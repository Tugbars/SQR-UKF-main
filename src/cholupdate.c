/**
 * @file cholupdate.c (REFACTORED)
 * @brief Rank-k Cholesky update with GEMM-accelerated QR
 * @author TUGBARS
 * @date 2025
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <immintrin.h>
#include "linalg_simd.h"
#include "../qr/qr.h"
#include "../gemm_2/gemm_utils.h"

#ifndef CHOLK_COL_TILE
#define CHOLK_COL_TILE 32
#endif

#ifndef CHOLK_AVX_MIN_N
#define CHOLK_AVX_MIN_N 16
#endif

#ifndef CHOLK_USE_ALIGNED_SIMD
#define CHOLK_USE_ALIGNED_SIMD 0
#endif

#if CHOLK_USE_ALIGNED_SIMD
#define CHOLK_MM256_LOAD_PS(ptr) _mm256_load_ps(ptr)
#define CHOLK_MM256_STORE_PS(ptr, val) _mm256_store_ps(ptr, val)
#else
#define CHOLK_MM256_LOAD_PS(ptr) _mm256_loadu_ps(ptr)
#define CHOLK_MM256_STORE_PS(ptr, val) _mm256_storeu_ps(ptr, val)
#endif

//==============================================================================
// WORKSPACE STRUCTURE
//==============================================================================

typedef struct cholupdate_workspace_s
{
    uint16_t n_max;
    uint16_t k_max;

    float *xbuf;  // n_max
    float *M;     // (n+k)×n for QR path (vertical stacking)
    float *R;     // n×n for QR result
    float *Utmp;  // n×n for transpose operations

    qr_workspace *qr_ws;
    size_t total_bytes;
} cholupdate_workspace;

//==============================================================================
// WORKSPACE API
//==============================================================================


cholupdate_workspace *cholupdate_workspace_alloc(uint16_t n_max, uint16_t k_max)
{
    if (n_max == 0)
        return NULL;

    cholupdate_workspace *ws = (cholupdate_workspace *)calloc(1, sizeof(cholupdate_workspace));
    if (!ws)
        return NULL;

    ws->n_max = n_max;
    ws->k_max = k_max;
    ws->total_bytes = sizeof(cholupdate_workspace);

#define ALLOC_BUF(ptr, count)                             \
    do                                                    \
    {                                                     \
        size_t bytes = (size_t)(count) * sizeof(float);   \
        ws->ptr = (float *)gemm_aligned_alloc(32, bytes); \
        if (!ws->ptr)                                     \
            goto cleanup_fail;                            \
        ws->total_bytes += bytes;                         \
    } while (0)

    ALLOC_BUF(xbuf, n_max);

    if (k_max > 0)
    {
        if ((size_t)n_max + k_max > UINT16_MAX)
            goto cleanup_fail;

        const size_t m_rows = (size_t)n_max + k_max;
        const uint16_t qr_m = (uint16_t)m_rows;
        const uint16_t qr_n = n_max;

        ALLOC_BUF(M, m_rows * n_max);
        ALLOC_BUF(R, m_rows * n_max);  // (n+k)×n, NOT n×n
        ALLOC_BUF(Utmp, (size_t)n_max * n_max);

        ws->qr_ws = qr_workspace_alloc_ex(qr_m, qr_n, 0, false);

        if (!ws->qr_ws)
            goto cleanup_fail;

        ws->total_bytes += qr_workspace_bytes(ws->qr_ws);
    }

#undef ALLOC_BUF

    return ws;

cleanup_fail:
    if (ws->xbuf) gemm_aligned_free(ws->xbuf);
    if (ws->M) gemm_aligned_free(ws->M);
    if (ws->R) gemm_aligned_free(ws->R);
    if (ws->Utmp) gemm_aligned_free(ws->Utmp);
    if (ws->qr_ws) qr_workspace_free(ws->qr_ws);
    free(ws);
    return NULL;
}

void cholupdate_workspace_free(cholupdate_workspace *ws)
{
    if (!ws)
        return;

    gemm_aligned_free(ws->xbuf);
    gemm_aligned_free(ws->M);
    gemm_aligned_free(ws->R);
    gemm_aligned_free(ws->Utmp);
    qr_workspace_free(ws->qr_ws);
    free(ws);
}

size_t cholupdate_workspace_bytes(const cholupdate_workspace *ws)
{
    return ws ? ws->total_bytes : 0;
}

//==============================================================================
// RANK-1 UPDATE (Gill-Golub-Murray-Saunders)
//==============================================================================

//==============================================================================
// RANK-1 UPDATE FIX (Gill-Golub-Murray-Saunders)
//==============================================================================

static int cholupdate_rank1_update(float *restrict L,
                                   float *restrict x,
                                   uint16_t n,
                                   bool is_upper)
{
#if __AVX2__
    const int use_avx = n >= (uint32_t)CHOLK_AVX_MIN_N;
#else
    const int use_avx = 0;
#endif

    for (uint32_t j = 0; j < n; ++j)
    {
        if (j + 4 < n)
        {
            const size_t prefetch_idx = (size_t)(j + 4) * n + (j + 4);
            _mm_prefetch((const char *)&L[prefetch_idx], _MM_HINT_T0);
        }

        const size_t dj = (size_t)j * n + j;
        const float Ljj = L[dj];
        const float xj = x[j];

        const double Ljj_sq = (double)Ljj * Ljj;
        const double xj_sq = (double)xj * xj;
        const double r_sq = Ljj_sq + xj_sq;

        if (r_sq <= 0.0 || !isfinite(r_sq))
            return -EDOM;

        const float r = (float)sqrt(r_sq);
        const float c = Ljj / r;
        const float s = xj / r;

        L[dj] = r;

        if (xj == 0.0f)
            continue;

        if (!use_avx || (j + 8 >= n))
        {
            for (uint32_t k = j + 1; k < n; ++k)
            {
                const size_t off = is_upper ? (size_t)j * n + k
                                            : (size_t)k * n + j;
                const float Lkj = L[off];
                const float xk = x[k];
                L[off] = c * Lkj + s * xk;
                x[k] = c * xk - s * Lkj;
            }
            continue;
        }

#if __AVX2__
        uint32_t k = j + 1;

#if CHOLK_USE_ALIGNED_SIMD
        while ((k < n) && ((uintptr_t)(&x[k]) & 31u))
        {
            const size_t off = is_upper ? (size_t)j * n + k
                                        : (size_t)k * n + j;
            const float Lkj = L[off];
            const float xk = x[k];
            L[off] = c * Lkj + s * xk;
            x[k] = c * xk - s * Lkj;
            ++k;
        }
#endif

        const __m256 c_v = _mm256_set1_ps(c);
        const __m256 s_v = _mm256_set1_ps(s);

        for (; k + 7 < n; k += 8)
        {
            float *baseL = is_upper ? &L[(size_t)j * n + k]
                                    : &L[(size_t)k * n + j];
            __m256 Lkj;

            if (is_upper)
            {
                Lkj = _mm256_loadu_ps(baseL);
            }
            else
            {
#ifdef __AVX2__
                int idx[8];
                for (int t = 0; t < 8; ++t)
                    idx[t] = t * (int)n;
                __m256i idx_vec = _mm256_loadu_si256((const __m256i *)idx);
                Lkj = _mm256_i32gather_ps(baseL, idx_vec, sizeof(float));
#else
                float tmp[8];
                for (int t = 0; t < 8; ++t)
                    tmp[t] = baseL[(size_t)t * n];
                Lkj = _mm256_loadu_ps(tmp);
#endif
            }

            __m256 xk = CHOLK_MM256_LOAD_PS(&x[k]);

            __m256 Lkj_new = _mm256_fmadd_ps(c_v, Lkj, _mm256_mul_ps(s_v, xk));
            __m256 xk_new = _mm256_fnmadd_ps(s_v, Lkj, _mm256_mul_ps(c_v, xk));

            CHOLK_MM256_STORE_PS(&x[k], xk_new);

            if (is_upper)
            {
                _mm256_storeu_ps(baseL, Lkj_new);
            }
            else
            {
                float tmp[8];
                _mm256_storeu_ps(tmp, Lkj_new);
                for (int t = 0; t < 8; ++t)
                    baseL[(size_t)t * n] = tmp[t];
            }
        }

        for (; k < n; ++k)
        {
            const size_t off = is_upper ? (size_t)j * n + k
                                        : (size_t)k * n + j;
            const float Lkj = L[off];
            const float xk = x[k];
            L[off] = c * Lkj + s * xk;
            x[k] = c * xk - s * Lkj;
        }
#endif
    }

    return 0;
}


//==============================================================================
// RANK-1 DOWNDATE (Hyperbolic rotation)
//==============================================================================

static int cholupdate_rank1_downdate(float *restrict L,
                                     float *restrict x,
                                     uint16_t n,
                                     bool is_upper)
{
#if __AVX2__
    const int use_avx = n >= (uint32_t)CHOLK_AVX_MIN_N;
#else
    const int use_avx = 0;
#endif

    for (uint32_t j = 0; j < n; ++j)
    {
        if (j + 4 < n)
        {
            const size_t prefetch_idx = (size_t)(j + 4) * n + (j + 4);
            _mm_prefetch((const char *)&L[prefetch_idx], _MM_HINT_T0);
        }

        const size_t dj = (size_t)j * n + j;
        const float Ljj = L[dj];
        const float xj = x[j];

        const float t = (Ljj != 0.0f) ? (xj / Ljj) : 0.0f;
        const float r2 = 1.0f - t * t;

        if (r2 <= 0.0f || !isfinite(r2))
            return -EDOM;

        const float c = sqrtf(r2);
        const float s = t;
        L[dj] = c * Ljj;

        if (xj == 0.0f)
            continue;

        if (!use_avx || (j + 8 >= n))
        {
            for (uint32_t k = j + 1; k < n; ++k)
            {
                const size_t off = is_upper ? (size_t)j * n + k
                                            : (size_t)k * n + j;
                const float Lkj = L[off];
                const float xk = x[k];
                L[off] = (Lkj - s * xk) / c;
                x[k] = c * xk - s * Lkj;
            }
            continue;
        }

#if __AVX2__
        uint32_t k = j + 1;

#if CHOLK_USE_ALIGNED_SIMD
        while ((k < n) && ((uintptr_t)(&x[k]) & 31u))
        {
            const size_t off = is_upper ? (size_t)j * n + k
                                        : (size_t)k * n + j;
            const float Lkj = L[off];
            const float xk = x[k];
            L[off] = (Lkj - s * xk) / c;
            x[k] = c * xk - s * Lkj;
            ++k;
        }
#endif

        const __m256 c_v = _mm256_set1_ps(c);
        const __m256 s_v = _mm256_set1_ps(s);
        const __m256 rcp_c = _mm256_set1_ps(1.0f / c);

        for (; k + 7 < n; k += 8)
        {
            float *baseL = is_upper ? &L[(size_t)j * n + k]
                                    : &L[(size_t)k * n + j];
            __m256 Lkj;

            if (is_upper)
            {
                Lkj = _mm256_loadu_ps(baseL);
            }
            else
            {
#ifdef __AVX2__
                int idx[8];
                for (int t = 0; t < 8; ++t)
                    idx[t] = t * (int)n;
                __m256i idx_vec = _mm256_loadu_si256((const __m256i *)idx);
                Lkj = _mm256_i32gather_ps(baseL, idx_vec, sizeof(float));
#else
                float tmp[8];
                for (int t = 0; t < 8; ++t)
                    tmp[t] = baseL[(size_t)t * n];
                Lkj = _mm256_loadu_ps(tmp);
#endif
            }

            __m256 xk = CHOLK_MM256_LOAD_PS(&x[k]);

            __m256 Lkj_new = _mm256_mul_ps(_mm256_fnmadd_ps(s_v, xk, Lkj), rcp_c);
            __m256 xk_new = _mm256_fnmadd_ps(s_v, Lkj, _mm256_mul_ps(c_v, xk));

            CHOLK_MM256_STORE_PS(&x[k], xk_new);

            if (is_upper)
            {
                _mm256_storeu_ps(baseL, Lkj_new);
            }
            else
            {
                float tmp[8];
                _mm256_storeu_ps(tmp, Lkj_new);
                for (int t = 0; t < 8; ++t)
                    baseL[(size_t)t * n] = tmp[t];
            }
        }

        for (; k < n; ++k)
        {
            const size_t off = is_upper ? (size_t)j * n + k
                                        : (size_t)k * n + j;
            const float Lkj = L[off];
            const float xk = x[k];
            L[off] = (Lkj - s * xk) / c;
            x[k] = c * xk - s * Lkj;
        }
#endif
    }

    return 0;
}

//==============================================================================
// TILED RANK-K
//==============================================================================

static int cholupdatek_tiled_ws(cholupdate_workspace *ws,
                                float *restrict L,
                                const float *restrict X,
                                uint16_t n, uint16_t k,
                                bool is_upper, int add)
{
    if (!ws || !L)
        return -EINVAL;
    if (n == 0)
        return -EINVAL;
    if (k == 0)
        return 0;
    if (!X)
        return -EINVAL;
    if (add != +1 && add != -1)
        return -EINVAL;
    if (n > ws->n_max || k > ws->k_max)
        return -EOVERFLOW;

    const uint16_t T = (CHOLK_COL_TILE == 0) ? 32 : (uint16_t)CHOLK_COL_TILE;
    float *xbuf = ws->xbuf;

    int rc = 0;

    for (uint16_t p0 = 0; p0 < k; p0 += T)
    {
        const uint16_t jb = (uint16_t)((p0 + T <= k) ? T : (k - p0));

        for (uint16_t t = 0; t < jb; ++t)
        {
            const float *xcol = X + (p0 + t);
            for (uint16_t r = 0; r < n; ++r)
                xbuf[r] = xcol[(size_t)r * k];

            if (add > 0)
                rc = cholupdate_rank1_update(L, xbuf, n, is_upper);
            else
                rc = cholupdate_rank1_downdate(L, xbuf, n, is_upper);

            if (rc)
                return rc;
        }
    }

    return 0;
}

//==============================================================================
// QR-BASED UPDATE (UPDATE ONLY, VERTICAL STACKING)
//==============================================================================

static void copy_upper_nxn(float *restrict dst,
                          const float *restrict src,
                          uint16_t n, uint16_t ld_src)
{
    for (uint16_t i = 0; i < n; ++i)
    {
        const float *row = src + (size_t)i * ld_src;

        for (uint16_t j = 0; j < i; ++j)
            dst[(size_t)i * n + j] = 0.0f;

        memcpy(dst + (size_t)i * n + i, row + i, (size_t)(n - i) * sizeof(float));
    }
}

//==============================================================================
// FIX 1: Sign correction after QR extraction
//==============================================================================

static void copy_upper_nxn_with_positive_diag(float *restrict dst,
                                               const float *restrict src,
                                               uint16_t n, uint16_t ld_src)
{
    for (uint16_t i = 0; i < n; ++i)
    {
        const float *row = src + (size_t)i * ld_src;
        float *dst_row = dst + (size_t)i * n;

        // Zero below diagonal
        for (uint16_t j = 0; j < i; ++j)
            dst_row[j] = 0.0f;

        // Copy upper triangle
        memcpy(dst_row + i, row + i, (size_t)(n - i) * sizeof(float));

        // ✅ FIX: Enforce positive diagonal by negating row if needed
        if (dst_row[i] < 0.0f)
        {
            for (uint16_t j = i; j < n; ++j)
                dst_row[j] = -dst_row[j];
        }
    }
}

int cholupdatek_blockqr_ws(cholupdate_workspace *ws,
                           float *restrict L_or_U,
                           const float *restrict X,
                           uint16_t n, uint16_t k,
                           bool is_upper, int add)
{
    if (!ws || !L_or_U)
        return -EINVAL;
    if (n == 0)
        return -EINVAL;
    if (k == 0)
        return 0;
    if (!X)
        return -EINVAL;
    if (add != +1)
        return -EINVAL;
    if (n > ws->n_max || k > ws->k_max)
        return -EOVERFLOW;
    if (!ws->qr_ws)
        return -EINVAL;

    if ((size_t)n + k > UINT16_MAX)
        return -EOVERFLOW;

    const uint16_t m_rows = (uint16_t)(n + k);

    if (m_rows > ws->qr_ws->m_max || n > ws->qr_ws->n_max)
        return -EOVERFLOW;

    float *M = ws->M;
    float *R = ws->R;
    float *Utmp = ws->Utmp;

    if (is_upper)
    {
        for (uint16_t i = 0; i < n; ++i)
        {
            float *dst = M + (size_t)i * n;
            const float *src = L_or_U + (size_t)i * n;

            for (uint16_t j = 0; j < i; ++j)
                dst[j] = 0.0f;

            memcpy(dst + i, src + i, (size_t)(n - i) * sizeof(float));
        }
    }
    else
    {
        for (uint16_t i = 0; i < n; ++i)
        {
            float *dst = M + (size_t)i * n;

            for (uint16_t j = 0; j < i; ++j)
                dst[j] = 0.0f;

            dst[i] = L_or_U[(size_t)i * n + i];

            for (uint16_t j = i + 1; j < n; ++j)
                dst[j] = L_or_U[(size_t)j * n + i];
        }
    }

    for (uint16_t i = 0; i < k; ++i)
    {
        float *dst = M + (size_t)(n + i) * n;
        const float *src = X + i;

        for (uint16_t j = 0; j < n; ++j)
            dst[j] = src[(size_t)j * k];
    }

    int rc = qr_ws_blocked_inplace(ws->qr_ws, M, NULL, R, m_rows, n, true);
    if (rc)
        return rc;

    if (is_upper)
    {
        copy_upper_nxn_with_positive_diag(L_or_U, R, n, n);
    }
    else
    {
        copy_upper_nxn_with_positive_diag(Utmp, R, n, n);

        for (uint16_t i = 0; i < n; ++i)
        {
            for (uint16_t j = 0; j < i; ++j)
                L_or_U[(size_t)i * n + j] = Utmp[(size_t)j * n + i];

            L_or_U[(size_t)i * n + i] = Utmp[(size_t)i * n + i];

            for (uint16_t j = i + 1; j < n; ++j)
                L_or_U[(size_t)i * n + j] = 0.0f;
        }
    }

    return 0;
}

//==============================================================================
// ALGORITHM SELECTION
//==============================================================================

static inline int choose_cholupdate_method(uint16_t n, uint16_t k, int add)
{
    if (add < 0)
        return 0;  // Downdates must use rank-1

    if (k == 1)
        return 0;

    uint16_t estimated_qr_ib;
    if (n < 32)
        estimated_qr_ib = 8;
    else if (n < 128)
        estimated_qr_ib = 32;
    else if (n < 512)
        estimated_qr_ib = 64;
    else
        estimated_qr_ib = 96;

    if (k >= estimated_qr_ib / 2 && n >= 32)
        return 1;

    if (k < 8)
        return 0;

    if (n < 32)
        return 0;

    if (k >= 8 && n >= 32)
        return 1;

    return 0;
}

//==============================================================================
// AUTO-DISPATCH
//==============================================================================

int cholupdatek_auto_ws(cholupdate_workspace *ws,
                        float *restrict L_or_U,
                        const float *restrict X,
                        uint16_t n, uint16_t k,
                        bool is_upper, int add)
{
    if (!ws || !L_or_U)
        return -EINVAL;
    if (n == 0)
        return -EINVAL;
    if (k == 0)
        return 0;
    if (!X)
        return -EINVAL;
    if (add != +1 && add != -1)
        return -EINVAL;

    int method = choose_cholupdate_method(n, k, add);

    if (method == 1)
        return cholupdatek_blockqr_ws(ws, L_or_U, X, n, k, is_upper, add);
    else
        return cholupdatek_tiled_ws(ws, L_or_U, X, n, k, is_upper, add);
}


//==============================================================================
// EXPLICIT PATH SELECTION
//==============================================================================

int cholupdatek_ws(cholupdate_workspace *ws,
                   float *restrict L,
                   const float *restrict X,
                   uint16_t n, uint16_t k,
                   bool is_upper, int add)
{
    return cholupdatek_tiled_ws(ws, L, X, n, k, is_upper, add);
}

//==============================================================================
// LEGACY API
//==============================================================================

int cholupdatek(float *restrict L,
                const float *restrict X,
                uint16_t n, uint16_t k,
                bool is_upper, int add)
{
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
        return -ENOMEM;

    int ret = cholupdatek_tiled_ws(ws, L, X, n, k, is_upper, add);
    cholupdate_workspace_free(ws);

    return ret;
}

int cholupdatek_blockqr(float *restrict L_or_U,
                        const float *restrict X,
                        uint16_t n, uint16_t k,
                        bool is_upper, int add)
{
    if (add != +1)
        return -EINVAL;

    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
        return -ENOMEM;

    int ret = cholupdatek_blockqr_ws(ws, L_or_U, X, n, k, is_upper, add);
    cholupdate_workspace_free(ws);

    return ret;
}

int cholupdatek_blas3(float *restrict L_or_U,
                      const float *restrict X,
                      uint16_t n, uint16_t k,
                      bool is_upper, int add)
{
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
        return -ENOMEM;

    int ret = cholupdatek_auto_ws(ws, L_or_U, X, n, k, is_upper, add);
    cholupdate_workspace_free(ws);

    return ret;
}