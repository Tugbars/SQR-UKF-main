/**
 * @file cholupdate.c
 * @brief Rank-k Cholesky update with workspace-based zero-malloc design
 *
 * @details
 * FFTW-style planning: all allocations happen once in workspace creation,
 * hot-path execution (`cholupdatek_ws`, `cholupdatek_blockqr_ws`) is allocation-free.
 *
 * Provides both tiled AVX2-optimized and BLAS-3 QR-based implementations
 * for rank-k Cholesky updates/downdates:
 * \f[
 *    L L^T \leftarrow L L^T \pm X X^T
 * \f]
 *
 * ### Key improvements over original:
 * - Zero hot-path allocations (workspace pre-allocated)
 * - Thread-safe by design (each workspace is independent)
 * - Backward-compatible legacy API (allocates internally)
 * - BLAS-3 capable with blocked QR
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <immintrin.h>
#include "linalg_simd.h"

#ifndef CHOLK_COL_TILE
#define CHOLK_COL_TILE 32
#endif

#ifndef CHOLK_AVX_MIN_N
#define CHOLK_AVX_MIN_N LINALG_SMALL_N_THRESH
#endif

#ifndef QRW_IB_DEFAULT
#define QRW_IB_DEFAULT 64
#endif

// Forward declaration for QR (if not in header)
extern int qrw_geqrf_blocked_wy(float *A, uint16_t m, uint16_t n,
                                uint16_t ib, float *tau);

//==============================================================================
// PLATFORM-SPECIFIC ALLOCATION
//==============================================================================

static void *aligned_alloc32(size_t size)
{
#if defined(_WIN32)
    return _aligned_malloc(size, 32);
#else
    void *p = NULL;
    if (posix_memalign(&p, 32, size) != 0)
        return NULL;
    return p;
#endif
}

static void aligned_free32(void *p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

//==============================================================================
// WORKSPACE STRUCTURE
//==============================================================================

/**
 * @brief Cholesky update workspace - all buffers pre-allocated
 *
 * @note Thread-safe: each thread should have its own workspace
 * @note Sized for maximum dimensions (n_max, k_max) specified at creation
 */
typedef struct cholupdate_workspace_s
{
    // Dimensions
    uint16_t n_max;
    uint16_t k_max;

    // Tiled rank-1 update buffers (32-byte aligned)
    float *xbuf; // n_max - work vector for rank-1 updates

    // Blocked QR buffers (32-byte aligned)
    float *M;    // (n+k)×n - augmented matrix [U | X]
    float *tau;  // min(n,n+k) - Householder scalars for QR
    float *Utmp; // n×n - temporary for transpose operations

    // Memory accounting
    size_t total_bytes;
} cholupdate_workspace;

//==============================================================================
// WORKSPACE API
//==============================================================================

/**
 * @brief Allocate Cholesky update workspace
 *
 * @param n_max Maximum matrix dimension
 * @param k_max Maximum rank of updates
 *
 * @return Workspace pointer on success, NULL on allocation failure
 *
 * @note COLD PATH - call once, reuse many times
 */
cholupdate_workspace *cholupdate_workspace_alloc(uint16_t n_max, uint16_t k_max)
{
    if (n_max == 0)
        return NULL;

    cholupdate_workspace *ws = (cholupdate_workspace *)malloc(sizeof(cholupdate_workspace));
    if (!ws)
        return NULL;

    memset(ws, 0, sizeof(cholupdate_workspace));
    ws->n_max = n_max;
    ws->k_max = k_max;
    ws->total_bytes = sizeof(cholupdate_workspace);

// Allocate buffers
#define ALLOC_BUF(ptr, count)                           \
    do                                                  \
    {                                                   \
        size_t bytes = (size_t)(count) * sizeof(float); \
        ws->ptr = (float *)aligned_alloc32(bytes);      \
        if (!ws->ptr)                                   \
            goto cleanup_fail;                          \
        ws->total_bytes += bytes;                       \
    } while (0)

    ALLOC_BUF(xbuf, n_max);

    if (k_max > 0)
    {
        const size_t m_cols = (size_t)n_max + k_max;
        ALLOC_BUF(M, (size_t)n_max * m_cols);
        ALLOC_BUF(tau, n_max < m_cols ? n_max : m_cols);
        ALLOC_BUF(Utmp, (size_t)n_max * n_max);
    }

#undef ALLOC_BUF

    return ws;

cleanup_fail:
    if (ws->xbuf)
        aligned_free32(ws->xbuf);
    if (ws->M)
        aligned_free32(ws->M);
    if (ws->tau)
        aligned_free32(ws->tau);
    if (ws->Utmp)
        aligned_free32(ws->Utmp);
    free(ws);
    return NULL;
}

/**
 * @brief Free Cholesky update workspace
 *
 * @param ws Workspace to free (NULL-safe)
 */
void cholupdate_workspace_free(cholupdate_workspace *ws)
{
    if (!ws)
        return;

    aligned_free32(ws->xbuf);
    aligned_free32(ws->M);
    aligned_free32(ws->tau);
    aligned_free32(ws->Utmp);
    free(ws);
}

/**
 * @brief Query workspace memory usage
 *
 * @param ws Workspace to query
 * @return Total bytes allocated (0 if ws=NULL)
 */
size_t cholupdate_workspace_bytes(const cholupdate_workspace *ws)
{
    return ws ? ws->total_bytes : 0;
}

//==============================================================================
// INTERNAL: RANK-1 UPDATE KERNEL (UNCHANGED ALGORITHM)
//==============================================================================

/**
 * @brief Internal robust rank-1 Cholesky update/downdate kernel
 *
 * @note Algorithm unchanged - only uses workspace xbuf instead of allocating
 */
static int cholupdate_rank1_core(float *RESTRICT L,
                                 float *RESTRICT x, /* workspace buffer */
                                 uint16_t n,
                                 bool is_upper,
                                 int add)
{
    const float sign = (add >= 0) ? 1.0f : -1.0f;

#if LINALG_SIMD_ENABLE
    const int use_avx = linalg_has_avx2() && n >= (uint32_t)CHOLK_AVX_MIN_N;
#else
    const int use_avx = 0;
#endif

    for (uint32_t i = 0; i < n; ++i)
    {
        const size_t di = (size_t)i * n + i;
        const float Lii = L[di];
        const float xi = x[i];

        const float t = (Lii != 0.0f) ? (xi / Lii) : 0.0f;
        const float r2 = 1.0f + sign * t * t;
        if (r2 <= 0.0f || !isfinite(r2))
            return -EDOM;

        const float c = sqrtf(r2);
        const float s = t;
        L[di] = c * Lii;

        if (xi == 0.0f)
            continue;

        if (!use_avx || (i + 8 >= n))
        {
            for (uint32_t k = i + 1; k < n; ++k)
            {
                const size_t off = is_upper ? (size_t)i * n + k
                                            : (size_t)k * n + i;
                const float Lik = L[off];
                const float xk = x[k];
                L[off] = (Lik + sign * s * xk) / c;
                x[k] = c * xk - s * Lik;
            }
            continue;
        }

#if LINALG_SIMD_ENABLE
        uint32_t k = (uint32_t)i + 1;

        // Align to 32B
        while ((k < n) && ((uintptr_t)(&x[k]) & 31u))
        {
            const size_t off = is_upper ? (size_t)i * n + k
                                        : (size_t)k * n + i;
            const float Lik = L[off];
            const float xk = x[k];
            L[off] = (Lik + sign * s * xk) / c;
            x[k] = c * xk - s * Lik;
            ++k;
        }

        const __m256 c_v = _mm256_set1_ps(c);
        const __m256 s_v = _mm256_set1_ps(s);
        const __m256 ss_v = _mm256_set1_ps(sign * s);

        for (; k + 7 < n; k += 8)
        {
            float *baseL = is_upper ? &L[(size_t)i * n + k]
                                    : &L[(size_t)k * n + i];
            __m256 Lik;
            if (is_upper)
            {
                Lik = _mm256_loadu_ps(baseL);
            }
            else
            {
#ifdef __AVX2__
                alignas(32) int idx[8];
                for (int t = 0; t < 8; ++t)
                    idx[t] = t * (int)n;
                Lik = _mm256_i32gather_ps(baseL, _mm256_load_si256((const __m256i *)idx), sizeof(float));
#else
                alignas(32) float tmp[8];
                for (int t = 0; t < 8; ++t)
                    tmp[t] = baseL[(size_t)t * n];
                Lik = _mm256_load_ps(tmp);
#endif
            }

            __m256 xk = _mm256_load_ps(&x[k]);

            __m256 Lik_new = _mm256_mul_ps(_mm256_fmadd_ps(ss_v, xk, Lik),
                                           _mm256_div_ps(_mm256_set1_ps(1.0f), c_v));
            __m256 xk_new = _mm256_fnmadd_ps(s_v, Lik, _mm256_mul_ps(c_v, xk));

            _mm256_store_ps(&x[k], xk_new);

            if (is_upper)
            {
                _mm256_storeu_ps(baseL, Lik_new);
            }
            else
            {
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, Lik_new);
                for (int t = 0; t < 8; ++t)
                    baseL[(size_t)t * n] = tmp[t];
            }
        }

        for (; k < n; ++k)
        {
            const size_t off = is_upper ? (size_t)i * n + k
                                        : (size_t)k * n + i;
            const float Lik = L[off];
            const float xk = x[k];
            L[off] = (Lik + sign * s * xk) / c;
            x[k] = c * xk - s * Lik;
        }
#endif
    }
    return 0;
}

//==============================================================================
// TILED RANK-K UPDATE WITH WORKSPACE (ZERO MALLOC)
//==============================================================================

/**
 * @brief Tiled rank-k Cholesky update using workspace (HOT PATH - ZERO MALLOC)
 *
 * @param ws Pre-allocated workspace
 * @param L In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension (must be ≤ n_max from workspace)
 * @param k Rank of update (must be ≤ k_max from workspace)
 * @param is_upper True for upper-triangular, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure
 *
 * @note ZERO allocations - fully cache-optimized
 */
int cholupdatek_ws(cholupdate_workspace *ws,
                   float *RESTRICT L,
                   const float *RESTRICT X,
                   uint16_t n, uint16_t k,
                   bool is_upper, int add)
{
    if (!ws || !L || !X)
        return -EINVAL;

    if (n == 0)
        return -EINVAL;
    if (k == 0)
        return 0;
    if (add != +1 && add != -1)
        return -EINVAL;

    // User's responsibility to provide correct dimensions
    // (Could add assert(n <= ws->n_max && k <= ws->k_max) in debug)

    const uint16_t T = (CHOLK_COL_TILE == 0) ? 32 : (uint16_t)CHOLK_COL_TILE;

    // Use workspace buffer (NO MALLOC)
    float *xbuf = ws->xbuf;

    int rc = 0;
    for (uint16_t p0 = 0; p0 < k; p0 += T)
    {
        const uint16_t jb = (uint16_t)((p0 + T <= k) ? T : (k - p0));

        for (uint16_t t = 0; t < jb; ++t)
        {
            // Gather column into contiguous buffer
            const float *xcol = X + (p0 + t);
            for (uint16_t r = 0; r < n; ++r)
                xbuf[r] = xcol[(size_t)r * k];

            rc = cholupdate_rank1_core(L, xbuf, n, is_upper, add);
            if (rc)
                return rc;
        }
    }

    return 0;
}

//==============================================================================
// BLOCKED QR UPDATE WITH WORKSPACE (ZERO MALLOC)
//==============================================================================

/**
 * @brief Extract upper-triangular block from QR result
 */
static void copy_upper_nxn_from_qr(float *RESTRICT Udst,
                                   const float *RESTRICT Rsrc,
                                   uint16_t n, uint16_t ldR)
{
    for (uint16_t i = 0; i < n; ++i)
    {
        const float *row = Rsrc + (size_t)i * ldR;
        for (uint16_t j = 0; j < i; ++j)
            Udst[(size_t)i * n + j] = 0.0f;
        memcpy(Udst + (size_t)i * n + i, row + i, (size_t)(n - i) * sizeof(float));
    }
}

/**
 * @brief BLAS-3 rank-k Cholesky update using workspace (HOT PATH - ZERO MALLOC)
 *
 * @param ws Pre-allocated workspace
 * @param L_or_U In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension (must be ≤ n_max)
 * @param k Rank (must be ≤ k_max)
 * @param is_upper True for upper, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure
 *
 * @note Uses blocked QR for BLAS-3 efficiency
 * @note ZERO allocations - all buffers from workspace
 */
int cholupdatek_blockqr_ws(cholupdate_workspace *ws,
                           float *RESTRICT L_or_U,
                           const float *RESTRICT X,
                           uint16_t n, uint16_t k,
                           bool is_upper, int add)
{
    if (!ws || !L_or_U || !X)
        return -EINVAL;

    if (n == 0)
        return -EINVAL;
    if (k == 0)
        return 0;
    if (add != +1 && add != -1)
        return -EINVAL;

    // Use workspace buffers (NO MALLOC)
    const uint16_t m_cols = (uint16_t)(n + k);
    float *M = ws->M;
    float *tau = ws->tau;
    float *Utmp = ws->Utmp;

    // Build M = [U | s*X]
    if (is_upper)
    {
        // Copy U into M[:,0:n]
        for (uint16_t i = 0; i < n; ++i)
        {
            float *dst = M + (size_t)i * m_cols;
            const float *src = L_or_U + (size_t)i * n;
            for (uint16_t j = 0; j < i; ++j)
                dst[j] = 0.0f;
            memcpy(dst + i, src + i, (size_t)(n - i) * sizeof(float));
        }
    }
    else
    {
        // Build U = L^T into M[:,0:n]
        for (uint16_t i = 0; i < n; ++i)
        {
            float *dst = M + (size_t)i * m_cols;
            for (uint16_t j = 0; j < i; ++j)
                dst[j] = L_or_U[(size_t)j * n + i];
            dst[i] = L_or_U[(size_t)i * n + i];
            for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
                dst[j] = 0.0f;
        }
    }

    // Copy scaled X into M[:,n:n+k]
    const float s = (add >= 0) ? 1.0f : -1.0f;
    for (uint16_t i = 0; i < n; ++i)
    {
        float *dst = M + (size_t)i * m_cols + n;
        const float *src = X + (size_t)i * k;
        for (uint16_t j = 0; j < k; ++j)
            dst[j] = s * src[j];
    }

    // QR factorization (uses workspace tau - NO MALLOC)
    int rc = qrw_geqrf_blocked_wy(M, n, m_cols, QRW_IB_DEFAULT, tau);
    if (rc)
        return rc;

    // Extract new Cholesky factor
    if (is_upper)
    {
        copy_upper_nxn_from_qr(L_or_U, M, n, m_cols);
    }
    else
    {
        // Extract U, transpose to L (uses workspace Utmp - NO MALLOC)
        copy_upper_nxn_from_qr(Utmp, M, n, m_cols);

        for (uint16_t i = 0; i < n; ++i)
        {
            for (uint16_t j = 0; j < i; ++j)
            {
                L_or_U[(size_t)i * n + j] = Utmp[(size_t)j * n + i];
            }
            L_or_U[(size_t)i * n + i] = Utmp[(size_t)i * n + i];
            for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
            {
                L_or_U[(size_t)i * n + j] = 0.0f;
            }
        }
    }

    return 0;
}

//==============================================================================
// LEGACY API (BACKWARD COMPATIBLE - CREATES TEMPORARY WORKSPACE)
//==============================================================================

/**
 * @brief Legacy tiled rank-k update (UNCHANGED API - backward compatible)
 *
 * @note Allocates workspace internally - for performance-critical code, use cholupdatek_ws()
 */
int cholupdatek(float *RESTRICT L,
                const float *RESTRICT X,
                uint16_t n, uint16_t k,
                bool is_upper, int add)
{
    // Create temporary workspace
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
        return -ENOMEM;

    // Execute using workspace
    int ret = cholupdatek_ws(ws, L, X, n, k, is_upper, add);

    // Cleanup
    cholupdate_workspace_free(ws);

    return ret;
}

/**
 * @brief Legacy blocked QR update (UNCHANGED API - backward compatible)
 *
 * @note Allocates workspace internally - for performance-critical code, use cholupdatek_blockqr_ws()
 */
int cholupdatek_blockqr(float *RESTRICT L_or_U,
                        const float *RESTRICT X,
                        uint16_t n, uint16_t k,
                        bool is_upper, int add)
{
    // Create temporary workspace
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
        return -ENOMEM;

    // Execute using workspace
    int ret = cholupdatek_blockqr_ws(ws, L_or_U, X, n, k, is_upper, add);

    // Cleanup
    cholupdate_workspace_free(ws);

    return ret;
}

/**
 * @brief Legacy BLAS-3 dispatcher (UNCHANGED API)
 */
int cholupdatek_blas3(float *RESTRICT L_or_U,
                      const float *RESTRICT X,
                      uint16_t n, uint16_t k,
                      bool is_upper, int add)
{
    int rc = cholupdatek_blockqr(L_or_U, X, n, k, is_upper, add);
    if (rc == -ENOTSUP)
    {
        return cholupdatek(L_or_U, X, n, k, is_upper, add);
    }
    return rc;
}