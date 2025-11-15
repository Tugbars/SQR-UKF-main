/**
 * @file cholupdate.c
 * @brief Rank-k Cholesky update with GEMM-accelerated QR
 *
 * @details
 * Zero-allocation hot path with automatic algorithm selection:
 * - Rank-1 path: Tiled SIMD updates for small k
 * - QR path: BLAS-3 blocked QR for large k (leverages GEMM acceleration)
 *
 * Implements rank-k Cholesky updates/downdates:
 * \f[
 *    L L^T \leftarrow L L^T \pm X X^T
 * \f]
 *
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

#ifndef CHOLK_COL_TILE
#define CHOLK_COL_TILE 32
#endif

#ifndef CHOLK_AVX_MIN_N
#define CHOLK_AVX_MIN_N 16
#endif

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
    float *R;    // n×(n+k) - QR result storage
    float *Utmp; // n×n - temporary for transpose operations

    // QR workspace (embedded for zero-malloc QR)
    qr_workspace *qr_ws;

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

    cholupdate_workspace *ws = (cholupdate_workspace *)calloc(1, sizeof(cholupdate_workspace));
    if (!ws)
        return NULL;

    ws->n_max = n_max;
    ws->k_max = k_max;
    ws->total_bytes = sizeof(cholupdate_workspace);

#define ALLOC_BUF(ptr, count)                           \
    do                                                  \
    {                                                   \
        size_t bytes = (size_t)(count) * sizeof(float); \
        ws->ptr = (float *)aligned_alloc32(bytes);      \
        if (!ws->ptr)                                   \
            goto cleanup_fail;                          \
        ws->total_bytes += bytes;                       \
    } while (0)

    // Always allocate rank-1 buffer
    ALLOC_BUF(xbuf, n_max);

    // Allocate QR buffers if k_max > 0
    if (k_max > 0)
    {
        const size_t m_cols = (size_t)n_max + k_max;

        ALLOC_BUF(M, (size_t)n_max * m_cols);
        ALLOC_BUF(R, (size_t)n_max * m_cols);
        ALLOC_BUF(Utmp, (size_t)n_max * n_max);

        // ============================================
        // QR WORKSPACE ALLOCATION (OPTIMIZED)
        // ============================================
        // - ib=0: Auto-select block size from GEMM tuning
        // - store_reflectors=false: We only need R, not Q
        //   Saves memory (~n²×k floats) and skips reflector storage
        ws->qr_ws = qr_workspace_alloc_ex(
            n_max,            // m_max
            (uint16_t)m_cols, // n_max
            0,                // ib=0 → auto-select
            false             // don't store reflectors
        );

        if (!ws->qr_ws)
            goto cleanup_fail;

        ws->total_bytes += qr_workspace_bytes(ws->qr_ws);
    }

#undef ALLOC_BUF

    return ws;

cleanup_fail:
    if (ws->xbuf)
        aligned_free32(ws->xbuf);
    if (ws->M)
        aligned_free32(ws->M);
    if (ws->R)
        aligned_free32(ws->R);
    if (ws->Utmp)
        aligned_free32(ws->Utmp);
    if (ws->qr_ws)
        qr_workspace_free(ws->qr_ws);
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
    aligned_free32(ws->R);
    aligned_free32(ws->Utmp);
    qr_workspace_free(ws->qr_ws);

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
// INTERNAL: OPTIMIZED RANK-1 UPDATE KERNEL
//==============================================================================

/**
 * @brief Internal robust rank-1 Cholesky update/downdate kernel
 *
 * @details Uses hyperbolic rotations (Givens-like) to maintain positive definiteness
 * Algorithm: For each row i, apply rotation [c s; s c] to maintain triangular form
 *
 * @note Enhanced with prefetching for better cache behavior
 */
static int cholupdate_rank1_core(float *restrict L,
                                 float *restrict x,
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
        // Prefetch next diagonal element for better pipeline utilization
        if (i + 4 < n)
        {
            const size_t prefetch_idx = (size_t)(i + 4) * n + (i + 4);
            _mm_prefetch((const char *)&L[prefetch_idx], _MM_HINT_T0);
        }

        const size_t di = (size_t)i * n + i;
        const float Lii = L[di];
        const float xi = x[i];

        // Compute rotation parameters
        const float t = (Lii != 0.0f) ? (xi / Lii) : 0.0f;
        const float r2 = 1.0f + sign * t * t;

        if (r2 <= 0.0f || !isfinite(r2))
            return -EDOM; // Matrix would become indefinite

        const float c = sqrtf(r2);
        const float s = t;
        L[di] = c * Lii;

        if (xi == 0.0f)
            continue; // No work needed for this row

        // =====================================================================
        // Apply rotation to remaining elements
        // =====================================================================

        if (!use_avx || (i + 8 >= n))
        {
            // Scalar fallback for small tail or when SIMD disabled
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
        // =====================================================================
        // AVX2 SIMD PATH
        // =====================================================================
        uint32_t k = (uint32_t)i + 1;

        // Align to 32-byte boundary for optimal SIMD performance
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
        const __m256 rcp_c = _mm256_set1_ps(1.0f / c);

        for (; k + 7 < n; k += 8)
        {
            float *baseL = is_upper ? &L[(size_t)i * n + k]
                                    : &L[(size_t)k * n + i];
            __m256 Lik;

            if (is_upper)
            {
                // Contiguous load for upper triangular
                Lik = _mm256_loadu_ps(baseL);
            }
            else
            {
                // Strided gather for lower triangular
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

            // Compute: Lik_new = (Lik + sign*s*xk) / c
            __m256 Lik_new = _mm256_mul_ps(_mm256_fmadd_ps(ss_v, xk, Lik), rcp_c);

            // Compute: xk_new = c*xk - s*Lik
            __m256 xk_new = _mm256_fnmadd_ps(s_v, Lik, _mm256_mul_ps(c_v, xk));

            _mm256_store_ps(&x[k], xk_new);

            if (is_upper)
            {
                // Contiguous store
                _mm256_storeu_ps(baseL, Lik_new);
            }
            else
            {
                // Strided scatter
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, Lik_new);
                for (int t = 0; t < 8; ++t)
                    baseL[(size_t)t * n] = tmp[t];
            }
        }

        // Scalar tail
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
// ALGORITHM SELECTION HEURISTICS
//==============================================================================

/**
 * @brief Choose optimal algorithm based on problem characteristics
 *
 * @details Heuristics based on operation counts and cache behavior:
 * - Rank-1 tiled: O(n²k) operations, k sequential passes
 * - QR blocked: O(n²k + nk²) operations, BLAS-3 dominated
 *
 * Crossover analysis:
 * - k=1: Rank-1 always wins (specialized, no overhead)
 * - k∈[2,7]: Tiled rank-1 competitive (good cache, already SIMD)
 * - k≥8: QR scales better (GEMM dominates, ~20 GFLOPS on 14900)
 * - Large n: QR startup cost amortized
 * - Small n: Tiled overhead minimal
 *
 * @return 0=tiled rank-1, 1=blocked QR
 */
static inline int choose_cholupdate_method(uint16_t n, uint16_t k)
{
    // Single column: always use specialized rank-1
    if (k == 1)
        return 0;

    // Small rank: tiled is competitive and simpler
    if (k < 8)
        return 0;

    // Very small matrix: QR overhead not worth it
    if (n < 32)
        return 0;

    // Large rank with reasonable matrix size: QR's BLAS-3 wins
    // At k=8, n=32: QR does ~32²×8 + 32×8² ≈ 10K flops via GEMM
    //               Tiled does ~32²×8 ≈ 8K flops but with worse cache
    if (k >= 8 && n >= 32)
        return 1;

    // Default: tiled (conservative, always correct)
    return 0;
}

//==============================================================================
// TILED RANK-K UPDATE (HOT PATH - ZERO MALLOC)
//==============================================================================

/**
 * @brief Tiled rank-k Cholesky update using workspace
 *
 * @details Applies k rank-1 updates in tiles for cache efficiency
 * Each update costs O(n²), total O(n²k)
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
static int cholupdatek_tiled_ws(cholupdate_workspace *ws,
                                float *restrict L,
                                const float *restrict X,
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

    const uint16_t T = (CHOLK_COL_TILE == 0) ? 32 : (uint16_t)CHOLK_COL_TILE;
    float *xbuf = ws->xbuf;

    int rc = 0;

    // Process X in tiles of T columns for cache locality
    for (uint16_t p0 = 0; p0 < k; p0 += T)
    {
        const uint16_t jb = (uint16_t)((p0 + T <= k) ? T : (k - p0));

        for (uint16_t t = 0; t < jb; ++t)
        {
            // Gather column from X into contiguous buffer
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
// BLOCKED QR UPDATE (HOT PATH - ZERO MALLOC, BLAS-3)
//==============================================================================

/**
 * @brief Extract upper-triangular n×n block from QR result
 */
static void copy_upper_nxn_from_qr(float *restrict Udst,
                                   const float *restrict Rsrc,
                                   uint16_t n, uint16_t ldR)
{
    for (uint16_t i = 0; i < n; ++i)
    {
        const float *row = Rsrc + (size_t)i * ldR;

        // Zero below diagonal
        for (uint16_t j = 0; j < i; ++j)
            Udst[(size_t)i * n + j] = 0.0f;

        // Copy upper triangle
        memcpy(Udst + (size_t)i * n + i, row + i, (size_t)(n - i) * sizeof(float));
    }
}

/**
 * @brief BLAS-3 rank-k Cholesky update using workspace
 *
 * @details Algorithm:
 * 1. Build augmented matrix M = [U | ±X] where U is current Cholesky factor
 * 2. Compute QR decomposition: M = QR
 * 3. Extract R[1:n, 1:n] as new Cholesky factor
 *
 * Correctness: R^T R = M^T M = U^T U ± X^T X (since Q is orthogonal)
 *
 * Cost: O(n²k) for QR (dominated by GEMM at ~20 GFLOPS)
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
 * @note Uses blocked QR with GEMM for BLAS-3 efficiency
 * @note ZERO allocations - all buffers from workspace
 */
static int cholupdatek_blockqr_ws(cholupdate_workspace *ws,
                                  float *restrict L_or_U,
                                  const float *restrict X,
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

    const uint16_t m_cols = (uint16_t)(n + k);
    float *M = ws->M;
    float *R = ws->R;
    float *Utmp = ws->Utmp;

    // =========================================================================
    // Step 1: Build M = [U | ±X]
    // =========================================================================

    if (is_upper)
    {
        // Copy upper triangle of U to first n columns
        for (uint16_t i = 0; i < n; ++i)
        {
            float *dst = M + (size_t)i * m_cols;
            const float *src = L_or_U + (size_t)i * n;

            // Zero below diagonal
            for (uint16_t j = 0; j < i; ++j)
                dst[j] = 0.0f;

            // Copy upper triangle
            memcpy(dst + i, src + i, (size_t)(n - i) * sizeof(float));
        }
    }
    else
    {
        // Extract upper triangle from lower triangular storage
        for (uint16_t i = 0; i < n; ++i)
        {
            float *dst = M + (size_t)i * m_cols;

            // Build U^T from L (L is stored lower triangular)
            for (uint16_t j = 0; j < i; ++j)
                dst[j] = L_or_U[(size_t)j * n + i];

            dst[i] = L_or_U[(size_t)i * n + i];

            for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
                dst[j] = 0.0f;
        }
    }

    // Copy scaled X into columns [n : n+k]
    const float s = (add >= 0) ? 1.0f : -1.0f;
    for (uint16_t i = 0; i < n; ++i)
    {
        float *dst = M + (size_t)i * m_cols + n;
        const float *src = X + (size_t)i * k;

        for (uint16_t j = 0; j < k; ++j)
            dst[j] = s * src[j];
    }

    // =========================================================================
    // Step 2: QR decomposition M = QR (GEMM-ACCELERATED)
    // =========================================================================
    // only_R=true: We don't need Q, just R
    // This saves computation and is safe since we pre-allocated R

    int rc = qr_ws_blocked_inplace(ws->qr_ws, M, NULL, R, n, m_cols, true);
    if (rc)
        return rc;

    // =========================================================================
    // Step 3: Extract new Cholesky factor from R[1:n, 1:n]
    // =========================================================================

    if (is_upper)
    {
        // Directly copy upper triangle to output
        copy_upper_nxn_from_qr(L_or_U, R, n, m_cols);
    }
    else
    {
        // Need to transpose: L = U^T
        copy_upper_nxn_from_qr(Utmp, R, n, m_cols);

        // Transpose Utmp to L_or_U (lower triangular storage)
        for (uint16_t i = 0; i < n; ++i)
        {
            for (uint16_t j = 0; j < i; ++j)
            {
                L_or_U[(size_t)i * n + j] = Utmp[(size_t)j * n + i];
            }
            L_or_U[(size_t)i * n + i] = Utmp[(size_t)i * n + i];

            // Zero above diagonal (strictly lower triangular storage)
            for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
            {
                L_or_U[(size_t)i * n + j] = 0.0f;
            }
        }
    }

    return 0;
}

//==============================================================================
// SMART AUTO-DISPATCH API
//==============================================================================

/**
 * @brief Smart rank-k Cholesky update with automatic algorithm selection
 *
 * @details Automatically chooses between:
 * - Tiled rank-1: Best for small k (k < 8) or small matrices
 * - Blocked QR: Best for large k (k ≥ 8) on reasonably sized matrices
 *
 * See choose_cholupdate_method() for selection heuristics
 *
 * @param ws Pre-allocated workspace
 * @param L_or_U In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension
 * @param k Rank of update
 * @param is_upper True for upper-triangular, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure
 *
 * @note Recommended API for most use cases
 * @note HOT PATH - ZERO allocations
 */
int cholupdatek_auto_ws(cholupdate_workspace *ws,
                        float *restrict L_or_U,
                        const float *restrict X,
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

    int method = choose_cholupdate_method(n, k);

    if (method == 1)
    {
        // Large k: Use blocked QR (BLAS-3, ~20 GFLOPS)
        return cholupdatek_blockqr_ws(ws, L_or_U, X, n, k, is_upper, add);
    }
    else
    {
        // Small k: Use tiled rank-1 (cache-friendly, already SIMD)
        return cholupdatek_tiled_ws(ws, L_or_U, X, n, k, is_upper, add);
    }
}

//==============================================================================
// EXPLICIT PATH SELECTION (FOR MANUAL CONTROL)
//==============================================================================

/**
 * @brief Tiled rank-k update (explicit path selection)
 *
 * @note Use cholupdatek_auto_ws() unless you have specific reasons
 */
int cholupdatek_ws(cholupdate_workspace *ws,
                   float *restrict L,
                   const float *restrict X,
                   uint16_t n, uint16_t k,
                   bool is_upper, int add)
{
    return cholupdatek_tiled_ws(ws, L, X, n, k, is_upper, add);
}

/**
 * @brief Blocked QR update (explicit path selection)
 *
 * @note Use cholupdatek_auto_ws() unless you have specific reasons
 */
int cholupdatek_blockqr_ws(cholupdate_workspace *ws,
                           float *restrict L_or_U,
                           const float *restrict X,
                           uint16_t n, uint16_t k,
                           bool is_upper, int add)
{
    return cholupdatek_blockqr_ws(ws, L_or_U, X, n, k, is_upper, add);
}

//==============================================================================
// LEGACY API (BACKWARD COMPATIBLE - ALLOCATES TEMPORARY WORKSPACE)
//==============================================================================

/**
 * @brief Legacy tiled rank-k update
 *
 * @note Allocates workspace internally
 * @note For performance-critical code, use cholupdatek_auto_ws()
 * @deprecated Use workspace-based API for hot paths
 */
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

/**
 * @brief Legacy blocked QR update
 *
 * @note Allocates workspace internally
 * @note For performance-critical code, use cholupdatek_auto_ws()
 * @deprecated Use workspace-based API for hot paths
 */
int cholupdatek_blockqr(float *restrict L_or_U,
                        const float *restrict X,
                        uint16_t n, uint16_t k,
                        bool is_upper, int add)
{
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
        return -ENOMEM;

    int ret = cholupdatek_blockqr_ws(ws, L_or_U, X, n, k, is_upper, add);
    cholupdate_workspace_free(ws);

    return ret;
}

/**
 * @brief Legacy auto-dispatch with BLAS-3
 *
 * @note Allocates workspace internally
 * @note For performance-critical code, use cholupdatek_auto_ws()
 * @deprecated Use workspace-based API for hot paths
 */
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