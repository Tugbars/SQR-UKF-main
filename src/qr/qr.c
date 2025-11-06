/**
 * @file
 * @brief Blocked compact-WY QR (single-precision) with AVX2/FMA kernels.
 *
 * @details
 * This implementation factors an m×n row-major matrix A into Q·R using
 * Householder reflections. It follows the LAPACK/BLAS pattern:
 *  - **Panel factorization (unblocked)**: GEQR2 over a panel of width @ref QRW_IB_DEFAULT.
 *  - **Form T (compact-WY)**: LARFT builds the ib×ib triangular T for the panel V.
 *  - **Blocked application to trailing matrix**: LARFB-style update via three BLAS-3 shaped
 *    steps: Y = Vᵀ·C, Z = T·Y, C ← C − V·Z. These are implemented with small packers and
 *    AVX2/FMA vectorized kernels (dual accumulators, contiguous loads across kc).
 *
 * **Data layout and outputs**
 *  - Input is row-major A (m×n). The routine copies A→R and factors **in-place**.
 *  - On return, the **upper triangle of R** is the R factor. The **strict lower triangle**
 *    stores the Householder reflectors V; the corresponding scalars τ are kept internally.
 *  - Q is **not** formed unless requested. When needed, ORGQR builds Q (m×m) using the same
 *    blocked machinery (no per-reflector rank-1 updates).
 *
 * **Dispatch**
 *  - For small problems (mn < @ref LINALG_SMALL_N_THRESH) or when AVX2 is unavailable,
 *    a scalar reference QR path is used.
 *  - Otherwise, the blocked compact-WY path is selected.
 *
 * **SIMD & alignment**
 *  - AVX2/FMA kernels assume 32-byte alignment for workspace allocations
 *    (enforced by linalg_aligned_alloc). Unaligned loads are used where layout
 *    prohibits alignment guarantees (e.g., packed tiles), but hot buffers are aligned.
 *
 * **Tuning knobs**
 *  - @ref QRW_IB_DEFAULT : Panel width (ib). Try 64–96 on Intel 14900KF.
 *  - @ref LINALG_BLOCK_KC : Trailing-block tile width (kc) for packed updates, e.g., 256–320.
 *  - @ref LINALG_SMALL_N_THRESH : Switch to scalar path for small mn.
 *
 * **API overview**
 *  - `int qr(const float* A, float* Q, float* R, uint16_t m, uint16_t n, bool only_R);`
 *      - Copies A→R, computes R and (optionally) Q.
 *      - Returns 0 on success, negative errno on failure.
 *  - Internal helpers: blocked GEQRF (in-place reflectors + τ), ORGQR (forms Q on demand),
 *    tiny pack/unpack, and AVX2 kernels for Y/Z/VZ.
 *  - Optional: a minimal CPQR (`geqp3_blocked`) is provided but not wired into `qr()`.
 *
 * **Numerics**
 *  - Householder vectors use a robust constructor with scaling by ‖x‖∞ to avoid
 *    overflow/underflow and Parlett’s choice for β to minimize cancellation.
 *  - Compact-WY preserves the numerical stability of classical Householder QR while
 *    improving performance via BLAS-3-like updates.
 *
 * @note  Single-precision build by default. Hooks exist to mirror s/d and add c/z variants.
 * @note  This file is single-threaded by design; parallelization can be layered around
 *        the GEMM-shaped updates if needed.
 * @warning Q and R must not alias A. All buffers must be valid and sized.
 * @warning The reflectors (V) overwrite the strict lower triangle of R; if you need A later,
 *          keep your own copy.
 *
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <immintrin.h>
#include "qr.h"

#include "linalg_simd.h" // RESTRICT, linalg_has_avx2(), LINALG_* knobs, linalg_aligned_alloc/free
// also exports mul(), inv() that qr_scalar uses

#if LINALG_SIMD_ENABLE
#include "linalg_qr_avx2_kernels.h" // Highly optimized AVX2 kernels with 6x16 register blocking
#endif

#ifndef LINALG_SMALL_N_THRESH
#define LINALG_SMALL_N_THRESH 48
#endif

#ifndef LINALG_BLOCK_KC
#define LINALG_BLOCK_KC 256
#endif

#ifndef LINALG_BLOCK_JC
#define LINALG_BLOCK_JC 64
#endif

#ifndef LINALG_BLOCK_MC
#define LINALG_BLOCK_MC 192 // Row blocking for L2 cache
#endif

#ifndef LINALG_BLOCK_NC
#define LINALG_BLOCK_NC 4096 // Column blocking for L3 cache
#endif

#ifndef QRW_IB_DEFAULT
#define QRW_IB_DEFAULT 64 // try 64 or 96 on 14900KF
#endif

#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE (36 * 1024 * 1024) // 36MB for 14900KF
#endif

_Static_assert(LINALG_DEFAULT_ALIGNMENT >= 32, "Need 32B alignment for AVX loads");

/* ===========================================================================================
 * Scalar (reference) QR (unchanged, small matrices or no-AVX fallback)
 * ===========================================================================================
 */
static int qr_scalar(const float *RESTRICT A, float *RESTRICT Q,
                     float *RESTRICT R, uint16_t m, uint16_t n, bool only_R)
{
    const uint16_t l = (m - 1 < n) ? (m - 1) : n;

    memcpy(R, A, (size_t)m * n * sizeof(float));

    float *H = (float *)malloc((size_t)m * m * sizeof(float));
    float *W = (float *)malloc((size_t)m * sizeof(float));
    float *WW = (float *)malloc((size_t)m * m * sizeof(float));
    float *Hi = (float *)malloc((size_t)m * m * sizeof(float));
    float *HiH = (float *)malloc((size_t)m * m * sizeof(float));
    float *HiR = (float *)malloc((size_t)m * n * sizeof(float));
    if (!H || !W || !WW || !Hi || !HiH || !HiR)
    {
        free(H);
        free(W);
        free(WW);
        free(Hi);
        free(HiH);
        free(HiR);
        return -ENOMEM;
    }

    memset(H, 0, (size_t)m * m * sizeof(float));
    for (uint16_t i = 0; i < m; ++i)
        H[(size_t)i * m + i] = 1.0f;

    for (uint16_t k = 0; k < l; ++k)
    {
        float s = 0.0f;
        for (uint16_t i = k; i < m; ++i)
        {
            float x = R[(size_t)i * n + k];
            s += x * x;
        }
        s = sqrtf(s);
        float Rk = R[(size_t)k * n + k];
        if (s == 0.0f)
            continue; // guard: nothing to do on this column
        if (Rk < 0.0f)
            s = -s;
        float r = sqrtf(2.0f * s * (Rk + s));
        if (r == 0.0f)
            continue; // guard: avoid division by zero

        memset(W, 0, (size_t)m * sizeof(float));
        W[k] = (Rk + s) / r;
        for (uint16_t i = k + 1; i < m; ++i)
            W[i] = R[(size_t)i * n + k] / r;

        mul(WW, W, W, m, 1, 1, m); // WW = W * Wᵀ
        for (size_t i = 0; i < (size_t)m * m; ++i)
            Hi[i] = -2.0f * WW[i];
        for (uint16_t i = 0; i < m; ++i)
            Hi[(size_t)i * m + i] += 1.0f;

        if (!only_R)
        {
            mul(HiH, Hi, H, m, m, m, m);
            memcpy(H, HiH, (size_t)m * m * sizeof(float));
        }
        mul(HiR, Hi, R, m, m, m, n);
        memcpy(R, HiR, (size_t)m * n * sizeof(float));
    }

    if (!only_R)
    {
        float *Hin = (float *)malloc((size_t)m * m * sizeof(float));
        if (!Hin)
        {
            free(H);
            free(W);
            free(WW);
            free(Hi);
            free(HiH);
            free(HiR);
            return -ENOMEM;
        }
        memcpy(Hin, H, (size_t)m * m * sizeof(float));
        int rc = inv(H, Hin, m); // separate input/output
        free(Hin);
        if (rc != 0)
        {
            free(H);
            free(W);
            free(WW);
            free(Hi);
            free(HiH);
            free(HiR);
            return -ENOTSUP;
        }
        memcpy(Q, H, (size_t)m * m * sizeof(float));
    }

    free(H);
    free(W);
    free(WW);
    free(Hi);
    free(HiH);
    free(HiR);
    return 0;
}

/* ===========================================================================================
 * Blocked compact-WY QR (single precision; scalar + AVX2 kernels)
 * ===========================================================================================
 */

typedef float qrw_t;

//==============================================================================
// HOUSEHOLDER + PANEL FACTORIZATION - UNCHANGED ALGORITHMS
//==============================================================================

/**
 * @brief Robust Householder constructor for contiguous vector
 *
 * @note UNCHANGED - numerically stable with scaling by ||x||∞
 */
static qrw_t qrw_householder_robust(qrw_t *RESTRICT x, uint16_t len, qrw_t *beta_out)
{
    if (len == 0)
    {
        *beta_out = 0;
        return 0;
    }

    qrw_t amax = 0;
    for (uint16_t i = 0; i < len; ++i)
    {
        qrw_t ax = (qrw_t)fabs((double)x[i]);
        if (ax > amax)
            amax = ax;
    }
    if (amax == 0)
    {
        *beta_out = 0;
        x[0] = 1;
        return 0;
    }

    qrw_t alpha = x[0] / amax;
    qrw_t normy2 = 0;
    for (uint16_t i = 0; i < len; ++i)
    {
        qrw_t yi = x[i] / amax;
        normy2 += yi * yi;
    }
    qrw_t sigma = normy2 - alpha * alpha;
    if (sigma <= 0)
    {
        *beta_out = -x[0];
        x[0] = 1;
        return 0;
    }

    qrw_t normy = (qrw_t)sqrt((double)(alpha * alpha + sigma));
    qrw_t beta_scaled = (alpha <= 0) ? (alpha - normy) : (-sigma / (alpha + normy));
    qrw_t beta = beta_scaled * amax;
    qrw_t b2 = beta_scaled * beta_scaled;
    qrw_t tau = (qrw_t)2.0 * b2 / (sigma + b2);

    qrw_t invb = 1.0f / beta;
    for (uint16_t i = 1; i < len; ++i)
        x[i] *= invb;
    x[0] = 1.0f;
    *beta_out = beta;
    return tau;
}

/**
 * @brief Panel QR (unblocked Householders) - VECTORIZED
 *
 * @note UNCHANGED - all optimizations preserved (4-column vectorization)
 */
static void qrw_panel_geqr2(qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                            uint16_t k, uint16_t ib, qrw_t *RESTRICT tau_panel,
                            qrw_t *RESTRICT tmp /* len >= m */)
{
    const uint16_t end = (uint16_t)((k + ib <= n) ? (k + ib) : n);

    for (uint16_t j = k; j < end; ++j)
    {
        uint16_t rows = (uint16_t)(m - j);
        qrw_t *colj0 = A + (size_t)j * n + j;

        // Gather column (strided)
        for (uint16_t r = 0; r < rows; ++r)
            tmp[r] = colj0[(size_t)r * n];

        qrw_t beta;
        qrw_t tauj = qrw_householder_robust(tmp, rows, &beta);
        tau_panel[j - k] = tauj;

        // Scatter back
        for (uint16_t r = 0; r < rows; ++r)
            colj0[(size_t)r * n] = tmp[r];
        *(A + (size_t)j * n + j) = -beta;

        // Apply Householder to trailing columns (VECTORIZED - UNCHANGED)
        if (tauj != 0 && j + 1 < end)
        {
            const uint16_t nc = (uint16_t)(end - (j + 1));

#if LINALG_SIMD_ENABLE
            // Process 4 columns at once (UNCHANGED OPTIMIZATION)
            uint16_t c = (uint16_t)(j + 1);
            for (; c + 3 < end; c += 4)
            {
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();

                // Dot products with 4 columns
                uint16_t r = 0;
                for (; r + 7 < rows; r += 8)
                {
                    __m256 v = _mm256_loadu_ps(&colj0[(size_t)r * n]);
                    sum0 = _mm256_fmadd_ps(v, _mm256_loadu_ps(&A[(size_t)(j + r) * n + c + 0]), sum0);
                    sum1 = _mm256_fmadd_ps(v, _mm256_loadu_ps(&A[(size_t)(j + r) * n + c + 1]), sum1);
                    sum2 = _mm256_fmadd_ps(v, _mm256_loadu_ps(&A[(size_t)(j + r) * n + c + 2]), sum2);
                    sum3 = _mm256_fmadd_ps(v, _mm256_loadu_ps(&A[(size_t)(j + r) * n + c + 3]), sum3);
                }

                // Horizontal sums (uses kernel function - UNCHANGED)
                float s0 = qrw_hsum8_opt(sum0);
                float s1 = qrw_hsum8_opt(sum1);
                float s2 = qrw_hsum8_opt(sum2);
                float s3 = qrw_hsum8_opt(sum3);

                // Scalar remainder
                for (; r < rows; ++r)
                {
                    float v_val = colj0[(size_t)r * n];
                    s0 += v_val * A[(size_t)(j + r) * n + c + 0];
                    s1 += v_val * A[(size_t)(j + r) * n + c + 1];
                    s2 += v_val * A[(size_t)(j + r) * n + c + 2];
                    s3 += v_val * A[(size_t)(j + r) * n + c + 3];
                }

                s0 *= tauj;
                s1 *= tauj;
                s2 *= tauj;
                s3 *= tauj;

                // Apply updates (vectorized)
                __m256 s0_vec = _mm256_set1_ps(s0);
                __m256 s1_vec = _mm256_set1_ps(s1);
                __m256 s2_vec = _mm256_set1_ps(s2);
                __m256 s3_vec = _mm256_set1_ps(s3);

                r = 0;
                for (; r + 7 < rows; r += 8)
                {
                    __m256 v = _mm256_loadu_ps(&colj0[(size_t)r * n]);

                    __m256 a0 = _mm256_loadu_ps(&A[(size_t)(j + r) * n + c + 0]);
                    __m256 a1 = _mm256_loadu_ps(&A[(size_t)(j + r) * n + c + 1]);
                    __m256 a2 = _mm256_loadu_ps(&A[(size_t)(j + r) * n + c + 2]);
                    __m256 a3 = _mm256_loadu_ps(&A[(size_t)(j + r) * n + c + 3]);

                    a0 = _mm256_fnmadd_ps(v, s0_vec, a0);
                    a1 = _mm256_fnmadd_ps(v, s1_vec, a1);
                    a2 = _mm256_fnmadd_ps(v, s2_vec, a2);
                    a3 = _mm256_fnmadd_ps(v, s3_vec, a3);

                    _mm256_storeu_ps(&A[(size_t)(j + r) * n + c + 0], a0);
                    _mm256_storeu_ps(&A[(size_t)(j + r) * n + c + 1], a1);
                    _mm256_storeu_ps(&A[(size_t)(j + r) * n + c + 2], a2);
                    _mm256_storeu_ps(&A[(size_t)(j + r) * n + c + 3], a3);
                }

                // Scalar remainder rows
                for (; r < rows; ++r)
                {
                    float v_val = colj0[(size_t)r * n];
                    A[(size_t)(j + r) * n + c + 0] -= v_val * s0;
                    A[(size_t)(j + r) * n + c + 1] -= v_val * s1;
                    A[(size_t)(j + r) * n + c + 2] -= v_val * s2;
                    A[(size_t)(j + r) * n + c + 3] -= v_val * s3;
                }
            }

            // Remainder columns (scalar)
            for (; c < end; ++c)
#else
            for (uint16_t c = (uint16_t)(j + 1); c < end; ++c)
#endif
            {
                qrw_t sum = 0;
                for (uint16_t r = 0; r < rows; ++r)
                    sum += colj0[(size_t)r * n] * A[(size_t)(j + r) * n + c];
                sum *= tauj;
                for (uint16_t r = 0; r < rows; ++r)
                    A[(size_t)(j + r) * n + c] -= colj0[(size_t)r * n] * sum;
            }
        }
    }
}

/**
 * @brief Build T matrix (LARFT) - UNCHANGED
 */
static void qrw_larft(qrw_t *RESTRICT T, uint16_t ib,
                      const qrw_t *RESTRICT A, uint16_t m, uint16_t n, uint16_t k,
                      const qrw_t *RESTRICT tau_panel)
{
    for (uint16_t i = 0; i < ib; ++i)
        for (uint16_t j = 0; j < ib; ++j)
            T[(size_t)i * ib + j] = 0;

    for (uint16_t j = 0; j < ib; ++j)
    {
        for (uint16_t i = 0; i < j; ++i)
        {
            const qrw_t *vi = A + (size_t)(k + i) * n + (k + i);
            const qrw_t *vj = A + (size_t)(k + j) * n + (k + j);
            uint16_t len_j = (uint16_t)(m - (k + j));
            qrw_t sum = vi[(size_t)(j - i) * n]; // vj[0] == 1
            for (uint16_t r = 1; r < len_j; ++r)
                sum += vi[(size_t)(j - i + r) * n] * vj[(size_t)r * n];
            T[(size_t)i * ib + j] = -tau_panel[j] * sum;
        }
        T[(size_t)j * ib + j] = tau_panel[j];

        for (int i = (int)j - 1; i >= 0; --i)
        {
            qrw_t acc = T[(size_t)i * ib + j];
            for (uint16_t p = (uint16_t)(i + 1); p < j; ++p)
                acc += T[(size_t)i * ib + p] * T[(size_t)p * ib + j];
            T[(size_t)i * ib + j] = acc;
        }
    }
}

//==============================================================================
// PACKERS - UNCHANGED
//==============================================================================

static void qrw_pack_C(const qrw_t *RESTRICT C, uint16_t ld, uint16_t m_sub,
                       uint16_t c0, uint16_t kc, qrw_t *RESTRICT Cp)
{
    for (uint16_t r = 0; r < m_sub; ++r)
    {
        const qrw_t *src = C + (size_t)r * ld + c0;
        memcpy(Cp + (size_t)r * kc, src, (size_t)kc * sizeof(qrw_t));
    }
}

static void qrw_unpack_C(qrw_t *RESTRICT C, uint16_t ld, uint16_t m_sub,
                         uint16_t c0, uint16_t kc, const qrw_t *RESTRICT Cp)
{
    for (uint16_t r = 0; r < m_sub; ++r)
    {
        qrw_t *dst = C + (size_t)r * ld + c0;
        memcpy(dst, Cp + (size_t)r * kc, (size_t)kc * sizeof(qrw_t));
    }
}

//==============================================================================
// SCALAR LEVEL-3 FALLBACKS - UNCHANGED
//==============================================================================

static void qrw_compute_Y_scalar(const qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                                 uint16_t k, uint16_t ib,
                                 const qrw_t *RESTRICT Cpack, uint16_t m_sub,
                                 uint16_t kc, qrw_t *RESTRICT Y)
{
    for (uint16_t j = 0; j < kc; ++j)
    {
        for (uint16_t p = 0; p < ib; ++p)
        {
            const qrw_t *vp = A + (size_t)(k + p) * n + (k + p);
            uint16_t len = (uint16_t)(m - (k + p));
            qrw_t sum = 0;
            for (uint16_t r = 0; r < len; ++r)
                sum += vp[(size_t)r * n] * Cpack[(size_t)(r + p) * kc + j];
            Y[(size_t)p * kc + j] = sum;
        }
    }
}

static void qrw_compute_Z_scalar(const qrw_t *RESTRICT T, uint16_t ib,
                                 const qrw_t *RESTRICT Y, uint16_t kc,
                                 qrw_t *RESTRICT Z)
{
    for (uint16_t i = 0; i < ib; ++i)
    {
        for (uint16_t j = 0; j < kc; ++j)
        {
            qrw_t sum = 0;
            for (uint16_t p = 0; p < ib; ++p)
                sum += T[(size_t)i * ib + p] * Y[(size_t)p * kc + j];
            Z[(size_t)i * kc + j] = sum;
        }
    }
}

static void qrw_apply_VZ_scalar(qrw_t *RESTRICT Cpack, uint16_t m_sub, uint16_t kc,
                                const qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                                uint16_t k, uint16_t ib,
                                const qrw_t *RESTRICT Z)
{
    for (uint16_t j = 0; j < kc; ++j)
    {
        for (uint16_t p = 0; p < ib; ++p)
        {
            const qrw_t *vp = A + (size_t)(k + p) * n + (k + p);
            uint16_t len = (uint16_t)(m - (k + p));
            qrw_t zp = Z[(size_t)p * kc + j];
            for (uint16_t r = 0; r < len; ++r)
                Cpack[(size_t)(r + p) * kc + j] -= vp[(size_t)r * n] * zp;
        }
    }
}

//==============================================================================
// BLOCKED GEQRF WITH WORKSPACE (ZERO MALLOC)
//==============================================================================

/**
 * @brief Blocked compact-WY QR using workspace (ZERO HOT-PATH ALLOCATIONS)
 *
 * @note ALL compute code unchanged - only pointer sources changed
 * @note Preserves all optimizations: 6-row blocking, fast-path, prefetch, streaming
 */
static int qrw_geqrf_blocked_wy_ws(qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                                   qr_workspace *ws)
{
    if (m == 0 || n == 0)
        return 0;

    const uint16_t ib = ws->ib;
    const uint16_t kmax = (m < n) ? m : n;

    // Use workspace buffers (NO MALLOC)
    qrw_t *tau_out = ws->tau;
    qrw_t *tmp = ws->tmp;
    qrw_t *T = ws->T;
    qrw_t *Cpack = ws->Cpack;
    qrw_t *Y = ws->Y;
    qrw_t *Z = ws->Z;

    uint16_t k = 0;
    while (k < kmax)
    {
        uint16_t ib_k = (uint16_t)((k + ib <= kmax) ? ib : (kmax - k));
        qrw_t *tau_panel = tau_out + k;

        // 1) Panel factorization (UNCHANGED - uses tmp from workspace)
        qrw_panel_geqr2(A, m, n, k, ib_k, tau_panel, tmp);

        // 2) Build T (UNCHANGED - reuses T from workspace)
        qrw_larft(T, ib_k, A, m, n, k, tau_panel);

        // 3) Apply block reflector to trailing matrix (UNCHANGED ALGORITHM)
        const uint16_t m_sub = (uint16_t)(m - k);
        const uint16_t nc = (uint16_t)(n - (k + ib_k));

        if (nc)
        {
            const uint16_t nc_tile = (uint16_t)LINALG_BLOCK_NC;
            const uint16_t mc_tile = (uint16_t)LINALG_BLOCK_MC;
            const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;

            qrw_t *C = A + (size_t)k * n + (k + ib_k);

            // Column blocking (UNCHANGED)
            for (uint16_t jc = 0; jc < nc; jc += nc_tile)
            {
                uint16_t nc_chunk = (uint16_t)((jc + nc_tile <= nc) ? nc_tile : (nc - jc));

                // Row blocking (UNCHANGED)
                for (uint16_t ic = 0; ic < m_sub; ic += mc_tile)
                {
                    uint16_t mc_chunk = (uint16_t)((ic + mc_tile <= m_sub) ? mc_tile : (m_sub - ic));

                    // Inner kc blocking (UNCHANGED)
                    for (uint16_t c0 = 0; c0 < nc_chunk; c0 += kc_tile)
                    {
                        uint16_t kc = (uint16_t)((c0 + kc_tile <= nc_chunk) ? kc_tile : (nc_chunk - c0));

                        // Pack (uses Cpack from workspace)
                        for (uint16_t r = 0; r < mc_chunk; ++r)
                        {
                            const qrw_t *src = C + (size_t)(ic + r) * n + (jc + c0);
                            memcpy(Cpack + (size_t)r * kc, src, (size_t)kc * sizeof(qrw_t));
                        }

#if LINALG_SIMD_ENABLE
                        // UNCHANGED - all kernel calls identical, use workspace Y/Z
                        qrw_compute_Y_avx_opt(A, m, n, k, ib_k, Cpack, mc_chunk, kc, Y);
                        qrw_compute_Z_avx_opt(T, ib_k, Y, kc, Z);
                        qrw_apply_VZ_avx_opt(A, m, n, k, ib_k, Cpack, mc_chunk, kc, Z);
#else
                        qrw_compute_Y_scalar(A, m, n, k, ib_k, Cpack, mc_chunk, kc, Y);
                        qrw_compute_Z_scalar(T, ib_k, Y, kc, Z);
                        qrw_apply_VZ_scalar(Cpack, mc_chunk, kc, A, m, n, k, ib_k, Z);
#endif

                        // Unpack
                        for (uint16_t r = 0; r < mc_chunk; ++r)
                        {
                            qrw_t *dst = C + (size_t)(ic + r) * n + (jc + c0);
                            memcpy(dst, Cpack + (size_t)r * kc, (size_t)kc * sizeof(qrw_t));
                        }
                    }
                }
            }
        }

        k = (uint16_t)(k + ib_k);
    }

    return 0;
}

//==============================================================================
// ORGQR (FORM Q) WITH WORKSPACE (ZERO MALLOC)
//==============================================================================

/**
 * @brief Form Q explicitly using workspace (ZERO HOT-PATH ALLOCATIONS)
 *
 * @note ALL compute code unchanged - only pointer sources changed
 */
static int qrw_orgqr_full_ws(qrw_t *RESTRICT Q, uint16_t m,
                             const qrw_t *RESTRICT A, uint16_t n,
                             const qrw_t *RESTRICT tau, uint16_t kreflect,
                             qr_workspace *ws)
{
    // Initialize Q = I
    for (uint16_t r = 0; r < m; ++r)
    {
        for (uint16_t c = 0; c < m; ++c)
            Q[(size_t)r * m + c] = (r == c) ? 1.0f : 0.0f;
    }

    if (kreflect == 0)
        return 0;

    const uint16_t ib_def = ws->ib;

    // Use workspace buffers (NO MALLOC)
    qrw_t *T = ws->T;
    qrw_t *Cpack = ws->Cpack;
    qrw_t *Y = ws->Y;
    qrw_t *Z = ws->Z;

    const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;

    int64_t kk = (int64_t)kreflect;
    while (kk > 0)
    {
        uint16_t ib_k = (uint16_t)((kk >= ib_def) ? ib_def : kk);
        uint16_t kstart = (uint16_t)(kk - ib_k);

        // Build T (UNCHANGED - reuses workspace buffer)
        qrw_larft(T, ib_k, A, m, n, kstart, tau + kstart);

        const uint16_t m_sub = (uint16_t)(m - kstart);

        for (uint16_t c0 = 0; c0 < m; c0 += kc_tile)
        {
            uint16_t kc = (uint16_t)((c0 + kc_tile <= m) ? kc_tile : (m - c0));
            qrw_t *C = Q + (size_t)kstart * m;

            // Pack (uses Cpack from workspace)
            for (uint16_t r = 0; r < m_sub; ++r)
                memcpy(Cpack + (size_t)r * kc, C + (size_t)r * m + c0, (size_t)kc * sizeof(qrw_t));

#if LINALG_SIMD_ENABLE
            // UNCHANGED - all kernel calls identical
            qrw_compute_Y_avx_opt(A, m, n, kstart, ib_k, Cpack, m_sub, kc, Y);
            qrw_compute_Z_avx_opt(T, ib_k, Y, kc, Z);
            qrw_apply_VZ_avx_opt(A, m, n, kstart, ib_k, Cpack, m_sub, kc, Z);
#else
            qrw_compute_Y_scalar(A, m, n, kstart, ib_k, Cpack, m_sub, kc, Y);
            qrw_compute_Z_scalar(T, ib_k, Y, kc, Z);
            qrw_apply_VZ_scalar(Cpack, m_sub, kc, A, m, n, kstart, ib_k, Z);
#endif

            // Unpack
            for (uint16_t r = 0; r < m_sub; ++r)
                memcpy(C + (size_t)r * m + c0, Cpack + (size_t)r * kc, (size_t)kc * sizeof(qrw_t));
        }

        kk -= ib_k;
    }

    return 0;
}

//==============================================================================
// CPQR HELPER FUNCTIONS (C-compatible, no lambdas)
//==============================================================================

/**
 * @brief Compute 2-norm of column from row i0 onwards
 */
static inline float cpqr_nrm2_from(const float *A, uint16_t m, uint16_t n,
                                   uint16_t i0, uint16_t j)
{
    double s = 0.0;
    for (uint16_t r = i0; r < m; ++r)
    {
        float v = A[(size_t)r * n + j];
        s += (double)v * (double)v;
    }
    return (float)sqrt(s);
}

/**
 * @brief Swap two columns in row-major matrix
 */
static inline void cpqr_swap_cols(float *A, uint16_t m, uint16_t n,
                                  uint16_t j1, uint16_t j2)
{
    if (j1 == j2)
        return;
    for (uint16_t r = 0; r < m; ++r)
    {
        float t = A[(size_t)r * n + j1];
        A[(size_t)r * n + j1] = A[(size_t)r * n + j2];
        A[(size_t)r * n + j2] = t;
    }
}

/**
 * @brief Robust Householder constructor (CPQR variant)
 */
static float cpqr_hh_robust(float *RESTRICT x, uint16_t len, float *beta_out)
{
    if (len == 0)
    {
        *beta_out = 0.0f;
        return 0.0f;
    }

    float amax = 0.0f;
    for (uint16_t i = 0; i < len; ++i)
    {
        float ax = fabsf(x[i]);
        if (ax > amax)
            amax = ax;
    }
    if (amax == 0.0f)
    {
        *beta_out = 0.0f;
        x[0] = 1.0f;
        for (uint16_t i = 1; i < len; ++i)
            x[i] = 0.0f;
        return 0.0f;
    }

    float alpha = x[0] / amax;
    double normy2 = 0.0;
    for (uint16_t i = 0; i < len; ++i)
    {
        double y = (double)x[i] / (double)amax;
        normy2 += y * y;
    }
    double sigma = normy2 - (double)alpha * (double)alpha;
    if (sigma <= 0.0)
    {
        *beta_out = -x[0];
        x[0] = 1.0f;
        for (uint16_t i = 1; i < len; ++i)
            x[i] = 0.0f;
        return 0.0f;
    }

    double normy = sqrt(alpha * alpha + sigma);
    double beta_scaled = (alpha <= 0.0f) ? (alpha - normy) : (-sigma / (alpha + normy));
    float beta = (float)(beta_scaled * amax);
    double b2 = beta_scaled * beta_scaled;
    float taui = (float)(2.0 * b2 / (sigma + b2));

    float invb = 1.0f / beta;
    for (uint16_t i = 1; i < len; ++i)
        x[i] *= invb;
    x[0] = 1.0f;
    *beta_out = beta;
    return taui;
}

//==============================================================================
// CPQR UNBLOCKED (FOR SMALL MATRICES)
//==============================================================================

/**
 * @brief Unblocked CPQR with workspace (ZERO MALLOC)
 */
static int geqp3_unblocked_ws(float *RESTRICT A, uint16_t m, uint16_t n,
                              float *RESTRICT tau, int *RESTRICT jpvt,
                              qr_workspace *ws)
{
    const uint16_t kmax = (m < n) ? m : n;
    for (uint16_t j = 0; j < n; ++j)
        jpvt[j] = (int)j;

    // Use workspace buffers (NO MALLOC)
    float *vn1 = ws->vn1;
    float *vn2 = ws->vn2;
    float *work = ws->work;

    // Initial norms
    for (uint16_t j = 0; j < n; ++j)
    {
        float v = cpqr_nrm2_from(A, m, n, 0, j);
        vn1[j] = vn2[j] = v;
    }
    const float tol = 0.1f;

    for (uint16_t i = 0; i < kmax; ++i)
    {
        // Choose pivot: max vn1[j] over i..n-1
        uint16_t pvt = i;
        float best = vn1[pvt];
        for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
            if (vn1[j] > best)
            {
                best = vn1[j];
                pvt = j;
            }

        if (pvt != i)
        {
            cpqr_swap_cols(A, m, n, i, pvt);
            int ti = jpvt[i];
            jpvt[i] = jpvt[pvt];
            jpvt[pvt] = ti;
            float tv = vn1[i];
            vn1[i] = vn1[pvt];
            vn1[pvt] = tv;
            tv = vn2[i];
            vn2[i] = vn2[pvt];
            vn2[pvt] = tv;
        }

        // Householder on A[i:m-1, i]
        const uint16_t rows = (uint16_t)(m - i);
        float *aii = A + (size_t)i * n + i;
        for (uint16_t r = 0; r < rows; ++r)
            work[r] = aii[(size_t)r * n];

        float beta;
        float taui = cpqr_hh_robust(work, rows, &beta);
        tau[i] = taui;

        for (uint16_t r = 0; r < rows; ++r)
            aii[(size_t)r * n] = work[r];
        A[(size_t)i * n + i] = -beta;

        // Apply H_i to columns j=i+1..n-1
        if (taui != 0.0f && (uint16_t)(i + 1) < n)
        {
            for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
            {
                double dot = 0.0;
                for (uint16_t r = 0; r < rows; ++r)
                    dot += (double)work[r] * (double)A[(size_t)(i + r) * n + j];
                float tdot = (float)(taui * dot);
                for (uint16_t r = 0; r < rows; ++r)
                    A[(size_t)(i + r) * n + j] -= work[r] * tdot;
            }
        }

        // Downdate norms and occasional recompute
        for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
        {
            float old = vn1[j];
            if (old != 0.0f)
            {
                float aij = A[(size_t)i * n + j];
                float t = fabsf(aij) / old;
                float new2 = old * old * (1.0f - t * t);
                vn1[j] = (new2 > 0.0f) ? sqrtf(new2) : 0.0f;
            }
            if (vn1[j] <= tol * vn2[j])
            {
                vn1[j] = cpqr_nrm2_from(A, m, n, (uint16_t)(i + 1), j);
                vn2[j] = vn1[j];
            }
        }
    }

    return 0;
}

//==============================================================================
// CPQR HYBRID (BLOCKED + WINDOWED) WITH WORKSPACE (ZERO HOT-PATH MALLOC)
//==============================================================================

/**
 * @brief Hybrid blocked CPQR using workspace (ALL HOT-LOOP ALLOCATIONS ELIMINATED)
 *
 * @note Preserves ALL optimizations: windowed pivoting, blocked updates, AVX2 kernels
 * @note Hot loop now uses workspace buffers - ZERO malloc/free per panel
 */
int geqp3_hybrid_ws(float *RESTRICT A, uint16_t m, uint16_t n,
                    uint16_t ib, uint16_t kw,
                    float *RESTRICT tau, int *RESTRICT jpvt,
                    qr_workspace *ws)
{
#if defined(CPQR_SMALL_N_THRESH)
    const uint16_t mn = (m < n) ? m : n;
    if (mn < (uint16_t)CPQR_SMALL_N_THRESH)
    {
        return geqp3_unblocked_ws(A, m, n, tau, jpvt, ws);
    }
#endif

    if (m == 0 || n == 0)
        return 0;
    if (ib == 0)
        ib = QRW_IB_DEFAULT;

    const uint16_t kmax = (m < n) ? m : n;
    for (uint16_t j = 0; j < n; ++j)
        jpvt[j] = (int)j;

    // Use workspace buffers (NO MALLOC)
    float *vn1 = ws->vn1;
    float *vn2 = ws->vn2;
    float *work = ws->work;
    float *T = ws->T;
    float *Cpack = ws->Cpack;
    float *Y = ws->Y;
    float *Z = ws->Z;

    // Initial norms
    for (uint16_t j = 0; j < n; ++j)
    {
        float v = cpqr_nrm2_from(A, m, n, 0, j);
        vn1[j] = vn2[j] = v;
    }

    const float tol = 0.1f;
    const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;

    // HOT LOOP - NOW ALLOCATION-FREE
    for (uint16_t k = 0; k < kmax; k = (uint16_t)(k + ib))
    {
        uint16_t ib_k = (uint16_t)((k + ib <= kmax) ? ib : (kmax - k));
        uint16_t win_beg = (uint16_t)(k + ib_k);
        uint16_t win_end = (uint16_t)((win_beg + kw <= n) ? (win_beg + kw) : n);

        // ---- Panel: CPQR within [k .. k+ib_k-1], pivoting over [col .. win_end-1] ----
        for (uint16_t i = 0; i < ib_k; ++i)
        {
            uint16_t col = (uint16_t)(k + i);

            // 1) Choose pivot in [col .. win_end-1] by vn1
            uint16_t pvt = col;
            float best = vn1[pvt];
            for (uint16_t j = (uint16_t)(col + 1); j < win_end; ++j)
                if (vn1[j] > best)
                {
                    best = vn1[j];
                    pvt = j;
                }

            // Swap columns (A, jpvt, vn1, vn2)
            if (pvt != col)
            {
                cpqr_swap_cols(A, m, n, col, pvt);
                int tj = jpvt[col];
                jpvt[col] = jpvt[pvt];
                jpvt[pvt] = tj;
                float tv = vn1[col];
                vn1[col] = vn1[pvt];
                vn1[pvt] = tv;
                tv = vn2[col];
                vn2[col] = vn2[pvt];
                vn2[pvt] = tv;
            }

            // 2) Householder on A[col:m-1, col]
            const uint16_t rows = (uint16_t)(m - col);
            float *a_colcol = A + (size_t)col * n + col;
            for (uint16_t r = 0; r < rows; ++r)
                work[r] = a_colcol[(size_t)r * n];

            float beta, taui = cpqr_hh_robust(work, rows, &beta);
            tau[col] = taui;

            for (uint16_t r = 0; r < rows; ++r)
                a_colcol[(size_t)r * n] = work[r];
            A[(size_t)col * n + col] = -beta;

            // 3) Apply H_i to columns j∈(col+1 .. win_end-1) only (panel + window)
            if (taui != 0.0f && (uint16_t)(col + 1) < win_end)
            {
                for (uint16_t j = (uint16_t)(col + 1); j < win_end; ++j)
                {
                    double dot = 0.0;
                    for (uint16_t r = 0; r < rows; ++r)
                        dot += (double)work[r] * (double)A[(size_t)(col + r) * n + j];
                    float tdot = (float)(taui * dot);
                    for (uint16_t r = 0; r < rows; ++r)
                        A[(size_t)(col + r) * n + j] -= work[r] * tdot;
                }
            }

            // 4) Update norms vn1 for j∈(col+1 .. win_end-1); recompute if needed
            for (uint16_t j = (uint16_t)(col + 1); j < win_end; ++j)
            {
                float old = vn1[j];
                if (old != 0.0f)
                {
                    float aij = A[(size_t)col * n + j];
                    float t = fabsf(aij) / old;
                    float new2 = old * old * (1.0f - t * t);
                    vn1[j] = (new2 > 0.0f) ? sqrtf(new2) : 0.0f;
                }
                if (vn1[j] <= tol * vn2[j])
                {
                    vn1[j] = cpqr_nrm2_from(A, m, n, (uint16_t)(col + 1), j);
                    vn2[j] = vn1[j];
                }
            }
        }

        // ---- End of panel: apply block reflector to FAR-RIGHT remainder [win_end .. n) ----
        // THIS SECTION NOW USES WORKSPACE BUFFERS - NO MALLOC
        const uint16_t m_sub = (uint16_t)(m - k);
        const uint16_t far_beg = win_end;

        if (far_beg < n)
        {
            // Build T (uses workspace buffer T - NO MALLOC)
            qrw_larft(T, ib_k, A, m, n, k, tau + k);

            float *C = A + (size_t)k * n + far_beg;
            const uint16_t nc_far = (uint16_t)(n - far_beg);

            // Apply block reflector (uses workspace Cpack, Y, Z - NO MALLOC)
            for (uint16_t c0 = 0; c0 < nc_far; c0 += kc_tile)
            {
                uint16_t kc = (uint16_t)((c0 + kc_tile <= nc_far) ? kc_tile : (nc_far - c0));
                qrw_pack_C(C, n, m_sub, c0, kc, Cpack);

#if LINALG_SIMD_ENABLE
                // UNCHANGED - all kernel calls identical
                qrw_compute_Y_avx_opt(A, m, n, k, ib_k, Cpack, m_sub, kc, Y);
                qrw_compute_Z_avx_opt(T, ib_k, Y, kc, Z);
                qrw_apply_VZ_avx_opt(A, m, n, k, ib_k, Cpack, m_sub, kc, Z);
#else
                qrw_compute_Y_scalar(A, m, n, k, ib_k, Cpack, m_sub, kc, Y);
                qrw_compute_Z_scalar(T, ib_k, Y, kc, Z);
                qrw_apply_VZ_scalar(Cpack, m_sub, kc, A, m, n, k, ib_k, Z);
#endif

                qrw_unpack_C(C, n, m_sub, c0, kc, Cpack);
            }
        }

        // ---- Slide window: columns entering the next window get exact norms once ----
        uint16_t next_k = (uint16_t)(k + ib_k);
        if (next_k < kmax)
        {
            uint16_t beg = (uint16_t)((next_k + ib <= n) ? (next_k + ib) : n);
            uint16_t end = (uint16_t)((beg + kw <= n) ? (beg + kw) : n);
            for (uint16_t j = beg; j < end; ++j)
            {
                vn1[j] = vn2[j] = cpqr_nrm2_from(A, m, n, next_k, j);
            }
        }
    } // END HOT LOOP - NOW COMPLETELY ALLOCATION-FREE

    return 0;
}

//==============================================================================
// WORKSPACE-BASED PUBLIC API
//==============================================================================

/**
 * @brief Execute QR using workspace (HOT PATH - ZERO MALLOC)
 *
 * @param ws Pre-allocated workspace (from qr_workspace_alloc)
 * @param A Input matrix (m×n, row-major), read-only
 * @param Q Output orthogonal matrix (m×m, row-major), may be NULL if only_R=true
 * @param R Output upper-triangular (m×n, row-major)
 * @param m Rows (must be ≤ m_max from workspace)
 * @param n Columns (must be ≤ n_max from workspace)
 * @param only_R If true, skip Q formation
 *
 * @return 0 on success, negative errno on failure
 *
 * @note ZERO allocations - fully cache-optimized
 * @note User responsible for correct dimensions (m ≤ m_max, n ≤ n_max)
 */
int qr_ws(qr_workspace *ws,
          const float *RESTRICT A,
          float *RESTRICT Q,
          float *RESTRICT R,
          uint16_t m, uint16_t n,
          bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;

    if (m == 0 || n == 0)
        return -EINVAL;

    // User's responsibility to provide correct dimensions
    // (We could add assert(m <= ws->m_max && n <= ws->n_max) in debug builds)

    const uint16_t mn = (m < n) ? m : n;

    // Small matrix or no AVX2: use scalar fallback
    if (mn < LINALG_SMALL_N_THRESH || !linalg_has_avx2())
    {
        return qr_scalar(A, Q, R, m, n, only_R);
    }

    // Copy A → R
    memcpy(R, A, (size_t)m * n * sizeof(float));

    // Factor R in-place using workspace (ZERO MALLOC)
    int rc = qrw_geqrf_blocked_wy_ws(R, m, n, ws);
    if (rc)
        return rc;

    // Form Q if requested (uses workspace - ZERO MALLOC)
    if (!only_R)
    {
        rc = qrw_orgqr_full_ws(Q, m, R, n, ws->tau, mn, ws);
        if (rc)
            return rc;
    }

    return 0;
}

/**
 * @brief Execute CPQR using workspace (HOT PATH - ZERO MALLOC)
 *
 * @param ws Pre-allocated workspace
 * @param A Input/output matrix (m×n, row-major) - factored in-place
 * @param m Rows (must be ≤ m_max from workspace)
 * @param n Columns (must be ≤ n_max from workspace)
 * @param tau Output Householder scalars (length ≥ min(m,n))
 * @param jpvt Output column permutation (length ≥ n)
 * @param ib Panel width (0 = use workspace default)
 * @param kw Look-ahead window width
 *
 * @return 0 on success, negative errno on failure
 */
int geqp3_ws(qr_workspace *ws,
             float *RESTRICT A, uint16_t m, uint16_t n,
             float *RESTRICT tau, int *RESTRICT jpvt,
             uint16_t ib, uint16_t kw)
{
    if (!ws || !A || !tau || !jpvt)
        return -EINVAL;

    if (ib == 0)
        ib = ws->ib;

    return geqp3_hybrid_ws(A, m, n, ib, kw, tau, jpvt, ws);
}

//==============================================================================
// LEGACY API (BACKWARD COMPATIBLE - CREATES TEMPORARY WORKSPACE)
//==============================================================================

/**
 * @brief Legacy QR function (UNCHANGED API - backward compatible)
 *
 * @note Allocates workspace internally - for performance-critical code, use qr_ws()
 */
int qr(const float *RESTRICT A,
       float *RESTRICT Q,
       float *RESTRICT R,
       uint16_t m, uint16_t n,
       bool only_R)
{
    if (m == 0 || n == 0)
        return -EINVAL;

    const uint16_t mn = (m < n) ? m : n;

    // Small matrix fallback
    if (mn < LINALG_SMALL_N_THRESH || !linalg_has_avx2())
    {
        return qr_scalar(A, Q, R, m, n, only_R);
    }

    // Create temporary workspace
    qr_workspace *ws = qr_workspace_alloc(m, n, 0);
    if (!ws)
        return -ENOMEM;

    // Execute using workspace
    int ret = qr_ws(ws, A, Q, R, m, n, only_R);

    // Cleanup
    qr_workspace_free(ws);

    return ret;
}

int qr_ws_inplace(qr_workspace *ws, float *RESTRICT A_inout,
                  float *RESTRICT Q, uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A_inout) return -EINVAL;
    const uint16_t mn = (m < n) ? m : n;
    
    if (mn < LINALG_SMALL_N_THRESH || !linalg_has_avx2()) {
        // Need temp for scalar path
        float *tmp = (float*)aligned_alloc32((size_t)m*n*sizeof(float));
        if (!tmp) return -ENOMEM;
        memcpy(tmp, A_inout, (size_t)m*n*sizeof(float));
        int ret = qr_scalar(tmp, Q, A_inout, m, n, only_R);
        aligned_free32(tmp);
        return ret;
    }
    
    int rc = qrw_geqrf_blocked_wy_ws(A_inout, m, n, ws);
    if (rc) return rc;
    
    if (!only_R) {
        rc = qrw_orgqr_full_ws(Q, m, A_inout, n, ws->tau, mn, ws);
        if (rc) return rc;
    }
    return 0;
}

/**
 * @brief Legacy CPQR function (UNCHANGED API - backward compatible)
 *
 * @note Allocates workspace internally - for performance-critical code, use geqp3_ws()
 */
int geqp3(float *RESTRICT A, uint16_t m, uint16_t n,
          float *RESTRICT tau, int *RESTRICT jpvt,
          uint16_t ib, uint16_t kw)
{
    // Create temporary workspace
    qr_workspace *ws = qr_workspace_alloc(m, n, ib);
    if (!ws)
        return -ENOMEM;

    // Execute using workspace
    int ret = geqp3_ws(ws, A, m, n, tau, jpvt, ib, kw);

    // Cleanup
    qr_workspace_free(ws);

    return ret;
}

/**
 * @brief Wrapper for blocked CPQR (UNCHANGED API)
 */
int geqp3_blocked(float *RESTRICT A, uint16_t m, uint16_t n,
                  uint16_t ib, float *RESTRICT tau, int *RESTRICT jpvt)
{
    return geqp3(A, m, n, tau, jpvt, ib, 128); // Default kw=128
}