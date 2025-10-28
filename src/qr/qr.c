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

/* ------------------ Householder + Panel (unblocked) ------------------ */

// robust Householder for a contiguous vector x[0..len-1]
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

// Panel QR (unblocked Householders) - vectorized
static void qrw_panel_geqr2(qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                            uint16_t k, uint16_t ib, qrw_t *RESTRICT tau_panel,
                            qrw_t *RESTRICT tmp /* len >= m */)
{
    const uint16_t end = (uint16_t)((k + ib <= n) ? (k + ib) : n);

    for (uint16_t j = k; j < end; ++j)
    {
        uint16_t rows = (uint16_t)(m - j);
        qrw_t *colj0 = A + (size_t)j * n + j;

        // Gather column (strided, keep scalar)
        for (uint16_t r = 0; r < rows; ++r)
            tmp[r] = colj0[(size_t)r * n];

        qrw_t beta;
        qrw_t tauj = qrw_householder_robust(tmp, rows, &beta);
        tau_panel[j - k] = tauj;

        // Scatter back
        for (uint16_t r = 0; r < rows; ++r)
            colj0[(size_t)r * n] = tmp[r];
        *(A + (size_t)j * n + j) = -beta;

        // Apply Householder to trailing columns (vectorized)
        if (tauj != 0 && j + 1 < end)
        {
            const uint16_t nc = (uint16_t)(end - (j + 1));

#if LINALG_SIMD_ENABLE
            // Process 4 columns at once
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

                // Horizontal sums
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

/* ------------------ Build T (LARFT) ------------------ */

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

/* ------------------ Packers ------------------ */

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

/* ------------------ Scalar Level-3 (fallback) ------------------ */

// Y = V^T * Cpack   (ib × kc)
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

// Z = T * Y   (ib × kc)
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

// Cpack = Cpack − V * Z
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

/* ------------------ AVX2 Level-3 (vectorized) ------------------ */

#if LINALG_SIMD_ENABLE
// horizontal sum for __m256
static inline float qrw_hsum8(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, s);
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

// Vectorized: Y = V^T * Cpack  (ib × kc)
// We vectorize across columns j in chunks of 16 (two 8-lane accumulators).
static void qrw_compute_Y_avx(const qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                              uint16_t k, uint16_t ib,
                              const qrw_t *RESTRICT Cpack, uint16_t m_sub,
                              uint16_t kc, qrw_t *RESTRICT Y)
{
    (void)m_sub; // not needed, we derive from m,k,p
    for (uint16_t p = 0; p < ib; ++p)
    {
        const float *vp = A + (size_t)(k + p) * n + (k + p);
        const uint16_t len = (uint16_t)(m - (k + p));

        uint16_t j = 0;
        // alignment peel for Cpack row start (row offset p)
        // Each row r contributes Cpack[(r+p)*kc + j]
        // We'll just use unaligned loads for simplicity on Cpack; we still peel to make stores aligned if desired.
        for (; j + 15 < kc; j += 16)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            // accumulate over r
            for (uint16_t r = 0; r < len; ++r)
            {
                const __m256 vv = _mm256_set1_ps(vp[(size_t)r * n]);
                const float *cptr = Cpack + (size_t)(r + p) * kc + j;
                acc0 = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr + 0), acc0);
                acc1 = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr + 8), acc1);
            }
            _mm256_storeu_ps(Y + (size_t)p * kc + j + 0, acc0);
            _mm256_storeu_ps(Y + (size_t)p * kc + j + 8, acc1);
        }
        for (; j + 7 < kc; j += 8)
        {
            __m256 acc = _mm256_setzero_ps();
            for (uint16_t r = 0; r < len; ++r)
            {
                const __m256 vv = _mm256_set1_ps(vp[(size_t)r * n]);
                const float *cptr = Cpack + (size_t)(r + p) * kc + j;
                acc = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr), acc);
            }
            _mm256_storeu_ps(Y + (size_t)p * kc + j, acc);
        }
        for (; j < kc; ++j)
        {
            float sum = 0.0f;
            for (uint16_t r = 0; r < len; ++r)
                sum += vp[(size_t)r * n] * Cpack[(size_t)(r + p) * kc + j];
            Y[(size_t)p * kc + j] = sum;
        }
    }
}

// Vectorized: Z = T * Y  (ib × kc)
// Vectorize across kc columns with 16-wide chunks; broadcast T(i,p).
static void qrw_compute_Z_avx(const qrw_t *RESTRICT T, uint16_t ib,
                              const qrw_t *RESTRICT Y, uint16_t kc,
                              qrw_t *RESTRICT Z)
{
    for (uint16_t i = 0; i < ib; ++i)
    {
        uint16_t j = 0;
        for (; j + 15 < kc; j += 16)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (uint16_t p = 0; p < ib; ++p)
            {
                const __m256 t = _mm256_set1_ps(T[(size_t)i * ib + p]);
                const float *y = Y + (size_t)p * kc + j;
                acc0 = _mm256_fmadd_ps(t, _mm256_loadu_ps(y + 0), acc0);
                acc1 = _mm256_fmadd_ps(t, _mm256_loadu_ps(y + 8), acc1);
            }
            _mm256_storeu_ps(Z + (size_t)i * kc + j + 0, acc0);
            _mm256_storeu_ps(Z + (size_t)i * kc + j + 8, acc1);
        }
        for (; j + 7 < kc; j += 8)
        {
            __m256 acc = _mm256_setzero_ps();
            for (uint16_t p = 0; p < ib; ++p)
            {
                const __m256 t = _mm256_set1_ps(T[(size_t)i * ib + p]);
                const float *y = Y + (size_t)p * kc + j;
                acc = _mm256_fmadd_ps(t, _mm256_loadu_ps(y), acc);
            }
            _mm256_storeu_ps(Z + (size_t)i * kc + j, acc);
        }
        for (; j < kc; ++j)
        {
            float sum = 0.0f;
            for (uint16_t p = 0; p < ib; ++p)
                sum += T[(size_t)i * ib + p] * Y[(size_t)p * kc + j];
            Z[(size_t)i * kc + j] = sum;
        }
    }
}

// Vectorized: Cpack = Cpack − V * Z
// Vectorize across kc columns similarly; broadcast each v_p[r] and subtract v*z row by row.
static void qrw_apply_VZ_avx(qrw_t *RESTRICT Cpack, uint16_t m_sub, uint16_t kc,
                             const qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                             uint16_t k, uint16_t ib,
                             const qrw_t *RESTRICT Z)
{
    for (uint16_t p = 0; p < ib; ++p)
    {
        const float *vp = A + (size_t)(k + p) * n + (k + p);
        const uint16_t len = (uint16_t)(m - (k + p));

        uint16_t j = 0;
        for (; j + 15 < kc; j += 16)
        {
            for (uint16_t r = 0; r < len; ++r)
            {
                const __m256 vz0 = _mm256_mul_ps(_mm256_set1_ps(vp[(size_t)r * n]),
                                                 _mm256_loadu_ps(Z + (size_t)p * kc + j + 0));
                const __m256 vz1 = _mm256_mul_ps(_mm256_set1_ps(vp[(size_t)r * n]),
                                                 _mm256_loadu_ps(Z + (size_t)p * kc + j + 8));
                float *cptr = Cpack + (size_t)(r + p) * kc + j;
                __m256 c0 = _mm256_loadu_ps(cptr + 0);
                __m256 c1 = _mm256_loadu_ps(cptr + 8);
                c0 = _mm256_sub_ps(c0, vz0);
                c1 = _mm256_sub_ps(c1, vz1);
                _mm256_storeu_ps(cptr + 0, c0);
                _mm256_storeu_ps(cptr + 8, c1);
            }
        }
        for (; j + 7 < kc; j += 8)
        {
            for (uint16_t r = 0; r < len; ++r)
            {
                const __m256 vz = _mm256_mul_ps(_mm256_set1_ps(vp[(size_t)r * n]),
                                                _mm256_loadu_ps(Z + (size_t)p * kc + j));
                float *cptr = Cpack + (size_t)(r + p) * kc + j;
                __m256 c = _mm256_loadu_ps(cptr);
                c = _mm256_sub_ps(c, vz);
                _mm256_storeu_ps(cptr, c);
            }
        }
        for (; j < kc; ++j)
        {
            for (uint16_t r = 0; r < len; ++r)
                Cpack[(size_t)(r + p) * kc + j] -= vp[(size_t)r * n] * Z[(size_t)p * kc + j];
        }
    }
}
#endif /* LINALG_SIMD_ENABLE */

/* ------------------ Blocked driver: GEQRF (in-place reflectors + tau) ------------------ */

static int qrw_geqrf_blocked_wy(qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                                uint16_t ib, qrw_t *RESTRICT tau_out)
{
    if (m == 0 || n == 0)
        return 0;
    if (ib == 0)
        ib = QRW_IB_DEFAULT;

    const uint16_t kmax = (m < n) ? m : n;
    qrw_t *tmp = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m * sizeof(qrw_t));
    if (!tmp)
        return -ENOMEM;

    uint16_t k = 0;
    while (k < kmax)
    {
        uint16_t ib_k = (uint16_t)((k + ib <= kmax) ? ib : (kmax - k));
        qrw_t *tau_panel = tau_out + k;

        // 1) Panel factorization
        qrw_panel_geqr2(A, m, n, k, ib_k, tau_panel, tmp);

        // 2) Build T
        qrw_t *T = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * ib_k * sizeof(qrw_t));
        if (!T)
        {
            linalg_aligned_free(tmp);
            return -ENOMEM;
        }
        qrw_larft(T, ib_k, A, m, n, k, tau_panel);

        // 3) Apply block reflector to trailing matrix C = A[k:m, k+ib_k:n]
        const uint16_t m_sub = (uint16_t)(m - k);
        const uint16_t nc = (uint16_t)(n - (k + ib_k));
        if (nc)
        {
            const uint16_t nc_tile = (uint16_t)LINALG_BLOCK_NC;
            const uint16_t mc_tile = (uint16_t)LINALG_BLOCK_MC;
            const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;

            // Allocate max-sized buffers for mc × kc tiles
            const uint16_t mc_actual = (m_sub < mc_tile) ? m_sub : mc_tile;
            qrw_t *Cpack = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)mc_actual * kc_tile * sizeof(qrw_t));
            qrw_t *Y = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
            qrw_t *Z = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
            if (!Cpack || !Y || !Z)
            {
                if (Cpack)
                    linalg_aligned_free(Cpack);
                if (Y)
                    linalg_aligned_free(Y);
                if (Z)
                    linalg_aligned_free(Z);
                linalg_aligned_free(T);
                linalg_aligned_free(tmp);
                return -ENOMEM;
            }

            qrw_t *C = A + (size_t)k * n + (k + ib_k);

            // Column blocking (nc)
            for (uint16_t jc = 0; jc < nc; jc += nc_tile)
            {
                uint16_t nc_chunk = (uint16_t)((jc + nc_tile <= nc) ? nc_tile : (nc - jc));

                // Row blocking (mc) within each column block
                for (uint16_t ic = 0; ic < m_sub; ic += mc_tile)
                {
                    uint16_t mc_chunk = (uint16_t)((ic + mc_tile <= m_sub) ? mc_tile : (m_sub - ic));

                    // Inner kc blocking
                    for (uint16_t c0 = 0; c0 < nc_chunk; c0 += kc_tile)
                    {
                        uint16_t kc = (uint16_t)((c0 + kc_tile <= nc_chunk) ? kc_tile : (nc_chunk - c0));

                        // Pack mc_chunk rows of C
                        for (uint16_t r = 0; r < mc_chunk; ++r)
                        {
                            const qrw_t *src = C + (size_t)(ic + r) * n + (jc + c0);
                            memcpy(Cpack + (size_t)r * kc, src, (size_t)kc * sizeof(qrw_t));
                        }

#if LINALG_SIMD_ENABLE
                        // Compute Y = V[ic:ic+mc_chunk]^T * C
                        // Only process V rows that overlap with this row block
                        qrw_compute_Y_avx_opt(A, m, n, k, ib_k, Cpack, mc_chunk, kc, Y);
                        qrw_compute_Z_avx_opt(T, ib_k, Y, kc, Z);
                        qrw_apply_VZ_avx_opt(Cpack, mc_chunk, kc, A, m, n, k, ib_k, Z);
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

            linalg_aligned_free(Cpack);
            linalg_aligned_free(Y);
            linalg_aligned_free(Z);
        }

        linalg_aligned_free(T);
        k = (uint16_t)(k + ib_k);
    }

    linalg_aligned_free(tmp);
    return 0;
}

/* ------------------ ORGQR: form Q explicitly from (A,V,tau) ------------------ */

static int qrw_orgqr_full(qrw_t *RESTRICT Q, uint16_t m,
                          const qrw_t *RESTRICT A, uint16_t n,
                          const qrw_t *RESTRICT tau, uint16_t kreflect)
{
    for (uint16_t r = 0; r < m; ++r)
    {
        for (uint16_t c = 0; c < m; ++c)
            Q[(size_t)r * m + c] = (r == c) ? 1.0f : 0.0f;
    }

    if (kreflect == 0)
        return 0;

    const uint16_t ib_def = QRW_IB_DEFAULT;

    int64_t kk = (int64_t)kreflect;
    while (kk > 0)
    {
        uint16_t ib_k = (uint16_t)((kk >= ib_def) ? ib_def : kk);
        uint16_t kstart = (uint16_t)(kk - ib_k);

        qrw_t *T = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * ib_k * sizeof(qrw_t));
        if (!T)
            return -ENOMEM;
        qrw_larft(T, ib_k, A, m, n, kstart, tau + kstart);

        const uint16_t m_sub = (uint16_t)(m - kstart);
        const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;

        qrw_t *Cpack = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m_sub * kc_tile * sizeof(qrw_t));
        qrw_t *Y = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
        qrw_t *Z = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
        if (!Cpack || !Y || !Z)
        {
            if (Cpack)
                linalg_aligned_free(Cpack);
            if (Y)
                linalg_aligned_free(Y);
            if (Z)
                linalg_aligned_free(Z);
            linalg_aligned_free(T);
            return -ENOMEM;
        }

        for (uint16_t c0 = 0; c0 < m; c0 += kc_tile)
        {
            uint16_t kc = (uint16_t)((c0 + kc_tile <= m) ? kc_tile : (m - c0));
            qrw_t *C = Q + (size_t)kstart * m;
            for (uint16_t r = 0; r < m_sub; ++r)
                memcpy(Cpack + (size_t)r * kc, C + (size_t)r * m + c0, (size_t)kc * sizeof(qrw_t));
#if LINALG_SIMD_ENABLE
            qrw_compute_Y_avx_opt(A, m, n, kstart, ib_k, Cpack, m_sub, kc, Y);
            qrw_compute_Z_avx_opt(T, ib_k, Y, kc, Z);
            qrw_apply_VZ_avx_opt(Cpack, m_sub, kc, A, m, n, kstart, ib_k, Z);
#else
            qrw_compute_Y_scalar(A, m, n, kstart, ib_k, Cpack, m_sub, kc, Y);
            qrw_compute_Z_scalar(T, ib_k, Y, kc, Z);
            qrw_apply_VZ_scalar(Cpack, m_sub, kc, A, m, n, kstart, ib_k, Z);
#endif
            for (uint16_t r = 0; r < m_sub; ++r)
                memcpy(C + (size_t)r * m + c0, Cpack + (size_t)r * kc, (size_t)kc * sizeof(qrw_t));
        }

        linalg_aligned_free(Cpack);
        linalg_aligned_free(Y);
        linalg_aligned_free(Z);
        linalg_aligned_free(T);
        kk -= ib_k;
    }
    return 0;
}

/* ===========================================================================================
 * Optional: CPQR (GEQP3) minimal (unchanged from previous drop)
 * ===========================================================================================
 */

static void qrw_swap_cols(qrw_t *A, uint16_t m, uint16_t n, uint16_t j1, uint16_t j2)
{
    if (j1 == j2)
        return;
    for (uint16_t r = 0; r < m; ++r)
    {
        qrw_t tmp = A[(size_t)r * n + j1];
        A[(size_t)r * n + j1] = A[(size_t)r * n + j2];
        A[(size_t)r * n + j2] = tmp;
    }
}

static void qrw_colnorms(const qrw_t *RESTRICT A, uint16_t m, uint16_t n, uint16_t k,
                         qrw_t *RESTRICT nrms)
{
    for (uint16_t j = k; j < n; ++j)
    {
        qrw_t s = 0;
        for (uint16_t r = k; r < m; ++r)
        {
            qrw_t v = A[(size_t)r * n + j];
            s += v * v;
        }
        nrms[j] = (qrw_t)sqrt((double)s);
    }
}

static float qrw_nrm2_tail(const float *A, uint16_t m, uint16_t n,
                           uint16_t i, uint16_t j)
{
    // ||A[i+1:m-1, j]||_2 in row-major (stride n)
    double s = 0.0;
    for (uint16_t r = (uint16_t)(i + 1); r < m; ++r)
    {
        float v = A[(size_t)r * n + j];
        s += (double)v * (double)v;
    }
    return (float)sqrt(s);
}

static inline float nrm2_from(const float *A, uint16_t m, uint16_t n,
                              uint16_t i /*start row*/, uint16_t j /*col*/)
{
    double s = 0.0;
    for (uint16_t r = i; r < m; ++r)
    {
        float v = A[(size_t)r * n + j];
        s += (double)v * (double)v;
    }
    return (float)sqrt(s);
}

static inline void swap_cols(float *A, uint16_t m, uint16_t n,
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

/* Robust Householder on contiguous vector x[0..len-1].
   Returns tau; sets v in-place with v[0]=1; outputs beta via *beta_out. */
static float hh_robust(float *RESTRICT x, uint16_t len, float *beta_out)
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
    float tau = (float)(2.0 * b2 / (sigma + b2));

    float invb = 1.0f / beta;
    for (uint16_t i = 1; i < len; ++i)
        x[i] *= invb;
    x[0] = 1.0f;
    *beta_out = beta;
    return tau;
}

/* ---------- Correct CPQR: blocked outer loop, unblocked in-panel (Level-2 updates) ---------- */

int geqp3_blocked(float *RESTRICT A, uint16_t m, uint16_t n,
                  uint16_t ib, float *RESTRICT tau, int *RESTRICT jpvt)
{
    if (m == 0 || n == 0)
        return 0;
    if (ib == 0)
        ib = QRW_IB_DEFAULT;

    const uint16_t kmax = (m < n) ? m : n;

    for (uint16_t j = 0; j < n; ++j)
        jpvt[j] = (int)j;

    float *vn1 = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    float *vn2 = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    float *work = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m * sizeof(float));
    if (!vn1 || !vn2 || !work)
    {
        if (vn1)
            linalg_aligned_free(vn1);
        if (vn2)
            linalg_aligned_free(vn2);
        if (work)
            linalg_aligned_free(work);
        return -ENOMEM;
    }

    /* Initial column norms */
    for (uint16_t j = 0; j < n; ++j)
    {
        vn1[j] = vn2[j] = nrm2_from(A, m, n, 0, j);
    }

    const float recompute_tol = 0.1f; /* similar to LAPACK’s threshold */

    for (uint16_t k = 0; k < kmax; k = (uint16_t)(k + ib))
    {
        uint16_t ib_k = (uint16_t)((k + ib <= kmax) ? ib : (kmax - k));

        /* ---- Panel (columns k .. k+ib_k-1), with full in-panel pivoting ---- */
        for (uint16_t i = 0; i < ib_k; ++i)
        {
            uint16_t col = (uint16_t)(k + i);

            /* 1) Choose pivot among col..n-1 */
            uint16_t pvt = col;
            float best = vn1[pvt];
            for (uint16_t j = (uint16_t)(col + 1); j < n; ++j)
                if (vn1[j] > best)
                {
                    best = vn1[j];
                    pvt = j;
                }

            /* swap columns col <-> pvt */
            if (pvt != col)
            {
                swap_cols(A, m, n, col, pvt);
                int tmpi = jpvt[col];
                jpvt[col] = jpvt[pvt];
                jpvt[pvt] = tmpi;
                float t = vn1[col];
                vn1[col] = vn1[pvt];
                vn1[pvt] = t;
                t = vn2[col];
                vn2[col] = vn2[pvt];
                vn2[pvt] = t;
            }

            /* 2) Householder on A[col:m-1, col] */
            const uint16_t rows = (uint16_t)(m - col);
            float *a_colcol = A + (size_t)col * n + col; /* A[col,col], then stride n downward */
            for (uint16_t r = 0; r < rows; ++r)
                work[r] = a_colcol[(size_t)r * n];

            float beta, taui = hh_robust(work, rows, &beta);
            tau[col] = taui;

            /* scatter v back; set R(col,col) = -beta */
            for (uint16_t r = 0; r < rows; ++r)
                a_colcol[(size_t)r * n] = work[r];
            A[(size_t)col * n + col] = -beta;

            /* 3) Apply H_i to trailing columns j = col+1..n-1 (Level-2) */
            if (taui != 0.0f && col + 1 < n)
            {
                for (uint16_t j = (uint16_t)(col + 1); j < n; ++j)
                {
                    /* dot = v^T * A[col:m-1, j] */
                    double dot = 0.0;
                    for (uint16_t r = 0; r < rows; ++r)
                        dot += (double)work[r] * (double)A[(size_t)(col + r) * n + j];
                    float tdot = (float)(taui * dot);
                    /* A[col:m-1, j] -= v * tdot */
                    for (uint16_t r = 0; r < rows; ++r)
                        A[(size_t)(col + r) * n + j] -= work[r] * tdot;
                }
            }

            /* 4) Update norms vn1 for j = col+1..n-1 (downdate + occasional recompute) */
            for (uint16_t j = (uint16_t)(col + 1); j < n; ++j)
            {
                float old = vn1[j];
                if (old != 0.0f)
                {
                    float aij = A[(size_t)col * n + j]; /* just-updated element at row 'col' */
                    float temp = fabsf(aij) / old;
                    float new2 = old * old * (1.0f - temp * temp);
                    vn1[j] = (new2 > 0.0f) ? sqrtf(new2) : 0.0f;
                }
                /* If estimate became unreliable, recompute exactly */
                if (vn1[j] <= recompute_tol * vn2[j])
                {
                    vn1[j] = nrm2_from(A, m, n, (uint16_t)(col + 1), j);
                    vn2[j] = vn1[j];
                }
            }
        }

        /* At this point the panel reflectors (k..k+ib_k-1) have been applied
           to ALL trailing columns already, so no extra LARFB is required. */
    }

    linalg_aligned_free(vn1);
    linalg_aligned_free(vn2);
    linalg_aligned_free(work);
    return 0;
}

/** @brief Unblocked CPQR (fallback for tiny sizes).
 *  @details Correct DGEQP3-style loop with vn1/vn2 maintenance.
 *           Overwrites A in-place: R in upper-tri, V (reflectors) below.
 *           jpvt records the column permutations; tau holds Householder scalars.
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
    float tau = (float)(2.0 * b2 / (sigma + b2));

    float invb = 1.0f / beta;
    for (uint16_t i = 1; i < len; ++i)
        x[i] *= invb;
    x[0] = 1.0f;
    *beta_out = beta;
    return tau;
}

/** @brief Correct, unblocked CPQR fallback (tiny sizes).
 *  @return 0 on success, -ENOMEM on allocation failure.
 */
static int geqp3_unblocked(float *RESTRICT A, uint16_t m, uint16_t n,
                           float *RESTRICT tau, int *RESTRICT jpvt)
{
    const uint16_t kmax = (m < n) ? m : n;
    for (uint16_t j = 0; j < n; ++j)
        jpvt[j] = (int)j;

    float *vn1 = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    float *vn2 = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    float *work = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m * sizeof(float));
    if (!vn1 || !vn2 || !work)
    {
        if (vn1)
            linalg_aligned_free(vn1);
        if (vn2)
            linalg_aligned_free(vn2);
        if (work)
            linalg_aligned_free(work);
        return -ENOMEM;
    }

    for (uint16_t j = 0; j < n; ++j)
    {
        float v = cpqr_nrm2_from(A, m, n, 0, j);
        vn1[j] = vn2[j] = v;
    }
    const float tol = 0.1f;

    for (uint16_t i = 0; i < kmax; ++i)
    {
        /* pivot: choose j maximizing vn1[j] over i..n-1 */
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

        /* Householder on A[i:m-1, i] */
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

        /* Apply H_i to columns j=i+1..n-1 */
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

        /* Downdate norms and occasional recompute */
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

    linalg_aligned_free(vn1);
    linalg_aligned_free(vn2);
    linalg_aligned_free(work);
    return 0;
}

/**
 * @brief Column-Pivoted QR (GEQP3), hybrid blocked + windowed.
 *
 * @details
 * Factors A with column pivoting so that A·P = Q·R.
 * - Panel of width @p ib is processed with a DGEQP3-style loop:
 *   pivot by @p vn1 within the active window, swap columns, form H_i,
 *   and apply H_i to the **panel + look-ahead window** only.
 *   Norms for in-window columns are downdated every step; when they
 *   deteriorate (vn1 <= tol*vn2), they are recomputed exactly.
 * - At the end of the panel, build compact-WY (V,T) and apply the
 *   **block reflector** to the **far-right remainder** (columns past the
 *   window) via GEMM-shaped AVX2 kernels: Y=VᵀC, Z=T·Y, C←C−V·Z.
 *
 * @param[in,out] A    Row-major m×n matrix; on return, R in the upper-tri,
 *                     Householder vectors (V) in the strict lower-tri.
 * @param[in]     m    Rows.
 * @param[in]     n    Cols.
 * @param[in]     ib   Panel width (e.g., 64 or 96 on 14900KF).
 * @param[in]     kw   Look-ahead window width (e.g., 128). Window is the set
 *                     of columns immediately to the right of the panel that
 *                     participate in in-panel updates/pivoting.
 * @param[out]    tau  Length ≥ min(m,n); Householder scalars.
 * @param[out]    jpvt Permutation array (size n): column j maps to jpvt[j].
 *
 * @retval 0          Success.
 * @retval -ENOMEM    Allocation failure.
 */
int geqp3_hybrid(float *RESTRICT A, uint16_t m, uint16_t n,
                 uint16_t ib, uint16_t kw,
                 float *RESTRICT tau, int *RESTRICT jpvt)
{
#if defined(CPQR_SMALL_N_THRESH)
    const uint16_t mn = (m < n) ? m : n;
    if (mn < (uint16_t)CPQR_SMALL_N_THRESH)
    {
        return geqp3_unblocked(A, m, n, tau, jpvt);
    }
#endif

    if (m == 0 || n == 0)
        return 0;
    if (ib == 0)
        ib = QRW_IB_DEFAULT;

    const uint16_t kmax = (m < n) ? m : n;
    for (uint16_t j = 0; j < n; ++j)
        jpvt[j] = (int)j;

    /* --- helpers (inline duplicates if not already present) --- */
    auto nrm2_from = [](const float *A_, uint16_t m_, uint16_t n_, uint16_t i0, uint16_t j_)
    {
        double s = 0.0;
        for (uint16_t r = i0; r < m_; ++r)
        {
            float v = A_[(size_t)r * n_ + j_];
            s += (double)v * v;
        }
        return (float)sqrt(s);
    };
    auto swap_cols = [](float *A_, uint16_t m_, uint16_t n_, uint16_t j1, uint16_t j2)
    {
        if (j1 == j2)
            return;
        for (uint16_t r = 0; r < m_; ++r)
        {
            float t = A_[(size_t)r * n_ + j1];
            A_[(size_t)r * n_ + j1] = A_[(size_t)r * n_ + j2];
            A_[(size_t)r * n_ + j2] = t;
        }
    };
    auto hh_robust = [](float *RESTRICT x, uint16_t len, float *beta_out)
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
    };

    /* --- working buffers --- */
    float *vn1 = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    float *vn2 = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    float *work = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m * sizeof(float));
    if (!vn1 || !vn2 || !work)
    {
        if (vn1)
            linalg_aligned_free(vn1);
        if (vn2)
            linalg_aligned_free(vn2);
        if (work)
            linalg_aligned_free(work);
        return -ENOMEM;
    }

    /* initial norms */
    for (uint16_t j = 0; j < n; ++j)
    {
        float v = nrm2_from(A, m, n, 0, j);
        vn1[j] = vn2[j] = v;
    }

    const float tol = 0.1f; /* recompute threshold */
    const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;

    for (uint16_t k = 0; k < kmax; k = (uint16_t)(k + ib))
    {
        uint16_t ib_k = (uint16_t)((k + ib <= kmax) ? ib : (kmax - k));
        uint16_t win_beg = (uint16_t)(k + ib_k);
        uint16_t win_end = (uint16_t)((win_beg + kw <= n) ? (win_beg + kw) : n);

        /* ---- Panel: CPQR within [k .. k+ib_k-1], pivoting over [col .. win_end-1] ---- */
        for (uint16_t i = 0; i < ib_k; ++i)
        {
            uint16_t col = (uint16_t)(k + i);

            /* 1) choose pivot in [col .. win_end-1] by vn1 */
            uint16_t pvt = col;
            float best = vn1[pvt];
            for (uint16_t j = (uint16_t)(col + 1); j < win_end; ++j)
                if (vn1[j] > best)
                {
                    best = vn1[j];
                    pvt = j;
                }

            /* swap columns (A, jpvt, vn1, vn2) */
            if (pvt != col)
            {
                swap_cols(A, m, n, col, pvt);
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

            /* 2) Householder on A[col:m-1, col] */
            const uint16_t rows = (uint16_t)(m - col);
            float *a_colcol = A + (size_t)col * n + col; /* A[col,col] then +n per row */
            for (uint16_t r = 0; r < rows; ++r)
                work[r] = a_colcol[(size_t)r * n];

            float beta, taui = hh_robust(work, rows, &beta);
            tau[col] = taui;

            for (uint16_t r = 0; r < rows; ++r)
                a_colcol[(size_t)r * n] = work[r];
            A[(size_t)col * n + col] = -beta;

            /* 3) Apply H_i to columns j∈(col+1 .. win_end-1) only (panel + window) */
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

            /* 4) Update norms vn1 for j∈(col+1 .. win_end-1); recompute if needed */
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
                    vn1[j] = nrm2_from(A, m, n, (uint16_t)(col + 1), j);
                    vn2[j] = vn1[j];
                }
            }

            /* 5) For columns entering the window on next i, refresh exact norms once */
            if (win_end < n)
            {
                uint16_t enter_beg = win_end; /* none enter until panel ends */
                (void)enter_beg;              /* kept for clarity */
            }
        }

        /* ---- End of panel: apply block reflector to FAR-RIGHT remainder [win_end .. n) ---- */
        const uint16_t m_sub = (uint16_t)(m - k);
        const uint16_t far_beg = win_end;
        if (far_beg < n)
        {
            /* Build T for this panel (V is stored in A at rows/cols k..k+ib_k-1) */
            float *T = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * ib_k * sizeof(float));
            if (!T)
            {
                linalg_aligned_free(vn1);
                linalg_aligned_free(vn2);
                linalg_aligned_free(work);
                return -ENOMEM;
            }
            qrw_larft(T, ib_k, A, m, n, k, tau + k);

            float *Cpack = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m_sub * kc_tile * sizeof(float));
            float *Y = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(float));
            float *Z = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(float));
            if (!Cpack || !Y || !Z)
            {
                if (Cpack)
                    linalg_aligned_free(Cpack);
                if (Y)
                    linalg_aligned_free(Y);
                if (Z)
                    linalg_aligned_free(Z);
                linalg_aligned_free(T);
                linalg_aligned_free(vn1);
                linalg_aligned_free(vn2);
                linalg_aligned_free(work);
                return -ENOMEM;
            }

            float *C = A + (size_t)k * n + far_beg; /* trailing far-right block */
            const uint16_t nc_far = (uint16_t)(n - far_beg);

            for (uint16_t c0 = 0; c0 < nc_far; c0 += kc_tile)
            {
                uint16_t kc = (uint16_t)((c0 + kc_tile <= nc_far) ? kc_tile : (nc_far - c0));
                qrw_pack_C(C, n, m_sub, c0, kc, Cpack);
#if LINALG_SIMD_ENABLE
                qrw_compute_Y_avx_opt(A, m, n, k, ib_k, Cpack, m_sub, kc, Y);
                qrw_compute_Z_avx_opt(T, ib_k, Y, kc, Z);
                qrw_apply_VZ_avx_opt(Cpack, m_sub, kc, A, m, n, k, ib_k, Z);
#else
                qrw_compute_Y_scalar(A, m, n, k, ib_k, Cpack, m_sub, kc, Y);
                qrw_compute_Z_scalar(T, ib_k, Y, kc, Z);
                qrw_apply_VZ_scalar(Cpack, m_sub, kc, A, m, n, k, ib_k, Z);
#endif
                qrw_unpack_C(C, n, m_sub, c0, kc, Cpack);
            }

            linalg_aligned_free(Cpack);
            linalg_aligned_free(Y);
            linalg_aligned_free(Z);
            linalg_aligned_free(T);
        }

        /* ---- Slide window: columns entering the next window get exact norms once ---- */
        uint16_t next_k = (uint16_t)(k + ib_k);
        if (next_k < kmax)
        {
            uint16_t next_win_beg = (uint16_t)(next_k + ((next_k < kmax) ? ((uint16_t)0) : 0));
            uint16_t next_win_end = (uint16_t)((next_k + ib <= kmax ? next_k + ib : kmax) + kw);
            uint16_t beg = (uint16_t)((next_k + ib <= n) ? (next_k + ib) : n);
            uint16_t end = (uint16_t)((beg + kw <= n) ? (beg + kw) : n);
            for (uint16_t j = beg; j < end; ++j)
            {
                vn1[j] = vn2[j] = nrm2_from(A, m, n, next_k, j);
            }
        }
    }

    linalg_aligned_free(vn1);
    linalg_aligned_free(vn2);
    linalg_aligned_free(work);
    return 0;
}

int geqp3(float *A, uint16_t m, uint16_t n,
          float *tau, int *jpvt,
          uint16_t ib, uint16_t kw)
{
#if defined(CPQR_SMALL_N_THRESH)
    uint16_t mn = (m < n) ? m : n;
    if (mn < (uint16_t)CPQR_SMALL_N_THRESH)
        return geqp3_unblocked(A, m, n, tau, jpvt);
#endif
    return geqp3_hybrid(A, m, n, ib, kw, tau, jpvt);
}

/* ===========================================================================================
 * Public entry: qr() — chooses blocked WY or scalar path; forms Q only if requested
 * ===========================================================================================
 */

int qr(const float *RESTRICT A, float *RESTRICT Q, float *RESTRICT R,
       uint16_t m, uint16_t n, bool only_R)
{
    if (m == 0 || n == 0)
        return -EINVAL;

    const uint16_t mn = (m < n) ? m : n;

    if (mn < LINALG_SMALL_N_THRESH || !linalg_has_avx2())
    {
        return qr_scalar(A, Q, R, m, n, only_R);
    }

    memcpy(R, A, (size_t)m * n * sizeof(float)); // Factor in R
    float *tau = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)mn * sizeof(float));
    if (!tau)
        return -ENOMEM;

    int rc = qrw_geqrf_blocked_wy(R, m, n, QRW_IB_DEFAULT, tau);
    if (rc)
    {
        linalg_aligned_free(tau);
        return rc;
    }

    if (!only_R)
    {
        rc = qrw_orgqr_full(Q, m, R, n, tau, mn);
        if (rc)
        {
            linalg_aligned_free(tau);
            return rc;
        }
    }

    linalg_aligned_free(tau);
    return 0;
}

/**
 * @file
 * @brief Column-Pivoted QR (CPQR, GEQP3) – Hybrid Blocked + Windowed Implementation.
 *
 * @details
 * This routine computes a rank-revealing QR factorization with column pivoting:
 * \f[
 *      A \, P = Q \, R
 * \f]
 * where A is m×n, P is a permutation matrix represented by @ref jpvt, Q is orthogonal,
 * and R is upper-triangular.  The algorithm follows the **hybrid compact-WY + windowed**
 * strategy used in high-performance libraries (e.g., LAPACK DGEQP3, OpenBLAS, MKL, MAGMA).
 *
 * ### Algorithm overview
 *  1. **Panel factorization (width `ib`):**
 *     - Work on a vertical block (panel) of `ib` columns starting at index `k`.
 *     - Within this panel, perform unblocked CPQR (Householder + pivoting)
 *       using *accurate* column norms (`vn1`, `vn2`) and incremental downdates.
 *     - After each Householder reflector is built, apply it **only** to the
 *       **panel + look-ahead window** of width `kw`.
 *       This keeps pivoting decisions numerically correct for all columns
 *       that can be chosen within the window.
 *
 *  2. **Norm management:**
 *     - For columns inside the window: maintain norms accurately (downdate
 *       after each reflector; recompute when `vn1[j] <= tol * vn2[j]`).
 *     - For columns beyond the window: keep *approximate* norms and
 *       recompute them exactly when they slide into the active window.
 *
 *  3. **End of panel – compact-WY update:**
 *     - Form the block reflector matrices `V` and `T` for the current panel.
 *     - Apply the block reflector \f$ Q_b = I - V T V^T \f$ to the
 *       **far-right remainder** (columns past the window) using
 *       **GEMM-shaped AVX2 kernels**:
 *       ```
 *       Y = Vᵀ · C
 *       Z = T · Y
 *       C ← C − V · Z
 *       ```
 *       These steps use packed buffers for L2/L3 reuse (`kc` width tiles)
 *       and dual-accumulator AVX2/FMA microkernels, similar to SGEMM.
 *
 *  4. **Advance window:** Move to the next panel (`k ← k + ib`),
 *     sliding the window to cover the next `kw` columns; newly-entered
 *     columns get exact norm recomputation.
 *
 * ### Data layout and outputs
 *  - Input: row-major A (m×n).
 *  - On output:
 *    - Upper triangle of A holds **R**.
 *    - Strict lower triangle stores **Householder vectors (V)**.
 *    - Array `tau[k]` stores the scalar coefficients for each reflector.
 *    - `jpvt[j]` records the column permutations (so that A·P = Q·R).
 *  - `Q` can be formed explicitly with ORGQR if needed.
 *
 * ### Numerical properties
 *  - Pivoting is **rank-revealing** and stable: norms are
 *    updated after every Householder within the window,
 *    guaranteeing correct pivot order up to machine precision.
 *  - The hybrid scheme preserves the numerical behavior of
 *    LAPACK’s DGEQP3 while achieving near-GEMM throughput.
 *
 * ### Performance characteristics
 *  - Most flops reside in the trailing-update GEMM phase (BLAS-3).
 *  - The in-window operations are Level-2 but limited to `kw` columns,
 *    maintaining cache locality and predictable cost.
 *  - All buffers (`V`, `T`, packed `Y`, `Z`) are 32-byte aligned.
 *
 * ### Tuning parameters
 *  - `ib`  : panel width (typ. 48–96 on Intel 14900KF).
 *  - `kw`  : look-ahead window width (typ. 64–192, default ≈128).
 *  - `kc`  : trailing-update tile width for packed GEMM (256–320).
 *  - `tol` : norm recompute threshold (default ≈0.1).
 *
 * ### API sketch
 * @code
 * int geqp3_hybrid(float *A, uint16_t m, uint16_t n,
 *                  uint16_t ib, uint16_t kw,
 *                  float *tau, int *jpvt);
 * @endcode
 *
 * @note Single-precision only. The algorithm can be mirrored to s/d/c/z variants.
 * @note Thread-safe if called with independent matrices.
 * @warning A is overwritten in-place; copy it if needed later.
 */