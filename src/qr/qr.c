/**
 * @file qr_blocked.c
 * @brief GEMM-Accelerated Blocked QR Decomposition (OpenBLAS-style)
 *
 * Features:
 * - Level-3 BLAS for trailing matrix updates (90%+ of flops)
 * - SIMD-optimized Householder reflection generation
 * - Adaptive block size selection based on GEMM tuning
 * - Pre-planned GEMM execution for maximum performance
 * - Blocked Q formation (Level-3 instead of Level-2)
 * - In-place operation support
 * - Optional reflector storage for fast Q formation
 *
 * @author TUGBARS
 * @date 2025
 */

#include "qr.h"
#include "../gemm_2/gemm.h"          // ← FIXED: was "gemm.h"
#include "../gemm_2/gemm_planning.h" // ← FIXED: was "gemm_planning.h"
#include "../gemm_2/gemm_utils.h"    // ← FIXED: was "gemm_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>

#define USE_NAIVE_GEMM_DEBUG 0 // Set to 1 to use naive GEMM, 0 for optimized

#if USE_NAIVE_GEMM_DEBUG

/**
 * @brief Naive GEMM: C = alpha * A * B + beta * C
 *
 * All matrices row-major:
 * - A[m×k]: m rows, k columns
 * - B[k×n]: k rows, n columns
 * - C[m×n]: m rows, n columns
 */
static int naive_gemm(float *C, const float *A, const float *B,
                      uint16_t m, uint16_t k, uint16_t n,
                      float alpha, float beta)
{
    if (!C || !A || !B)
        return -1;

    //printf("    [NAIVE_GEMM] C[%d×%d] = %.2f*A[%d×%d]*B[%d×%d] + %.2f*C\n",
    //       m, n, alpha, m, k, k, n, beta);

    // Step 1: Scale existing C by beta
    if (beta == 0.0f)
    {
        memset(C, 0, (size_t)m * n * sizeof(float));
    }
    else if (beta != 1.0f)
    {
        for (uint16_t i = 0; i < m; i++)
        {
            for (uint16_t j = 0; j < n; j++)
            {
                C[i * n + j] *= beta;
            }
        }
    }

    // Step 2: C += alpha * A * B
    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (uint16_t p = 0; p < k; p++)
            {
                sum += (double)A[i * k + p] * (double)B[p * n + j];
            }
            C[i * n + j] += alpha * (float)sum;
        }
    }

    return 0;
}

// Wrapper to match GEMM_CALL signature
static int gemm_debug(float *C, const float *A, const float *B,
                      uint16_t m, uint16_t k, uint16_t n,
                      float alpha, float beta)
{
    return naive_gemm(C, A, B, m, k, n, alpha, beta);
}

#define GEMM_CALL gemm_debug

#else

// Use optimized GEMM
#define GEMM_CALL gemm_dynamic

#endif

//==============================================================================
// GEMM PLAN MANAGEMENT
//==============================================================================

#ifdef MIN
#undef MIN
#endif

#ifdef MAX
#undef MAX
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static void destroy_panel_plans(qr_gemm_plans_t *plans)
{
    if (!plans)
        return;
    gemm_plan_destroy(plans->plan_yt_c);
    gemm_plan_destroy(plans->plan_t_z);
    gemm_plan_destroy(plans->plan_y_z);
    free(plans);
}

static qr_gemm_plans_t *create_panel_plans(uint16_t m, uint16_t n, uint16_t ib)
{
    if (n == 0 || m == 0 || ib == 0)
        return NULL;

    qr_gemm_plans_t *plans = (qr_gemm_plans_t *)calloc(1, sizeof(qr_gemm_plans_t));
    if (!plans)
        return NULL;

    plans->plan_m = m;
    plans->plan_n = n;
    plans->plan_ib = ib;

    plans->plan_yt_c = gemm_plan_create(ib, m, n);
    plans->plan_t_z = gemm_plan_create(ib, ib, n);
    plans->plan_y_z = gemm_plan_create(m, ib, n);

    if (!plans->plan_yt_c || !plans->plan_t_z || !plans->plan_y_z)
    {
        destroy_panel_plans(plans);
        return NULL;
    }

    return plans;
}

//==============================================================================
// ADAPTIVE BLOCK SIZE SELECTION (FIXED CLAMPING LOGIC)
//==============================================================================

static uint16_t select_optimal_ib(uint16_t m, uint16_t n)
{
    if (m == 0 || n == 0)
        return 16;

    size_t MC, KC, NC, MR, NR;
    const uint16_t min_dim = (m < n) ? m : n;
    gemm_get_tuning(m, min_dim, n, &MC, &KC, &NC, &MR, &NR);

    // Base on KC (keeps T in L1 cache)
    uint16_t ib = (uint16_t)MIN(KC, 64);

    // Matrix shape adjustments
    const double aspect_ratio = (double)m / (double)n;

    if (m < 128 || n < 128)
    {
        ib = MIN(ib, 16);
    }
    else if (aspect_ratio > 2.0)
    {
        ib = MIN(ib, 64);
    }
    else if (aspect_ratio < 0.5)
    {
        ib = MIN(ib, 32);
    }

    // ✅ FIXED: Clamp in correct order
    // Don't let MR exceed min_dim
    if ((size_t)MR <= min_dim)
    {
        if (ib < (uint16_t)MR)
            ib = (uint16_t)MR;
    }

    // Final clamps
    if (ib > min_dim)
        ib = min_dim;
    if (ib < 8)
        ib = 8;

    return ib;
}

//==============================================================================
// SIMD-OPTIMIZED HOUSEHOLDER REFLECTION
//==============================================================================

static void householder_reflection_simd(float *x, uint16_t m, float *tau, float *beta_out)
{
    if (m == 0)
    {
        *tau = 0.0f;
        if (beta_out)
            *beta_out = 0.0f;
        return;
    }

    if (m == 1)
    {
        *tau = 0.0f;
        if (beta_out)
            *beta_out = x[0];
        x[0] = 1.0f;
        return;
    }

    const float x0 = x[0];

    // Compute norm² of x[1:m] using AVX2 with double precision accumulation
    double norm_sq = 0.0;
    size_t i = 1;

    __m256d sum = _mm256_setzero_pd();

    for (; i + 7 < m; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&x[i]);

        // Split into low/high and convert to double for accuracy
        __m128 v_lo = _mm256_castps256_ps128(v);
        __m128 v_hi = _mm256_extractf128_ps(v, 1);

        __m256d v_lo_d = _mm256_cvtps_pd(v_lo);
        __m256d v_hi_d = _mm256_cvtps_pd(v_hi);

        sum = _mm256_fmadd_pd(v_lo_d, v_lo_d, sum);
        sum = _mm256_fmadd_pd(v_hi_d, v_hi_d, sum);
    }

    // Horizontal reduction
    __m128d sum_lo = _mm256_castpd256_pd128(sum);
    __m128d sum_hi = _mm256_extractf128_pd(sum, 1);
    sum_lo = _mm_add_pd(sum_lo, sum_hi);
    sum_lo = _mm_add_pd(sum_lo, _mm_shuffle_pd(sum_lo, sum_lo, 1));
    norm_sq = _mm_cvtsd_f64(sum_lo);

    // Scalar tail
    for (; i < m; ++i)
    {
        double xi = (double)x[i];
        norm_sq += xi * xi;
    }

    if (norm_sq == 0.0)
    {
        *tau = 0.0f;
        if (beta_out)
            *beta_out = x0;
        x[0] = 1.0f;
        return;
    }

    // Compute beta and scale factor
    double alpha = (double)x0;
    double beta = -copysign(sqrt(alpha * alpha + norm_sq), alpha);
    double scale = 1.0 / (alpha - beta);

    // SIMD scaling of x[1:m]
    __m256 vscale = _mm256_set1_ps((float)scale);
    i = 1;

    for (; i + 7 < m; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&x[i]);
        v = _mm256_mul_ps(v, vscale);
        _mm256_storeu_ps(&x[i], v);
    }

    // Scalar tail
    for (; i < m; ++i)
    {
        x[i] *= (float)scale;
    }

    *tau = (float)((beta - alpha) / beta);
    if (beta_out)
        *beta_out = (float)beta; // ✅ RETURN beta for R diagonal
    x[0] = 1.0f;                 // Reflector vector v[0] = 1
}

//==============================================================================
// SCALAR REFLECTOR APPLICATION WITH EXPLICIT LEADING DIMENSION
//==============================================================================

//==============================================================================
// SIMD-OPTIMIZED REFLECTOR APPLICATION (CORRECTED)
//==============================================================================
// MEMORY LAYOUT:
//   A[m×n]:     row-major submatrix, lda = leading dimension
//   Element:    A[r * lda + c]
//   v[m]:       contiguous vector (stride 1)
//
// LANE ORDERING CONVENTION:
//   All __m256 vectors use natural order: lane[0..7] = data[i..i+7]
//   Use _mm256_setr_ps (set-reverse) for clarity
//
// ALGORITHM:
//   For each column j: A[:, j] -= tau * v * (v^T * A[:, j])
//==============================================================================

static void apply_reflector_simd(float *A, uint16_t m, uint16_t n,
                                 uint16_t lda, const float *v, float tau)
{
    if (tau == 0.0f)
        return;

    __m256 vtau = _mm256_set1_ps(tau);

    for (uint16_t j = 0; j < n; ++j)
    {
        // =====================================================================
        // Step 1: Compute dot = v^T * A[:, j] with SIMD
        // =====================================================================
        double dot = 0.0;
        uint16_t i = 0;

        __m256d sum = _mm256_setzero_pd();

        for (; i + 7 < m; i += 8)
        {
            // ✅ FIXED: Use setr_ps for natural lane ordering
            // lane[0..7] = A[rows i..i+7, column j]
            __m256 a = _mm256_setr_ps(
                A[(i + 0) * lda + j], A[(i + 1) * lda + j], A[(i + 2) * lda + j], A[(i + 3) * lda + j],
                A[(i + 4) * lda + j], A[(i + 5) * lda + j], A[(i + 6) * lda + j], A[(i + 7) * lda + j]);

            // v is contiguous: lane[0..7] = v[i..i+7]
            __m256 vv = _mm256_loadu_ps(&v[i]);

            // Convert to double for accurate accumulation
            __m128 a_lo = _mm256_castps256_ps128(a);
            __m128 a_hi = _mm256_extractf128_ps(a, 1);
            __m128 v_lo = _mm256_castps256_ps128(vv);
            __m128 v_hi = _mm256_extractf128_ps(vv, 1);

            __m256d a_lo_d = _mm256_cvtps_pd(a_lo);
            __m256d a_hi_d = _mm256_cvtps_pd(a_hi);
            __m256d v_lo_d = _mm256_cvtps_pd(v_lo);
            __m256d v_hi_d = _mm256_cvtps_pd(v_hi);

            sum = _mm256_fmadd_pd(a_lo_d, v_lo_d, sum);
            sum = _mm256_fmadd_pd(a_hi_d, v_hi_d, sum);
        }

        // Horizontal reduction
        __m128d sum_lo = _mm256_castpd256_pd128(sum);
        __m128d sum_hi = _mm256_extractf128_pd(sum, 1);
        sum_lo = _mm_add_pd(sum_lo, sum_hi);
        sum_lo = _mm_add_pd(sum_lo, _mm_shuffle_pd(sum_lo, sum_lo, 1));
        dot = _mm_cvtsd_f64(sum_lo);

        // Scalar tail
        for (; i < m; ++i)
        {
            dot += (double)v[i] * (double)A[i * lda + j];
        }

        // =====================================================================
        // Step 2: Update A[:, j] -= tau * v * dot with SIMD
        // =====================================================================
        __m256 vdot = _mm256_set1_ps((float)dot);
        i = 0;

        for (; i + 7 < m; i += 8)
        {
            // Gather column (same pattern as above)
            __m256 a = _mm256_setr_ps(
                A[(i + 0) * lda + j], A[(i + 1) * lda + j], A[(i + 2) * lda + j], A[(i + 3) * lda + j],
                A[(i + 4) * lda + j], A[(i + 5) * lda + j], A[(i + 6) * lda + j], A[(i + 7) * lda + j]);

            __m256 vv = _mm256_loadu_ps(&v[i]);

            // a -= tau * v * dot
            a = _mm256_fnmadd_ps(_mm256_mul_ps(vtau, vv), vdot, a);

            // ✅ FIXED: Scatter directly, no tmp buffer
            A[(i + 0) * lda + j] = ((float *)&a)[0];
            A[(i + 1) * lda + j] = ((float *)&a)[1];
            A[(i + 2) * lda + j] = ((float *)&a)[2];
            A[(i + 3) * lda + j] = ((float *)&a)[3];
            A[(i + 4) * lda + j] = ((float *)&a)[4];
            A[(i + 5) * lda + j] = ((float *)&a)[5];
            A[(i + 6) * lda + j] = ((float *)&a)[6];
            A[(i + 7) * lda + j] = ((float *)&a)[7];
        }

        // Scalar tail
        for (; i < m; ++i)
        {
            A[i * lda + j] -= tau * v[i] * (float)dot;
        }
    }
}

//==============================================================================
// MULTI-COLUMN SIMD VERSION (FIXED)
//==============================================================================

static void apply_reflector_simd_multi(float *A, uint16_t m, uint16_t n,
                                       uint16_t lda, const float *v, float tau)
{
    if (tau == 0.0f)
        return;

    __m256 vtau = _mm256_set1_ps(tau);
    uint16_t j = 0;

    // Process 4 columns at a time
    for (; j + 3 < n; j += 4)
    {
        // =====================================================================
        // Compute 4 dot products
        // =====================================================================
        double dots[4] = {0.0, 0.0, 0.0, 0.0};
        uint16_t i = 0;

        __m256d sums[4] = {
            _mm256_setzero_pd(), _mm256_setzero_pd(),
            _mm256_setzero_pd(), _mm256_setzero_pd()};

        for (; i + 7 < m; i += 8)
        {
            // Load v once (reused for all 4 columns)
            __m256 vv = _mm256_loadu_ps(&v[i]);
            __m128 v_lo = _mm256_castps256_ps128(vv);
            __m128 v_hi = _mm256_extractf128_ps(vv, 1);
            __m256d v_lo_d = _mm256_cvtps_pd(v_lo);
            __m256d v_hi_d = _mm256_cvtps_pd(v_hi);

            // Process 4 columns
            for (uint16_t jj = 0; jj < 4; ++jj)
            {
                // ✅ FIXED: setr_ps for natural ordering
                __m256 a = _mm256_setr_ps(
                    A[(i + 0) * lda + j + jj], A[(i + 1) * lda + j + jj],
                    A[(i + 2) * lda + j + jj], A[(i + 3) * lda + j + jj],
                    A[(i + 4) * lda + j + jj], A[(i + 5) * lda + j + jj],
                    A[(i + 6) * lda + j + jj], A[(i + 7) * lda + j + jj]);

                __m128 a_lo = _mm256_castps256_ps128(a);
                __m128 a_hi = _mm256_extractf128_ps(a, 1);
                __m256d a_lo_d = _mm256_cvtps_pd(a_lo);
                __m256d a_hi_d = _mm256_cvtps_pd(a_hi);

                sums[jj] = _mm256_fmadd_pd(a_lo_d, v_lo_d, sums[jj]);
                sums[jj] = _mm256_fmadd_pd(a_hi_d, v_hi_d, sums[jj]);
            }
        }

        // Reduce accumulators
        for (uint16_t jj = 0; jj < 4; ++jj)
        {
            __m128d sum_lo = _mm256_castpd256_pd128(sums[jj]);
            __m128d sum_hi = _mm256_extractf128_pd(sums[jj], 1);
            sum_lo = _mm_add_pd(sum_lo, sum_hi);
            sum_lo = _mm_add_pd(sum_lo, _mm_shuffle_pd(sum_lo, sum_lo, 1));
            dots[jj] = _mm_cvtsd_f64(sum_lo);
        }

        // Scalar tail for dots
        for (; i < m; ++i)
        {
            for (uint16_t jj = 0; jj < 4; ++jj)
            {
                dots[jj] += (double)v[i] * (double)A[i * lda + j + jj];
            }
        }

        // =====================================================================
        // Update 4 columns
        // =====================================================================
        i = 0;
        for (; i + 7 < m; i += 8)
        {
            __m256 vv = _mm256_loadu_ps(&v[i]);

            for (uint16_t jj = 0; jj < 4; ++jj)
            {
                // ✅ FIXED: setr_ps for clarity
                __m256 a = _mm256_setr_ps(
                    A[(i + 0) * lda + j + jj], A[(i + 1) * lda + j + jj],
                    A[(i + 2) * lda + j + jj], A[(i + 3) * lda + j + jj],
                    A[(i + 4) * lda + j + jj], A[(i + 5) * lda + j + jj],
                    A[(i + 6) * lda + j + jj], A[(i + 7) * lda + j + jj]);

                __m256 dot_vec = _mm256_set1_ps((float)dots[jj]);
                a = _mm256_fnmadd_ps(_mm256_mul_ps(vtau, vv), dot_vec, a);

                // ✅ FIXED: Direct scatter, no tmp buffer
                A[(i + 0) * lda + j + jj] = ((float *)&a)[0];
                A[(i + 1) * lda + j + jj] = ((float *)&a)[1];
                A[(i + 2) * lda + j + jj] = ((float *)&a)[2];
                A[(i + 3) * lda + j + jj] = ((float *)&a)[3];
                A[(i + 4) * lda + j + jj] = ((float *)&a)[4];
                A[(i + 5) * lda + j + jj] = ((float *)&a)[5];
                A[(i + 6) * lda + j + jj] = ((float *)&a)[6];
                A[(i + 7) * lda + j + jj] = ((float *)&a)[7];
            }
        }

        // Scalar tail
        for (; i < m; ++i)
        {
            for (uint16_t jj = 0; jj < 4; ++jj)
            {
                A[i * lda + j + jj] -= tau * v[i] * (float)dots[jj];
            }
        }
    }

    // Process remaining columns one at a time
    for (; j < n; ++j)
    {
        double dot = 0.0;
        uint16_t i = 0;

        __m256d sum = _mm256_setzero_pd();

        for (; i + 7 < m; i += 8)
        {
            __m256 a = _mm256_setr_ps(
                A[(i + 0) * lda + j], A[(i + 1) * lda + j], A[(i + 2) * lda + j], A[(i + 3) * lda + j],
                A[(i + 4) * lda + j], A[(i + 5) * lda + j], A[(i + 6) * lda + j], A[(i + 7) * lda + j]);
            __m256 vv = _mm256_loadu_ps(&v[i]);

            __m128 a_lo = _mm256_castps256_ps128(a);
            __m128 a_hi = _mm256_extractf128_ps(a, 1);
            __m128 v_lo = _mm256_castps256_ps128(vv);
            __m128 v_hi = _mm256_extractf128_ps(vv, 1);

            __m256d a_lo_d = _mm256_cvtps_pd(a_lo);
            __m256d a_hi_d = _mm256_cvtps_pd(a_hi);
            __m256d v_lo_d = _mm256_cvtps_pd(v_lo);
            __m256d v_hi_d = _mm256_cvtps_pd(v_hi);

            sum = _mm256_fmadd_pd(a_lo_d, v_lo_d, sum);
            sum = _mm256_fmadd_pd(a_hi_d, v_hi_d, sum);
        }

        __m128d sum_lo = _mm256_castpd256_pd128(sum);
        __m128d sum_hi = _mm256_extractf128_pd(sum, 1);
        sum_lo = _mm_add_pd(sum_lo, sum_hi);
        sum_lo = _mm_add_pd(sum_lo, _mm_shuffle_pd(sum_lo, sum_lo, 1));
        dot = _mm_cvtsd_f64(sum_lo);

        for (; i < m; ++i)
        {
            dot += (double)v[i] * (double)A[i * lda + j];
        }

        __m256 vdot = _mm256_set1_ps((float)dot);
        i = 0;

        for (; i + 7 < m; i += 8)
        {
            __m256 a = _mm256_setr_ps(
                A[(i + 0) * lda + j], A[(i + 1) * lda + j], A[(i + 2) * lda + j], A[(i + 3) * lda + j],
                A[(i + 4) * lda + j], A[(i + 5) * lda + j], A[(i + 6) * lda + j], A[(i + 7) * lda + j]);
            __m256 vv = _mm256_loadu_ps(&v[i]);

            a = _mm256_fnmadd_ps(_mm256_mul_ps(vtau, vv), vdot, a);

            A[(i + 0) * lda + j] = ((float *)&a)[0];
            A[(i + 1) * lda + j] = ((float *)&a)[1];
            A[(i + 2) * lda + j] = ((float *)&a)[2];
            A[(i + 3) * lda + j] = ((float *)&a)[3];
            A[(i + 4) * lda + j] = ((float *)&a)[4];
            A[(i + 5) * lda + j] = ((float *)&a)[5];
            A[(i + 6) * lda + j] = ((float *)&a)[6];
            A[(i + 7) * lda + j] = ((float *)&a)[7];
        }

        for (; i < m; ++i)
        {
            A[i * lda + j] -= tau * v[i] * (float)dot;
        }
    }
}

// Scalar fallback
static void apply_reflector_scalar(float *A, uint16_t m, uint16_t n,
                                   uint16_t lda, const float *v, float tau)
{
    if (tau == 0.0f)
        return;

    for (uint16_t j = 0; j < n; ++j)
    {
        double dot = 0.0;
        for (uint16_t i = 0; i < m; ++i)
            dot += v[i] * A[i * lda + j];

        for (uint16_t i = 0; i < m; ++i)
            A[i * lda + j] -= tau * v[i] * (float)dot;
    }
}

// Dispatch (unchanged)
static inline void apply_reflector(float *A, uint16_t m, uint16_t n,
                                   uint16_t lda, const float *v, float tau)
{
    if (n >= 8 && m >= 32)
    {
        apply_reflector_simd_multi(A, m, n, lda, v, tau);
    }
    else if (m >= 16)
    {
        apply_reflector_simd(A, m, n, lda, v, tau);
    }
    else
    {
        apply_reflector_scalar(A, m, n, lda, v, tau);
    }
}

//==============================================================================
// PANEL FACTORIZATION WITH GATHER/SCATTER FOR STRIDED COLUMNS
//==============================================================================

//==============================================================================
// PANEL FACTORIZATION (CLEAN, CORRECT VERSION)
//==============================================================================
// panel:       pointer to submatrix A[k:m, k:k+ib-1] in row-major storage
//              global element (row r, col c) → panel[(r-k) * lda + (c-k)]
// Y (out):     m_panel × ib, row-major, stores Householder vectors
// YT (out):    ib × m_panel, row-major, Y^T (for GEMM)
// tau (out):   tau_base points at tau[k] in workspace; we write tau_base[j]
// m:           m_panel = m_global - k  (rows in this panel, starting at row k)
// panel_stride: lda of the *full* matrix A (n, number of columns of A)
// ib:          block size (number of columns in this panel)
// tmp_col:     scratch buffer of length at least m_panel
//==============================================================================

static void panel_qr_simd(float *panel, float *Y, float *YT, float *tau_base,
                          uint16_t m, uint16_t panel_stride, uint16_t ib,
                          float *tmp_col)
{
    // m = number of rows in this panel (m_global - k)
    // panel is A[k: , k: ] but viewed with local row index r = 0..m-1

    for (uint16_t j = 0; j < ib; ++j)
    {
        // Number of rows affected by reflector j: rows j..m-1 in panel coords
        const uint16_t rows_this = (m > j) ? (m - j) : 0;
        if (rows_this == 0)
            break;

        // Pointer to pivot element A_local(j,j) = A_global(k+j, k+j)
        float *col0 = &panel[j * panel_stride + j];

        // ---------------------------------------------------------------------
        // 1. Gather x = A(j:m-1, j) into tmp_col[0:rows_this-1]
        //    x[i] = A_local(j + i, j) = panel[(j + i) * lda + j]
        //       → with col0 = &panel[j * lda + j], this is col0[i * lda]
        // ---------------------------------------------------------------------
        for (uint16_t i = 0; i < rows_this; ++i)
        {
            tmp_col[i] = col0[i * panel_stride];
        }

        // ---------------------------------------------------------------------
        // 2. Build Householder reflector: tmp_col → v, tau, beta
        //    v[0] = 1, v[1:] = scaled tail; beta is R diagonal
        // ---------------------------------------------------------------------
        float beta;
        householder_reflection_simd(tmp_col, rows_this, &tau_base[j], &beta);

        // ---------------------------------------------------------------------
        // 3. Scatter back:
        //    - A(j,j)   ← beta        (R diagonal)
        //    - A(j+1:,j)← v[1:]      (reflector tail)
        // ---------------------------------------------------------------------
        col0[0] = beta;
        for (uint16_t i = 1; i < rows_this; ++i)
        {
            col0[i * panel_stride] = tmp_col[i];
        }

        // ---------------------------------------------------------------------
        // 4. Apply reflector H_j to trailing columns j+1 .. ib-1 of the panel
        //    Submatrix is A(j:m-1, j+1:), with leading dimension = panel_stride
        // ---------------------------------------------------------------------
        if (j + 1 < ib)
        {
            float *trailing = &panel[j * panel_stride + (j + 1)];
            apply_reflector(trailing,
                            rows_this,  // rows in this block
                            ib - j - 1, // number of trailing columns
                            panel_stride,
                            tmp_col, // full v (v[0]=1, v[1:]=tail)
                            tau_base[j]);
        }

        // ---------------------------------------------------------------------
        // 5. Store full reflector v into Y (m×ib) and YT (ib×m)
        //
        //    In local panel coords:
        //      v_local = [ 0 .. 0, 1, v_tail ]' of length m
        //      index mapping:
        //        i < j      → Y[i,j] = 0
        //        i == j     → Y[j,j] = 1
        //        i > j      → Y[i,j] = v[i-j]
        //
        //    Y  is m×ib row-major:   Y[i*ib + j]
        //    YT is ib×m row-major:   YT[j*m + i]
        // ---------------------------------------------------------------------
        for (uint16_t i = 0; i < j; ++i)
        {
            Y[i * ib + j] = 0.0f;
            YT[j * m + i] = 0.0f;
        }

        for (uint16_t i = j; i < m; ++i)
        {
            float val = (i == j) ? 1.0f : tmp_col[i - j];
            Y[i * ib + j] = val;
            YT[j * m + i] = val;
        }
    }
}

//==============================================================================
// BUILD T MATRIX (COMPACT WY REPRESENTATION)
//==============================================================================
/**
 * @brief Build the T matrix for compact WY representation of blocked reflectors
 *
 * **Mathematical Background:**
 * 
 * Given a sequence of Householder reflectors H_0, H_1, ..., H_{ib-1}, where
 * each H_j = I - τ_j * v_j * v_j^T, we want to express their product as:
 * 
 *   H_0 * H_1 * ... * H_{ib-1} = I - Y * T * Y^T
 * 
 * where:
 *   - Y[m×ib] is a matrix whose columns are the Householder vectors v_j
 *   - T[ib×ib] is an upper triangular matrix (the "WY" factor)
 * 
 * This compact form allows us to apply all reflectors at once using Level-3
 * BLAS (matrix-matrix multiplications), which is 10-100× faster than applying
 * reflectors one at a time.
 * 
 * **Algorithm (Schreiber-Van Loan, 1989):**
 * 
 * The T matrix is built recursively. For column i, we have:
 * 
 *   T[i,i] = τ_i                                    (diagonal)
 *   T[0:i, i] = -τ_i * T[0:i, 0:i] * Y^T * Y[:,i]  (off-diagonal)
 * 
 * The recursion ensures that T remains upper triangular.
 * 
 * **References:**
 * - Schreiber, R. and Van Loan, C. (1989). "A storage-efficient WY 
 *   representation for products of Householder transformations."
 *   SIAM J. Sci. Stat. Comput., 10(1), 53-57.
 * - Bischof, C. and Van Loan, C. (1987). "The WY representation for 
 *   products of Householder matrices." SIAM J. Sci. Stat. Comput., 8(1).
 * 
 * @param Y[in]     Householder vectors [m×ib], row-major
 *                  Y[:,j] = j-th Householder vector v_j (with v_j[0] = 1)
 * @param tau[in]   Scaling factors [ib], where H_j = I - τ_j * v_j * v_j^T
 * @param T[out]    Output T matrix [ib×ib], row-major, upper triangular
 * @param m         Number of rows in Y (length of each Householder vector)
 * @param ib        Block size (number of reflectors = number of columns in Y)
 * 
 * @complexity O(ib² * m) for the dot products + O(ib³) for the triangular solve
 *             Total: O(ib² * (m + ib))
 * 
 * @note This function uses double precision internally for the dot products
 *       and matrix-vector multiply to maintain numerical accuracy.
 * 
 * @note For large ib (> 64), uses heap allocation to avoid stack overflow.
 */
static void build_T_matrix(const float *Y, const float *tau, float *T,
                           uint16_t m, uint16_t ib)
{
    //==========================================================================
    // INITIALIZATION
    //==========================================================================
    
    // Zero the entire T matrix
    memset(T, 0, (size_t)ib * ib * sizeof(float));
    
    if (ib == 0)
        return;

    //==========================================================================
    // WORKSPACE ALLOCATION
    //==========================================================================
    
    // We need a temporary vector w[0:i-1] to store intermediate results
    // when building column i of T.
    //
    // Strategy:
    //   - For ib ≤ 64: use stack allocation (fast, no malloc overhead)
    //   - For ib > 64: use heap allocation (avoid stack overflow)
    
    double *w = NULL;
    double w_stack[64];  // Stack buffer for typical block sizes
    
    if (ib <= 64)
    {
        w = w_stack;  // Use fast stack allocation
    }
    else
    {
        // Large block size - use heap to avoid stack overflow
        w = (double *)malloc(ib * sizeof(double));
        if (!w)
        {
            // Allocation failed - silently return (could add error handling)
            // In production code, you might want to return an error code
            return;
        }
    }

    //==========================================================================
    // BUILD T COLUMN BY COLUMN (RECURSIVE ALGORITHM)
    //==========================================================================
    
    // We build T one column at a time, from left to right (column 0 to ib-1).
    // Each column i depends only on columns 0:i-1, which have already been
    // computed. This is why the algorithm works.
    
    for (uint16_t i = 0; i < ib; ++i)
    {
        // Get the scaling factor for reflector H_i
        float tau_i = tau[i];
        
        //----------------------------------------------------------------------
        // STEP 1: Set diagonal element T[i,i] = τ_i
        //----------------------------------------------------------------------
        // 
        // The diagonal of T contains the Householder scaling factors.
        // This comes directly from the definition:
        //   H_i = I - τ_i * v_i * v_i^T
        
        T[i * ib + i] = tau_i;
        
        //----------------------------------------------------------------------
        // EARLY EXIT: First column or zero reflector
        //----------------------------------------------------------------------
        //
        // For the first column (i=0), there are no previous reflectors to
        // couple with, so T[0,0] = τ_0 and we're done.
        //
        // If τ_i = 0, the i-th reflector is the identity (H_i = I), so it
        // doesn't interact with previous reflectors.
        
        if (tau_i == 0.0f || i == 0)
        {
            continue;  // T[0:i-1, i] remains zero
        }
        
        //----------------------------------------------------------------------
        // STEP 2: Compute w = -τ_i * Y[:,0:i-1]^T * Y[:,i]
        //----------------------------------------------------------------------
        //
        // This is the key step. We need to compute how reflector H_i interacts
        // with all previous reflectors H_0, ..., H_{i-1}.
        //
        // Mathematically:
        //   w[j] = -τ_i * (Y[:,j]^T * Y[:,i])    for j = 0, 1, ..., i-1
        //
        // This is a set of dot products between column i and each previous
        // column j.
        //
        // Complexity: O(i * m) ≈ O(ib * m) for this step
        
        for (uint16_t j = 0; j < i; ++j)
        {
            // Compute dot product: Y[:,j]^T * Y[:,i]
            double dot = 0.0;
            
            for (uint16_t r = 0; r < m; ++r)
            {
                // Y is row-major, so element (r,c) is at Y[r * ib + c]
                double y_rj = (double)Y[r * ib + j];  // Y[r,j]
                double y_ri = (double)Y[r * ib + i];  // Y[r,i]
                
                dot += y_rj * y_ri;
            }
            
            // Store -τ_i times the dot product
            w[j] = -(double)tau_i * dot;
        }
        
        //----------------------------------------------------------------------
        // STEP 3: Compute T[0:i-1, i] = T[0:i-1, 0:i-1] * w
        //----------------------------------------------------------------------
        //
        // This is a matrix-vector multiply with the upper-left (i×i) block
        // of T that we've already computed.
        //
        // Mathematically:
        //   T[j,i] = sum_{k=0}^{i-1} T[j,k] * w[k]    for j = 0, 1, ..., i-1
        //
        // Since T is upper triangular, T[j,k] = 0 for k < j, so we only need
        // to sum from k=j to k=i-1:
        //   T[j,i] = sum_{k=j}^{i-1} T[j,k] * w[k]
        //
        // However, for clarity and simplicity, we sum over all k and rely on
        // the fact that T[j,k] = 0 for k < j.
        //
        // Complexity: O(i²) ≈ O(ib²) for this step
        
        for (uint16_t j = 0; j < i; ++j)
        {
            // Compute T[j,:] * w (row j of T times vector w)
            double sum = 0.0;
            
            for (uint16_t k = 0; k < i; ++k)
            {
                // T is row-major, so element (row,col) is at T[row * ib + col]
                double t_jk = (double)T[j * ib + k];  // T[j,k]
                
                sum += t_jk * w[k];
            }
            
            // Store the result in column i of T
            T[j * ib + i] = (float)sum;
        }
        
        //----------------------------------------------------------------------
        // Column i is now complete!
        //----------------------------------------------------------------------
        //
        // At this point, T[0:i, 0:i] is fully computed and upper triangular.
        // We can proceed to the next column.
    }
    
    //==========================================================================
    // CLEANUP
    //==========================================================================
    
    // Free heap-allocated workspace if we used it
    if (ib > 64)
        free(w);
}

//==============================================================================
// OPTIMIZED UPPER TRIANGULAR MATRIX MULTIPLY (SIMD)
//==============================================================================

static void gemm_trmm_upper(
    float *C, const float *T, const float *B,
    uint16_t ib, uint16_t n)
{
    const size_t buffer_size = (size_t)ib * n * sizeof(float);
    const size_t L3_CACHE_SIZE = 36 * 1024 * 1024; // 36MB for 14900
    const bool use_streaming = (buffer_size > L3_CACHE_SIZE / 4);

    memset(C, 0, buffer_size);

    for (uint16_t i = 0; i < ib; ++i)
    {
        const float *t_row = &T[i * ib];
        float *c_row = &C[i * n];

        for (uint16_t k = i; k < ib; ++k)
        {
            const float t_ik = t_row[k];
            const float *b_row = &B[k * n];

            uint16_t j = 0;
            __m256 vt = _mm256_set1_ps(t_ik);

            for (; j + 7 < n; j += 8)
            {
                __m256 c = _mm256_loadu_ps(&c_row[j]);
                __m256 b = _mm256_loadu_ps(&b_row[j]);
                c = _mm256_fmadd_ps(vt, b, c);

                // ✅ Use streaming store for large buffers
                if (use_streaming)
                {
                    _mm256_stream_ps(&c_row[j], c);
                }
                else
                {
                    _mm256_storeu_ps(&c_row[j], c);
                }
            }

            for (; j < n; j++)
            {
                c_row[j] += t_ik * b_row[j];
            }
        }
    }

    if (use_streaming)
    {
        _mm_sfence(); // Ensure NT stores complete
    }
}

//==============================================================================
// OPTIMIZED BLOCK REFLECTOR APPLICATION (GEMM-ACCELERATED)
//==============================================================================

static int apply_block_reflector_optimized(
    float *C,        // [m×n] trailing matrix, row-major
    const float *Y,  // [m×ib] Householder vectors, row-major
    const float *YT, // [ib×m] transposed Y, row-major
    const float *T,  // [ib×ib] WY matrix (upper triangular), row-major
    uint16_t m, uint16_t n, uint16_t ib,
    float *Z,               // [ib×n] workspace
    float *Z_temp,          // [ib×n] workspace
    qr_gemm_plans_t *plans) // Pre-created GEMM plans (or NULL)
{
    int ret;

    // Use plans only if dimensions match exactly
    const bool use_plans = (plans &&
                            m == plans->plan_m &&
                            n == plans->plan_n &&
                            ib == plans->plan_ib);

    // ============================================================
    // Step 1: Z = Y^T * C  [ib×n] = [ib×m] * [m×n]
    // ============================================================
    if (use_plans)
    {
        ret = gemm_execute_plan(plans->plan_yt_c, Z, YT, C, 1.0f, 0.0f);
    }
    else
    {
        ret = GEMM_CALL(Z, YT, C, ib, m, n, 1.0f, 0.0f);
    }
    if (ret != 0)
        return ret;

    // ============================================================
    // Step 2: Z_temp = T * Z  [ib×n] = [ib×ib] * [ib×n]
    // ============================================================
    // T is upper triangular - use specialized TRMM for small problems
    if ((size_t)ib * n < 4096)
    {
        gemm_trmm_upper(Z_temp, T, Z, ib, n);
    }
    else
    {
        // Large problem - GEMM overhead negligible
        if (use_plans)
        {
            ret = gemm_execute_plan(plans->plan_t_z, Z_temp, T, Z, 1.0f, 0.0f);
        }
        else
        {
            ret = GEMM_CALL(Z_temp, T, Z, ib, ib, n, 1.0f, 0.0f);
        }
        if (ret != 0)
            return ret;
    }

    // ============================================================
    // Step 3: C -= Y * Z_temp  [m×n] -= [m×ib] * [ib×n]
    // ============================================================
    if (use_plans)
    {
        ret = gemm_execute_plan(plans->plan_y_z, C, Y, Z_temp, -1.0f, 1.0f);
    }
    else
    {
        ret = GEMM_CALL(C, Y, Z_temp, m, ib, n, -1.0f, 1.0f);
    }

    return ret;
}

//==============================================================================
// BLOCKED Q FORMATION (LEVEL-3 BLAS)
//==============================================================================

static int form_Q_blocked(qr_workspace *ws, float *Q, uint16_t m, uint16_t n)
{
    const uint16_t kmax = (m < n) ? m : n;

    // Initialize Q = I
    memset(Q, 0, (size_t)m * m * sizeof(float));
    for (uint16_t i = 0; i < m; ++i)
        Q[i * m + i] = 1.0f;

    // Apply blocks in reverse order
    for (int kt = (kmax + ws->ib - 1) / ws->ib - 1; kt >= 0; --kt)
    {
        uint16_t k = kt * ws->ib;
        uint16_t block_size = (k + ws->ib <= kmax) ? ws->ib : (kmax - k);
        uint16_t rows_below = m - k;

        // Retrieve stored Y and T for this block
        if (!ws->Y_stored || !ws->T_stored)
        {
            return -EINVAL;
        }

        float *Y_src = &ws->Y_stored[kt * ws->Y_block_stride];
        float *T_src = &ws->T_stored[kt * ws->T_block_stride];

        memcpy(ws->Y, Y_src, rows_below * block_size * sizeof(float));
        memcpy(ws->T, T_src, block_size * block_size * sizeof(float));

        // ✅ FIXED: Rebuild YT with correct stride (rows_below, not m)
        for (uint16_t j = 0; j < block_size; ++j)
        {
            for (uint16_t i = 0; i < rows_below; ++i)
            {
                ws->YT[j * rows_below + i] = ws->Y[i * block_size + j];
            }
        }

        // Apply block reflector to Q[k:m, :]
        qr_gemm_plans_t *plans_to_use = NULL;
        if (ws->q_formation_plans && k == 0 &&
            rows_below == ws->q_formation_plans->plan_m &&
            m == ws->q_formation_plans->plan_n &&
            block_size == ws->q_formation_plans->plan_ib)
        {
            plans_to_use = ws->q_formation_plans;
        }

        int ret = apply_block_reflector_optimized(
            &Q[k * m],
            ws->Y, ws->YT, ws->T,
            rows_below, m, block_size,
            ws->Z, ws->Z_temp,
            plans_to_use);

        if (ret != 0)
            return ret;
    }

    return 0;
}

//==============================================================================
// WORKSPACE ALLOCATION (FIXED LAYOUT)
//==============================================================================

//==============================================================================
// WORKSPACE ALLOCATION (FIXED: Z/Z_temp sized for max(m_max, n_max))
//==============================================================================
/**
 * @brief Allocate workspace for blocked QR decomposition with optional reflector storage
 *
 * **Memory Layout:**
 * 
 * The workspace contains several categories of buffers:
 * 
 * 1. **Panel Factorization Buffers:**
 *    - tau[mn]:        Householder scaling factors (one per reflector)
 *    - tmp[m_max]:     Column gather/scatter buffer for strided access
 *    - work[m_max]:    General-purpose working buffer
 * 
 * 2. **WY Representation Buffers:**
 *    - T[ib×ib]:       Compact WY factor for current block (upper triangular)
 *    - Y[m_max×ib]:    Current block's Householder vectors (row-major)
 *    - YT[ib×m_max]:   Transposed Y for efficient GEMM access
 * 
 * 3. **GEMM Working Buffers:**
 *    - Z[ib×n_big]:    First GEMM workspace (Y^T * C)
 *    - Z_temp[ib×n_big]: Second GEMM workspace (T * Z)
 *    - n_big = max(m_max, n_max) to handle both:
 *        * Trailing matrix updates during factorization (n ≤ n_max)
 *        * Q formation where Q is m×m (n = m ≤ m_max)
 * 
 * 4. **Copy/Packing Buffer:**
 *    - Cpack[m_max×n_max]: Aligned copy of input matrix for in-place operation
 * 
 * 5. **Column Pivoting Buffers (for future RRQR support):**
 *    - vn1[n_max]:     Column norms (first pass)
 *    - vn2[n_max]:     Column norms (second pass / verification)
 * 
 * 6. **Reflector Storage (optional, for fast Q formation):**
 *    - Y_stored[num_blocks × m_max × ib]: All Householder vectors
 *    - T_stored[num_blocks × ib × ib]:    All WY factors
 * 
 * **Critical Fix (2025):**
 * 
 * Prior versions allocated Z/Z_temp as [ib × n_max], which caused buffer
 * overflows during Q formation for tall matrices (m > n):
 * 
 *   - During factorization: Z needs [ib × n_max] (trailing updates)
 *   - During Q formation:   Z needs [ib × m_max] (Q is m×m)
 * 
 * The fix allocates Z/Z_temp as [ib × max(m_max, n_max)] to handle both cases.
 * 
 * @param m_max              Maximum number of rows
 * @param n_max              Maximum number of columns
 * @param ib                 Block size (0 = auto-select based on cache tuning)
 * @param store_reflectors   If true, allocate storage for Y and T matrices
 *                           (required for Q formation)
 * 
 * @return Allocated workspace, or NULL on failure
 * 
 * @retval NULL if m_max or n_max is zero
 * @retval NULL if any memory allocation fails
 * 
 * @note All GEMM buffers (T, Y, YT, Z, Z_temp, Cpack) are 32-byte aligned
 *       for optimal AVX2/AVX-512 performance
 * 
 * @note Must be freed with qr_workspace_free()
 * 
 * @see qr_workspace_free()
 * @see qr_workspace_alloc() (simplified wrapper)
 */
qr_workspace *qr_workspace_alloc_ex(uint16_t m_max, uint16_t n_max,
                                    uint16_t ib, bool store_reflectors)
{
    //==========================================================================
    // INPUT VALIDATION
    //==========================================================================
    
    if (!m_max || !n_max)
        return NULL;

    //==========================================================================
    // ALLOCATE WORKSPACE STRUCTURE
    //==========================================================================
    
    qr_workspace *ws = (qr_workspace *)calloc(1, sizeof(qr_workspace));
    if (!ws)
        return NULL;

    //==========================================================================
    // DETERMINE BLOCK SIZE
    //==========================================================================
    
    // If ib=0, automatically select based on GEMM cache tuning
    // Otherwise, use the user-specified block size
    const uint16_t min_dim = (m_max < n_max) ? m_max : n_max;
    ws->m_max = m_max;
    ws->n_max = n_max;
    ws->ib = ib ? ib : select_optimal_ib(m_max, n_max);
    
    // Number of blocks needed for a min_dim×min_dim matrix
    // Example: min_dim=128, ib=16 → num_blocks = (128+15)/16 = 8
    ws->num_blocks = (min_dim + ws->ib - 1) / ws->ib;

    //==========================================================================
    // COMPUTE STORAGE STRIDES
    //==========================================================================
    
    // Each block k stores:
    //   Y_k[rows_below_k × ib]  where rows_below_k = m - k*ib
    //   T_k[ib × ib]
    //
    // For simplicity, we allocate the worst case (first block with m_max rows):
    //   Y_block_stride = m_max * ib  (bytes = m_max * ib * sizeof(float))
    //   T_block_stride = ib * ib     (bytes = ib * ib * sizeof(float))
    
    ws->Y_block_stride = (size_t)m_max * ws->ib;
    ws->T_block_stride = (size_t)ws->ib * ws->ib;

    //==========================================================================
    // ✅ CRITICAL FIX: Determine maximum column dimension for Z/Z_temp
    //==========================================================================
    
    // Z and Z_temp are used in apply_block_reflector_optimized for:
    //
    //   1. Trailing matrix updates (during factorization):
    //      - Input matrix C is [rows_below × cols_right]
    //      - Z = Y^T * C has dimension [ib × cols_right]
    //      - cols_right ≤ n_max
    //      → Need: Z[ib × n_max]
    //
    //   2. Q formation (after factorization):
    //      - Input matrix Q[k:m, :] is [rows_below × m]
    //      - Z = Y^T * Q[k:m, :] has dimension [ib × m]
    //      - m ≤ m_max
    //      → Need: Z[ib × m_max]
    //
    // Therefore, we need Z[ib × max(m_max, n_max)] to handle both cases.
    // Prior versions used n_max only, causing buffer overflows for tall matrices.
    
    const uint16_t n_big = (m_max > n_max) ? m_max : n_max;

    //==========================================================================
    // ALLOCATE PANEL FACTORIZATION BUFFERS
    //==========================================================================
    
    // tau: Householder scaling factors (one per reflector)
    // Size: min_dim (at most min(m,n) reflectors in any QR)
    ws->tau = (float *)malloc(min_dim * sizeof(float));
    
    // tmp: Column gather/scatter buffer (for strided access in panel_qr_simd)
    // Size: m_max (longest column we'll ever see)
    ws->tmp = (float *)malloc(m_max * sizeof(float));
    
    // work: General-purpose working buffer (reserved for future use)
    // Size: m_max
    ws->work = (float *)malloc(m_max * sizeof(float));

    //==========================================================================
    // ALLOCATE WY REPRESENTATION BUFFERS (32-byte aligned for SIMD)
    //==========================================================================
    
    // T: Upper triangular WY factor for current block [ib×ib]
    ws->T = (float *)gemm_aligned_alloc(32, ws->ib * ws->ib * sizeof(float));
    
    // Y: Householder vectors for current block [m_max×ib], row-major
    // Y[:,j] = j-th Householder vector (with v[0]=1, stored implicitly)
    ws->Y = (float *)gemm_aligned_alloc(32, (size_t)m_max * ws->ib * sizeof(float));
    
    // YT: Transposed Y [ib×m_max], row-major (= Y in column-major)
    // Precomputed transpose for efficient GEMM: Z = Y^T * C
    ws->YT = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * m_max * sizeof(float));

    //==========================================================================
    // ALLOCATE GEMM WORKING BUFFERS (32-byte aligned, FIXED SIZE)
    //==========================================================================
    
    // Z: First GEMM workspace [ib × n_big]
    // Used for: Z = Y^T * C in block reflector application
    ws->Z = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_big * sizeof(float));
    
    // Z_temp: Second GEMM workspace [ib × n_big]
    // Used for: Z_temp = T * Z in block reflector application
    ws->Z_temp = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_big * sizeof(float));

    //==========================================================================
    // ALLOCATE COPY/PACKING BUFFER (32-byte aligned)
    //==========================================================================
    
    // Cpack: Aligned copy of input matrix [m_max × n_max]
    // Used for in-place operation in qr_ws_blocked()
    ws->Cpack = (float *)gemm_aligned_alloc(32, (size_t)m_max * n_max * sizeof(float));

    //==========================================================================
    // ALLOCATE COLUMN PIVOTING BUFFERS (for future RRQR support)
    //==========================================================================
    
    // vn1, vn2: Column norms for rank-revealing QR (not yet used)
    // Size: n_max (one norm per column)
    ws->vn1 = (float *)malloc(n_max * sizeof(float));
    ws->vn2 = (float *)malloc(n_max * sizeof(float));

    //==========================================================================
    // COMPUTE TOTAL ALLOCATED BYTES
    //==========================================================================
    
    size_t bytes = 
        // Panel factorization buffers
        min_dim * sizeof(float) +                          // tau
        m_max * sizeof(float) * 2 +                        // tmp, work
        
        // WY representation buffers
        ws->ib * ws->ib * sizeof(float) +                  // T
        (size_t)m_max * ws->ib * sizeof(float) * 2 +       // Y, YT
        
        // GEMM working buffers (FIXED)
        (size_t)ws->ib * n_big * sizeof(float) * 2 +       // Z, Z_temp
        
        // Copy/packing buffer
        (size_t)m_max * n_max * sizeof(float) +            // Cpack
        
        // Column pivoting buffers
        n_max * sizeof(float) * 2;                         // vn1, vn2

    //==========================================================================
    // ALLOCATE REFLECTOR STORAGE (optional, for fast Q formation)
    //==========================================================================
    
    if (store_reflectors)
    {
        // Y_stored: All Householder vectors for all blocks
        // Layout: [num_blocks][m_max][ib]
        // Total: num_blocks * (m_max * ib) floats
        ws->Y_stored = (float *)gemm_aligned_alloc(32,
            ws->num_blocks * ws->Y_block_stride * sizeof(float));
        
        // T_stored: All WY factors for all blocks
        // Layout: [num_blocks][ib][ib]
        // Total: num_blocks * (ib * ib) floats
        ws->T_stored = (float *)gemm_aligned_alloc(32,
            ws->num_blocks * ws->T_block_stride * sizeof(float));
        
        // Update byte count
        bytes += ws->num_blocks * ws->Y_block_stride * sizeof(float);
        bytes += ws->num_blocks * ws->T_block_stride * sizeof(float);
    }
    else
    {
        ws->Y_stored = NULL;
        ws->T_stored = NULL;
    }

    //==========================================================================
    // VALIDATE ALL ALLOCATIONS
    //==========================================================================
    
    if (!ws->tau || !ws->tmp || !ws->work || !ws->T || !ws->Cpack ||
        !ws->Y || !ws->YT || !ws->Z || !ws->Z_temp || !ws->vn1 || !ws->vn2)
    {
        // At least one allocation failed - clean up and return NULL
        qr_workspace_free(ws);
        return NULL;
    }

    //==========================================================================
    // CREATE GEMM EXECUTION PLANS (optional optimization)
    //==========================================================================
    
    // Pre-plan GEMM operations for the first panel (exact dimensions known)
    // This avoids re-planning overhead on every block
    
    const uint16_t first_panel_cols = (n_max > ws->ib) ? (n_max - ws->ib) : 0;
    
    if (first_panel_cols > 0)
    {
        // Plans for trailing matrix update: C[m_max × first_panel_cols]
        ws->trailing_plans = create_panel_plans(m_max, first_panel_cols, ws->ib);
    }
    else
    {
        ws->trailing_plans = NULL;
    }

    // Plans for Q formation: Q[m_max × m_max]
    if (m_max >= ws->ib)
    {
        ws->q_formation_plans = create_panel_plans(m_max, m_max, ws->ib);
    }
    else
    {
        ws->q_formation_plans = NULL;
    }

    //==========================================================================
    // FINALIZE AND RETURN
    //==========================================================================
    
    ws->total_bytes = bytes;
    return ws;
}

qr_workspace *qr_workspace_alloc(uint16_t m_max, uint16_t n_max, uint16_t ib)
{
    return qr_workspace_alloc_ex(m_max, n_max, ib, true);
}

void qr_workspace_free(qr_workspace *ws)
{
    if (!ws)
        return;

    destroy_panel_plans(ws->trailing_plans);
    destroy_panel_plans(ws->q_formation_plans);

    free(ws->tau);
    free(ws->tmp);
    free(ws->work);
    gemm_aligned_free(ws->T);
    gemm_aligned_free(ws->Cpack);
    gemm_aligned_free(ws->Y);
    gemm_aligned_free(ws->YT);
    gemm_aligned_free(ws->Z);
    gemm_aligned_free(ws->Z_temp);
    gemm_aligned_free(ws->Y_stored);
    gemm_aligned_free(ws->T_stored);
    free(ws->vn1);
    free(ws->vn2);
    free(ws);
}

size_t qr_workspace_bytes(const qr_workspace *ws)
{
    return ws ? ws->total_bytes : 0;
}

//==============================================================================
// COMPLETE IN-PLACE BLOCKED QR (FULLY FIXED)
//==============================================================================

#define DEBUG_QR 0

int qr_ws_blocked_inplace(qr_workspace *ws, float *A, float *Q, float *R,
                          uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;
    if (m > ws->m_max || n > ws->n_max)
        return -EINVAL;

#if DEBUG_QR
    printf("[qr_ws_blocked_inplace] m=%d, n=%d, ib=%d, only_R=%d\n",
           m, n, ws->ib, only_R);
#endif

    float *Awork = A;
    const uint16_t kmax = (m < n) ? m : n;

    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        const uint16_t block_size = (k + ws->ib <= kmax) ? ws->ib : (kmax - k);
        const uint16_t rows_below = m - k;
        const uint16_t cols_right = n - k - block_size;

#if DEBUG_QR
        printf("  Block k=%d: block_size=%d, rows_below=%d, cols_right=%d\n",
               k, block_size, rows_below, cols_right);
#endif

        // Panel factorization
#if DEBUG_QR
        printf("  Calling panel_qr_simd...\n");
#endif
        panel_qr_simd(&Awork[k * n + k], ws->Y, ws->YT, &ws->tau[k],
                      rows_below, n, block_size, ws->tmp);

#if DEBUG_QR
        printf("  panel_qr_simd completed\n");
        printf("  Calling build_T_matrix...\n");
#endif

        // Build T matrix
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size);

#if DEBUG_QR
        printf("  build_T_matrix completed\n");
#endif

        // Store reflectors
        if (ws->Y_stored && ws->T_stored)
        {
            uint16_t block_idx = k / ws->ib;
#if DEBUG_QR
            printf("  Storing block %d (offset %zu in Y_stored)\n", 
                   block_idx, block_idx * ws->Y_block_stride);
#endif
            
            float *Y_dst = &ws->Y_stored[block_idx * ws->Y_block_stride];
            float *T_dst = &ws->T_stored[block_idx * ws->T_block_stride];

            memcpy(Y_dst, ws->Y, rows_below * block_size * sizeof(float));
            memcpy(T_dst, ws->T, block_size * block_size * sizeof(float));
        }

        // Apply block reflector to trailing matrix
        if (cols_right > 0)
        {
#if DEBUG_QR
            printf("  Applying block reflector to trailing matrix...\n");
#endif
            qr_gemm_plans_t *plans_to_use = NULL;

            if (ws->trailing_plans && k == 0 &&
                rows_below == ws->trailing_plans->plan_m &&
                cols_right == ws->trailing_plans->plan_n &&
                block_size == ws->trailing_plans->plan_ib)
            {
                plans_to_use = ws->trailing_plans;
            }

            int ret = apply_block_reflector_optimized(
                &Awork[k * n + k + block_size],
                ws->Y, ws->YT, ws->T,
                rows_below, cols_right, block_size,
                ws->Z, ws->Z_temp,
                plans_to_use);

            if (ret != 0)
            {
#if DEBUG_QR
                printf("  ERROR: apply_block_reflector_optimized returned %d\n", ret);
#endif
                return ret;
            }
#if DEBUG_QR
            printf("  Block reflector applied successfully\n");
#endif
        }

#if DEBUG_QR
        printf("  Block %d completed\n\n", k / ws->ib);
#endif
    }

    // Extract R
#if DEBUG_QR
    printf("Extracting R matrix...\n");
#endif
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            R[i * n + j] = (i <= j) ? Awork[i * n + j] : 0.0f;
        }
    }

    // Form Q
    if (!only_R && Q)
    {
#if DEBUG_QR
        printf("Forming Q matrix...\n");
#endif
        if (!ws->Y_stored || !ws->T_stored)
        {
            return -EINVAL;
        }

        int ret = form_Q_blocked(ws, Q, m, n);
        if (ret != 0)
            return ret;
    }

    return 0;
}

//==============================================================================
// SAFE WRAPPER (COPIES TO ALIGNED WORKSPACE)
//==============================================================================

int qr_ws_blocked(qr_workspace *ws, const float *A, float *Q, float *R,
                  uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;

    memcpy(ws->Cpack, A, (size_t)m * n * sizeof(float));
    return qr_ws_blocked_inplace(ws, ws->Cpack, Q, R, m, n, only_R);
}

//==============================================================================
// SIMPLE API (AUTO-ALLOCATES WORKSPACE)
//==============================================================================

int qr_blocked(const float *A, float *Q, float *R,
               uint16_t m, uint16_t n, bool only_R)
{
    qr_workspace *ws = qr_workspace_alloc(m, n, 0);
    if (!ws)
        return -ENOMEM;

    int ret = qr_ws_blocked(ws, A, Q, R, m, n, only_R);
    qr_workspace_free(ws);
    return ret;
}