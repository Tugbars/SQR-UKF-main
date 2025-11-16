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
#include "../gemm_2/gemm.h"           // ← FIXED: was "gemm.h"
#include "../gemm_2/gemm_planning.h"  // ← FIXED: was "gemm_planning.h"
#include "../gemm_2/gemm_utils.h"     // ← FIXED: was "gemm_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>


#define USE_NAIVE_GEMM_DEBUG 0  // Set to 1 to use naive GEMM, 0 for optimized

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

    printf("    [NAIVE_GEMM] C[%d×%d] = %.2f*A[%d×%d]*B[%d×%d] + %.2f*C\n",
           m, n, alpha, m, k, k, n, beta);

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
        if (beta_out) *beta_out = 0.0f;
        return;
    }

    if (m == 1)
    {
        *tau = 0.0f;
        if (beta_out) *beta_out = x[0];
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
        if (beta_out) *beta_out = x0;
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
    if (beta_out) *beta_out = (float)beta;  // ✅ RETURN beta for R diagonal
    x[0] = 1.0f;  // Reflector vector v[0] = 1
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
// OPTIMIZED PANEL FACTORIZATION WITH EQUAL-LENGTH FAST PATH
//==============================================================================
// MEMORY LAYOUT:
//   panel[m×ib]:   row-major submatrix starting at A[k,k], lda = panel_stride
//   Element (r,c): panel[r * panel_stride + c]
//   Column c:      &panel[c * panel_stride + c], stride = panel_stride
//   Y[m×ib]:       row-major output, lda = ib (Householder vectors)
//   YT[ib×m]:      row-major output, lda = m (transposed Y)
//   tmp_col[m]:    workspace for gather/scatter (stride-1 buffer)
//
// OPTIMIZATION STRATEGY:
//   Fast path: columns 0..(m >= ib ? ib : 0) have equal length → no bounds checks
//   Slow path: remaining columns have shrinking length → bounds checks needed
//==============================================================================

static void panel_qr_simd(float *panel, float *Y, float *YT, float *tau,
                          uint16_t m, uint16_t panel_stride, uint16_t ib,
                          float *tmp_col)
{
    const uint16_t fast_end = (m >= ib) ? ib : 0;

    // =========================================================================
    // FAST PATH: EQUAL-LENGTH REFLECTORS
    // =========================================================================
    for (uint16_t j = 0; j < fast_end; ++j)
    {
        const uint16_t rows_below = m - j;

        // Prefetch next column
        if (j + 1 < fast_end)
        {
            float *next_col = &panel[(j + 1) * panel_stride + (j + 1)];
            _mm_prefetch((const char *)next_col, _MM_HINT_T0);
            if (rows_below > 8)
            {
                _mm_prefetch((const char *)(next_col + 8 * panel_stride), _MM_HINT_T0);
            }
        }

        // GATHER: Strided column → contiguous buffer
        float *col = &panel[j * panel_stride + j];
        for (uint16_t i = 0; i < rows_below; ++i)
        {
            tmp_col[i] = col[i * panel_stride];
        }

        // SIMD Householder reflection
        float beta;
        householder_reflection_simd(tmp_col, rows_below, &tau[j], &beta);

        // ✅ FIXED SCATTER: diagonal gets beta (R), below gets v_tail
        col[0] = beta;  // Store R diagonal element
        for (uint16_t i = 1; i < rows_below; ++i)
        {
            col[i * panel_stride] = tmp_col[i];  // Store reflector v[1:]
        }

        // Apply reflector to trailing columns [j+1 : ib)
        if (j + 1 < ib)
        {
            float *trailing = &panel[j * panel_stride + (j + 1)];
            apply_reflector(trailing, rows_below, ib - j - 1,
                            panel_stride, tmp_col, tau[j]);
        }

        // Store to Y[m×ib] and YT[ib×m] (full reflector v with v[0]=1)
        for (uint16_t i = j; i < m; ++i)
        {
            float val = (i == j) ? 1.0f : tmp_col[i - j];
            Y[i * ib + j] = val;
            YT[j * m + i] = val;
        }

        // Zero upper triangle
        for (uint16_t i = 0; i < j; ++i)
        {
            Y[i * ib + j] = 0.0f;
            YT[j * m + i] = 0.0f;
        }
    }

    // =========================================================================
    // SLOW PATH: RAGGED TAIL
    // =========================================================================
    for (uint16_t j = fast_end; j < ib; ++j)
    {
        const uint16_t rows_below = m - j;

        if (rows_below == 0)
            break;

        // Prefetch
        if (j + 1 < ib && j + 1 < m)
        {
            float *next_col = &panel[(j + 1) * panel_stride + (j + 1)];
            _mm_prefetch((const char *)next_col, _MM_HINT_T0);
        }

        // GATHER
        float *col = &panel[j * panel_stride + j];
        for (uint16_t i = 0; i < rows_below; ++i)
        {
            tmp_col[i] = col[i * panel_stride];
        }

        // SIMD Householder reflection
        float beta;
        householder_reflection_simd(tmp_col, rows_below, &tau[j], &beta);

        // ✅ FIXED SCATTER: diagonal gets beta, below gets v_tail
        col[0] = beta;
        for (uint16_t i = 1; i < rows_below; ++i)
        {
            col[i * panel_stride] = tmp_col[i];
        }

        // Apply reflector to trailing columns
        if (j + 1 < ib)
        {
            float *trailing = &panel[j * panel_stride + (j + 1)];
            apply_reflector(trailing, rows_below, ib - j - 1,
                            panel_stride, tmp_col, tau[j]);
        }

        // Store to Y and YT
        for (uint16_t i = j; i < m; ++i)
        {
            float val = (i == j) ? 1.0f : tmp_col[i - j];
            Y[i * ib + j] = val;
            YT[j * m + i] = val;
        }

        for (uint16_t i = 0; i < j; ++i)
        {
            Y[i * ib + j] = 0.0f;
            YT[j * m + i] = 0.0f;
        }
    }
}

//==============================================================================
// BUILD T MATRIX (WY REPRESENTATION, SCALAR - CORRECT FOR ROW-MAJOR Y)
//==============================================================================

static void build_T_matrix(const float *Y, const float *tau, float *T,
                           uint16_t m, uint16_t ib)
{
    // Build upper triangular T: Q = I - Y*T*Y^T
    // Y[m×ib] row-major, T[ib×ib] row-major
    // Note: Column dot products with row-major Y are not vectorizable
    // without gather instructions, so we keep this scalar

    memset(T, 0, (size_t)ib * ib * sizeof(float));

    T[0] = tau[0];

    for (uint16_t k = 1; k < ib; ++k)
    {
        T[k * ib + k] = tau[k];

        // Compute w[j] = Y[:, j]^T * Y[:, k]
        for (uint16_t j = 0; j < k; ++j)
        {
            double dot = 0.0;
            for (uint16_t i = 0; i < m; ++i)
            {
                dot += (double)Y[i * ib + j] * (double)Y[i * ib + k];
            }
            T[j * ib + k] = (float)dot;
        }

        // T[0:k, k] = -tau[k] * T[0:k, 0:k] * w (backward substitution)
        for (int j = k - 1; j >= 0; --j)
        {
            double sum = 0.0;
            for (uint16_t i = j; i < k; ++i)
            {
                sum += (double)T[j * ib + i] * (double)T[i * ib + k];
            }
            T[j * ib + k] = (float)(-tau[k] * sum);
        }
    }
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

qr_workspace *qr_workspace_alloc_ex(uint16_t m_max, uint16_t n_max,
                                    uint16_t ib, bool store_reflectors)
{
    if (!m_max || !n_max)
        return NULL;

    qr_workspace *ws = (qr_workspace *)calloc(1, sizeof(qr_workspace));
    if (!ws)
        return NULL;

    const uint16_t mn = (m_max < n_max) ? m_max : n_max;
    ws->m_max = m_max;
    ws->n_max = n_max;
    ws->ib = ib ? ib : select_optimal_ib(m_max, n_max);
    ws->num_blocks = (mn + ws->ib - 1) / ws->ib;

    // Correct block strides
    ws->Y_block_stride = (size_t)m_max * ws->ib;
    ws->T_block_stride = (size_t)ws->ib * ws->ib;

    size_t bytes = 0;

    // Standard buffers
    ws->tau = (float *)malloc(mn * sizeof(float));
    ws->tmp = (float *)malloc(m_max * sizeof(float));
    ws->work = (float *)malloc(m_max * sizeof(float));
    ws->T = (float *)gemm_aligned_alloc(32, ws->ib * ws->ib * sizeof(float));
    ws->Cpack = (float *)gemm_aligned_alloc(32, (size_t)m_max * n_max * sizeof(float));
    ws->Y = (float *)gemm_aligned_alloc(32, (size_t)m_max * ws->ib * sizeof(float));
    ws->YT = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * m_max * sizeof(float));
    ws->Z = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_max * sizeof(float));
    ws->Z_temp = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_max * sizeof(float));
    ws->vn1 = (float *)malloc(n_max * sizeof(float));
    ws->vn2 = (float *)malloc(n_max * sizeof(float));

    bytes = mn * sizeof(float) +
            m_max * sizeof(float) * 2 +
            ws->ib * ws->ib * sizeof(float) +
            (size_t)m_max * n_max * sizeof(float) * 2 +
            (size_t)m_max * ws->ib * sizeof(float) * 2 +
            (size_t)ws->ib * n_max * sizeof(float) * 2 +
            n_max * sizeof(float) * 2;

    // Correct storage layout: [num_blocks][rows][cols]
    if (store_reflectors)
    {
        ws->Y_stored = (float *)gemm_aligned_alloc(32,
                                                       ws->num_blocks * ws->Y_block_stride * sizeof(float));
        ws->T_stored = (float *)gemm_aligned_alloc(32,
                                                       ws->num_blocks * ws->T_block_stride * sizeof(float));
        bytes += ws->num_blocks * ws->Y_block_stride * sizeof(float);
        bytes += ws->num_blocks * ws->T_block_stride * sizeof(float);
    }
    else
    {
        ws->Y_stored = NULL;
        ws->T_stored = NULL;
    }

    if (!ws->tau || !ws->tmp || !ws->work || !ws->T || !ws->Cpack ||
        !ws->Y || !ws->YT || !ws->Z || !ws->Z_temp || !ws->vn1 || !ws->vn2)
    {
        qr_workspace_free(ws);
        return NULL;
    }

    // Create GEMM plans for first panel (exact dimensions)
    const uint16_t first_panel_cols = (n_max > ws->ib) ? (n_max - ws->ib) : 0;
    if (first_panel_cols > 0)
    {
        ws->trailing_plans = create_panel_plans(m_max, first_panel_cols, ws->ib);
    }
    else
    {
        ws->trailing_plans = NULL;
    }

    // Plans for Q formation (m×m problem)
    if (m_max >= ws->ib)
    {
        ws->q_formation_plans = create_panel_plans(m_max, m_max, ws->ib);
    }
    else
    {
        ws->q_formation_plans = NULL;
    }

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

int qr_ws_blocked_inplace(qr_workspace *ws, float *A, float *Q, float *R,
                          uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;
    if (m > ws->m_max || n > ws->n_max)
        return -EINVAL;

    float *Awork = A;
    const uint16_t kmax = (m < n) ? m : n;

    //==========================================================================
    // MAIN BLOCKED FACTORIZATION LOOP
    //==========================================================================
    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        const uint16_t block_size = (k + ws->ib <= kmax) ? ws->ib : (kmax - k);
        const uint16_t rows_below = m - k;
        const uint16_t cols_right = n - k - block_size;

        // ------------------------------------------------------------------
        // 1. Panel factorization with SIMD + prefetching
        // ------------------------------------------------------------------
        // ✅ FIXED: Pass ws->tmp for gather/scatter operations
        panel_qr_simd(&Awork[k * n + k], ws->Y, ws->YT, &ws->tau[k],
                      rows_below, n, block_size, ws->tmp);

        // ------------------------------------------------------------------
        // 2. Build T matrix (WY representation)
        // ------------------------------------------------------------------
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size);

        // ------------------------------------------------------------------
        // 3. Store reflectors for fast Q formation
        // ------------------------------------------------------------------
        if (ws->Y_stored && ws->T_stored)
        {
            uint16_t block_idx = k / ws->ib;
            float *Y_dst = &ws->Y_stored[block_idx * ws->Y_block_stride];
            float *T_dst = &ws->T_stored[block_idx * ws->T_block_stride];

            memcpy(Y_dst, ws->Y, rows_below * block_size * sizeof(float));
            memcpy(T_dst, ws->T, block_size * block_size * sizeof(float));
        }

        // ------------------------------------------------------------------
        // 4. Apply block reflector to trailing matrix (Level-3 BLAS!)
        // ------------------------------------------------------------------
        if (cols_right > 0)
        {
            qr_gemm_plans_t *plans_to_use = NULL;

            // Use pre-created plans if dimensions match exactly
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
                return ret;
        }
    }

    //==========================================================================
    // EXTRACT R (Upper Triangular)
    //==========================================================================
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            R[i * n + j] = (i <= j) ? Awork[i * n + j] : 0.0f;
        }
    }

    //==========================================================================
    // FORM Q (BLOCKED, LEVEL-3 BLAS)
    //==========================================================================
    if (!only_R && Q)
    {
        if (!ws->Y_stored || !ws->T_stored)
        {
            return -EINVAL; // Need stored reflectors for Q formation
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