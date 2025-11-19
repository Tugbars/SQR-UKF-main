/**
 * @file qr_blocked.c (COMPLETE FIXED VERSION)
 * @brief GEMM-Accelerated Blocked QR with Recursive Panel Factorization
 */

#include "qr.h"
#include "../gemm_2/gemm.h"
#include "../gemm_2/gemm_planning.h"
#include "../gemm_2/gemm_utils.h"
#include "qr_kernels_avx2.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>

// Forward declarations
static void build_T_matrix(const float *Y, const float *tau, float *T,
                           uint16_t m, uint16_t ib, uint16_t ldy);

                           //==============================================================================
// NAIVE CONTIGUOUS GEMM (for debugging)
//==============================================================================

/**
 * @brief Naive contiguous GEMM: C = alpha*A*B + beta*C
 * 
 * Assumes all matrices are contiguous (row-major, stride = n)
 * 
 * @param C Output matrix [m × n]
 * @param A Input matrix [m × k]
 * @param B Input matrix [k × n]
 * @param m Number of rows in A and C
 * @param k Number of columns in A, rows in B
 * @param n Number of columns in B and C
 * @param alpha Scalar for A*B
 * @param beta Scalar for C
 */
static int naive_gemm_contiguous(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    uint16_t m, uint16_t k, uint16_t n,
    float alpha, float beta)
{
    // C = beta * C
    if (beta == 0.0f)
    {
        for (uint16_t i = 0; i < m; ++i)
            for (uint16_t j = 0; j < n; ++j)
                C[i * n + j] = 0.0f;
    }
    else if (beta != 1.0f)
    {
        for (uint16_t i = 0; i < m; ++i)
            for (uint16_t j = 0; j < n; ++j)
                C[i * n + j] *= beta;
    }

    // C += alpha * A * B
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (uint16_t p = 0; p < k; ++p)
            {
                sum += (double)A[i * k + p] * (double)B[p * n + j];
            }
            C[i * n + j] += alpha * (float)sum;
        }
    }
    
    return 0;  // Success
}

#define GEMM_CALL naive_gemm_contiguous

#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//==============================================================================
// NAIVE STRIDED GEMM (for debugging)
//==============================================================================

/**
 * @brief Naive strided GEMM: C = alpha*A*B + beta*C
 * 
 * @param C Output matrix [m × n], stride ldc
 * @param A Input matrix [m × k], stride lda
 * @param B Input matrix [k × n], stride ldb
 * @param m Number of rows in A and C
 * @param k Number of columns in A, rows in B
 * @param n Number of columns in B and C
 * @param ldc Stride of C (elements between rows)
 * @param lda Stride of A
 * @param ldb Stride of B
 * @param alpha Scalar for A*B
 * @param beta Scalar for C
 */
static void naive_gemm_strided(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    uint16_t m, uint16_t k, uint16_t n,
    uint16_t ldc, uint16_t lda, uint16_t ldb,
    float alpha, float beta)
{
    // C = beta * C
    if (beta == 0.0f)
    {
        for (uint16_t i = 0; i < m; ++i)
            for (uint16_t j = 0; j < n; ++j)
                C[i * ldc + j] = 0.0f;
    }
    else if (beta != 1.0f)
    {
        for (uint16_t i = 0; i < m; ++i)
            for (uint16_t j = 0; j < n; ++j)
                C[i * ldc + j] *= beta;
    }

    // C += alpha * A * B
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (uint16_t p = 0; p < k; ++p)
            {
                sum += (double)A[i * lda + p] * (double)B[p * ldb + j];
            }
            C[i * ldc + j] += alpha * (float)sum;
        }
    }
}

//==============================================================================
// GEMM PLAN MANAGEMENT
//==============================================================================

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
// BLOCK SIZE SELECTION
//==============================================================================

static uint16_t select_optimal_ib(uint16_t m, uint16_t n)
{
    const uint16_t min_dim = MIN(m, n);
    uint16_t ib = MIN(64, min_dim);

    if (min_dim < 32)
        ib = MIN(16, min_dim);
    else if (min_dim < 64)
        ib = MIN(32, min_dim);

    if (ib < 8)
        ib = MIN(8, min_dim);

    return ib;
}

/**
 * @brief Check for extreme values that need safe path
 *
 * @param x Vector to check
 * @param len Vector length
 * @param max_abs [out] Maximum absolute value found
 * @return true if safe path needed (NaN/Inf or extreme values)
 */
static inline bool needs_safe_householder(const float *x, uint16_t len, float *max_abs)
{
    // Thresholds based on float32 range
    const float OVERFLOW_THRESHOLD = 1e19f;   // √FLT_MAX ≈ 1.84e19
    const float UNDERFLOW_THRESHOLD = 1e-19f; // √FLT_MIN ≈ 1.08e-19

    // Check for NaN/Inf first
    if (has_nan_or_inf(x, len))
        return true;

    // Find maximum absolute value
    float max_val = 0.0f;

#ifdef __AVX2__
    if (len >= 8)
    {
        __m256 max_vec = _mm256_setzero_ps();

        for (uint16_t i = 0; i < len - 7; i += 8)
        {
            __m256 v = _mm256_loadu_ps(&x[i]);
            __m256 abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
            max_vec = _mm256_max_ps(max_vec, abs_v);
        }

        // Horizontal max
        float vals[8];
        _mm256_storeu_ps(vals, max_vec);
        for (int i = 0; i < 8; ++i)
            if (vals[i] > max_val)
                max_val = vals[i];
    }
#endif

    // Scalar tail
    for (uint16_t i = (len / 8) * 8; i < len; ++i)
    {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_val)
            max_val = abs_val;
    }

    *max_abs = max_val;

    // Check if extreme
    if (max_val > OVERFLOW_THRESHOLD)
        return true; // Risk of overflow in x²

    if (max_val > 0.0f && max_val < UNDERFLOW_THRESHOLD)
        return true; // Risk of underflow in x²

    return false;
}

//==============================================================================
// HOUSEHOLDER REFLECTION PRIMITIVES
//==============================================================================

/**
 * @brief Robust Householder reflector generation with fast/safe path selection
 *
 * Uses fast AVX2 norm computation by default, falls back to scaled algorithm
 * only when overflow/underflow risk detected.
 *
 * @param x [in/out] Input vector, output: normalized reflector (x[0]=1 implicit)
 * @param m Vector length
 * @param tau [out] Householder scaling factor τ
 * @param beta [out] Resulting diagonal element β
 *
 * @note Fast path: Direct norm² computation (most common)
 * @note Safe path: LAPACK DLASSQ scaling (rare, extreme values)
 */
static void compute_householder_robust(float *restrict x, uint16_t m,
                                       float *restrict tau, float *restrict beta)
{
    if (m == 0)
    {
        *tau = 0.0f;
        if (beta)
            *beta = 0.0f;
        return;
    }

    if (m == 1)
    {
        *tau = 0.0f;
        if (beta)
            *beta = x[0];
        x[0] = 1.0f;
        return;
    }

    //==========================================================================
    // Check if we need safe path
    //==========================================================================

    float tail_max;
    bool need_safe = needs_safe_householder(&x[1], m - 1, &tail_max);

    // Also check alpha
    float abs_alpha = fabsf(x[0]);
    if (!need_safe)
    {
        const float OVERFLOW_THRESHOLD = 1e19f;
        const float UNDERFLOW_THRESHOLD = 1e-19f;

        if (!isfinite(x[0]) ||
            abs_alpha > OVERFLOW_THRESHOLD ||
            (abs_alpha > 0.0f && abs_alpha < UNDERFLOW_THRESHOLD))
        {
            need_safe = true;
        }
    }

    //==========================================================================
    // SAFE PATH: Use scaled computation if needed
    //==========================================================================

    if (need_safe)
    {
        double alpha = (double)x[0];

        // Compute safe norm using LAPACK DLASSQ algorithm
        double scale = 0.0;
        double sumsq = 1.0;

        // Add tail elements
        for (uint16_t i = 1; i < m; ++i)
        {
            double absxi = fabs((double)x[i]);
            if (absxi > scale)
            {
                double ratio = scale / absxi;
                sumsq = 1.0 + sumsq * ratio * ratio;
                scale = absxi;
            }
            else if (absxi > 0.0)
            {
                double ratio = absxi / scale;
                sumsq += ratio * ratio;
            }
        }

        // Add alpha
        double absalpha = fabs(alpha);
        if (absalpha > scale)
        {
            double ratio = scale / absalpha;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = absalpha;
        }
        else if (absalpha > 0.0)
        {
            double ratio = absalpha / scale;
            sumsq += ratio * ratio;
        }

        // Handle zero/NaN case
        if (scale == 0.0 || !isfinite(sumsq))
        {
            *tau = 0.0f;
            if (beta)
                *beta = x[0];
            x[0] = 1.0f;
            return;
        }

        // Reconstruct norm
        double norm = scale * sqrt(sumsq);

        // Compute beta
        double beta_val = -copysign(norm, alpha);

        // Compute tau
        *tau = (float)((beta_val - alpha) / beta_val);

        if (beta)
            *beta = (float)beta_val;

        // Normalize reflector with safe scaling
        double scale_factor = 1.0 / (alpha - beta_val);

        for (uint16_t i = 1; i < m; ++i)
            x[i] *= (float)scale_factor;

        x[0] = 1.0f;
        return;
    }

    //==========================================================================
    // FAST PATH: Normal computation (no overflow risk)
    //==========================================================================

    double norm_sq;

#ifdef __AVX2__
    norm_sq = (m > 9) ? compute_norm_sq_avx2(&x[1], m - 1) : 0.0;
    if (m <= 9)
#endif
    {
        norm_sq = 0.0;
        for (uint16_t i = 1; i < m; ++i)
        {
            double xi = (double)x[i];
            norm_sq += xi * xi;
        }
    }

    // Check for zero tail
    if (norm_sq == 0.0)
    {
        *tau = 0.0f;
        if (beta)
            *beta = x[0];
        x[0] = 1.0f;
        return;
    }

    double alpha = (double)x[0];
    double beta_val = -copysign(sqrt(alpha * alpha + norm_sq), alpha);
    double scale = 1.0 / (alpha - beta_val);

    // Vectorized scaling
#ifdef __AVX2__
    if (m > 9)
    {
        __m256 scale_vec = _mm256_set1_ps((float)scale);
        uint16_t i = 1;
        for (; i + 7 < m; i += 8)
        {
            __m256 v = _mm256_loadu_ps(&x[i]);
            v = _mm256_mul_ps(v, scale_vec);
            _mm256_storeu_ps(&x[i], v);
        }
        for (; i < m; ++i)
            x[i] *= (float)scale;
    }
    else
#endif
    {
        for (uint16_t i = 1; i < m; ++i)
            x[i] *= (float)scale;
    }

    *tau = (float)((beta_val - alpha) / beta_val);
    if (beta)
        *beta = (float)beta_val;
    x[0] = 1.0f;
}

static void apply_householder_clean(float *restrict C, uint16_t m, uint16_t n,
                                    uint16_t ldc, const float *restrict v, float tau)
{
#ifdef __AVX2__
    if (n >= 8)
    {
        apply_householder_avx2(C, m, n, ldc, v, tau);
        return;
    }
#endif

    if (tau == 0.0f)
        return;

    for (uint16_t j = 0; j < n; ++j)
    {
        double dot = 0.0;
        for (uint16_t i = 0; i < m; ++i)
            dot += (double)v[i] * (double)C[i * ldc + j];

        float tau_dot = tau * (float)dot;
        for (uint16_t i = 0; i < m; ++i)
            C[i * ldc + j] -= v[i] * tau_dot;
    }
}

//==============================================================================
// PANEL FACTORIZATION (WITH STRIDE SUPPORT)
//==============================================================================

/**
 * @brief Classical panel factorization with proper stride handling
 *
 * @param panel  Panel to factor [m × ib], stride lda
 * @param Y      Output Householder vectors [m × ib], stride ldy
 * @param tau    Output tau values [ib]
 * @param m      Rows in panel
 * @param ib     Columns in panel
 * @param lda    Stride of panel
 * @param ldy    Stride of Y (CRITICAL: may differ from ib!)
 * @param work   Workspace [m]
 */
static void panel_factor_clean(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,
    uint16_t ldy, // ✅ ADDED
    float *restrict work)
{
#ifdef __AVX2__
    if (ib >= 8)
    {
        zero_fill_strided_avx2(Y, m, ib, ldy);
    }
    else
#endif
        for (uint16_t i = 0; i < m; ++i)
        {
            for (uint16_t j = 0; j < ib; ++j)
            {
                Y[i * ldy + j] = 0.0f;
            }
        }

    for (uint16_t j = 0; j < ib && j < m; ++j)
    {
        uint16_t col_len = m - j;

        // Extract column j from panel
        float *restrict col_ptr = &panel[j * lda + j];
        for (uint16_t i = 0; i < col_len; ++i)
            work[i] = col_ptr[i * lda];

        // Compute Householder reflector
        float beta;
        compute_householder_robust(work, col_len, &tau[j], &beta);

        // Write beta to R diagonal
        col_ptr[0] = beta;

        // Write reflector tail back to panel
        for (uint16_t i = 1; i < col_len; ++i)
            col_ptr[i * lda] = work[i];

        // ✅ Store complete reflector in Y with proper stride
        for (uint16_t i = 0; i < j; ++i)
            Y[i * ldy + j] = 0.0f;
        for (uint16_t i = 0; i < col_len; ++i)
            Y[(j + i) * ldy + j] = work[i];

        // Apply reflector to trailing columns
        if (j + 1 < ib)
        {
            float *restrict trailing = &panel[j * lda + (j + 1)];
            apply_householder_clean(trailing, col_len, ib - j - 1,
                                    lda, work, tau[j]);
        }
    }
}

/**
 * @brief Recursive panel factorization with Level 3 BLAS
 */
static void panel_factor_recursive(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    float *restrict T_workspace,
    float *restrict Z_workspace,
    float *restrict Y_temp, // Still pass down, but don't use for allocation
    float *restrict work,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,
    uint16_t ldy,
    uint16_t threshold)
{
    if (ib <= threshold || ib < 2)
    {
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, work);
        return;
    }

    uint16_t ib1 = ib / 2;
    uint16_t ib2 = ib - ib1;

    // ✅ FIX: Use malloc for Y buffers (small, infrequent allocations)
    // T and Z workspace are still pre-allocated (larger, reused)
    float *Y_left = (float *)malloc(m * ib1 * sizeof(float));
    float *Y_right = (float *)malloc((m - ib1) * ib2 * sizeof(float));

    if (!Y_left || !Y_right)
    {
        free(Y_left);
        free(Y_right);
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, work);
        return;
    }

    // Left recursion
    panel_factor_recursive(
        panel, Y_left, tau,
        T_workspace, Z_workspace, Y_temp, // Y_temp unused now
        work,
        m, ib1, lda, ib1, threshold);

    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < ib1; ++j)
            Y[i * ldy + j] = Y_left[i * ib1 + j];

    // Apply left reflectors
    float *right_cols = &panel[ib1];
    for (uint16_t j = 0; j < ib1; ++j)
    {
        uint16_t rows_affected = m - j;
        for (uint16_t i = 0; i < rows_affected; ++i)
            work[i] = Y_left[(j + i) * ib1 + j];

        float *right_start = &right_cols[j * lda];
        apply_householder_clean(right_start, rows_affected, ib2,
                                lda, work, tau[j]);
    }

    // Right recursion
    float *right_panel = &panel[ib1 * lda + ib1];
    panel_factor_recursive(
        right_panel, Y_right, &tau[ib1],
        T_workspace, Z_workspace, Y_temp, // Y_temp unused now
        work,
        m - ib1, ib2, lda, ib2, threshold);

    // Copy results
    for (uint16_t i = 0; i < ib1; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            Y[i * ldy + (ib1 + j)] = 0.0f;

    for (uint16_t i = 0; i < m - ib1; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            Y[(ib1 + i) * ldy + (ib1 + j)] = Y_right[i * ib2 + j];

    free(Y_left);
    free(Y_right);
}

static void panel_factor_optimized(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,
    uint16_t ldy,
    qr_workspace *workspace)
{
    uint16_t threshold;
    if (ib < 16)
    {
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, workspace->tmp);
        return;
    }
    else if (ib < 32)
        threshold = 8;
    else if (ib < 64)
        threshold = 12;
    else
        threshold = 16;

    // ✅ NO MALLOC: Use pre-allocated workspace
    panel_factor_recursive(
        panel, Y, tau,
        workspace->panel_T_temp, // ✅ Pre-allocated
        workspace->panel_Z_temp, // ✅ Pre-allocated
        workspace->panel_Y_temp, // ✅ Pre-allocated
        workspace->tmp,
        m, ib, lda, ldy, threshold);
}

//==============================================================================
// BUILD T MATRIX (WITH STRIDE SUPPORT)
//==============================================================================

static void build_T_matrix(const float *restrict Y, const float *restrict tau,
                           float *restrict T, uint16_t m, uint16_t ib,
                           uint16_t ldy) // ✅ ADDED
{
    memset(T, 0, (size_t)ib * ib * sizeof(float));
    if (ib == 0)
        return;

    double w_stack[64];
    double *w = (ib <= 64) ? w_stack : (double *)malloc(ib * sizeof(double));
    if (!w)
        return;

    for (uint16_t i = 0; i < ib; ++i)
    {
        T[i * ib + i] = tau[i];

        if (tau[i] == 0.0f || i == 0)
            continue;

        for (uint16_t j = 0; j < i; ++j)
        {
#ifdef __AVX2__
            double dot = (m >= 16) ? dot_product_strided_avx2(&Y[j], &Y[i], m, ldy, ldy) : 0.0;
            if (m < 16)
#endif
            {
                dot = 0.0;
                for (uint16_t r = 0; r < m; ++r)
                    dot += (double)Y[r * ldy + j] * (double)Y[r * ldy + i];
            }
            w[j] = -(double)tau[i] * dot;
        }

        for (uint16_t j = 0; j < i; ++j)
        {
            double sum = 0.0;
            for (uint16_t k = 0; k < i; ++k)
                sum += (double)T[j * ib + k] * w[k];
            T[j * ib + i] = (float)sum;
        }
    }

    if (ib > 64)
        free(w);
}

//==============================================================================
// BLOCK REFLECTOR APPLICATION
//==============================================================================

static int apply_block_reflector_clean(
    float *restrict C,
    const float *restrict Y,
    const float *restrict T,
    uint16_t m, uint16_t n, uint16_t ib,
    uint16_t ldy,
    float *restrict Z,
    float *restrict Z_temp,
    float *restrict YT) // Always provided by caller
{
#ifdef __AVX2__
    if (m >= 8 && ib >= 8)
    {
        transpose_avx2_8x8(Y, YT, m, ib, ldy, m);
    }
    else
#endif
        for (uint16_t i = 0; i < ib; ++i)
            for (uint16_t j = 0; j < m; ++j)
                YT[i * m + j] = Y[j * ldy + i];

    // Z = Y^T * C
    int ret = GEMM_CALL(Z, YT, C, ib, m, n, 1.0f, 0.0f);
    if (ret != 0)
        return ret;

    // Z_temp = T * Z
    ret = GEMM_CALL(Z_temp, T, Z, ib, ib, n, 1.0f, 0.0f);
    if (ret != 0)
        return ret;

    // ✅ Use YT as temporary storage for contiguous Y (reuse the buffer)
    // Since YT is [ib × m] and we need [m × ib], we have enough space
    float *Y_contig = YT; // Reuse YT buffer temporarily

    // Copy Y to contiguous buffer
#ifdef __AVX2__
    if (m >= 8 && ib >= 8)
    {
        copy_strided_to_contiguous_avx2(Y_contig, Y, m, ib, ldy);
    }
    else
#endif
        for (uint16_t i = 0; i < m; ++i)
            for (uint16_t j = 0; j < ib; ++j)
                Y_contig[i * ib + j] = Y[i * ldy + j];

    // C -= Y_contig * Z_temp
    ret = GEMM_CALL(C, Y_contig, Z_temp, m, ib, n, -1.0f, 1.0f);

    return ret;
}

static int apply_block_reflector_strided(
    float *restrict C,
    const float *restrict Y,
    const float *restrict T,
    uint16_t m, uint16_t n, uint16_t ib,
    uint16_t ldc,
    uint16_t ldy,
    float *restrict Z,
    float *restrict Z_temp,
    float *restrict YT)
{
    // Transpose Y: YT[ib × m]
    for (uint16_t i = 0; i < ib; ++i)
        for (uint16_t j = 0; j < m; ++j)
            YT[i * m + j] = Y[j * ldy + i];

    // ✅ Z = Y^T * C using NAIVE strided GEMM
    printf("    [NAIVE] Z = YT * C: [%d×%d] × [%d×%d] → [%d×%d]\n",
           ib, m, m, n, ib, n);
    printf("            Strides: YT=%d, C=%d, Z=%d\n", m, ldc, n);
    
    naive_gemm_strided(Z, YT, C,
                      ib, m, n,    // dimensions
                      n, m, ldc,   // strides
                      1.0f, 0.0f);

    // ✅ Z_temp = T * Z using NAIVE strided GEMM
    printf("    [NAIVE] Z_temp = T * Z: [%d×%d] × [%d×%d] → [%d×%d]\n",
           ib, ib, ib, n, ib, n);
    
    naive_gemm_strided(Z_temp, T, Z,
                      ib, ib, n,
                      n, ib, n,    // all contiguous
                      1.0f, 0.0f);

    // ✅ C = C - Y * Z_temp using NAIVE strided GEMM
    printf("    [NAIVE] C -= Y * Z_temp: [%d×%d] × [%d×%d] → [%d×%d]\n",
           m, ib, ib, n, m, n);
    printf("            Strides: C=%d, Y=%d, Z_temp=%d\n", ldc, ldy, n);
    
    naive_gemm_strided(C, Y, Z_temp,
                      m, ib, n,
                      ldc, ldy, n,
                      -1.0f, 1.0f);

    return 0;
}

//==============================================================================
// MAIN QR ALGORITHM
//==============================================================================

int qr_ws_blocked_inplace(qr_workspace *ws, float *A, float *Q, float *R,
                          uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;
    if (m > ws->m_max || n > ws->n_max)
        return -EINVAL;



    const uint16_t kmax = MIN(m, n);
    uint16_t block_count = 0;

    //==========================================================================
    // FACTORIZATION PHASE: Process matrix block by block
    //==========================================================================
    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;
        uint16_t cols_right = (n > k + block_size) ? (n - k - block_size) : 0;

        // Factor current panel: A[k:m, k:k+block_size]
        panel_factor_optimized(&A[k * n + k], ws->Y, &ws->tau[k],
                               rows_below, block_size, n, ws->ib, ws);

        // Build compact WY representation: T = -τ * Y^T * Y (upper triangular)
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size, ws->ib);

        // Store Y and T for Q formation
if (ws->Y_stored && ws->T_stored)
{
    size_t y_offset = block_count * ws->Y_block_stride;
    size_t t_offset = block_count * ws->T_block_stride;

    printf("  [STORING Block %d] k=%d, rows_below=%d, block_size=%d\n",
           block_count, k, rows_below, block_size);
    printf("    y_offset=%lu, Y_block_stride=%lu\n", 
           (unsigned long)y_offset, (unsigned long)ws->Y_block_stride);
    printf("    Storing Y[0:%d, 0:%d] with stride ws->ib=%d\n", 
           rows_below, block_size, ws->ib);
    
    // Store first few elements as sanity check
    printf("    Y[0,0]=%.4f, Y[1,0]=%.4f, Y[0,1]=%.4f\n",
           ws->Y[0], ws->Y[ws->ib], ws->Y[1]);

    // Copy Y to storage
    for (uint16_t i = 0; i < rows_below; ++i)
        for (uint16_t j = 0; j < block_size; ++j)
            ws->Y_stored[y_offset + i * ws->ib + j] = ws->Y[i * ws->ib + j];

    // Verify storage
    printf("    Y_stored[y_offset+0]=%.4f, Y_stored[y_offset+ws->ib]=%.4f\n",
           ws->Y_stored[y_offset], ws->Y_stored[y_offset + ws->ib]);

    // Copy T to storage
    memcpy(&ws->T_stored[t_offset], ws->T,
           block_size * block_size * sizeof(float));
}

        //======================================================================
        // ✅ BLAS LEVEL 3: Apply block reflector to trailing matrix
        //======================================================================
        // Apply (I - Y*T*Y^T) to A[k:m, k+block_size:n]
        if (cols_right > 0)
        {
            float *trailing_matrix = &A[k * n + (k + block_size)];

            int ret = apply_block_reflector_strided(
                trailing_matrix,  // C: trailing submatrix [rows_below × cols_right]
                ws->Y,           // Y: Householder vectors [rows_below × block_size]
                ws->T,           // T: compact WY factor [block_size × block_size]
                rows_below,      // m: number of rows
                cols_right,      // n: number of columns
                block_size,      // ib: block size
                n,               // ldc: stride of A (full matrix width)
                ws->ib,          // ldy: stride of Y
                ws->Z,           // workspace [ib × n]
                ws->Z_temp,      // workspace [ib × n]
                ws->YT);         // workspace [ib × m]

            if (ret != 0)
                return ret;
        }

        block_count++;
    }

    //==========================================================================
    // EXTRACT R: Upper triangular part of factored A
    //==========================================================================
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? A[i * n + j] : 0.0f;

    //==========================================================================
    // ✅ FORM Q: Apply stored reflectors in reverse order
    //==========================================================================
    if (!only_R && Q)
    {
        // Initialize Q = I
        memset(Q, 0, (size_t)m * m * sizeof(float));
        for (uint16_t i = 0; i < m; ++i)
            Q[i * m + i] = 1.0f;

        // Apply reflectors in reverse order: Q = H_n * ... * H_2 * H_1
for (int blk = block_count - 1; blk >= 0; blk--)
{
    uint16_t k = blk * ws->ib;
    uint16_t block_size = MIN(ws->ib, kmax - k);
    uint16_t rows_below = m - k;

    size_t y_offset = blk * ws->Y_block_stride;
    size_t t_offset = blk * ws->T_block_stride;

    printf("  [LOADING Block %d] k=%d, rows_below=%d, block_size=%d\n",
           blk, k, rows_below, block_size);
    printf("    y_offset=%lu, Y_block_stride=%lu\n", 
           (unsigned long)y_offset, (unsigned long)ws->Y_block_stride);
    printf("    Y_stored[y_offset+0]=%.4f, Y_stored[y_offset+ws->ib]=%.4f\n",
           ws->Y_stored[y_offset], ws->Y_stored[y_offset + ws->ib]);

    // Load Y
    for (uint16_t i = 0; i < rows_below; ++i)
        for (uint16_t j = 0; j < block_size; ++j)
            ws->Y[i * ws->ib + j] = ws->Y_stored[y_offset + i * ws->ib + j];

    printf("    Loaded Y[0,0]=%.4f, Y[1,0]=%.4f, Y[0,1]=%.4f\n",
           ws->Y[0], ws->Y[ws->ib], ws->Y[1]);

            // Load T matrix
            memcpy(ws->T, &ws->T_stored[t_offset],
                   block_size * block_size * sizeof(float));

            //==================================================================
            // ✅ FIX 2: Apply to Q submatrix starting at row k
            //==================================================================
            // Q_sub points to Q[k:m, 0:m] - the rows affected by this block
            float *Q_sub = &Q[k * m + 0];

            //==================================================================
            // ✅ FIX 3: Use strided version with correct dimensions
            //==================================================================
            // Apply (I - Y*T*Y^T) to Q[k:m, :]
            int ret = apply_block_reflector_strided(
                Q_sub,           // C: Q[k:m, 0:m] submatrix
                ws->Y,           // Y: LOCAL coordinates [rows_below × block_size]
                ws->T,           // T: compact WY factor [block_size × block_size]
                rows_below,      // m: number of rows (m - k)
                m,               // n: full width of Q
                block_size,      // ib: block size
                m,               // ldc: stride of Q (full matrix width)
                ws->ib,          // ldy: stride of Y
                ws->Z,           // workspace [ib × n]
                ws->Z_temp,      // workspace [ib × n]
                ws->YT);         // workspace [ib × m]

            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

//==============================================================================
// WRAPPER FUNCTIONS
//==============================================================================

int qr_ws_blocked(qr_workspace *ws, const float *A, float *Q, float *R,
                  uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;
    memcpy(ws->Cpack, A, (size_t)m * n * sizeof(float));
    return qr_ws_blocked_inplace(ws, ws->Cpack, Q, R, m, n, only_R);
}

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


qr_workspace *qr_workspace_alloc_ex(uint16_t m_max, uint16_t n_max,
                                    uint16_t ib, bool store_reflectors)
{
    if (!m_max || !n_max)
        return NULL;

    qr_workspace *ws = (qr_workspace *)calloc(1, sizeof(qr_workspace));
    if (!ws)
        return NULL;

    const uint16_t min_dim = (m_max < n_max) ? m_max : n_max;
    ws->m_max = m_max;
    ws->n_max = n_max;
    ws->ib = ib ? ib : select_optimal_ib(m_max, n_max);
    ws->num_blocks = (min_dim + ws->ib - 1) / ws->ib;

    //==========================================================================
    // ✅ CRITICAL: Y_stored stride calculation
    //==========================================================================

    // Y_stored stores each block in PACKED format:
    //   Block k stores Y_k[rows_below_k × block_size_k]
    //   where rows_below_k = m - k*ib
    //         block_size_k = min(ib, min_dim - k*ib)
    //
    // We allocate worst-case: first block with m_max rows and ib columns
    // This gives Y_block_stride = m_max * ib elements per block
    //
    // IMPORTANT: Y_stored uses PACKED stride (block_size varies),
    //            while ws->Y uses FIXED stride (always ws->ib)

    ws->Y_block_stride = (size_t)m_max * ws->ib; // ✅ Max size per block
    ws->T_block_stride = (size_t)ws->ib * ws->ib;

    const uint16_t n_big = (m_max > n_max) ? m_max : n_max;

    //==========================================================================
    // Allocate buffers
    //==========================================================================

    ws->tau = (float *)malloc(min_dim * sizeof(float));
    ws->tmp = (float *)malloc(m_max * sizeof(float));
    ws->work = (float *)malloc(m_max * sizeof(float));
    ws->T = (float *)gemm_aligned_alloc(32, ws->ib * ws->ib * sizeof(float));

    // ✅ Y has FIXED stride ws->ib (used during factorization)
    ws->Y = (float *)gemm_aligned_alloc(32, (size_t)m_max * ws->ib * sizeof(float));

    ws->YT = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * m_max * sizeof(float));
    ws->Z = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_big * sizeof(float));
    ws->Z_temp = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_big * sizeof(float));
    ws->Cpack = (float *)gemm_aligned_alloc(32, (size_t)m_max * n_max * sizeof(float));
    ws->vn1 = (float *)malloc(n_max * sizeof(float));
    ws->vn2 = (float *)malloc(n_max * sizeof(float));

    // panel_Y_temp: Used for Y_left and Y_right during recursion
    // Max size: m_max × ib (worst case for Y_left at first level)
    ws->panel_Y_temp = (float *)gemm_aligned_alloc(32,
                                                   (size_t)m_max * ws->ib * sizeof(float));

    // panel_T_temp: T matrix workspace for recursion
    ws->panel_T_temp = (float *)gemm_aligned_alloc(32,
                                                   ws->ib * ws->ib * sizeof(float));

    // panel_Z_temp: Z workspace for recursion
    ws->panel_Z_temp = (float *)gemm_aligned_alloc(32,
                                                   ws->ib * ws->ib * sizeof(float));

    size_t bytes =
        min_dim * sizeof(float) +
        m_max * sizeof(float) * 2 +
        ws->ib * ws->ib * sizeof(float) +
        (size_t)m_max * ws->ib * sizeof(float) * 2 +
        (size_t)ws->ib * n_big * sizeof(float) * 2 +
        (size_t)m_max * n_max * sizeof(float) +
        n_max * sizeof(float) * 2;

    bytes += (size_t)m_max * ws->ib * sizeof(float); // panel_Y_temp
    bytes += ws->ib * ws->ib * sizeof(float) * 2;    // panel_T_temp, panel_Z_temp

    //==========================================================================
    // ✅ Allocate Y_stored and T_stored with proper layout
    //==========================================================================

    if (store_reflectors)
    {
        // Y_stored: All blocks stored in packed format
        // Layout: [block0: m_max × ib][block1: (m_max-ib) × ib][...]
        //
        // We allocate worst-case for ALL blocks:
        //   Total = num_blocks × (m_max × ib) floats
        //
        // Note: This over-allocates because later blocks have fewer rows,
        //       but it simplifies indexing (y_offset = block_count * Y_block_stride)

        ws->Y_stored = (float *)gemm_aligned_alloc(32,
                                                   ws->num_blocks * ws->Y_block_stride * sizeof(float));

        // T_stored: All T matrices (each is ib × ib)
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

    //==========================================================================
    // Validate allocations
    //==========================================================================

    if (!ws->tau || !ws->tmp || !ws->work || !ws->T || !ws->Cpack ||
        !ws->Y || !ws->YT || !ws->Z || !ws->Z_temp || !ws->vn1 || !ws->vn2 ||
        !ws->panel_Y_temp || !ws->panel_T_temp || !ws->panel_Z_temp)
    {
        qr_workspace_free(ws);
        return NULL;
    }

    if (store_reflectors && (!ws->Y_stored || !ws->T_stored))
    {
        qr_workspace_free(ws);
        return NULL;
    }

    //==========================================================================
    // Optional: Create GEMM plans
    //==========================================================================

    const uint16_t first_panel_cols = (n_max > ws->ib) ? (n_max - ws->ib) : 0;

    if (first_panel_cols > 0)
        ws->trailing_plans = create_panel_plans(m_max, first_panel_cols, ws->ib);
    else
        ws->trailing_plans = NULL;

    if (m_max >= ws->ib)
        ws->q_formation_plans = create_panel_plans(m_max, m_max, ws->ib);
    else
        ws->q_formation_plans = NULL;

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

    // ✅ Free new buffers
    gemm_aligned_free(ws->panel_Y_temp);
    gemm_aligned_free(ws->panel_T_temp);
    gemm_aligned_free(ws->panel_Z_temp);

    free(ws->vn1);
    free(ws->vn2);
    free(ws);
}

size_t qr_workspace_bytes(const qr_workspace *ws)
{
    return ws ? ws->total_bytes : 0;
}