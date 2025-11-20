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

#define GEMM_CALL gemm_dynamic

#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef QR_ENABLE_PREFETCH
    #define QR_ENABLE_PREFETCH 1  // Enable by default
#endif

#if QR_ENABLE_PREFETCH && defined(__AVX2__)
    #define QR_PREFETCH_ENABLED
#endif

#ifndef QR_PREFETCH_DISTANCE_NEAR
    #define QR_PREFETCH_DISTANCE_NEAR 1  // Iterations ahead for T0
#endif

#ifndef QR_PREFETCH_DISTANCE_FAR
    #define QR_PREFETCH_DISTANCE_FAR 2   // Iterations ahead for T1
#endif

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
        return true;  // Risk of overflow in x²
    
    if (max_val > 0.0f && max_val < UNDERFLOW_THRESHOLD)
        return true;  // Risk of underflow in x²
    
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
        //======================================================================
        // ✅ PREFETCH: Next columns
        //======================================================================
        
#ifdef __AVX2__
        // Prefetch next column (1 iteration ahead)
        if (j + 1 < n)
        {
            uint16_t prefetch_rows = MIN(64, m);
            for (uint16_t i = 0; i < prefetch_rows; i += 8)
            {
                _mm_prefetch((const char*)&C[i * ldc + (j + 1)], _MM_HINT_T0);
            }
        }
        
        // Prefetch column after next (2 iterations ahead)
        if (j + 2 < n)
        {
            uint16_t prefetch_rows = MIN(32, m);
            for (uint16_t i = 0; i < prefetch_rows; i += 8)
            {
                _mm_prefetch((const char*)&C[i * ldc + (j + 2)], _MM_HINT_T1);
            }
        }
#endif

        //======================================================================
        // Compute dot product: v^T * C[:,j]
        //======================================================================
        
        double dot = 0.0;
        for (uint16_t i = 0; i < m; ++i)
            dot += (double)v[i] * (double)C[i * ldc + j];

        float tau_dot = tau * (float)dot;
        
        //======================================================================
        // Apply update: C[:,j] -= tau * v * (v^T * C[:,j])
        //======================================================================
        
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
    uint16_t ldy,
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

        //======================================================================
        // ✅ PREFETCH: Next column (1-2 iterations ahead)
        //======================================================================
        
#ifdef __AVX2__
        // Prefetch next column
        if (j + 1 < ib && j + 1 < m)
        {
            float *next_col = &panel[(j + 1) * lda + (j + 1)];
            uint16_t prefetch_len = MIN(64, m - j - 1);
            
            for (uint16_t i = 0; i < prefetch_len; i += 8)
            {
                _mm_prefetch((const char*)&next_col[i * lda], _MM_HINT_T0);
            }
        }
        
        // Prefetch column after next (for deeper pipeline)
        if (j + 2 < ib && j + 2 < m)
        {
            float *next_next_col = &panel[(j + 2) * lda + (j + 2)];
            uint16_t prefetch_len = MIN(32, m - j - 2);
            
            for (uint16_t i = 0; i < prefetch_len; i += 8)
            {
                _mm_prefetch((const char*)&next_next_col[i * lda], _MM_HINT_T1);
            }
        }
#endif

        //======================================================================
        // Extract column j from panel
        //======================================================================
        
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

        // Store complete reflector in Y with proper stride
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
    float *restrict workspace,      // ✅ Pre-allocated workspace buffer
    size_t workspace_size,          // ✅ Available space (in floats)
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

    // ✅ Calculate required workspace
    size_t y_left_size = (size_t)m * ib1;
    size_t y_right_size = (size_t)(m - ib1) * ib2;
    size_t required = y_left_size + y_right_size;
    
    // ✅ Fallback to base case if insufficient workspace
    if (workspace_size < required)
    {
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, work);
        return;
    }
    
    // ✅ Partition workspace (no malloc!)
    float *Y_left = workspace;
    float *Y_right = workspace + y_left_size;
    float *workspace_next = workspace + required;
    size_t workspace_next_size = workspace_size - required;

    //==========================================================================
    // Left recursion
    //==========================================================================
    
    panel_factor_recursive(
        panel, Y_left, tau,
        workspace_next, workspace_next_size,  // ✅ Pass remaining workspace
        work,
        m, ib1, lda, ib1, threshold);

    // Copy Y_left to output Y
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < ib1; ++j)
            Y[i * ldy + j] = Y_left[i * ib1 + j];

    //==========================================================================
    // Apply left reflectors to right columns
    //==========================================================================
    
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

    //==========================================================================
    // Right recursion
    //==========================================================================
    
    float *right_panel = &panel[ib1 * lda + ib1];
    panel_factor_recursive(
        right_panel, Y_right, &tau[ib1],
        workspace_next, workspace_next_size,  // ✅ Pass remaining workspace
        work,
        m - ib1, ib2, lda, ib2, threshold);

    // Copy Y_right to output Y
    for (uint16_t i = 0; i < ib1; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            Y[i * ldy + (ib1 + j)] = 0.0f;

    for (uint16_t i = 0; i < m - ib1; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            Y[(ib1 + i) * ldy + (ib1 + j)] = Y_right[i * ib2 + j];

    // ✅ No free() needed!
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

    // ✅ Calculate available workspace
    size_t workspace_size = 2 * (size_t)workspace->m_max * workspace->ib;
    
    panel_factor_recursive(
        panel, Y, tau,
        workspace->panel_Y_temp,     // ✅ Pre-allocated buffer
        workspace_size,              // ✅ Size in floats
        workspace->tmp,
        m, ib, lda, ldy, threshold);
}

//==============================================================================
// BUILD T MATRIX (WITH STRIDE SUPPORT)
//==============================================================================


static void build_T_matrix(const float *restrict Y, const float *restrict tau,
                           float *restrict T, uint16_t m, uint16_t ib,
                           uint16_t ldy)
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

        //======================================================================
        // ✅ PREFETCH: Next Y column
        //======================================================================
        
#ifdef __AVX2__
        if (i + 1 < ib)
        {
            // Prefetch next column of Y
            uint16_t prefetch_rows = MIN(64, m);
            for (uint16_t r = 0; r < prefetch_rows; r += 8)
            {
                _mm_prefetch((const char*)&Y[r * ldy + (i + 1)], _MM_HINT_T0);
            }
        }
#endif

        //======================================================================
        // Compute w = -tau[i] * Y^T[:,0:i] * Y[:,i]
        //======================================================================
        
        for (uint16_t j = 0; j < i; ++j)
        {
#ifdef __AVX2__
            // Prefetch ahead in the dot product
            if (j + 8 < i)
            {
                for (uint16_t r = 0; r < MIN(32, m); r += 8)
                {
                    _mm_prefetch((const char*)&Y[r * ldy + (j + 8)], _MM_HINT_T1);
                }
            }
            
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

        //======================================================================
        // Compute T[:,i] = T * w
        //======================================================================
        
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
    float *restrict YT)
{
    //==========================================================================
    // Step 1: Transpose Y → YT
    //==========================================================================
    
#ifdef __AVX2__
    if (m >= 8 && ib >= 8)
    {
        transpose_avx2_8x8(Y, YT, m, ib, ldy, m);
    }
    else
#endif
    {
        for (uint16_t i = 0; i < ib; ++i)
        {
#ifdef __AVX2__
            // Prefetch next row of Y
            if (i + 1 < ib)
            {
                for (uint16_t j = 0; j < m; j += 16)
                {
                    _mm_prefetch((const char*)&Y[j * ldy + (i + 1)], _MM_HINT_T0);
                }
            }
#endif
            
            for (uint16_t j = 0; j < m; ++j)
                YT[i * m + j] = Y[j * ldy + i];
        }
    }

    //==========================================================================
    // Step 2: Z = Y^T * C (prefetch C during GEMM)
    //==========================================================================
    
#ifdef __AVX2__
    // Prefetch C for the GEMM operation
    for (uint16_t i = 0; i < MIN(64, m); i += 8)
    {
        for (uint16_t j = 0; j < MIN(64, n); j += 16)
        {
            _mm_prefetch((const char*)&C[i * n + j], _MM_HINT_T0);
        }
    }
#endif

    int ret = GEMM_CALL(Z, YT, C, ib, m, n, 1.0f, 0.0f);
    if (ret != 0)
        return ret;

    //==========================================================================
    // Step 3: Z_temp = T * Z (prefetch T)
    //==========================================================================
    
#ifdef __AVX2__
    // Prefetch T matrix
    for (uint16_t i = 0; i < ib; i += 16)
    {
        _mm_prefetch((const char*)&T[i], _MM_HINT_T0);
    }
#endif

    ret = GEMM_CALL(Z_temp, T, Z, ib, ib, n, 1.0f, 0.0f);
    if (ret != 0)
        return ret;

    //==========================================================================
    // Step 4: Copy Y to contiguous buffer (reuse YT)
    //==========================================================================
    
    float *Y_contig = YT;

#ifdef __AVX2__
    if (m >= 8 && ib >= 8)
    {
        copy_strided_to_contiguous_avx2(Y_contig, Y, m, ib, ldy);
    }
    else
#endif   
    {
        for (uint16_t i = 0; i < m; ++i)
        {
#ifdef __AVX2__
            // Prefetch ahead
            if (i + 8 < m)
            {
                _mm_prefetch((const char*)&Y[(i + 8) * ldy], _MM_HINT_T0);
            }
#endif
            
            for (uint16_t j = 0; j < ib; ++j)
                Y_contig[i * ib + j] = Y[i * ldy + j];
        }
    }

    //==========================================================================
    // Step 5: C -= Y * Z_temp (prefetch C for update)
    //==========================================================================
    
#ifdef __AVX2__
    // Prefetch C for the final update
    for (uint16_t i = 0; i < MIN(64, m); i += 8)
    {
        for (uint16_t j = 0; j < MIN(64, n); j += 16)
        {
            _mm_prefetch((const char*)&C[i * n + j], _MM_HINT_T0);
        }
    }
#endif

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

    // ✅ Z = Y^T * C using strided GEMM
    // Y is m×ib with stride ldy, C is m×n with stride ldc
    // We need Y^T * C = [ib×m] × [m×n] = [ib×n]
    
    // Since YT is already transposed and contiguous, use it directly
    naive_gemm_strided(Z, YT, C, 
                          ib, m, n,           // logical dimensions
                          n, m, ldc,          // strides: Z is ib×n, YT is ib×m, C has stride ldc
                          1.0f, 0.0f);

    // ✅ Z_temp = T * Z (both contiguous)
    naive_gemm_strided(Z_temp, T, Z,
                      ib, ib, n,
                      n, ib, n,              // all contiguous
                      1.0f, 0.0f);
    

    // ✅ C = C - Y * Z_temp using strided GEMM
    // Y is m×ib with stride ldy, Z_temp is ib×n (contiguous)
    // C is m×n with stride ldc
    naive_gemm_strided(C, Y, Z_temp,
                      m, ib, n,
                      ldc, ldy, n,           // C has stride ldc, Y has stride ldy
                      -1.0f, 1.0f);
    
    return 0;
}


//==============================================================================
// LEFT-LOOKING BLOCKED QR FACTORIZATION
//==============================================================================

/**
 * @brief Apply a stored block reflector to a panel
 * 
 * Computes: panel = (I - Y*T*Y^T) * panel
 * where Y and T are loaded from storage (previous block)
 * 
 * @param ws Workspace
 * @param A Full matrix (for indexing)
 * @param panel_col Starting column of panel to update
 * @param panel_width Width of panel (number of columns)
 * @param block_idx Which stored block to load (0, 1, 2, ...)
 * @param m Total rows in matrix
 * @param n Total columns in matrix
 * @param kmax min(m, n)
 * @return 0 on success, negative on error
 */
static int apply_stored_block_to_panel(
    qr_workspace *ws,
    float *A,
    uint16_t panel_col,
    uint16_t panel_width,
    uint16_t block_idx,
    uint16_t m,
    uint16_t n,
    uint16_t kmax)
{
    // Compute dimensions of the stored block
    uint16_t blk_k = block_idx * ws->ib;
    uint16_t blk_size = MIN(ws->ib, kmax - blk_k);
    uint16_t blk_rows_below = m - blk_k;
    
    // The reflector affects rows [blk_k : m-1]
    // The panel spans columns [panel_col : panel_col+panel_width-1]
    // So we update A[blk_k:m-1, panel_col:panel_col+panel_width-1]
    
    uint16_t update_rows = blk_rows_below;
    uint16_t update_cols = panel_width;
    
    if (update_rows == 0 || update_cols == 0)
        return 0;
    
    // Load stored Y and T for this block
    size_t y_offset = block_idx * ws->Y_block_stride;
    size_t t_offset = block_idx * ws->T_block_stride;
    
    // Y is stored in packed format: [blk_rows_below × blk_size]
    // We need to load it into ws->Y with proper layout
    memset(ws->Y, 0, (size_t)m * ws->ib * sizeof(float));
    
    for (uint16_t i = 0; i < blk_rows_below; ++i)
        for (uint16_t j = 0; j < blk_size; ++j)
            ws->Y[(blk_k + i) * ws->ib + j] = 
                ws->Y_stored[y_offset + i * blk_size + j];
    
    // Load T matrix
    memcpy(ws->T, &ws->T_stored[t_offset],
           blk_size * blk_size * sizeof(float));
    
    // Apply block reflector to panel: A[blk_k:m, panel_col:panel_col+width]
    // This is a strided GEMM operation because the panel is within A
    
    float *panel_ptr = &A[blk_k * n + panel_col];
    
    return apply_block_reflector_strided(
        panel_ptr,           // Panel to update (strided within A)
        ws->Y,               // Householder vectors [m × blk_size]
        ws->T,               // T matrix [blk_size × blk_size]
        update_rows,         // Number of rows to update
        update_cols,         // Number of columns in panel
        blk_size,            // Block size
        n,                   // Stride of panel (columns in A)
        ws->ib,              // Stride of Y
        ws->Z,               // Workspace
        ws->Z_temp,          // Workspace
        ws->YT);             // Workspace
}

/**
 * @brief Left-looking blocked QR factorization
 * 
 * For each panel k:
 *   1. Apply all previous reflectors H_0, ..., H_{k-1} to panel k
 *   2. Factor the updated panel
 *   3. Apply new reflector H_k to trailing matrix
 * 
 * Better cache locality: All updates to panel k happen together before factorization.
 * 
 * @param ws Workspace
 * @param A [in/out] Matrix to factor [m×n]
 * @param m Number of rows
 * @param n Number of columns
 * @return Number of blocks processed, or negative error code
 */
static int qr_factor_blocked_left_looking(qr_workspace *ws, float *A,
                                          uint16_t m, uint16_t n)
{
    const uint16_t kmax = MIN(m, n);
    uint16_t block_count = 0;

    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;
        uint16_t cols_right = (n > k + block_size) ? (n - k - block_size) : 0;

        //======================================================================
        // LEFT-LOOKING STEP: Apply all previous block reflectors to panel k
        //======================================================================
        
        for (uint16_t prev_blk = 0; prev_blk < block_count; prev_blk++)
        {
            int ret = apply_stored_block_to_panel(
                ws, A,
                k,              // Panel starts at column k
                block_size,     // Panel width
                prev_blk,       // Which previous block to apply
                m, n, kmax);
            
            if (ret != 0)
                return ret;
        }

        //======================================================================
        // Factor the updated panel
        //======================================================================
        
        panel_factor_optimized(&A[k * n + k], ws->Y, &ws->tau[k],
                               rows_below, block_size, n, ws->ib, ws);

        // Build T matrix
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size, ws->ib);

        //======================================================================
        // Store Y and T for future panels (left-looking needs this!)
        //======================================================================
        
        if (ws->Y_stored && ws->T_stored)
        {
            size_t y_offset = block_count * ws->Y_block_stride;
            size_t t_offset = block_count * ws->T_block_stride;

            for (uint16_t i = 0; i < rows_below; ++i)
                for (uint16_t j = 0; j < block_size; ++j)
                    ws->Y_stored[y_offset + i * block_size + j] =
                        ws->Y[i * ws->ib + j];

            memcpy(&ws->T_stored[t_offset], ws->T,
                   block_size * block_size * sizeof(float));
        }
        else
        {
            // Left-looking REQUIRES reflector storage!
            return -EINVAL;
        }

        //======================================================================
        // Apply to trailing matrix (same as right-looking)
        //======================================================================
        
        if (cols_right > 0)
        {
            for (uint16_t j = 0; j < block_size && (k + j) < m; ++j)
            {
                uint16_t reflector_len = m - (k + j);
                float *col_j = &A[(k + j) * n + (k + j)];

                ws->tmp[0] = 1.0f;
                for (uint16_t i = 1; i < reflector_len; ++i)
                {
                    ws->tmp[i] = col_j[i * n];
                }

                uint16_t row_start = k + j;
                const uint16_t col_block = 32;

                for (uint16_t jj = 0; jj < cols_right; jj += col_block)
                {
                    uint16_t n_block = MIN(col_block, cols_right - jj);

                    apply_householder_clean(
                        &A[row_start * n + (k + block_size + jj)],
                        reflector_len,
                        n_block,
                        n,
                        ws->tmp,
                        ws->tau[k + j]);
                }
            }
        }

        block_count++;
    }

    return block_count;
}

//==============================================================================
// PHASE 1: BLOCKED QR FACTORIZATION
//==============================================================================

/**
 * @brief Perform blocked QR factorization, storing reflectors in A and Y/T
 * 
 * @param ws Workspace
 * @param A [in/out] Matrix to factor [m×n], gets overwritten with R and reflectors
 * @param m Number of rows
 * @param n Number of columns
 * @return Number of blocks processed, or negative error code
 */
static int qr_factor_blocked(qr_workspace *ws, float *A, uint16_t m, uint16_t n)
{
    const uint16_t kmax = MIN(m, n);
    uint16_t block_count = 0;

    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;
        uint16_t cols_right = (n > k + block_size) ? (n - k - block_size) : 0;

        //======================================================================
        // ✅ PREFETCH: Next panel (2 blocks ahead for better timing)
        //======================================================================
        
#ifdef __AVX2__
        if (k + 2 * ws->ib < kmax)
        {
            uint16_t next_k = k + 2 * ws->ib;
            uint16_t prefetch_rows = MIN(64, m - next_k);
            uint16_t next_cols = MIN(ws->ib, kmax - next_k);
            
            float *next_panel = &A[next_k * n + next_k];
            
            // Prefetch panel in 64-byte cache line chunks
            for (uint16_t i = 0; i < prefetch_rows; i += 8)
            {
                for (uint16_t j = 0; j < next_cols; j += 16)
                {
                    _mm_prefetch((const char*)&next_panel[i * n + j], _MM_HINT_T1);
                }
            }
        }
        
        // Also prefetch trailing matrix start (if exists)
        if (cols_right > 0 && k + ws->ib < kmax)
        {
            float *trailing_start = &A[k * n + (k + block_size)];
            uint16_t prefetch_rows = MIN(32, rows_below);
            uint16_t prefetch_cols = MIN(32, cols_right);
            
            for (uint16_t i = 0; i < prefetch_rows; i += 8)
            {
                for (uint16_t j = 0; j < prefetch_cols; j += 16)
                {
                    _mm_prefetch((const char*)&trailing_start[i * n + j], _MM_HINT_T1);
                }
            }
        }
#endif

        //======================================================================
        // Factor current panel
        //======================================================================
        
        panel_factor_optimized(&A[k * n + k], ws->Y, &ws->tau[k],
                               rows_below, block_size, n, ws->ib, ws);

        // Build T matrix
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size, ws->ib);

        // Store Y and T for Q formation
        if (ws->Y_stored && ws->T_stored)
        {
            size_t y_offset = block_count * ws->Y_block_stride;
            size_t t_offset = block_count * ws->T_block_stride;

            for (uint16_t i = 0; i < rows_below; ++i)
                for (uint16_t j = 0; j < block_size; ++j)
                    ws->Y_stored[y_offset + i * block_size + j] =
                        ws->Y[i * ws->ib + j];

            memcpy(&ws->T_stored[t_offset], ws->T,
                   block_size * block_size * sizeof(float));
        }

        // Apply to trailing matrix
        if (cols_right > 0)
        {
            for (uint16_t j = 0; j < block_size && (k + j) < m; ++j)
            {
                uint16_t reflector_len = m - (k + j);
                float *col_j = &A[(k + j) * n + (k + j)];

                ws->tmp[0] = 1.0f;
                for (uint16_t i = 1; i < reflector_len; ++i)
                {
                    ws->tmp[i] = col_j[i * n];
                }

                uint16_t row_start = k + j;
                const uint16_t col_block = 32;

                for (uint16_t jj = 0; jj < cols_right; jj += col_block)
                {
                    uint16_t n_block = MIN(col_block, cols_right - jj);

                    apply_householder_clean(
                        &A[row_start * n + (k + block_size + jj)],
                        reflector_len,
                        n_block,
                        n,
                        ws->tmp,
                        ws->tau[k + j]);
                }
            }
        }

        block_count++;
    }

    return block_count;
}
//==============================================================================
// PHASE 2: EXTRACT R MATRIX
//==============================================================================

/**
 * @brief Extract upper triangular R from factored matrix A
 * 
 * @param R [out] Output R matrix [m×n]
 * @param A [in] Factored matrix (R in upper triangle)
 * @param m Number of rows
 * @param n Number of columns
 */
static void qr_extract_r(float *restrict R, const float *restrict A,
                         uint16_t m, uint16_t n)
{
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            R[i * n + j] = (i <= j) ? A[i * n + j] : 0.0f;
        }
    }
}

//==============================================================================
// PHASE 3: FORM ORTHOGONAL Q MATRIX
//==============================================================================

/**
 * @brief Form orthogonal matrix Q from stored Householder reflectors
 * 
 * Applies reflectors in reverse order: Q = H(1) * H(2) * ... * H(k)
 * Uses block reflector representation: H = I - Y*T*Y^T
 * 
 * @param ws Workspace containing stored Y and T matrices
 * @param Q [out] Output Q matrix [m×m]
 * @param m Number of rows
 * @param n Number of columns (original matrix)
 * @param block_count Number of blocks to process
 * @return 0 on success, negative error code on failure
 */
static int qr_form_q(qr_workspace *ws, float *Q, uint16_t m, uint16_t n,
                     uint16_t block_count)
{
    if (!ws->Y_stored || !ws->T_stored)
        return -EINVAL;

    const uint16_t kmax = MIN(m, n);

    // Initialize Q = I
    memset(Q, 0, (size_t)m * m * sizeof(float));
    for (uint16_t i = 0; i < m; ++i)
        Q[i * m + i] = 1.0f;

    // Apply blocks in reverse order
    for (int blk = block_count - 1; blk >= 0; blk--)
    {
        uint16_t k = blk * ws->ib;
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;

        size_t y_offset = blk * ws->Y_block_stride;
        size_t t_offset = blk * ws->T_block_stride;

        //======================================================================
        // ✅ PREFETCH: Next block's Y and T (if exists)
        //======================================================================
        
#ifdef __AVX2__
        if (blk > 0)
        {
            size_t next_y_offset = (blk - 1) * ws->Y_block_stride;
            size_t next_t_offset = (blk - 1) * ws->T_block_stride;
            
            // Prefetch next Y_stored
            for (size_t i = 0; i < MIN(1024, ws->Y_block_stride); i += 16)
            {
                _mm_prefetch((const char*)&ws->Y_stored[next_y_offset + i], _MM_HINT_T1);
            }
            
            // Prefetch next T_stored
            for (size_t i = 0; i < ws->T_block_stride; i += 16)
            {
                _mm_prefetch((const char*)&ws->T_stored[next_t_offset + i], _MM_HINT_T1);
            }
        }
#endif

        //======================================================================
        // Load stored Y matrix for this block
        //======================================================================
        
        memset(ws->Y, 0, (size_t)m * ws->ib * sizeof(float));
        for (uint16_t i = 0; i < rows_below; ++i)
            for (uint16_t j = 0; j < block_size; ++j)
                ws->Y[(k + i) * ws->ib + j] =
                    ws->Y_stored[y_offset + i * block_size + j];

        // Load stored T matrix for this block
        memcpy(ws->T, &ws->T_stored[t_offset],
               block_size * block_size * sizeof(float));

        // Apply block reflector: Q = Q * (I - Y*T*Y^T)
        int ret = apply_block_reflector_clean(
            Q, ws->Y, ws->T,
            m, m, block_size, ws->ib,
            ws->Z, ws->Z_temp, ws->YT);

        if (ret != 0)
            return ret;
    }

    return 0;
}

//==============================================================================
// MAIN WRAPPER (UNCHANGED SIGNATURE)
//==============================================================================

/**
 * @brief Blocked QR decomposition with in-place factorization
 * 
 * Computes A = Q*R where Q is orthogonal and R is upper triangular.
 * 
 * @param ws Pre-allocated workspace
 * @param A [in/out] Input matrix [m×n], gets overwritten during factorization
 * @param Q [out] Orthogonal matrix [m×m] (if !only_R)
 * @param R [out] Upper triangular matrix [m×n]
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R If true, skip Q formation (faster for least squares)
 * @return 0 on success, negative error code on failure
 */
int qr_ws_blocked_inplace(qr_workspace *ws, float *A, float *Q, float *R,
                          uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;
    if (m > ws->m_max || n > ws->n_max)
        return -EINVAL;

    //==========================================================================
    // Phase 1: Factorization (A → R + reflectors)
    //==========================================================================
    
    int block_count = qr_factor_blocked(ws, A, m, n);
    if (block_count < 0)
        return block_count;

    //==========================================================================
    // Phase 2: Extract R
    //==========================================================================
    
    qr_extract_r(R, A, m, n);

    //==========================================================================
    // Phase 3: Form Q (optional)
    //==========================================================================
    
    if (!only_R && Q)
    {
        return qr_form_q(ws, Q, m, n, block_count);
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

//==============================================================================
// WRAPPER WITH ALGORITHM SELECTION
//==============================================================================

/**
 * @brief Blocked QR with algorithm selection
 * 
 * @param ws Workspace
 * @param A [in/out] Matrix to factor
 * @param Q [out] Orthogonal matrix (if !only_R)
 * @param R [out] Upper triangular matrix
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R Skip Q formation
 * @param left_looking If true, use left-looking algorithm
 * @return 0 on success
 */
int qr_ws_blocked_inplace_ex(qr_workspace *ws, float *A, float *Q, float *R,
                             uint16_t m, uint16_t n, bool only_R,
                             bool left_looking)
{
    if (!ws || !A || !R)
        return -EINVAL;
    if (m > ws->m_max || n > ws->n_max)
        return -EINVAL;

    //==========================================================================
    // Phase 1: Factorization (choose algorithm)
    //==========================================================================
    
    int block_count;
    
    if (left_looking)
    {
        block_count = qr_factor_blocked_left_looking(ws, A, m, n);
    }
    else
    {
        block_count = qr_factor_blocked(ws, A, m, n);
    }
    
    if (block_count < 0)
        return block_count;

    //==========================================================================
    // Phase 2: Extract R
    //==========================================================================
    
    qr_extract_r(R, A, m, n);

    //==========================================================================
    // Phase 3: Form Q (optional)
    //==========================================================================
    
    if (!only_R && Q)
    {
        return qr_form_q(ws, Q, m, n, block_count);
    }

    return 0;
}


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
                                               2 * (size_t)m_max * ws->ib * sizeof(float));

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

    bytes += 2 * (size_t)m_max * ws->ib * sizeof(float); // panel_Y_temp (was 1×, now 2×)
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