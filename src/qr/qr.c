/**
 * @file qr_blocked.c (COMPLETE FIXED VERSION)
 * @brief GEMM-Accelerated Blocked QR with Recursive Panel Factorization
 */

#include "qr.h"
#include "../gemm_2/gemm.h"
#include "../gemm_2/gemm_planning.h"
#include "../gemm_2/gemm_utils.h"
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

//==============================================================================
// HOUSEHOLDER REFLECTION PRIMITIVES
//==============================================================================

#ifdef __AVX2__
static inline double compute_norm_sq_avx2(const float *restrict x, uint16_t len)
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    uint16_t i = 0;
    for (; i + 7 < len; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&x[i]);
        __m128 v_lo = _mm256_castps256_ps128(v);
        __m128 v_hi = _mm256_extractf128_ps(v, 1);
        __m256d v_lo_d = _mm256_cvtps_pd(v_lo);
        __m256d v_hi_d = _mm256_cvtps_pd(v_hi);
        acc0 = _mm256_fmadd_pd(v_lo_d, v_lo_d, acc0);
        acc1 = _mm256_fmadd_pd(v_hi_d, v_hi_d, acc1);
    }

    acc0 = _mm256_add_pd(acc0, acc1);
    __m128d lo = _mm256_castpd256_pd128(acc0);
    __m128d hi = _mm256_extractf128_pd(acc0, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    double result = _mm_cvtsd_f64(sum);

    for (; i < len; ++i)
    {
        double xi = (double)x[i];
        result += xi * xi;
    }

    return result;
}
#endif

static void compute_householder_clean(float *restrict x, uint16_t m,
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

#ifdef __AVX2__
    double norm_sq = (m > 9) ? compute_norm_sq_avx2(&x[1], m - 1) : 0.0;
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

#ifdef __AVX2__
static void apply_householder_avx2(float *restrict C, uint16_t m, uint16_t n,
                                   uint16_t ldc, const float *restrict v, float tau)
{
    if (tau == 0.0f)
        return;

    uint16_t j = 0;
    for (; j + 7 < n; j += 8)
    {
        __m256d dot_acc_lo = _mm256_setzero_pd();
        __m256d dot_acc_hi = _mm256_setzero_pd();

        uint16_t i = 0;
        for (; i + 1 < m; i += 2)
        {
            if (i + 8 < m)
                _mm_prefetch((const char *)(&C[(i + 8) * ldc + j]), _MM_HINT_T0);

            __m256 c_row = _mm256_loadu_ps(&C[i * ldc + j]);
            __m128 c_lo = _mm256_castps256_ps128(c_row);
            __m128 c_hi = _mm256_extractf128_ps(c_row, 1);
            __m256d c_lo_d = _mm256_cvtps_pd(c_lo);
            __m256d c_hi_d = _mm256_cvtps_pd(c_hi);
            __m256d v_d = _mm256_set1_pd((double)v[i]);
            dot_acc_lo = _mm256_fmadd_pd(v_d, c_lo_d, dot_acc_lo);
            dot_acc_hi = _mm256_fmadd_pd(v_d, c_hi_d, dot_acc_hi);

            c_row = _mm256_loadu_ps(&C[(i + 1) * ldc + j]);
            c_lo = _mm256_castps256_ps128(c_row);
            c_hi = _mm256_extractf128_ps(c_row, 1);
            c_lo_d = _mm256_cvtps_pd(c_lo);
            c_hi_d = _mm256_cvtps_pd(c_hi);
            v_d = _mm256_set1_pd((double)v[i + 1]);
            dot_acc_lo = _mm256_fmadd_pd(v_d, c_lo_d, dot_acc_lo);
            dot_acc_hi = _mm256_fmadd_pd(v_d, c_hi_d, dot_acc_hi);
        }

        for (; i < m; ++i)
        {
            __m256 c_row = _mm256_loadu_ps(&C[i * ldc + j]);
            __m128 c_lo = _mm256_castps256_ps128(c_row);
            __m128 c_hi = _mm256_extractf128_ps(c_row, 1);
            __m256d c_lo_d = _mm256_cvtps_pd(c_lo);
            __m256d c_hi_d = _mm256_cvtps_pd(c_hi);
            __m256d v_d = _mm256_set1_pd((double)v[i]);
            dot_acc_lo = _mm256_fmadd_pd(v_d, c_lo_d, dot_acc_lo);
            dot_acc_hi = _mm256_fmadd_pd(v_d, c_hi_d, dot_acc_hi);
        }

        __m128 dot_lo_f = _mm256_cvtpd_ps(dot_acc_lo);
        __m128 dot_hi_f = _mm256_cvtpd_ps(dot_acc_hi);
        __m256 dot_f = _mm256_insertf128_ps(_mm256_castps128_ps256(dot_lo_f), dot_hi_f, 1);
        __m256 tau_dot = _mm256_mul_ps(_mm256_set1_ps(tau), dot_f);

        i = 0;
        for (; i + 1 < m; i += 2)
        {
            __m256 v_bc = _mm256_set1_ps(v[i]);
            __m256 c_r = _mm256_loadu_ps(&C[i * ldc + j]);
            __m256 upd = _mm256_fnmadd_ps(v_bc, tau_dot, c_r);
            _mm256_storeu_ps(&C[i * ldc + j], upd);

            v_bc = _mm256_set1_ps(v[i + 1]);
            c_r = _mm256_loadu_ps(&C[(i + 1) * ldc + j]);
            upd = _mm256_fnmadd_ps(v_bc, tau_dot, c_r);
            _mm256_storeu_ps(&C[(i + 1) * ldc + j], upd);
        }

        for (; i < m; ++i)
        {
            __m256 v_bc = _mm256_set1_ps(v[i]);
            __m256 c_r = _mm256_loadu_ps(&C[i * ldc + j]);
            __m256 upd = _mm256_fnmadd_ps(v_bc, tau_dot, c_r);
            _mm256_storeu_ps(&C[i * ldc + j], upd);
        }
    }

    for (; j < n; ++j)
    {
        double dot = 0.0;
        for (uint16_t i = 0; i < m; ++i)
            dot += (double)v[i] * (double)C[i * ldc + j];

        float tau_dot = tau * (float)dot;
        for (uint16_t i = 0; i < m; ++i)
            C[i * ldc + j] -= v[i] * tau_dot;
    }
}
#endif

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
    uint16_t ldy,       // ✅ ADDED
    float *restrict work)
{
    // ✅ Clear Y with proper stride
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
        compute_householder_clean(work, col_len, &tau[j], &beta);

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

    float *Y_left = (float*)malloc(m * ib1 * sizeof(float));
    float *Y_right = (float*)malloc((m - ib1) * ib2 * sizeof(float));  // ✅ Only m-ib1 rows needed
    
    if (!Y_left || !Y_right)
    {
        free(Y_left);
        free(Y_right);
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, work);
        return;
    }

    //--------------------------------------------------------------------------
    // STEP 1: Factor LEFT HALF (rows 0:m, cols 0:ib1)
    //--------------------------------------------------------------------------
    
    panel_factor_recursive(
        panel, Y_left, tau,
        T_workspace, Z_workspace, work,
        m, ib1, lda, ib1, threshold);

    // Copy to main Y
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < ib1; ++j)
            Y[i * ldy + j] = Y_left[i * ib1 + j];

    //--------------------------------------------------------------------------
    // STEP 2: Apply left reflectors to right columns (one at a time)
    //--------------------------------------------------------------------------
    
    float *right_cols = &panel[ib1];  // Right columns start at col ib1
    
    for (uint16_t j = 0; j < ib1; ++j)
    {
        // Reflector j affects rows [j:m]
        uint16_t rows_affected = m - j;
        
        // Extract reflector j from Y_left
        for (uint16_t i = 0; i < rows_affected; ++i)
            work[i] = Y_left[(j + i) * ib1 + j];
        
        // Apply to right columns, starting at row j
        float *right_start = &right_cols[j * lda];
        apply_householder_clean(right_start, rows_affected, ib2,
                               lda, work, tau[j]);
    }

    //--------------------------------------------------------------------------
    // STEP 3: Factor RIGHT HALF ✅ FIXED
    //--------------------------------------------------------------------------
    
    // ✅ Key fix: right panel's diagonal starts at row ib1, not row 0!
    float *right_panel = &panel[ib1 * lda + ib1];  // Start at element (ib1, ib1)
    
    panel_factor_recursive(
        right_panel,                // Panel starting at diagonal
        Y_right,                    // Temp Y [m-ib1 × ib2]
        &tau[ib1],                  // tau for right columns
        T_workspace,
        Z_workspace,
        work,
        m - ib1,                    // ✅ Only remaining rows!
        ib2,                        // Columns in right panel
        lda,                        // Stride unchanged
        ib2,                        // Y_right stride
        threshold);

    //--------------------------------------------------------------------------
    // STEP 4: Copy Y_right to main Y ✅ FIXED
    //--------------------------------------------------------------------------
    
    // Zero out above-diagonal part (rows 0:ib1 of right columns)
    for (uint16_t i = 0; i < ib1; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            Y[i * ldy + (ib1 + j)] = 0.0f;
    
    // Copy the actual reflectors (starting at global row ib1)
    for (uint16_t i = 0; i < m - ib1; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            Y[(ib1 + i) * ldy + (ib1 + j)] = Y_right[i * ib2 + j];

    free(Y_left);
    free(Y_right);
}

/**
 * @brief Wrapper with automatic threshold selection
 */
static void panel_factor_optimized(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,
    uint16_t ldy,       // ✅ ADDED
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

    float *T_workspace = (float*)malloc(ib * ib * sizeof(float));
    float *Z_workspace = (float*)malloc(ib * ib * sizeof(float));

    panel_factor_recursive(
        panel, Y, tau,
        T_workspace, Z_workspace, workspace->tmp,
        m, ib, lda, ldy, threshold);  // ✅ Pass ldy

    free(T_workspace);
    free(Z_workspace);
}

//==============================================================================
// BUILD T MATRIX (WITH STRIDE SUPPORT)
//==============================================================================

#ifdef __AVX2__
static inline double dot_product_strided_avx2(const float *restrict a,
                                              const float *restrict b,
                                              uint16_t len,
                                              uint16_t stride_a,
                                              uint16_t stride_b)
{
    __m256d acc = _mm256_setzero_pd();
    uint16_t i = 0;

    for (; i + 3 < len; i += 4)
    {
        __m128 va = _mm_setr_ps(a[i * stride_a], a[(i + 1) * stride_a],
                                a[(i + 2) * stride_a], a[(i + 3) * stride_a]);
        __m128 vb = _mm_setr_ps(b[i * stride_b], b[(i + 1) * stride_b],
                                b[(i + 2) * stride_b], b[(i + 3) * stride_b]);
        __m256d va_d = _mm256_cvtps_pd(va);
        __m256d vb_d = _mm256_cvtps_pd(vb);
        acc = _mm256_fmadd_pd(va_d, vb_d, acc);
    }

    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    double result = _mm_cvtsd_f64(sum);

    for (; i < len; ++i)
        result += (double)a[i * stride_a] * (double)b[i * stride_b];

    return result;
}
#endif

static void build_T_matrix(const float *restrict Y, const float *restrict tau,
                           float *restrict T, uint16_t m, uint16_t ib,
                           uint16_t ldy)  // ✅ ADDED
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
            double dot = (m >= 16) ? 
                dot_product_strided_avx2(&Y[j], &Y[i], m, ldy, ldy) : 0.0;
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
    float *restrict YT)
{
    float *YT_local = NULL;
    if (!YT)
    {
        YT_local = (float *)malloc(ib * m * sizeof(float));
        if (!YT_local)
            return -ENOMEM;
        YT = YT_local;
    }

    // Transpose with proper stride
    for (uint16_t i = 0; i < ib; ++i)
        for (uint16_t j = 0; j < m; ++j)
            YT[i * m + j] = Y[j * ldy + i];

    // Z = Y^T * C
    int ret = GEMM_CALL(Z, YT, C, ib, m, n, 1.0f, 0.0f);
    if (ret != 0)
    {
        if (YT_local)
            free(YT_local);
        return ret;
    }

    // Z_temp = T * Z
    ret = GEMM_CALL(Z_temp, T, Z, ib, ib, n, 1.0f, 0.0f);
    if (ret != 0)
    {
        if (YT_local)
            free(YT_local);
        return ret;
    }

    // ✅ FIXED: Make Y contiguous, then use regular GEMM
    float *Y_contig = (float*)malloc(m * ib * sizeof(float));
    if (!Y_contig)
    {
        if (YT_local)
            free(YT_local);
        return -ENOMEM;
    }
    
    // Copy Y to contiguous buffer
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < ib; ++j)
            Y_contig[i * ib + j] = Y[i * ldy + j];
    
    // C -= Y_contig * Z_temp
    ret = GEMM_CALL(C, Y_contig, Z_temp, m, ib, n, -1.0f, 1.0f);
    
    free(Y_contig);
    if (YT_local)
        free(YT_local);
    return ret;
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

    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;
        uint16_t cols_right = (n > k + block_size) ? (n - k - block_size) : 0;

        // Factor current panel
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

        // ✅ FIXED: Apply to trailing matrix
        if (cols_right > 0)
        {
            for (uint16_t j = 0; j < block_size && (k + j) < m; ++j)
            {
                // Reflector j affects rows [k+j .. m-1]
                uint16_t reflector_len = m - (k + j);
                
                // ✅ CRITICAL FIX: Extract reflector from A (not ws->Y!)
                // Reflector is stored in A[k+j:m-1, k+j] with implicit v[0]=1
                float *col_j = &A[(k + j) * n + (k + j)];
                
                ws->tmp[0] = 1.0f;  // Implicit first component
                for (uint16_t i = 1; i < reflector_len; ++i)
                {
                    ws->tmp[i] = col_j[i * n];  // Extract from A's lower triangle
                }

                // Apply to trailing matrix in column blocks
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

    // Extract R
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? A[i * n + j] : 0.0f;

    // Form Q (unchanged - uses stored Y which is correct)
    if (!only_R && Q)
    {
        memset(Q, 0, (size_t)m * m * sizeof(float));
        for (uint16_t i = 0; i < m; ++i)
            Q[i * m + i] = 1.0f;

        for (int blk = block_count - 1; blk >= 0; blk--)
        {
            uint16_t k = blk * ws->ib;
            uint16_t block_size = MIN(ws->ib, kmax - k);
            uint16_t rows_below = m - k;

            size_t y_offset = blk * ws->Y_block_stride;
            size_t t_offset = blk * ws->T_block_stride;

            memset(ws->Y, 0, (size_t)m * ws->ib * sizeof(float));
            for (uint16_t i = 0; i < rows_below; ++i)
                for (uint16_t j = 0; j < block_size; ++j)
                    ws->Y[(k + i) * ws->ib + j] = 
                        ws->Y_stored[y_offset + i * block_size + j];

            memcpy(ws->T, &ws->T_stored[t_offset], 
                   block_size * block_size * sizeof(float));

            int ret = apply_block_reflector_clean(
                Q, ws->Y, ws->T,
                m, m, block_size, ws->ib,
                ws->Z, ws->Z_temp, ws->YT);

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

    ws->Y_block_stride = (size_t)m_max * ws->ib;  // ✅ Max size per block
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

     
    bytes += (size_t)m_max * ws->ib * sizeof(float);      // panel_Y_temp
    bytes += ws->ib * ws->ib * sizeof(float) * 2;         // panel_T_temp, panel_Z_temp
    

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