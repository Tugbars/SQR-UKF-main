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

static void build_T_matrix(const float *Y, const float *tau, float *T,
                           uint16_t m, uint16_t ib);

#define GEMM_CALL gemm_dynamic

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
// BLOCK SIZE SELECTION (simplified but compatible)
//==============================================================================
static uint16_t select_optimal_ib(uint16_t m, uint16_t n)
{
    const uint16_t min_dim = MIN(m, n);
    uint16_t ib = MIN(64, min_dim); // Default block size

    // Adjust for small matrices
    if (min_dim < 32)
        ib = MIN(16, min_dim);
    else if (min_dim < 64)
        ib = MIN(32, min_dim);

    // Ensure minimum block size
    if (ib < 8)
        ib = MIN(8, min_dim);

    return ib;
}

//==============================================================================
// CLEAN HOUSEHOLDER REFLECTION
//==============================================================================

/**
 * @brief AVX2-optimized squared norm: ||x||^2
 * Uses FMA for better accuracy and performance
 */
static inline double compute_norm_sq_avx2(const float *restrict x, uint16_t len)
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    uint16_t i = 0;

    // Process 8 floats at a time (split into 2x4 for double accumulation)
    for (; i + 7 < len; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&x[i]);

        // Split into low/high 4 floats
        __m128 v_lo = _mm256_castps256_ps128(v);
        __m128 v_hi = _mm256_extractf128_ps(v, 1);

        // Convert to double precision
        __m256d v_lo_d = _mm256_cvtps_pd(v_lo);
        __m256d v_hi_d = _mm256_cvtps_pd(v_hi);

        // FMA: acc += v * v
        acc0 = _mm256_fmadd_pd(v_lo_d, v_lo_d, acc0);
        acc1 = _mm256_fmadd_pd(v_hi_d, v_hi_d, acc1);
    }

    // Horizontal reduction
    acc0 = _mm256_add_pd(acc0, acc1);
    __m128d lo = _mm256_castpd256_pd128(acc0);
    __m128d hi = _mm256_extractf128_pd(acc0, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);

    double result = _mm_cvtsd_f64(sum);

    // Scalar tail
    for (; i < len; ++i)
    {
        double xi = (double)x[i];
        result += xi * xi;
    }

    return result;
}

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

    // Compute norm of x[1:m]
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

    // Compute beta and scale
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
        {
            x[i] *= (float)scale;
        }
    }
    else
#endif
    {
        for (uint16_t i = 1; i < m; ++i)
        {
            x[i] *= (float)scale;
        }
    }

    *tau = (float)((beta_val - alpha) / beta_val);
    if (beta)
        *beta = (float)beta_val;
    x[0] = 1.0f;
}


#ifdef __AVX2__
/**
 * @brief AVX2-optimized Householder with 2-way unrolling (no spilling)
 */
static void apply_householder_avx2(float *restrict C, uint16_t m, uint16_t n,
                                   uint16_t ldc, const float *restrict v, float tau)
{
    if (tau == 0.0f)
        return;

    uint16_t j = 0;
    for (; j + 7 < n; j += 8)
    {
        // Step 1: Compute dot product with 2-way unrolling
        __m256d dot_acc_lo = _mm256_setzero_pd();
        __m256d dot_acc_hi = _mm256_setzero_pd();
        
        uint16_t i = 0;
        for (; i + 1 < m; i += 2)
        {
            // Prefetch
            if (i + 8 < m)
                _mm_prefetch((const char*)(&C[(i + 8) * ldc + j]), _MM_HINT_T0);
            
            // Iteration 0
            __m256 c_row = _mm256_loadu_ps(&C[i * ldc + j]);
            __m128 c_lo = _mm256_castps256_ps128(c_row);
            __m128 c_hi = _mm256_extractf128_ps(c_row, 1);
            __m256d c_lo_d = _mm256_cvtps_pd(c_lo);
            __m256d c_hi_d = _mm256_cvtps_pd(c_hi);
            __m256d v_d = _mm256_set1_pd((double)v[i]);
            dot_acc_lo = _mm256_fmadd_pd(v_d, c_lo_d, dot_acc_lo);
            dot_acc_hi = _mm256_fmadd_pd(v_d, c_hi_d, dot_acc_hi);
            
            // Iteration 1 (reuses same temp registers)
            c_row = _mm256_loadu_ps(&C[(i + 1) * ldc + j]);
            c_lo = _mm256_castps256_ps128(c_row);
            c_hi = _mm256_extractf128_ps(c_row, 1);
            c_lo_d = _mm256_cvtps_pd(c_lo);
            c_hi_d = _mm256_cvtps_pd(c_hi);
            v_d = _mm256_set1_pd((double)v[i + 1]);
            dot_acc_lo = _mm256_fmadd_pd(v_d, c_lo_d, dot_acc_lo);
            dot_acc_hi = _mm256_fmadd_pd(v_d, c_hi_d, dot_acc_hi);
        }
        
        // Tail
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
        
        // Convert and scale
        __m128 dot_lo_f = _mm256_cvtpd_ps(dot_acc_lo);
        __m128 dot_hi_f = _mm256_cvtpd_ps(dot_acc_hi);
        __m256 dot_f = _mm256_insertf128_ps(_mm256_castps128_ps256(dot_lo_f), dot_hi_f, 1);
        __m256 tau_dot = _mm256_mul_ps(_mm256_set1_ps(tau), dot_f);
        
        // Step 2: Update with 2-way unrolling
        i = 0;
        for (; i + 1 < m; i += 2)
        {
            // Iteration 0
            __m256 v_bc = _mm256_set1_ps(v[i]);
            __m256 c_r = _mm256_loadu_ps(&C[i * ldc + j]);
            __m256 upd = _mm256_fnmadd_ps(v_bc, tau_dot, c_r);
            _mm256_storeu_ps(&C[i * ldc + j], upd);
            
            // Iteration 1
            v_bc = _mm256_set1_ps(v[i + 1]);
            c_r = _mm256_loadu_ps(&C[(i + 1) * ldc + j]);
            upd = _mm256_fnmadd_ps(v_bc, tau_dot, c_r);
            _mm256_storeu_ps(&C[(i + 1) * ldc + j], upd);
        }
        
        // Tail
        for (; i < m; ++i)
        {
            __m256 v_bc = _mm256_set1_ps(v[i]);
            __m256 c_r = _mm256_loadu_ps(&C[i * ldc + j]);
            __m256 upd = _mm256_fnmadd_ps(v_bc, tau_dot, c_r);
            _mm256_storeu_ps(&C[i * ldc + j], upd);
        }
    }
    
    // Scalar tail
    for (; j < n; ++j)
    {
        double dot = 0.0;
        for (uint16_t i = 0; i < m; ++i)
        {
            dot += (double)v[i] * (double)C[i * ldc + j];
        }
        
        float tau_dot = tau * (float)dot;
        for (uint16_t i = 0; i < m; ++i)
        {
            C[i * ldc + j] -= v[i] * tau_dot;
        }
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
        {
            dot += (double)v[i] * (double)C[i * ldc + j];
        }

        float tau_dot = tau * (float)dot;
        for (uint16_t i = 0; i < m; ++i)
        {
            C[i * ldc + j] -= v[i] * tau_dot;
        }
    }
}

//==============================================================================
// CLEAN PANEL FACTORIZATION
//==============================================================================

/**
 * @brief Factor a panel - clean and correct version
 */
static void panel_factor_clean(
    float *restrict panel, // Panel to factor [m × ib], stride = lda
    float *restrict Y,     // Output: Householder vectors [m × ib]
    float *restrict tau,   // Output: tau values [ib]
    uint16_t m,            // Rows in panel
    uint16_t ib,           // Columns in panel
    uint16_t lda,          // Stride of full matrix
    float *restrict work)  // Workspace [m]
{
    // Clear Y first
    memset(Y, 0, m * ib * sizeof(float));

    for (uint16_t j = 0; j < ib && j < m; ++j)
    {
        uint16_t col_len = m - j;

        // Extract column j from panel
        float *restrict col_ptr = &panel[j * lda + j];
        for (uint16_t i = 0; i < col_len; ++i)
        {
            work[i] = col_ptr[i * lda];
        }

        // Compute Householder reflector
        float beta;
        compute_householder_clean(work, col_len, &tau[j], &beta);

        // Write beta to R diagonal
        col_ptr[0] = beta;

        // Write reflector tail back to panel
        for (uint16_t i = 1; i < col_len; ++i)
        {
            col_ptr[i * lda] = work[i];
        }

        // Store complete reflector in Y
        for (uint16_t i = 0; i < j; ++i)
        {
            Y[i * ib + j] = 0.0f; // Zero above diagonal
        }
        for (uint16_t i = 0; i < col_len; ++i)
        {
            Y[(j + i) * ib + j] = work[i]; // Store reflector
        }

        // Apply reflector to trailing columns
        if (j + 1 < ib)
        {
            float *restrict trailing = &panel[j * lda + (j + 1)];
            apply_householder_clean(trailing, col_len, ib - j - 1,
                                    lda, work, tau[j]);
        }
    }
}


//==============================================================================
// BUILD T MATRIX (keep existing implementation)
//==============================================================================

#ifdef __AVX2__
/**
 * @brief AVX2 dot product with stride (for column access in row-major matrices)
 */
static inline double dot_product_strided_avx2(const float *restrict a,
                                              const float *restrict b,
                                              uint16_t len,
                                              uint16_t stride_a,
                                              uint16_t stride_b)
{
    __m256d acc = _mm256_setzero_pd();

    uint16_t i = 0;

    // Process 4 elements at a time (stride makes 8 impractical)
    for (; i + 3 < len; i += 4)
    {
        // Gather 4 elements manually
        __m128 va = _mm_setr_ps(a[i * stride_a],
                                a[(i + 1) * stride_a],
                                a[(i + 2) * stride_a],
                                a[(i + 3) * stride_a]);
        __m128 vb = _mm_setr_ps(b[i * stride_b],
                                b[(i + 1) * stride_b],
                                b[(i + 2) * stride_b],
                                b[(i + 3) * stride_b]);

        __m256d va_d = _mm256_cvtps_pd(va);
        __m256d vb_d = _mm256_cvtps_pd(vb);

        acc = _mm256_fmadd_pd(va_d, vb_d, acc);
    }

    // Horizontal reduction
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);

    double result = _mm_cvtsd_f64(sum);

    // Scalar tail
    for (; i < len; ++i)
    {
        result += (double)a[i * stride_a] * (double)b[i * stride_b];
    }

    return result;
}
#endif

static void build_T_matrix(const float *restrict Y, const float *restrict tau,
                           float *restrict T, uint16_t m, uint16_t ib)
{
    memset(T, 0, (size_t)ib * ib * sizeof(float));
    if (ib == 0)
        return;

    double *w = NULL;
    double w_stack[64];

    if (ib <= 64)
    {
        w = w_stack;
    }
    else
    {
        w = (double *)malloc(ib * sizeof(double));
        if (!w)
            return;
    }

    for (uint16_t i = 0; i < ib; ++i)
    {
        T[i * ib + i] = tau[i];

        if (tau[i] == 0.0f || i == 0)
            continue;

        // Compute w = -tau[i] * Y^T[:,0:i-1] * Y[:,i]
        for (uint16_t j = 0; j < i; ++j)
        {
#ifdef __AVX2__
            double dot = (m >= 16) ? dot_product_strided_avx2(&Y[j], &Y[i], m, ib, ib) : 0.0;

            if (m < 16)
#endif
            {
                dot = 0.0;
                for (uint16_t r = 0; r < m; ++r)
                {
                    dot += (double)Y[r * ib + j] * (double)Y[r * ib + i];
                }
            }
            w[j] = -(double)tau[i] * dot;
        }

        // T[0:i-1, i] = T[0:i-1, 0:i-1] * w
        for (uint16_t j = 0; j < i; ++j)
        {
            double sum = 0.0;
            for (uint16_t k = 0; k < i; ++k)
            {
                sum += (double)T[j * ib + k] * w[k];
            }
            T[j * ib + i] = (float)sum;
        }
    }

    if (ib > 64)
        free(w);
}

//==============================================================================
// APPLY BLOCK REFLECTOR (keep existing but cleaned up)
//==============================================================================
static int apply_block_reflector_clean(
    float *restrict C,       // Matrix to update [m × n]
    const float *restrict Y, // Householder vectors [m × ib]
    const float *restrict T, // T matrix [ib × ib]
    uint16_t m, uint16_t n, uint16_t ib,
    float *restrict Z,      // Workspace [ib × n]
    float *restrict Z_temp, // Workspace [ib × n]
    float *restrict YT)     // Y transposed [ib × m]
{
    // Build YT if not provided
    float *YT_local = NULL;
    if (!YT)
    {
        YT_local = (float *)malloc(ib * m * sizeof(float));
        if (!YT_local)
            return -ENOMEM;
        YT = YT_local;
    }

    // Transpose Y to YT
    for (uint16_t i = 0; i < ib; ++i)
    {
        for (uint16_t j = 0; j < m; ++j)
        {
            YT[i * m + j] = Y[j * ib + i];
        }
    }

    // Step 1: Z = Y^T * C
    int ret = GEMM_CALL(Z, YT, C, ib, m, n, 1.0f, 0.0f);
    if (ret != 0)
    {
        if (YT_local)
            free(YT_local);
        return ret;
    }

    // Step 2: Z_temp = T * Z
    ret = GEMM_CALL(Z_temp, T, Z, ib, ib, n, 1.0f, 0.0f);
    if (ret != 0)
    {
        if (YT_local)
            free(YT_local);
        return ret;
    }

    // Step 3: C -= Y * Z_temp
    ret = GEMM_CALL(C, Y, Z_temp, m, ib, n, -1.0f, 1.0f);

    if (YT_local)
        free(YT_local);
    return ret;
}

//==============================================================================
// OPTIMIZED LEVEL-3 BLOCK REFLECTOR APPLICATION
//==============================================================================

/**
 * @brief Apply block reflector with optimized strided access
 *
 * Computes: C := (I - Y*T*Y^T) * C
 * where C is [m × n] with leading dimension ldc
 */
static int apply_block_reflector_strided_opt(
    float *restrict C,       // Submatrix to update [m × n], stride ldc
    const float *restrict Y, // Householder vectors [m × ib]
    const float *restrict T, // T matrix [ib × ib]
    uint16_t m,              // Rows in submatrix
    uint16_t n,              // Columns in submatrix
    uint16_t ib,             // Block size
    uint16_t ldc,            // Leading dimension (stride) of C
    float *restrict Z,       // Workspace [ib × n]
    float *restrict Z_temp)  // Workspace [ib × n]
{
    // Fast path: if already contiguous, skip packing
    if (ldc == n)
    {
        return apply_block_reflector_clean(C, Y, T, m, n, ib, Z, Z_temp, NULL);
    }

    // Otherwise: pack → compute → unpack
    // Note: For small n, overhead is acceptable; for large n, amortized across many rows

    float *C_packed = (float *)malloc((size_t)m * n * sizeof(float));
    if (!C_packed)
        return -ENOMEM;

    // Pack: Convert strided C to contiguous
    for (uint16_t i = 0; i < m; ++i)
    {
        memcpy(C_packed + i * n, C + i * ldc, n * sizeof(float));
    }

    // Apply block reflector on packed buffer
    int ret = apply_block_reflector_clean(C_packed, Y, T, m, n, ib, Z, Z_temp, NULL);

    if (ret == 0)
    {
        // Unpack: Write back to strided C
        for (uint16_t i = 0; i < m; ++i)
        {
            memcpy(C + i * ldc, C_packed + i * n, n * sizeof(float));
        }
    }

    free(C_packed);
    return ret;
}

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
        panel_factor_clean(&A[k * n + k], ws->Y, &ws->tau[k],
                           rows_below, block_size, n, ws->tmp);

        // Build T matrix
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size);

        // Store Y and T for Q formation
        if (ws->Y_stored && ws->T_stored)
        {
            size_t y_offset = block_count * ws->Y_block_stride;
            size_t t_offset = block_count * ws->T_block_stride;
            memcpy(&ws->Y_stored[y_offset], ws->Y, rows_below * block_size * sizeof(float));
            memcpy(&ws->T_stored[t_offset], ws->T, block_size * block_size * sizeof(float));
        }

        // Apply block reflector to trailing matrix
        // Apply block's reflectors to the trailing matrix (columns to the right)
        if (cols_right > 0)
        {
            // We apply each Householder reflector in the block explicitly
            // to the trailing submatrix A[row_start:m, k+block_size:n].
            for (uint16_t j = 0; j < block_size && (k + j) < m; ++j)
            {
                uint16_t row_start = k + j;     // global row where reflector j starts
                uint16_t m_sub = m - row_start; // rows affected by this reflector
                uint16_t n_sub = cols_right;    // number of trailing columns

                if (n_sub == 0)
                    break;

                // Build the v-vector for reflector j into ws->tmp[0..m_sub-1].
                // In ws->Y (rows_below × block_size), reflector j lives in
                // rows j..rows_below-1 (local), which correspond to global
                // rows (k + j)..(m - 1).
                for (uint16_t i = 0; i < m_sub; ++i)
                {
                    ws->tmp[i] = ws->Y[(j + i) * block_size + j];
                }

                // Apply H_j = I - tau_j * v v^T to the trailing block of A.
                apply_householder_clean(
                    &A[row_start * n + (k + block_size)], // C starting at (row_start, k+block_size)
                    m_sub,                                // rows
                    n_sub,                                // cols (cols_right)
                    n,                                    // ldc = full width of A
                    ws->tmp,                              // v
                    ws->tau[k + j]);                      // tau_j
            }
        }

        block_count++;
    }

    // Extract R from upper triangle
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            R[i * n + j] = (i <= j) ? A[i * n + j] : 0.0f;
        }
    }

    // Form Q
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
            float *Y_src = &ws->Y_stored[y_offset];
            float *T_src = &ws->T_stored[t_offset];

            float *Y_full = ws->Y;
            memset(Y_full, 0, m * block_size * sizeof(float));
            for (uint16_t i = 0; i < rows_below; ++i)
            {
                for (uint16_t j = 0; j < block_size; ++j)
                {
                    Y_full[(k + i) * block_size + j] = Y_src[i * block_size + j];
                }
            }

            memcpy(ws->T, T_src, block_size * block_size * sizeof(float));

            int ret = apply_block_reflector_clean(
                Q, Y_full, ws->T,
                m, m, block_size,
                ws->Z, ws->Z_temp, ws->YT);

            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

//==============================================================================
// WRAPPER FUNCTIONS (keep existing signatures)
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
        min_dim * sizeof(float) +   // tau
        m_max * sizeof(float) * 2 + // tmp, work

        // WY representation buffers
        ws->ib * ws->ib * sizeof(float) +            // T
        (size_t)m_max * ws->ib * sizeof(float) * 2 + // Y, YT

        // GEMM working buffers (FIXED)
        (size_t)ws->ib * n_big * sizeof(float) * 2 + // Z, Z_temp

        // Copy/packing buffer
        (size_t)m_max * n_max * sizeof(float) + // Cpack

        // Column pivoting buffers
        n_max * sizeof(float) * 2; // vn1, vn2

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
