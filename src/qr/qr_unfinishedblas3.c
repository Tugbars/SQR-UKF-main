/**
 * @file qr_blocked.c (ROW-MAJOR CORRECTED VERSION)
 * @brief GEMM-Accelerated Blocked QR with Row-Major Layout
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

    return 0;
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

static void naive_gemm_strided(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    uint16_t m, uint16_t k, uint16_t n,
    uint16_t ldc, uint16_t lda, uint16_t ldb,
    float alpha, float beta)
{
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

static inline bool needs_safe_householder(const float *x, uint16_t len, float *max_abs)
{
    const float OVERFLOW_THRESHOLD = 1e19f;
    const float UNDERFLOW_THRESHOLD = 1e-19f;

    if (has_nan_or_inf(x, len))
        return true;

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

        float vals[8];
        _mm256_storeu_ps(vals, max_vec);
        for (int i = 0; i < 8; ++i)
            if (vals[i] > max_val)
                max_val = vals[i];
    }
#endif

    for (uint16_t i = (len / 8) * 8; i < len; ++i)
    {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_val)
            max_val = abs_val;
    }

    *max_abs = max_val;

    if (max_val > OVERFLOW_THRESHOLD)
        return true;

    if (max_val > 0.0f && max_val < UNDERFLOW_THRESHOLD)
        return true;

    return false;
}

//==============================================================================
// HOUSEHOLDER REFLECTION PRIMITIVES
//==============================================================================

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

    float tail_max;
    bool need_safe = needs_safe_householder(&x[1], m - 1, &tail_max);

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

    if (need_safe)
    {
        double alpha = (double)x[0];

        double scale = 0.0;
        double sumsq = 1.0;

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

        if (scale == 0.0 || !isfinite(sumsq))
        {
            *tau = 0.0f;
            if (beta)
                *beta = x[0];
            x[0] = 1.0f;
            return;
        }

        double norm = scale * sqrt(sumsq);
        double beta_val = -copysign(norm, alpha);

        *tau = (float)((beta_val - alpha) / beta_val);

        if (beta)
            *beta = (float)beta_val;

        double scale_factor = 1.0 / (alpha - beta_val);

        for (uint16_t i = 1; i < m; ++i)
            x[i] *= (float)scale_factor;

        x[0] = 1.0f;
        return;
    }

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
// PANEL FACTORIZATION (ROW-MAJOR CORRECTED)
//==============================================================================

/**
 * @brief Panel factorization with ROW-MAJOR layout
 * 
 * All indexing is now consistent with row-major storage:
 * - panel[i * lda + j] = element at row i, column j
 * - lda = full row stride (typically = n)
 * 
 * This matches the layout of your trailing matrix updates and AVX2 kernels.
 */
static void panel_factor_clean(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,  // Row stride (= n for full matrix)
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

        // ✅ ROW-MAJOR: Extract column j starting from row j
        // panel[row * lda + col]
        for (uint16_t i = 0; i < col_len; ++i)
            work[i] = panel[(j + i) * lda + j];

        float beta;
        compute_householder_robust(work, col_len, &tau[j], &beta);

        // Write back: panel[row j, column j] = beta
        panel[j * lda + j] = beta;

        for (uint16_t i = 1; i < col_len; ++i)
            panel[(j + i) * lda + j] = work[i];

        // Store in Y (row-major)
        for (uint16_t i = 0; i < j; ++i)
            Y[i * ldy + j] = 0.0f;
        for (uint16_t i = 0; i < col_len; ++i)
            Y[(j + i) * ldy + j] = work[i];

        // Apply to trailing columns
        if (j + 1 < ib)
        {
            // Trailing submatrix starts at row j, column (j+1)
            float *restrict trailing = &panel[j * lda + (j + 1)];
            apply_householder_clean(trailing, col_len, ib - j - 1,
                                    lda, work, tau[j]);
        }
    }
}

/**
 * @brief Recursive panel factorization with ROW-MAJOR layout
 */
static void panel_factor_recursive(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    float *restrict T_workspace,
    float *restrict Z_workspace,
    float *restrict Y_temp,
    float *restrict work,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,  // Row stride
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

    float *Y_left = (float *)malloc(m * ib1 * sizeof(float));
    float *Y_right = (float *)malloc((m - ib1) * ib2 * sizeof(float));

    if (!Y_left || !Y_right)
    {
        free(Y_left);
        free(Y_right);
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, work);
        return;
    }

    // Factor left panel
    panel_factor_recursive(
        panel, Y_left, tau,
        T_workspace, Z_workspace, Y_temp,
        work,
        m, ib1, lda, ib1, threshold);

    // Copy Y_left to output Y
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < ib1; ++j)
            Y[i * ldy + j] = Y_left[i * ib1 + j];

    // Build T matrix for left panel
    build_T_matrix(Y_left, tau, T_workspace, m, ib1, ib1);

    // Apply block reflector to right columns
    float *right_cols = &panel[ib1];  // Start at column ib1, row 0

    // Copy right columns to contiguous buffer
    float *C_contig = Z_workspace;
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            C_contig[i * ib2 + j] = right_cols[i * lda + j];

    // Transpose Y_left for GEMM
    float *YT = Y_temp;
    for (uint16_t i = 0; i < ib1; ++i)
        for (uint16_t j = 0; j < m; ++j)
            YT[i * m + j] = Y_left[j * ib1 + i];

    // Z1 = Y^T * C
    float *Z1 = Z_workspace + m * ib2;
    GEMM_CALL(Z1, YT, C_contig, ib1, m, ib2, 1.0f, 0.0f);

    // Z2 = T * Z1
    float *Z2 = Z1 + ib1 * ib2;
    GEMM_CALL(Z2, T_workspace, Z1, ib1, ib1, ib2, 1.0f, 0.0f);

    // C = C - Y * Z2
    GEMM_CALL(C_contig, Y_left, Z2, m, ib1, ib2, -1.0f, 1.0f);

    // Copy back to strided layout
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            right_cols[i * lda + j] = C_contig[i * ib2 + j];

    // Factor right panel (starting at row ib1, column ib1)
    float *right_panel = &panel[ib1 * lda + ib1];
    panel_factor_recursive(
        right_panel, Y_right, &tau[ib1],
        T_workspace, Z_workspace, Y_temp,
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

    panel_factor_recursive(
        panel, Y, tau,
        workspace->panel_T_temp,
        workspace->panel_Z_temp,
        workspace->panel_Y_temp,
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
    float *restrict YT)
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

    float *Y_contig = YT; // Reuse YT buffer

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
    for (uint16_t i = 0; i < ib; ++i)
        for (uint16_t j = 0; j < m; ++j)
            YT[i * m + j] = Y[j * ldy + i];

    naive_gemm_strided(Z, YT, C,
                       ib, m, n,
                       n, m, ldc,
                       1.0f, 0.0f);

    naive_gemm_strided(Z_temp, T, Z,
                       ib, ib, n,
                       n, ib, n,
                       1.0f, 0.0f);

    naive_gemm_strided(C, Y, Z_temp,
                       m, ib, n,
                       ldc, ldy, n,
                       -1.0f, 1.0f);
    return 0;
}

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

static int apply_trailing_update(
    float *A,
    const float *Y,
    const float *T,
    uint16_t k,
    uint16_t m,
    uint16_t n,
    uint16_t block_size,
    uint16_t rows_below,
    uint16_t cols_right,
    qr_workspace *ws)
{
    if (cols_right == 0)
        return 0;
    
    float *trailing_matrix = &A[k * n + k + block_size];
    
    // Copy Y to contiguous buffer
    float *Y_contig = ws->panel_Y_temp;
    for (uint16_t i = 0; i < rows_below; ++i)
        for (uint16_t j = 0; j < block_size; ++j)
            Y_contig[i * block_size + j] = Y[i * ws->ib + j];
    
    // Copy trailing matrix to contiguous buffer
    float *C_contig = ws->C_temp;
    for (uint16_t i = 0; i < rows_below; ++i)
        for (uint16_t j = 0; j < cols_right; ++j)
            C_contig[i * cols_right + j] = trailing_matrix[i * n + j];
    
    // Apply block reflector: C = C - Y*(T*(Y^T*C))
    
    // Step 1: Z = Y^T * C
    float *YT = ws->YT;
    for (uint16_t i = 0; i < block_size; ++i)
        for (uint16_t j = 0; j < rows_below; ++j)
            YT[i * rows_below + j] = Y_contig[j * block_size + i];
    
    int ret = GEMM_CALL(ws->Z, YT, C_contig,
                        block_size, rows_below, cols_right, 1.0f, 0.0f);
    if (ret != 0)
        return ret;
    
    // Step 2: Z_temp = T * Z
    ret = GEMM_CALL(ws->Z_temp, T, ws->Z,
                    block_size, block_size, cols_right, 1.0f, 0.0f);
    if (ret != 0)
        return ret;
    
    // Step 3: C = C - Y * Z_temp
    ret = GEMM_CALL(C_contig, Y_contig, ws->Z_temp,
                    rows_below, block_size, cols_right, -1.0f, 1.0f);
    if (ret != 0)
        return ret;
    
    // Copy back to strided layout
    for (uint16_t i = 0; i < rows_below; ++i)
        for (uint16_t j = 0; j < cols_right; ++j)
            trailing_matrix[i * n + j] = C_contig[i * cols_right + j];
    
    // Debug output
    if (k == 0 && cols_right > 0) {
        printf("DEBUG Block 0: After trailing update\n");
        printf("  A[0,4:8] = ");
        for (int j = 4; j < 8; j++) printf("%.4f ", A[0*n + j]);
        printf("\n  A[4,4:8] = ");
        for (int j = 4; j < 8; j++) printf("%.4f ", A[4*n + j]);
        printf("\n");
    }
    
    return 0;
}

static void extract_R(const float *A, float *R, uint16_t m, uint16_t n)
{
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? A[i * n + j] : 0.0f;
}

static int form_Q(
    float *Q,
    uint16_t m,
    uint16_t kmax,
    uint16_t block_count,
    qr_workspace *ws)
{
    memset(Q, 0, (size_t)m * m * sizeof(float));
    for (uint16_t i = 0; i < m; ++i)
        Q[i * m + i] = 1.0f;
    
    printf("DEBUG: Starting Q formation\n");
    
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
        
        printf("DEBUG: After block %d Q formation\n", blk);
        printf("  Q[0,0:4] = ");
        for (int j = 0; j < 4; j++) printf("%.4f ", Q[0*m + j]);
        printf("\n  Q[0,4:8] = ");
        for (int j = 4; j < 8; j++) printf("%.4f ", Q[0*m + j]);
        printf("\n");
        
        if (ret != 0)
            return ret;
    }
    
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
    
    // ========== FACTORIZATION LOOP ==========
    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;
        uint16_t cols_right = (n > k + block_size) ? (n - k - block_size) : 0;
        
        if (k == 4) {
            printf("DEBUG Block 1: Before panel factor\n");
            printf("  A[4,4:8] = ");
            for (int j = 4; j < 8; j++) printf("%.4f ", A[4*n + j]);
            printf("\n");
        }
        
        // 1. Factor current panel (starting at row k, column k)
        panel_factor_optimized(&A[k * n + k], ws->Y, &ws->tau[k],
                               rows_below, block_size, n, ws->ib, ws);
        
        // 2. Build T matrix
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size, ws->ib);
        
        // 3. Store Y and T for Q formation
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
        
        // 4. Apply block reflector to trailing matrix
        int ret = apply_trailing_update(A, ws->Y, ws->T,
                                        k, m, n, block_size, rows_below, cols_right,
                                        ws);
        if (ret != 0)
            return ret;
        
        block_count++;
    }
    
    // ========== EXTRACT R ==========
    extract_R(A, R, m, n);
    
    printf("DEBUG: R[4,4] = %.6f (from A[%d])\n", R[4*n + 4], 4*n + 4);
    printf("DEBUG: A[4,4] after factorization = %.6f\n", A[4*n + 4]);
    
    // ========== FORM Q ==========
    if (!only_R && Q)
    {
        int ret = form_Q(Q, m, kmax, block_count, ws);
        if (ret != 0)
            return ret;
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
// WORKSPACE ALLOCATION
//==============================================================================

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

    ws->Y_block_stride = (size_t)m_max * ws->ib;
    ws->T_block_stride = (size_t)ws->ib * ws->ib;

    const uint16_t n_big = (m_max > n_max) ? m_max : n_max;

    ws->tau = (float *)malloc(min_dim * sizeof(float));
    ws->tmp = (float *)malloc(m_max * sizeof(float));
    ws->work = (float *)malloc(m_max * sizeof(float));
    ws->T = (float *)gemm_aligned_alloc(32, ws->ib * ws->ib * sizeof(float));

    ws->Y = (float *)gemm_aligned_alloc(32, (size_t)m_max * ws->ib * sizeof(float));

    ws->YT = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * m_max * sizeof(float));
    ws->Z = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_big * sizeof(float));
    ws->Z_temp = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_big * sizeof(float));
    ws->Cpack = (float *)gemm_aligned_alloc(32, (size_t)m_max * n_max * sizeof(float));
    ws->vn1 = (float *)malloc(n_max * sizeof(float));
    ws->vn2 = (float *)malloc(n_max * sizeof(float));

    ws->panel_Y_temp = (float *)gemm_aligned_alloc(32,
                                                   (size_t)m_max * ws->ib * sizeof(float));

    ws->panel_T_temp = (float *)gemm_aligned_alloc(32,
                                                   ws->ib * ws->ib * sizeof(float));

    size_t panel_z_size = m_max * (ws->ib / 2) + 2 * (ws->ib / 2) * (ws->ib / 2);
    ws->panel_Z_temp = (float *)gemm_aligned_alloc(32, panel_z_size * sizeof(float));

    ws->C_temp = (float *)gemm_aligned_alloc(32, (size_t)m_max * n_max * sizeof(float));

    size_t bytes =
        min_dim * sizeof(float) +
        m_max * sizeof(float) * 2 +
        ws->ib * ws->ib * sizeof(float) +
        (size_t)m_max * ws->ib * sizeof(float) * 2 +
        (size_t)ws->ib * n_big * sizeof(float) * 2 +
        (size_t)m_max * n_max * sizeof(float) +
        n_max * sizeof(float) * 2;

    bytes += (size_t)m_max * ws->ib * sizeof(float);
    bytes += ws->ib * ws->ib * sizeof(float) * 2;
    bytes += (size_t)m_max * n_max * sizeof(float);

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

    if (!ws->tau || !ws->tmp || !ws->work || !ws->T || !ws->Cpack || !ws->C_temp ||
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
    gemm_aligned_free(ws->C_temp);

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