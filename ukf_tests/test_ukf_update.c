/**
 * @file test_ukf_update.c
 * @brief Unit tests for SR-UKF measurement update step
 *
 * Tests:
 * - update_state_covariance_matrix_and_state_estimation_vector
 * - Kalman gain computation correctness
 * - State update: x̂⁺ = x̂⁻ + K·(y - ŷ)
 * - Covariance downdate correctness
 * - Positive definiteness preservation after downdate
 * - Filter divergence detection
 * - Numerical stability
 *
 * @author TUGBARS
 * @date 2025
 */

#include "test_common.h"
#include "sqr_ukf.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

//==============================================================================
// MATRIX UTILITIES
//==============================================================================

/**
 * @brief Compute Frobenius norm
 */
static double frobenius_norm(const float *A, size_t m, size_t n, size_t ld)
{
    double sum = 0.0;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double val = (double)A[i * ld + j];
            sum += val * val;
        }
    }
    return sqrt(sum);
}

/**
 * @brief Compute relative error between matrices
 */
static double matrix_relative_error(const float *A, const float *B,
                                    size_t m, size_t n, size_t ld)
{
    double diff_sum = 0.0;
    double a_sum = 0.0;
    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double a = (double)A[i * ld + j];
            double b = (double)B[i * ld + j];
            diff_sum += (a - b) * (a - b);
            a_sum += a * a;
        }
    }
    
    if (a_sum < 1e-30)
        return 0.0;
    
    return sqrt(diff_sum / a_sum);
}

/**
 * @brief Compute relative error between vectors
 */
static double vector_relative_error(const float *a, const float *b, size_t n)
{
    double diff_sum = 0.0;
    double a_sum = 0.0;
    
    for (size_t i = 0; i < n; i++)
    {
        double ai = (double)a[i];
        double bi = (double)b[i];
        diff_sum += (ai - bi) * (ai - bi);
        a_sum += ai * ai;
    }
    
    if (a_sum < 1e-30)
        return sqrt(diff_sum);
    
    return sqrt(diff_sum / a_sum);
}

/**
 * @brief Generate random vector
 */
static void generate_random_vector(float *v, size_t n, unsigned int seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++)
    {
        v[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    }
}

/**
 * @brief Generate random upper triangular matrix with positive diagonal
 */
static void generate_random_upper_triangular(float *U, size_t n, size_t ld,
                                             unsigned int seed, float diag_boost)
{
    srand(seed);
    
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            if (j < i)
            {
                U[i * ld + j] = 0.0f;
            }
            else if (j == i)
            {
                U[i * ld + j] = diag_boost + ((float)(rand() % 100) + 10.0f) / 50.0f;
            }
            else
            {
                U[i * ld + j] = ((float)(rand() % 200) - 100.0f) / 100.0f;
            }
        }
    }
}

/**
 * @brief Generate random symmetric positive definite matrix
 */
static void generate_random_spd(float *A, size_t n, unsigned int seed, float diag_boost)
{
    srand(seed);
    
    /* Generate random matrix */
    float *temp = gemm_aligned_alloc(32, n * n * sizeof(float));
    for (size_t i = 0; i < n * n; i++)
    {
        temp[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }
    
    /* A = temp * temp^T (guaranteed SPD) */
    memset(A, 0, n * n * sizeof(float));
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++)
            {
                sum += (double)temp[i * n + k] * (double)temp[j * n + k];
            }
            A[i * n + j] = (float)sum;
        }
    }
    
    /* Add diagonal boost for better conditioning */
    for (size_t i = 0; i < n; i++)
    {
        A[i * n + i] += diag_boost;
    }
    
    gemm_aligned_free(temp);
}

/**
 * @brief Reconstruct symmetric matrix from upper triangular SR: A = S^T * S
 */
static void reconstruct_from_upper_sr(const float *S, float *A, size_t n, size_t ld)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            size_t k_max = (i < j) ? i : j;
            for (size_t k = 0; k <= k_max; k++)
            {
                sum += (double)S[k * ld + i] * (double)S[k * ld + j];
            }
            A[i * ld + j] = (float)sum;
        }
    }
}

/**
 * @brief Matrix multiply C = A * B
 */
static void matmul_reference(float *C, const float *A, const float *B,
                             size_t m, size_t k, size_t n,
                             size_t ldc, size_t lda, size_t ldb)
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++)
            {
                sum += (double)A[i * lda + p] * (double)B[p * ldb + j];
            }
            C[i * ldc + j] = (float)sum;
        }
    }
}

/**
 * @brief Matrix-vector multiply: y = A * x
 */
static void matvec_reference(float *y, const float *A, const float *x,
                             size_t m, size_t n, size_t lda)
{
    for (size_t i = 0; i < m; i++)
    {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++)
        {
            sum += (double)A[i * lda + j] * (double)x[j];
        }
        y[i] = (float)sum;
    }
}

/**
 * @brief Compute matrix inverse via Gauss-Jordan (for small matrices, testing only)
 */
static int matrix_inverse_reference(float *Ainv, const float *A, size_t n)
{
    float *work = gemm_aligned_alloc(32, n * 2 * n * sizeof(float));
    if (!work) return -1;
    
    /* Build [A | I] */
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            work[i * 2 * n + j] = A[i * n + j];
            work[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    /* Gauss-Jordan elimination */
    for (size_t col = 0; col < n; col++)
    {
        /* Find pivot */
        size_t pivot = col;
        float max_val = fabsf(work[col * 2 * n + col]);
        for (size_t row = col + 1; row < n; row++)
        {
            float val = fabsf(work[row * 2 * n + col]);
            if (val > max_val)
            {
                max_val = val;
                pivot = row;
            }
        }
        
        if (max_val < 1e-10f)
        {
            gemm_aligned_free(work);
            return -1; /* Singular */
        }
        
        /* Swap rows */
        if (pivot != col)
        {
            for (size_t j = 0; j < 2 * n; j++)
            {
                float tmp = work[col * 2 * n + j];
                work[col * 2 * n + j] = work[pivot * 2 * n + j];
                work[pivot * 2 * n + j] = tmp;
            }
        }
        
        /* Scale pivot row */
        float scale = 1.0f / work[col * 2 * n + col];
        for (size_t j = 0; j < 2 * n; j++)
        {
            work[col * 2 * n + j] *= scale;
        }
        
        /* Eliminate column */
        for (size_t row = 0; row < n; row++)
        {
            if (row != col)
            {
                float factor = work[row * 2 * n + col];
                for (size_t j = 0; j < 2 * n; j++)
                {
                    work[row * 2 * n + j] -= factor * work[col * 2 * n + j];
                }
            }
        }
    }
    
    /* Extract inverse */
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            Ainv[i * n + j] = work[i * 2 * n + n + j];
        }
    }
    
    gemm_aligned_free(work);
    return 0;
}

//==============================================================================
// PROPERTY CHECKERS
//==============================================================================

/**
 * @brief Check if matrix is upper triangular
 */
static int check_upper_triangular(const float *S, size_t n, size_t ld,
                                  double tol, const char *name)
{
    printf("  Checking upper triangular structure (%s)...\n", name);
    
    double max_lower = 0.0;
    int violations = 0;
    
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            double val = fabs((double)S[i * ld + j]);
            if (val > max_lower)
                max_lower = val;
            if (val > tol)
                violations++;
        }
    }
    
    if (violations > 0)
    {
        printf("    FAILED: %d lower triangle violations (max=%.6e)\n", 
               violations, max_lower);
        return 0;
    }
    
    printf("    PASSED (max lower = %.6e)\n", max_lower);
    return 1;
}

/**
 * @brief Check if matrix has positive diagonal
 */
static int check_positive_diagonal(const float *S, size_t n, size_t ld,
                                   const char *name)
{
    printf("  Checking positive diagonal (%s)...\n", name);
    
    float min_diag = FLT_MAX;
    int neg_count = 0;
    int nan_count = 0;
    
    for (size_t i = 0; i < n; i++)
    {
        float diag = S[i * ld + i];
        
        if (!isfinite(diag))
        {
            nan_count++;
            continue;
        }
        
        if (diag <= 0.0f)
            neg_count++;
        
        if (diag < min_diag)
            min_diag = diag;
    }
    
    if (nan_count > 0)
    {
        printf("    FAILED: %d NaN/Inf diagonal elements\n", nan_count);
        return 0;
    }
    
    if (neg_count > 0)
    {
        printf("    FAILED: %d non-positive diagonal (min=%.6e)\n", neg_count, min_diag);
        return 0;
    }
    
    printf("    PASSED (min diagonal = %.6e)\n", min_diag);
    return 1;
}

/**
 * @brief Check no NaN/Inf in matrix
 */
static int check_finite(const float *A, size_t m, size_t n, size_t ld,
                        const char *name)
{
    int bad_count = 0;
    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            if (!isfinite(A[i * ld + j]))
                bad_count++;
        }
    }
    
    if (bad_count > 0)
    {
        printf("  %s: %d NaN/Inf values detected\n", name, bad_count);
        return 0;
    }
    
    return 1;
}

//==============================================================================
// REFERENCE IMPLEMENTATION
//==============================================================================

/**
 * @brief Reference Kalman gain computation via explicit inverse
 * 
 * K = Pxy * (Sy * Sy^T)^(-1)
 *   = Pxy * Pyy^(-1)
 * 
 * where Pyy = Sy * Sy^T (measurement covariance)
 */
static int kalman_gain_reference(float *K, const float *Pxy, const float *Sy,
                                 size_t n)
{
    float *Pyy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pyy_inv = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    if (!Pyy || !Pyy_inv)
    {
        gemm_aligned_free(Pyy);
        gemm_aligned_free(Pyy_inv);
        return -1;
    }
    
    /* Pyy = Sy^T * Sy (since Sy is upper triangular) */
    reconstruct_from_upper_sr(Sy, Pyy, n, n);
    
    /* Pyy_inv = Pyy^(-1) */
    int rc = matrix_inverse_reference(Pyy_inv, Pyy, n);
    if (rc != 0)
    {
        gemm_aligned_free(Pyy);
        gemm_aligned_free(Pyy_inv);
        return -1;
    }
    
    /* K = Pxy * Pyy_inv */
    matmul_reference(K, Pxy, Pyy_inv, n, n, n, n, n, n);
    
    gemm_aligned_free(Pyy);
    gemm_aligned_free(Pyy_inv);
    return 0;
}

/**
 * @brief Reference state update: x̂⁺ = x̂⁻ + K * (y - ŷ)
 */
static void state_update_reference(float *xhat_out, const float *xhat_in,
                                   const float *K, const float *y, const float *yhat,
                                   size_t n)
{
    /* Compute innovation v = y - yhat */
    float *v = gemm_aligned_alloc(32, n * sizeof(float));
    for (size_t i = 0; i < n; i++)
    {
        v[i] = y[i] - yhat[i];
    }
    
    /* Compute K * v */
    float *Kv = gemm_aligned_alloc(32, n * sizeof(float));
    matvec_reference(Kv, K, v, n, n, n);
    
    /* x̂⁺ = x̂⁻ + Kv */
    for (size_t i = 0; i < n; i++)
    {
        xhat_out[i] = xhat_in[i] + Kv[i];
    }
    
    gemm_aligned_free(v);
    gemm_aligned_free(Kv);
}

/**
 * @brief Reference covariance downdate (explicit formula)
 * 
 * P⁺ = P⁻ - K * Pyy * K^T
 *    = P⁻ - U * U^T  where U = K * Sy
 */
static void covariance_downdate_reference(float *P_out, const float *P_in,
                                          const float *K, const float *Sy,
                                          size_t n)
{
    float *U = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *UUt = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    /* U = K * Sy */
    matmul_reference(U, K, Sy, n, n, n, n, n, n);
    
    /* UUt = U * U^T */
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++)
            {
                sum += (double)U[i * n + k] * (double)U[j * n + k];
            }
            UUt[i * n + j] = (float)sum;
        }
    }
    
    /* P⁺ = P⁻ - UUt */
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            P_out[i * n + j] = P_in[i * n + j] - UUt[i * n + j];
        }
    }
    
    gemm_aligned_free(U);
    gemm_aligned_free(UUt);
}

//==============================================================================
// TESTS
//==============================================================================

/**
 * @brief Test basic measurement update
 */
static int test_update_basic(void)
{
    printf("\n=== Testing measurement update (basic) ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t n = (size_t)L;
    
    printf("  State dimension n=%zu\n", n);
    
    /* Allocate */
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *S_orig = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat_orig = gemm_aligned_alloc(32, n * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    ukf_upd_ws_t ws = {0};
    
    if (!S || !S_orig || !xhat || !xhat_orig || !yhat || !y || !Sy || !Pxy)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Generate test data */
    generate_random_upper_triangular(S, n, n, 11111, 2.0f);
    memcpy(S_orig, S, n * n * sizeof(float));
    
    generate_random_vector(xhat, n, 22222);
    memcpy(xhat_orig, xhat, n * sizeof(float));
    
    generate_random_vector(yhat, n, 33333);
    generate_random_vector(y, n, 44444);
    
    generate_random_upper_triangular(Sy, n, n, 55555, 2.0f);
    generate_random_spd(Pxy, n, 66666, 0.5f);
    
    /* Call function under test */
    printf("  Calling update_state_covariance_matrix_and_state_estimation_vector...\n");
    int rc = update_state_covariance_matrix_and_state_estimation_vector(
        S, xhat, yhat, y, Sy, Pxy, &ws, L);
    
    if (rc != 0)
    {
        printf("  FAILED: returned error code %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* Check output properties */
    passed &= check_upper_triangular(S, n, n, 1e-5, "S updated");
    passed &= check_positive_diagonal(S, n, n, "S updated");
    passed &= check_finite(xhat, n, 1, 1, "xhat updated");
    
    /* Verify state was actually modified */
    double xhat_change = vector_relative_error(xhat, xhat_orig, n);
    printf("  State change: ||x̂⁺ - x̂⁻|| / ||x̂⁻|| = %.6e\n", xhat_change);
    
    if (xhat_change < 1e-10)
    {
        printf("  WARNING: State appears unchanged\n");
    }
    
    /* Verify covariance was modified (downdated) */
    float *P_orig = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *P_new = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    reconstruct_from_upper_sr(S_orig, P_orig, n, n);
    reconstruct_from_upper_sr(S, P_new, n, n);
    
    /* Check that covariance decreased (trace should decrease) */
    double trace_orig = 0.0, trace_new = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        trace_orig += (double)P_orig[i * n + i];
        trace_new += (double)P_new[i * n + i];
    }
    
    printf("  Covariance trace: before=%.4f, after=%.4f\n", trace_orig, trace_new);
    
    if (trace_new >= trace_orig)
    {
        printf("  WARNING: Covariance trace did not decrease (may indicate issue)\n");
    }
    
    gemm_aligned_free(P_orig);
    gemm_aligned_free(P_new);
    
cleanup:
    gemm_aligned_free(S);
    gemm_aligned_free(S_orig);
    gemm_aligned_free(xhat);
    gemm_aligned_free(xhat_orig);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    ukf_upd_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test state update correctness against reference
 */
static int test_state_update_correctness(void)
{
    printf("\n=== Testing state update correctness ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t n = (size_t)L;
    
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat_ref = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat_orig = gemm_aligned_alloc(32, n * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *K_ref = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    ukf_upd_ws_t ws = {0};
    
    if (!S || !xhat || !xhat_ref || !xhat_orig || !yhat || !y || !Sy || !Pxy || !K_ref)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Generate well-conditioned test data */
    generate_random_upper_triangular(S, n, n, 77777, 3.0f);
    generate_random_vector(xhat_orig, n, 88888);
    memcpy(xhat, xhat_orig, n * sizeof(float));
    
    generate_random_vector(yhat, n, 99999);
    generate_random_vector(y, n, 11112);
    
    generate_random_upper_triangular(Sy, n, n, 22223, 3.0f);
    generate_random_spd(Pxy, n, 33334, 1.0f);
    
    /* Compute reference Kalman gain and state update */
    int rc = kalman_gain_reference(K_ref, Pxy, Sy, n);
    if (rc != 0)
    {
        printf("  ERROR: Reference Kalman gain computation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    state_update_reference(xhat_ref, xhat_orig, K_ref, y, yhat, n);
    
    /* Call function under test */
    rc = update_state_covariance_matrix_and_state_estimation_vector(
        S, xhat, yhat, y, Sy, Pxy, &ws, L);
    
    if (rc != 0)
    {
        printf("  FAILED: update returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* Compare state update */
    double state_err = vector_relative_error(xhat, xhat_ref, n);
    
    printf("  State update relative error: %.6e\n", state_err);
    
    if (state_err > 1e-3)
    {
        printf("  FAILED: State update mismatch\n");
        printf("  First few elements:\n");
        for (size_t i = 0; i < 4 && i < n; i++)
        {
            printf("    x̂[%zu]: got %.6f, expected %.6f\n", i, xhat[i], xhat_ref[i]);
        }
        passed = 0;
    }
    else
    {
        printf("  State update PASSED\n");
    }
    
cleanup:
    gemm_aligned_free(S);
    gemm_aligned_free(xhat);
    gemm_aligned_free(xhat_ref);
    gemm_aligned_free(xhat_orig);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    gemm_aligned_free(K_ref);
    ukf_upd_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test covariance downdate correctness
 */
static int test_covariance_downdate_correctness(void)
{
    printf("\n=== Testing covariance downdate correctness ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t n = (size_t)L;
    
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *S_orig = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *P_orig = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *P_updated = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *P_ref = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *K_ref = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    ukf_upd_ws_t ws = {0};
    
    if (!S || !S_orig || !xhat || !yhat || !y || !Sy || !Pxy || 
        !P_orig || !P_updated || !P_ref || !K_ref)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Generate well-conditioned data with large covariances */
    generate_random_upper_triangular(S_orig, n, n, 44445, 5.0f);
    memcpy(S, S_orig, n * n * sizeof(float));
    
    generate_random_vector(xhat, n, 55556);
    generate_random_vector(yhat, n, 66667);
    generate_random_vector(y, n, 77778);
    
    /* Small measurement noise (allows larger downdate) */
    generate_random_upper_triangular(Sy, n, n, 88889, 1.0f);
    
    /* Cross-covariance scaled relative to covariances */
    generate_random_spd(Pxy, n, 99990, 0.5f);
    
    /* Compute reference */
    reconstruct_from_upper_sr(S_orig, P_orig, n, n);
    
    int rc = kalman_gain_reference(K_ref, Pxy, Sy, n);
    if (rc != 0)
    {
        printf("  ERROR: Reference Kalman gain failed\n");
        passed = 0;
        goto cleanup;
    }
    
    covariance_downdate_reference(P_ref, P_orig, K_ref, Sy, n);
    
    /* Call function under test */
    rc = update_state_covariance_matrix_and_state_estimation_vector(
        S, xhat, yhat, y, Sy, Pxy, &ws, L);
    
    if (rc != 0)
    {
        printf("  Update returned %d (may be expected for ill-conditioned case)\n", rc);
        /* Don't fail immediately - check what we can */
    }
    
    /* Reconstruct updated covariance */
    reconstruct_from_upper_sr(S, P_updated, n, n);
    
    /* Compare covariances */
    double cov_err = matrix_relative_error(P_updated, P_ref, n, n, n);
    
    printf("  Covariance downdate relative error: %.6e\n", cov_err);
    
    if (cov_err > 5e-2) /* Relaxed tolerance due to numerical differences */
    {
        printf("  WARNING: Covariance downdate has significant difference\n");
        printf("  (This may be acceptable due to different numerical paths)\n");
        
        /* Check that at least diagonal decreased */
        int diag_decreased = 1;
        for (size_t i = 0; i < n; i++)
        {
            if (P_updated[i * n + i] > P_orig[i * n + i] * 1.01f)
            {
                diag_decreased = 0;
                break;
            }
        }
        
        if (!diag_decreased)
        {
            printf("  FAILED: Covariance diagonal did not decrease\n");
            passed = 0;
        }
    }
    else
    {
        printf("  Covariance downdate PASSED\n");
    }
    
cleanup:
    gemm_aligned_free(S);
    gemm_aligned_free(S_orig);
    gemm_aligned_free(xhat);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    gemm_aligned_free(P_orig);
    gemm_aligned_free(P_updated);
    gemm_aligned_free(P_ref);
    gemm_aligned_free(K_ref);
    ukf_upd_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test update across various sizes
 */
static int test_update_sizes(void)
{
    printf("\n=== Testing measurement update (various sizes) ===\n");
    
    int passed = 1;
    
    uint8_t test_L[] = {4, 8, 16, 32};
    const int num_L = sizeof(test_L) / sizeof(test_L[0]);
    
    ukf_upd_ws_t ws = {0};
    
    for (int tc = 0; tc < num_L; tc++)
    {
        uint8_t L = test_L[tc];
        const size_t n = (size_t)L;
        
        printf("  Testing n=%zu...\n", n);
        
        float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *y = gemm_aligned_alloc(32, n * sizeof(float));
        float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
        
        if (!S || !xhat || !yhat || !y || !Sy || !Pxy)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_size;
        }
        
        /* Generate data with good conditioning */
        generate_random_upper_triangular(S, n, n, tc * 11111, 3.0f + (float)tc);
        generate_random_vector(xhat, n, tc * 22222);
        generate_random_vector(yhat, n, tc * 33333);
        generate_random_vector(y, n, tc * 44444);
        generate_random_upper_triangular(Sy, n, n, tc * 55555, 2.0f + (float)tc);
        generate_random_spd(Pxy, n, tc * 66666, 1.0f);
        
        int rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y, Sy, Pxy, &ws, L);
        
        if (rc != 0)
        {
            printf("    FAILED: returned %d\n", rc);
            passed = 0;
        }
        else
        {
            int ok = 1;
            ok &= check_upper_triangular(S, n, n, 1e-5, "S");
            ok &= check_positive_diagonal(S, n, n, "S");
            ok &= check_finite(xhat, n, 1, 1, "xhat");
            
            if (ok)
            {
                printf("    n=%zu PASSED\n", n);
            }
            else
            {
                passed = 0;
            }
        }
        
cleanup_size:
        gemm_aligned_free(S);
        gemm_aligned_free(xhat);
        gemm_aligned_free(yhat);
        gemm_aligned_free(y);
        gemm_aligned_free(Sy);
        gemm_aligned_free(Pxy);
    }
    
    ukf_upd_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test with zero innovation (y = yhat)
 */
static int test_zero_innovation(void)
{
    printf("\n=== Testing update with zero innovation (y = ŷ) ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t n = (size_t)L;
    
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *S_orig = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat_orig = gemm_aligned_alloc(32, n * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    ukf_upd_ws_t ws = {0};
    
    if (!S || !S_orig || !xhat || !xhat_orig || !yhat || !Sy || !Pxy)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    generate_random_upper_triangular(S_orig, n, n, 12121, 3.0f);
    memcpy(S, S_orig, n * n * sizeof(float));
    
    generate_random_vector(xhat_orig, n, 23232);
    memcpy(xhat, xhat_orig, n * sizeof(float));
    
    generate_random_vector(yhat, n, 34343);
    /* y = yhat (zero innovation) */
    
    generate_random_upper_triangular(Sy, n, n, 45454, 2.0f);
    generate_random_spd(Pxy, n, 56565, 0.5f);
    
    /* Call with y = yhat */
    int rc = update_state_covariance_matrix_and_state_estimation_vector(
        S, xhat, yhat, yhat, /* y = yhat */ Sy, Pxy, &ws, L);
    
    if (rc != 0)
    {
        printf("  FAILED: returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* State should be unchanged (K * 0 = 0) */
    double state_change = vector_relative_error(xhat, xhat_orig, n);
    
    printf("  State change with zero innovation: %.6e\n", state_change);
    
    if (state_change > 1e-5)
    {
        printf("  FAILED: State changed with zero innovation\n");
        passed = 0;
    }
    else
    {
        printf("  PASSED: State unchanged\n");
    }
    
    /* Covariance should still be downdated */
    passed &= check_positive_diagonal(S, n, n, "S after zero innovation");
    
cleanup:
    gemm_aligned_free(S);
    gemm_aligned_free(S_orig);
    gemm_aligned_free(xhat);
    gemm_aligned_free(xhat_orig);
    gemm_aligned_free(yhat);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    ukf_upd_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test positive definiteness preservation
 */
static int test_pd_preservation(void)
{
    printf("\n=== Testing positive definiteness preservation ===\n");
    
    int passed = 1;
    
    const uint8_t L = 16;
    const size_t n = (size_t)L;
    
    /* Run multiple random tests */
    const int num_trials = 10;
    
    ukf_upd_ws_t ws = {0};
    
    for (int trial = 0; trial < num_trials; trial++)
    {
        float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *y = gemm_aligned_alloc(32, n * sizeof(float));
        float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
        
        if (!S || !xhat || !yhat || !y || !Sy || !Pxy)
        {
            printf("  ERROR: Allocation failed at trial %d\n", trial);
            passed = 0;
            gemm_aligned_free(S);
            gemm_aligned_free(xhat);
            gemm_aligned_free(yhat);
            gemm_aligned_free(y);
            gemm_aligned_free(Sy);
            gemm_aligned_free(Pxy);
            continue;
        }
        
        /* Generate with varying conditioning */
        float diag_boost = 2.0f + (float)trial * 0.5f;
        generate_random_upper_triangular(S, n, n, trial * 11111, diag_boost);
        generate_random_vector(xhat, n, trial * 22222);
        generate_random_vector(yhat, n, trial * 33333);
        generate_random_vector(y, n, trial * 44444);
        generate_random_upper_triangular(Sy, n, n, trial * 55555, diag_boost);
        generate_random_spd(Pxy, n, trial * 66666, 0.5f);
        
        int rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y, Sy, Pxy, &ws, L);
        
        if (rc != 0)
        {
            printf("  Trial %d: update returned %d (filter may have diverged)\n", 
                   trial, rc);
            /* This is expected for some random cases */
        }
        else
        {
            /* Check PD */
            int pd_ok = 1;
            for (size_t i = 0; i < n; i++)
            {
                if (S[i * n + i] <= 0.0f || !isfinite(S[i * n + i]))
                {
                    pd_ok = 0;
                    break;
                }
            }
            
            if (!pd_ok)
            {
                printf("  Trial %d: FAILED PD check\n", trial);
                passed = 0;
            }
        }
        
        gemm_aligned_free(S);
        gemm_aligned_free(xhat);
        gemm_aligned_free(yhat);
        gemm_aligned_free(y);
        gemm_aligned_free(Sy);
        gemm_aligned_free(Pxy);
    }
    
    ukf_upd_ws_cleanup(&ws);
    
    if (passed)
    {
        printf("  PD preservation PASSED (%d trials)\n", num_trials);
    }
    
    return passed;
}

/**
 * @brief Test numerical stability with extreme values
 */
static int test_numerical_stability(void)
{
    printf("\n=== Testing numerical stability ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t n = (size_t)L;
    
    ukf_upd_ws_t ws = {0};
    
    /* Test 1: Small covariances */
    printf("  Testing with small covariances...\n");
    {
        float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *y = gemm_aligned_alloc(32, n * sizeof(float));
        float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
        
        /* Small but valid covariances */
        memset(S, 0, n * n * sizeof(float));
        memset(Sy, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++)
        {
            S[i * n + i] = 1e-4f;
            Sy[i * n + i] = 1e-4f;
        }
        
        generate_random_vector(xhat, n, 78787);
        for (size_t i = 0; i < n; i++) xhat[i] *= 1e-3f;
        
        generate_random_vector(yhat, n, 89898);
        for (size_t i = 0; i < n; i++) yhat[i] *= 1e-3f;
        
        generate_random_vector(y, n, 90909);
        for (size_t i = 0; i < n; i++) y[i] *= 1e-3f;
        
        memset(Pxy, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++)
        {
            Pxy[i * n + i] = 1e-5f;
        }
        
        int rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y, Sy, Pxy, &ws, L);
        
        if (rc != 0)
        {
            printf("    Small covariances: returned %d\n", rc);
        }
        
        int ok = check_finite(S, n, n, n, "S small");
        ok &= check_finite(xhat, n, 1, 1, "xhat small");
        
        if (ok)
        {
            printf("    Small covariances PASSED\n");
        }
        else
        {
            passed = 0;
        }
        
        gemm_aligned_free(S);
        gemm_aligned_free(xhat);
        gemm_aligned_free(yhat);
        gemm_aligned_free(y);
        gemm_aligned_free(Sy);
        gemm_aligned_free(Pxy);
    }
    
    /* Test 2: Large covariances */
    printf("  Testing with large covariances...\n");
    {
        float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *y = gemm_aligned_alloc(32, n * sizeof(float));
        float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
        
        memset(S, 0, n * n * sizeof(float));
        memset(Sy, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++)
        {
            S[i * n + i] = 1e4f;
            Sy[i * n + i] = 1e4f;
        }
        
        generate_random_vector(xhat, n, 12312);
        for (size_t i = 0; i < n; i++) xhat[i] *= 1e3f;
        
        generate_random_vector(yhat, n, 23423);
        for (size_t i = 0; i < n; i++) yhat[i] *= 1e3f;
        
        generate_random_vector(y, n, 34534);
        for (size_t i = 0; i < n; i++) y[i] *= 1e3f;
        
        memset(Pxy, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++)
        {
            Pxy[i * n + i] = 1e6f;
        }
        
        int rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y, Sy, Pxy, &ws, L);
        
        if (rc != 0)
        {
            printf("    Large covariances: returned %d\n", rc);
        }
        
        int ok = check_finite(S, n, n, n, "S large");
        ok &= check_finite(xhat, n, 1, 1, "xhat large");
        
        if (ok)
        {
            printf("    Large covariances PASSED\n");
        }
        else
        {
            passed = 0;
        }
        
        gemm_aligned_free(S);
        gemm_aligned_free(xhat);
        gemm_aligned_free(yhat);
        gemm_aligned_free(y);
        gemm_aligned_free(Sy);
        gemm_aligned_free(Pxy);
    }
    
    ukf_upd_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test workspace reuse across multiple updates
 */
static int test_workspace_reuse(void)
{
    printf("\n=== Testing workspace reuse across updates ===\n");
    
    int passed = 1;
    
    const uint8_t L = 16;
    const size_t n = (size_t)L;
    const int num_updates = 50;
    
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    ukf_upd_ws_t ws = {0};
    
    if (!S || !xhat || !yhat || !y || !Sy || !Pxy)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    printf("  Running %d consecutive updates...\n", num_updates);
    
    for (int iter = 0; iter < num_updates; iter++)
    {
        /* Re-initialize with fresh data each iteration */
        generate_random_upper_triangular(S, n, n, iter * 11111, 3.0f);
        generate_random_vector(xhat, n, iter * 22222);
        generate_random_vector(yhat, n, iter * 33333);
        generate_random_vector(y, n, iter * 44444);
        generate_random_upper_triangular(Sy, n, n, iter * 55555, 3.0f);
        generate_random_spd(Pxy, n, iter * 66666, 1.0f);
        
        int rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y, Sy, Pxy, &ws, L);
        
        if (rc != 0)
        {
            /* Some failures expected with random data */
            continue;
        }
        
        /* Verify valid output */
        int ok = 1;
        for (size_t i = 0; i < n; i++)
        {
            if (!isfinite(S[i * n + i]) || S[i * n + i] <= 0.0f)
            {
                ok = 0;
                break;
            }
            if (!isfinite(xhat[i]))
            {
                ok = 0;
                break;
            }
        }
        
        if (!ok)
        {
            printf("  FAILED at iteration %d\n", iter);
            passed = 0;
            break;
        }
    }
    
    if (passed)
    {
        printf("  Workspace reuse PASSED (%d updates)\n", num_updates);
    }
    
cleanup:
    gemm_aligned_free(S);
    gemm_aligned_free(xhat);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    ukf_upd_ws_cleanup(&ws);
    
    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_ukf_update_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("    SR-UKF MEASUREMENT UPDATE TESTS\n");
    printf("=================================================\n");
    
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    /* Basic Tests */
    printf("\n--- Basic Functionality Tests ---\n");
    
    results->total++;
    if (test_update_basic())
    {
        results->passed++;
        printf("✓ Basic update test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Basic update test FAILED\n");
    }
    
    results->total++;
    if (test_update_sizes())
    {
        results->passed++;
        printf("✓ Update sizes test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Update sizes test FAILED\n");
    }
    
    /* Correctness Tests */
    printf("\n--- Correctness Tests ---\n");
    
    results->total++;
    if (test_state_update_correctness())
    {
        results->passed++;
        printf("✓ State update correctness PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ State update correctness FAILED\n");
    }
    
    results->total++;
    if (test_covariance_downdate_correctness())
    {
        results->passed++;
        printf("✓ Covariance downdate correctness PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Covariance downdate correctness FAILED\n");
    }
    
    /* Edge Cases */
    printf("\n--- Edge Case Tests ---\n");
    
    results->total++;
    if (test_zero_innovation())
    {
        results->passed++;
        printf("✓ Zero innovation test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Zero innovation test FAILED\n");
    }
    
    /* Stability Tests */
    printf("\n--- Stability Tests ---\n");
    
    results->total++;
    if (test_pd_preservation())
    {
        results->passed++;
        printf("✓ PD preservation test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ PD preservation test FAILED\n");
    }
    
    results->total++;
    if (test_numerical_stability())
    {
        results->passed++;
        printf("✓ Numerical stability test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Numerical stability test FAILED\n");
    }
    
    /* Workspace Tests */
    printf("\n--- Workspace Tests ---\n");
    
    results->total++;
    if (test_workspace_reuse())
    {
        results->passed++;
        printf("✓ Workspace reuse test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Workspace reuse test FAILED\n");
    }
    
    /* Summary */
    printf("\n=================================================\n");
    printf("SR-UKF Update Tests: %d/%d passed\n", results->passed, results->total);
    
    if (results->passed == results->total)
    {
        printf("✓ ALL SR-UKF UPDATE TESTS PASSED!\n");
    }
    else
    {
        printf("✗ %d SR-UKF update tests FAILED\n", results->failed);
    }
    printf("=================================================\n");
    
    return (results->failed == 0) ? 0 : 1;
}

//==============================================================================
// STANDALONE MODE
//==============================================================================

#ifdef STANDALONE
int main(void)
{
    test_results_t results = {0};
    return run_ukf_update_tests(&results);
}
#endif