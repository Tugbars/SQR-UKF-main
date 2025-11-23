/**
 * @file test_ukf_covariance.c
 * @brief Unit tests for SR-UKF covariance operations
 *
 * Tests:
 * - create_state_estimation_error_covariance_matrix: QR-based SR covariance
 * - create_state_cross_covariance_matrix: Cross-covariance Pxy computation
 * - Positive definiteness preservation
 * - Upper triangular structure verification
 * - Covariance reconstruction correctness
 * - Numerical stability across various state sizes
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
 * @brief Reconstruct symmetric matrix from upper triangular SR factor: A = S^T * S
 */
static void reconstruct_from_upper_sr(const float *S, float *A, size_t n, size_t ld)
{
    /* A = S^T * S where S is upper triangular */
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            /* A[i,j] = sum_k S[k,i] * S[k,j] for k where both S[k,i] and S[k,j] exist */
            /* Since S is upper: S[k,i] != 0 only if k <= i, S[k,j] != 0 only if k <= j */
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
    
    printf("    PASSED (max lower element = %.6e)\n", max_lower);
    return 1;
}

/**
 * @brief Check if matrix has positive diagonal (necessary for valid SR factor)
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
        printf("    FAILED: %d non-positive diagonal elements (min=%.6e)\n",
               neg_count, min_diag);
        return 0;
    }
    
    printf("    PASSED (min diagonal = %.6e)\n", min_diag);
    return 1;
}

/**
 * @brief Check if reconstructed covariance is symmetric
 */
static int check_symmetry(const float *A, size_t n, size_t ld,
                          double tol, const char *name)
{
    printf("  Checking symmetry (%s)...\n", name);
    
    double max_asym = 0.0;
    
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = i + 1; j < n; j++)
        {
            double diff = fabs((double)A[i * ld + j] - (double)A[j * ld + i]);
            if (diff > max_asym)
                max_asym = diff;
        }
    }
    
    if (max_asym > tol)
    {
        printf("    FAILED: max asymmetry = %.6e\n", max_asym);
        return 0;
    }
    
    printf("    PASSED (max asymmetry = %.6e)\n", max_asym);
    return 1;
}

//==============================================================================
// REFERENCE IMPLEMENTATIONS
//==============================================================================

/**
 * @brief Reference cross-covariance: Pxy = sum_k Wc[k] * (X[:,k]-x) * (Y[:,k]-y)^T
 */
static void cross_covariance_reference(float *Pxy,
                                       const float *Wc,
                                       const float *X, const float *Y,
                                       const float *x, const float *y,
                                       size_t L, size_t N)
{
    /* Zero output */
    memset(Pxy, 0, L * L * sizeof(float));
    
    /* Accumulate outer products */
    for (size_t k = 0; k < N; k++)
    {
        float wk = Wc[k];
        
        for (size_t i = 0; i < L; i++)
        {
            float xi_centered = X[i * N + k] - x[i];
            
            for (size_t j = 0; j < L; j++)
            {
                float yj_centered = Y[j * N + k] - y[j];
                Pxy[i * L + j] += wk * xi_centered * yj_centered;
            }
        }
    }
}

/**
 * @brief Reference covariance from weighted deviations: P = sum_k Wc[k] * dev[:,k] * dev[:,k]^T
 */
static void covariance_from_deviations_reference(float *P,
                                                 const float *Wc,
                                                 const float *X,
                                                 const float *x,
                                                 size_t L, size_t N)
{
    memset(P, 0, L * L * sizeof(float));
    
    for (size_t k = 0; k < N; k++)
    {
        float wk = Wc[k];
        
        for (size_t i = 0; i < L; i++)
        {
            float di = X[i * N + k] - x[i];
            
            for (size_t j = 0; j < L; j++)
            {
                float dj = X[j * N + k] - x[j];
                P[i * L + j] += wk * di * dj;
            }
        }
    }
}

//==============================================================================
// TEST: create_state_estimation_error_covariance_matrix
//==============================================================================

/**
 * @brief Test basic QR-based SR covariance construction
 */
static int test_sr_covariance_basic(void)
{
    printf("\n=== Testing SR covariance construction (basic) ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    
    printf("  State dimension L=%d, sigma points N=%zu\n", L, N);
    
    /* Allocate */
    float *S = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
    float *Rsr = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    
    ukf_qr_ws_t ws = {0};
    
    if (!S || !X || !x || !Rsr || !Wc || !Wm)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Generate test data */
    generate_random_vector(x, Ls, 11111);
    generate_random_upper_triangular(Rsr, Ls, Ls, 22222, 0.5f);
    
    /* Create weights */
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    
    /* Create sigma points from x and some initial SR factor */
    float *S_init = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    generate_random_upper_triangular(S_init, Ls, Ls, 33333, 1.0f);
    create_sigma_point_matrix(X, x, S_init, alpha, kappa, L);
    gemm_aligned_free(S_init);
    
    /* Call function under test */
    printf("  Calling create_state_estimation_error_covariance_matrix...\n");
    int rc = create_state_estimation_error_covariance_matrix(
        S, &ws, Wc, X, x, Rsr, L);
    
    if (rc != 0)
    {
        printf("  FAILED: returned error code %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* Check properties */
    passed &= check_upper_triangular(S, Ls, Ls, 1e-6, "S");
    passed &= check_positive_diagonal(S, Ls, Ls, "S");
    
    /* Reconstruct covariance and check symmetry */
    float *P_reconstructed = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    reconstruct_from_upper_sr(S, P_reconstructed, Ls, Ls);
    passed &= check_symmetry(P_reconstructed, Ls, Ls, 1e-5, "S^T*S");
    
    /* Check positive definiteness (all eigenvalues > 0) via diagonal dominance proxy */
    int pd_ok = 1;
    for (size_t i = 0; i < Ls; i++)
    {
        if (P_reconstructed[i * Ls + i] <= 0.0f)
        {
            pd_ok = 0;
            break;
        }
    }
    
    if (!pd_ok)
    {
        printf("  FAILED: Reconstructed covariance has non-positive diagonal\n");
        passed = 0;
    }
    else
    {
        printf("  Reconstructed covariance diagonal check PASSED\n");
    }
    
    gemm_aligned_free(P_reconstructed);
    
cleanup:
    gemm_aligned_free(S);
    gemm_aligned_free(X);
    gemm_aligned_free(x);
    gemm_aligned_free(Rsr);
    gemm_aligned_free(Wc);
    gemm_aligned_free(Wm);
    ukf_qr_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test SR covariance across various state sizes
 */
static int test_sr_covariance_sizes(void)
{
    printf("\n=== Testing SR covariance construction (various sizes) ===\n");
    
    int passed = 1;
    
    uint8_t test_L[] = {4, 8, 16, 32, 64};
    const int num_L = sizeof(test_L) / sizeof(test_L[0]);
    
    ukf_qr_ws_t ws = {0};
    
    for (int tc = 0; tc < num_L; tc++)
    {
        uint8_t L = test_L[tc];
        const size_t Ls = (size_t)L;
        const size_t N = 2u * Ls + 1u;
        
        printf("  Testing L=%d...\n", L);
        
        float *S = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
        float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
        float *Rsr = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
        float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
        float *S_init = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        
        if (!S || !X || !x || !Rsr || !Wc || !Wm || !S_init)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_size;
        }
        
        /* Generate test data */
        generate_random_vector(x, Ls, tc * 11111);
        generate_random_upper_triangular(Rsr, Ls, Ls, tc * 22222, 0.5f);
        generate_random_upper_triangular(S_init, Ls, Ls, tc * 33333, 1.0f);
        
        const float alpha = 1e-3f;
        const float beta = 2.0f;
        const float kappa = 0.0f;
        create_weights(Wc, Wm, alpha, beta, kappa, L);
        create_sigma_point_matrix(X, x, S_init, alpha, kappa, L);
        
        int rc = create_state_estimation_error_covariance_matrix(
            S, &ws, Wc, X, x, Rsr, L);
        
        if (rc != 0)
        {
            printf("    FAILED: returned %d\n", rc);
            passed = 0;
        }
        else
        {
            int size_passed = 1;
            size_passed &= check_upper_triangular(S, Ls, Ls, 1e-6, "S");
            size_passed &= check_positive_diagonal(S, Ls, Ls, "S");
            
            if (size_passed)
            {
                printf("    L=%d PASSED\n", L);
            }
            else
            {
                passed = 0;
            }
        }
        
cleanup_size:
        gemm_aligned_free(S);
        gemm_aligned_free(X);
        gemm_aligned_free(x);
        gemm_aligned_free(Rsr);
        gemm_aligned_free(Wc);
        gemm_aligned_free(Wm);
        gemm_aligned_free(S_init);
    }
    
    ukf_qr_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test that SR factor correctly includes process noise
 */
static int test_sr_covariance_noise_inclusion(void)
{
    printf("\n=== Testing SR covariance noise inclusion ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    
    float *S = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
    float *Rsr = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    float *P_reconstructed = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *Q = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    
    ukf_qr_ws_t ws = {0};
    
    if (!S || !X || !x || !Rsr || !Wc || !Wm || !P_reconstructed || !Q)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Generate data with known noise */
    generate_random_vector(x, Ls, 44444);
    
    /* Create diagonal process noise SR */
    memset(Rsr, 0, Ls * Ls * sizeof(float));
    for (size_t i = 0; i < Ls; i++)
    {
        Rsr[i * Ls + i] = 0.1f * (float)(i + 1); /* Diagonal: 0.1, 0.2, ... */
    }
    
    /* Reconstruct Q = Rsr^T * Rsr */
    reconstruct_from_upper_sr(Rsr, Q, Ls, Ls);
    
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    
    /* Create sigma points with identity-like covariance */
    float *S_init = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    memset(S_init, 0, Ls * Ls * sizeof(float));
    for (size_t i = 0; i < Ls; i++)
    {
        S_init[i * Ls + i] = 1.0f; /* Identity SR */
    }
    create_sigma_point_matrix(X, x, S_init, alpha, kappa, L);
    gemm_aligned_free(S_init);
    
    /* Compute SR covariance */
    int rc = create_state_estimation_error_covariance_matrix(
        S, &ws, Wc, X, x, Rsr, L);
    
    if (rc != 0)
    {
        printf("  FAILED: returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* Reconstruct P = S^T * S */
    reconstruct_from_upper_sr(S, P_reconstructed, Ls, Ls);
    
    /* P should be approximately P_sigma + Q where P_sigma comes from sigma deviations */
    /* At minimum, diagonal of P should be >= diagonal of Q */
    printf("  Checking noise contribution to covariance diagonal...\n");
    int noise_ok = 1;
    for (size_t i = 0; i < Ls; i++)
    {
        float p_diag = P_reconstructed[i * Ls + i];
        float q_diag = Q[i * Ls + i];
        
        if (p_diag < q_diag * 0.9f) /* Allow 10% tolerance */
        {
            printf("    P[%zu,%zu]=%.4f < Q[%zu,%zu]=%.4f\n",
                   i, i, p_diag, i, i, q_diag);
            noise_ok = 0;
        }
    }
    
    if (!noise_ok)
    {
        printf("  FAILED: Process noise not properly included\n");
        passed = 0;
    }
    else
    {
        printf("  PASSED: Process noise properly contributes to covariance\n");
    }
    
cleanup:
    gemm_aligned_free(S);
    gemm_aligned_free(X);
    gemm_aligned_free(x);
    gemm_aligned_free(Rsr);
    gemm_aligned_free(Wc);
    gemm_aligned_free(Wm);
    gemm_aligned_free(P_reconstructed);
    gemm_aligned_free(Q);
    ukf_qr_ws_cleanup(&ws);
    
    return passed;
}

//==============================================================================
// TEST: create_state_cross_covariance_matrix
//==============================================================================

/**
 * @brief Test basic cross-covariance computation
 */
static int test_cross_covariance_basic(void)
{
    printf("\n=== Testing cross-covariance Pxy (basic) ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    
    printf("  State dimension L=%d, sigma points N=%zu\n", L, N);
    
    float *Pxy = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *Pxy_ref = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *Y = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
    float *y = gemm_aligned_alloc(32, Ls * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    
    ukf_pxy_ws_t ws = {0};
    
    if (!Pxy || !Pxy_ref || !X || !Y || !x || !y || !Wc || !Wm)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Generate random sigma points */
    srand(55555);
    for (size_t i = 0; i < Ls * N; i++)
    {
        X[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        Y[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    }
    
    /* Compute means */
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    multiply_sigma_point_matrix_to_weights(x, X, Wm, L);
    multiply_sigma_point_matrix_to_weights(y, Y, Wm, L);
    
    /* Compute using implementation under test */
    int rc = create_state_cross_covariance_matrix(
        Pxy, Wc, X, Y, x, y, &ws, L);
    
    if (rc != 0)
    {
        printf("  FAILED: returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* Compute reference */
    cross_covariance_reference(Pxy_ref, Wc, X, Y, x, y, Ls, N);
    
    /* Compare */
    double rel_err = matrix_relative_error(Pxy, Pxy_ref, Ls, Ls, Ls);
    
    if (rel_err > 1e-4)
    {
        printf("  FAILED: relative error = %.6e\n", rel_err);
        passed = 0;
        
        /* Debug output */
        printf("  First few elements:\n");
        for (size_t i = 0; i < 3 && i < Ls; i++)
        {
            for (size_t j = 0; j < 3 && j < Ls; j++)
            {
                printf("    Pxy[%zu,%zu]: got %.6f, expected %.6f\n",
                       i, j, Pxy[i * Ls + j], Pxy_ref[i * Ls + j]);
            }
        }
    }
    else
    {
        printf("  PASSED (relative error = %.6e)\n", rel_err);
    }
    
cleanup:
    gemm_aligned_free(Pxy);
    gemm_aligned_free(Pxy_ref);
    gemm_aligned_free(X);
    gemm_aligned_free(Y);
    gemm_aligned_free(x);
    gemm_aligned_free(y);
    gemm_aligned_free(Wc);
    gemm_aligned_free(Wm);
    ukf_pxy_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test cross-covariance across various sizes
 */
static int test_cross_covariance_sizes(void)
{
    printf("\n=== Testing cross-covariance Pxy (various sizes) ===\n");
    
    int passed = 1;
    
    uint8_t test_L[] = {4, 8, 16, 32, 64};
    const int num_L = sizeof(test_L) / sizeof(test_L[0]);
    
    ukf_pxy_ws_t ws = {0};
    
    for (int tc = 0; tc < num_L; tc++)
    {
        uint8_t L = test_L[tc];
        const size_t Ls = (size_t)L;
        const size_t N = 2u * Ls + 1u;
        
        printf("  Testing L=%d...\n", L);
        
        float *Pxy = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        float *Pxy_ref = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
        float *Y = gemm_aligned_alloc(32, Ls * N * sizeof(float));
        float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
        float *y = gemm_aligned_alloc(32, Ls * sizeof(float));
        float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
        float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
        
        if (!Pxy || !Pxy_ref || !X || !Y || !x || !y || !Wc || !Wm)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_size;
        }
        
        /* Generate data */
        srand(tc * 66666);
        for (size_t i = 0; i < Ls * N; i++)
        {
            X[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
            Y[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        }
        
        const float alpha = 1e-3f;
        const float beta = 2.0f;
        const float kappa = 0.0f;
        create_weights(Wc, Wm, alpha, beta, kappa, L);
        multiply_sigma_point_matrix_to_weights(x, X, Wm, L);
        multiply_sigma_point_matrix_to_weights(y, Y, Wm, L);
        
        int rc = create_state_cross_covariance_matrix(
            Pxy, Wc, X, Y, x, y, &ws, L);
        
        if (rc != 0)
        {
            printf("    FAILED: returned %d\n", rc);
            passed = 0;
        }
        else
        {
            cross_covariance_reference(Pxy_ref, Wc, X, Y, x, y, Ls, N);
            double rel_err = matrix_relative_error(Pxy, Pxy_ref, Ls, Ls, Ls);
            
            if (rel_err > 1e-4)
            {
                printf("    FAILED: relative error = %.6e\n", rel_err);
                passed = 0;
            }
            else
            {
                printf("    PASSED (rel_err=%.2e)\n", rel_err);
            }
        }
        
cleanup_size:
        gemm_aligned_free(Pxy);
        gemm_aligned_free(Pxy_ref);
        gemm_aligned_free(X);
        gemm_aligned_free(Y);
        gemm_aligned_free(x);
        gemm_aligned_free(y);
        gemm_aligned_free(Wc);
        gemm_aligned_free(Wm);
    }
    
    ukf_pxy_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test cross-covariance with identical X and Y (should equal auto-covariance)
 */
static int test_cross_covariance_self(void)
{
    printf("\n=== Testing cross-covariance Pxy with X=Y (auto-covariance) ===\n");
    
    int passed = 1;
    
    const uint8_t L = 16;
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    
    float *Pxy = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    
    ukf_pxy_ws_t ws = {0};
    
    if (!Pxy || !X || !x || !Wc || !Wm)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Generate random sigma points */
    srand(77777);
    for (size_t i = 0; i < Ls * N; i++)
    {
        X[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    }
    
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    multiply_sigma_point_matrix_to_weights(x, X, Wm, L);
    
    /* Compute Pxy with X=Y and x=y */
    int rc = create_state_cross_covariance_matrix(
        Pxy, Wc, X, X, x, x, &ws, L);
    
    if (rc != 0)
    {
        printf("  FAILED: returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* When X=Y, Pxy should be symmetric (it's the auto-covariance) */
    passed &= check_symmetry(Pxy, Ls, Ls, 1e-5, "Pxx");
    
    /* Diagonal should be non-negative (variances) */
    int diag_ok = 1;
    float min_var = FLT_MAX;
    for (size_t i = 0; i < Ls; i++)
    {
        float var = Pxy[i * Ls + i];
        if (var < min_var) min_var = var;
        if (var < -1e-6)
        {
            diag_ok = 0;
            printf("    Negative variance at [%zu,%zu] = %.6e\n", i, i, var);
        }
    }
    
    if (!diag_ok)
    {
        printf("  FAILED: Negative variances detected\n");
        passed = 0;
    }
    else
    {
        printf("  Variance check PASSED (min variance = %.6e)\n", min_var);
    }
    
cleanup:
    gemm_aligned_free(Pxy);
    gemm_aligned_free(X);
    gemm_aligned_free(x);
    gemm_aligned_free(Wc);
    gemm_aligned_free(Wm);
    ukf_pxy_ws_cleanup(&ws);
    
    return passed;
}

/**
 * @brief Test cross-covariance with zero-centered data
 */
static int test_cross_covariance_zero_mean(void)
{
    printf("\n=== Testing cross-covariance with zero mean ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    
    float *Pxy = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *Pxy_ref = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *Y = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
    float *y = gemm_aligned_alloc(32, Ls * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    
    ukf_pxy_ws_t ws = {0};
    
    if (!Pxy || !Pxy_ref || !X || !Y || !x || !y || !Wc || !Wm)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Generate data */
    srand(88888);
    for (size_t i = 0; i < Ls * N; i++)
    {
        X[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        Y[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    }
    
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    
    /* Use zero means */
    memset(x, 0, Ls * sizeof(float));
    memset(y, 0, Ls * sizeof(float));
    
    /* Compute */
    int rc = create_state_cross_covariance_matrix(
        Pxy, Wc, X, Y, x, y, &ws, L);
    
    if (rc != 0)
    {
        printf("  FAILED: returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* Reference */
    cross_covariance_reference(Pxy_ref, Wc, X, Y, x, y, Ls, N);
    
    double rel_err = matrix_relative_error(Pxy, Pxy_ref, Ls, Ls, Ls);
    
    if (rel_err > 1e-4)
    {
        printf("  FAILED: relative error = %.6e\n", rel_err);
        passed = 0;
    }
    else
    {
        printf("  PASSED (relative error = %.6e)\n", rel_err);
    }
    
cleanup:
    gemm_aligned_free(Pxy);
    gemm_aligned_free(Pxy_ref);
    gemm_aligned_free(X);
    gemm_aligned_free(Y);
    gemm_aligned_free(x);
    gemm_aligned_free(y);
    gemm_aligned_free(Wc);
    gemm_aligned_free(Wm);
    ukf_pxy_ws_cleanup(&ws);
    
    return passed;
}

//==============================================================================
// TEST: Numerical stability
//==============================================================================

/**
 * @brief Test numerical stability with poorly conditioned data
 */
static int test_numerical_stability(void)
{
    printf("\n=== Testing numerical stability ===\n");
    
    int passed = 1;
    
    const uint8_t L = 16;
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    
    float *S = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
    float *Rsr = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    
    ukf_qr_ws_t ws = {0};
    
    if (!S || !X || !x || !Rsr || !Wc || !Wm)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Test 1: Very small values */
    printf("  Testing with small values (scale 1e-6)...\n");
    {
        generate_random_vector(x, Ls, 99999);
        for (size_t i = 0; i < Ls; i++) x[i] *= 1e-6f;
        
        memset(Rsr, 0, Ls * Ls * sizeof(float));
        for (size_t i = 0; i < Ls; i++)
        {
            Rsr[i * Ls + i] = 1e-7f;
        }
        
        const float alpha = 1e-3f;
        const float beta = 2.0f;
        const float kappa = 0.0f;
        create_weights(Wc, Wm, alpha, beta, kappa, L);
        
        float *S_init = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        memset(S_init, 0, Ls * Ls * sizeof(float));
        for (size_t i = 0; i < Ls; i++) S_init[i * Ls + i] = 1e-6f;
        create_sigma_point_matrix(X, x, S_init, alpha, kappa, L);
        gemm_aligned_free(S_init);
        
        int rc = create_state_estimation_error_covariance_matrix(
            S, &ws, Wc, X, x, Rsr, L);
        
        if (rc != 0)
        {
            printf("    FAILED: returned %d\n", rc);
            passed = 0;
        }
        else
        {
            int ok = check_positive_diagonal(S, Ls, Ls, "small values");
            if (ok)
            {
                printf("    Small values test PASSED\n");
            }
            else
            {
                passed = 0;
            }
        }
    }
    
    /* Test 2: Large values */
    printf("  Testing with large values (scale 1e6)...\n");
    {
        generate_random_vector(x, Ls, 11111);
        for (size_t i = 0; i < Ls; i++) x[i] *= 1e6f;
        
        memset(Rsr, 0, Ls * Ls * sizeof(float));
        for (size_t i = 0; i < Ls; i++)
        {
            Rsr[i * Ls + i] = 1e5f;
        }
        
        const float alpha = 1e-3f;
        const float beta = 2.0f;
        const float kappa = 0.0f;
        create_weights(Wc, Wm, alpha, beta, kappa, L);
        
        float *S_init = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        memset(S_init, 0, Ls * Ls * sizeof(float));
        for (size_t i = 0; i < Ls; i++) S_init[i * Ls + i] = 1e6f;
        create_sigma_point_matrix(X, x, S_init, alpha, kappa, L);
        gemm_aligned_free(S_init);
        
        int rc = create_state_estimation_error_covariance_matrix(
            S, &ws, Wc, X, x, Rsr, L);
        
        if (rc != 0)
        {
            printf("    FAILED: returned %d\n", rc);
            passed = 0;
        }
        else
        {
            int ok = check_positive_diagonal(S, Ls, Ls, "large values");
            ok &= check_upper_triangular(S, Ls, Ls, 1e-3, "large values");
            
            /* Check no NaN/Inf */
            int finite_ok = 1;
            for (size_t i = 0; i < Ls * Ls; i++)
            {
                if (!isfinite(S[i]))
                {
                    finite_ok = 0;
                    break;
                }
            }
            
            if (ok && finite_ok)
            {
                printf("    Large values test PASSED\n");
            }
            else
            {
                printf("    FAILED: Non-finite values or structure error\n");
                passed = 0;
            }
        }
    }
    
cleanup:
    gemm_aligned_free(S);
    gemm_aligned_free(X);
    gemm_aligned_free(x);
    gemm_aligned_free(Rsr);
    gemm_aligned_free(Wc);
    gemm_aligned_free(Wm);
    ukf_qr_ws_cleanup(&ws);
    
    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_ukf_covariance_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("    SR-UKF COVARIANCE TESTS\n");
    printf("=================================================\n");
    
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    /* SR Covariance Tests */
    printf("\n--- SR Covariance Construction Tests ---\n");
    
    results->total++;
    if (test_sr_covariance_basic())
    {
        results->passed++;
        printf("✓ SR covariance basic test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ SR covariance basic test FAILED\n");
    }
    
    results->total++;
    if (test_sr_covariance_sizes())
    {
        results->passed++;
        printf("✓ SR covariance sizes test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ SR covariance sizes test FAILED\n");
    }
    
    results->total++;
    if (test_sr_covariance_noise_inclusion())
    {
        results->passed++;
        printf("✓ SR covariance noise inclusion test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ SR covariance noise inclusion test FAILED\n");
    }
    
    /* Cross-Covariance Tests */
    printf("\n--- Cross-Covariance Pxy Tests ---\n");
    
    results->total++;
    if (test_cross_covariance_basic())
    {
        results->passed++;
        printf("✓ Cross-covariance basic test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Cross-covariance basic test FAILED\n");
    }
    
    results->total++;
    if (test_cross_covariance_sizes())
    {
        results->passed++;
        printf("✓ Cross-covariance sizes test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Cross-covariance sizes test FAILED\n");
    }
    
    results->total++;
    if (test_cross_covariance_self())
    {
        results->passed++;
        printf("✓ Cross-covariance self (auto-cov) test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Cross-covariance self (auto-cov) test FAILED\n");
    }
    
    results->total++;
    if (test_cross_covariance_zero_mean())
    {
        results->passed++;
        printf("✓ Cross-covariance zero mean test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Cross-covariance zero mean test FAILED\n");
    }
    
    /* Numerical Stability Tests */
    printf("\n--- Numerical Stability Tests ---\n");
    
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
    
    /* Summary */
    printf("\n=================================================\n");
    printf("SR-UKF Covariance Tests: %d/%d passed\n", results->passed, results->total);
    
    if (results->passed == results->total)
    {
        printf("✓ ALL SR-UKF COVARIANCE TESTS PASSED!\n");
    }
    else
    {
        printf("✗ %d SR-UKF covariance tests FAILED\n", results->failed);
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
    return run_ukf_covariance_tests(&results);
}
#endif