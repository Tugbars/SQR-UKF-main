/**
 * @file test_cholupdate.c
 * @brief Unit tests for rank-k Cholesky updates with QR acceleration
 *
 * Tests:
 * - Rank-1 updates/downdates (Givens rotations)
 * - Rank-k updates (tiled vs QR-based)
 * - Algorithm selection heuristics
 * - Numerical stability and positive definiteness
 * - Upper vs lower triangular storage
 * - Workspace reuse and memory efficiency
 * - Edge cases and error handling
 *
 * @author TUGBARS
 * @date 2025
 */

#include "test_common.h"
#include "cholupdate.h"
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
 * @brief Compute Frobenius norm: ||A||_F = sqrt(sum(A[i,j]^2))
 */
static double frobenius_norm(const float *A, uint16_t m, uint16_t n)
{
    double sum = 0.0;
    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double val = (double)A[i * n + j];
            sum += val * val;
        }
    }
    return sqrt(sum);
}

/**
 * @brief Compute relative error: ||A - B||_F / ||A||_F
 */
static double relative_error(const float *A, const float *B, uint16_t m, uint16_t n)
{
    double diff_norm = 0.0;
    double a_norm = 0.0;

    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double a = (double)A[i * n + j];
            double b = (double)B[i * n + j];
            double diff = a - b;

            diff_norm += diff * diff;
            a_norm += a * a;
        }
    }

    if (a_norm < 1e-30)
        return 0.0;

    return sqrt(diff_norm / a_norm);
}

/**
 * @brief Print matrix for debugging
 */
static void print_matrix_debug(const char *name, const float *M,
                               uint16_t rows, uint16_t cols, uint16_t max_display)
{
    printf("%s (%dx%d):\n", name, rows, cols);
    uint16_t display_rows = MIN(rows, max_display);
    uint16_t display_cols = MIN(cols, max_display);

    for (uint16_t i = 0; i < display_rows; i++)
    {
        for (uint16_t j = 0; j < display_cols; j++)
        {
            printf("%8.4f ", M[i * cols + j]);
        }
        if (cols > max_display)
            printf("...");
        printf("\n");
    }
    if (rows > max_display)
        printf("...\n");
    printf("\n");
}

/**
 * @brief Generate random symmetric positive definite matrix via A*A^T
 */
static void generate_spd_matrix(float *A, uint16_t n, unsigned int seed)
{
    srand(seed);
    
    // Generate random matrix
    float *temp = gemm_aligned_alloc(32, n * n * sizeof(float));
    for (uint16_t i = 0; i < n * n; i++)
    {
        temp[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }
    
    // A = temp * temp^T (guaranteed SPD)
    memset(A, 0, n * n * sizeof(float));
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (uint16_t k = 0; k < n; k++)
            {
                sum += (double)temp[i * n + k] * (double)temp[j * n + k];
            }
            A[i * n + j] = (float)sum;
        }
    }
    
    // Add diagonal dominance for better conditioning
    for (uint16_t i = 0; i < n; i++)
    {
        A[i * n + i] += (float)n;
    }
    
    gemm_aligned_free(temp);
}

/**
 * @brief Compute Cholesky factorization (simple unblocked, for testing)
 */
static int cholesky_factorize_simple(float *A, uint16_t n, bool is_upper)
{
    if (is_upper)
    {
        // Upper triangular: U^T * U = A
        for (uint16_t j = 0; j < n; j++)
        {
            // Compute U[j,j]
            double sum = (double)A[j * n + j];
            for (uint16_t k = 0; k < j; k++)
            {
                double ujk = (double)A[k * n + j];
                sum -= ujk * ujk;
            }
            
            if (sum <= 0.0)
                return -1; // Not positive definite
            
            A[j * n + j] = (float)sqrt(sum);
            
            // Compute U[j,j+1:n]
            for (uint16_t i = j + 1; i < n; i++)
            {
                sum = (double)A[j * n + i];
                for (uint16_t k = 0; k < j; k++)
                {
                    sum -= (double)A[k * n + j] * (double)A[k * n + i];
                }
                A[j * n + i] = (float)(sum / A[j * n + j]);
            }
        }
        
        // Zero out lower triangle
        for (uint16_t i = 0; i < n; i++)
        {
            for (uint16_t j = 0; j < i; j++)
            {
                A[i * n + j] = 0.0f;
            }
        }
    }
    else
    {
        // Lower triangular: L * L^T = A
        for (uint16_t j = 0; j < n; j++)
        {
            double sum = (double)A[j * n + j];
            for (uint16_t k = 0; k < j; k++)
            {
                double ljk = (double)A[j * n + k];
                sum -= ljk * ljk;
            }
            
            if (sum <= 0.0)
                return -1;
            
            A[j * n + j] = (float)sqrt(sum);
            
            for (uint16_t i = j + 1; i < n; i++)
            {
                sum = (double)A[i * n + j];
                for (uint16_t k = 0; k < j; k++)
                {
                    sum -= (double)A[i * n + k] * (double)A[j * n + k];
                }
                A[i * n + j] = (float)(sum / A[j * n + j]);
            }
        }
        
        // Zero out upper triangle
        for (uint16_t i = 0; i < n; i++)
        {
            for (uint16_t j = i + 1; j < n; j++)
            {
                A[i * n + j] = 0.0f;
            }
        }
    }
    
    return 0;
}

/**
 * @brief Reconstruct matrix from Cholesky factor: A = L*L^T or U^T*U
 */
static void reconstruct_from_cholesky(const float *L, float *A, uint16_t n, bool is_upper)
{
    memset(A, 0, n * n * sizeof(float));
    
    if (is_upper)
    {
        // A = U^T * U
        for (uint16_t i = 0; i < n; i++)
        {
            for (uint16_t j = i; j < n; j++)
            {
                double sum = 0.0;
                for (uint16_t k = 0; k <= i; k++)
                {
                    sum += (double)L[k * n + i] * (double)L[k * n + j];
                }
                A[i * n + j] = (float)sum;
                A[j * n + i] = (float)sum; // Symmetric
            }
        }
    }
    else
    {
        // A = L * L^T
        for (uint16_t i = 0; i < n; i++)
        {
            for (uint16_t j = 0; j <= i; j++)
            {
                double sum = 0.0;
                for (uint16_t k = 0; k <= j; k++)
                {
                    sum += (double)L[i * n + k] * (double)L[j * n + k];
                }
                A[i * n + j] = (float)sum;
                A[j * n + i] = (float)sum; // Symmetric
            }
        }
    }
}

//==============================================================================
// CHOLESKY UPDATE PROPERTY CHECKERS
//==============================================================================

/**
 * @brief Check if L*L^T matches expected matrix
 */
static int check_cholesky_factorization(const float *L, const float *A_expected,
                                        uint16_t n, bool is_upper, double tol,
                                        const char *test_name)
{
    printf("  Checking Cholesky factorization...\n");
    
    float *A_reconstructed = gemm_aligned_alloc(32, n * n * sizeof(float));
    if (!A_reconstructed)
    {
        printf("    ERROR: Allocation failed\n");
        return 0;
    }
    
    reconstruct_from_cholesky(L, A_reconstructed, n, is_upper);
    
    double rel_err = relative_error(A_expected, A_reconstructed, n, n);
    double a_norm = frobenius_norm(A_expected, n, n);
    
    gemm_aligned_free(A_reconstructed);
    
    if (rel_err > tol)
    {
        printf("    %s: Factorization check FAILED\n", test_name);
        printf("    ||A_expected||_F = %.6e\n", a_norm);
        printf("    ||A_expected - L*L^T||_F / ||A_expected||_F = %.6e (tol: %.6e)\n",
               rel_err, tol);
        return 0;
    }
    
    printf("    %s: Factorization check PASSED\n", test_name);
    printf("    Relative error: %.6e\n", rel_err);
    return 1;
}

/**
 * @brief Check update correctness: L_new*L_new^T = L*L^T + X*X^T
 */
static int check_update_correctness(const float *L_original, const float *L_updated,
                                    const float *X, uint16_t n, uint16_t k,
                                    bool is_upper, int add, double tol,
                                    const char *test_name)
{
    printf("  Checking update correctness...\n");
    
    float *A_original = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *A_updated = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *A_expected = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    if (!A_original || !A_updated || !A_expected)
    {
        printf("    ERROR: Allocation failed\n");
        gemm_aligned_free(A_original);
        gemm_aligned_free(A_updated);
        gemm_aligned_free(A_expected);
        return 0;
    }
    
    // Reconstruct original: A_original = L*L^T
    reconstruct_from_cholesky(L_original, A_original, n, is_upper);
    
    // Reconstruct updated: A_updated = L_new*L_new^T
    reconstruct_from_cholesky(L_updated, A_updated, n, is_upper);
    
    // Compute expected: A_expected = A_original ± X*X^T
    memcpy(A_expected, A_original, n * n * sizeof(float));
    
    const float sign = (add >= 0) ? 1.0f : -1.0f;
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (uint16_t p = 0; p < k; p++)
            {
                sum += (double)X[i * k + p] * (double)X[j * k + p];
            }
            A_expected[i * n + j] += sign * (float)sum;
        }
    }
    
    // Compare
    double rel_err = relative_error(A_expected, A_updated, n, n);
    
    gemm_aligned_free(A_original);
    gemm_aligned_free(A_updated);
    gemm_aligned_free(A_expected);
    
    if (rel_err > tol)
    {
        printf("    %s: Update correctness FAILED\n", test_name);
        printf("    ||A_expected - A_updated||_F / ||A_expected||_F = %.6e (tol: %.6e)\n",
               rel_err, tol);
        return 0;
    }
    
    printf("    %s: Update correctness PASSED\n", test_name);
    printf("    Relative error: %.6e\n", rel_err);
    return 1;
}

/**
 * @brief Check if matrix is positive definite (all diagonal elements > 0)
 */
static int check_positive_definite(const float *L, uint16_t n, const char *test_name)
{
    printf("  Checking positive definiteness...\n");
    
    int errors = 0;
    double min_diag = FLT_MAX;
    
    for (uint16_t i = 0; i < n; i++)
    {
        double diag = (double)L[i * n + i];
        if (diag < min_diag)
            min_diag = diag;
        
        if (diag <= 0.0 || !isfinite(diag))
        {
            if (errors < 3)
            {
                printf("    Diagonal [%d,%d] = %.6e (not positive!)\n", i, i, diag);
            }
            errors++;
        }
    }
    
    if (errors > 0)
    {
        printf("    %s: NOT positive definite (%d errors)\n", test_name, errors);
        return 0;
    }
    
    printf("    %s: Positive definite (min diagonal = %.6e)\n", test_name, min_diag);
    return 1;
}

//==============================================================================
// INDIVIDUAL TESTS
//==============================================================================

/**
 * @brief Test rank-1 update (small matrix, 8×8)
 */
static int test_rank1_update_small(void)
{
    printf("\n=== Testing Rank-1 Update (8×8) ===\n");
    
    const uint16_t n = 8;
    const uint16_t k = 1;
    
    float *A = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L_original = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
    
    // Generate SPD matrix
    generate_spd_matrix(A, n, 12345);
    
    // Compute initial Cholesky factorization
    memcpy(L, A, n * n * sizeof(float));
    int ret = cholesky_factorize_simple(L, n, false);
    if (ret != 0)
    {
        printf("  ERROR: Initial factorization failed\n");
        goto cleanup;
    }
    
    memcpy(L_original, L, n * n * sizeof(float));
    
    // Generate update vector
    srand(54321);
    for (uint16_t i = 0; i < n; i++)
    {
        X[i] = ((float)(rand() % 100)) / 50.0f;
    }
    
    printf("  Performing rank-1 update (no workspace)...\n");
    ret = cholupdatek(L, X, n, k, false, +1);
    
    if (ret != 0)
    {
        printf("  ERROR: cholupdatek returned %d\n", ret);
        goto cleanup;
    }
    
    int passed = 1;
    passed &= check_positive_definite(L, n, "rank-1");
    passed &= check_update_correctness(L_original, L, X, n, k, false, +1, 1e-4, "rank-1");
    
cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(L);
    gemm_aligned_free(L_original);
    gemm_aligned_free(X);
    
    return passed;
}

/**
 * @brief Test rank-1 downdate
 */
static int test_rank1_downdate(void)
{
    printf("\n=== Testing Rank-1 Downdate (16×16) ===\n");
    
    const uint16_t n = 16;
    const uint16_t k = 1;
    
    float *A = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L_original = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
    
    // Generate SPD matrix with large diagonal for stable downdate
    generate_spd_matrix(A, n, 99999);
    for (uint16_t i = 0; i < n; i++)
    {
        A[i * n + i] += 100.0f; // Extra diagonal dominance
    }
    
    memcpy(L, A, n * n * sizeof(float));
    int ret = cholesky_factorize_simple(L, n, false);
    if (ret != 0)
    {
        printf("  ERROR: Initial factorization failed\n");
        goto cleanup;
    }
    
    memcpy(L_original, L, n * n * sizeof(float));
    
    // Small update vector (downdate must be small to maintain SPD)
    srand(11111);
    for (uint16_t i = 0; i < n; i++)
    {
        X[i] = ((float)(rand() % 50)) / 100.0f; // Small values
    }
    
    printf("  Performing rank-1 downdate...\n");
    ret = cholupdatek(L, X, n, k, false, -1);
    
    if (ret != 0)
    {
        printf("  ERROR: cholupdatek returned %d\n", ret);
        goto cleanup;
    }
    
    int passed = 1;
    passed &= check_positive_definite(L, n, "rank-1 downdate");
    passed &= check_update_correctness(L_original, L, X, n, k, false, -1, 1e-3, "rank-1 downdate");
    
cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(L);
    gemm_aligned_free(L_original);
    gemm_aligned_free(X);
    
    return passed;
}

/**
 * @brief Test rank-4 update (should use tiled path)
 */
static int test_rank4_tiled(void)
{
    printf("\n=== Testing Rank-4 Update (32×32, tiled path) ===\n");
    
    const uint16_t n = 32;
    const uint16_t k = 4;
    
    float *A = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L_original = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
    
    generate_spd_matrix(A, n, 77777);
    
    memcpy(L, A, n * n * sizeof(float));
    int ret = cholesky_factorize_simple(L, n, false);
    if (ret != 0)
    {
        printf("  ERROR: Initial factorization failed\n");
        goto cleanup;
    }
    
    memcpy(L_original, L, n * n * sizeof(float));
    
    // Generate update matrix
    srand(22222);
    for (uint16_t i = 0; i < n * k; i++)
    {
        X[i] = ((float)(rand() % 100) - 50.0f) / 100.0f;
    }
    
    // Allocate workspace
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
    {
        printf("  ERROR: Workspace allocation failed\n");
        goto cleanup;
    }
    
    printf("  Performing rank-4 update (tiled path expected)...\n");
    ret = cholupdatek_auto_ws(ws, L, X, n, k, false, +1);
    
    if (ret != 0)
    {
        printf("  ERROR: cholupdatek_auto_ws returned %d\n", ret);
        cholupdate_workspace_free(ws);
        goto cleanup;
    }
    
    cholupdate_workspace_free(ws);
    
    int passed = 1;
    passed &= check_positive_definite(L, n, "rank-4 tiled");
    passed &= check_update_correctness(L_original, L, X, n, k, false, +1, 1e-3, "rank-4 tiled");
    
cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(L);
    gemm_aligned_free(L_original);
    gemm_aligned_free(X);
    
    return passed;
}

/**
 * @brief Test rank-16 update (should use QR path)
 */
static int test_rank16_qr(void)
{
    printf("\n=== Testing Rank-16 Update (64×64, QR path) ===\n");
    
    const uint16_t n = 64;
    const uint16_t k = 16;
    
    float *A = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L_original = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
    
    generate_spd_matrix(A, n, 33333);
    
    memcpy(L, A, n * n * sizeof(float));
    int ret = cholesky_factorize_simple(L, n, false);
    if (ret != 0)
    {
        printf("  ERROR: Initial factorization failed\n");
        goto cleanup;
    }
    
    memcpy(L_original, L, n * n * sizeof(float));
    
    // Generate update matrix
    srand(44444);
    for (uint16_t i = 0; i < n * k; i++)
    {
        X[i] = ((float)(rand() % 100) - 50.0f) / 50.0f;
    }
    
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
    {
        printf("  ERROR: Workspace allocation failed\n");
        goto cleanup;
    }
    
    printf("  Performing rank-16 update (QR path expected)...\n");
    ret = cholupdatek_auto_ws(ws, L, X, n, k, false, +1);
    
    if (ret != 0)
    {
        printf("  ERROR: cholupdatek_auto_ws returned %d\n", ret);
        cholupdate_workspace_free(ws);
        goto cleanup;
    }
    
    cholupdate_workspace_free(ws);
    
    int passed = 1;
    passed &= check_positive_definite(L, n, "rank-16 QR");
    passed &= check_update_correctness(L_original, L, X, n, k, false, +1, 1e-3, "rank-16 QR");
    
cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(L);
    gemm_aligned_free(L_original);
    gemm_aligned_free(X);
    
    return passed;
}

/**
 * @brief Test upper triangular storage
 */
static int test_upper_triangular(void)
{
    printf("\n=== Testing Upper Triangular Storage (32×32, k=8) ===\n");
    
    const uint16_t n = 32;
    const uint16_t k = 8;
    
    float *A = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *U = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *U_original = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
    
    generate_spd_matrix(A, n, 55555);
    
    memcpy(U, A, n * n * sizeof(float));
    int ret = cholesky_factorize_simple(U, n, true); // Upper triangular
    if (ret != 0)
    {
        printf("  ERROR: Initial factorization failed\n");
        goto cleanup;
    }
    
    memcpy(U_original, U, n * n * sizeof(float));
    
    srand(66666);
    for (uint16_t i = 0; i < n * k; i++)
    {
        X[i] = ((float)(rand() % 100)) / 50.0f;
    }
    
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
    {
        printf("  ERROR: Workspace allocation failed\n");
        goto cleanup;
    }
    
    printf("  Performing update on upper triangular...\n");
    ret = cholupdatek_auto_ws(ws, U, X, n, k, true, +1); // is_upper=true
    
    if (ret != 0)
    {
        printf("  ERROR: cholupdatek_auto_ws returned %d\n", ret);
        cholupdate_workspace_free(ws);
        goto cleanup;
    }
    
    cholupdate_workspace_free(ws);
    
    int passed = 1;
    passed &= check_positive_definite(U, n, "upper triangular");
    passed &= check_update_correctness(U_original, U, X, n, k, true, +1, 1e-3, "upper triangular");
    
cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(U);
    gemm_aligned_free(U_original);
    gemm_aligned_free(X);
    
    return passed;
}

/**
 * @brief Test workspace reuse
 */
static int test_workspace_reuse(void)
{
    printf("\n=== Testing Workspace Reuse ===\n");
    
    const uint16_t n_max = 64;
    const uint16_t k_max = 32;
    
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n_max, k_max);
    if (!ws)
    {
        printf("  ERROR: Workspace allocation failed\n");
        return 0;
    }
    
    printf("  Workspace size: %.2f KB\n", cholupdate_workspace_bytes(ws) / 1024.0);
    
    int passed = 1;
    
    // Test multiple updates with same workspace
    uint16_t test_configs[][2] = {
        {16, 4},
        {32, 8},
        {48, 16},
        {64, 24}
    };
    
    for (int test_idx = 0; test_idx < 4; test_idx++)
    {
        uint16_t n = test_configs[test_idx][0];
        uint16_t k = test_configs[test_idx][1];
        
        printf("  Testing %d×%d with k=%d...\n", n, n, k);
        
        float *A = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *L_original = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
        
        generate_spd_matrix(A, n, test_idx * 11111);
        memcpy(L, A, n * n * sizeof(float));
        cholesky_factorize_simple(L, n, false);
        memcpy(L_original, L, n * n * sizeof(float));
        
        srand(test_idx * 22222);
        for (uint16_t i = 0; i < n * k; i++)
        {
            X[i] = ((float)(rand() % 100)) / 50.0f;
        }
        
        int ret = cholupdatek_auto_ws(ws, L, X, n, k, false, +1);
        
        if (ret != 0)
        {
            printf("    ERROR: Update failed with code %d\n", ret);
            passed = 0;
        }
        else
        {
            int test_passed = check_update_correctness(L_original, L, X, n, k, 
                                                       false, +1, 1e-3, "workspace reuse");
            passed &= test_passed;
            
            if (test_passed)
            {
                printf("    Test %d PASSED\n", test_idx);
            }
            else
            {
                printf("    Test %d FAILED\n", test_idx);
            }
        }
        
        gemm_aligned_free(A);
        gemm_aligned_free(L);
        gemm_aligned_free(L_original);
        gemm_aligned_free(X);
    }
    
    cholupdate_workspace_free(ws);
    
    return passed;
}

/**
 * @brief Test algorithm selection (tiled vs QR)
 */
static int test_algorithm_selection(void)
{
    printf("\n=== Testing Algorithm Selection ===\n");
    
    const uint16_t n = 64;
    
    int passed = 1;
    
    // Test crossover point (k=8 should be near crossover)
    uint16_t test_k_values[] = {1, 4, 8, 16, 32};
    
    for (int test_idx = 0; test_idx < 5; test_idx++)
    {
        uint16_t k = test_k_values[test_idx];
        
        printf("  Testing with k=%d...\n", k);
        
        float *A = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *L_auto = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *L_tiled = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *L_qr = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
        
        generate_spd_matrix(A, n, test_idx * 99999);
        
        // Compute initial Cholesky factor
        memcpy(L_auto, A, n * n * sizeof(float));
        cholesky_factorize_simple(L_auto, n, false);
        memcpy(L_tiled, L_auto, n * n * sizeof(float));
        memcpy(L_qr, L_auto, n * n * sizeof(float));
        
        srand(test_idx * 88888);
        for (uint16_t i = 0; i < n * k; i++)
        {
            X[i] = ((float)(rand() % 100)) / 50.0f;
        }
        
        cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
        if (!ws)
        {
            printf("    ERROR: Workspace allocation failed\n");
            passed = 0;
            goto cleanup_algo;
        }
        
        // Test auto selection
        int ret_auto = cholupdatek_auto_ws(ws, L_auto, X, n, k, false, +1);
        
        // Test explicit tiled
        int ret_tiled = cholupdatek_ws(ws, L_tiled, X, n, k, false, +1);
        
        // Test explicit QR (skip if k=1, not supported)
        int ret_qr = (k > 1) ? cholupdatek_blockqr_ws(ws, L_qr, X, n, k, false, +1) : 0;
        
        if (ret_auto != 0 || ret_tiled != 0 || (k > 1 && ret_qr != 0))
        {
            printf("    ERROR: One or more methods failed\n");
            passed = 0;
        }
        else
        {
            // Compare results
            double diff_auto_tiled = relative_error(L_auto, L_tiled, n, n);
            double diff_auto_qr = (k > 1) ? relative_error(L_auto, L_qr, n, n) : 0.0;
            
            printf("    ||L_auto - L_tiled||_F / ||L_auto||_F = %.6e\n", diff_auto_tiled);
            if (k > 1)
            {
                printf("    ||L_auto - L_qr||_F / ||L_auto||_F    = %.6e\n", diff_auto_qr);
            }
            
            if (diff_auto_tiled > 1e-4 || (k > 1 && diff_auto_qr > 1e-4))
            {
                printf("    WARNING: Methods produce different results!\n");
                passed = 0;
            }
        }
        
        cholupdate_workspace_free(ws);
        
cleanup_algo:
        gemm_aligned_free(A);
        gemm_aligned_free(L_auto);
        gemm_aligned_free(L_tiled);
        gemm_aligned_free(L_qr);
        gemm_aligned_free(X);
    }
    
    return passed;
}

/**
 * @brief Test edge cases
 */
static int test_edge_cases(void)
{
    printf("\n=== Testing Edge Cases ===\n");
    
    int passed = 1;
    
    // Test 1: k=0 (should be no-op)
    printf("  Testing k=0 (no-op)...\n");
    {
        const uint16_t n = 16;
        float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *L_original = gemm_aligned_alloc(32, n * n * sizeof(float));
        
        // Identity matrix
        memset(L, 0, n * n * sizeof(float));
        for (uint16_t i = 0; i < n; i++)
        {
            L[i * n + i] = 1.0f;
        }
        memcpy(L_original, L, n * n * sizeof(float));
        
        cholupdate_workspace *ws = cholupdate_workspace_alloc(n, 0);
        int ret = cholupdatek_auto_ws(ws, L, NULL, n, 0, false, +1);
        
        if (ret != 0)
        {
            printf("    ERROR: k=0 update failed\n");
            passed = 0;
        }
        else
        {
            double diff = relative_error(L_original, L, n, n);
            if (diff > 1e-10)
            {
                printf("    ERROR: k=0 modified matrix (diff=%.6e)\n", diff);
                passed = 0;
            }
            else
            {
                printf("    k=0 test PASSED\n");
            }
        }
        
        cholupdate_workspace_free(ws);
        gemm_aligned_free(L);
        gemm_aligned_free(L_original);
    }
    
    // Test 2: n=1 (scalar case)
    printf("  Testing n=1 (scalar)...\n");
    {
        const uint16_t n = 1, k = 1;
        float L[1] = {2.0f}; // L*L^T = 4
        float L_orig[1] = {2.0f};
        float X[1] = {3.0f}; // X*X^T = 9
        
        int ret = cholupdatek(&L[0], &X[0], n, k, false, +1);
        
        if (ret != 0)
        {
            printf("    ERROR: Scalar update failed\n");
            passed = 0;
        }
        else
        {
            // Expected: L_new*L_new^T = 4 + 9 = 13, so L_new = sqrt(13)
            float expected = sqrtf(13.0f);
            float diff = fabsf(L[0] - expected);
            
            if (diff > 1e-5)
            {
                printf("    ERROR: Scalar result wrong (got %.6f, expected %.6f)\n",
                       L[0], expected);
                passed = 0;
            }
            else
            {
                printf("    Scalar test PASSED\n");
            }
        }
    }
    
    // Test 3: Downdate that would make matrix indefinite (should fail)
    printf("  Testing downdate failure detection...\n");
    {
        const uint16_t n = 8, k = 1;
        float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
        
        // Very small SPD matrix
        memset(L, 0, n * n * sizeof(float));
        for (uint16_t i = 0; i < n; i++)
        {
            L[i * n + i] = 0.1f; // L*L^T = 0.01*I (very small)
        }
        
        // Large downdate vector
        for (uint16_t i = 0; i < n; i++)
        {
            X[i] = 1.0f; // X*X^T much larger than L*L^T
        }
        
        int ret = cholupdatek(L, X, n, k, false, -1);
        
        if (ret == 0)
        {
            printf("    WARNING: Downdate should have failed but succeeded\n");
            // Check if result is still positive definite
            int still_pd = check_positive_definite(L, n, "downdate edge case");
            if (!still_pd)
            {
                printf("    ERROR: Result is not positive definite\n");
                passed = 0;
            }
        }
        else
        {
            printf("    Downdate correctly failed with code %d\n", ret);
        }
        
        gemm_aligned_free(L);
        gemm_aligned_free(X);
    }
    
    return passed;
}

/**
 * @brief Test large matrix (stress test)
 */
static int test_large_matrix(void)
{
    printf("\n=== Testing Large Matrix (256×256, k=32) ===\n");
    
    const uint16_t n = 256;
    const uint16_t k = 32;
    
    printf("  Allocating %.2f MB...\n",
           (n * n * 3 + n * k) * sizeof(float) / (1024.0 * 1024.0));
    
    float *A = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *L_original = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * k * sizeof(float));
    
    if (!A || !L || !L_original || !X)
    {
        printf("  ERROR: Allocation failed\n");
        gemm_aligned_free(A);
        gemm_aligned_free(L);
        gemm_aligned_free(L_original);
        gemm_aligned_free(X);
        return 0;
    }
    
    printf("  Generating SPD matrix...\n");
    generate_spd_matrix(A, n, 123456);
    
    printf("  Computing initial Cholesky factorization...\n");
    memcpy(L, A, n * n * sizeof(float));
    int ret = cholesky_factorize_simple(L, n, false);
    if (ret != 0)
    {
        printf("  ERROR: Initial factorization failed\n");
        goto cleanup;
    }
    
    memcpy(L_original, L, n * n * sizeof(float));
    
    printf("  Generating rank-32 update...\n");
    srand(654321);
    for (uint32_t i = 0; i < (uint32_t)n * k; i++)
    {
        X[i] = ((float)(rand() % 100)) / 50.0f;
    }
    
    cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
    if (!ws)
    {
        printf("  ERROR: Workspace allocation failed\n");
        goto cleanup;
    }
    
    printf("  Performing rank-32 update (QR path)...\n");
    ret = cholupdatek_auto_ws(ws, L, X, n, k, false, +1);
    
    if (ret != 0)
    {
        printf("  ERROR: cholupdatek_auto_ws returned %d\n", ret);
        cholupdate_workspace_free(ws);
        goto cleanup;
    }
    
    cholupdate_workspace_free(ws);
    
    printf("  Verifying results...\n");
    int passed = 1;
    
    // Use relaxed tolerances for large matrices
    passed &= check_positive_definite(L, n, "large matrix");
    passed &= check_update_correctness(L_original, L, X, n, k, false, +1, 5e-3, "large matrix");
    
cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(L);
    gemm_aligned_free(L_original);
    gemm_aligned_free(X);
    
    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_cholupdate_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("    CHOLESKY RANK-K UPDATE TESTS\n");
    printf("=================================================\n");
    
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    // Run all tests
    printf("\n--- Basic Functionality Tests ---\n");
    
    results->total++;
    if (test_rank1_update_small())
    {
        results->passed++;
        printf("✓ Rank-1 update test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Rank-1 update test FAILED\n");
    }
    
    results->total++;
    if (test_rank1_downdate())
    {
        results->passed++;
        printf("✓ Rank-1 downdate test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Rank-1 downdate test FAILED\n");
    }
    
    results->total++;
    if (test_rank4_tiled())
    {
        results->passed++;
        printf("✓ Rank-4 tiled test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Rank-4 tiled test FAILED\n");
    }
    
    results->total++;
    if (test_rank16_qr())
    {
        results->passed++;
        printf("✓ Rank-16 QR test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Rank-16 QR test FAILED\n");
    }
    
    results->total++;
    if (test_upper_triangular())
    {
        results->passed++;
        printf("✓ Upper triangular test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Upper triangular test FAILED\n");
    }
    
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
    
    printf("\n--- Algorithm Selection Tests ---\n");
    
    results->total++;
    if (test_algorithm_selection())
    {
        results->passed++;
        printf("✓ Algorithm selection test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Algorithm selection test FAILED\n");
    }
    
    printf("\n--- Edge Case Tests ---\n");
    
    results->total++;
    if (test_edge_cases())
    {
        results->passed++;
        printf("✓ Edge cases PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Edge cases FAILED\n");
    }
    
    printf("\n--- Stress Tests ---\n");
    
    results->total++;
    if (test_large_matrix())
    {
        results->passed++;
        printf("✓ Large matrix test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Large matrix test FAILED\n");
    }
    
    printf("\n=================================================\n");
    printf("Cholesky Update Tests: %d/%d passed\n", results->passed, results->total);
    
    if (results->passed == results->total)
    {
        printf("✓ ALL CHOLESKY UPDATE TESTS PASSED!\n");
    }
    else
    {
        printf("✗ %d Cholesky update tests FAILED\n", results->failed);
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
    return run_cholupdate_tests(&results);
}
#endif