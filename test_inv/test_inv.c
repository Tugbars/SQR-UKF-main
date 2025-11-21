/**
 * @file test_inv.c
 * @brief Unit tests for matrix inversion module (inv_blas3_gemm)
 *
 * Tests:
 * 1. Basic inversion (small matrices)
 * 2. Numerical accuracy (verify A*inv(A) = I)
 * 3. Special matrices (identity, diagonal, SPD, etc.)
 * 4. Error handling (singular, invalid inputs)
 * 5. Various matrix sizes
 * 6. Edge cases and robustness
 * 7. In-place inversion
 * 8. Performance characteristics
 *
 * @author TUGBARS
 * @date 2025
 */

#include "inv_blas3_gemm.h"
#include "test_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>

// Portable size_t printf format
#define FMT_SIZE_T "%lu"
#define CAST_SIZE_T(x) ((unsigned long)(x))

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Compute maximum absolute error |A*Ainv - I|_∞
 */
static double compute_inverse_error(const float *A, const float *Ainv, uint16_t n)
{
    double max_err = 0.0;
    
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            // Compute (A*Ainv)[i,j]
            double sum = 0.0;
            for (uint16_t k = 0; k < n; k++)
            {
                sum += (double)A[i*n + k] * (double)Ainv[k*n + j];
            }
            
            // Compare to identity
            double expected = (i == j) ? 1.0 : 0.0;
            double err = fabs(sum - expected);
            
            if (err > max_err)
                max_err = err;
        }
    }
    
    return max_err;
}

/**
 * @brief Create symmetric positive definite matrix (always invertible)
 */
static void create_spd_matrix(float *A, uint16_t n)
{
    // A = B^T * B + n*I (guaranteed SPD)
    float *B = malloc(n * n * sizeof(float));
    
    for (uint16_t i = 0; i < n * n; i++)
    {
        B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // A = B^T * B
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (uint16_t k = 0; k < n; k++)
            {
                sum += B[k*n + i] * B[k*n + j];
            }
            A[i*n + j] = sum;
        }
    }
    
    // Add diagonal dominance: A += n*I
    for (uint16_t i = 0; i < n; i++)
    {
        A[i*n + i] += (float)n;
    }
    
    free(B);
}

/**
 * @brief Compute Frobenius norm ||A||_F
 */
static double frobenius_norm(const float *A, uint16_t n)
{
    double sum = 0.0;
    for (uint16_t i = 0; i < n * n; i++)
    {
        sum += (double)A[i] * (double)A[i];
    }
    return sqrt(sum);
}

//==============================================================================
// TEST 1: Basic Inversion (Small Matrices)
//==============================================================================

static int test_inv_2x2_simple(void)
{
    printf("  Testing: 2×2 matrix inversion\n");
    
    float A[4] = {
        2.0f, 1.0f,
        1.0f, 3.0f
    };
    float Ainv[4];
    
    int rc = inv(Ainv, A, 2);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d (expected 0)\n", rc);
        return 0;
    }
    
    // Verify numerical accuracy
    double err = compute_inverse_error(A, Ainv, 2);
    printf("    Max error: %.2e (threshold: 1e-5)\n", err);
    
    if (err > 1e-5)
    {
        printf("    FAIL: Error too large\n");
        return 0;
    }
    
    return 1;
}

static int test_inv_3x3_simple(void)
{
    printf("  Testing: 3×3 matrix inversion\n");
    
    float A[9] = {
        4.0f, 3.0f, 2.0f,
        3.0f, 4.0f, 1.0f,
        2.0f, 1.0f, 5.0f
    };
    float Ainv[9];
    
    int rc = inv(Ainv, A, 3);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, 3);
    printf("    Max error: %.2e\n", err);
    
    if (err > 1e-5)
    {
        printf("    FAIL: Error too large\n");
        return 0;
    }
    
    return 1;
}

static int test_inv_4x4_random(void)
{
    printf("  Testing: 4×4 random SPD matrix\n");
    
    float A[16];
    float Ainv[16];
    
    create_spd_matrix(A, 4);
    
    int rc = inv(Ainv, A, 4);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, 4);
    printf("    Max error: %.2e\n", err);
    
    if (err > 1e-4)
    {
        printf("    FAIL: Error too large for 4×4\n");
        return 0;
    }
    
    return 1;
}

//==============================================================================
// TEST 2: Special Matrices
//==============================================================================

static int test_inv_identity(void)
{
    printf("  Testing: Identity matrix (inv(I) = I)\n");
    
    uint16_t n = 8;
    float *I = calloc(n * n, sizeof(float));
    float *Iinv = malloc(n * n * sizeof(float));
    
    // Create identity
    for (uint16_t i = 0; i < n; i++)
    {
        I[i*n + i] = 1.0f;
    }
    
    int rc = inv(Iinv, I, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(I);
        free(Iinv);
        return 0;
    }
    
    // Verify inv(I) = I
    double max_diff = 0.0;
    for (uint16_t i = 0; i < n * n; i++)
    {
        double diff = fabs((double)Iinv[i] - (double)I[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    
    printf("    Max difference: %.2e\n", max_diff);
    
    free(I);
    free(Iinv);
    
    return (max_diff < 1e-6);
}

static int test_inv_diagonal(void)
{
    printf("  Testing: Diagonal matrix\n");
    
    uint16_t n = 10;
    float *D = calloc(n * n, sizeof(float));
    float *Dinv = malloc(n * n * sizeof(float));
    
    // Create diagonal matrix with entries 1, 2, 3, ..., n
    for (uint16_t i = 0; i < n; i++)
    {
        D[i*n + i] = (float)(i + 1);
    }
    
    int rc = inv(Dinv, D, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(D);
        free(Dinv);
        return 0;
    }
    
    // Verify diagonal entries are reciprocals
    int correct = 1;
    for (uint16_t i = 0; i < n; i++)
    {
        float expected = 1.0f / (float)(i + 1);
        float diff = fabsf(Dinv[i*n + i] - expected);
        
        if (diff > 1e-6f)
        {
            printf("    FAIL: Dinv[%d,%d] = %f, expected %f\n",
                   i, i, Dinv[i*n + i], expected);
            correct = 0;
            break;
        }
    }
    
    if (correct)
        printf("    All diagonal elements correct\n");
    
    free(D);
    free(Dinv);
    
    return correct;
}

static int test_inv_symmetric(void)
{
    printf("  Testing: Symmetric matrix preservation\n");
    
    uint16_t n = 12;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    
    // Create symmetric SPD matrix
    create_spd_matrix(A, n);
    
    int rc = inv(Ainv, A, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(Ainv);
        return 0;
    }
    
    // Verify inv(A) is also symmetric
    double max_asym = 0.0;
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = i + 1; j < n; j++)
        {
            double diff = fabs((double)Ainv[i*n + j] - (double)Ainv[j*n + i]);
            if (diff > max_asym)
                max_asym = diff;
        }
    }
    
    printf("    Max asymmetry: %.2e\n", max_asym);
    
    free(A);
    free(Ainv);
    
    return (max_asym < 1e-4);
}

static int test_inv_lower_triangular(void)
{
    printf("  Testing: Lower triangular matrix\n");
    
    uint16_t n = 8;
    float *L = calloc(n * n, sizeof(float));
    float *Linv = malloc(n * n * sizeof(float));
    
    // Create lower triangular with unit diagonal
    for (uint16_t i = 0; i < n; i++)
    {
        L[i*n + i] = 2.0f; // Non-unit diagonal
        for (uint16_t j = 0; j < i; j++)
        {
            L[i*n + j] = ((float)rand() / RAND_MAX) * 0.1f;
        }
    }
    
    int rc = inv(Linv, L, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(L);
        free(Linv);
        return 0;
    }
    
    double err = compute_inverse_error(L, Linv, n);
    printf("    Max error: %.2e\n", err);
    
    free(L);
    free(Linv);
    
    return (err < 1e-4);
}

//==============================================================================
// TEST 3: Error Handling
//==============================================================================

static int test_inv_singular_exact(void)
{
    printf("  Testing: Exactly singular matrix (rank deficient)\n");
    
    float A[9] = {
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,  // Row 2 = 2 * Row 1
        4.0f, 5.0f, 6.0f
    };
    float Ainv[9];
    
    int rc = inv(Ainv, A, 3);
    
    if (rc == 0)
    {
        printf("    FAIL: Should have detected singularity\n");
        return 0;
    }
    
    if (rc != -ENOTSUP)
    {
        printf("    FAIL: Expected -ENOTSUP, got %d\n", rc);
        return 0;
    }
    
    printf("    Correctly detected singularity (rc=%d)\n", rc);
    return 1;
}

static int test_inv_singular_near(void)
{
    printf("  Testing: Nearly singular matrix (small pivot)\n");
    
    float A[9] = {
        1e-8f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    float Ainv[9];
    
    int rc = inv(Ainv, A, 3);
    
    if (rc == 0)
    {
        printf("    FAIL: Should have detected near-singularity\n");
        return 0;
    }
    
    printf("    Correctly detected near-singularity (rc=%d)\n", rc);
    return 1;
}

static int test_inv_zero_dimension(void)
{
    printf("  Testing: Zero dimension (n=0)\n");
    
    float *A = NULL;
    float *Ainv = NULL;
    
    int rc = inv(Ainv, A, 0);
    
    if (rc == 0)
    {
        printf("    FAIL: Should have rejected n=0\n");
        return 0;
    }
    
    if (rc != -EINVAL)
    {
        printf("    FAIL: Expected -EINVAL, got %d\n", rc);
        return 0;
    }
    
    printf("    Correctly rejected n=0 (rc=%d)\n", rc);
    return 1;
}

static int test_inv_ill_conditioned(void)
{
    printf("  Testing: Ill-conditioned matrix (large condition number)\n");
    
    // Hilbert matrix H[i,j] = 1/(i+j+1) - notoriously ill-conditioned
    uint16_t n = 6;
    float *H = malloc(n * n * sizeof(float));
    float *Hinv = malloc(n * n * sizeof(float));
    
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            H[i*n + j] = 1.0f / (float)(i + j + 1);
        }
    }
    
    int rc = inv(Hinv, H, n);
    
    if (rc != 0)
    {
        printf("    WARN: Failed to invert Hilbert matrix (expected, high κ)\n");
        printf("    Return code: %d\n", rc);
        free(H);
        free(Hinv);
        return 1; // Not a failure - ill-conditioned is expected to fail
    }
    
    // If it succeeded, check if error is reasonable
    double err = compute_inverse_error(H, Hinv, n);
    printf("    Max error: %.2e (ill-conditioned, high error expected)\n", err);
    
    free(H);
    free(Hinv);
    
    return 1; // Pass either way - this is a robustness test
}

//==============================================================================
// TEST 4: Various Matrix Sizes
//==============================================================================

static int test_inv_size_16(void)
{
    printf("  Testing: 16×16 random SPD matrix\n");
    
    uint16_t n = 16;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    
    int rc = inv(Ainv, A, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(Ainv);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, n);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 1e-3);
    
    free(A);
    free(Ainv);
    
    return pass;
}

static int test_inv_size_64(void)
{
    printf("  Testing: 64×64 random SPD matrix\n");
    
    uint16_t n = 64;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    
    int rc = inv(Ainv, A, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(Ainv);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, n);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 1e-3);
    
    free(A);
    free(Ainv);
    
    return pass;
}

static int test_inv_size_128(void)
{
    printf("  Testing: 128×128 random SPD matrix\n");
    
    uint16_t n = 128;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    
    int rc = inv(Ainv, A, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(Ainv);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, n);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 5e-3);
    
    free(A);
    free(Ainv);
    
    return pass;
}

static int test_inv_size_256(void)
{
    printf("  Testing: 256×256 random SPD matrix\n");
    
    uint16_t n = 256;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    
    clock_t start = clock();
    int rc = inv(Ainv, A, n);
    clock_t end = clock();
    
    double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(Ainv);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, n);
    double gflops = (2.0/3.0 * n * n * n) / (time_ms / 1000.0) / 1e9;
    
    printf("    Time: %.2f ms (%.1f GFLOPS)\n", time_ms, gflops);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 1e-2);
    
    free(A);
    free(Ainv);
    
    return pass;
}

static int test_inv_size_512(void)
{
    printf("  Testing: 512×512 random SPD matrix (stress test)\n");
    
    uint16_t n = 512;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    
    if (!A || !Ainv)
    {
        printf("    SKIP: Memory allocation failed\n");
        free(A);
        free(Ainv);
        return 1; // Not a test failure
    }
    
    create_spd_matrix(A, n);
    
    clock_t start = clock();
    int rc = inv(Ainv, A, n);
    clock_t end = clock();
    
    double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(Ainv);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, n);
    double gflops = (2.0/3.0 * n * n * n) / (time_ms / 1000.0) / 1e9;
    
    printf("    Time: %.2f ms (%.1f GFLOPS)\n", time_ms, gflops);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 2e-2);
    
    free(A);
    free(Ainv);
    
    return pass;
}

//==============================================================================
// TEST 5: In-Place Inversion
//==============================================================================

static int test_inv_inplace_small(void)
{
    printf("  Testing: In-place inversion (4×4)\n");
    
    uint16_t n = 4;
    float *A = malloc(n * n * sizeof(float));
    float *A_copy = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    memcpy(A_copy, A, n * n * sizeof(float));
    
    // In-place: A = inv(A)
    int rc = inv(A, A, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(A_copy);
        return 0;
    }
    
    // Verify correctness using copy
    double err = compute_inverse_error(A_copy, A, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(A_copy);
    
    return (err < 1e-4);
}

static int test_inv_inplace_large(void)
{
    printf("  Testing: In-place inversion (128×128)\n");
    
    uint16_t n = 128;
    float *A = malloc(n * n * sizeof(float));
    float *A_copy = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    memcpy(A_copy, A, n * n * sizeof(float));
    
    int rc = inv(A, A, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(A_copy);
        return 0;
    }
    
    double err = compute_inverse_error(A_copy, A, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(A_copy);
    
    return (err < 5e-3);
}

//==============================================================================
// TEST 6: Edge Cases
//==============================================================================

static int test_inv_single_element(void)
{
    printf("  Testing: 1×1 matrix (scalar inverse)\n");
    
    float A[1] = {5.0f};
    float Ainv[1];
    
    int rc = inv(Ainv, A, 1);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        return 0;
    }
    
    float expected = 0.2f; // 1/5
    float err = fabsf(Ainv[0] - expected);
    
    printf("    Result: %.6f (expected %.6f)\n", Ainv[0], expected);
    printf("    Error: %.2e\n", err);
    
    return (err < 1e-6f);
}

static int test_inv_odd_size(void)
{
    printf("  Testing: Odd dimension (127×127)\n");
    
    uint16_t n = 127;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    
    int rc = inv(Ainv, A, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(Ainv);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(Ainv);
    
    return (err < 1e-2);
}

static int test_inv_prime_size(void)
{
    printf("  Testing: Prime dimension (101×101)\n");
    
    uint16_t n = 101;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    
    int rc = inv(Ainv, A, n);
    
    if (rc != 0)
    {
        printf("    FAIL: inv() returned %d\n", rc);
        free(A);
        free(Ainv);
        return 0;
    }
    
    double err = compute_inverse_error(A, Ainv, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(Ainv);
    
    return (err < 1e-2);
}

//==============================================================================
// TEST 7: Numerical Stability
//==============================================================================

static int test_inv_double_inverse(void)
{
    printf("  Testing: Double inversion (inv(inv(A)) = A)\n");
    
    uint16_t n = 32;
    float *A = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    float *A2 = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    
    // First inversion
    int rc1 = inv(Ainv, A, n);
    if (rc1 != 0)
    {
        printf("    FAIL: First inv() returned %d\n", rc1);
        free(A);
        free(Ainv);
        free(A2);
        return 0;
    }
    
    // Second inversion
    int rc2 = inv(A2, Ainv, n);
    if (rc2 != 0)
    {
        printf("    FAIL: Second inv() returned %d\n", rc2);
        free(A);
        free(Ainv);
        free(A2);
        return 0;
    }
    
    // Compare A2 to original A
    double max_diff = 0.0;
    for (uint16_t i = 0; i < n * n; i++)
    {
        double diff = fabs((double)A2[i] - (double)A[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    
    printf("    Max difference: %.2e\n", max_diff);
    
    free(A);
    free(Ainv);
    free(A2);
    
    return (max_diff < 1e-2);
}

static int test_inv_scaled_matrix(void)
{
    printf("  Testing: Scaled matrix (A vs 10*A)\n");
    
    uint16_t n = 16;
    float *A = malloc(n * n * sizeof(float));
    float *A_scaled = malloc(n * n * sizeof(float));
    float *Ainv = malloc(n * n * sizeof(float));
    float *Ainv_scaled = malloc(n * n * sizeof(float));
    
    create_spd_matrix(A, n);
    
    // Create 10*A
    for (uint16_t i = 0; i < n * n; i++)
    {
        A_scaled[i] = 10.0f * A[i];
    }
    
    inv(Ainv, A, n);
    inv(Ainv_scaled, A_scaled, n);
    
    // inv(10*A) should equal (1/10)*inv(A)
    double max_diff = 0.0;
    for (uint16_t i = 0; i < n * n; i++)
    {
        double expected = 0.1 * Ainv[i];
        double diff = fabs((double)Ainv_scaled[i] - expected);
        if (diff > max_diff)
            max_diff = diff;
    }
    
    printf("    Max scaling error: %.2e\n", max_diff);
    
    free(A);
    free(A_scaled);
    free(Ainv);
    free(Ainv_scaled);
    
    return (max_diff < 1e-4);
}

//==============================================================================
// TEST SUITE RUNNER
//==============================================================================

int run_inv_tests(test_results_t *results)
{
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    srand(12345); // Reproducible random tests
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  Matrix Inversion Module - Comprehensive Test Suite      ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Group 1: Basic Inversion
    printf("\n═══ Test Group 1: Basic Inversion (Small Matrices) ═══\n");
    RUN_TEST(results, test_inv_2x2_simple);
    RUN_TEST(results, test_inv_3x3_simple);
    RUN_TEST(results, test_inv_4x4_random);
    
    // Group 2: Special Matrices
    printf("\n═══ Test Group 2: Special Matrices ═══\n");
    RUN_TEST(results, test_inv_identity);
    RUN_TEST(results, test_inv_diagonal);
    RUN_TEST(results, test_inv_symmetric);
    RUN_TEST(results, test_inv_lower_triangular);
    
    // Group 3: Error Handling
    printf("\n═══ Test Group 3: Error Handling ═══\n");
    RUN_TEST(results, test_inv_singular_exact);
    RUN_TEST(results, test_inv_singular_near);
    RUN_TEST(results, test_inv_zero_dimension);
    RUN_TEST(results, test_inv_ill_conditioned);
    
    // Group 4: Various Sizes
    printf("\n═══ Test Group 4: Various Matrix Sizes ═══\n");
    RUN_TEST(results, test_inv_size_16);
    RUN_TEST(results, test_inv_size_64);
    RUN_TEST(results, test_inv_size_128);
    RUN_TEST(results, test_inv_size_256);
    RUN_TEST(results, test_inv_size_512);
    
    // Group 5: In-Place Inversion
    printf("\n═══ Test Group 5: In-Place Inversion ═══\n");
    RUN_TEST(results, test_inv_inplace_small);
    RUN_TEST(results, test_inv_inplace_large);
    
    // Group 6: Edge Cases
    printf("\n═══ Test Group 6: Edge Cases ═══\n");
    RUN_TEST(results, test_inv_single_element);
    RUN_TEST(results, test_inv_odd_size);
    RUN_TEST(results, test_inv_prime_size);
    
    // Group 7: Numerical Stability
    printf("\n═══ Test Group 7: Numerical Stability ═══\n");
    RUN_TEST(results, test_inv_double_inverse);
    RUN_TEST(results, test_inv_scaled_matrix);
    
    print_test_results("Matrix Inversion Module", results);
    
    return (results->failed == 0) ? 0 : 1;
}

#ifdef STANDALONE
int main(void)
{
    test_results_t results;
    int ret = run_inv_tests(&results);
    
    if (ret == 0)
    {
        printf("\n🎉 " TEST_PASS " All tests passed!\n\n");
    }
    else
    {
        printf("\n❌ " TEST_FAIL " %d test(s) failed\n\n", results.failed);
    }
    
    return ret;
}
#endif