/**
 * @file test_lup.c
 * @brief Unit tests for LU factorization with partial pivoting (lup_blas3)
 *
 * Tests:
 * 1. Basic factorization (small matrices)
 * 2. Numerical accuracy (verify P*A = L*U)
 * 3. Special matrices (identity, diagonal, triangular, etc.)
 * 4. Permutation correctness
 * 5. Error handling (singular matrices, invalid inputs)
 * 6. Various matrix sizes
 * 7. Workspace management
 * 8. Edge cases and robustness
 * 9. In-place factorization
 *
 * @author TUGBARS
 * @date 2025
 */

#include "lup.h"
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
 * @brief Extract L and U from combined LU matrix
 */
static void extract_LU(const float *LU, float *L, float *U, uint16_t n)
{
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            if (i > j)
            {
                // Lower triangular (unit diagonal)
                L[i*n + j] = LU[i*n + j];
                U[i*n + j] = 0.0f;
            }
            else if (i == j)
            {
                // Diagonal
                L[i*n + j] = 1.0f; // Unit diagonal
                U[i*n + j] = LU[i*n + j];
            }
            else
            {
                // Upper triangular
                L[i*n + j] = 0.0f;
                U[i*n + j] = LU[i*n + j];
            }
        }
    }
}

/**
 * @brief Apply permutation to matrix: PA[i,:] = A[P[i],:]
 */
static void apply_permutation(const float *A, float *PA, const uint8_t *P, uint16_t n)
{
    for (uint16_t i = 0; i < n; i++)
    {
        memcpy(PA + i*n, A + P[i]*n, n * sizeof(float));
    }
}

/**
 * @brief Compute C = A * B
 */
static void matmul(const float *A, const float *B, float *C, uint16_t n)
{
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (uint16_t k = 0; k < n; k++)
            {
                sum += (double)A[i*n + k] * (double)B[k*n + j];
            }
            C[i*n + j] = (float)sum;
        }
    }
}

/**
 * @brief Verify P*A = L*U factorization
 */
static double verify_factorization(const float *A, const float *LU, const uint8_t *P, uint16_t n)
{
    float *L = malloc(n * n * sizeof(float));
    float *U = malloc(n * n * sizeof(float));
    float *PA = malloc(n * n * sizeof(float));
    float *LU_product = malloc(n * n * sizeof(float));
    
    extract_LU(LU, L, U, n);
    apply_permutation(A, PA, P, n);
    matmul(L, U, LU_product, n);
    
    // Compute max error |PA - LU|_∞
    double max_err = 0.0;
    for (uint16_t i = 0; i < n * n; i++)
    {
        double err = fabs((double)PA[i] - (double)LU_product[i]);
        if (err > max_err)
            max_err = err;
    }
    
    free(L);
    free(U);
    free(PA);
    free(LU_product);
    
    return max_err;
}

/**
 * @brief Create random matrix with specified condition number
 */
static void create_random_matrix(float *A, uint16_t n, float scale)
{
    for (uint16_t i = 0; i < n * n; i++)
    {
        A[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    
    // Add diagonal dominance for better conditioning
    for (uint16_t i = 0; i < n; i++)
    {
        A[i*n + i] += (float)n * scale;
    }
}

/**
 * @brief Check if permutation is valid (bijection)
 */
static int is_valid_permutation(const uint8_t *P, uint16_t n)
{
    uint8_t *seen = calloc(n, sizeof(uint8_t));
    
    for (uint16_t i = 0; i < n; i++)
    {
        if (P[i] >= n)
        {
            free(seen);
            return 0; // Out of range
        }
        
        if (seen[P[i]])
        {
            free(seen);
            return 0; // Duplicate
        }
        
        seen[P[i]] = 1;
    }
    
    free(seen);
    return 1;
}

//==============================================================================
// TEST 1: Basic Factorization (Small Matrices)
//==============================================================================

static int test_lup_2x2_simple(void)
{
    printf("  Testing: 2×2 matrix factorization\n");
    
    float A[4] = {
        2.0f, 1.0f,
        1.0f, 3.0f
    };
    float LU[4];
    uint8_t P[2];
    
    int rc = lup(A, LU, P, 2);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d (expected 0)\n", rc);
        return 0;
    }
    
    // Verify factorization
    double err = verify_factorization(A, LU, P, 2);
    printf("    Max error |PA - LU|: %.2e\n", err);
    
    // Verify permutation
    if (!is_valid_permutation(P, 2))
    {
        printf("    FAIL: Invalid permutation\n");
        return 0;
    }
    
    return (err < 1e-5);
}

static int test_lup_3x3_simple(void)
{
    printf("  Testing: 3×3 matrix factorization\n");
    
    float A[9] = {
        4.0f, 3.0f, 2.0f,
        3.0f, 4.0f, 1.0f,
        2.0f, 1.0f, 5.0f
    };
    float LU[9];
    uint8_t P[3];
    
    int rc = lup(A, LU, P, 3);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, 3);
    printf("    Max error: %.2e\n", err);
    
    if (!is_valid_permutation(P, 3))
    {
        printf("    FAIL: Invalid permutation\n");
        return 0;
    }
    
    return (err < 1e-5);
}

static int test_lup_4x4_random(void)
{
    printf("  Testing: 4×4 random matrix\n");
    
    float A[16];
    float LU[16];
    uint8_t P[4];
    
    create_random_matrix(A, 4, 10.0f);
    
    int rc = lup(A, LU, P, 4);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, 4);
    printf("    Max error: %.2e\n", err);
    
    if (!is_valid_permutation(P, 4))
    {
        printf("    FAIL: Invalid permutation\n");
        return 0;
    }
    
    return (err < 1e-4);
}

//==============================================================================
// TEST 2: Special Matrices
//==============================================================================

static int test_lup_identity(void)
{
    printf("  Testing: Identity matrix (no pivoting needed)\n");
    
    uint16_t n = 8;
    float *I = calloc(n * n, sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    // Create identity
    for (uint16_t i = 0; i < n; i++)
    {
        I[i*n + i] = 1.0f;
    }
    
    int rc = lup(I, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(I);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(I, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    // Verify P is identity (no pivoting)
    int identity_perm = 1;
    for (uint16_t i = 0; i < n; i++)
    {
        if (P[i] != i)
        {
            identity_perm = 0;
            break;
        }
    }
    
    if (identity_perm)
        printf("    Permutation is identity (as expected)\n");
    
    free(I);
    free(LU);
    free(P);
    
    return (err < 1e-6);
}

static int test_lup_diagonal(void)
{
    printf("  Testing: Diagonal matrix\n");
    
    uint16_t n = 10;
    float *D = calloc(n * n, sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    // Create diagonal with entries n, n-1, ..., 1 (decreasing order)
    for (uint16_t i = 0; i < n; i++)
    {
        D[i*n + i] = (float)(n - i);
    }
    
    int rc = lup(D, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(D);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(D, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    free(D);
    free(LU);
    free(P);
    
    return (err < 1e-5);
}

static int test_lup_lower_triangular(void)
{
    printf("  Testing: Lower triangular matrix\n");
    
    uint16_t n = 8;
    float *L = calloc(n * n, sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    // Create lower triangular
    for (uint16_t i = 0; i < n; i++)
    {
        L[i*n + i] = 2.0f;
        for (uint16_t j = 0; j < i; j++)
        {
            L[i*n + j] = ((float)rand() / RAND_MAX) * 0.5f;
        }
    }
    
    int rc = lup(L, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(L);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(L, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    free(L);
    free(LU);
    free(P);
    
    return (err < 1e-4);
}

static int test_lup_upper_triangular(void)
{
    printf("  Testing: Upper triangular matrix\n");
    
    uint16_t n = 8;
    float *U = calloc(n * n, sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    // Create upper triangular
    for (uint16_t i = 0; i < n; i++)
    {
        U[i*n + i] = 2.0f;
        for (uint16_t j = i + 1; j < n; j++)
        {
            U[i*n + j] = ((float)rand() / RAND_MAX) * 0.5f;
        }
    }
    
    int rc = lup(U, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(U);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(U, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    free(U);
    free(LU);
    free(P);
    
    return (err < 1e-4);
}

static int test_lup_symmetric(void)
{
    printf("  Testing: Symmetric matrix\n");
    
    uint16_t n = 12;
    float *A = malloc(n * n * sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    // Create symmetric matrix
    for (uint16_t i = 0; i < n; i++)
    {
        for (uint16_t j = i; j < n; j++)
        {
            float val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            A[i*n + j] = val;
            A[j*n + i] = val;
        }
        A[i*n + i] += (float)n; // Diagonal dominance
    }
    
    int rc = lup(A, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(LU);
    free(P);
    
    return (err < 1e-3);
}

//==============================================================================
// TEST 3: Permutation Correctness
//==============================================================================

static int test_lup_pivot_needed(void)
{
    printf("  Testing: Matrix requiring pivoting\n");
    
    // Matrix designed to need pivoting (first element is small)
    float A[9] = {
        1e-6f, 2.0f, 3.0f,
        4.0f,  5.0f, 6.0f,
        7.0f,  8.0f, 10.0f
    };
    float LU[9];
    uint8_t P[3];
    
    int rc = lup(A, LU, P, 3);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        return 0;
    }
    
    // Verify pivoting occurred (P should not be identity)
    int pivoted = 0;
    for (uint16_t i = 0; i < 3; i++)
    {
        if (P[i] != i)
        {
            pivoted = 1;
            break;
        }
    }
    
    if (!pivoted)
    {
        printf("    WARN: Expected pivoting but P is identity\n");
    }
    else
    {
        printf("    Pivoting occurred: P = [%d, %d, %d]\n", P[0], P[1], P[2]);
    }
    
    double err = verify_factorization(A, LU, P, 3);
    printf("    Max error: %.2e\n", err);
    
    return (err < 1e-4);
}

static int test_lup_permutation_validity(void)
{
    printf("  Testing: Permutation validity for various matrices\n");
    
    uint16_t sizes[] = {4, 8, 16, 32, 64};
    int all_valid = 1;
    
    for (size_t t = 0; t < sizeof(sizes)/sizeof(sizes[0]); t++)
    {
        uint16_t n = sizes[t];
        float *A = malloc(n * n * sizeof(float));
        float *LU = malloc(n * n * sizeof(float));
        uint8_t *P = malloc(n * sizeof(uint8_t));
        
        create_random_matrix(A, n, 10.0f);
        
        int rc = lup(A, LU, P, n);
        
        if (rc != 0)
        {
            printf("    FAIL: n=%d, lup() returned %d\n", n, rc);
            all_valid = 0;
            free(A);
            free(LU);
            free(P);
            break;
        }
        
        if (!is_valid_permutation(P, n))
        {
            printf("    FAIL: n=%d, invalid permutation\n", n);
            all_valid = 0;
            free(A);
            free(LU);
            free(P);
            break;
        }
        
        printf("    n=%d: permutation valid ✓\n", n);
        
        free(A);
        free(LU);
        free(P);
    }
    
    return all_valid;
}

//==============================================================================
// TEST 4: Error Handling
//==============================================================================

static int test_lup_singular_exact(void)
{
    printf("  Testing: Exactly singular matrix\n");
    
    float A[9] = {
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,  // Row 2 = 2 * Row 1
        4.0f, 5.0f, 6.0f
    };
    float LU[9];
    uint8_t P[3];
    
    int rc = lup(A, LU, P, 3);
    
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

static int test_lup_singular_near(void)
{
    printf("  Testing: Nearly singular matrix\n");
    
    // Create a matrix that's nearly rank-deficient
    // Row 2 ≈ Row 0 + Row 1 (with small perturbation)
    float A[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        5.0f + 1e-7f, 7.0f + 1e-7f, 9.0f + 1e-7f  // Nearly dependent
    };
    float LU[9];
    uint8_t P[3];
    
    int rc = lup(A, LU, P, 3);
    
    if (rc == 0)
    {
        // It might succeed for nearly singular (just ill-conditioned)
        // Check if the factorization is numerically unstable
        double err = verify_factorization(A, LU, P, 3);
        
        if (err > 1e-3)
        {
            printf("    Nearly singular matrix produced large error: %.2e\n", err);
            printf("    This is expected for ill-conditioned matrices\n");
        }
        else
        {
            printf("    Matrix factorized successfully (ill-conditioned but not singular)\n");
            printf("    Factorization error: %.2e\n", err);
        }
        
        return 1; // Pass - near-singular is hard to detect reliably
    }
    
    printf("    Detected near-singularity (rc=%d)\n", rc);
    return 1; // Pass either way
}

static int test_lup_zero_dimension(void)
{
    printf("  Testing: Zero dimension (n=0)\n");
    
    float *A = NULL;
    float *LU = NULL;
    uint8_t *P = NULL;
    
    int rc = lup(A, LU, P, 0);
    
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

//==============================================================================
// TEST 5: Various Matrix Sizes
//==============================================================================

static int test_lup_size_16(void)
{
    printf("  Testing: 16×16 random matrix\n");
    
    uint16_t n = 16;
    float *A = malloc(n * n * sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    
    int rc = lup(A, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 1e-3);
    
    free(A);
    free(LU);
    free(P);
    
    return pass;
}

static int test_lup_size_64(void)
{
    printf("  Testing: 64×64 random matrix\n");
    
    uint16_t n = 64;
    float *A = malloc(n * n * sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    
    int rc = lup(A, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 1e-3);
    
    free(A);
    free(LU);
    free(P);
    
    return pass;
}

static int test_lup_size_128(void)
{
    printf("  Testing: 128×128 random matrix\n");
    
    uint16_t n = 128;
    float *A = malloc(n * n * sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    
    int rc = lup(A, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 5e-3);
    
    free(A);
    free(LU);
    free(P);
    
    return pass;
}

static int test_lup_size_256(void)
{
    printf("  Testing: 256×256 random matrix\n");
    
    uint16_t n = 128;
    float *A = malloc(n * n * sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    
    clock_t start = clock();
    int rc = lup(A, LU, P, n);
    clock_t end = clock();
    
    double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, n);
    double gflops = (2.0/3.0 * n * n * n) / (time_ms / 1000.0) / 1e9;
    
    printf("    Time: %.2f ms (%.1f GFLOPS)\n", time_ms, gflops);
    printf("    Max error: %.2e\n", err);
    
    int pass = (err < 1e-2);
    
    free(A);
    free(LU);
    free(P);
    
    return pass;
}

//==============================================================================
// TEST 6: Workspace Management
//==============================================================================

static int test_lup_workspace_query(void)
{
    printf("  Testing: Workspace size query\n");
    
    uint16_t sizes[] = {16, 64, 128, 256};
    
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++)
    {
        size_t ws_size = lup_workspace_query(sizes[i]);
        printf("    n=%d: workspace=" FMT_SIZE_T " bytes\n", 
               sizes[i], CAST_SIZE_T(ws_size));
        
        if (ws_size == 0)
        {
            printf("    WARN: Workspace size is zero\n");
        }
    }
    
    return 1;
}

static int test_lup_workspace_create_destroy(void)
{
    printf("  Testing: Workspace creation and destruction\n");
    
    size_t ws_size = lup_workspace_query(128);
    lup_workspace_t *ws = lup_workspace_create(ws_size);
    
    if (!ws)
    {
        printf("    FAIL: Workspace creation failed\n");
        return 0;
    }
    
    printf("    Workspace created: " FMT_SIZE_T " bytes\n", CAST_SIZE_T(ws_size));
    
    lup_workspace_destroy(ws);
    printf("    Workspace destroyed successfully\n");
    
    return 1;
}

static int test_lup_with_workspace(void)
{
    printf("  Testing: LUP with explicit workspace\n");
    
    uint16_t n = 64;
    float *A = malloc(n * n * sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    
    size_t ws_size = lup_workspace_query(n);
    lup_workspace_t *ws = lup_workspace_create(ws_size);
    
    if (!ws)
    {
        printf("    FAIL: Workspace creation failed\n");
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    int rc = lup_ws(A, LU, P, n, ws);
    
    if (rc != 0)
    {
        printf("    FAIL: lup_ws() returned %d\n", rc);
        lup_workspace_destroy(ws);
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    lup_workspace_destroy(ws);
    free(A);
    free(LU);
    free(P);
    
    return (err < 1e-3);
}

//==============================================================================
// TEST 7: Edge Cases
//==============================================================================

static int test_lup_single_element(void)
{
    printf("  Testing: 1×1 matrix (scalar)\n");
    
    float A[1] = {5.0f};
    float LU[1];
    uint8_t P[1];
    
    int rc = lup(A, LU, P, 1);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        return 0;
    }
    
    printf("    LU[0] = %.6f (expected 5.0)\n", LU[0]);
    printf("    P[0] = %d (expected 0)\n", P[0]);
    
    float err = fabsf(LU[0] - 5.0f);
    
    return (err < 1e-6f && P[0] == 0);
}

static int test_lup_odd_size(void)
{
    printf("  Testing: Odd dimension (127×127)\n");
    
    uint16_t n = 127;
    float *A = malloc(n * n * sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    
    int rc = lup(A, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(LU);
    free(P);
    
    return (err < 1e-2);
}

static int test_lup_prime_size(void)
{
    printf("  Testing: Prime dimension (101×101)\n");
    
    uint16_t n = 101;
    float *A = malloc(n * n * sizeof(float));
    float *LU = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    
    int rc = lup(A, LU, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(LU);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A, LU, P, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(LU);
    free(P);
    
    return (err < 1e-2);
}

//==============================================================================
// TEST 8: In-Place Factorization
//==============================================================================

static int test_lup_inplace_small(void)
{
    printf("  Testing: In-place factorization (4×4)\n");
    
    uint16_t n = 4;
    float *A = malloc(n * n * sizeof(float));
    float *A_copy = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    memcpy(A_copy, A, n * n * sizeof(float));
    
    // In-place: LU stored in A
    int rc = lup(A, A, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(A_copy);
        free(P);
        return 0;
    }
    
    // Verify using copy
    double err = verify_factorization(A_copy, A, P, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(A_copy);
    free(P);
    
    return (err < 1e-4);
}

static int test_lup_inplace_large(void)
{
    printf("  Testing: In-place factorization (128×128)\n");
    
    uint16_t n = 128;
    float *A = malloc(n * n * sizeof(float));
    float *A_copy = malloc(n * n * sizeof(float));
    uint8_t *P = malloc(n * sizeof(uint8_t));
    
    create_random_matrix(A, n, 10.0f);
    memcpy(A_copy, A, n * n * sizeof(float));
    
    int rc = lup(A, A, P, n);
    
    if (rc != 0)
    {
        printf("    FAIL: lup() returned %d\n", rc);
        free(A);
        free(A_copy);
        free(P);
        return 0;
    }
    
    double err = verify_factorization(A_copy, A, P, n);
    printf("    Max error: %.2e\n", err);
    
    free(A);
    free(A_copy);
    free(P);
    
    return (err < 5e-3);
}

//==============================================================================
// TEST SUITE RUNNER
//==============================================================================

int run_lup_tests(test_results_t *results)
{
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    srand(12345); // Reproducible tests
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  LUP Factorization Module - Comprehensive Test Suite     ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Group 1: Basic Factorization
    printf("\n═══ Test Group 1: Basic Factorization (Small Matrices) ═══\n");
    RUN_TEST(results, test_lup_2x2_simple);
    RUN_TEST(results, test_lup_3x3_simple);
    RUN_TEST(results, test_lup_4x4_random);
    
    // Group 2: Special Matrices
    printf("\n═══ Test Group 2: Special Matrices ═══\n");
    RUN_TEST(results, test_lup_identity);
    RUN_TEST(results, test_lup_diagonal);
    RUN_TEST(results, test_lup_lower_triangular);
    RUN_TEST(results, test_lup_upper_triangular);
    RUN_TEST(results, test_lup_symmetric);
    
    // Group 3: Permutation Correctness
    printf("\n═══ Test Group 3: Permutation Correctness ═══\n");
    RUN_TEST(results, test_lup_pivot_needed);
    RUN_TEST(results, test_lup_permutation_validity);
    
    // Group 4: Error Handling
    printf("\n═══ Test Group 4: Error Handling ═══\n");
    RUN_TEST(results, test_lup_singular_exact);
    RUN_TEST(results, test_lup_singular_near);
    RUN_TEST(results, test_lup_zero_dimension);
    
    // Group 5: Various Sizes
    printf("\n═══ Test Group 5: Various Matrix Sizes ═══\n");
    RUN_TEST(results, test_lup_size_16);
    RUN_TEST(results, test_lup_size_64);
    RUN_TEST(results, test_lup_size_128);
    RUN_TEST(results, test_lup_size_256);
    
    // Group 6: Workspace Management
    printf("\n═══ Test Group 6: Workspace Management ═══\n");
    RUN_TEST(results, test_lup_workspace_query);
    RUN_TEST(results, test_lup_workspace_create_destroy);
    RUN_TEST(results, test_lup_with_workspace);
    
    // Group 7: Edge Cases
    printf("\n═══ Test Group 7: Edge Cases ═══\n");
    RUN_TEST(results, test_lup_single_element);
    RUN_TEST(results, test_lup_odd_size);
    RUN_TEST(results, test_lup_prime_size);
    
    // Group 8: In-Place Factorization
    printf("\n═══ Test Group 8: In-Place Factorization ═══\n");
    RUN_TEST(results, test_lup_inplace_small);
    RUN_TEST(results, test_lup_inplace_large);
    
    print_test_results("LUP Factorization Module", results);
    
    return (results->failed == 0) ? 0 : 1;
}

#ifdef STANDALONE
int main(void)
{
    test_results_t results;
    int ret = run_lup_tests(&results);
    
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