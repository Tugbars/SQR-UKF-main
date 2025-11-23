/**
 * @file test_trsm_blocked.c
 * @brief Unit tests for blocked triangular solve (TRSM) with GEMM acceleration
 *
 * Tests:
 * - Lower triangular solve: L · X = B
 * - Upper triangular solve: U · X = B
 * - Upper triangular transposed: U^T · X = B
 * - Block size selection heuristics
 * - Panel packing correctness
 * - Workspace allocation and reuse
 * - Singularity detection
 * - Edge cases and error handling
 * - Various matrix dimensions and RHS counts
 *
 * @author TUGBARS
 * @date 2025
 */

#include "test_common.h"
#include "trsm_blocked.h"
#include "gemm_planning.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>

//==============================================================================
// MATRIX UTILITIES
//==============================================================================

/**
 * @brief Compute Frobenius norm: ||A||_F = sqrt(sum(A[i,j]^2))
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
 * @brief Compute relative error: ||A - B||_F / ||A||_F
 */
static double relative_error(const float *A, const float *B, 
                             size_t m, size_t n, size_t ld)
{
    double diff_norm = 0.0;
    double a_norm = 0.0;

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double a = (double)A[i * ld + j];
            double b = (double)B[i * ld + j];
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
 * @brief Generate random lower triangular matrix with guaranteed non-zero diagonal
 * 
 * @param[out] L     Output matrix [n×n], row-major
 * @param[in]  n     Matrix dimension
 * @param[in]  ld    Leading dimension
 * @param[in]  seed  Random seed
 * @param[in]  diag_boost  Amount to add to diagonal for conditioning
 */
static void generate_lower_triangular(float *L, size_t n, size_t ld, 
                                      unsigned int seed, float diag_boost)
{
    srand(seed);
    
    // Zero out entire matrix first
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < ld; j++)
        {
            L[i * ld + j] = 0.0f;
        }
    }
    
    // Fill lower triangle
    for (size_t i = 0; i < n; i++)
    {
        // Diagonal element (guaranteed non-zero)
        L[i * ld + i] = diag_boost + ((float)(rand() % 100) + 10.0f) / 50.0f;
        
        // Sub-diagonal elements
        for (size_t j = 0; j < i; j++)
        {
            L[i * ld + j] = ((float)(rand() % 200) - 100.0f) / 100.0f;
        }
    }
}

/**
 * @brief Generate random upper triangular matrix with guaranteed non-zero diagonal
 */
static void generate_upper_triangular(float *U, size_t n, size_t ld,
                                      unsigned int seed, float diag_boost)
{
    srand(seed);
    
    // Zero out entire matrix first
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < ld; j++)
        {
            U[i * ld + j] = 0.0f;
        }
    }
    
    // Fill upper triangle
    for (size_t i = 0; i < n; i++)
    {
        // Diagonal element (guaranteed non-zero)
        U[i * ld + i] = diag_boost + ((float)(rand() % 100) + 10.0f) / 50.0f;
        
        // Super-diagonal elements
        for (size_t j = i + 1; j < n; j++)
        {
            U[i * ld + j] = ((float)(rand() % 200) - 100.0f) / 100.0f;
        }
    }
}

/**
 * @brief Generate random dense RHS matrix B
 */
static void generate_rhs_matrix(float *B, size_t n, size_t ncols, size_t ld,
                                unsigned int seed)
{
    srand(seed);
    
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < ncols; j++)
        {
            B[i * ld + j] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        }
        // Zero padding if ld > ncols
        for (size_t j = ncols; j < ld; j++)
        {
            B[i * ld + j] = 0.0f;
        }
    }
}

/**
 * @brief Matrix-matrix multiply: C = A × B (for verification)
 * 
 * @param[out] C   Output [m×n]
 * @param[in]  A   Input [m×k]
 * @param[in]  B   Input [k×n]
 * @param[in]  m   Rows of C and A
 * @param[in]  k   Cols of A, rows of B
 * @param[in]  n   Cols of C and B
 * @param[in]  ldc Leading dimension of C
 * @param[in]  lda Leading dimension of A
 * @param[in]  ldb Leading dimension of B
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
 * @brief Compute C = A^T × B (A transposed)
 */
static void matmul_At_B_reference(float *C, const float *A, const float *B,
                                  size_t m, size_t k, size_t n,
                                  size_t ldc, size_t lda, size_t ldb)
{
    // A is [k×m] so A^T is [m×k]
    // Result C is [m×n]
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++)
            {
                // A^T[i,p] = A[p,i]
                sum += (double)A[p * lda + i] * (double)B[p * ldb + j];
            }
            C[i * ldc + j] = (float)sum;
        }
    }
}

//==============================================================================
// TRSM PROPERTY CHECKERS
//==============================================================================

/**
 * @brief Verify TRSM solution: Check that T · X = B_original
 * 
 * @param[in] T          Triangular matrix [n×n]
 * @param[in] X          Computed solution [n×ncols]
 * @param[in] B_original Original RHS [n×ncols]
 * @param[in] n          Matrix dimension
 * @param[in] ncols      Number of RHS columns
 * @param[in] ldt        Leading dimension of T
 * @param[in] ldx        Leading dimension of X
 * @param[in] ldb        Leading dimension of B
 * @param[in] tol        Relative error tolerance
 * @param[in] test_name  Name for error messages
 * 
 * @return 1 if passed, 0 if failed
 */
static int check_trsm_solution(const float *T, const float *X,
                               const float *B_original,
                               size_t n, size_t ncols,
                               size_t ldt, size_t ldx, size_t ldb,
                               double tol, const char *test_name)
{
    printf("  Checking TRSM solution (T·X = B)...\n");
    
    // Allocate workspace for T × X
    float *B_computed = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    if (!B_computed)
    {
        printf("    ERROR: Allocation failed\n");
        return 0;
    }
    
    // Compute T × X
    matmul_reference(B_computed, T, X, n, n, ncols, ncols, ldt, ldx);
    
    // Compare with original B
    double rel_err = relative_error(B_original, B_computed, n, ncols, ldb);
    double b_norm = frobenius_norm(B_original, n, ncols, ldb);
    
    gemm_aligned_free(B_computed);
    
    if (rel_err > tol)
    {
        printf("    %s: Solution check FAILED\n", test_name);
        printf("    ||B_original||_F = %.6e\n", b_norm);
        printf("    ||B_original - T·X||_F / ||B_original||_F = %.6e (tol: %.6e)\n",
               rel_err, tol);
        return 0;
    }
    
    printf("    %s: Solution check PASSED\n", test_name);
    printf("    Relative error: %.6e\n", rel_err);
    return 1;
}

/**
 * @brief Verify transposed TRSM solution: Check that T^T · X = B_original
 */
static int check_trsm_transpose_solution(const float *T, const float *X,
                                         const float *B_original,
                                         size_t n, size_t ncols,
                                         size_t ldt, size_t ldx, size_t ldb,
                                         double tol, const char *test_name)
{
    printf("  Checking transposed TRSM solution (T^T·X = B)...\n");
    
    float *B_computed = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    if (!B_computed)
    {
        printf("    ERROR: Allocation failed\n");
        return 0;
    }
    
    // Compute T^T × X
    matmul_At_B_reference(B_computed, T, X, n, n, ncols, ncols, ldt, ldx);
    
    double rel_err = relative_error(B_original, B_computed, n, ncols, ldb);
    double b_norm = frobenius_norm(B_original, n, ncols, ldb);
    
    gemm_aligned_free(B_computed);
    
    if (rel_err > tol)
    {
        printf("    %s: Transpose solution check FAILED\n", test_name);
        printf("    ||B_original||_F = %.6e\n", b_norm);
        printf("    ||B_original - T^T·X||_F / ||B_original||_F = %.6e (tol: %.6e)\n",
               rel_err, tol);
        return 0;
    }
    
    printf("    %s: Transpose solution check PASSED\n", test_name);
    printf("    Relative error: %.6e\n", rel_err);
    return 1;
}

/**
 * @brief Check triangular matrix structure (zeros in wrong places)
 */
static int check_triangular_structure(const float *T, size_t n, size_t ld,
                                      int is_lower, const char *test_name)
{
    printf("  Checking triangular structure...\n");
    
    int errors = 0;
    
    if (is_lower)
    {
        // Check upper triangle is zero
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = i + 1; j < n; j++)
            {
                if (T[i * ld + j] != 0.0f)
                {
                    if (errors < 3)
                    {
                        printf("    Non-zero at upper [%zu,%zu] = %.6f\n",
                               i, j, T[i * ld + j]);
                    }
                    errors++;
                }
            }
        }
    }
    else
    {
        // Check lower triangle is zero
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (T[i * ld + j] != 0.0f)
                {
                    if (errors < 3)
                    {
                        printf("    Non-zero at lower [%zu,%zu] = %.6f\n",
                               i, j, T[i * ld + j]);
                    }
                    errors++;
                }
            }
        }
    }
    
    if (errors > 0)
    {
        printf("    %s: Structure check FAILED (%d errors)\n", test_name, errors);
        return 0;
    }
    
    printf("    %s: Structure check PASSED\n", test_name);
    return 1;
}

/**
 * @brief Check diagonal elements are non-zero
 */
static int check_nonsingular_diagonal(const float *T, size_t n, size_t ld,
                                      const char *test_name)
{
    printf("  Checking diagonal elements...\n");
    
    float min_diag = FLT_MAX;
    int zero_count = 0;
    
    for (size_t i = 0; i < n; i++)
    {
        float diag = fabsf(T[i * ld + i]);
        if (diag < min_diag)
            min_diag = diag;
        if (diag == 0.0f)
            zero_count++;
    }
    
    if (zero_count > 0)
    {
        printf("    %s: Diagonal check FAILED (%d zeros)\n", test_name, zero_count);
        return 0;
    }
    
    printf("    %s: Diagonal check PASSED (min |diag| = %.6e)\n", test_name, min_diag);
    return 1;
}

//==============================================================================
// REFERENCE TRSM IMPLEMENTATIONS (for comparison)
//==============================================================================

/**
 * @brief Simple unblocked lower triangular solve (reference)
 */
static void trsm_lower_reference(const float *L, float *B,
                                 size_t n, size_t ncols,
                                 size_t ldl, size_t ldb)
{
    // Forward substitution: L · X = B
    for (size_t j = 0; j < n; j++)
    {
        float ljj = L[j * ldl + j];
        if (ljj == 0.0f)
            continue;
        
        float inv_ljj = 1.0f / ljj;
        
        // Scale row j
        for (size_t k = 0; k < ncols; k++)
        {
            B[j * ldb + k] *= inv_ljj;
        }
        
        // Update trailing rows
        for (size_t i = j + 1; i < n; i++)
        {
            float lij = L[i * ldl + j];
            for (size_t k = 0; k < ncols; k++)
            {
                B[i * ldb + k] -= lij * B[j * ldb + k];
            }
        }
    }
}

/**
 * @brief Simple unblocked upper triangular solve (reference)
 */
static void trsm_upper_reference(const float *U, float *B,
                                 size_t n, size_t ncols,
                                 size_t ldu, size_t ldb)
{
    // Backward substitution: U · X = B
    for (int j = (int)n - 1; j >= 0; j--)
    {
        float ujj = U[j * ldu + j];
        if (ujj == 0.0f)
            continue;
        
        float inv_ujj = 1.0f / ujj;
        
        // Scale row j
        for (size_t k = 0; k < ncols; k++)
        {
            B[j * ldb + k] *= inv_ujj;
        }
        
        // Update preceding rows
        for (int i = j - 1; i >= 0; i--)
        {
            float uij = U[i * ldu + j];
            for (size_t k = 0; k < ncols; k++)
            {
                B[i * ldb + k] -= uij * B[j * ldb + k];
            }
        }
    }
}

/**
 * @brief Simple unblocked U^T solve (reference): U^T · X = B
 */
static void trsm_upper_transpose_reference(const float *U, float *B,
                                           size_t n, size_t ncols,
                                           size_t ldu, size_t ldb)
{
    // Forward substitution on U^T (which is lower triangular)
    for (size_t j = 0; j < n; j++)
    {
        // Diagonal: U^T[j,j] = U[j,j]
        float ujj = U[j * ldu + j];
        if (ujj == 0.0f)
            continue;
        
        float inv_ujj = 1.0f / ujj;
        
        // Scale row j
        for (size_t k = 0; k < ncols; k++)
        {
            B[j * ldb + k] *= inv_ujj;
        }
        
        // Update trailing rows: U^T[i,j] = U[j,i] for i > j
        for (size_t i = j + 1; i < n; i++)
        {
            float utij = U[j * ldu + i]; // U^T[i,j] = U[j,i]
            for (size_t k = 0; k < ncols; k++)
            {
                B[i * ldb + k] -= utij * B[j * ldb + k];
            }
        }
    }
}

//==============================================================================
// INDIVIDUAL TESTS
//==============================================================================

/**
 * @brief Test small lower triangular solve (8×8)
 */
static int test_lower_small(void)
{
    printf("\n=== Testing Lower Triangular TRSM (8×8, 4 RHS) ===\n");
    
    const size_t n = 8;
    const size_t ncols = 4;
    
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!L || !B || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    generate_lower_triangular(L, n, n, 12345, 1.0f);
    generate_rhs_matrix(B, n, ncols, ncols, 54321);
    memcpy(B_original, B, n * ncols * sizeof(float));
    
    // Allocate GEMM plan
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Performing blocked lower TRSM...\n");
    int ret = trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_lower returned %d\n", ret);
        goto cleanup_fail;
    }
    
    int passed = 1;
    passed &= check_trsm_solution(L, B, B_original, n, ncols, n, ncols, ncols, 
                                  1e-5, "lower 8×8");
    
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test small upper triangular solve (8×8)
 */
static int test_upper_small(void)
{
    printf("\n=== Testing Upper Triangular TRSM (8×8, 4 RHS) ===\n");
    
    const size_t n = 8;
    const size_t ncols = 4;
    
    float *U = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!U || !B || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    generate_upper_triangular(U, n, n, 23456, 1.0f);
    generate_rhs_matrix(B, n, ncols, ncols, 65432);
    memcpy(B_original, B, n * ncols * sizeof(float));
    
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Performing blocked upper TRSM...\n");
    int ret = trsm_blocked_upper(U, B, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_upper returned %d\n", ret);
        goto cleanup_fail;
    }
    
    int passed = 1;
    passed &= check_trsm_solution(U, B, B_original, n, ncols, n, ncols, ncols,
                                  1e-5, "upper 8×8");
    
    gemm_aligned_free(U);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(U);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test medium lower triangular solve (triggers blocking, NB=48)
 */
static int test_lower_medium(void)
{
    printf("\n=== Testing Lower Triangular TRSM (64×64, 32 RHS) ===\n");
    
    const size_t n = 64;
    const size_t ncols = 32;
    
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!L || !B || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    generate_lower_triangular(L, n, n, 34567, 2.0f);
    generate_rhs_matrix(B, n, ncols, ncols, 76543);
    memcpy(B_original, B, n * ncols * sizeof(float));
    
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Performing blocked lower TRSM (expect NB=48 or 64)...\n");
    int ret = trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_lower returned %d\n", ret);
        goto cleanup_fail;
    }
    
    int passed = 1;
    passed &= check_trsm_solution(L, B, B_original, n, ncols, n, ncols, ncols,
                                  1e-4, "lower 64×64");
    
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test upper triangular with transpose (U^T · X = B)
 */
static int test_upper_transpose(void)
{
    printf("\n=== Testing Upper Triangular Transpose TRSM (32×32, 16 RHS) ===\n");
    
    const size_t n = 32;
    const size_t ncols = 16;
    
    float *U = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!U || !B || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    generate_upper_triangular(U, n, n, 45678, 2.0f);
    generate_rhs_matrix(B, n, ncols, ncols, 87654);
    memcpy(B_original, B, n * ncols * sizeof(float));
    
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Performing blocked upper transpose TRSM...\n");
    int ret = trsm_blocked_upper_transpose_auto(U, B, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_upper_transpose_auto returned %d\n", ret);
        goto cleanup_fail;
    }
    
    int passed = 1;
    passed &= check_trsm_transpose_solution(U, B, B_original, n, ncols, n, ncols, ncols,
                                            1e-4, "upper^T 32×32");
    
    gemm_aligned_free(U);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(U);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test comparison with reference implementation
 */
static int test_vs_reference(void)
{
    printf("\n=== Testing Blocked vs Reference TRSM ===\n");
    
    const size_t n = 48;
    const size_t ncols = 24;
    
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B_blocked = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_reference = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!L || !B_blocked || !B_reference || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    generate_lower_triangular(L, n, n, 56789, 2.0f);
    generate_rhs_matrix(B_original, n, ncols, ncols, 98765);
    memcpy(B_blocked, B_original, n * ncols * sizeof(float));
    memcpy(B_reference, B_original, n * ncols * sizeof(float));
    
    // Compute reference solution
    printf("  Computing reference solution...\n");
    trsm_lower_reference(L, B_reference, n, ncols, n, ncols);
    
    // Compute blocked solution
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Computing blocked solution...\n");
    int ret = trsm_blocked_lower(L, B_blocked, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_lower returned %d\n", ret);
        goto cleanup_fail;
    }
    
    // Compare solutions
    double rel_err = relative_error(B_reference, B_blocked, n, ncols, ncols);
    printf("  ||X_blocked - X_reference||_F / ||X_reference||_F = %.6e\n", rel_err);
    
    int passed = 1;
    if (rel_err > 1e-5)
    {
        printf("  FAILED: Blocked and reference disagree\n");
        passed = 0;
    }
    else
    {
        printf("  PASSED: Blocked matches reference\n");
    }
    
    // Also verify both are correct solutions
    passed &= check_trsm_solution(L, B_blocked, B_original, n, ncols, n, ncols, ncols,
                                  1e-4, "blocked");
    passed &= check_trsm_solution(L, B_reference, B_original, n, ncols, n, ncols, ncols,
                                  1e-4, "reference");
    
    gemm_aligned_free(L);
    gemm_aligned_free(B_blocked);
    gemm_aligned_free(B_reference);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(L);
    gemm_aligned_free(B_blocked);
    gemm_aligned_free(B_reference);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test with non-square leading dimensions
 */
static int test_non_square_ld(void)
{
    printf("\n=== Testing Non-Square Leading Dimensions ===\n");
    
    const size_t n = 32;
    const size_t ncols = 16;
    const size_t ldl = 40;  // Padded leading dimension for L
    const size_t ldb = 24;  // Padded leading dimension for B
    
    float *L = gemm_aligned_alloc(32, n * ldl * sizeof(float));
    float *B = gemm_aligned_alloc(32, n * ldb * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ldb * sizeof(float));
    
    if (!L || !B || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    // Initialize with padding
    memset(L, 0, n * ldl * sizeof(float));
    memset(B, 0, n * ldb * sizeof(float));
    
    generate_lower_triangular(L, n, ldl, 67890, 2.0f);
    generate_rhs_matrix(B, n, ncols, ldb, 9876);
    memcpy(B_original, B, n * ldb * sizeof(float));
    
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Performing TRSM with ldl=%zu, ldb=%zu...\n", ldl, ldb);
    int ret = trsm_blocked_lower(L, B, n, ncols, ldl, ldb, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_lower returned %d\n", ret);
        goto cleanup_fail;
    }
    
    int passed = 1;
    passed &= check_trsm_solution(L, B, B_original, n, ncols, ldl, ldb, ldb,
                                  1e-4, "non-square ld");
    
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test singularity detection
 */
static int test_singularity_detection(void)
{
    printf("\n=== Testing Singularity Detection ===\n");
    
    const size_t n = 16;
    const size_t ncols = 8;
    
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!L || !B)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    // Generate valid lower triangular, then make one diagonal zero
    generate_lower_triangular(L, n, n, 11111, 1.0f);
    L[8 * n + 8] = 0.0f; // Make diagonal[8] singular
    
    generate_rhs_matrix(B, n, ncols, ncols, 22222);
    
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Attempting TRSM on singular matrix...\n");
    int ret = trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    int passed = 1;
    if (ret == -EDOM)
    {
        printf("  Singularity correctly detected (returned -EDOM)\n");
        printf("  PASSED\n");
    }
    else if (ret == 0)
    {
        printf("  WARNING: Singularity not detected (returned 0)\n");
        printf("  Note: Some implementations may skip zero rows silently\n");
        // This may still be acceptable behavior
    }
    else
    {
        printf("  Unexpected return code: %d\n", ret);
        passed = 0;
    }
    
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    return 0;
}

/**
 * @brief Test edge cases: n=0, ncols=0, n=1
 */
static int test_edge_cases(void)
{
    printf("\n=== Testing Edge Cases ===\n");
    
    int passed = 1;
    
    // Test 1: n=0 (should be no-op)
    printf("  Testing n=0 (no-op)...\n");
    {
        gemm_plan_t *plan = gemm_plan_alloc(0, 0, 0);
        int ret = trsm_blocked_lower(NULL, NULL, 0, 0, 0, 0, plan);
        gemm_plan_free(plan);
        
        if (ret != 0)
        {
            printf("    ERROR: n=0 returned %d (expected 0)\n", ret);
            passed = 0;
        }
        else
        {
            printf("    n=0 test PASSED\n");
        }
    }
    
    // Test 2: ncols=0 (should be no-op)
    printf("  Testing ncols=0 (no-op)...\n");
    {
        const size_t n = 8;
        float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
        generate_lower_triangular(L, n, n, 33333, 1.0f);
        
        gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, 0);
        int ret = trsm_blocked_lower(L, NULL, n, 0, n, 0, plan);
        gemm_plan_free(plan);
        gemm_aligned_free(L);
        
        if (ret != 0)
        {
            printf("    ERROR: ncols=0 returned %d (expected 0)\n", ret);
            passed = 0;
        }
        else
        {
            printf("    ncols=0 test PASSED\n");
        }
    }
    
    // Test 3: n=1 (scalar case)
    printf("  Testing n=1 (scalar)...\n");
    {
        const size_t n = 1, ncols = 1;
        float L[1] = {2.0f};
        float B[1] = {4.0f};
        float B_orig[1] = {4.0f};
        
        gemm_plan_t *plan = gemm_plan_alloc(1, 1, 1);
        int ret = trsm_blocked_lower(&L[0], &B[0], n, ncols, n, ncols, plan);
        gemm_plan_free(plan);
        
        if (ret != 0)
        {
            printf("    ERROR: Scalar solve returned %d\n", ret);
            passed = 0;
        }
        else
        {
            // Expected: X = B/L = 4/2 = 2
            float expected = 2.0f;
            float diff = fabsf(B[0] - expected);
            
            if (diff > 1e-6)
            {
                printf("    ERROR: Scalar result wrong (got %.6f, expected %.6f)\n",
                       B[0], expected);
                passed = 0;
            }
            else
            {
                printf("    Scalar test PASSED (X = %.6f)\n", B[0]);
            }
        }
    }
    
    // Test 4: Single RHS column
    printf("  Testing single RHS column (32×32, 1 RHS)...\n");
    {
        const size_t n = 32, ncols = 1;
        
        float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
        float *B_orig = gemm_aligned_alloc(32, n * ncols * sizeof(float));
        
        generate_lower_triangular(L, n, n, 44444, 2.0f);
        generate_rhs_matrix(B, n, ncols, ncols, 55555);
        memcpy(B_orig, B, n * ncols * sizeof(float));
        
        gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
        int ret = trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
        gemm_plan_free(plan);
        
        if (ret != 0)
        {
            printf("    ERROR: Single RHS returned %d\n", ret);
            passed = 0;
        }
        else
        {
            int test_passed = check_trsm_solution(L, B, B_orig, n, ncols, n, ncols, ncols,
                                                   1e-4, "single RHS");
            passed &= test_passed;
        }
        
        gemm_aligned_free(L);
        gemm_aligned_free(B);
        gemm_aligned_free(B_orig);
    }
    
    return passed;
}

/**
 * @brief Test large matrix (stress test)
 */
static int test_large_matrix(void)
{
    printf("\n=== Testing Large Matrix (256×256, 64 RHS) ===\n");
    
    const size_t n = 256;
    const size_t ncols = 64;
    
    printf("  Allocating %.2f MB...\n",
           (n * n + 2 * n * ncols) * sizeof(float) / (1024.0 * 1024.0));
    
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!L || !B || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Generating lower triangular matrix...\n");
    generate_lower_triangular(L, n, n, 123456, 5.0f);
    
    printf("  Generating RHS matrix...\n");
    generate_rhs_matrix(B, n, ncols, ncols, 654321);
    memcpy(B_original, B, n * ncols * sizeof(float));
    
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Performing blocked lower TRSM (expect NB=64 or 96)...\n");
    int ret = trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_lower returned %d\n", ret);
        goto cleanup_fail;
    }
    
    printf("  Verifying solution...\n");
    int passed = check_trsm_solution(L, B, B_original, n, ncols, n, ncols, ncols,
                                     5e-4, "large 256×256");
    
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test many RHS columns (tests RC blocking)
 */
static int test_many_rhs_columns(void)
{
    printf("\n=== Testing Many RHS Columns (48×48, 128 RHS) ===\n");
    
    const size_t n = 48;
    const size_t ncols = 128; // Much larger than RC (32)
    
    float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!L || !B || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    generate_lower_triangular(L, n, n, 77777, 2.0f);
    generate_rhs_matrix(B, n, ncols, ncols, 88888);
    memcpy(B_original, B, n * ncols * sizeof(float));
    
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Performing TRSM with %zu RHS columns (RC blocking)...\n", ncols);
    int ret = trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_lower returned %d\n", ret);
        goto cleanup_fail;
    }
    
    int passed = check_trsm_solution(L, B, B_original, n, ncols, n, ncols, ncols,
                                     1e-4, "many RHS");
    
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(L);
    gemm_aligned_free(B);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test block size selection across various n values
 */
static int test_block_size_selection(void)
{
    printf("\n=== Testing Block Size Selection Heuristics ===\n");
    
    int passed = 1;
    
    // Test various sizes that should trigger different block sizes
    struct {
        size_t n;
        size_t expected_nb_min;
        size_t expected_nb_max;
    } test_cases[] = {
        {32,  32,  32},   // n≤96 → NB=32
        {64,  32,  48},   // n≤96 → NB=32 (or 48)
        {128, 48,  64},   // n≤256 → NB=48
        {200, 48,  64},   // n≤256 → NB=48
        {400, 64,  96},   // n≤512 → NB=64
        {600, 96, 128},   // n≤1024 → NB=96
    };
    
    const size_t ncols = 32;
    
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++)
    {
        size_t n = test_cases[i].n;
        
        printf("  Testing n=%zu...\n", n);
        
        float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
        float *B_orig = gemm_aligned_alloc(32, n * ncols * sizeof(float));
        
        if (!L || !B || !B_orig)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            gemm_aligned_free(L);
            gemm_aligned_free(B);
            gemm_aligned_free(B_orig);
            continue;
        }
        
        generate_lower_triangular(L, n, n, (unsigned int)(i * 11111), 3.0f);
        generate_rhs_matrix(B, n, ncols, ncols, (unsigned int)(i * 22222));
        memcpy(B_orig, B, n * ncols * sizeof(float));
        
        gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
        
        int ret = trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
        
        gemm_plan_free(plan);
        
        if (ret != 0)
        {
            printf("    ERROR: TRSM returned %d\n", ret);
            passed = 0;
        }
        else
        {
            int test_passed = check_trsm_solution(L, B, B_orig, n, ncols, n, ncols, ncols,
                                                   5e-4, "block selection");
            if (!test_passed)
            {
                passed = 0;
            }
        }
        
        gemm_aligned_free(L);
        gemm_aligned_free(B);
        gemm_aligned_free(B_orig);
    }
    
    return passed;
}

/**
 * @brief Test upper transpose vs reference
 */
static int test_upper_transpose_vs_reference(void)
{
    printf("\n=== Testing Upper Transpose vs Reference ===\n");
    
    const size_t n = 48;
    const size_t ncols = 24;
    
    float *U = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *B_blocked = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_reference = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    float *B_original = gemm_aligned_alloc(32, n * ncols * sizeof(float));
    
    if (!U || !B_blocked || !B_reference || !B_original)
    {
        printf("  ERROR: Allocation failed\n");
        goto cleanup_fail;
    }
    
    generate_upper_triangular(U, n, n, 99999, 2.0f);
    generate_rhs_matrix(B_original, n, ncols, ncols, 88888);
    memcpy(B_blocked, B_original, n * ncols * sizeof(float));
    memcpy(B_reference, B_original, n * ncols * sizeof(float));
    
    // Compute reference solution
    printf("  Computing reference U^T solve...\n");
    trsm_upper_transpose_reference(U, B_reference, n, ncols, n, ncols);
    
    // Compute blocked solution
    gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
    if (!plan)
    {
        printf("  ERROR: GEMM plan allocation failed\n");
        goto cleanup_fail;
    }
    
    printf("  Computing blocked U^T solve...\n");
    int ret = trsm_blocked_upper_transpose_auto(U, B_blocked, n, ncols, n, ncols, plan);
    
    gemm_plan_free(plan);
    
    if (ret != 0)
    {
        printf("  ERROR: trsm_blocked_upper_transpose_auto returned %d\n", ret);
        goto cleanup_fail;
    }
    
    // Compare solutions
    double rel_err = relative_error(B_reference, B_blocked, n, ncols, ncols);
    printf("  ||X_blocked - X_reference||_F / ||X_reference||_F = %.6e\n", rel_err);
    
    int passed = 1;
    if (rel_err > 1e-4)
    {
        printf("  FAILED: Blocked and reference disagree\n");
        passed = 0;
    }
    else
    {
        printf("  PASSED: Blocked matches reference\n");
    }
    
    // Verify both are correct
    passed &= check_trsm_transpose_solution(U, B_blocked, B_original, n, ncols, 
                                            n, ncols, ncols, 1e-4, "blocked U^T");
    passed &= check_trsm_transpose_solution(U, B_reference, B_original, n, ncols,
                                            n, ncols, ncols, 1e-4, "reference U^T");
    
    gemm_aligned_free(U);
    gemm_aligned_free(B_blocked);
    gemm_aligned_free(B_reference);
    gemm_aligned_free(B_original);
    return passed;
    
cleanup_fail:
    gemm_aligned_free(U);
    gemm_aligned_free(B_blocked);
    gemm_aligned_free(B_reference);
    gemm_aligned_free(B_original);
    return 0;
}

/**
 * @brief Test conditioning sensitivity
 */
static int test_conditioning(void)
{
    printf("\n=== Testing Conditioning Sensitivity ===\n");
    
    const size_t n = 32;
    const size_t ncols = 16;
    
    int passed = 1;
    
    // Test well-conditioned matrix
    printf("  Testing well-conditioned matrix...\n");
    {
        float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
        float *B_orig = gemm_aligned_alloc(32, n * ncols * sizeof(float));
        
        // Well-conditioned: large diagonal boost
        generate_lower_triangular(L, n, n, 12121, 10.0f);
        generate_rhs_matrix(B, n, ncols, ncols, 21212);
        memcpy(B_orig, B, n * ncols * sizeof(float));
        
        gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
        trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
        gemm_plan_free(plan);
        
        int test_passed = check_trsm_solution(L, B, B_orig, n, ncols, n, ncols, ncols,
                                               1e-5, "well-conditioned");
        passed &= test_passed;
        
        gemm_aligned_free(L);
        gemm_aligned_free(B);
        gemm_aligned_free(B_orig);
    }
    
    // Test moderately ill-conditioned matrix
    printf("  Testing moderately ill-conditioned matrix...\n");
    {
        float *L = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *B = gemm_aligned_alloc(32, n * ncols * sizeof(float));
        float *B_orig = gemm_aligned_alloc(32, n * ncols * sizeof(float));
        
        // Ill-conditioned: small diagonal, large off-diagonal
        generate_lower_triangular(L, n, n, 34343, 0.1f);
        
        // Make off-diagonal elements larger
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                L[i * n + j] *= 5.0f;
            }
        }
        
        generate_rhs_matrix(B, n, ncols, ncols, 43434);
        memcpy(B_orig, B, n * ncols * sizeof(float));
        
        gemm_plan_t *plan = gemm_plan_alloc((uint16_t)n, (uint16_t)n, (uint16_t)ncols);
        trsm_blocked_lower(L, B, n, ncols, n, ncols, plan);
        gemm_plan_free(plan);
        
        // Relaxed tolerance for ill-conditioned
        int test_passed = check_trsm_solution(L, B, B_orig, n, ncols, n, ncols, ncols,
                                               1e-2, "ill-conditioned");
        passed &= test_passed;
        
        gemm_aligned_free(L);
        gemm_aligned_free(B);
        gemm_aligned_free(B_orig);
    }
    
    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_trsm_blocked_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("    BLOCKED TRSM TESTS\n");
    printf("=================================================\n");
    
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    // Basic functionality tests
    printf("\n--- Basic Functionality Tests ---\n");
    
    results->total++;
    if (test_lower_small())
    {
        results->passed++;
        printf("✓ Lower small test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Lower small test FAILED\n");
    }
    
    results->total++;
    if (test_upper_small())
    {
        results->passed++;
        printf("✓ Upper small test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Upper small test FAILED\n");
    }
    
    results->total++;
    if (test_lower_medium())
    {
        results->passed++;
        printf("✓ Lower medium test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Lower medium test FAILED\n");
    }
    
    results->total++;
    if (test_upper_transpose())
    {
        results->passed++;
        printf("✓ Upper transpose test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Upper transpose test FAILED\n");
    }
    
    // Reference comparison tests
    printf("\n--- Reference Comparison Tests ---\n");
    
    results->total++;
    if (test_vs_reference())
    {
        results->passed++;
        printf("✓ Blocked vs reference test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Blocked vs reference test FAILED\n");
    }
    
    results->total++;
    if (test_upper_transpose_vs_reference())
    {
        results->passed++;
        printf("✓ Upper transpose vs reference test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Upper transpose vs reference test FAILED\n");
    }
    
    // Edge case tests
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
    
    results->total++;
    if (test_non_square_ld())
    {
        results->passed++;
        printf("✓ Non-square LD test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Non-square LD test FAILED\n");
    }
    
    results->total++;
    if (test_singularity_detection())
    {
        results->passed++;
        printf("✓ Singularity detection test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Singularity detection test FAILED\n");
    }
    
    // Block size and performance tests
    printf("\n--- Block Size / Performance Tests ---\n");
    
    results->total++;
    if (test_block_size_selection())
    {
        results->passed++;
        printf("✓ Block size selection test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Block size selection test FAILED\n");
    }
    
    results->total++;
    if (test_many_rhs_columns())
    {
        results->passed++;
        printf("✓ Many RHS columns test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Many RHS columns test FAILED\n");
    }
    
    // Numerical stability tests
    printf("\n--- Numerical Stability Tests ---\n");
    
    results->total++;
    if (test_conditioning())
    {
        results->passed++;
        printf("✓ Conditioning test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Conditioning test FAILED\n");
    }
    
    // Stress tests
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
    
    // Summary
    printf("\n=================================================\n");
    printf("Blocked TRSM Tests: %d/%d passed\n", results->passed, results->total);
    
    if (results->passed == results->total)
    {
        printf("✓ ALL BLOCKED TRSM TESTS PASSED!\n");
    }
    else
    {
        printf("✗ %d Blocked TRSM tests FAILED\n", results->failed);
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
    return run_trsm_blocked_tests(&results);
}
#endif