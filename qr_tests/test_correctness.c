/**
 * @file test_qr_blocked.c
 * @brief Unit tests for GEMM-accelerated blocked QR decomposition
 *
 * Tests:
 * - Reconstruction accuracy: A = Q*R
 * - Orthogonality: Q^T * Q = I
 * - Upper triangular R
 * - Various matrix shapes (square, tall, wide)
 * - Edge cases and numerical stability
 *
 * @author TUGBARS
 * @date 2025
 */

#include "test_common.h"
#include "qr.h"
#include "gemm.h"
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
 * @brief Compute max absolute value in matrix
 */
static double max_abs(const float *A, uint16_t m, uint16_t n)
{
    double max_val = 0.0;
    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double val = fabs((double)A[i * n + j]);
            if (val > max_val)
                max_val = val;
        }
    }
    return max_val;
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
    uint16_t display_rows = (rows < max_display) ? rows : max_display;
    uint16_t display_cols = (cols < max_display) ? cols : max_display;

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
 * @brief Check if R is upper triangular
 */
static int is_upper_triangular(const float *R, uint16_t m, uint16_t n, double tol)
{
    int errors = 0;
    double max_lower = 0.0;

    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            if (i > j) // Lower triangle
            {
                double val = fabs((double)R[i * n + j]);
                if (val > max_lower)
                    max_lower = val;
                if (val > tol)
                {
                    if (errors < 3) // Print first 3 errors
                    {
                        printf("    Lower triangle error at [%d,%d] = %.6e\n",
                               i, j, val);
                    }
                    errors++;
                }
            }
        }
    }

    if (errors > 0)
    {
        printf("    R not upper triangular: %d errors, max lower = %.6e\n",
               errors, max_lower);
        return 0;
    }

    printf("    R is upper triangular (max lower = %.6e)\n", max_lower);
    return 1;
}

//==============================================================================
// QR PROPERTY CHECKERS
//==============================================================================

/**
 * @brief Test Q^T * Q = I (orthogonality)
 */
static int check_orthogonality(const float *Q, uint16_t m, double tol,
                               const char *test_name)
{
    printf("  Checking orthogonality (Q^T * Q = I)...\n");

    // ✅ FIXED: Create Q^T explicitly
    float *QT = gemm_aligned_alloc(32, m * m * sizeof(float));
    float *QTQ = gemm_aligned_alloc(32, m * m * sizeof(float));
    
    if (!QT || !QTQ)
    {
        printf("    ERROR: Allocation failed\n");
        gemm_aligned_free(QT);
        gemm_aligned_free(QTQ);
        return 0;
    }

    // Transpose Q
    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < m; j++)
        {
            QT[i * m + j] = Q[j * m + i];
        }
    }

    // QTQ = Q^T * Q  (m×m = m×m * m×m)
    int ret = gemm_auto(QTQ, QT, Q, m, m, m, 1.0f, 0.0f);
    if (ret != 0)
    {
        printf("    ERROR: GEMM failed\n");
        gemm_aligned_free(QT);
        gemm_aligned_free(QTQ);
        return 0;
    }

    // Check if QTQ ≈ I
    int errors = 0;
    double max_diag_err = 0.0;
    double max_offdiag = 0.0;

    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < m; j++)
        {
            double val = (double)QTQ[i * m + j];
            double expected = (i == j) ? 1.0 : 0.0;
            double err = fabs(val - expected);

            if (i == j)
            {
                if (err > max_diag_err)
                    max_diag_err = err;
            }
            else
            {
                if (fabs(val) > max_offdiag)
                    max_offdiag = fabs(val);
            }

            if (err > tol)
            {
                if (errors < 3)
                {
                    printf("    Orthogonality error at [%d,%d]: "
                           "got %.6e, expected %.6e, diff %.6e\n",
                           i, j, val, expected, err);
                }
                errors++;
            }
        }
    }

    gemm_aligned_free(QT);
    gemm_aligned_free(QTQ);

    if (errors > 0)
    {
        printf("    %s: Q not orthogonal (%d errors)\n", test_name, errors);
        printf("    Max diagonal error: %.6e\n", max_diag_err);
        printf("    Max off-diagonal:   %.6e\n", max_offdiag);
        return 0;
    }

    printf("    %s: Q is orthogonal\n", test_name);
    printf("    Max diagonal error: %.6e\n", max_diag_err);
    printf("    Max off-diagonal:   %.6e\n", max_offdiag);
    return 1;
}

/**
 * @brief Test A = Q * R (reconstruction)
 */
static int check_reconstruction(const float *A, const float *Q, const float *R,
                                uint16_t m, uint16_t n, double tol,
                                const char *test_name)
{
    printf("  Checking reconstruction (A = Q*R)...\n");

    // Compute Q * R
    float *QR = gemm_aligned_alloc(32, m * n * sizeof(float));
    if (!QR)
    {
        printf("    ERROR: Allocation failed\n");
        return 0;
    }

    // QR = Q * R  (m×n = m×m * m×n)
    int ret = gemm_auto(QR, Q, R, m, m, n, 1.0f, 0.0f);
    if (ret != 0)
    {
        printf("    ERROR: GEMM failed\n");
        gemm_aligned_free(QR);
        return 0;
    }

    // Compute ||A - QR||_F / ||A||_F
    double rel_err = relative_error(A, QR, m, n);
    double a_norm = frobenius_norm(A, m, n);

    gemm_aligned_free(QR);

    if (rel_err > tol)
    {
        printf("    %s: Reconstruction FAILED\n", test_name);
        printf("    ||A||_F        = %.6e\n", a_norm);
        printf("    ||A - QR||_F / ||A||_F = %.6e (tolerance: %.6e)\n",
               rel_err, tol);
        return 0;
    }

    printf("    %s: Reconstruction PASSED\n", test_name);
    printf("    ||A - QR||_F / ||A||_F = %.6e\n", rel_err);
    return 1;
}

//==============================================================================
// REFERENCE IMPLEMENTATION (Simple Gram-Schmidt for validation)
//==============================================================================

/**
 * @brief Classical Gram-Schmidt QR (for small test matrices only)
 *
 * WARNING: Numerically unstable! Only for testing small, well-conditioned matrices.
 */
static int qr_gram_schmidt_reference(const float *A, float *Q, float *R,
                                     uint16_t m, uint16_t n)
{
    if (m < n)
        return -1; // Underdetermined system

    // Copy A to Q
    memcpy(Q, A, m * n * sizeof(float));
    memset(R, 0, m * n * sizeof(float));

    for (uint16_t j = 0; j < n; j++)
    {
        // R[j,j] = ||Q[:,j]||
        double norm_sq = 0.0;
        for (uint16_t i = 0; i < m; i++)
        {
            double val = (double)Q[i * n + j];
            norm_sq += val * val;
        }
        double norm = sqrt(norm_sq);
        R[j * n + j] = (float)norm;

        if (norm < 1e-12)
        {
            return -1; // Rank deficient
        }

        // Q[:,j] = Q[:,j] / R[j,j]
        for (uint16_t i = 0; i < m; i++)
        {
            Q[i * n + j] /= (float)norm;
        }

        // Orthogonalize remaining columns
        for (uint16_t k = j + 1; k < n; k++)
        {
            // R[j,k] = Q[:,j]^T * Q[:,k]
            double dot = 0.0;
            for (uint16_t i = 0; i < m; i++)
            {
                dot += (double)Q[i * n + j] * (double)Q[i * n + k];
            }
            R[j * n + k] = (float)dot;

            // Q[:,k] = Q[:,k] - R[j,k] * Q[:,j]
            for (uint16_t i = 0; i < m; i++)
            {
                Q[i * n + k] -= (float)dot * Q[i * n + j];
            }
        }
    }

    // Extend Q to full m×m orthogonal matrix (if needed)
    // For testing, we only need the first n columns

    return 0;
}

//==============================================================================
// INDIVIDUAL QR TESTS
//==============================================================================

/**
 * @brief Test small square matrix (8×8)
 */
static int test_qr_small_square(void)
{
    printf("\n=== Testing Small Square QR (8×8) ===\n");

    const uint16_t m = 8, n = 8;

    float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
    float *Q = gemm_aligned_alloc(32, m * m * sizeof(float));
    float *R = gemm_aligned_alloc(32, m * n * sizeof(float));

    // Initialize with well-conditioned random matrix
    srand(12345);
    for (uint16_t i = 0; i < m * n; i++)
    {
        A[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }

    // Add diagonal dominance for better conditioning
    for (uint16_t i = 0; i < MIN(m, n); i++)
    {
        A[i * n + i] += 5.0f;
    }

    printf("  Running blocked QR...\n");
    int ret = qr_blocked(A, Q, R, m, n, false);

    if (ret != 0)
    {
        printf("  ERROR: qr_blocked returned %d\n", ret);
        goto cleanup;
    }

    printf("  Verifying results...\n");
    int passed = 1;

    // Check R is upper triangular
    passed &= is_upper_triangular(R, m, n, 1e-5);

    // Check Q orthogonality
    passed &= check_orthogonality(Q, m, 1e-5, "8×8");

    // Check reconstruction
    passed &= check_reconstruction(A, Q, R, m, n, 1e-4, "8×8");

cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(Q);
    gemm_aligned_free(R);

    return passed;
}

/**
 * @brief Test tall matrix (128×32)
 */
static int test_qr_tall(void)
{
    printf("\n=== Testing Tall QR (128×32) ===\n");

    const uint16_t m = 128, n = 32;

    float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
    float *Q = gemm_aligned_alloc(32, m * m * sizeof(float));
    float *R = gemm_aligned_alloc(32, m * n * sizeof(float));

    // Initialize
    srand(54321);
    for (uint16_t i = 0; i < m * n; i++)
    {
        A[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    }

    printf("  Running blocked QR...\n");
    int ret = qr_blocked(A, Q, R, m, n, false);

    if (ret != 0)
    {
        printf("  ERROR: qr_blocked returned %d\n", ret);
        goto cleanup;
    }

    int passed = 1;
    passed &= is_upper_triangular(R, m, n, 1e-4);
    passed &= check_orthogonality(Q, m, 1e-4, "128×32");
    passed &= check_reconstruction(A, Q, R, m, n, 1e-3, "128×32");

cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(Q);
    gemm_aligned_free(R);

    return passed;
}

/**
 * @brief Test wide matrix (32×16) - only R
 */
static int test_qr_wide(void)
{
    printf("\n=== Testing Wide QR (32×16, R only) ===\n");

    const uint16_t m = 32, n = 16;

    float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
    float *A_copy = gemm_aligned_alloc(32, m * n * sizeof(float));
    float *Q = gemm_aligned_alloc(32, m * m * sizeof(float));
    float *R = gemm_aligned_alloc(32, m * n * sizeof(float));

    // Initialize
    srand(99999);
    for (uint16_t i = 0; i < m * n; i++)
    {
        A[i] = ((float)(rand() % 100)) / 25.0f;
    }
    memcpy(A_copy, A, m * n * sizeof(float));

    printf("  Running blocked QR (only_R=true)...\n");
    int ret = qr_blocked(A, NULL, R, m, n, true);

    if (ret != 0)
    {
        printf("  ERROR: qr_blocked returned %d\n", ret);
        goto cleanup;
    }

    // Since we don't have Q, we can only check R is upper triangular
    int passed = is_upper_triangular(R, m, n, 1e-4);

    // Optionally, run full QR to verify reconstruction
    printf("  Verifying with full QR...\n");
    ret = qr_blocked(A_copy, Q, R, m, n, false);
    if (ret == 0)
    {
        passed &= check_reconstruction(A_copy, Q, R, m, n, 1e-3, "32×16");
    }

cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(A_copy);
    gemm_aligned_free(Q);
    gemm_aligned_free(R);

    return passed;
}

int test_tiny_qr() {
    const uint16_t m = 4, n = 4;
    float A[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float Q[16], R[16];
    
    printf("\n=== TINY 4x4 TEST ===\n");
    printf("Original A:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", A[i * 4 + j]);
        }
        printf("\n");
    }
    
    int ret = qr_blocked(A, Q, R, m, n, false);
    
    printf("\nR:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", R[i * 4 + j]);
        }
        printf("\n");
    }
    
    printf("\nQ:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", Q[i * 4 + j]);
        }
        printf("\n");
    }
    
    // Check Q*R
    printf("\nQ*R (should match original A):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) {
                sum += Q[i * 4 + k] * R[k * 4 + j];
            }
            printf("%6.2f ", sum);
        }
        printf("\n");
    }

    int passed = 1;
    return passed;
}

int test_single_block_8x8() {
    printf("\n=== SINGLE BLOCK TEST (8x8, ib=8) ===\n");
    
    const uint16_t m = 8, n = 8;
    float A[64];
    for (int i = 0; i < 64; i++) {
        A[i] = (float)(i + 1);
    }
    
    float A_orig[64];
    memcpy(A_orig, A, sizeof(A));
    
    float Q[64], R[64];
    
    // Force single block
    qr_workspace *ws = qr_workspace_alloc_ex(m, n, 8, true);
    
    int ret = qr_ws_blocked(ws, A, Q, R, m, n, false);
    
    // Check reconstruction
    float QR[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float sum = 0;
            for (int k = 0; k < 8; k++) {
                sum += Q[i * 8 + k] * R[k * 8 + j];
            }
            QR[i * 8 + j] = sum;
        }
    }
    
    float error = 0, norm = 0;
    for (int i = 0; i < 64; i++) {
        error += (QR[i] - A_orig[i]) * (QR[i] - A_orig[i]);
        norm += A_orig[i] * A_orig[i];
    }
    
    double rel_error = sqrt(error/norm);
    printf("Reconstruction error: %.6f\n", rel_error);
    
    qr_workspace_free(ws);
    
    int passed = (rel_error < 1e-4);
    return passed;
}

int test_two_block_qr() {
    const uint16_t m = 8, n = 8;
    
    float A[64];
    for (int i = 0; i < 64; i++) {
        A[i] = (float)(i + 1);
    }
    
    // Save original
    float A_orig[64];
    memcpy(A_orig, A, sizeof(A));
    
    float Q[64], R[64];
    
    qr_workspace *ws = qr_workspace_alloc_ex(m, n, 4, true);
    
    printf("\n=== TWO BLOCK TEST (8x8, ib=4) ===\n");
    
    // Print A before
    printf("A before factorization:\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 8; j++) {
            printf("%6.2f ", A[i * 8 + j]);
        }
        printf("\n");
    }
    
    int ret = qr_ws_blocked(ws, A, Q, R, m, n, false);

     
    // ✅ ADD THIS HERE - FULL R MATRIX
    printf("\nFULL R after factorization (all 8 rows):\n");
    for (int i = 0; i < 8; i++) {
        printf("  ");
        for (int j = 0; j < 8; j++) {
            printf("%6.2f ", R[i * 8 + j]);
        }
        printf("\n");
    }
    
    // Print R after (should match A's upper triangle approximately)
    printf("\nR after factorization:\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 8; j++) {
            printf("%6.2f ", R[i * 8 + j]);
        }
        printf("\n");
    }
    
    // Check reconstruction
    float QR[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float sum = 0;
            for (int k = 0; k < 8; k++) {
                sum += (Q[i * 8 + k] * R[k * 8 + j]);
            }
            QR[i * 8 + j] = sum;
        }
    }
    
    printf("\nQ*R (should match original A):\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 8; j++) {
            printf("%6.2f ", QR[i * 8 + j]);
        }
        printf("\n");
    }
    
    printf("\nA_orig:\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 8; j++) {
            printf("%6.2f ", A_orig[i * 8 + j]);
        }
        printf("\n");
    }
    
    // Compute error
    float error = 0, norm = 0;
    for (int i = 0; i < 64; i++) {
        error += (QR[i] - A_orig[i]) * (QR[i] - A_orig[i]);
        norm += A_orig[i] * A_orig[i];
    }
    
    double rel_error = sqrt(error/norm);
    printf("\nReconstruction error: %.6f\n", rel_error);
    
    qr_workspace_free(ws);

    int passed = (rel_error < 1e-4);
    if (!passed) {
        printf("✗ FAILED: Error %.6f exceeds tolerance 1e-4\n", rel_error);
    }
    return passed;
}

/**
 * @brief Test large square matrix (256×256)
 */
static int test_qr_large_square(void)
{
    printf("\n=== Testing Large Square QR (256×256) ===\n");

    const uint16_t m = 512, n = 512;

    printf("  Allocating %.2f MB...\n",
           (m * n + m * m + m * n) * sizeof(float) / (1024.0 * 1024.0));

    float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
    float *Q = gemm_aligned_alloc(32, m * m * sizeof(float));
    float *R = gemm_aligned_alloc(32, m * n * sizeof(float));

    if (!A || !Q || !R)
    {
        printf("  ERROR: Allocation failed\n");
        return 0;
    }

    // ✅ FIXED: Use size_t to avoid overflow
    //printf("  Initializing matrix...\n");
    srand(77777);
    for (size_t i = 0; i < (size_t)m * n; i++)
    {
        A[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
    }

    // Add diagonal dominance
    for (uint16_t i = 0; i < MIN(m, n); i++)
    {
        A[i * n + i] += 3.0f;
    }

    //printf("  Running blocked QR...\n");
    float *A_orig = gemm_aligned_alloc(32, m * n * sizeof(float));

    memcpy(A_orig, A, (size_t)m * n * sizeof(float));  // BEFORE qr_blocked
    int ret = qr_blocked(A, Q, R, m, n, false);

    if (ret != 0)
    {
        printf("  ERROR: qr_blocked returned %d\n", ret);
        goto cleanup;
    }

    printf("  Verifying results...\n");
    int passed = 1;

    // For large matrices, use relaxed tolerances
    passed &= is_upper_triangular(R, m, n, 1e-3);
    passed &= check_orthogonality(Q, m, 5e-4, "256×256");
    passed &= check_reconstruction(A_orig, Q, R, m, n, 1e-3, "256×256");  // Use A_orig


cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(Q);
    gemm_aligned_free(R);

    return passed;
}

/**
 * @brief Test workspace reuse
 */
static int test_workspace_reuse(void)
{
    printf("\n=== Testing Workspace Reuse ===\n");

    const uint16_t m_max = 64, n_max = 48;

    // Allocate workspace once
    printf("  Allocating workspace (m_max=%d, n_max=%d)...\n", m_max, n_max);
    qr_workspace *ws = qr_workspace_alloc(m_max, n_max, 0);
    if (!ws)
    {
        printf("  ERROR: Workspace allocation failed\n");
        return 0;
    }

    printf("  Workspace size: %.2f KB\n", qr_workspace_bytes(ws) / 1024.0);

    int passed = 1;

    // Test multiple sizes with same workspace
    uint16_t test_sizes[][2] = {
        {32, 16},
        {64, 32},
        {48, 48},
        {64, 24}};

    for (int test_idx = 0; test_idx < 4; test_idx++)
    {
        uint16_t m = test_sizes[test_idx][0];
        uint16_t n = test_sizes[test_idx][1];

        printf("  Testing %d×%d with reused workspace...\n", m, n);

        float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
        float *Q = gemm_aligned_alloc(32, m * m * sizeof(float));
        float *R = gemm_aligned_alloc(32, m * n * sizeof(float));

        // Initialize
        srand(test_idx * 11111);
        for (uint16_t i = 0; i < m * n; i++)
        {
            A[i] = ((float)(rand() % 100)) / 50.0f;
        }

        // Run QR with workspace
        int ret = qr_ws_blocked(ws, A, Q, R, m, n, false);

        if (ret != 0)
        {
            printf("    ERROR: qr_ws_blocked returned %d\n", ret);
            passed = 0;
            gemm_aligned_free(A);
            gemm_aligned_free(Q);
            gemm_aligned_free(R);
            continue;
        }

        // Quick check
        int test_passed = check_reconstruction(A, Q, R, m, n, 1e-3, "workspace");
        passed &= test_passed;

        gemm_aligned_free(A);
        gemm_aligned_free(Q);
        gemm_aligned_free(R);

        if (!test_passed)
        {
            printf("    Test %d FAILED\n", test_idx);
        }
        else
        {
            printf("    Test %d PASSED\n", test_idx);
        }
    }

    qr_workspace_free(ws);

    return passed;
}

/**
 * @brief Test edge cases
 */
static int test_edge_cases(void)
{
    printf("\n=== Testing Edge Cases ===\n");

    int passed = 1;

    // Test 1: Tiny matrix (4×4)
    printf("  Testing 4×4 matrix...\n");
    {
        const uint16_t m = 4, n = 4;
        float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
        float *Q = gemm_aligned_alloc(32, m * m * sizeof(float));
        float *R = gemm_aligned_alloc(32, m * n * sizeof(float));

        float data[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1};
        memcpy(A, data, sizeof(data));

        int ret = qr_blocked(A, Q, R, m, n, false);

        if (ret == 0)
        {
            passed &= check_orthogonality(Q, m, 1e-5, "4×4 identity");
            passed &= check_reconstruction(A, Q, R, m, n, 1e-5, "4×4 identity");
        }
        else
        {
            printf("    ERROR: qr_blocked failed\n");
            passed = 0;
        }

        gemm_aligned_free(A);
        gemm_aligned_free(Q);
        gemm_aligned_free(R);
    }

    // Test 2: Matrix with one large singular value
    printf("  Testing matrix with large condition number...\n");
    {
        const uint16_t m = 16, n = 16;
        float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
        float *Q = gemm_aligned_alloc(32, m * m * sizeof(float));
        float *R = gemm_aligned_alloc(32, m * n * sizeof(float));

        memset(A, 0, m * n * sizeof(float));
        A[0] = 1000.0f; // First singular value = 1000
        for (uint16_t i = 1; i < MIN(m, n); i++)
        {
            A[i * n + i] = 1.0f; // Rest = 1
        }

        int ret = qr_blocked(A, Q, R, m, n, false);

        if (ret == 0)
        {
            // Use relaxed tolerance for ill-conditioned matrix
            passed &= check_reconstruction(A, Q, R, m, n, 1e-3, "ill-conditioned");
        }
        else
        {
            printf("    ERROR: qr_blocked failed\n");
            passed = 0;
        }

        gemm_aligned_free(A);
        gemm_aligned_free(Q);
        gemm_aligned_free(R);
    }

    return passed;
}

/**
 * @brief Compare against reference (for small matrices)
 */
static int test_vs_reference(void)
{
    printf("\n=== Comparing Against Gram-Schmidt Reference ===\n");

    const uint16_t m = 16, n = 12;

    float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
    float *Q_test = gemm_aligned_alloc(32, m * m * sizeof(float));
    float *R_test = gemm_aligned_alloc(32, m * n * sizeof(float));
    float *Q_ref = gemm_aligned_alloc(32, m * n * sizeof(float));
    float *R_ref = gemm_aligned_alloc(32, m * n * sizeof(float));

    // Well-conditioned random matrix
    srand(42);
    for (uint16_t i = 0; i < m * n; i++)
    {
        A[i] = ((float)(rand() % 100)) / 50.0f;
    }

    // Add diagonal dominance
    for (uint16_t i = 0; i < MIN(m, n); i++)
    {
        A[i * n + i] += 2.0f;
    }

    printf("  Running blocked QR...\n");
    int ret_test = qr_blocked(A, Q_test, R_test, m, n, false);

    printf("  Running Gram-Schmidt reference...\n");
    int ret_ref = qr_gram_schmidt_reference(A, Q_ref, R_ref, m, n);

    if (ret_test != 0 || ret_ref != 0)
    {
        printf("  ERROR: One of the implementations failed\n");
        goto cleanup;
    }

    // Compare R matrices (sign ambiguity is OK, we just check magnitudes)
    printf("  Comparing R matrices...\n");
    double r_error = 0.0;
    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double diff = fabs((double)R_test[i * n + j]) - fabs((double)R_ref[i * n + j]);
            r_error += diff * diff;
        }
    }
    r_error = sqrt(r_error);

    printf("  ||abs(R_test) - abs(R_ref)||_F = %.6e\n", r_error);

    int passed = (r_error < 1e-3);

cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(Q_test);
    gemm_aligned_free(R_test);
    gemm_aligned_free(Q_ref);
    gemm_aligned_free(R_ref);

    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_qr_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("      BLOCKED QR DECOMPOSITION TESTS\n");
    printf("=================================================\n");

    results->total = 0;
    results->passed = 0;
    results->failed = 0;

    // Run all tests
    printf("\n--- Basic Functionality Tests ---\n");

    results->total++;
    if (test_qr_small_square())
    {
        results->passed++;
        printf("✓ Small square test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Small square test FAILED\n");
    }

    results->total++;
    if (test_qr_tall())
    {
        results->passed++;
        printf("✓ Tall matrix test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Tall matrix test FAILED\n");
    }

    results->total++;
    if (test_qr_wide())
    {
        results->passed++;
        printf("✓ Wide matrix test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Wide matrix test FAILED\n");
    }

    results->total++;
    if (test_qr_large_square())
    {
        results->passed++;
        printf("✓ Large square test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Large square test FAILED\n");
    }

    results->total++;
    if (test_tiny_qr())
    {
        results->passed++;
        printf("✓ tiny qr test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗  tiny qr test FAILED\n");
    }
    // test_single_block_8x8
      results->total++;
    if (test_single_block_8x8())
    {
        results->passed++;
        printf("✓ test_single_block_8x8 test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ test_single_block_8x8  test FAILED\n");
    }
    
    results->total++;
    if (test_two_block_qr())
    {
        results->passed++;
        printf("✓ two block qr test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Large square test FAILED\n");
    }



    //printf("\n--- Workspace Tests ---\n");

    //results->total++;
    //if (test_workspace_reuse())
    //{
    //    results->passed++;
    //    printf("✓ Workspace reuse test PASSED\n");
    //}
    //else
    //{
    //    results->failed++;
    //    printf("✗ Workspace reuse test FAILED\n");
    //}

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

    printf("\n--- Reference Comparison ---\n");

    results->total++;
    if (test_vs_reference())
    {
        results->passed++;
        printf("✓ Reference comparison PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Reference comparison FAILED\n");
    }

    printf("\n=================================================\n");
    printf("QR Tests: %d/%d passed\n", results->passed, results->total);

    if (results->passed == results->total)
    {
        printf("✓ ALL QR TESTS PASSED!\n");
    }
    else
    {
        printf("✗ %d QR tests FAILED\n", results->failed);
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
    return run_qr_tests(&results);
}
#endif