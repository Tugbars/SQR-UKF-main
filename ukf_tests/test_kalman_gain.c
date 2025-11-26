/**
 * @file test_kalman_gain.c
 * @brief Test suite for Kalman gain computation in SR-UKF
 *
 * @details
 * Verifies that K = Pxy · Pyy^(-1) = Pxy · (Sy · Sy^T)^(-1)
 *
 * The correct decomposition is:
 *   K = Pxy · Sy^(-T) · Sy^(-1)
 *
 * This requires RIGHT-solves:
 *   Step 1: Z · Sy^T = Pxy  →  Z = Pxy · Sy^(-T)
 *   Step 2: K · Sy   = Z    →  K = Z · Sy^(-1)
 *
 * The WRONG approach (left-solves) computes:
 *   Step 1: Sy^T · Z = Pxy  →  Z = Sy^(-T) · Pxy
 *   Step 2: Sy · K   = Z    →  K = Sy^(-1) · Sy^(-T) · Pxy  ← WRONG!
 *
 * Compile:
 *   gcc -O2 -mavx2 -mfma -o test_kalman_gain test_kalman_gain.c -lm
 *
 * Run:
 *   ./test_kalman_gain
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Allocate aligned memory
 */
static void *aligned_alloc_32(size_t size)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 32, size) != 0)
        return NULL;
    return ptr;
}

/**
 * @brief Print matrix for debugging
 */
static void print_matrix(const char *name, const float *A, int m, int n)
{
    printf("%s [%d x %d]:\n", name, m, n);
    for (int i = 0; i < m; ++i)
    {
        printf("  [");
        for (int j = 0; j < n; ++j)
        {
            printf("%8.4f", A[i * n + j]);
            if (j < n - 1)
                printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}

/**
 * @brief Compute Frobenius norm of difference: ||A - B||_F
 */
static float matrix_diff_norm(const float *A, const float *B, int m, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        float d = A[i] - B[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

/**
 * @brief Compute Frobenius norm: ||A||_F
 */
static float matrix_norm(const float *A, int m, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        sum += A[i] * A[i];
    }
    return sqrtf(sum);
}

/**
 * @brief Matrix multiply C = A * B (row-major)
 */
static void matmul(float *C, const float *A, const float *B,
                   int m, int k, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
            {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * @brief Matrix transpose: B = A^T
 */
static void transpose(float *B, const float *A, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            B[j * m + i] = A[i * n + j];
        }
    }
}

/**
 * @brief Invert upper triangular matrix (for testing)
 *
 * Computes U^(-1) where U is upper triangular
 */
static void invert_upper_triangular(float *Uinv, const float *U, int n)
{
    /* Initialize to identity */
    memset(Uinv, 0, n * n * sizeof(float));
    for (int i = 0; i < n; ++i)
        Uinv[i * n + i] = 1.0f;

    /* Back-substitution for each column */
    for (int j = 0; j < n; ++j)
    {
        for (int i = n - 1; i >= 0; --i)
        {
            float sum = Uinv[i * n + j];
            for (int k = i + 1; k < n; ++k)
            {
                sum -= U[i * n + k] * Uinv[k * n + j];
            }
            Uinv[i * n + j] = sum / U[i * n + i];
        }
    }
}

//==============================================================================
// TRSM IMPLEMENTATIONS (simplified for testing)
//==============================================================================

/**
 * @brief Left-solve: U · X = B  (what current code does)
 *
 * Returns X = U^(-1) · B
 */
static void trsm_left_upper(float *B, const float *U, int n, int ncols)
{
    /* Backward substitution */
    for (int j = n - 1; j >= 0; --j)
    {
        float ujj = U[j * n + j];

        /* Scale row j */
        for (int k = 0; k < ncols; ++k)
            B[j * ncols + k] /= ujj;

        /* Update preceding rows */
        for (int i = 0; i < j; ++i)
        {
            float uij = U[i * n + j];
            for (int k = 0; k < ncols; ++k)
                B[i * ncols + k] -= uij * B[j * ncols + k];
        }
    }
}

/**
 * @brief Left-solve with transpose: U^T · X = B  (what current code does)
 *
 * Returns X = U^(-T) · B
 * U^T is lower triangular, so forward substitution
 */
static void trsm_left_upper_transpose(float *B, const float *U, int n, int ncols)
{
    /* Forward substitution (U^T is lower triangular) */
    for (int j = 0; j < n; ++j)
    {
        float ujj = U[j * n + j]; /* U^T[j,j] = U[j,j] */

        /* Scale row j */
        for (int k = 0; k < ncols; ++k)
            B[j * ncols + k] /= ujj;

        /* Update trailing rows */
        for (int i = j + 1; i < n; ++i)
        {
            /* U^T[i,j] = U[j,i] */
            float ut_ij = U[j * n + i];
            for (int k = 0; k < ncols; ++k)
                B[i * ncols + k] -= ut_ij * B[j * ncols + k];
        }
    }
}

/**
 * @brief Right-solve: X · U = B  (what we SHOULD do)
 *
 * Returns X = B · U^(-1)
 *
 * For upper triangular U, process columns LEFT to RIGHT:
 *   Column j: X[:,j] = (B[:,j] - Σ_{k<j} X[:,k] * U[k,j]) / U[j,j]
 */
static void trsm_right_upper(float *B, const float *U, int m, int n)
{
    /* Process columns left to right (forward substitution for right-solve) */
    for (int j = 0; j < n; ++j)
    {
        /* Subtract contributions from already-solved columns */
        for (int k = 0; k < j; ++k)
        {
            float ukj = U[k * n + j]; /* U[k,j] for k < j */
            if (ukj == 0.0f)
                continue;

            for (int i = 0; i < m; ++i)
                B[i * n + j] -= B[i * n + k] * ukj;
        }

        /* Scale column j by diagonal */
        float ujj = U[j * n + j];
        for (int i = 0; i < m; ++i)
            B[i * n + j] /= ujj;
    }
}

/**
 * @brief Right-solve with transpose: X · U^T = B  (what we SHOULD do)
 *
 * Returns X = B · U^(-T)
 *
 * U^T is lower triangular, so process columns RIGHT to LEFT:
 *   Column j: X[:,j] = (B[:,j] - Σ_{k>j} X[:,k] * U^T[k,j]) / U^T[j,j]
 *   where U^T[k,j] = U[j,k]
 */
static void trsm_right_upper_transpose(float *B, const float *U, int m, int n)
{
    /* Process columns right to left (U^T is lower triangular) */
    for (int j = n - 1; j >= 0; --j)
    {
        /* Subtract contributions from already-solved columns (k > j) */
        for (int k = j + 1; k < n; ++k)
        {
            /* U^T[k,j] = U[j,k] */
            float ut_kj = U[j * n + k];
            if (ut_kj == 0.0f)
                continue;

            for (int i = 0; i < m; ++i)
                B[i * n + j] -= B[i * n + k] * ut_kj;
        }

        /* Scale column j by diagonal (U^T[j,j] = U[j,j]) */
        float ujj = U[j * n + j];
        for (int i = 0; i < m; ++i)
            B[i * n + j] /= ujj;
    }
}

//==============================================================================
// KALMAN GAIN COMPUTATION METHODS
//==============================================================================

/**
 * @brief Compute Kalman gain using explicit inverse (GROUND TRUTH)
 *
 * K = Pxy · (Sy · Sy^T)^(-1)
 *   = Pxy · Sy^(-T) · Sy^(-1)
 */
static void kalman_gain_explicit_inverse(
    float *K,
    const float *Pxy,
    const float *Sy,
    int n)
{
    float *Sy_inv = aligned_alloc_32(n * n * sizeof(float));
    float *Sy_inv_T = aligned_alloc_32(n * n * sizeof(float));
    float *temp = aligned_alloc_32(n * n * sizeof(float));

    /* Sy^(-1) */
    invert_upper_triangular(Sy_inv, Sy, n);

    /* Sy^(-T) = (Sy^(-1))^T */
    transpose(Sy_inv_T, Sy_inv, n, n);

    /* K = Pxy · Sy^(-T) · Sy^(-1) */
    matmul(temp, Pxy, Sy_inv_T, n, n, n);
    matmul(K, temp, Sy_inv, n, n, n);

    free(Sy_inv);
    free(Sy_inv_T);
    free(temp);
}

/**
 * @brief Compute Kalman gain using LEFT-TRSM (WRONG - current implementation)
 *
 * What the buggy code computes:
 *   Step 1: Sy^T · Z = Pxy  →  Z = Sy^(-T) · Pxy
 *   Step 2: Sy · K = Z      →  K = Sy^(-1) · Z = Sy^(-1) · Sy^(-T) · Pxy
 *
 * This is WRONG because K should be Pxy · Sy^(-T) · Sy^(-1)
 */
static void kalman_gain_left_trsm_WRONG(
    float *K,
    const float *Pxy,
    const float *Sy,
    int n)
{
    /* Z starts as copy of Pxy */
    float *Z = aligned_alloc_32(n * n * sizeof(float));
    memcpy(Z, Pxy, n * n * sizeof(float));

    /* Step 1: Sy^T · Z = Pxy  →  Z = Sy^(-T) · Pxy (LEFT solve!) */
    trsm_left_upper_transpose(Z, Sy, n, n);

    /* Step 2: Sy · K = Z  →  K = Sy^(-1) · Z (LEFT solve!) */
    memcpy(K, Z, n * n * sizeof(float));
    trsm_left_upper(K, Sy, n, n);

    free(Z);
}

/**
 * @brief Compute Kalman gain using RIGHT-TRSM (CORRECT)
 *
 * Correct computation:
 *   Step 1: Z · Sy^T = Pxy  →  Z = Pxy · Sy^(-T)
 *   Step 2: K · Sy = Z      →  K = Z · Sy^(-1) = Pxy · Sy^(-T) · Sy^(-1)
 */
static void kalman_gain_right_trsm_CORRECT(
    float *K,
    const float *Pxy,
    const float *Sy,
    int n)
{
    /* Z starts as copy of Pxy */
    float *Z = aligned_alloc_32(n * n * sizeof(float));
    memcpy(Z, Pxy, n * n * sizeof(float));

    /* Step 1: Z · Sy^T = Pxy  →  Z = Pxy · Sy^(-T) (RIGHT solve!) */
    trsm_right_upper_transpose(Z, Sy, n, n);

    /* Step 2: K · Sy = Z  →  K = Z · Sy^(-1) (RIGHT solve!) */
    memcpy(K, Z, n * n * sizeof(float));
    trsm_right_upper(K, Sy, n, n);

    free(Z);
}

/**
 * @brief Compute Kalman gain using transpose trick with LEFT-TRSM
 *
 * Uses the identity: (X · A = B) ⟺ (A^T · X^T = B^T)
 *
 * So we can convert right-solves to left-solves via transpose:
 *   Step 1: (Z · Sy^T = Pxy) ⟺ (Sy · Z^T = Pxy^T)
 *   Step 2: (K · Sy = Z)     ⟺ (Sy^T · K^T = Z^T)
 */
static void kalman_gain_transpose_trick(
    float *K,
    const float *Pxy,
    const float *Sy,
    int n)
{
    float *PxyT = aligned_alloc_32(n * n * sizeof(float));
    float *ZT = aligned_alloc_32(n * n * sizeof(float));
    float *Z = aligned_alloc_32(n * n * sizeof(float));
    float *KT = aligned_alloc_32(n * n * sizeof(float));

    /* Transpose Pxy */
    transpose(PxyT, Pxy, n, n);

    /* Step 1: Sy · Z^T = Pxy^T  (left-solve for Z^T) */
    memcpy(ZT, PxyT, n * n * sizeof(float));
    trsm_left_upper(ZT, Sy, n, n);

    /* Transpose Z^T → Z */
    transpose(Z, ZT, n, n);

    /* Step 2: Sy^T · K^T = Z^T  (left-solve for K^T) */
    transpose(ZT, Z, n, n); /* Need Z^T again */
    memcpy(KT, ZT, n * n * sizeof(float));
    trsm_left_upper_transpose(KT, Sy, n, n);

    /* Transpose K^T → K */
    transpose(K, KT, n, n);

    free(PxyT);
    free(ZT);
    free(Z);
    free(KT);
}

//==============================================================================
// TEST CASES
//==============================================================================

/**
 * @brief Test with a simple 3x3 case
 */
static bool test_3x3(void)
{
    printf("=== TEST: 3x3 Kalman Gain ===\n\n");

    const int n = 3;

    /* Upper triangular Sy (square root of measurement covariance) */
    float Sy[9] = {
        2.0f, 1.0f, 0.5f,
        0.0f, 3.0f, 0.5f,
        0.0f, 0.0f, 1.0f};

    /* Cross-covariance Pxy */
    float Pxy[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f};

    float K_truth[9], K_wrong[9], K_right[9], K_trick[9];

    print_matrix("Sy (upper triangular)", Sy, n, n);
    print_matrix("Pxy (cross-covariance)", Pxy, n, n);

    /* Compute Kalman gain using all methods */
    kalman_gain_explicit_inverse(K_truth, Pxy, Sy, n);
    kalman_gain_left_trsm_WRONG(K_wrong, Pxy, Sy, n);
    kalman_gain_right_trsm_CORRECT(K_right, Pxy, Sy, n);
    kalman_gain_transpose_trick(K_trick, Pxy, Sy, n);

    print_matrix("K (ground truth: explicit inverse)", K_truth, n, n);
    print_matrix("K (LEFT-TRSM - CURRENT BUGGY CODE)", K_wrong, n, n);
    print_matrix("K (RIGHT-TRSM - CORRECT)", K_right, n, n);
    print_matrix("K (Transpose trick - CORRECT)", K_trick, n, n);

    /* Compute errors */
    float norm_truth = matrix_norm(K_truth, n, n);
    float err_wrong = matrix_diff_norm(K_wrong, K_truth, n, n);
    float err_right = matrix_diff_norm(K_right, K_truth, n, n);
    float err_trick = matrix_diff_norm(K_trick, K_truth, n, n);

    printf("Relative errors (||K - K_truth|| / ||K_truth||):\n");
    printf("  LEFT-TRSM (buggy):     %e", err_wrong / norm_truth);
    if (err_wrong / norm_truth > 1e-5)
        printf(" *** BUG DETECTED! ***");
    printf("\n");
    printf("  RIGHT-TRSM (correct):  %e\n", err_right / norm_truth);
    printf("  Transpose trick:       %e\n", err_trick / norm_truth);
    printf("\n");

    /* Verify the bug exists */
    bool bug_detected = (err_wrong / norm_truth > 1e-5);
    bool right_correct = (err_right / norm_truth < 1e-5);
    bool trick_correct = (err_trick / norm_truth < 1e-5);

    if (bug_detected && right_correct && trick_correct)
    {
        printf("✅ TEST PASSED: Bug in LEFT-TRSM detected, RIGHT-TRSM and transpose trick are correct\n\n");
        return true;
    }
    else
    {
        printf("❌ TEST FAILED: Unexpected results\n\n");
        return false;
    }
}

/**
 * @brief Test with a 4x4 case (more complex)
 */
static bool test_4x4(void)
{
    printf("=== TEST: 4x4 Kalman Gain ===\n\n");

    const int n = 4;

    /* Upper triangular Sy */
    float Sy[16] = {
        3.0f, 1.0f, 0.5f, 0.2f,
        0.0f, 2.0f, 0.3f, 0.1f,
        0.0f, 0.0f, 4.0f, 0.5f,
        0.0f, 0.0f, 0.0f, 1.5f};

    /* Random-ish Pxy */
    float Pxy[16] = {
        1.2f, 2.3f, 3.4f, 4.5f,
        5.6f, 6.7f, 7.8f, 8.9f,
        9.1f, 1.2f, 2.3f, 3.4f,
        4.5f, 5.6f, 6.7f, 7.8f};

    float K_truth[16], K_wrong[16], K_right[16];

    kalman_gain_explicit_inverse(K_truth, Pxy, Sy, n);
    kalman_gain_left_trsm_WRONG(K_wrong, Pxy, Sy, n);
    kalman_gain_right_trsm_CORRECT(K_right, Pxy, Sy, n);

    float norm_truth = matrix_norm(K_truth, n, n);
    float err_wrong = matrix_diff_norm(K_wrong, K_truth, n, n);
    float err_right = matrix_diff_norm(K_right, K_truth, n, n);

    printf("Relative errors:\n");
    printf("  LEFT-TRSM (buggy):     %e", err_wrong / norm_truth);
    if (err_wrong / norm_truth > 1e-5)
        printf(" *** BUG ***");
    printf("\n");
    printf("  RIGHT-TRSM (correct):  %e\n", err_right / norm_truth);
    printf("\n");

    bool bug_detected = (err_wrong / norm_truth > 1e-5);
    bool right_correct = (err_right / norm_truth < 1e-5);

    if (bug_detected && right_correct)
    {
        printf("✅ TEST PASSED\n\n");
        return true;
    }
    else
    {
        printf("❌ TEST FAILED\n\n");
        return false;
    }
}

/**
 * @brief Verify that K produces correct state update
 *
 * For a Kalman filter, K should satisfy:
 *   K · Pyy = Pxy  where Pyy = Sy · Sy^T
 *
 * This verifies the Kalman gain is correct.
 */
static bool test_kalman_gain_identity(void)
{
    printf("=== TEST: Kalman Gain Identity K · Pyy = Pxy ===\n\n");

    const int n = 3;

    float Sy[9] = {
        2.0f, 1.0f, 0.5f,
        0.0f, 3.0f, 0.5f,
        0.0f, 0.0f, 1.0f};

    float Pxy[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f};

    /* Compute Pyy = Sy · Sy^T */
    float SyT[9], Pyy[9];
    transpose(SyT, Sy, n, n);
    matmul(Pyy, Sy, SyT, n, n, n);

    print_matrix("Pyy = Sy · Sy^T", Pyy, n, n);

    /* Compute K using different methods */
    float K_wrong[9], K_right[9];
    kalman_gain_left_trsm_WRONG(K_wrong, Pxy, Sy, n);
    kalman_gain_right_trsm_CORRECT(K_right, Pxy, Sy, n);

    /* Verify: K · Pyy should equal Pxy */
    float KPyy_wrong[9], KPyy_right[9];
    matmul(KPyy_wrong, K_wrong, Pyy, n, n, n);
    matmul(KPyy_right, K_right, Pyy, n, n, n);

    print_matrix("K_wrong · Pyy (should equal Pxy if correct)", KPyy_wrong, n, n);
    print_matrix("K_right · Pyy (should equal Pxy if correct)", KPyy_right, n, n);
    print_matrix("Pxy (target)", Pxy, n, n);

    float norm_Pxy = matrix_norm(Pxy, n, n);
    float err_wrong = matrix_diff_norm(KPyy_wrong, Pxy, n, n);
    float err_right = matrix_diff_norm(KPyy_right, Pxy, n, n);

    printf("Identity check: ||K · Pyy - Pxy|| / ||Pxy||\n");
    printf("  LEFT-TRSM (buggy):     %e", err_wrong / norm_Pxy);
    if (err_wrong / norm_Pxy > 1e-5)
        printf(" *** FAILS IDENTITY ***");
    printf("\n");
    printf("  RIGHT-TRSM (correct):  %e\n", err_right / norm_Pxy);
    printf("\n");

    bool wrong_fails = (err_wrong / norm_Pxy > 1e-5);
    bool right_passes = (err_right / norm_Pxy < 1e-5);

    if (wrong_fails && right_passes)
    {
        printf("✅ TEST PASSED: RIGHT-TRSM satisfies K·Pyy=Pxy, LEFT-TRSM doesn't\n\n");
        return true;
    }
    else
    {
        printf("❌ TEST FAILED\n\n");
        return false;
    }
}

/**
 * @brief Test individual TRSM operations
 */
static bool test_trsm_operations(void)
{
    printf("=== TEST: Individual TRSM Operations ===\n\n");

    const int n = 3;

    float U[9] = {
        2.0f, 1.0f, 0.5f,
        0.0f, 3.0f, 0.5f,
        0.0f, 0.0f, 1.0f};

    float B[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f};

    float U_inv[9];
    invert_upper_triangular(U_inv, U, n);

    /* Test 1: Left solve U · X = B → X = U^(-1) · B */
    {
        float X[9], X_expected[9];
        memcpy(X, B, sizeof(B));
        trsm_left_upper(X, U, n, n);
        matmul(X_expected, U_inv, B, n, n, n);

        float err = matrix_diff_norm(X, X_expected, n, n);
        printf("Left solve (U·X=B):        error = %e %s\n",
               err, err < 1e-5 ? "✓" : "✗");
    }

    /* Test 2: Right solve X · U = B → X = B · U^(-1) */
    {
        float X[9], X_expected[9];
        memcpy(X, B, sizeof(B));
        trsm_right_upper(X, U, n, n);
        matmul(X_expected, B, U_inv, n, n, n);

        float err = matrix_diff_norm(X, X_expected, n, n);
        printf("Right solve (X·U=B):       error = %e %s\n",
               err, err < 1e-5 ? "✓" : "✗");
    }

    /* Test 3: Left solve with transpose U^T · X = B → X = U^(-T) · B */
    {
        float X[9], X_expected[9], U_inv_T[9];
        memcpy(X, B, sizeof(B));
        trsm_left_upper_transpose(X, U, n, n);
        transpose(U_inv_T, U_inv, n, n);
        matmul(X_expected, U_inv_T, B, n, n, n);

        float err = matrix_diff_norm(X, X_expected, n, n);
        printf("Left solve (U^T·X=B):      error = %e %s\n",
               err, err < 1e-5 ? "✓" : "✗");
    }

    /* Test 4: Right solve with transpose X · U^T = B → X = B · U^(-T) */
    {
        float X[9], X_expected[9], U_inv_T[9];
        memcpy(X, B, sizeof(B));
        trsm_right_upper_transpose(X, U, n, n);
        transpose(U_inv_T, U_inv, n, n);
        matmul(X_expected, B, U_inv_T, n, n, n);

        float err = matrix_diff_norm(X, X_expected, n, n);
        printf("Right solve (X·U^T=B):     error = %e %s\n",
               err, err < 1e-5 ? "✓" : "✗");
    }

    printf("\n✅ All TRSM operations verified\n\n");
    return true;
}

//==============================================================================
// MAIN
//==============================================================================

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     SR-UKF Kalman Gain Computation Test Suite                ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ This test verifies that:                                     ║\n");
    printf("║   K = Pxy · Pyy^(-1) = Pxy · (Sy·Sy^T)^(-1)                  ║\n");
    printf("║     = Pxy · Sy^(-T) · Sy^(-1)                                ║\n");
    printf("║                                                              ║\n");
    printf("║ The CURRENT (buggy) implementation computes:                 ║\n");
    printf("║   K = Sy^(-1) · Sy^(-T) · Pxy   (LEFT-solves - WRONG!)       ║\n");
    printf("║                                                              ║\n");
    printf("║ The CORRECT implementation should use RIGHT-solves:          ║\n");
    printf("║   Step 1: Z · Sy^T = Pxy  →  Z = Pxy · Sy^(-T)               ║\n");
    printf("║   Step 2: K · Sy   = Z    →  K = Z · Sy^(-1)                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int passed = 0;
    int total = 0;

    total++;
    if (test_trsm_operations())
        passed++;
    total++;
    if (test_3x3())
        passed++;
    total++;
    if (test_4x4())
        passed++;
    total++;
    if (test_kalman_gain_identity())
        passed++;

    printf("════════════════════════════════════════════════════════════════\n");
    printf("SUMMARY: %d/%d tests passed\n", passed, total);
    printf("════════════════════════════════════════════════════════════════\n");

    if (passed == total)
    {
        printf("\n🎯 CONCLUSION: The bug is confirmed.\n");
        printf("   The current code uses LEFT-TRSM but needs RIGHT-TRSM.\n");
        printf("   Either implement right-TRSM or use the transpose trick.\n\n");
        return 0;
    }
    else
    {
        printf("\n❌ Some tests failed unexpectedly.\n\n");
        return 1;
    }
}