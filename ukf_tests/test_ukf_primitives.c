/**
 * @file test_ukf_primitives.c
 * @brief Unit tests for SR-UKF primitive functions
 *
 * Tests:
 * - create_weights: UKF weight computation (Wm, Wc)
 * - create_sigma_point_matrix: Sigma point generation from mean + SR covariance
 * - multiply_sigma_point_matrix_to_weights: Weighted mean computation
 * - compute_transition_function: Applying dynamics to sigma points
 * - build_YTc_fused: Fused centering + transpose
 * - build_Aprime_column_major: Augmented matrix construction for QR
 * - transpose_square_inplace: In-place matrix transpose
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
// TEST UTILITIES
//==============================================================================

/**
 * @brief Compute absolute error
 */
static float abs_error(float a, float b)
{
    return fabsf(a - b);
}

/**
 * @brief Compute relative error (handles near-zero values)
 */
static double rel_error(double a, double b)
{
    double max_abs = fmax(fabs(a), fabs(b));
    if (max_abs < 1e-10)
        return 0.0;
    return fabs(a - b) / max_abs;
}

/**
 * @brief Compute Frobenius norm of matrix difference
 */
static double matrix_diff_norm(const float *A, const float *B, 
                               size_t m, size_t n, size_t lda, size_t ldb)
{
    double sum = 0.0;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            double diff = (double)A[i * lda + j] - (double)B[i * ldb + j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
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
                U[i * ld + j] = 0.0f; // Lower triangle = 0
            }
            else if (j == i)
            {
                // Positive diagonal
                U[i * ld + j] = diag_boost + ((float)(rand() % 100) + 10.0f) / 50.0f;
            }
            else
            {
                // Upper triangle
                U[i * ld + j] = ((float)(rand() % 200) - 100.0f) / 100.0f;
            }
        }
    }
}

//==============================================================================
// REFERENCE IMPLEMENTATIONS (for comparison)
//==============================================================================

/**
 * @brief Reference implementation of create_weights
 */
static void create_weights_reference(float *Wc, float *Wm,
                                     float alpha, float beta, float kappa,
                                     uint8_t L)
{
    const size_t N = 2u * L + 1u;
    const float Lf = (float)L;
    
    // λ = α²(L + κ) - L
    const float lambda = alpha * alpha * (Lf + kappa) - Lf;
    
    // Wm[0] = λ / (L + λ)
    Wm[0] = lambda / (Lf + lambda);
    
    // Wc[0] = Wm[0] + (1 - α² + β)
    Wc[0] = Wm[0] + 1.0f - alpha * alpha + beta;
    
    // Wm[i] = Wc[i] = 1 / (2(L + λ)) for i > 0
    const float w_tail = 0.5f / (Lf + lambda);
    for (size_t i = 1; i < N; i++)
    {
        Wm[i] = w_tail;
        Wc[i] = w_tail;
    }
}

/**
 * @brief Reference implementation of sigma point matrix creation
 */
static void create_sigma_point_matrix_reference(float *X,
                                                const float *x,
                                                const float *S,
                                                float alpha, float kappa,
                                                uint8_t L)
{
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    const float gamma = alpha * sqrtf((float)L + kappa);
    
    for (size_t i = 0; i < Ls; i++)
    {
        float *Xi = X + i * N;
        const float *Si = S + i * Ls;
        
        // Column 0: mean
        Xi[0] = x[i];
        
        // Columns 1..L: x + γ*S[i,:]
        for (size_t j = 0; j < Ls; j++)
        {
            Xi[1 + j] = x[i] + gamma * Si[j];
        }
        
        // Columns L+1..2L: x - γ*S[i,:]
        for (size_t j = 0; j < Ls; j++)
        {
            Xi[1 + Ls + j] = x[i] - gamma * Si[j];
        }
    }
}

/**
 * @brief Reference implementation of weighted mean
 */
static void weighted_mean_reference(float *result, const float *X, const float *W,
                                    size_t L, size_t N)
{
    for (size_t i = 0; i < L; i++)
    {
        double sum = 0.0;
        for (size_t j = 0; j < N; j++)
        {
            sum += (double)W[j] * (double)X[i * N + j];
        }
        result[i] = (float)sum;
    }
}

/**
 * @brief Reference implementation of YTc = (Y - y)^T (transposed centered Y)
 */
static void build_YTc_reference(float *YTc, const float *Y, const float *y,
                                size_t L, size_t N, size_t N8)
{
    // YTc[j,i] = Y[i,j] - y[i]
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < L; i++)
        {
            YTc[j * L + i] = Y[i * N + j] - y[i];
        }
    }
    
    // Zero-pad to N8
    for (size_t j = N; j < N8; j++)
    {
        for (size_t i = 0; i < L; i++)
        {
            YTc[j * L + i] = 0.0f;
        }
    }
}

/**
 * @brief Reference implementation of Aprime column-major construction
 */
static void build_Aprime_column_major_reference(float *Aprime,
                                                const float *X, const float *x,
                                                const float *Rsr,
                                                float w1s,
                                                size_t L, size_t N, size_t M, size_t K)
{
    // Build each column i of Aprime
    for (size_t i = 0; i < L; i++)
    {
        float *col = Aprime + i * M;
        const float *Xi = X + i * N;
        const float xi = x[i];
        
        // Rows 0..K-1: weighted deviations
        for (size_t r = 0; r < K; r++)
        {
            col[r] = w1s * (Xi[r + 1] - xi);
        }
        
        // Rows K..M-1: SR noise
        const float *Rsri = Rsr + i * L;
        for (size_t t = 0; t < L; t++)
        {
            col[K + t] = Rsri[t];
        }
    }
}

/**
 * @brief Reference implementation of in-place square transpose
 */
static void transpose_square_reference(float *A, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = i + 1; j < n; j++)
        {
            float tmp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = tmp;
        }
    }
}

//==============================================================================
// TEST: create_weights
//==============================================================================

/**
 * @brief Test UKF weight computation
 */
static int test_create_weights(void)
{
    printf("\n=== Testing create_weights ===\n");
    
    int passed = 1;
    
    // Test various parameter combinations
    struct {
        uint8_t L;
        float alpha;
        float beta;
        float kappa;
    } test_cases[] = {
        {4,  1e-3f, 2.0f, 0.0f},    // Standard UKF parameters
        {8,  1e-3f, 2.0f, 0.0f},    // Larger state
        {16, 1e-3f, 2.0f, 0.0f},    // Even larger
        {4,  0.5f,  2.0f, 0.0f},    // Different alpha
        {4,  1e-3f, 2.0f, 3.0f},    // Non-zero kappa
        {6,  0.1f,  1.0f, -3.0f},   // Negative kappa (κ = 3 - L)
    };
    
    const int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    
    for (int tc = 0; tc < num_cases; tc++)
    {
        uint8_t L = test_cases[tc].L;
        float alpha = test_cases[tc].alpha;
        float beta = test_cases[tc].beta;
        float kappa = test_cases[tc].kappa;
        
        const size_t N = 2u * L + 1u;
        
        printf("  Testing L=%d, α=%.3f, β=%.1f, κ=%.1f...\n", 
               L, alpha, beta, kappa);
        
        float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
        float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
        float *Wm_ref = gemm_aligned_alloc(32, N * sizeof(float));
        float *Wc_ref = gemm_aligned_alloc(32, N * sizeof(float));
        
        if (!Wm || !Wc || !Wm_ref || !Wc_ref)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_weights;
        }
        
        // Compute using implementation under test
        create_weights(Wc, Wm, alpha, beta, kappa, L);
        
        // Compute reference
        create_weights_reference(Wc_ref, Wm_ref, alpha, beta, kappa, L);
        
        // Compare
        double max_err_Wm = 0.0;
        double max_err_Wc = 0.0;
        
        for (size_t i = 0; i < N; i++)
        {
            double err_m = fabs((double)Wm[i] - (double)Wm_ref[i]);
            double err_c = fabs((double)Wc[i] - (double)Wc_ref[i]);
            if (err_m > max_err_Wm) max_err_Wm = err_m;
            if (err_c > max_err_Wc) max_err_Wc = err_c;
        }
        
        if (max_err_Wm > 1e-6 || max_err_Wc > 1e-6)
        {
            printf("    FAILED: max_err_Wm=%.6e, max_err_Wc=%.6e\n", 
                   max_err_Wm, max_err_Wc);
            passed = 0;
        }
        
        // Check that Wm sums to 1 (unscented transform property)
        double sum_Wm = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            sum_Wm += (double)Wm[i];
        }
        
        if (fabs(sum_Wm - 1.0) > 1e-5)
        {
            printf("    FAILED: sum(Wm) = %.6f (expected 1.0)\n", sum_Wm);
            passed = 0;
        }
        
        // Check tail weights are equal
        int tail_equal = 1;
        for (size_t i = 2; i < N; i++)
        {
            if (fabs(Wm[i] - Wm[1]) > 1e-7 || fabs(Wc[i] - Wc[1]) > 1e-7)
            {
                tail_equal = 0;
                break;
            }
        }
        
        if (!tail_equal)
        {
            printf("    FAILED: tail weights not equal\n");
            passed = 0;
        }
        
        if (max_err_Wm <= 1e-6 && max_err_Wc <= 1e-6 && 
            fabs(sum_Wm - 1.0) <= 1e-5 && tail_equal)
        {
            printf("    PASSED\n");
        }
        
cleanup_weights:
        gemm_aligned_free(Wm);
        gemm_aligned_free(Wc);
        gemm_aligned_free(Wm_ref);
        gemm_aligned_free(Wc_ref);
    }
    
    return passed;
}

//==============================================================================
// TEST: create_sigma_point_matrix
//==============================================================================

/**
 * @brief Test sigma point matrix generation
 */
static int test_create_sigma_point_matrix(void)
{
    printf("\n=== Testing create_sigma_point_matrix ===\n");
    
    int passed = 1;
    
    uint8_t test_L[] = {4, 8, 16, 32};
    const int num_L = sizeof(test_L) / sizeof(test_L[0]);
    
    for (int tc = 0; tc < num_L; tc++)
    {
        uint8_t L = test_L[tc];
        const size_t Ls = (size_t)L;
        const size_t N = 2u * Ls + 1u;
        
        printf("  Testing L=%d...\n", L);
        
        float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
        float *S = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
        float *X_ref = gemm_aligned_alloc(32, Ls * N * sizeof(float));
        
        if (!x || !S || !X || !X_ref)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_sigma;
        }
        
        // Generate test data
        generate_random_vector(x, Ls, tc * 11111);
        generate_random_upper_triangular(S, Ls, Ls, tc * 22222, 1.0f);
        
        const float alpha = 1e-3f;
        const float kappa = 0.0f;
        
        // Compute using implementation under test
        create_sigma_point_matrix(X, x, S, alpha, kappa, L);
        
        // Compute reference
        create_sigma_point_matrix_reference(X_ref, x, S, alpha, kappa, L);
        
        // Compare
        double diff = matrix_diff_norm(X, X_ref, Ls, N, N, N);
        double ref_norm = matrix_diff_norm(X_ref, X_ref, Ls, N, N, N);
        double rel_diff = (ref_norm > 1e-10) ? diff / ref_norm : diff;
        
        if (rel_diff > 1e-5)
        {
            printf("    FAILED: relative diff = %.6e\n", rel_diff);
            passed = 0;
        }
        
        // Check property: X[:,0] = x (mean in first column)
        double mean_err = 0.0;
        for (size_t i = 0; i < Ls; i++)
        {
            mean_err += fabs((double)X[i * N + 0] - (double)x[i]);
        }
        
        if (mean_err > 1e-6)
        {
            printf("    FAILED: mean not in column 0 (err=%.6e)\n", mean_err);
            passed = 0;
        }
        
        // Check property: X[:,1+j] + X[:,L+1+j] = 2*x (symmetry)
        double sym_err = 0.0;
        for (size_t i = 0; i < Ls; i++)
        {
            for (size_t j = 0; j < Ls; j++)
            {
                double plus = (double)X[i * N + 1 + j];
                double minus = (double)X[i * N + 1 + Ls + j];
                double expected_sum = 2.0 * (double)x[i];
                sym_err += fabs(plus + minus - expected_sum);
            }
        }
        
        if (sym_err > 1e-4)
        {
            printf("    FAILED: symmetry check (err=%.6e)\n", sym_err);
            passed = 0;
        }
        
        if (rel_diff <= 1e-5 && mean_err <= 1e-6 && sym_err <= 1e-4)
        {
            printf("    PASSED (rel_diff=%.2e, sym_err=%.2e)\n", rel_diff, sym_err);
        }
        
cleanup_sigma:
        gemm_aligned_free(x);
        gemm_aligned_free(S);
        gemm_aligned_free(X);
        gemm_aligned_free(X_ref);
    }
    
    return passed;
}

//==============================================================================
// TEST: multiply_sigma_point_matrix_to_weights
//==============================================================================

/**
 * @brief Test weighted mean computation
 */
static int test_weighted_mean(void)
{
    printf("\n=== Testing multiply_sigma_point_matrix_to_weights ===\n");
    
    int passed = 1;
    
    uint8_t test_L[] = {4, 8, 16, 32, 64};
    const int num_L = sizeof(test_L) / sizeof(test_L[0]);
    
    for (int tc = 0; tc < num_L; tc++)
    {
        uint8_t L = test_L[tc];
        const size_t Ls = (size_t)L;
        const size_t N = 2u * Ls + 1u;
        
        printf("  Testing L=%d (N=%zu)...\n", L, N);
        
        float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
        float *W = gemm_aligned_alloc(32, N * sizeof(float));
        float *result = gemm_aligned_alloc(32, Ls * sizeof(float));
        float *result_ref = gemm_aligned_alloc(32, Ls * sizeof(float));
        
        if (!X || !W || !result || !result_ref)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_mean;
        }
        
        // Generate random sigma points and weights
        srand(tc * 33333);
        for (size_t i = 0; i < Ls * N; i++)
        {
            X[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        }
        
        // Generate weights that sum to 1
        float sum = 0.0f;
        for (size_t i = 0; i < N; i++)
        {
            W[i] = ((float)(rand() % 100) + 1.0f) / 100.0f;
            sum += W[i];
        }
        for (size_t i = 0; i < N; i++)
        {
            W[i] /= sum;
        }
        
        // Compute using implementation under test
        multiply_sigma_point_matrix_to_weights(result, X, W, L);
        
        // Compute reference
        weighted_mean_reference(result_ref, X, W, Ls, N);
        
        // Compare
        double max_err = 0.0;
        for (size_t i = 0; i < Ls; i++)
        {
            double err = fabs((double)result[i] - (double)result_ref[i]);
            if (err > max_err) max_err = err;
        }
        
        if (max_err > 1e-4)
        {
            printf("    FAILED: max_err = %.6e\n", max_err);
            passed = 0;
        }
        else
        {
            printf("    PASSED (max_err=%.2e)\n", max_err);
        }
        
cleanup_mean:
        gemm_aligned_free(X);
        gemm_aligned_free(W);
        gemm_aligned_free(result);
        gemm_aligned_free(result_ref);
    }
    
    return passed;
}

//==============================================================================
// TEST: build_YTc_fused
//==============================================================================

/**
 * @brief Test fused centering + transpose
 */
static int test_build_YTc_fused(void)
{
    printf("\n=== Testing build_YTc_fused ===\n");
    
    int passed = 1;
    
    struct {
        size_t L;
        size_t N;
    } test_cases[] = {
        {4, 9},      // Small (scalar path)
        {8, 17},     // Medium
        {16, 33},    // Triggers SIMD
        {32, 65},    // Larger
        {64, 129},   // Large
    };
    
    const int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    
    for (int tc = 0; tc < num_cases; tc++)
    {
        size_t L = test_cases[tc].L;
        size_t N = test_cases[tc].N;
        size_t N8 = (N + 7u) & ~7u; // Round up to multiple of 8
        
        printf("  Testing L=%zu, N=%zu, N8=%zu...\n", L, N, N8);
        
        float *Y = gemm_aligned_alloc(32, L * N * sizeof(float));
        float *y = gemm_aligned_alloc(32, L * sizeof(float));
        float *YTc = gemm_aligned_alloc(32, N8 * L * sizeof(float));
        float *YTc_ref = gemm_aligned_alloc(32, N8 * L * sizeof(float));
        
        if (!Y || !y || !YTc || !YTc_ref)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_ytc;
        }
        
        // Generate random data
        srand(tc * 44444);
        for (size_t i = 0; i < L * N; i++)
        {
            Y[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        }
        for (size_t i = 0; i < L; i++)
        {
            y[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        }
        
        // Initialize outputs
        memset(YTc, 0, N8 * L * sizeof(float));
        memset(YTc_ref, 0, N8 * L * sizeof(float));
        
        // Compute using implementation under test
        build_YTc_fused(YTc, Y, y, L, N, N8);
        
        // Compute reference
        build_YTc_reference(YTc_ref, Y, y, L, N, N8);
        
        // Compare
        double diff = matrix_diff_norm(YTc, YTc_ref, N8, L, L, L);
        
        if (diff > 1e-5)
        {
            printf("    FAILED: diff = %.6e\n", diff);
            passed = 0;
            
            // Debug: show first mismatch
            for (size_t j = 0; j < N8 && j < 5; j++)
            {
                for (size_t i = 0; i < L && i < 5; i++)
                {
                    float got = YTc[j * L + i];
                    float exp = YTc_ref[j * L + i];
                    if (fabsf(got - exp) > 1e-6)
                    {
                        printf("      YTc[%zu,%zu]: got %.6f, expected %.6f\n",
                               j, i, got, exp);
                    }
                }
            }
        }
        else
        {
            printf("    PASSED (diff=%.2e)\n", diff);
        }
        
cleanup_ytc:
        gemm_aligned_free(Y);
        gemm_aligned_free(y);
        gemm_aligned_free(YTc);
        gemm_aligned_free(YTc_ref);
    }
    
    return passed;
}

//==============================================================================
// TEST: build_Aprime_column_major
//==============================================================================

/**
 * @brief Test Aprime construction for QR
 */
static int test_build_Aprime_column_major(void)
{
    printf("\n=== Testing build_Aprime_column_major ===\n");
    
    int passed = 1;
    
    uint8_t test_L[] = {4, 8, 16, 32};
    const int num_L = sizeof(test_L) / sizeof(test_L[0]);
    
    for (int tc = 0; tc < num_L; tc++)
    {
        size_t L = (size_t)test_L[tc];
        size_t N = 2u * L + 1u;
        size_t K = 2u * L;
        size_t M = 3u * L;
        
        printf("  Testing L=%zu (M=%zu, K=%zu)...\n", L, M, K);
        
        float *X = gemm_aligned_alloc(32, L * N * sizeof(float));
        float *x = gemm_aligned_alloc(32, L * sizeof(float));
        float *Rsr = gemm_aligned_alloc(32, L * L * sizeof(float));
        float *Aprime = gemm_aligned_alloc(32, M * L * sizeof(float));
        float *Aprime_ref = gemm_aligned_alloc(32, M * L * sizeof(float));
        
        if (!X || !x || !Rsr || !Aprime || !Aprime_ref)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_aprime;
        }
        
        // Generate random data
        srand(tc * 55555);
        for (size_t i = 0; i < L * N; i++)
        {
            X[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        }
        for (size_t i = 0; i < L; i++)
        {
            x[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        }
        generate_random_upper_triangular(Rsr, L, L, tc * 66666, 1.0f);
        
        float w1s = sqrtf(0.5f / (float)L); // Typical weight sqrt
        
        // Initialize outputs
        memset(Aprime, 0, M * L * sizeof(float));
        memset(Aprime_ref, 0, M * L * sizeof(float));
        
        // Compute using implementation under test
        build_Aprime_column_major(Aprime, X, x, Rsr, w1s, L, N, M, K);
        
        // Compute reference
        build_Aprime_column_major_reference(Aprime_ref, X, x, Rsr, w1s, L, N, M, K);
        
        // Compare (Aprime is column-major: M rows, L columns)
        double max_err = 0.0;
        for (size_t col = 0; col < L; col++)
        {
            for (size_t row = 0; row < M; row++)
            {
                double got = (double)Aprime[col * M + row];
                double exp = (double)Aprime_ref[col * M + row];
                double err = fabs(got - exp);
                if (err > max_err) max_err = err;
            }
        }
        
        if (max_err > 1e-5)
        {
            printf("    FAILED: max_err = %.6e\n", max_err);
            passed = 0;
        }
        else
        {
            printf("    PASSED (max_err=%.2e)\n", max_err);
        }
        
        // Verify structure: check that deviations use X[i,1..N-1]
        // and SR noise uses Rsr[i,0..L-1]
        int structure_ok = 1;
        for (size_t col = 0; col < L; col++)
        {
            float *A_col = Aprime + col * M;
            const float *Xi = X + col * N;
            const float xi = x[col];
            const float *Rsri = Rsr + col * L;
            
            // Check deviation rows (0..K-1)
            for (size_t r = 0; r < K; r++)
            {
                float expected = w1s * (Xi[r + 1] - xi);
                if (fabsf(A_col[r] - expected) > 1e-6)
                {
                    structure_ok = 0;
                    break;
                }
            }
            
            // Check SR noise rows (K..M-1)
            for (size_t t = 0; t < L; t++)
            {
                if (fabsf(A_col[K + t] - Rsri[t]) > 1e-6)
                {
                    structure_ok = 0;
                    break;
                }
            }
        }
        
        if (!structure_ok)
        {
            printf("    FAILED: structure verification\n");
            passed = 0;
        }
        
cleanup_aprime:
        gemm_aligned_free(X);
        gemm_aligned_free(x);
        gemm_aligned_free(Rsr);
        gemm_aligned_free(Aprime);
        gemm_aligned_free(Aprime_ref);
    }
    
    return passed;
}

//==============================================================================
// TEST: transpose_square_inplace
//==============================================================================

/**
 * @brief Test in-place square transpose
 */
static int test_transpose_square_inplace(void)
{
    printf("\n=== Testing transpose_square_inplace ===\n");
    
    int passed = 1;
    
    uint16_t test_n[] = {4, 8, 16, 32, 64};
    const int num_n = sizeof(test_n) / sizeof(test_n[0]);
    
    for (int tc = 0; tc < num_n; tc++)
    {
        uint16_t n = test_n[tc];
        
        printf("  Testing n=%d...\n", n);
        
        float *A = gemm_aligned_alloc(32, (size_t)n * n * sizeof(float));
        float *A_orig = gemm_aligned_alloc(32, (size_t)n * n * sizeof(float));
        float *A_ref = gemm_aligned_alloc(32, (size_t)n * n * sizeof(float));
        
        if (!A || !A_orig || !A_ref)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_trans;
        }
        
        // Generate random matrix
        srand(tc * 77777);
        for (size_t i = 0; i < (size_t)n * n; i++)
        {
            A[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
        }
        memcpy(A_orig, A, (size_t)n * n * sizeof(float));
        memcpy(A_ref, A, (size_t)n * n * sizeof(float));
        
        // Compute using implementation under test
        transpose_square_inplace(A, n);
        
        // Compute reference
        transpose_square_reference(A_ref, n);
        
        // Compare
        double diff = matrix_diff_norm(A, A_ref, n, n, n, n);
        
        if (diff > 1e-6)
        {
            printf("    FAILED: diff = %.6e\n", diff);
            passed = 0;
        }
        
        // Verify transpose property: A[i,j] should now be A_orig[j,i]
        double prop_err = 0.0;
        for (uint16_t i = 0; i < n; i++)
        {
            for (uint16_t j = 0; j < n; j++)
            {
                prop_err += fabs((double)A[i * n + j] - (double)A_orig[j * n + i]);
            }
        }
        
        if (prop_err > 1e-6)
        {
            printf("    FAILED: transpose property (err=%.6e)\n", prop_err);
            passed = 0;
        }
        
        // Verify double transpose = identity
        transpose_square_inplace(A, n);
        double identity_err = matrix_diff_norm(A, A_orig, n, n, n, n);
        
        if (identity_err > 1e-6)
        {
            printf("    FAILED: A^T^T != A (err=%.6e)\n", identity_err);
            passed = 0;
        }
        
        if (diff <= 1e-6 && prop_err <= 1e-6 && identity_err <= 1e-6)
        {
            printf("    PASSED\n");
        }
        
cleanup_trans:
        gemm_aligned_free(A);
        gemm_aligned_free(A_orig);
        gemm_aligned_free(A_ref);
    }
    
    return passed;
}

//==============================================================================
// TEST: compute_transition_function
//==============================================================================

/**
 * @brief Simple test transition function: dx = 2*x + u
 */
static void test_transition_F(float dx[], float x[], float u[])
{
    // Assume L is known from context (we'll use L=4 for testing)
    const int L = 4;
    for (int i = 0; i < L; i++)
    {
        dx[i] = 2.0f * x[i] + u[i];
    }
}

/**
 * @brief Test sigma point propagation through transition function
 */
static int test_compute_transition_function(void)
{
    printf("\n=== Testing compute_transition_function ===\n");
    
    int passed = 1;
    
    const uint8_t L = 4;
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    
    printf("  Testing L=%d, N=%zu...\n", L, N);
    
    float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *Xstar = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *u = gemm_aligned_alloc(32, Ls * sizeof(float));
    
    if (!X || !Xstar || !u)
    {
        printf("    ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup_trans_func;
    }
    
    // Generate random input
    srand(88888);
    for (size_t i = 0; i < Ls * N; i++)
    {
        X[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    }
    for (size_t i = 0; i < Ls; i++)
    {
        u[i] = ((float)(rand() % 100)) / 50.0f;
    }
    
    memset(Xstar, 0, Ls * N * sizeof(float));
    
    // Compute using implementation under test
    compute_transition_function(Xstar, X, u, test_transition_F, L);
    
    // Verify: Xstar[:,j] = F(X[:,j], u) = 2*X[:,j] + u
    double max_err = 0.0;
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < Ls; i++)
        {
            float expected = 2.0f * X[i * N + j] + u[i];
            float got = Xstar[i * N + j];
            double err = fabs((double)got - (double)expected);
            if (err > max_err) max_err = err;
        }
    }
    
    if (max_err > 1e-5)
    {
        printf("    FAILED: max_err = %.6e\n", max_err);
        passed = 0;
    }
    else
    {
        printf("    PASSED (max_err=%.2e)\n", max_err);
    }
    
cleanup_trans_func:
    gemm_aligned_free(X);
    gemm_aligned_free(Xstar);
    gemm_aligned_free(u);
    
    return passed;
}

//==============================================================================
// TEST: Identity observation model H
//==============================================================================

/**
 * @brief Test identity observation model
 */
static int test_identity_H(void)
{
    printf("\n=== Testing identity observation model H ===\n");
    
    int passed = 1;
    
    const uint8_t L = 8;
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;
    
    printf("  Testing L=%d...\n", L);
    
    float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    float *Y = gemm_aligned_alloc(32, Ls * N * sizeof(float));
    
    if (!X || !Y)
    {
        printf("    ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup_H;
    }
    
    // Generate random input
    srand(99999);
    for (size_t i = 0; i < Ls * N; i++)
    {
        X[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    }
    
    memset(Y, 0, Ls * N * sizeof(float));
    
    // Apply H (identity)
    H(Y, X, L);
    
    // Verify Y == X
    double diff = matrix_diff_norm(Y, X, Ls, N, N, N);
    
    if (diff > 1e-10)
    {
        printf("    FAILED: Y != X (diff=%.6e)\n", diff);
        passed = 0;
    }
    else
    {
        printf("    PASSED\n");
    }
    
cleanup_H:
    gemm_aligned_free(X);
    gemm_aligned_free(Y);
    
    return passed;
}

//==============================================================================
// TEST: Round-trip sigma points → mean
//==============================================================================

/**
 * @brief Test that weighted mean of sigma points recovers original mean
 */
static int test_sigma_point_mean_roundtrip(void)
{
    printf("\n=== Testing sigma point mean round-trip ===\n");
    
    int passed = 1;
    
    uint8_t test_L[] = {4, 8, 16, 32};
    const int num_L = sizeof(test_L) / sizeof(test_L[0]);
    
    for (int tc = 0; tc < num_L; tc++)
    {
        uint8_t L = test_L[tc];
        const size_t Ls = (size_t)L;
        const size_t N = 2u * Ls + 1u;
        
        printf("  Testing L=%d...\n", L);
        
        float *x = gemm_aligned_alloc(32, Ls * sizeof(float));
        float *S = gemm_aligned_alloc(32, Ls * Ls * sizeof(float));
        float *X = gemm_aligned_alloc(32, Ls * N * sizeof(float));
        float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
        float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
        float *x_recovered = gemm_aligned_alloc(32, Ls * sizeof(float));
        
        if (!x || !S || !X || !Wm || !Wc || !x_recovered)
        {
            printf("    ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_roundtrip;
        }
        
        // Generate random mean and SR covariance
        generate_random_vector(x, Ls, tc * 12121);
        generate_random_upper_triangular(S, Ls, Ls, tc * 21212, 1.0f);
        
        const float alpha = 1e-3f;
        const float beta = 2.0f;
        const float kappa = 0.0f;
        
        // Create weights
        create_weights(Wc, Wm, alpha, beta, kappa, L);
        
        // Create sigma points
        create_sigma_point_matrix(X, x, S, alpha, kappa, L);
        
        // Recover mean via weighted sum
        multiply_sigma_point_matrix_to_weights(x_recovered, X, Wm, L);
        
        // Compare recovered mean to original
        double max_err = 0.0;
        for (size_t i = 0; i < Ls; i++)
        {
            double err = fabs((double)x_recovered[i] - (double)x[i]);
            if (err > max_err) max_err = err;
        }
        
        if (max_err > 1e-4)
        {
            printf("    FAILED: mean recovery error = %.6e\n", max_err);
            passed = 0;
        }
        else
        {
            printf("    PASSED (max_err=%.2e)\n", max_err);
        }
        
cleanup_roundtrip:
        gemm_aligned_free(x);
        gemm_aligned_free(S);
        gemm_aligned_free(X);
        gemm_aligned_free(Wm);
        gemm_aligned_free(Wc);
        gemm_aligned_free(x_recovered);
    }
    
    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_ukf_primitives_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("    SR-UKF PRIMITIVES TESTS\n");
    printf("=================================================\n");
    
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    // Weight computation
    printf("\n--- Weight Computation Tests ---\n");
    
    results->total++;
    if (test_create_weights())
    {
        results->passed++;
        printf("✓ create_weights test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ create_weights test FAILED\n");
    }
    
    // Sigma point generation
    printf("\n--- Sigma Point Tests ---\n");
    
    results->total++;
    if (test_create_sigma_point_matrix())
    {
        results->passed++;
        printf("✓ create_sigma_point_matrix test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ create_sigma_point_matrix test FAILED\n");
    }
    
    // Weighted mean
    printf("\n--- Weighted Mean Tests ---\n");
    
    results->total++;
    if (test_weighted_mean())
    {
        results->passed++;
        printf("✓ weighted_mean test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ weighted_mean test FAILED\n");
    }
    
    // Transition function
    printf("\n--- Transition Function Tests ---\n");
    
    results->total++;
    if (test_compute_transition_function())
    {
        results->passed++;
        printf("✓ compute_transition_function test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ compute_transition_function test FAILED\n");
    }
    
    // Helper functions
    printf("\n--- Helper Function Tests ---\n");
    
    results->total++;
    if (test_build_YTc_fused())
    {
        results->passed++;
        printf("✓ build_YTc_fused test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ build_YTc_fused test FAILED\n");
    }
    
    results->total++;
    if (test_build_Aprime_column_major())
    {
        results->passed++;
        printf("✓ build_Aprime_column_major test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ build_Aprime_column_major test FAILED\n");
    }
    
    results->total++;
    if (test_transpose_square_inplace())
    {
        results->passed++;
        printf("✓ transpose_square_inplace test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ transpose_square_inplace test FAILED\n");
    }
    
    results->total++;
    if (test_identity_H())
    {
        results->passed++;
        printf("✓ identity H test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ identity H test FAILED\n");
    }
    
    // Integration tests
    printf("\n--- Round-trip Tests ---\n");
    
    results->total++;
    if (test_sigma_point_mean_roundtrip())
    {
        results->passed++;
        printf("✓ sigma point mean round-trip test PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ sigma point mean round-trip test FAILED\n");
    }
    
    // Summary
    printf("\n=================================================\n");
    printf("SR-UKF Primitives Tests: %d/%d passed\n", results->passed, results->total);
    
    if (results->passed == results->total)
    {
        printf("✓ ALL SR-UKF PRIMITIVES TESTS PASSED!\n");
    }
    else
    {
        printf("✗ %d SR-UKF primitives tests FAILED\n", results->failed);
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
    return run_ukf_primitives_tests(&results);
}
#endif