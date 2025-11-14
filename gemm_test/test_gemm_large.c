/**
 * @file test_gemm_large.c (contains test_kernels_individual)
 * @brief Unit tests for individual GEMM kernels
 */

#include "test_common.h"
#include "gemm_kernels_avx2.h"
#include "gemm_planning.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//==============================================================================
// TEST INFRASTRUCTURE
//==============================================================================

static void print_matrix(const char *name, const float *m, size_t rows, size_t cols, size_t ldc)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            printf("%8.3f ", m[i * ldc + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static int compare_matrices_verbose(
    const float *test,
    const float *ref,
    size_t rows,
    size_t cols,
    size_t ldc,
    float tol,
    const char *test_name)
{
    int errors = 0;
    float max_error = 0.0f;
    size_t error_i = 0, error_j = 0;

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            float t = test[i * ldc + j];
            float r = ref[i * ldc + j];
            float err = fabsf(t - r);

            if (err > max_error)
            {
                max_error = err;
                error_i = i;
                error_j = j;
            }

            if (err > tol)
            {
                errors++;
                if (errors <= 5)
                { // Print first 5 errors
#ifdef _WIN32
                    printf("  ERROR at [%llu,%llu]: test=%.6f, ref=%.6f, diff=%.6f\n",
                           (unsigned long long)i, (unsigned long long)j, t, r, err);
#else
                    printf("  ERROR at [%zu,%zu]: test=%.6f, ref=%.6f, diff=%.6f\n",
                           i, j, t, r, err);
#endif
                }
            }
        }
    }

    if (errors > 0)
    {
#ifdef _WIN32
        printf("  %s FAILED: %d errors, max error %.6f at [%llu,%llu]\n",
               test_name, errors, max_error,
               (unsigned long long)error_i, (unsigned long long)error_j);
#else
        printf("  %s FAILED: %d errors, max error %.6f at [%zu,%zu]\n",
               test_name, errors, max_error, error_i, error_j);
#endif
        return 0;
    }

    printf("  %s PASSED (max error: %.6e)\n", test_name, max_error);
    return 1;
}

//==============================================================================
// SIMPLE REFERENCE KERNELS
//==============================================================================

static void ref_gemm_simple(
    float *C, size_t ldc,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    size_t M, size_t K, size_t N,
    int accumulate)
{
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
            {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            if (accumulate)
            {
                C[i * ldc + j] += sum;
            }
            else
            {
                C[i * ldc + j] = sum;
            }
        }
    }
}

//==============================================================================
// PACKING TEST HELPERS
//==============================================================================

static void pack_A_for_test(float *Ap, const float *A, size_t M, size_t K, size_t mr)
{
    memset(Ap, 0, K * mr * sizeof(float));
    for (size_t k = 0; k < K; k++)
    {
        for (size_t i = 0; i < M && i < mr; i++)
        {
            Ap[k * mr + i] = A[i * K + k];
        }
    }
}

static void pack_B_for_test(float *Bp, const float *B, size_t K, size_t N)
{
    memset(Bp, 0, K * 16 * sizeof(float));
    for (size_t k = 0; k < K; k++)
    {
        for (size_t j = 0; j < N && j < 16; j++)
        {
            Bp[k * 16 + j] = B[k * N + j];
        }
    }
}

//==============================================================================
// TEST 8x8 KERNEL
//==============================================================================

static int test_kernel_8x8(void)
{
    printf("\n=== Testing 8x8 Kernel ===\n");

    const size_t M = 8, K = 32, N = 8;
    const size_t ldc = N;

    // Use library's aligned allocation
    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    // Initialize
    for (size_t i = 0; i < M * K; i++)
        A[i] = (i % 7) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i % 5) * 0.1f;

    pack_A_for_test(Ap, A, M, K, 8);
    pack_B_for_test(Bp, B, K, N);

    // Test STORE variant
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

    // Correct argument order
    gemm_8x8_panel_avx2fma_store(
        C_test, ldc, // C and ldc
        Ap, 8,       // packed A and stride
        Bp, 16,      // packed B and stride
        K,           // K dimension
        8, 8,        // m, n
        mask         // mask
    );

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x8 STORE");

    // Test ADD variant
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 1.0f;
        C_ref[i] = 1.0f;
    }

    gemm_8x8_panel_avx2fma_add(
        C_test, ldc,
        Ap, 8,
        Bp, 16,
        K,
        8, 8,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);

    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x8 ADD");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST 4x8 KERNEL
//==============================================================================

static int test_kernel_4x8(void)
{
    printf("\n=== Testing 4x8 Kernel ===\n");

    const size_t M = 4, K = 16, N = 8;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    for (size_t i = 0; i < M * K; i++)
        A[i] = (i + 1) * 0.01f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i + 1) * 0.01f;

    pack_A_for_test(Ap, A, M, K, 8);
    pack_B_for_test(Bp, B, K, N);

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

    // Correct argument order for 4x8
    gemm_4x8_panel_avx2fma_store(
        C_test, ldc,
        Ap, 8,
        Bp, 16,
        K,
        8, // jb (width)
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "4x8 STORE");

    // Test ADD
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 0.5f;
        C_ref[i] = 0.5f;
    }

    gemm_4x8_panel_avx2fma_add(
        C_test, ldc,
        Ap, 8,
        Bp, 16,
        K,
        8,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);

    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "4x8 ADD");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST 1x8 KERNEL
//==============================================================================

static int test_kernel_1x8(void)
{
    printf("\n=== Testing 1x8 Kernel ===\n");

    const size_t M = 1, K = 8, N = 8;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    for (size_t i = 0; i < M * K; i++)
        A[i] = (i + 1) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i + 1) * 0.1f;

    pack_A_for_test(Ap, A, M, K, 8);
    pack_B_for_test(Bp, B, K, N);

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

    // Correct argument order for 1x8
    gemm_1x8_panel_avx2fma_store(
        C_test, // No ldc for 1x8
        Ap, 8,
        Bp, 16,
        K,
        8, // jb
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    printf("  1x8 result: ");
    for (int i = 0; i < 8; i++)
        printf("%.3f ", C_test[i]);
    printf("\n");
    printf("  Reference:  ");
    for (int i = 0; i < 8; i++)
        printf("%.3f ", C_ref[i]);
    printf("\n");

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "1x8 STORE");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST KERNEL COMBINATIONS
//==============================================================================

static int test_kernel_combination(void)
{
    printf("\n=== Testing Kernel Combinations ===\n");

    const size_t M = 12, K = 16, N = 8;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    srand(42);
    for (size_t i = 0; i < M * K; i++)
        A[i] = (float)(rand() % 10) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (float)(rand() % 10) * 0.1f;

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    float *Ap8 = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Ap4 = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    pack_A_for_test(Ap8, A, 8, K, 8);
    pack_A_for_test(Ap4, A + 8 * K, 4, K, 8);
    pack_B_for_test(Bp, B, K, N);

    __m256i mask = _mm256_set1_epi32(-1);

    gemm_8x8_panel_avx2fma_store(
        C_test, ldc,
        Ap8, 8,
        Bp, 16,
        K, 8, 8, mask);

    gemm_4x8_panel_avx2fma_store(
        C_test + 8 * ldc, ldc,
        Ap4, 8,
        Bp, 16,
        K, 8, mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-4f, "12x8 combination");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap8);
    gemm_aligned_free(Ap4);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_gemm_kernel_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("         GEMM KERNEL UNIT TESTS\n");
    printf("=================================================\n");

    results->total = 0;
    results->passed = 0;
    results->failed = 0;

    // Test individual kernels
    printf("\n--- Testing Individual Kernels ---\n");

    results->total++;
    if (test_kernel_8x8())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    results->total++;
    if (test_kernel_4x8())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    results->total++;
    if (test_kernel_1x8())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    // Test combinations
    printf("\n--- Testing Kernel Combinations ---\n");

    results->total++;
    if (test_kernel_combination())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    printf("\n=================================================\n");
    printf("Kernel Tests: %d/%d passed\n", results->passed, results->total);

    if (results->passed == results->total)
    {
        printf("✓ All kernel tests PASSED!\n");
    }
    else
    {
        printf("✗ %d kernel tests FAILED\n", results->failed);
        printf("Debug the failing kernels before testing the full GEMM\n");
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
    return run_gemm_kernel_tests(&results);
}
#endif