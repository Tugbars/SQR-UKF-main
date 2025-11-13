/**
 * @file test_gemm_large.c
 * @brief Test suite for gemm_large.c (Tier 2)
 *
 * @author TUGBARS
 * @date 2025
 */

#include "test_common.h"
#include "gemm_planning.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Declare gemm_execute_plan
extern int gemm_execute_plan(
    gemm_plan_t *plan,
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    float alpha,
    float beta);

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

static void gemm_reference(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

static void init_random(float *A, size_t M, size_t N, unsigned int seed)
{
    srand(seed);
    for (size_t i = 0; i < M * N; i++)
    {
        A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

static int compare_matrices(
    const float *A,
    const float *B,
    size_t M, size_t N,
    double tolerance,
    double *max_error_out)
{
    double max_rel_error = 0.0;

    for (size_t i = 0; i < M * N; i++)
    {
        if (isnan(A[i]) || isnan(B[i]))
        {
            *max_error_out = -1.0;
            return 0;
        }

        double abs_error = fabs(A[i] - B[i]);
        double denominator = fmax(fabs(A[i]), fabs(B[i]));

        if (denominator > 1e-10)
        {
            double rel_error = abs_error / denominator;
            max_rel_error = fmax(max_rel_error, rel_error);
        }
    }

    *max_error_out = max_rel_error;
    return max_rel_error < tolerance;
}

//==============================================================================
// TEST CASES
//==============================================================================

static int test_basic_64x64x64(void)
{
    printf("  Testing: 64x64x64 square matrix\n");

    size_t M = 64, K = 64, N = 64;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        printf("    FAIL: Memory allocation\n");
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 42);
    init_random(B, K, N, 43);
    init_random(C_test, M, N, 44);
    memcpy(C_ref, C_test, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        printf("    FAIL: Plan creation\n");
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 1.0f, 0.0f);

    if (ret != 0)
    {
        printf("    FAIL: gemm_execute_plan returned error\n");
        gemm_plan_destroy(plan);
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    double max_error;
    int match = compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    if (!match)
    {
        printf("    FAIL: Max error = %.2e\n", max_error);
    }

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_odd_dimensions(void)
{
    printf("  Testing: 100x50x75 odd dimensions\n");

    size_t M = 100, K = 50, N = 75;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 50);
    init_random(B, K, N, 51);
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 1.0f, 0.0f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_prime_dimensions(void)
{
    printf("  Testing: 17x23x31 prime dimensions\n");

    size_t M = 17, K = 23, N = 31;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 60);
    init_random(B, K, N, 61);
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 1.0f, 0.0f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_alpha_scaling(void)
{
    printf("  Testing: Alpha scaling (alpha=2.0)\n");

    size_t M = 64, K = 64, N = 64;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 70);
    init_random(B, K, N, 71);
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 2.0f, 0.0f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 2.0f, 0.0f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_beta_scaling(void)
{
    printf("  Testing: Beta scaling (beta=0.5)\n");

    size_t M = 64, K = 64, N = 64;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 80);
    init_random(B, K, N, 81);
    init_random(C_test, M, N, 82);
    memcpy(C_ref, C_test, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.5f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 1.0f, 0.5f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_combined_alpha_beta(void)
{
    printf("  Testing: Combined alpha=2, beta=0.5\n");

    size_t M = 64, K = 64, N = 64;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 90);
    init_random(B, K, N, 91);
    init_random(C_test, M, N, 92);
    memcpy(C_ref, C_test, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 2.0f, 0.5f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 2.0f, 0.5f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_tall_matrix(void)
{
    printf("  Testing: Tall matrix 256x32x128\n");

    size_t M = 256, K = 32, N = 128;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 100);
    init_random(B, K, N, 101);
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 1.0f, 0.0f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_wide_matrix(void)
{
    printf("  Testing: Wide matrix 32x128x256\n");

    size_t M = 32, K = 128, N = 256;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 110);
    init_random(B, K, N, 111);
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 1.0f, 0.0f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_deep_matrix(void)
{
    printf("  Testing: Deep matrix 128x512x128\n");

    size_t M = 128, K = 512, N = 128;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 120);
    init_random(B, K, N, 121);
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 1.0f, 0.0f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

static int test_large_1024(void)
{
    printf("  Testing: Large 1024x1024x1024\n");

    size_t M = 1024, K = 1024, N = 1024;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_random(A, M, K, 130);
    init_random(B, K, N, 131);
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        return 0;
    }

    int ret = gemm_execute_plan(plan, C_test, A, B, 1.0f, 0.0f);

    double max_error;
    int match = (ret == 0) && compare_matrices(C_test, C_ref, M, N, 1e-4, &max_error);

    gemm_plan_destroy(plan);
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return match;
}

//==============================================================================
// TEST SUITE RUNNER
//==============================================================================

int run_gemm_large_tests(test_results_t *results)
{
    results->total = 0;
    results->passed = 0;
    results->failed = 0;

    printf("\n═══ Test Group 1: Basic Correctness ═══\n");
    RUN_TEST(results, test_basic_64x64x64);
    RUN_TEST(results, test_odd_dimensions);
    RUN_TEST(results, test_prime_dimensions);

    printf("\n═══ Test Group 2: Alpha/Beta Scaling ═══\n");
    RUN_TEST(results, test_alpha_scaling);
    RUN_TEST(results, test_beta_scaling);
    RUN_TEST(results, test_combined_alpha_beta);

    printf("\n═══ Test Group 3: Matrix Shapes ═══\n");
    RUN_TEST(results, test_tall_matrix);
    RUN_TEST(results, test_wide_matrix);
    RUN_TEST(results, test_deep_matrix);

    printf("\n═══ Test Group 4: Large Matrices ═══\n");
    RUN_TEST(results, test_large_1024);

    print_test_results("GEMM Large (Tier 2) - Results", results);

    return (results->failed == 0) ? 0 : 1;
}

#ifdef STANDALONE
int main(void)
{
    test_results_t results;
    int ret = run_gemm_large_tests(&results);

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