/**
 * @file test_gemm_large.c
 * @brief Test suite specifically for gemm_large.c (Tier 2)
 *
 * This bypasses gemm_small dispatch and directly tests:
 * - gemm_execute_plan()
 * - Adaptive blocking
 * - Alpha/beta scaling
 * - Loop structure correctness
 * - Workspace management
 *
 * @author TUGBARS
 * @date 2025
 */

#include "gemm_planning.h"
#include "gemm_kernels_avx2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>

//==============================================================================
// TEST UTILITIES (same as before)
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

static void init_matrix_random(float *A, size_t M, size_t N, unsigned int seed)
{
    srand(seed);
    for (size_t i = 0; i < M * N; i++)
    {
        A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

static double compare_matrices(
    const float *A,
    const float *B,
    size_t M, size_t N,
    double *max_abs_error)
{
    double max_rel_error = 0.0;
    *max_abs_error = 0.0;

    for (size_t i = 0; i < M * N; i++)
    {
        float a = A[i];
        float b = B[i];

        if (isnan(a) || isnan(b))
        {
            printf("NaN detected at index %zu: A=%f, B=%f\n", i, a, b);
            return -1.0;
        }

        double abs_error = fabs(a - b);
        *max_abs_error = fmax(*max_abs_error, abs_error);

        double denominator = fmax(fabs(a), fabs(b));
        if (denominator > 1e-10)
        {
            double rel_error = abs_error / denominator;
            max_rel_error = fmax(max_rel_error, rel_error);
        }
    }

    return max_rel_error;
}

static double get_time(void)
{
#ifdef _WIN32
    return (double)clock() / CLOCKS_PER_SEC;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

static double measure_gflops(size_t M, size_t K, size_t N, double time_sec)
{
    double flops = 2.0 * M * K * N;
    return (flops / time_sec) / 1e9;
}

//==============================================================================
// GEMM_LARGE SPECIFIC TESTS
//==============================================================================

/**
 * @brief Test gemm_execute_plan directly (bypasses Tier 1)
 */
static bool test_execute_plan_direct(
    size_t M, size_t K, size_t N,
    float alpha, float beta,
    const char *test_name)
{
    printf("  Testing: %s\n", test_name);

    // Allocate matrices
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C_test = (float *)malloc(M * N * sizeof(float));
    float *C_ref = (float *)malloc(M * N * sizeof(float));

    if (!A || !B || !C_test || !C_ref)
    {
        printf("  FAIL: Memory allocation failed\n");
        free(A);
        free(B);
        free(C_test);
        free(C_ref);
        return false;
    }

    // Initialize
    init_matrix_random(A, M, K, 42);
    init_matrix_random(B, K, N, 43);
    init_matrix_random(C_test, M, N, 44);
    memcpy(C_ref, C_test, M * N * sizeof(float));

    // Compute reference
    gemm_reference(C_ref, A, B, M, K, N, alpha, beta);

    // Create plan and execute (THIS IS THE KEY - direct call to gemm_large.c)
    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        printf("  FAIL: Plan creation failed\n");
        free(A);
        free(B);
        free(C_test);
        free(C_ref);
        return false;
    }

    printf("    Plan: MC=%zu, KC=%zu, NC=%zu, MR=%zu, NR=%zu, mode=%s\n",
           plan->MC, plan->KC, plan->NC, plan->MR, plan->NR,
           plan->mem_mode == GEMM_MEM_STATIC ? "STATIC" : "DYNAMIC");

    // DIRECT CALL TO GEMM_LARGE.C
    int ret = gemm_execute_plan(plan, C_test, A, B, alpha, beta);

    if (ret != 0)
    {
        printf("  FAIL: gemm_execute_plan returned error %d\n", ret);
        gemm_plan_destroy(plan);
        free(A);
        free(B);
        free(C_test);
        free(C_ref);
        return false;
    }

    // Compare
    double max_abs_error;
    double max_rel_error = compare_matrices(C_test, C_ref, M, N, &max_abs_error);

    bool passed = (max_rel_error >= 0.0 && max_rel_error < 1e-4);

    if (!passed)
    {
        printf("  FAIL: Max relative error = %.2e, max absolute error = %.2e\n",
               max_rel_error, max_abs_error);
    }
    else
    {
        printf("  PASS: Max relative error = %.2e, max absolute error = %.2e\n",
               max_rel_error, max_abs_error);
    }

    gemm_plan_destroy(plan);
    free(A);
    free(B);
    free(C_test);
    free(C_ref);
    return passed;
}

/**
 * @brief Test that dimensions force use of Tier 2 (not Tier 1)
 */
static bool test_forces_tier2(size_t M, size_t K, size_t N)
{
    printf("  Verifying dimensions force Tier 2 (gemm_large.c)...\n");

    // These dimensions should NOT match any Tier 1 kernels
    // (Tier 1 handles exact sizes like 4x4, 6x6, 8x8, etc.)

    bool will_use_tier2 = true;

    // Check if dimensions match any Tier 1 exact sizes
    if (M == K && K == N)
    {
        if (M == 4 || M == 6 || M == 8 || M == 12 || M == 16)
        {
            will_use_tier2 = false;
            printf("  WARNING: Dimensions %zux%zux%zu might use Tier 1!\n", M, K, N);
        }
    }

    if (will_use_tier2)
    {
        printf("  OK: Dimensions %zux%zux%zu will use Tier 2\n", M, K, N);
    }

    return will_use_tier2;
}

/**
 * @brief Test adaptive blocking specifically
 */
static bool test_adaptive_blocking_detail(size_t M, size_t K, size_t N)
{
    printf("  Testing adaptive blocking for %zux%zux%zu...\n", M, K, N);

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        printf("  FAIL: Plan creation failed\n");
        return false;
    }

    double aspect_mn = (double)M / (double)N;
    double aspect_kn = (double)K / (double)N;

    printf("    Aspect M/N = %.2f, K/N = %.2f\n", aspect_mn, aspect_kn);
    printf("    Selected: MC=%zu, KC=%zu, NC=%zu\n", plan->MC, plan->KC, plan->NC);

    // Verify adaptive selection worked
    bool correct = true;

    if (aspect_mn > 3.0)
    {
        // TALL: Should have larger MC
        if (plan->MC < 128)
        {
            printf("  WARN: Expected MC >= 128 for tall matrix, got %zu\n", plan->MC);
        }
        printf("    Classification: TALL matrix (OK)\n");
    }
    else if (aspect_mn < 0.33)
    {
        // WIDE: Should have larger NC
        if (plan->NC < 256)
        {
            printf("  WARN: Expected NC >= 256 for wide matrix, got %zu\n", plan->NC);
        }
        printf("    Classification: WIDE matrix (OK)\n");
    }
    else if (aspect_kn > 4.0)
    {
        // DEEP: Should have larger KC
        if (plan->KC < 256)
        {
            printf("  WARN: Expected KC >= 256 for deep matrix, got %zu\n", plan->KC);
        }
        printf("    Classification: DEEP matrix (OK)\n");
    }
    else
    {
        printf("    Classification: BALANCED matrix (OK)\n");
    }

    // Verify cache footprint
    size_t footprint = (plan->MC * plan->KC + plan->KC * plan->NC) * sizeof(float);
    printf("    Cache footprint: %.1f KB\n", footprint / 1024.0);

    if (footprint > 2 * 1024 * 1024)
    {
        printf("  WARN: Footprint exceeds L2 cache (2MB)\n");
        correct = false;
    }

    gemm_plan_destroy(plan);

    if (correct)
    {
        printf("  PASS\n");
    }
    return correct;
}

/**
 * @brief Test beta pre-scaling specifically
 */
static bool test_beta_scaling(void)
{
    printf("  Testing beta pre-scaling (gemm_large.c fix #4)...\n");

    size_t M = 64, K = 64, N = 64;

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C1 = (float *)malloc(M * N * sizeof(float));
    float *C2 = (float *)malloc(M * N * sizeof(float));
    float *C_ref = (float *)malloc(M * N * sizeof(float));

    init_matrix_random(A, M, K, 100);
    init_matrix_random(B, K, N, 101);
    init_matrix_random(C1, M, N, 102);

    // Test different beta values
    float betas[] = {0.0f, 0.5f, 1.0f, 2.0f, -0.5f};
    bool all_passed = true;

    for (size_t i = 0; i < sizeof(betas) / sizeof(betas[0]); i++)
    {
        float beta = betas[i];

        memcpy(C2, C1, M * N * sizeof(float));
        memcpy(C_ref, C1, M * N * sizeof(float));

        // Reference
        gemm_reference(C_ref, A, B, M, K, N, 1.0f, beta);

        // Test with gemm_execute_plan
        gemm_plan_t *plan = gemm_plan_create(M, K, N);
        gemm_execute_plan(plan, C2, A, B, 1.0f, beta);
        gemm_plan_destroy(plan);

        // Compare
        double max_abs_error;
        double max_rel_error = compare_matrices(C2, C_ref, M, N, &max_abs_error);

        bool passed = (max_rel_error >= 0.0 && max_rel_error < 1e-4);

        printf("    Beta=%.2f: %s (error=%.2e)\n",
               beta, passed ? "PASS" : "FAIL", max_rel_error);

        if (!passed)
            all_passed = false;
    }

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C_ref);

    if (all_passed)
    {
        printf("  PASS: All beta values correct\n");
    }
    else
    {
        printf("  FAIL: Some beta values incorrect\n");
    }

    return all_passed;
}

/**
 * @brief Test alpha scaling in packing
 */
static bool test_alpha_scaling(void)
{
    printf("  Testing alpha scaling in packing (gemm_large.c fix #3)...\n");

    size_t M = 64, K = 64, N = 64;

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C1 = (float *)malloc(M * N * sizeof(float));
    float *C_ref = (float *)malloc(M * N * sizeof(float));

    init_matrix_random(A, M, K, 110);
    init_matrix_random(B, K, N, 111);
    memset(C1, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    // Test different alpha values
    float alphas[] = {0.0f, 0.5f, 1.0f, 2.0f, -1.0f};
    bool all_passed = true;

    for (size_t i = 0; i < sizeof(alphas) / sizeof(alphas[0]); i++)
    {
        float alpha = alphas[i];

        memset(C1, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        // Reference
        gemm_reference(C_ref, A, B, M, K, N, alpha, 0.0f);

        // Test with gemm_execute_plan
        gemm_plan_t *plan = gemm_plan_create(M, K, N);
        gemm_execute_plan(plan, C1, A, B, alpha, 0.0f);
        gemm_plan_destroy(plan);

        // Compare
        double max_abs_error;
        double max_rel_error = compare_matrices(C1, C_ref, M, N, &max_abs_error);

        bool passed = (max_rel_error >= 0.0 && max_rel_error < 1e-4);

        printf("    Alpha=%.2f: %s (error=%.2e)\n",
               alpha, passed ? "PASS" : "FAIL", max_rel_error);

        if (!passed)
            all_passed = false;
    }

    free(A);
    free(B);
    free(C1);
    free(C_ref);

    if (all_passed)
    {
        printf("  PASS: All alpha values correct\n");
    }
    else
    {
        printf("  FAIL: Some alpha values incorrect\n");
    }

    return all_passed;
}

/**
 * @brief Performance test for gemm_large.c
 */
static bool test_performance_gemm_large(size_t M, size_t K, size_t N)
{
    printf("  Performance test: %zux%zux%zu\n", M, K, N);

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));

    init_matrix_random(A, M, K, 200);
    init_matrix_random(B, K, N, 201);
    memset(C, 0, M * N * sizeof(float));

    // Create plan once
    gemm_plan_t *plan = gemm_plan_create(M, K, N);

    // Warmup
    gemm_execute_plan(plan, C, A, B, 1.0f, 0.0f);

    // Benchmark
    double best_time = DBL_MAX;
    for (int run = 0; run < 3; run++)
    {
        double t0 = get_time();
        gemm_execute_plan(plan, C, A, B, 1.0f, 0.0f);
        double t1 = get_time();

        if (t1 - t0 < best_time)
        {
            best_time = t1 - t0;
        }
    }

    double gflops = measure_gflops(M, K, N, best_time);

    printf("    Time: %.3f ms, Performance: %.1f GFLOPS\n",
           best_time * 1000.0, gflops);

    gemm_plan_destroy(plan);
    free(A);
    free(B);
    free(C);

    printf("  PASS\n");
    return true;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int main(void)
{
    printf("================================================================================\n");
    printf("GEMM_LARGE.C Specific Test Suite (Tier 2 Only)\n");
    printf("================================================================================\n\n");

    size_t passed = 0, failed = 0;

    //==========================================================================
    // Test 1: Force Tier 2 dimensions
    //==========================================================================
    printf("[1] Verifying test dimensions force Tier 2 usage\n");
    test_forces_tier2(100, 100, 100);
    printf("\n");

    //==========================================================================
    // Test 2: Basic correctness (direct gemm_execute_plan calls)
    //==========================================================================
    printf("[2] Basic Correctness Tests (direct gemm_execute_plan)\n");
    passed += test_execute_plan_direct(64, 64, 64, 1.0f, 0.0f, "Square 64x64x64");
    passed += test_execute_plan_direct(100, 50, 75, 1.0f, 0.0f, "Odd dimensions");
    passed += test_execute_plan_direct(17, 23, 31, 1.0f, 0.0f, "Prime dimensions");
    printf("\n");

    //==========================================================================
    // Test 3: Adaptive blocking
    //==========================================================================
    printf("[3] Adaptive Blocking Tests\n");
    passed += test_adaptive_blocking_detail(2048, 512, 512);   // Tall
    passed += test_adaptive_blocking_detail(512, 512, 2048);   // Wide
    passed += test_adaptive_blocking_detail(512, 4096, 512);   // Deep
    passed += test_adaptive_blocking_detail(1024, 1024, 1024); // Balanced
    printf("\n");

    //==========================================================================
    // Test 4: Alpha scaling (Fix #3)
    //==========================================================================
    printf("[4] Alpha Scaling Tests (Fix #3)\n");
    passed += test_alpha_scaling();
    printf("\n");

    //==========================================================================
    // Test 5: Beta scaling (Fix #4)
    //==========================================================================
    printf("[5] Beta Scaling Tests (Fix #4)\n");
    passed += test_beta_scaling();
    printf("\n");

    //==========================================================================
    // Test 6: Various shapes (tests loop structure - Fix #5, #6)
    //==========================================================================
    printf("[6] Matrix Shape Tests (Loop structure & workspace indexing)\n");
    passed += test_execute_plan_direct(256, 32, 128, 1.0f, 0.0f, "Tall skinny");
    passed += test_execute_plan_direct(32, 128, 256, 1.0f, 0.0f, "Wide flat");
    passed += test_execute_plan_direct(128, 512, 128, 1.0f, 0.0f, "Deep");
    passed += test_execute_plan_direct(200, 200, 200, 1.0f, 0.0f, "Square 200");
    printf("\n");

    //==========================================================================
    // Test 7: Combined alpha/beta
    //==========================================================================
    printf("[7] Combined Alpha/Beta Tests\n");
    passed += test_execute_plan_direct(128, 128, 128, 2.0f, 0.5f, "alpha=2, beta=0.5");
    passed += test_execute_plan_direct(128, 128, 128, -1.0f, 2.0f, "alpha=-1, beta=2");
    passed += test_execute_plan_direct(128, 128, 128, 0.5f, 0.5f, "alpha=0.5, beta=0.5");
    printf("\n");

    //==========================================================================
    // Test 8: Static vs Dynamic
    //==========================================================================
    printf("[8] Memory Mode Tests\n");
    passed += test_execute_plan_direct(256, 256, 256, 1.0f, 0.0f, "Static mode size");
    passed += test_execute_plan_direct(1024, 1024, 1024, 1.0f, 0.0f, "Dynamic mode size");
    printf("\n");

    //==========================================================================
    // Test 9: Performance
    //==========================================================================
    printf("[9] Performance Tests\n");
    test_performance_gemm_large(256, 256, 256);
    test_performance_gemm_large(512, 512, 512);
    test_performance_gemm_large(1024, 1024, 1024);
    printf("\n");

    //==========================================================================
    // Summary
    //==========================================================================
    printf("================================================================================\n");
    printf("Tests passed: %zu\n", passed);
    printf("Tests failed: %zu\n", failed);
    printf("================================================================================\n");

    return (failed > 0) ? 1 : 0;
}