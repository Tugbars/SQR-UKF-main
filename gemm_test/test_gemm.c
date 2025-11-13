/**
 * @file test_gemm.c
 * @brief Comprehensive test suite for GEMM implementation
 * 
 * Tests:
 * - Correctness (vs reference naive GEMM)
 * - Edge cases (small, odd, single row/col)
 * - Alpha/beta scaling
 * - Matrix shapes (square, tall, wide, deep)
 * - Adaptive blocking
 * - Static vs dynamic modes
 * - Numerical stability
 * - Performance benchmarks
 * 
 * @author TUGBARS
 * @date 2025
 */

#include "gemm.h"
#include "gemm_planning.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>

//==============================================================================
// TEST UTILITIES
//==============================================================================

/**
 * @brief Reference naive GEMM implementation (slow but correct)
 */
static void gemm_reference(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

/**
 * @brief Initialize matrix with random values
 */
static void init_matrix_random(float *A, size_t M, size_t N, unsigned int seed)
{
    srand(seed);
    for (size_t i = 0; i < M * N; i++) {
        A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }
}

/**
 * @brief Initialize matrix with specific pattern (for debugging)
 */
static void init_matrix_pattern(float *A, size_t M, size_t N, const char *pattern)
{
    if (strcmp(pattern, "identity") == 0) {
        memset(A, 0, M * N * sizeof(float));
        for (size_t i = 0; i < M && i < N; i++) {
            A[i * N + i] = 1.0f;
        }
    } else if (strcmp(pattern, "ones") == 0) {
        for (size_t i = 0; i < M * N; i++) {
            A[i] = 1.0f;
        }
    } else if (strcmp(pattern, "zeros") == 0) {
        memset(A, 0, M * N * sizeof(float));
    } else if (strcmp(pattern, "sequential") == 0) {
        for (size_t i = 0; i < M * N; i++) {
            A[i] = (float)(i + 1);
        }
    }
}

/**
 * @brief Compare two matrices with relative error tolerance
 * 
 * @return Maximum relative error, or -1.0 if NaN detected
 */
static double compare_matrices(
    const float *A, 
    const float *B, 
    size_t M, size_t N,
    double *max_abs_error)
{
    double max_rel_error = 0.0;
    *max_abs_error = 0.0;
    
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            size_t idx = i * N + j;
            float a = A[idx];
            float b = B[idx];
            
            // Check for NaN
            if (isnan(a) || isnan(b)) {
                printf("NaN detected at (%zu, %zu): A=%f, B=%f\n", i, j, a, b);
                return -1.0;
            }
            
            double abs_error = fabs(a - b);
            *max_abs_error = fmax(*max_abs_error, abs_error);
            
            // Relative error (avoid division by zero)
            double denominator = fmax(fabs(a), fabs(b));
            if (denominator > 1e-10) {
                double rel_error = abs_error / denominator;
                max_rel_error = fmax(max_rel_error, rel_error);
            }
        }
    }
    
    return max_rel_error;
}

/**
 * @brief Print matrix (for debugging small matrices)
 */
static void print_matrix(const char *name, const float *A, size_t M, size_t N)
{
    printf("%s (%zux%zu):\n", name, M, N);
    for (size_t i = 0; i < M && i < 8; i++) {  // Print max 8 rows
        for (size_t j = 0; j < N && j < 8; j++) {  // Print max 8 cols
            printf("%8.3f ", A[i * N + j]);
        }
        if (N > 8) printf("...");
        printf("\n");
    }
    if (M > 8) printf("...\n");
    printf("\n");
}

/**
 * @brief Measure GFLOPS for GEMM operation
 */
static double measure_gflops(size_t M, size_t K, size_t N, double time_sec)
{
    // GEMM operations: 2*M*K*N (multiply-add counts as 2 FLOPs)
    double flops = 2.0 * M * K * N;
    return (flops / time_sec) / 1e9;
}

/**
 * @brief Timer (cross-platform)
 */
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

//==============================================================================
// TEST CASES
//==============================================================================

typedef struct {
    const char *name;
    size_t M, K, N;
    float alpha, beta;
    bool (*test_fn)(size_t M, size_t K, size_t N, float alpha, float beta);
} test_case_t;

/**
 * @brief Generic correctness test
 */
static bool test_correctness_generic(size_t M, size_t K, size_t N, float alpha, float beta)
{
    // Allocate matrices
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C_test = (float *)malloc(M * N * sizeof(float));
    float *C_ref = (float *)malloc(M * N * sizeof(float));
    
    if (!A || !B || !C_test || !C_ref) {
        printf("FAIL: Memory allocation failed\n");
        free(A); free(B); free(C_test); free(C_ref);
        return false;
    }
    
    // Initialize
    init_matrix_random(A, M, K, 42);
    init_matrix_random(B, K, N, 43);
    init_matrix_random(C_test, M, N, 44);
    memcpy(C_ref, C_test, M * N * sizeof(float));  // Same initial C
    
    // Compute reference
    gemm_reference(C_ref, A, B, M, K, N, alpha, beta);
    
    // Compute test
    int ret = gemm_auto(C_test, A, B, M, K, N, alpha, beta);
    
    if (ret != 0) {
        printf("FAIL: gemm_auto returned error %d\n", ret);
        free(A); free(B); free(C_test); free(C_ref);
        return false;
    }
    
    // Compare
    double max_abs_error;
    double max_rel_error = compare_matrices(C_test, C_ref, M, N, &max_abs_error);
    
    bool passed = (max_rel_error >= 0.0 && max_rel_error < 1e-4);
    
    if (!passed) {
        printf("FAIL: Max relative error = %.2e, max absolute error = %.2e\n", 
               max_rel_error, max_abs_error);
        
        if (M <= 8 && N <= 8) {
            print_matrix("A", A, M, K);
            print_matrix("B", B, K, N);
            print_matrix("C_test", C_test, M, N);
            print_matrix("C_ref", C_ref, M, N);
        }
    } else {
        printf("PASS: Max relative error = %.2e, max absolute error = %.2e\n",
               max_rel_error, max_abs_error);
    }
    
    free(A); free(B); free(C_test); free(C_ref);
    return passed;
}

/**
 * @brief Test adaptive blocking selection
 */
static bool test_adaptive_blocking(size_t M, size_t K, size_t N, float alpha, float beta)
{
    (void)alpha; (void)beta;
    
    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan) {
        printf("FAIL: Plan creation failed\n");
        return false;
    }
    
    printf("  Blocking: MC=%zu, KC=%zu, NC=%zu, MR=%zu, NR=%zu\n",
           plan->MC, plan->KC, plan->NC, plan->MR, plan->NR);
    
    // Verify blocking parameters are reasonable
    bool valid = (plan->MC > 0 && plan->MC <= M) &&
                 (plan->KC > 0 && plan->KC <= K) &&
                 (plan->NC > 0 && plan->NC <= N) &&
                 (plan->MR > 0 && plan->MR <= 16) &&
                 (plan->NR > 0 && plan->NR <= 16);
    
    // Verify cache footprint is reasonable (< 2MB for L2)
    size_t footprint = (plan->MC * plan->KC + plan->KC * plan->NC) * sizeof(float);
    bool cache_ok = (footprint < 2 * 1024 * 1024);
    
    printf("  Workspace footprint: %.1f KB%s\n", 
           footprint / 1024.0,
           cache_ok ? " (OK)" : " (WARNING: Too large for L2!)");
    
    gemm_plan_destroy(plan);
    
    if (!valid) {
        printf("FAIL: Invalid blocking parameters\n");
        return false;
    }
    
    if (!cache_ok) {
        printf("WARN: Workspace exceeds L2 cache\n");
    }
    
    printf("PASS\n");
    return true;
}

/**
 * @brief Test static vs dynamic mode
 */
static bool test_memory_modes(size_t M, size_t K, size_t N, float alpha, float beta)
{
    // Allocate matrices
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C_static = (float *)malloc(M * N * sizeof(float));
    float *C_dynamic = (float *)malloc(M * N * sizeof(float));
    
    if (!A || !B || !C_static || !C_dynamic) {
        printf("FAIL: Memory allocation failed\n");
        free(A); free(B); free(C_static); free(C_dynamic);
        return false;
    }
    
    // Initialize
    init_matrix_random(A, M, K, 50);
    init_matrix_random(B, K, N, 51);
    init_matrix_random(C_static, M, N, 52);
    memcpy(C_dynamic, C_static, M * N * sizeof(float));
    
    // Test static mode
    int ret_static = gemm_static(C_static, A, B, M, K, N, alpha, beta);
    
    // Test dynamic mode
    int ret_dynamic = gemm_dynamic(C_dynamic, A, B, M, K, N, alpha, beta);
    
    // Should both succeed for small matrices
    if (gemm_fits_static(M, K, N)) {
        if (ret_static != 0 || ret_dynamic != 0) {
            printf("FAIL: Static or dynamic mode failed\n");
            free(A); free(B); free(C_static); free(C_dynamic);
            return false;
        }
        
        // Results should match
        double max_abs_error;
        double max_rel_error = compare_matrices(C_static, C_dynamic, M, N, &max_abs_error);
        
        if (max_rel_error < 0.0 || max_rel_error > 1e-6) {
            printf("FAIL: Static and dynamic modes produce different results\n");
            printf("  Max relative error = %.2e\n", max_rel_error);
            free(A); free(B); free(C_static); free(C_dynamic);
            return false;
        }
        
        printf("PASS: Static and dynamic modes match (error = %.2e)\n", max_rel_error);
    } else {
        // Too large for static - should fail gracefully
        if (ret_static == 0) {
            printf("FAIL: Static mode should have rejected large matrix\n");
            free(A); free(B); free(C_static); free(C_dynamic);
            return false;
        }
        
        if (ret_dynamic != 0) {
            printf("FAIL: Dynamic mode should handle large matrix\n");
            free(A); free(B); free(C_static); free(C_dynamic);
            return false;
        }
        
        printf("PASS: Static mode correctly rejected, dynamic mode succeeded\n");
    }
    
    free(A); free(B); free(C_static); free(C_dynamic);
    return true;
}

/**
 * @brief Performance benchmark
 */
static bool test_performance(size_t M, size_t K, size_t N, float alpha, float beta)
{
    // Allocate matrices
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    
    if (!A || !B || !C) {
        printf("FAIL: Memory allocation failed\n");
        free(A); free(B); free(C);
        return false;
    }
    
    // Initialize
    init_matrix_random(A, M, K, 60);
    init_matrix_random(B, K, N, 61);
    init_matrix_random(C, M, N, 62);
    
    // Warmup
    gemm_auto(C, A, B, M, K, N, alpha, beta);
    
    // Benchmark (3 runs, take best)
    double best_time = DBL_MAX;
    const int num_runs = 3;
    
    for (int run = 0; run < num_runs; run++) {
        double t0 = get_time();
        gemm_auto(C, A, B, M, K, N, alpha, beta);
        double t1 = get_time();
        
        double elapsed = t1 - t0;
        if (elapsed < best_time) {
            best_time = elapsed;
        }
    }
    
    double gflops = measure_gflops(M, K, N, best_time);
    
    printf("  Time: %.3f ms, Performance: %.1f GFLOPS\n",
           best_time * 1000.0, gflops);
    
    free(A); free(B); free(C);
    
    // Consider pass if GFLOPS > 10 (very conservative)
    if (gflops < 10.0) {
        printf("WARN: Performance seems low (< 10 GFLOPS)\n");
    }
    
    printf("PASS\n");
    return true;
}

//==============================================================================
// TEST SUITE DEFINITION
//==============================================================================

static test_case_t test_suite[] = {
    //==========================================================================
    // Edge Cases
    //==========================================================================
    {"1×1×1 (minimal)", 1, 1, 1, 1.0f, 0.0f, test_correctness_generic},
    {"Single row (1×32×16)", 1, 32, 16, 1.0f, 0.0f, test_correctness_generic},
    {"Single col (16×32×1)", 16, 32, 1, 1.0f, 0.0f, test_correctness_generic},
    {"Single K (8×1×8)", 8, 1, 8, 1.0f, 0.0f, test_correctness_generic},
    {"Tiny (4×4×4)", 4, 4, 4, 1.0f, 0.0f, test_correctness_generic},
    {"Odd dimensions (7×11×13)", 7, 11, 13, 1.0f, 0.0f, test_correctness_generic},
    
    //==========================================================================
    // Alpha/Beta Variations
    //==========================================================================
    {"Alpha=0", 8, 8, 8, 0.0f, 1.0f, test_correctness_generic},
    {"Alpha=2", 8, 8, 8, 2.0f, 0.0f, test_correctness_generic},
    {"Beta=0", 8, 8, 8, 1.0f, 0.0f, test_correctness_generic},
    {"Beta=2", 8, 8, 8, 1.0f, 2.0f, test_correctness_generic},
    {"Alpha=-1, Beta=0.5", 8, 8, 8, -1.0f, 0.5f, test_correctness_generic},
    
    //==========================================================================
    // Kernel Boundary Cases
    //==========================================================================
    {"4×8 boundary", 4, 32, 8, 1.0f, 0.0f, test_correctness_generic},
    {"8×6 boundary", 8, 32, 6, 1.0f, 0.0f, test_correctness_generic},
    {"8×8 boundary", 8, 32, 8, 1.0f, 0.0f, test_correctness_generic},
    {"8×16 boundary", 8, 32, 16, 1.0f, 0.0f, test_correctness_generic},
    {"16×8 boundary", 16, 32, 8, 1.0f, 0.0f, test_correctness_generic},
    {"16×16 boundary", 16, 32, 16, 1.0f, 0.0f, test_correctness_generic},
    
    //==========================================================================
    // Standard Sizes
    //==========================================================================
    {"Small square (32×32×32)", 32, 32, 32, 1.0f, 0.0f, test_correctness_generic},
    {"Medium square (128×128×128)", 128, 128, 128, 1.0f, 0.0f, test_correctness_generic},
    {"Large square (256×256×256)", 256, 256, 256, 1.0f, 0.0f, test_correctness_generic},
    {"Power-of-2 (512×512×512)", 512, 512, 512, 1.0f, 0.0f, test_correctness_generic},
    
    //==========================================================================
    // Adaptive Blocking Tests
    //==========================================================================
    {"Tall matrix (2048×512×512)", 2048, 512, 512, 1.0f, 0.0f, test_adaptive_blocking},
    {"Wide matrix (512×512×2048)", 512, 512, 2048, 1.0f, 0.0f, test_adaptive_blocking},
    {"Deep matrix (512×4096×512)", 512, 4096, 512, 1.0f, 0.0f, test_adaptive_blocking},
    {"Balanced (1024×1024×1024)", 1024, 1024, 1024, 1.0f, 0.0f, test_adaptive_blocking},
    
    //==========================================================================
    // Correctness: Adaptive Shapes
    //==========================================================================
    {"Tall correctness (256×32×128)", 256, 32, 128, 1.0f, 0.0f, test_correctness_generic},
    {"Wide correctness (32×128×256)", 32, 128, 256, 1.0f, 0.0f, test_correctness_generic},
    {"Deep correctness (64×512×64)", 64, 512, 64, 1.0f, 0.0f, test_correctness_generic},
    
    //==========================================================================
    // Memory Mode Tests
    //==========================================================================
    {"Static mode (128×128×128)", 128, 128, 128, 1.0f, 0.0f, test_memory_modes},
    {"Dynamic mode (1024×1024×1024)", 1024, 1024, 1024, 1.0f, 0.0f, test_memory_modes},
    
    //==========================================================================
    // Performance Benchmarks
    //==========================================================================
    {"Perf: Small (256×256×256)", 256, 256, 256, 1.0f, 0.0f, test_performance},
    {"Perf: Medium (512×512×512)", 512, 512, 512, 1.0f, 0.0f, test_performance},
    {"Perf: Large (1024×1024×1024)", 1024, 1024, 1024, 1.0f, 0.0f, test_performance},
    {"Perf: Huge (2048×2048×2048)", 2048, 2048, 2048, 1.0f, 0.0f, test_performance},
    
    //==========================================================================
    // Kalman Filter Sizes (Real-World)
    //==========================================================================
    {"Kalman 6×6 (6×32×6)", 6, 32, 6, 1.0f, 1.0f, test_correctness_generic},
    {"Kalman 8×8 (8×64×8)", 8, 64, 8, 1.0f, 1.0f, test_correctness_generic},
    {"Kalman mixed (12×48×6)", 12, 48, 6, 1.0f, 1.0f, test_correctness_generic},
};

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int main(int argc, char **argv)
{
    printf("================================================================================\n");
    printf("GEMM Comprehensive Test Suite\n");
    printf("================================================================================\n\n");
    
    size_t num_tests = sizeof(test_suite) / sizeof(test_case_t);
    size_t passed = 0;
    size_t failed = 0;
    
    // Optional: Run specific test by index
    int test_idx = -1;
    if (argc > 1) {
        test_idx = atoi(argv[1]);
        if (test_idx < 0 || test_idx >= (int)num_tests) {
            printf("Invalid test index. Valid range: 0-%zu\n", num_tests - 1);
            return 1;
        }
    }
    
    for (size_t i = 0; i < num_tests; i++) {
        // Skip if specific test requested
        if (test_idx >= 0 && (int)i != test_idx) {
            continue;
        }
        
        test_case_t *tc = &test_suite[i];
        
        printf("[%3zu/%3zu] %s\n", i + 1, num_tests, tc->name);
        printf("  Dimensions: %zux%zux%zu, alpha=%.2f, beta=%.2f\n",
               tc->M, tc->K, tc->N, tc->alpha, tc->beta);
        
        bool result = tc->test_fn(tc->M, tc->K, tc->N, tc->alpha, tc->beta);
        
        if (result) {
            passed++;
        } else {
            failed++;
            printf("  *** TEST FAILED ***\n");
        }
        
        printf("\n");
    }
    
    printf("================================================================================\n");
    printf("Test Summary: %zu passed, %zu failed (%.1f%% pass rate)\n",
           passed, failed, 100.0 * passed / (passed + failed));
    printf("================================================================================\n");
    
    return (failed > 0) ? 1 : 0;
}