/**
 * @file test_qr_benchmark.c
 * @brief Performance benchmarks for blocked QR decomposition
 * 
 * Measures:
 * - QR factorization throughput (GFLOPS)
 * - Panel factorization time vs trailing update time
 * - Scaling with matrix size
 * - Block size sensitivity
 * 
 * @author TUGBARS
 * @date 2025
 */

#include "qr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

//==============================================================================
// TIMING UTILITIES
//==============================================================================

static double get_time_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

/**
 * @brief Compute FLOP count for QR factorization
 * 
 * QR factorization of m×n matrix requires approximately:
 *   2mn² - (2/3)n³  FLOPs (for m ≥ n)
 * 
 * This includes:
 * - Householder vector generation: O(mn)
 * - Trailing matrix updates: O(mn²)
 * - Q formation (if requested): additional O(m²n)
 */
static double qr_flops(uint16_t m, uint16_t n, bool form_q)
{
    double dm = (double)m;
    double dn = (double)n;
    
    // R factorization: 2mn² - (2/3)n³
    double flops_r = 2.0 * dm * dn * dn - (2.0 / 3.0) * dn * dn * dn;
    
    // Q formation: 2m²n (approximately)
    double flops_q = form_q ? (2.0 * dm * dm * dn) : 0.0;
    
    return flops_r + flops_q;
}

//==============================================================================
// TEST RESULT STRUCTURE
//==============================================================================

typedef struct {
    int total;
    int passed;
    int failed;
} test_results_t;

//==============================================================================
// MATRIX GENERATION
//==============================================================================

/**
 * @brief Generate random matrix with controlled condition number
 */
static void generate_test_matrix(float *A, uint16_t m, uint16_t n, uint32_t seed)
{
    srand(seed);
    
    // Generate random matrix with elements in [-1, 1]
    for (uint32_t i = 0; i < (uint32_t)m * n; i++)
    {
        A[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
    }
    
    // Add diagonal dominance for numerical stability
    uint16_t min_dim = (m < n) ? m : n;
    for (uint16_t i = 0; i < min_dim; i++)
    {
        A[i * n + i] += (float)min_dim;
    }
}

//==============================================================================
// VERIFICATION
//==============================================================================

/**
 * @brief Verify QR factorization: ||A - QR||_F / ||A||_F < tol
 */
static double verify_qr(const float *A_orig, const float *Q, const float *R,
                        uint16_t m, uint16_t n)
{
    // Compute QR
    float *QR = (float *)malloc((size_t)m * n * sizeof(float));
    if (!QR) return -1.0;
    
    memset(QR, 0, (size_t)m * n * sizeof(float));
    
    // QR = Q * R (Q is m×m, R is m×n)
    for (uint16_t i = 0; i < m; i++)
    {
        for (uint16_t j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (uint16_t k = 0; k < m; k++)
            {
                sum += (double)Q[i * m + k] * (double)R[k * n + j];
            }
            QR[i * n + j] = (float)sum;
        }
    }
    
    // Compute ||A - QR||_F and ||A||_F
    double norm_diff = 0.0, norm_A = 0.0;
    for (uint32_t i = 0; i < (uint32_t)m * n; i++)
    {
        double diff = (double)A_orig[i] - (double)QR[i];
        norm_diff += diff * diff;
        norm_A += (double)A_orig[i] * (double)A_orig[i];
    }
    
    free(QR);
    
    return sqrt(norm_diff) / sqrt(norm_A);
}

//==============================================================================
// BENCHMARK: SINGLE SIZE
//==============================================================================

typedef struct {
    uint16_t m;
    uint16_t n;
    uint16_t ib;
    double time_factor;      // Time for factorization only
    double time_form_q;      // Time to form Q
    double time_total;       // Total time including Q
    double gflops_factor;    // GFLOPS for factorization
    double gflops_total;     // GFLOPS including Q formation
    double rel_error;        // Reconstruction error
    int verified;            // 1 if passed verification
} benchmark_result_t;

/**
 * @brief Run benchmark for a single matrix size
 */
static benchmark_result_t benchmark_single(uint16_t m, uint16_t n, uint16_t ib,
                                           int num_warmup, int num_runs)
{
    benchmark_result_t result = {0};
    result.m = m;
    result.n = n;
    result.ib = ib;
    
    // Allocate matrices
    float *A = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    float *A_work = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    float *Q = (float *)aligned_alloc(32, (size_t)m * m * sizeof(float));
    float *R = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    
    if (!A || !A_work || !Q || !R)
    {
        printf("    ERROR: Allocation failed for %dx%d\n", m, n);
        free(A); free(A_work); free(Q); free(R);
        return result;
    }
    
    // Generate test matrix
    generate_test_matrix(A, m, n, 12345 + m * n);
    
    // Allocate workspace
    qr_workspace *ws = qr_workspace_alloc(m, n, ib);
    if (!ws)
    {
        printf("    ERROR: Workspace allocation failed\n");
        free(A); free(A_work); free(Q); free(R);
        return result;
    }
    
    // Warmup runs
    for (int i = 0; i < num_warmup; i++)
    {
        memcpy(A_work, A, (size_t)m * n * sizeof(float));
        qr_ws_blocked_inplace(ws, A_work, Q, R, m, n, false);
    }
    
    // Timed runs
    double total_time = 0.0;
    
    for (int i = 0; i < num_runs; i++)
    {
        memcpy(A_work, A, (size_t)m * n * sizeof(float));
        
        double t0 = get_time_seconds();
        qr_ws_blocked_inplace(ws, A_work, Q, R, m, n, false);
        double t1 = get_time_seconds();
        
        total_time += (t1 - t0);
    }
    
    result.time_total = total_time / num_runs;
    
    // Compute GFLOPS
    double flops = qr_flops(m, n, true);
    result.gflops_total = (flops / result.time_total) / 1e9;
    
    // Verify last result
    result.rel_error = verify_qr(A, Q, R, m, n);
    result.verified = (result.rel_error < 1e-4) ? 1 : 0;
    
    // Cleanup
    qr_workspace_free(ws);
    free(A);
    free(A_work);
    free(Q);
    free(R);
    
    return result;
}

//==============================================================================
// BENCHMARK: SCALING TEST
//==============================================================================

static int test_scaling_benchmark(void)
{
    printf("\n=== QR Scaling Benchmark ===\n");
    printf("  (3 warmup, 5 timed runs per size)\n\n");
    
    printf("  %-8s %-8s %-8s %-12s %-12s %-12s %s\n",
           "M", "N", "IB", "Time(ms)", "GFLOPS", "Error", "Status");
    printf("  %-8s %-8s %-8s %-12s %-12s %-12s %s\n",
           "--------", "--------", "--------", "------------",
           "------------", "------------", "------");
    
    // Test sizes: small to large
    typedef struct { uint16_t m; uint16_t n; } size_pair_t;
    size_pair_t sizes[] = {
        {32, 32},
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {128, 32},   // Tall
        {256, 64},   // Tall
        {512, 128},  // Tall
        {64, 128},   // Wide (R only would be faster)
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    int all_passed = 1;
    
    for (int i = 0; i < num_sizes; i++)
    {
        uint16_t m = sizes[i].m;
        uint16_t n = sizes[i].n;
        
        benchmark_result_t res = benchmark_single(m, n, 0, 3, 5);
        
        printf("  %-8d %-8d %-8d %-12.3f %-12.2f %-12.2e %s\n",
               m, n, res.ib,
               res.time_total * 1000.0,
               res.gflops_total,
               res.rel_error,
               res.verified ? "PASS" : "FAIL");
        
        if (!res.verified) all_passed = 0;
    }
    
    return all_passed;
}

//==============================================================================
// BENCHMARK: BLOCK SIZE SENSITIVITY
//==============================================================================

static int test_blocksize_benchmark(void)
{
    printf("\n=== Block Size Sensitivity (256×256) ===\n");
    printf("  (3 warmup, 10 timed runs per block size)\n\n");
    
    printf("  %-8s %-12s %-12s %-12s %s\n",
           "IB", "Time(ms)", "GFLOPS", "Error", "Status");
    printf("  %-8s %-12s %-12s %-12s %s\n",
           "--------", "------------", "------------", "------------", "------");
    
    const uint16_t m = 256, n = 256;
    uint16_t block_sizes[] = {8, 16, 24, 32, 48, 64, 96, 128};
    int num_bs = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    double best_time = 1e9;
    uint16_t best_ib = 0;
    int all_passed = 1;
    
    for (int i = 0; i < num_bs; i++)
    {
        uint16_t ib = block_sizes[i];
        if (ib > n) continue;
        
        benchmark_result_t res = benchmark_single(m, n, ib, 3, 10);
        
        printf("  %-8d %-12.3f %-12.2f %-12.2e %s\n",
               ib,
               res.time_total * 1000.0,
               res.gflops_total,
               res.rel_error,
               res.verified ? "PASS" : "FAIL");
        
        if (res.time_total < best_time && res.verified)
        {
            best_time = res.time_total;
            best_ib = ib;
        }
        
        if (!res.verified) all_passed = 0;
    }
    
    printf("\n  Best block size: IB=%d (%.3f ms, %.2f GFLOPS)\n",
           best_ib, best_time * 1000.0,
           qr_flops(m, n, true) / best_time / 1e9);
    
    return all_passed;
}

//==============================================================================
// BENCHMARK: TALL MATRIX PERFORMANCE
//==============================================================================

static int test_tall_matrix_benchmark(void)
{
    printf("\n=== Tall Matrix Performance (M >> N) ===\n");
    printf("  (3 warmup, 5 timed runs per size)\n\n");
    
    printf("  %-8s %-8s %-10s %-12s %-12s %-12s %s\n",
           "M", "N", "Aspect", "Time(ms)", "GFLOPS", "Error", "Status");
    printf("  %-8s %-8s %-10s %-12s %-12s %-12s %s\n",
           "--------", "--------", "----------", "------------",
           "------------", "------------", "------");
    
    typedef struct { uint16_t m; uint16_t n; } size_pair_t;
    size_pair_t sizes[] = {
        {256, 32},    // 8:1
        {512, 64},    // 8:1
        {1024, 128},  // 8:1
        {256, 64},    // 4:1
        {512, 128},   // 4:1
        {1024, 256},  // 4:1
        {512, 256},   // 2:1
        {1024, 512},  // 2:1
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    int all_passed = 1;
    
    for (int i = 0; i < num_sizes; i++)
    {
        uint16_t m = sizes[i].m;
        uint16_t n = sizes[i].n;
        double aspect = (double)m / (double)n;
        
        benchmark_result_t res = benchmark_single(m, n, 0, 3, 5);
        
        printf("  %-8d %-8d %-10.1f %-12.3f %-12.2f %-12.2e %s\n",
               m, n, aspect,
               res.time_total * 1000.0,
               res.gflops_total,
               res.rel_error,
               res.verified ? "PASS" : "FAIL");
        
        if (!res.verified) all_passed = 0;
    }
    
    return all_passed;
}

//==============================================================================
// BENCHMARK: R-ONLY MODE
//==============================================================================

static int test_r_only_benchmark(void)
{
    printf("\n=== R-Only Mode (Skip Q Formation) ===\n");
    printf("  Useful for least squares where Q is not needed\n\n");
    
    const uint16_t m = 512, n = 128;
    
    float *A = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    float *A_work = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    float *Q = (float *)aligned_alloc(32, (size_t)m * m * sizeof(float));
    float *R = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    
    if (!A || !A_work || !Q || !R)
    {
        printf("  ERROR: Allocation failed\n");
        free(A); free(A_work); free(Q); free(R);
        return 0;
    }
    
    generate_test_matrix(A, m, n, 99999);
    
    qr_workspace *ws = qr_workspace_alloc(m, n, 0);
    
    // Benchmark with Q
    int num_runs = 10;
    double time_with_q = 0.0;
    
    for (int i = 0; i < num_runs; i++)
    {
        memcpy(A_work, A, (size_t)m * n * sizeof(float));
        double t0 = get_time_seconds();
        qr_ws_blocked_inplace(ws, A_work, Q, R, m, n, false);
        double t1 = get_time_seconds();
        time_with_q += (t1 - t0);
    }
    time_with_q /= num_runs;
    
    // Benchmark R-only
    double time_r_only = 0.0;
    
    for (int i = 0; i < num_runs; i++)
    {
        memcpy(A_work, A, (size_t)m * n * sizeof(float));
        double t0 = get_time_seconds();
        qr_ws_blocked_inplace(ws, A_work, NULL, R, m, n, true);
        double t1 = get_time_seconds();
        time_r_only += (t1 - t0);
    }
    time_r_only /= num_runs;
    
    double speedup = time_with_q / time_r_only;
    double flops_r = qr_flops(m, n, false);
    double flops_qr = qr_flops(m, n, true);
    
    printf("  Matrix size: %d × %d\n", m, n);
    printf("  With Q formation:    %.3f ms (%.2f GFLOPS)\n",
           time_with_q * 1000.0, flops_qr / time_with_q / 1e9);
    printf("  R-only mode:         %.3f ms (%.2f GFLOPS)\n",
           time_r_only * 1000.0, flops_r / time_r_only / 1e9);
    printf("  Speedup (R-only):    %.2fx\n", speedup);
    
    qr_workspace_free(ws);
    free(A); free(A_work); free(Q); free(R);
    
    return 1;
}

//==============================================================================
// BENCHMARK: THROUGHPUT TEST
//==============================================================================

static int test_throughput_benchmark(void)
{
    printf("\n=== Sustained Throughput Test ===\n");
    printf("  Running 100 consecutive QR factorizations (128×128)\n\n");
    
    const uint16_t m = 128, n = 128;
    const int num_runs = 100;
    
    float *A = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    float *A_work = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    float *Q = (float *)aligned_alloc(32, (size_t)m * m * sizeof(float));
    float *R = (float *)aligned_alloc(32, (size_t)m * n * sizeof(float));
    
    if (!A || !A_work || !Q || !R)
    {
        printf("  ERROR: Allocation failed\n");
        free(A); free(A_work); free(Q); free(R);
        return 0;
    }
    
    generate_test_matrix(A, m, n, 11111);
    
    qr_workspace *ws = qr_workspace_alloc(m, n, 0);
    
    // Warmup
    for (int i = 0; i < 5; i++)
    {
        memcpy(A_work, A, (size_t)m * n * sizeof(float));
        qr_ws_blocked_inplace(ws, A_work, Q, R, m, n, false);
    }
    
    // Timed runs
    double times[100];
    double total_time = 0.0;
    
    for (int i = 0; i < num_runs; i++)
    {
        memcpy(A_work, A, (size_t)m * n * sizeof(float));
        
        double t0 = get_time_seconds();
        qr_ws_blocked_inplace(ws, A_work, Q, R, m, n, false);
        double t1 = get_time_seconds();
        
        times[i] = t1 - t0;
        total_time += times[i];
    }
    
    // Statistics
    double avg_time = total_time / num_runs;
    double min_time = times[0], max_time = times[0];
    double variance = 0.0;
    
    for (int i = 0; i < num_runs; i++)
    {
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
        double diff = times[i] - avg_time;
        variance += diff * diff;
    }
    variance /= num_runs;
    double stddev = sqrt(variance);
    
    double flops = qr_flops(m, n, true);
    
    printf("  Runs:          %d\n", num_runs);
    printf("  Avg time:      %.3f ms\n", avg_time * 1000.0);
    printf("  Min time:      %.3f ms\n", min_time * 1000.0);
    printf("  Max time:      %.3f ms\n", max_time * 1000.0);
    printf("  Std dev:       %.3f ms (%.1f%%)\n", 
           stddev * 1000.0, 100.0 * stddev / avg_time);
    printf("  Avg GFLOPS:    %.2f\n", flops / avg_time / 1e9);
    printf("  Peak GFLOPS:   %.2f\n", flops / min_time / 1e9);
    
    qr_workspace_free(ws);
    free(A); free(A_work); free(Q); free(R);
    
    return 1;
}

//==============================================================================
// MAIN BENCHMARK RUNNER
//==============================================================================

int run_qr_benchmarks(test_results_t *results)
{
    printf("=================================================\n");
    printf("    BLOCKED QR DECOMPOSITION BENCHMARKS\n");
    printf("=================================================\n");
    
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    // Run all benchmarks
    printf("\n--- Scaling Tests ---\n");
    
    results->total++;
    if (test_scaling_benchmark())
    {
        results->passed++;
        printf("✓ Scaling benchmark PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Scaling benchmark FAILED\n");
    }
    
    results->total++;
    if (test_blocksize_benchmark())
    {
        results->passed++;
        printf("✓ Block size benchmark PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Block size benchmark FAILED\n");
    }
    
    printf("\n--- Shape Tests ---\n");
    
    results->total++;
    if (test_tall_matrix_benchmark())
    {
        results->passed++;
        printf("✓ Tall matrix benchmark PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Tall matrix benchmark FAILED\n");
    }
    
    results->total++;
    if (test_r_only_benchmark())
    {
        results->passed++;
        printf("✓ R-only benchmark PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ R-only benchmark FAILED\n");
    }
    
    printf("\n--- Stability Tests ---\n");
    
    results->total++;
    if (test_throughput_benchmark())
    {
        results->passed++;
        printf("✓ Throughput benchmark PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Throughput benchmark FAILED\n");
    }
    
    printf("\n=================================================\n");
    printf("QR Benchmarks: %d/%d passed\n", results->passed, results->total);
    
    if (results->passed == results->total)
    {
        printf("✓ ALL QR BENCHMARKS PASSED!\n");
    }
    else
    {
        printf("✗ %d QR benchmarks FAILED\n", results->failed);
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
    return run_qr_benchmarks(&results);
}
#endif