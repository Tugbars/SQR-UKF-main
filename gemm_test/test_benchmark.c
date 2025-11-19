/**
 * @file test_benchmark.c
 * @brief GEMM Performance Benchmark - Accurate Timing
 *
 * Methodology:
 * - Warmup iterations (prime caches)
 * - Multiple timing runs (find minimum)
 * - Microsecond precision for small times
 * - Plan created ONCE (execution-only benchmark)
 */

#include "gemm.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

//==============================================================================
// UTILITIES
//==============================================================================

static void disable_denormals(void)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#ifdef _MM_DENORMALS_ZERO_ON
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

static double now_sec(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da < db) ? -1 : (da > db);
}

static double median(double *arr, int n)
{
    if (n == 0)
        return 0.0;
    qsort(arr, n, sizeof(double), cmp_double);
    return (n % 2) ? arr[n / 2] : (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
}

static double minimum(double *arr, int n)
{
    if (n == 0)
        return 0.0;
    double min_val = arr[0];
    for (int i = 1; i < n; i++)
    {
        if (arr[i] < min_val)
            min_val = arr[i];
    }
    return min_val;
}

//==============================================================================
// NAIVE GEMM
//==============================================================================

static void naive(float *C, const float *A, const float *B, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            const float *arow = A + (size_t)i * K;
            for (int k = 0; k < K; k++)
            {
                sum += arow[k] * B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = sum;
        }
    }
}

//==============================================================================
// VERIFICATION
//==============================================================================

static float rel_err_max(const float *x, const float *y, size_t n)
{
    float maxe = 0.0f, meanref = 0.0f;
    for (size_t i = 0; i < n; i++)
    {
        float e = fabsf(x[i] - y[i]);
        if (e > maxe)
            maxe = e;
        meanref += fabsf(y[i]);
    }
    return maxe / ((meanref / (float)n) + 1e-20f);
}

//==============================================================================
// BENCHMARK
//==============================================================================

void bench_case(int M, int K, int N, int warmup, int runs, int verify, const char *desc)
{
    size_t szA = (size_t)M * K;
    size_t szB = (size_t)K * N;
    size_t szC = (size_t)M * N;

    float *A = gemm_aligned_alloc(64, szA * sizeof(float));
    float *B = gemm_aligned_alloc(64, szB * sizeof(float));
    float *C = gemm_aligned_alloc(64, szC * sizeof(float));
    float *Cref = verify ? gemm_aligned_alloc(64, szC * sizeof(float)) : NULL;

    if (!A || !B || !C || (verify && !Cref))
    {
        fprintf(stderr, "Allocation failed\n");
        goto cleanup;
    }

    // Initialize with LCG
    for (size_t i = 0; i < szA; i++)
    {
        A[i] = (float)((int32_t)(i * 1103515245u + 12345u)) / (float)INT32_MAX;
    }
    for (size_t i = 0; i < szB; i++)
    {
        B[i] = (float)((int32_t)(i * 1664525u + 1013904223u)) / (float)INT32_MAX;
    }

    // Verify correctness
    if (verify)
    {
        memset(C, 0, szC * sizeof(float));
        gemm_auto(C, A, B, M, K, N, 1.0f, 0.0f);

        memset(Cref, 0, szC * sizeof(float));
        naive(Cref, A, B, M, K, N);

        float err = rel_err_max(C, Cref, szC);
        printf("  Verify %dx%dx%d: rel_err=%.2e %s\n",
               M, K, N, err, (err < 1e-3f ? "✓ PASS" : "✗ FAIL"));
        if (err >= 1e-3f)
            goto cleanup;
    }

    double *times_naive = (double *)malloc(runs * sizeof(double));
    double *times_opt = (double *)malloc(runs * sizeof(double));

    if (!times_naive || !times_opt)
    {
        fprintf(stderr, "Timing array allocation failed\n");
        goto cleanup;
    }

    //--------------------------------------------------------------------------
    // Benchmark NAIVE
    //--------------------------------------------------------------------------

    // Warmup
    for (int w = 0; w < warmup; w++)
    {
        memset(C, 0, szC * sizeof(float));
        naive(C, A, B, M, K, N);
    }

    // Timed runs
    for (int r = 0; r < runs; r++)
    {
        memset(C, 0, szC * sizeof(float));
        double t0 = now_sec();
        naive(C, A, B, M, K, N);
        times_naive[r] = now_sec() - t0;
    }

    //--------------------------------------------------------------------------
    // Benchmark OPTIMIZED (plan created ONCE)
    //--------------------------------------------------------------------------

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        fprintf(stderr, "Plan creation failed\n");
        free(times_naive);
        free(times_opt);
        goto cleanup;
    }

    // Warmup
    for (int w = 0; w < warmup; w++)
    {
        memset(C, 0, szC * sizeof(float));
        gemm_execute_plan(plan, C, A, B, 1.0f, 0.0f);
    }

    // Timed runs
    for (int r = 0; r < runs; r++)
    {
        memset(C, 0, szC * sizeof(float));
        double t0 = now_sec();
        gemm_execute_plan(plan, C, A, B, 1.0f, 0.0f);
        times_opt[r] = now_sec() - t0;
    }

    gemm_plan_destroy(plan);

    //--------------------------------------------------------------------------
    // Statistics
    //--------------------------------------------------------------------------

    double min_naive = minimum(times_naive, runs);
    double med_naive = median(times_naive, runs);
    double min_opt = minimum(times_opt, runs);
    double med_opt = median(times_opt, runs);

    free(times_naive);
    free(times_opt);

    //--------------------------------------------------------------------------
    // Report
    //--------------------------------------------------------------------------

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_naive = (flops / min_naive) / 1e9;
    double gflops_opt = (flops / min_opt) / 1e9;
    double speedup = min_naive / min_opt;

    printf("\n=== %s: %dx%dx%d (%d runs) ===\n", desc, M, K, N, runs);

    // Use appropriate time unit
    if (min_naive < 1e-3)
    {
        printf("  Naive:     %7.1f µs  (%.1f GFLOPS)\n", min_naive * 1e6, gflops_naive);
        printf("  Optimized: %7.1f µs  (%.1f GFLOPS)\n", min_opt * 1e6, gflops_opt);
    }
    else
    {
        printf("  Naive:     %7.3f ms  (%.1f GFLOPS)\n", min_naive * 1e3, gflops_naive);
        printf("  Optimized: %7.3f ms  (%.1f GFLOPS)\n", min_opt * 1e3, gflops_opt);
    }
    printf("  Speedup:   %.2fx faster\n", speedup);

cleanup:
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C);
    if (Cref)
        gemm_aligned_free(Cref);
}

//==============================================================================
// MAIN
//==============================================================================

int main(void)
{
    disable_denormals();

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          GEMM Performance Benchmark                        ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    const int verify = 1;

    printf("\n────────────────────────────────────────────────────────────\n");
    printf(" MEDIUM MATRICES (Single blocking)\n");
    printf("────────────────────────────────────────────────────────────\n");
    bench_case(128, 128, 128, 3, 10, verify, "Medium");
    bench_case(256, 256, 256, 2, 7, verify, "Medium");

    printf("\n────────────────────────────────────────────────────────────\n");
    printf(" LARGE MATRICES (Full blocking)\n");
    printf("────────────────────────────────────────────────────────────\n");
    bench_case(512, 512, 512, 2, 5, verify, "Large");
    bench_case(1024, 1024, 1024, 1, 3, verify, "Large");

    printf("\n────────────────────────────────────────────────────────────\n");
    printf(" IRREGULAR SIZES\n");
    printf("────────────────────────────────────────────────────────────\n");
    bench_case(100, 200, 150, 2, 7, verify, "Irregular");
    bench_case(129, 257, 255, 2, 5, verify, "Irregular");

    printf("\n────────────────────────────────────────────────────────────\n");
    printf(" NON-SQUARE MATRICES\n");
    printf("────────────────────────────────────────────────────────────\n");
    bench_case(1024, 256, 256, 2, 5, verify, "Tall");
    bench_case(2048, 128, 128, 2, 5, verify, "Very Tall");
    bench_case(256, 256, 1024, 2, 5, verify, "Wide");
    bench_case(128, 128, 2048, 2, 5, verify, "Very Wide");
    bench_case(256, 1024, 256, 2, 5, verify, "Deep");
    bench_case(128, 2048, 128, 2, 5, verify, "Very Deep");

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║                 Benchmark Complete!                        ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return 0;
}