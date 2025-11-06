// test_gemm_ref.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "../gemm/gemm.h"

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h> // _aligned_malloc/_aligned_free
#endif

//==============================
// High-resolution timer
//==============================
static double get_time_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

//==============================
// Aligned allocation helpers
//==============================
static inline size_t round_up(size_t x, size_t align) {
    return (x + (align - 1)) & ~(align - 1);
}

static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
#ifdef _WIN32
    // _aligned_malloc has no "size must be multiple of alignment" constraint
    return _aligned_malloc(size, alignment);
#elif defined(_POSIX_VERSION)
    // posix_memalign guarantees alignment without the C11 size multiple constraint
    void* p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
#else
    // C11 aligned_alloc requires size be a multiple of alignment
    size_t padded = round_up(size, alignment);
    return aligned_alloc(alignment, padded);
#endif
}

static void aligned_free_wrapper(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#define MAX_ERR_REL 1e-3f  // Relaxed for accumulation errors

//==============================
// Naive reference GEMM
//==============================
static void naive_gemm(float *C, const float *A, const float *B,
                       int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

//==============================
// One test case against reference
//==============================
static int compare_with_reference(int M, int K, int N, const char *name) {
    const size_t bytesA = (size_t)M * (size_t)K * sizeof(float);
    const size_t bytesB = (size_t)K * (size_t)N * sizeof(float);
    const size_t bytesC = (size_t)M * (size_t)N * sizeof(float);

    float *A      = (float*)aligned_alloc_wrapper(32, bytesA);
    float *B      = (float*)aligned_alloc_wrapper(32, bytesB);
    float *C_test = (float*)aligned_alloc_wrapper(32, bytesC);
    float *C_ref  = (float*)aligned_alloc_wrapper(32, bytesC);

    if (!A || !B || !C_test || !C_ref) {
        printf("  %s: ALLOC FAILED (A=%p B=%p C_test=%p C_ref=%p)\n",
               name, (void*)A, (void*)B, (void*)C_test, (void*)C_ref);
        aligned_free_wrapper(A);
        aligned_free_wrapper(B);
        aligned_free_wrapper(C_test);
        aligned_free_wrapper(C_ref);
        return 0;
    }

    // Random initialization (deterministic if caller seeds rand)
    for (size_t i = 0; i < (size_t)M * (size_t)K; i++)
        A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (size_t i = 0; i < (size_t)K * (size_t)N; i++)
        B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // Benchmark optimized GEMM
    double t_start = get_time_sec();
    int ret = mul(C_test, A, B,
                  (uint16_t)M, (uint16_t)K,
                  (uint16_t)K, (uint16_t)N);
    double t_opt = get_time_sec() - t_start;

    if (ret != 0) {
        printf("  %s: mul() returned error %d\n", name, ret);
        aligned_free_wrapper(A);
        aligned_free_wrapper(B);
        aligned_free_wrapper(C_test);
        aligned_free_wrapper(C_ref);
        return 0;
    }

    // Benchmark naive GEMM
    t_start = get_time_sec();
    naive_gemm(C_ref, A, B, M, K, N);
    double t_naive = get_time_sec() - t_start;

    // Compare
    float max_err = 0.0f, sum_ref = 0.0f;
    for (size_t i = 0; i < (size_t)M * (size_t)N; i++) {
        float err = fabsf(C_test[i] - C_ref[i]);
        if (err > max_err) max_err = err;
        sum_ref += fabsf(C_ref[i]);
    }

    float avg_ref = sum_ref / (float)((size_t)M * (size_t)N);
    float relative_err = max_err / (avg_ref + 1e-10f);

    // GFLOPS
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_opt = (flops / t_opt) * 1e-9;
    double gflops_naive = (flops / t_naive) * 1e-9;
    double speedup = t_naive / t_opt;

    int pass = (relative_err < MAX_ERR_REL);
    printf("  %s: %s (err=%.2e) | Opt: %.3f ms (%.1f GFLOPS) | Naive: %.3f ms (%.1f GFLOPS) | Speedup: %.1fx\n",
           name,
           pass ? "PASS" : "FAIL",
           relative_err,
           t_opt * 1000.0, gflops_opt,
           t_naive * 1000.0, gflops_naive,
           speedup);

    if (!pass) {
        for (size_t i = 0; i < (size_t)M * (size_t)N; i++) {
            if (fabsf(C_test[i] - C_ref[i]) > 1e-4f) {
                printf("    First error at [%zu]: got %.6f, expected %.6f\n",
                       i, C_test[i], C_ref[i]);
                break;
            }
        }
    }

    aligned_free_wrapper(A);
    aligned_free_wrapper(B);
    aligned_free_wrapper(C_test);
    aligned_free_wrapper(C_ref);
    return pass;
}

//==============================
// Test suites
//==============================
void test_reference_small(void) {
    compare_with_reference(8, 8, 8, "8×8×8");
    compare_with_reference(16, 16, 16, "16×16×16");
    compare_with_reference(32, 32, 32, "32×32×32");
    compare_with_reference(13, 17, 11, "13×17×11 (primes)");
    compare_with_reference(64, 64, 64, "64×64×64");
}

void test_reference_large(void) {
    compare_with_reference(128, 256, 256, "128×256×256 (exact blocks)");
    compare_with_reference(129, 257, 255, "129×257×255 (off-by-one)");
    compare_with_reference(256, 256, 256, "256×256×256");
    compare_with_reference(512, 512, 512, "512×512×512");
    compare_with_reference(100, 200, 150, "100×200×150 (irregular)");
}

