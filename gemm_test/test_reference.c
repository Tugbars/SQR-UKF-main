#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../gemm/gemm.h"


               // Aligned allocation helper
static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    size = (size + (alignment - 1)) & ~(alignment - 1);
    return aligned_alloc(alignment, size);
#endif
}
static void aligned_free_wrapper(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#define MAX_ERR_REL 1e-3f  // Relaxed for accumulation errors

// Naive reference GEMM
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

#include <malloc.h>  // <-- needed on Windows for _aligned_malloc/_aligned_free

static int compare_with_reference(int M, int K, int N, const char *name) {
#ifdef _WIN32
    float *A      = (float*)_aligned_malloc((size_t)M * K * sizeof(float), 32);
    float *B      = (float*)_aligned_malloc((size_t)K * N * sizeof(float), 32);
    float *C_test = (float*)_aligned_malloc((size_t)M * N * sizeof(float), 32);
    float *C_ref  = (float*)_aligned_malloc((size_t)M * N * sizeof(float), 32);
#else
    float *A = NULL, *B = NULL, *C_test = NULL, *C_ref = NULL;
    if (posix_memalign((void**)&A,      32, (size_t)M * K * sizeof(float)) != 0) A = NULL;
    if (posix_memalign((void**)&B,      32, (size_t)K * N * sizeof(float)) != 0) B = NULL;
    if (posix_memalign((void**)&C_test, 32, (size_t)M * N * sizeof(float)) != 0) C_test = NULL;
    if (posix_memalign((void**)&C_ref,  32, (size_t)M * N * sizeof(float)) != 0) C_ref  = NULL;
#endif

    if (!A || !B || !C_test || !C_ref) {
        printf("  %s: ALLOC FAILED\n", name);
#ifdef _WIN32
        if (A) _aligned_free(A);
        if (B) _aligned_free(B);
        if (C_test) _aligned_free(C_test);
        if (C_ref) _aligned_free(C_ref);
#else
        free(A); free(B); free(C_test); free(C_ref);
#endif
        return 0;
    }

    // Random init
    for (int i = 0; i < M * K; ++i) A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    int ret = mul(C_test, A, B, (uint16_t)M, (uint16_t)K, (uint16_t)K, (uint16_t)N);
    if (ret != 0) {
        printf("  %s: mul() returned error %d\n", name, ret);
#ifdef _WIN32
        _aligned_free(A); _aligned_free(B); _aligned_free(C_test); _aligned_free(C_ref);
#else
        free(A); free(B); free(C_test); free(C_ref);
#endif
        return 0;
    }

    naive_gemm(C_ref, A, B, M, K, N);

    float max_err = 0.f, sum_ref = 0.f;
    for (int i = 0; i < M * N; ++i) {
        float err = fabsf(C_test[i] - C_ref[i]);
        if (err > max_err) max_err = err;
        sum_ref += fabsf(C_ref[i]);
    }
    float rel = max_err / (sum_ref / (M * N) + 1e-10f);
    int pass = (rel < MAX_ERR_REL);
    printf("  %s: %s (abs_err=%.2e, rel_err=%.2e)\n", name, pass ? "PASS" : "FAIL", max_err, rel);

#ifdef _WIN32
    _aligned_free(A); _aligned_free(B); _aligned_free(C_test); _aligned_free(C_ref);
#else
    free(A); free(B); free(C_test); free(C_ref);
#endif
    return pass;
}


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