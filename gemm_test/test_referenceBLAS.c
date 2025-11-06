#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

// Your GEMM function
extern int mul(float *C, const float *A, const float *B,
               uint16_t row_a, uint16_t column_a,
               uint16_t row_b, uint16_t column_b);

#define MAX_ERR_REL 1e-3f  // Relaxed for accumulation errors

static int compare_with_openblas(int M, int K, int N, const char *name) {
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_test = (float*)malloc(M * N * sizeof(float));
    float *C_ref = (float*)malloc(M * N * sizeof(float));
    
    if (!A || !B || !C_test || !C_ref) {
        printf("  %s: ALLOC FAILED\n", name);
        free(A); free(B); free(C_test); free(C_ref);
        return 0;
    }
    
    // Random initialization
    for (int i = 0; i < M * K; i++)
        A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++)
        B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    
    // Your GEMM
    int ret = mul(C_test, A, B, (uint16_t)M, (uint16_t)K, (uint16_t)K, (uint16_t)N);
    if (ret != 0) {
        printf("  %s: mul() returned error %d\n", name, ret);
        free(A); free(B); free(C_test); free(C_ref);
        return 0;
    }
    
    // OpenBLAS reference
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C_ref, N);
    
    // Compare
    float max_err = 0.0f, sum_ref = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(C_test[i] - C_ref[i]);
        max_err = fmaxf(max_err, err);
        sum_ref += fabsf(C_ref[i]);
    }
    
    float avg_ref = sum_ref / (M * N);
    float relative_err = max_err / (avg_ref + 1e-10f);
    
    int pass = (relative_err < MAX_ERR_REL);
    printf("  %s: %s (abs_err=%.2e, rel_err=%.2e)\n", name, 
           pass ? "PASS" : "FAIL", max_err, relative_err);
    
    free(A); free(B); free(C_test); free(C_ref);
    return pass;
}

void test_reference_small_blas(void) {
    compare_with_openblas(8, 8, 8, "8×8×8");
    compare_with_openblas(16, 16, 16, "16×16×16");
    compare_with_openblas(32, 32, 32, "32×32×32");
    compare_with_openblas(13, 17, 11, "13×17×11 (primes)");
}

void test_reference_large_blas(void) {
    compare_with_openblas(128, 256, 256, "128×256×256 (exact blocks)");
    compare_with_openblas(129, 257, 255, "129×257×255 (off-by-one)");
    compare_with_openblas(256, 256, 256, "256×256×256");
}