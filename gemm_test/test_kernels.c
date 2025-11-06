#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

// Include your kernel headers (adjust path as needed)
// Note: You may need to stub out dependencies if these headers require full context
#include "../gemm_kernels_avx2_complete.h"

#define TOLERANCE 1e-4f
#define ALIGN_BYTES 32

// Helper: Build mask (copy from your code)
static inline __m256i gemm_build_mask_avx2(int lanes) {
    static const int mask_table[9][8] __attribute__((aligned(32))) = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {-1, 0, 0, 0, 0, 0, 0, 0},
        {-1, -1, 0, 0, 0, 0, 0, 0},
        {-1, -1, -1, 0, 0, 0, 0, 0},
        {-1, -1, -1, -1, 0, 0, 0, 0},
        {-1, -1, -1, -1, -1, 0, 0, 0},
        {-1, -1, -1, -1, -1, -1, 0, 0},
        {-1, -1, -1, -1, -1, -1, -1, 0},
        {-1, -1, -1, -1, -1, -1, -1, -1}
    };
    __m128i b8 = _mm_loadl_epi64((const __m128i *)mask_table[lanes]);
    return _mm256_cvtepi8_epi32(b8);
}

// Compare matrices with tolerance
static int compare_matrices(const float *C, const float *C_ref, int M, int N,
                           const char *name) {
    float max_err = 0.0f;
    int fail_count = 0;
    int first_fail_idx = -1;
    
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(C[i] - C_ref[i]);
        if (err > max_err) {
            max_err = err;
            if (fail_count == 0) first_fail_idx = i;
        }
        if (err > TOLERANCE) {
            fail_count++;
        }
    }
    
    if (fail_count > 0) {
        printf("  %s: FAILED (%d/%d errors, max_err=%.2e)\n",
               name, fail_count, M*N, max_err);
        if (first_fail_idx >= 0) {
            printf("    First error at [%d]: got %.6f, expected %.6f\n",
                   first_fail_idx, C[first_fail_idx], C_ref[first_fail_idx]);
        }
        return 0;
    }
    
    printf("  %s: PASSED (max_err=%.2e)\n", name, max_err);
    return 1;
}

// Naive reference GEMM
static void naive_gemm(float *C, const float *A, const float *B,
                      int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));
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

// Test 8×8 kernel (store mode)
static void test_8x8_store(void) {
    const int M = 8, K = 16, N = 8;
    
    __attribute__((aligned(ALIGN_BYTES))) float A[M * K];
    __attribute__((aligned(ALIGN_BYTES))) float B[K * N];
    __attribute__((aligned(ALIGN_BYTES))) float C[M * N];
    __attribute__((aligned(ALIGN_BYTES))) float C_expected[M * N];
    
    // Initialize with simple pattern
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 10 + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 10 + 1);
    
    naive_gemm(C_expected, A, B, M, K, N);
    
    // Pack A (8 rows, column-major: K × 8)
    __attribute__((aligned(ALIGN_BYTES))) float Ap[K * 8];
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++)
            Ap[k * 8 + i] = A[i * K + k];
    }
    
    // Pack B (8 columns, row-major: K × 8)
    __attribute__((aligned(ALIGN_BYTES))) float Bp[K * 8];
    for (int k = 0; k < K; k++)
        memcpy(Bp + k * 8, B + k * N, 8 * sizeof(float));
    
    // Call kernel
    memset(C, 0, M * N * sizeof(float));
    __m256i mask = gemm_build_mask_avx2(8);
    gemm_8x8_panel_avx2fma_store(C, N, Ap, Bp, K, M, N, mask);
    
    compare_matrices(C, C_expected, M, N, "8×8 STORE");
}

// Test 8×8 kernel (add mode)
static void test_8x8_add(void) {
    const int M = 8, K = 16, N = 8;
    
    __attribute__((aligned(ALIGN_BYTES))) float A[M * K];
    __attribute__((aligned(ALIGN_BYTES))) float B[K * N];
    __attribute__((aligned(ALIGN_BYTES))) float C[M * N];
    __attribute__((aligned(ALIGN_BYTES))) float C_expected[M * N];
    
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 10 + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 10 + 1);
    
    // Initialize C with existing values (ADD mode)
    for (int i = 0; i < M * N; i++) {
        C[i] = (float)(i % 5);
    }
    memcpy(C_expected, C, M * N * sizeof(float));
    
    // Expected = C_old + A*B
    float C_product[M * N];
    naive_gemm(C_product, A, B, M, K, N);
    for (int i = 0; i < M * N; i++)
        C_expected[i] += C_product[i];
    
    // Pack matrices
    __attribute__((aligned(ALIGN_BYTES))) float Ap[K * 8];
    __attribute__((aligned(ALIGN_BYTES))) float Bp[K * 8];
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++)
            Ap[k * 8 + i] = A[i * K + k];
        memcpy(Bp + k * 8, B + k * N, 8 * sizeof(float));
    }
    
    // Call kernel
    __m256i mask = gemm_build_mask_avx2(8);
    gemm_8x8_panel_avx2fma_add(C, N, Ap, Bp, K, M, N, mask);
    
    compare_matrices(C, C_expected, M, N, "8×8 ADD");
}

// Test 4×8 tail kernel
static void test_4x8_tail(void) {
    const int M = 4, K = 8, N = 8;
    
    __attribute__((aligned(ALIGN_BYTES))) float A[M * K];
    __attribute__((aligned(ALIGN_BYTES))) float B[K * N];
    __attribute__((aligned(ALIGN_BYTES))) float C[M * N];
    __attribute__((aligned(ALIGN_BYTES))) float C_expected[M * N];
    
    for (int i = 0; i < M * K; i++) A[i] = (float)(i + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i + 1);
    
    naive_gemm(C_expected, A, B, M, K, N);
    
    // Pack A (padded to 8 rows)
    __attribute__((aligned(ALIGN_BYTES))) float Ap[K * 8];
    memset(Ap, 0, K * 8 * sizeof(float));
    for (int k = 0; k < K; k++)
        for (int i = 0; i < M; i++)
            Ap[k * 8 + i] = A[i * K + k];
    
    // Pack B
    __attribute__((aligned(ALIGN_BYTES))) float Bp[K * 8];
    for (int k = 0; k < K; k++)
        memcpy(Bp + k * 8, B + k * N, 8 * sizeof(float));
    
    // Call 4×8 kernel
    memset(C, 0, M * N * sizeof(float));
    __m256i mask = gemm_build_mask_avx2(8);
    gemm_4x8_panel_avx2fma_store(C, N, Ap, Bp, K, N, mask);
    
    compare_matrices(C, C_expected, M, N, "4×8 TAIL");
}

// Test 1×8 tail kernel
static void test_1x8_tail(void) {
    const int M = 1, K = 8, N = 8;
    
    __attribute__((aligned(ALIGN_BYTES))) float A[K];
    __attribute__((aligned(ALIGN_BYTES))) float B[K * N];
    __attribute__((aligned(ALIGN_BYTES))) float C[N];
    __attribute__((aligned(ALIGN_BYTES))) float C_expected[N];
    
    for (int k = 0; k < K; k++) A[k] = (float)(k + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i + 1);
    
    naive_gemm(C_expected, A, B, M, K, N);
    
    // Pack A (padded to 8 rows)
    __attribute__((aligned(ALIGN_BYTES))) float Ap[K * 8];
    memset(Ap, 0, K * 8 * sizeof(float));
    for (int k = 0; k < K; k++)
        Ap[k * 8] = A[k];
    
    // Pack B
    __attribute__((aligned(ALIGN_BYTES))) float Bp[K * 8];
    for (int k = 0; k < K; k++)
        memcpy(Bp + k * 8, B + k * N, 8 * sizeof(float));
    
    // Call 1×8 kernel
    memset(C, 0, N * sizeof(float));
    __m256i mask = gemm_build_mask_avx2(8);
    gemm_1x8_panel_avx2fma_store(C, Ap, Bp, K, N, mask);
    
    compare_matrices(C, C_expected, M, N, "1×8 TAIL");
}

void test_kernels_unit(void) {
    printf("Testing 8×8 kernels:\n");
    test_8x8_store();
    test_8x8_add();
    
    printf("\nTesting tail kernels:\n");
    test_4x8_tail();
    test_1x8_tail();
}