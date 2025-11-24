// test_gemm_kernel.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Assembly kernel declarations
extern void gemm_kernel_8x16_store_asm(
    const float *A, const float *B, float *C, size_t ldc, size_t KC);

extern void gemm_kernel_8x16_add_asm(
    const float *A, const float *B, float *C, size_t ldc, size_t KC);

// Reference scalar implementation
static void gemm_8x16_ref_store(
    const float *A, const float *B, float *C, size_t ldc, size_t KC)
{
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < KC; k++) {
                sum += A[k * 8 + i] * B[k * 16 + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

static void gemm_8x16_ref_add(
    const float *A, const float *B, float *C, size_t ldc, size_t KC)
{
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < KC; k++) {
                sum += A[k * 8 + i] * B[k * 16 + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static int check_result(const float *C_ref, const float *C_test, 
                        size_t rows, size_t cols, size_t ldc, float tol)
{
    int errors = 0;
    float max_err = 0.0f;
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float ref = C_ref[i * ldc + j];
            float test = C_test[i * ldc + j];
            float err = fabsf(ref - test);
            float rel = err / (fabsf(ref) + 1e-6f);
            
            if (rel > tol) {
                if (errors < 5) {
                    printf("  MISMATCH [%zu,%zu]: ref=%.6f test=%.6f err=%.6e\n",
                           i, j, ref, test, err);
                }
                errors++;
            }
            if (err > max_err) max_err = err;
        }
    }
    
    if (errors == 0) {
        printf("  PASS (max_err=%.2e)\n", max_err);
    } else {
        printf("  FAIL: %d errors (max_err=%.2e)\n", errors, max_err);
    }
    
    return errors;
}

static void test_store_kernel(size_t KC)
{
    printf("Testing STORE kernel, KC=%zu\n", KC);
    
    float *A = aligned_alloc(32, 8 * KC * sizeof(float));
    float *B = aligned_alloc(32, KC * 16 * sizeof(float));
    float *C_ref = aligned_alloc(32, 8 * 16 * sizeof(float));
    float *C_test = aligned_alloc(32, 8 * 16 * sizeof(float));
    
    for (size_t i = 0; i < 8 * KC; i++) A[i] = randf();
    for (size_t i = 0; i < KC * 16; i++) B[i] = randf();
    
    for (size_t i = 0; i < 8 * 16; i++) {
        C_ref[i] = 999.0f;
        C_test[i] = 999.0f;
    }
    
    gemm_8x16_ref_store(A, B, C_ref, 16, KC);
    gemm_kernel_8x16_store_asm(A, B, C_test, 16, KC);
    
    float tol = KC * 1e-5f;
    check_result(C_ref, C_test, 8, 16, 16, tol);
    
    free(A); free(B); free(C_ref); free(C_test);
}

static void test_add_kernel(size_t KC)
{
    printf("Testing ADD kernel, KC=%zu\n", KC);
    
    float *A = aligned_alloc(32, 8 * KC * sizeof(float));
    float *B = aligned_alloc(32, KC * 16 * sizeof(float));
    float *C_ref = aligned_alloc(32, 8 * 16 * sizeof(float));
    float *C_test = aligned_alloc(32, 8 * 16 * sizeof(float));
    
    for (size_t i = 0; i < 8 * KC; i++) A[i] = randf();
    for (size_t i = 0; i < KC * 16; i++) B[i] = randf();
    
    for (size_t i = 0; i < 8 * 16; i++) {
        float val = randf();
        C_ref[i] = val;
        C_test[i] = val;
    }
    
    gemm_8x16_ref_add(A, B, C_ref, 16, KC);
    gemm_kernel_8x16_add_asm(A, B, C_test, 16, KC);
    
    float tol = KC * 1e-5f;
    check_result(C_ref, C_test, 8, 16, 16, tol);
    
    free(A); free(B); free(C_ref); free(C_test);
}

static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void benchmark_kernel(size_t KC, int iterations)
{
    printf("\nBenchmark KC=%zu, %d iterations:\n", KC, iterations);
    
    float *A = aligned_alloc(32, 8 * KC * sizeof(float));
    float *B = aligned_alloc(32, KC * 16 * sizeof(float));
    float *C = aligned_alloc(32, 8 * 16 * sizeof(float));
    
    for (size_t i = 0; i < 8 * KC; i++) A[i] = randf();
    for (size_t i = 0; i < KC * 16; i++) B[i] = randf();
    memset(C, 0, 8 * 16 * sizeof(float));
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        gemm_kernel_8x16_store_asm(A, B, C, 16, KC);
    }
    
    // Benchmark STORE
    double t0 = get_time();
    for (int i = 0; i < iterations; i++) {
        gemm_kernel_8x16_store_asm(A, B, C, 16, KC);
    }
    double t1 = get_time();
    
    double store_time = (t1 - t0) / iterations;
    double store_flops = 2.0 * 8 * 16 * KC;  // 2*M*N*K FLOPs per GEMM
    double store_gflops = store_flops / store_time / 1e9;
    
    printf("  STORE: %.3f ns/call, %.2f GFLOPS\n", store_time * 1e9, store_gflops);
    
    // Benchmark ADD
    t0 = get_time();
    for (int i = 0; i < iterations; i++) {
        gemm_kernel_8x16_add_asm(A, B, C, 16, KC);
    }
    t1 = get_time();
    
    double add_time = (t1 - t0) / iterations;
    double add_gflops = store_flops / add_time / 1e9;
    
    printf("  ADD:   %.3f ns/call, %.2f GFLOPS\n", add_time * 1e9, add_gflops);
    
    // Theoretical peak (2 FMA units * 8 floats/FMA * freq)
    // 14900KF P-core ~5.5 GHz, 2 FMA units, 8 floats = 176 GFLOPS/core theoretical
    printf("  (Theoretical peak ~176 GFLOPS/core on 14900KF P-core)\n");
    
    free(A); free(B); free(C);
}

static void test_noncontiguous_ldc(void)
{
    printf("\nTesting non-contiguous ldc (ldc=32, cols=16):\n");
    
    size_t KC = 64;
    size_t ldc = 32;  // Larger than 16
    
    float *A = aligned_alloc(32, 8 * KC * sizeof(float));
    float *B = aligned_alloc(32, KC * 16 * sizeof(float));
    float *C_ref = aligned_alloc(32, 8 * ldc * sizeof(float));
    float *C_test = aligned_alloc(32, 8 * ldc * sizeof(float));
    
    for (size_t i = 0; i < 8 * KC; i++) A[i] = randf();
    for (size_t i = 0; i < KC * 16; i++) B[i] = randf();
    
    // Fill with pattern to detect overwrites
    for (size_t i = 0; i < 8 * ldc; i++) {
        C_ref[i] = -777.0f;
        C_test[i] = -777.0f;
    }
    
    gemm_8x16_ref_store(A, B, C_ref, ldc, KC);
    gemm_kernel_8x16_store_asm(A, B, C_test, ldc, KC);
    
    // Check that columns 16-31 weren't touched
    int overwrites = 0;
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 16; j < ldc; j++) {
            if (C_test[i * ldc + j] != -777.0f) {
                printf("  OVERWRITE at [%zu,%zu]: %.6f\n", i, j, C_test[i * ldc + j]);
                overwrites++;
            }
        }
    }
    
    if (overwrites == 0) {
        printf("  No buffer overwrites detected.\n");
    }
    
    float tol = KC * 1e-5f;
    check_result(C_ref, C_test, 8, 16, ldc, tol);
    
    free(A); free(B); free(C_ref); free(C_test);
}

int main(int argc, char **argv)
{
    srand(42);  // Deterministic for reproducibility
    
    printf("=== GEMM 8x16 Kernel Tests ===\n\n");
    
    // Correctness tests
    printf("--- Correctness Tests ---\n");
    test_store_kernel(1);
    test_store_kernel(4);
    test_store_kernel(16);
    test_store_kernel(64);
    test_store_kernel(256);
    
    printf("\n");
    test_add_kernel(1);
    test_add_kernel(4);
    test_add_kernel(16);
    test_add_kernel(64);
    test_add_kernel(256);
    
    // Edge case: KC=0
    printf("\nTesting KC=0 (edge case):\n");
    test_store_kernel(0);
    test_add_kernel(0);
    
    // Non-contiguous ldc
    test_noncontiguous_ldc();
    
    // Performance benchmarks
    printf("\n--- Performance Benchmarks ---\n");
    benchmark_kernel(64, 1000000);
    benchmark_kernel(128, 500000);
    benchmark_kernel(256, 250000);
    benchmark_kernel(512, 100000);
    
    printf("\nDone.\n");
    return 0;
}