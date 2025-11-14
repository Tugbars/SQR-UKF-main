/**
 * @file test_gemm_validated.c
 * @brief Enhanced GEMM kernel tests with full validation integration
 * 
 * Compile with:
 *   Debug:   -DGEMM_VALIDATION_LEVEL=2
 *   Release: -DGEMM_VALIDATION_LEVEL=0 (or -DNDEBUG)
 */

#include "gemm_validation.h"  // ← Unified validation header
#include "test_common.h"
#include "gemm_kernels_avx2.h"
#include "gemm_planning.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//==============================================================================
// VALIDATION-AWARE ALLOCATION WRAPPER
//==============================================================================

static void* test_alloc_validated(size_t alignment, size_t size, const char *name)
{
    void *ptr;
    
#if GEMM_VALIDATION_LEVEL >= 2
    GEMM_ALLOC(ptr, size);
    
    if (((uintptr_t)ptr & (alignment - 1)) != 0) {
        printf("    WARNING: %s not " FMT_ZU "-byte aligned: %p\n", 
               name, CAST_ZU(alignment), ptr);
        GEMM_FREE(ptr);
        ptr = gemm_aligned_alloc(alignment, size);
        if (!ptr) {
            printf("    ERROR: Re-allocation failed for %s\n", name);
            return NULL;
        }
    }
#else
    ptr = gemm_aligned_alloc(alignment, size);
#endif
    
    if (!ptr) {
        printf("    ERROR: Allocation failed for %s (" FMT_ZU " bytes)\n", 
               name, CAST_ZU(size));
        return NULL;
    }
    
    return ptr;
}

static void test_free_validated(void *ptr)
{
#if GEMM_VALIDATION_LEVEL >= 2
    if (((uintptr_t)ptr & 31) == 0) {
        gemm_aligned_free(ptr);
    } else {
        GEMM_FREE(ptr);
    }
#else
    gemm_aligned_free(ptr);
#endif
}

//==============================================================================
// VALIDATED PACKING FUNCTIONS
//==============================================================================

static void pack_A_validated(float *Ap, const float *A, size_t M, size_t K, size_t mr)
{
    PACK_VALIDATE_PTR(Ap);
    PACK_VALIDATE_ALIGNED(Ap);
    PACK_CHECK_MR(mr);
    
    size_t buffer_size = K * mr * sizeof(float);
    memset(Ap, 0, buffer_size);
    
    for (size_t k = 0; k < K; k++)
    {
        for (size_t i = 0; i < M && i < mr; i++)
        {
            size_t dst_idx = k * mr + i;
            size_t src_idx = i * K + k;
            
#if GEMM_VALIDATION_LEVEL >= 2
            if (dst_idx >= K * mr) {
                VALIDATION_ERROR("pack_A: dst write OOB [" FMT_ZU "] >= " FMT_ZU,
                                 CAST_ZU(dst_idx), CAST_ZU(K * mr));
            }
            if (src_idx >= M * K) {
                VALIDATION_ERROR("pack_A: src read OOB [" FMT_ZU "] >= " FMT_ZU,
                                 CAST_ZU(src_idx), CAST_ZU(M * K));
            }
#endif
            
            Ap[dst_idx] = A[src_idx];
        }
    }
    
#if GEMM_VALIDATION_LEVEL >= 2
    for (size_t i = 0; i < K * mr; i++) {
        if (isnan(Ap[i]) || isinf(Ap[i])) {
            VALIDATION_ERROR("pack_A: NaN/Inf at [" FMT_ZU "]", CAST_ZU(i));
        }
    }
#endif
}

static void pack_B_validated(float *Bp, const float *B, size_t K, size_t N)
{
    PACK_VALIDATE_PTR(Bp);
    PACK_VALIDATE_ALIGNED(Bp);
    
    const size_t NR = 16;
    size_t buffer_size = K * NR * sizeof(float);
    memset(Bp, 0, buffer_size);
    
    for (size_t k = 0; k < K; k++)
    {
        for (size_t j = 0; j < N && j < NR; j++)
        {
            size_t dst_idx = k * NR + j;
            size_t src_idx = k * N + j;
            
#if GEMM_VALIDATION_LEVEL >= 2
            if (dst_idx >= K * NR) {
                VALIDATION_ERROR("pack_B: dst write OOB [" FMT_ZU "] >= " FMT_ZU,
                                 CAST_ZU(dst_idx), CAST_ZU(K * NR));
            }
            if (src_idx >= K * N) {
                VALIDATION_ERROR("pack_B: src read OOB [" FMT_ZU "] >= " FMT_ZU,
                                 CAST_ZU(src_idx), CAST_ZU(K * N));
            }
#endif
            
            Bp[dst_idx] = B[src_idx];
        }
    }
    
#if GEMM_VALIDATION_LEVEL >= 2
    for (size_t i = 0; i < K * NR; i++) {
        if (isnan(Bp[i]) || isinf(Bp[i])) {
            VALIDATION_ERROR("pack_B: NaN/Inf at [" FMT_ZU "]", CAST_ZU(i));
        }
    }
#endif
}

//==============================================================================
// SENTINEL VALUE CHECKING
//==============================================================================

#define SENTINEL_VALUE 999.0f

static void fill_sentinel(float *ptr, size_t count)
{
#if GEMM_VALIDATION_LEVEL >= 2
    for (size_t i = 0; i < count; i++) {
        ptr[i] = SENTINEL_VALUE;
    }
#else
    memset(ptr, 0, count * sizeof(float));
#endif
}

static int check_no_sentinels(const float *ptr, size_t count, const char *name)
{
#if GEMM_VALIDATION_LEVEL >= 2
    for (size_t i = 0; i < count; i++) {
        if (ptr[i] == SENTINEL_VALUE) {
            printf("  ERROR: %s has uninitialized output at [" FMT_ZU "] (sentinel still present)\n",
                   name, CAST_ZU(i));
            return 0;
        }
    }
    printf("  %s: Output fully initialized ✓\n", name);
#endif
    return 1;
}

//==============================================================================
// MASK GENERATION
//==============================================================================

inline void gemm_test_build_mask_avx2(size_t n, __m256i *out)
{
    if (n > 8) n = 8;
#if defined(__AVX2__)
    static const union {
        __m256i v;
        int32_t i[8];
    } lut[9] __attribute__((aligned(32))) = {
        {.i = {0, 0, 0, 0, 0, 0, 0, 0}},
        {.i = {-1, 0, 0, 0, 0, 0, 0, 0}},
        {.i = {-1, -1, 0, 0, 0, 0, 0, 0}},
        {.i = {-1, -1, -1, 0, 0, 0, 0, 0}},
        {.i = {-1, -1, -1, -1, 0, 0, 0, 0}},
        {.i = {-1, -1, -1, -1, -1, 0, 0, 0}},
        {.i = {-1, -1, -1, -1, -1, -1, 0, 0}},
        {.i = {-1, -1, -1, -1, -1, -1, -1, 0}},
        {.i = {-1, -1, -1, -1, -1, -1, -1, -1}}
    };
    memcpy(out, &lut[n].v, sizeof(__m256i));
#else
    int32_t tmp[8];
    for (size_t k = 0; k < 8; ++k) {
        tmp[k] = (k < n) ? -1 : 0;
    }
    memcpy(out, tmp, sizeof(__m256i));
#endif
}

inline void gemm_test_build_mask_pair16(size_t w, __m256i *lo, __m256i *hi)
{
    if (w >= 16) {
        gemm_test_build_mask_avx2(8, lo);
        gemm_test_build_mask_avx2(8, hi);
    } else if (w > 8) {
        gemm_test_build_mask_avx2(8, lo);
        gemm_test_build_mask_avx2(w - 8, hi);
    } else {
        gemm_test_build_mask_avx2(w, lo);
        gemm_test_build_mask_avx2(0, hi);
    }
}

//==============================================================================
// REFERENCE IMPLEMENTATION & COMPARISON
//==============================================================================

static void ref_gemm_simple(
    float *C, size_t ldc,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    size_t M, size_t K, size_t N,
    int accumulate)
{
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            if (accumulate) {
                C[i * ldc + j] += sum;
            } else {
                C[i * ldc + j] = sum;
            }
        }
    }
}

static int compare_matrices_verbose(
    const float *test,
    const float *ref,
    size_t rows,
    size_t cols,
    size_t ldc,
    float tol,
    const char *test_name)
{
    int errors = 0;
    float max_error = 0.0f;
    size_t error_i = 0, error_j = 0;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float t = test[i * ldc + j];
            float r = ref[i * ldc + j];
            float err = fabsf(t - r);

            if (err > max_error) {
                max_error = err;
                error_i = i;
                error_j = j;
            }

            if (err > tol) {
                errors++;
                if (errors <= 5) {
                    printf("  ERROR at [" FMT_ZU "," FMT_ZU "]: test=%.6f, ref=%.6f, diff=%.6f\n",
                           CAST_ZU(i), CAST_ZU(j), t, r, err);
                }
            }
        }
    }

    if (errors > 0) {
        printf("  %s FAILED: %d errors, max error %.6f at [" FMT_ZU "," FMT_ZU "]\n",
               test_name, errors, max_error,
               CAST_ZU(error_i), CAST_ZU(error_j));
        return 0;
    }

    printf("  %s PASSED (max error: %.6e)\n", test_name, max_error);
    return 1;
}

//==============================================================================
// VALIDATED TEST: 8x8 KERNEL
//==============================================================================

static int test_kernel_8x8_validated(void)
{
    printf("\n=== Testing 8x8 Kernel (VALIDATED) ===\n");
    fflush(stdout);

    const size_t M = 8, K = 32, N = 8;
    const size_t ldc = N;
    const size_t MR = 8;

    printf("  Allocating buffers...\n");
    
    float *A = test_alloc_validated(32, M * K * sizeof(float), "A");
    float *B = test_alloc_validated(32, K * N * sizeof(float), "B");
    float *C_test = test_alloc_validated(32, M * N * sizeof(float), "C_test");
    float *C_ref = test_alloc_validated(32, M * N * sizeof(float), "C_ref");
    float *Ap = test_alloc_validated(32, K * MR * sizeof(float), "Ap");
    float *Bp = test_alloc_validated(32, K * 16 * sizeof(float), "Bp");

    if (!A || !B || !C_test || !C_ref || !Ap || !Bp) {
        printf("  ERROR: Allocation failed!\n");
        return 0;
    }

    printf("  Initializing data...\n");
    for (size_t i = 0; i < M * K; i++)
        A[i] = (i % 7) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i % 5) * 0.1f;

    printf("  Packing A (MR=" FMT_ZU ")...\n", CAST_ZU(MR));
    pack_A_validated(Ap, A, M, K, MR);
    
#if GEMM_VALIDATION_LEVEL >= 2
    printf("  Validating Ap buffer...\n");
    GEMM_VALIDATE(Ap, K * MR * sizeof(float));
#endif

    printf("  Packing B (NR=16)...\n");
    pack_B_validated(Bp, B, K, N);
    
#if GEMM_VALIDATION_LEVEL >= 2
    printf("  Validating Bp buffer...\n");
    GEMM_VALIDATE(Bp, K * 16 * sizeof(float));
#endif

    printf("  Testing STORE variant...\n");
    fill_sentinel(C_test, M * N);
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

#if GEMM_VALIDATION_LEVEL >= 2
    printf("  Pre-kernel validation...\n");
    VALIDATE_KERNEL_PARAMS(C_test, ldc, Ap, MR, Bp, 16, K, M, N, MR);
#endif

    printf("  Calling gemm_8x8_panel_avx2fma_store...\n");
    fflush(stdout);
    
    gemm_8x8_panel_avx2fma_store(
        C_test, ldc,
        Ap, MR,
        Bp, 16,
        K,
        M, N,
        mask);

    printf("  Kernel completed\n");

#if GEMM_VALIDATION_LEVEL >= 2
    printf("  Post-kernel validation...\n");
    VALIDATE_KERNEL_POST(Ap, Bp, K, MR);
    
    if (!check_no_sentinels(C_test, M * N, "8x8 STORE")) {
        return 0;
    }
#endif

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);
    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x8 STORE");

    printf("  Testing ADD variant...\n");
    for (size_t i = 0; i < M * N; i++) {
        C_test[i] = 1.0f;
        C_ref[i] = 1.0f;
    }

#if GEMM_VALIDATION_LEVEL >= 2
    VALIDATE_KERNEL_PARAMS(C_test, ldc, Ap, MR, Bp, 16, K, M, N, MR);
#endif

    gemm_8x8_panel_avx2fma_add(
        C_test, ldc,
        Ap, MR,
        Bp, 16,
        K,
        M, N,
        mask);

#if GEMM_VALIDATION_LEVEL >= 2
    VALIDATE_KERNEL_POST(Ap, Bp, K, MR);
#endif

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);
    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x8 ADD");

    test_free_validated(A);
    test_free_validated(B);
    test_free_validated(C_test);
    test_free_validated(C_ref);
    test_free_validated(Ap);
    test_free_validated(Bp);

    return passed;
}

//==============================================================================
// VALIDATED TEST: 16x8 KERNEL
//==============================================================================

static int test_kernel_16x8_validated(void)
{
    printf("\n=== Testing 16x8 Kernel (VALIDATED) ===\n");

    const size_t M = 16, K = 32, N = 8;
    const size_t ldc = N;
    const size_t MR = 16;

    float *A = test_alloc_validated(32, M * K * sizeof(float), "A");
    float *B = test_alloc_validated(32, K * N * sizeof(float), "B");
    float *C_test = test_alloc_validated(32, M * N * sizeof(float), "C_test");
    float *C_ref = test_alloc_validated(32, M * N * sizeof(float), "C_ref");
    float *Ap = test_alloc_validated(32, K * MR * sizeof(float), "Ap");
    float *Bp = test_alloc_validated(32, K * 16 * sizeof(float), "Bp");

    for (size_t i = 0; i < M * K; i++)
        A[i] = (i + 1) * 0.01f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i + 1) * 0.01f;

    pack_A_validated(Ap, A, M, K, MR);
    pack_B_validated(Bp, B, K, N);

    fill_sentinel(C_test, M * N);
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

#if GEMM_VALIDATION_LEVEL >= 2
    VALIDATE_KERNEL_PARAMS(C_test, ldc, Ap, MR, Bp, 16, K, M, N, MR);
#endif

    gemm_16x8_panel_avx2fma_store(
        C_test, ldc,
        Ap, MR,
        Bp, 16,
        K,
        M, N,
        mask);

#if GEMM_VALIDATION_LEVEL >= 2
    VALIDATE_KERNEL_POST(Ap, Bp, K, MR);
    check_no_sentinels(C_test, M * N, "16x8 STORE");
#endif

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);
    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 5e-5f, "16x8 STORE");

    for (size_t i = 0; i < M * N; i++) {
        C_test[i] = 0.5f;
        C_ref[i] = 0.5f;
    }

    gemm_16x8_panel_avx2fma_add(C_test, ldc, Ap, MR, Bp, 16, K, M, N, mask);
    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);
    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 5e-5f, "16x8 ADD");

    test_free_validated(A);
    test_free_validated(B);
    test_free_validated(C_test);
    test_free_validated(C_ref);
    test_free_validated(Ap);
    test_free_validated(Bp);

    return passed;
}

//==============================================================================
// TEST RUNNER (callable from test_main.c)
//==============================================================================

int run_gemm_validated_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("    VALIDATED GEMM KERNEL UNIT TESTS\n");
#if GEMM_VALIDATION_LEVEL >= 2
    printf("    Validation Level: %d (FULL)\n", GEMM_VALIDATION_LEVEL);
#else
    printf("    Validation Level: %d (MINIMAL)\n", GEMM_VALIDATION_LEVEL);
#endif
    printf("=================================================\n");

    int total = 0;
    int passed = 0;

    total++;
    if (test_kernel_8x8_validated()) {
        passed++;
    }

    total++;
    if (test_kernel_16x8_validated()) {
        passed++;
    }

    // Populate results structure
    if (results) {
        results->total = total;
        results->passed = passed;
        results->failed = total - passed;
    }

    printf("\n=================================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    
    if (passed == total) {
        printf("✓ All validated tests PASSED!\n");
    } else {
        printf("✗ %d validated test(s) FAILED\n", total - passed);
    }
    printf("=================================================\n");

    return (passed == total) ? 1 : 0;
}

//==============================================================================
// STANDALONE MAIN (only when compiled as standalone executable)
//==============================================================================

#ifdef STANDALONE
int main(void)
{
    test_results_t results = {0};
    int success = run_gemm_validated_tests(&results);
    return success ? 0 : 1;
}
#endif