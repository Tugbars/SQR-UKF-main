/**
 * @file test_gemm_large.c (contains test_kernels_individual)
 * @brief Unit tests for individual GEMM kernels
 */

#include "test_common.h"
#include "gemm_kernels_avx2.h"
#include "gemm_planning.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//==============================================================================
// MASK GENERATION (copied to avoid linking dependencies)
//==============================================================================

/**
 * @brief Build AVX2 mask for partial vector (0-8 lanes)
 */
inline void gemm_test_build_mask_avx2(size_t n, __m256i *out)
{
    if (n > 8)
        n = 8;

#if defined(__AVX2__)
    static const union
    {
        __m256i v;
        int32_t i[8];
    } lut[9] __attribute__((aligned(32))) = {
        {.i = {0, 0, 0, 0, 0, 0, 0, 0}},        // 0
        {.i = {-1, 0, 0, 0, 0, 0, 0, 0}},       // 1
        {.i = {-1, -1, 0, 0, 0, 0, 0, 0}},      // 2
        {.i = {-1, -1, -1, 0, 0, 0, 0, 0}},     // 3
        {.i = {-1, -1, -1, -1, 0, 0, 0, 0}},    // 4
        {.i = {-1, -1, -1, -1, -1, 0, 0, 0}},   // 5
        {.i = {-1, -1, -1, -1, -1, -1, 0, 0}},  // 6
        {.i = {-1, -1, -1, -1, -1, -1, -1, 0}}, // 7
        {.i = {-1, -1, -1, -1, -1, -1, -1, -1}} // 8
    };
    memcpy(out, &lut[n].v, sizeof(__m256i));
#else
    int32_t tmp[8];
    for (size_t k = 0; k < 8; ++k)
    {
        tmp[k] = (k < n) ? -1 : 0;
    }
    memcpy(out, tmp, sizeof(__m256i));
#endif
}

/**
 * @brief Build mask pair for 16-wide panels
 */
inline void gemm_test_build_mask_pair16(size_t w, __m256i *lo, __m256i *hi)
{
    if (w >= 16)
    {
        gemm_test_build_mask_avx2(8, lo);
        gemm_test_build_mask_avx2(8, hi);
    }
    else if (w > 8)
    {
        gemm_test_build_mask_avx2(8, lo);
        gemm_test_build_mask_avx2(w - 8, hi);
    }
    else
    {
        gemm_test_build_mask_avx2(w, lo);
        gemm_test_build_mask_avx2(0, hi);
    }
}

//==============================================================================
// SAFE STORE MACRO (Reference Implementation)
//==============================================================================

/**
 * @brief Safe bounds-checked store for 8×16 tiles
 *
 * This macro ensures NO out-of-bounds writes by:
 * 1. Only storing rows r < m
 * 2. Only storing high 8 columns if n > 8
 * 3. Using masks for partial columns
 *
 * This is the CORRECT way to write microkernel output!
 */
#define GEMM_STORE_TILE_8x16(C, ldc, acc_lo, acc_hi, m, n, mask_lo, mask_hi)            \
    do                                                                                  \
    {                                                                                   \
        for (size_t rr = 0; rr < 8; rr++)                                               \
        {                                                                               \
            if (rr < (m))                                                               \
            {                                                                           \
                /* store columns 0–7 */                                                 \
                _mm256_maskstore_ps((C) + rr * (ldc) + 0, (mask_lo), (acc_lo)[rr]);     \
                /* store columns 8–15 only if needed */                                 \
                if ((n) > 8)                                                            \
                    _mm256_maskstore_ps((C) + rr * (ldc) + 8, (mask_hi), (acc_hi)[rr]); \
            }                                                                           \
        }                                                                               \
    } while (0)

//==============================================================================
// TEST INFRASTRUCTURE
//==============================================================================

static void print_matrix(const char *name, const float *m, size_t rows, size_t cols, size_t ldc)
{
    printf("%s:\n", name);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            printf("%8.3f ", m[i * ldc + j]);
        }
        printf("\n");
    }
    printf("\n");
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

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            float t = test[i * ldc + j];
            float r = ref[i * ldc + j];
            float err = fabsf(t - r);

            if (err > max_error)
            {
                max_error = err;
                error_i = i;
                error_j = j;
            }

            if (err > tol)
            {
                errors++;
                if (errors <= 5)
                { // Print first 5 errors
#ifdef _WIN32
                    printf("  ERROR at [%llu,%llu]: test=%.6f, ref=%.6f, diff=%.6f\n",
                           (unsigned long long)i, (unsigned long long)j, t, r, err);
#else
                    printf("  ERROR at [%zu,%zu]: test=%.6f, ref=%.6f, diff=%.6f\n",
                           i, j, t, r, err);
#endif
                }
            }
        }
    }

    if (errors > 0)
    {
#ifdef _WIN32
        printf("  %s FAILED: %d errors, max error %.6f at [%llu,%llu]\n",
               test_name, errors, max_error,
               (unsigned long long)error_i, (unsigned long long)error_j);
#else
        printf("  %s FAILED: %d errors, max error %.6f at [%zu,%zu]\n",
               test_name, errors, max_error, error_i, error_j);
#endif
        return 0;
    }

    printf("  %s PASSED (max error: %.6e)\n", test_name, max_error);
    return 1;
}

//==============================================================================
// SIMPLE REFERENCE KERNELS
//==============================================================================

static void ref_gemm_simple(
    float *C, size_t ldc,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    size_t M, size_t K, size_t N,
    int accumulate)
{
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
            {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            if (accumulate)
            {
                C[i * ldc + j] += sum;
            }
            else
            {
                C[i * ldc + j] = sum;
            }
        }
    }
}

//==============================================================================
// PACKING TEST HELPERS
//==============================================================================

static void pack_A_for_test(float *Ap, const float *A, size_t M, size_t K, size_t mr)
{
    memset(Ap, 0, K * mr * sizeof(float));
    for (size_t k = 0; k < K; k++)
    {
        for (size_t i = 0; i < M && i < mr; i++)
        {
            Ap[k * mr + i] = A[i * K + k];
        }
    }
}

static void pack_B_for_test(float *Bp, const float *B, size_t K, size_t N)
{
    memset(Bp, 0, K * 16 * sizeof(float));
    for (size_t k = 0; k < K; k++)
    {
        for (size_t j = 0; j < N && j < 16; j++)
        {
            Bp[k * 16 + j] = B[k * N + j];
        }
    }
}

//==============================================================================
// TEST 8x8 KERNEL
//==============================================================================

static int test_kernel_8x8(void)
{
    printf("\n=== Testing 8x8 Kernel ===\n");
    fflush(stdout);

    const size_t M = 8, K = 32, N = 8;
    const size_t ldc = N;

    printf("  Allocating buffers...\n");
    fflush(stdout);

    // Use library's aligned allocation
    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    if (!A || !B || !C_test || !C_ref || !Ap || !Bp)
    {
        printf("  ERROR: Allocation failed!\n");
        return 0;
    }

    printf("  Initializing data...\n");
    fflush(stdout);

    // Initialize
    for (size_t i = 0; i < M * K; i++)
        A[i] = (i % 7) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i % 5) * 0.1f;

    printf("  Packing A...\n");
    fflush(stdout);
    pack_A_for_test(Ap, A, M, K, 8);

    printf("  Packing B...\n");
    fflush(stdout);
    pack_B_for_test(Bp, B, K, N);

    // Test STORE variant
    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

    // Correct argument order
    gemm_8x8_panel_avx2fma_store(
        C_test, ldc, // C and ldc
        Ap, 8,       // packed A and stride
        Bp, 16,      // packed B and stride
        K,           // K dimension
        8, 8,        // m, n
        mask         // mask
    );

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x8 STORE");

    // Test ADD variant
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 1.0f;
        C_ref[i] = 1.0f;
    }

    gemm_8x8_panel_avx2fma_add(
        C_test, ldc,
        Ap, 8,
        Bp, 16,
        K,
        8, 8,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);

    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x8 ADD");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST 4x8 KERNEL
//==============================================================================

static int test_kernel_4x8(void)
{
    printf("\n=== Testing 4x8 Kernel ===\n");

    const size_t M = 4, K = 16, N = 8;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    for (size_t i = 0; i < M * K; i++)
        A[i] = (i + 1) * 0.01f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i + 1) * 0.01f;

    pack_A_for_test(Ap, A, M, K, 8);
    pack_B_for_test(Bp, B, K, N);

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

    // Correct argument order for 4x8
    gemm_4x8_panel_avx2fma_store(
    C_test, ldc,
    Ap, 8,
    Bp, 16,
    K,
    4,     // m_block (number of rows to actually write)
    8,     // jb (width)
    mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "4x8 STORE");

    // Test ADD
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 0.5f;
        C_ref[i] = 0.5f;
    }

    gemm_4x8_panel_avx2fma_add(
    C_test, ldc,
    Ap, 8,
    Bp, 16,
    K,
    4,     // m_block (number of rows to actually write)
    8,     // jb (width)
    mask);
    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);

    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "4x8 ADD");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST 1x8 KERNEL
//==============================================================================

static int test_kernel_1x8(void)
{
    printf("\n=== Testing 1x8 Kernel ===\n");

    const size_t M = 1, K = 8, N = 8;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    for (size_t i = 0; i < M * K; i++)
        A[i] = (i + 1) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i + 1) * 0.1f;

    pack_A_for_test(Ap, A, M, K, 8);
    pack_B_for_test(Bp, B, K, N);

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

    // Correct argument order for 1x8 (no ldc for single-row kernel)
    gemm_1x8_panel_avx2fma_store(
        C_test,
        Ap, 8,
        Bp, 16,
        K,
        8, // jb
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    printf("  1x8 result: ");
    for (int i = 0; i < 8; i++)
        printf("%.3f ", C_test[i]);
    printf("\n");
    printf("  Reference:  ");
    for (int i = 0; i < 8; i++)
        printf("%.3f ", C_ref[i]);
    printf("\n");

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "1x8 STORE");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST 16x8 KERNEL
//==============================================================================

static int test_kernel_16x8(void)
{
    printf("\n=== Testing 16x8 Kernel ===\n");

    const size_t M = 16, K = 32, N = 8;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 16 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    for (size_t i = 0; i < M * K; i++)
        A[i] = (i + 1) * 0.01f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i + 1) * 0.01f;

    // Pack with MR=16
    memset(Ap, 0, K * 16 * sizeof(float));
    for (size_t k = 0; k < K; k++)
    {
        for (size_t i = 0; i < M && i < 16; i++)
        {
            Ap[k * 16 + i] = A[i * K + k];
        }
    }

    pack_B_for_test(Bp, B, K, N);

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask = _mm256_set1_epi32(-1);

    // Test STORE variant
    gemm_16x8_panel_avx2fma_store(
        C_test, ldc,
        Ap, 16,
        Bp, 16,
        K,
        16, 8,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    // Note: Relaxed tolerance for 16-row accumulation
    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 5e-5f, "16x8 STORE");

    // Test ADD variant
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 0.5f;
        C_ref[i] = 0.5f;
    }

    gemm_16x8_panel_avx2fma_add(
        C_test, ldc,
        Ap, 16,
        Bp, 16,
        K,
        16, 8,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);

    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 5e-5f, "16x8 ADD");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST 8x16 KERNEL (Dual-mask variant)
//==============================================================================

static int test_kernel_8x16(void)
{
    printf("\n=== Testing 8x16 Kernel ===\n");

    const size_t M = 8, K = 32, N = 16;
    const size_t ldc = N;

    printf("  Allocating buffers...\n");
    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    printf("  Initializing data...\n");
    // Initialize with known pattern
    for (size_t i = 0; i < M * K; i++)
        A[i] = ((i * 3) % 17) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = ((i * 7) % 19) * 0.1f;

    // Initialize output to a sentinel value to detect uninitialized access
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 999.0f;
        C_ref[i] = 0.0f;
    }

#ifdef _WIN32
    printf("  Packing A (M=%llu, K=%llu, MR=8)...\n", (unsigned long long)M, (unsigned long long)K);
#else
    printf("  Packing A (M=%zu, K=%zu, MR=8)...\n", M, K);
#endif
    // Pack with explicit bounds checking
    pack_A_for_test(Ap, A, M, K, 8);

#ifdef _WIN32
    printf("  Packing B (K=%llu, N=%llu, NR=16)...\n", (unsigned long long)K, (unsigned long long)N);
#else
    printf("  Packing B (K=%zu, N=%zu, NR=16)...\n", K, N);
#endif
    // Pack B - ensure full 16-width coverage
    memset(Bp, 0, K * 16 * sizeof(float));
    for (size_t k = 0; k < K; k++)
    {
        for (size_t j = 0; j < N; j++)
        {
            Bp[k * 16 + j] = B[k * N + j];
        }
    }

    // Zero output before STORE test
    memset(C_test, 0, M * N * sizeof(float));

    // Build masks for 16-wide panel
    __m256i mask_lo, mask_hi;
    gemm_test_build_mask_pair16(N, &mask_lo, &mask_hi);

    printf("  Calling 8x16 STORE kernel...\n");
    fflush(stdout); // Ensure we see this before crash

    // Test STORE variant - CRITICAL: pass actual M and N!
    gemm_8x16_panel_avx2fma_store(
        C_test, ldc,
        Ap, 8,
        Bp, 16,
        K,
        M, N, // ← CRITICAL: Actual dimensions, not assumed 8×16
        mask_lo, mask_hi);

    printf("  STORE kernel completed\n");

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x16 STORE");

    printf("  Preparing ADD test...\n");
    // Test ADD variant
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 0.5f;
        C_ref[i] = 0.5f;
    }

    printf("  Calling 8x16 ADD kernel...\n");
    fflush(stdout);

    gemm_8x16_panel_avx2fma_add(
        C_test, ldc,
        Ap, 8,
        Bp, 16,
        K,
        M, N,
        mask_lo, mask_hi);

    printf("  ADD kernel completed\n");

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);

    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x16 ADD");

    printf("  Freeing buffers...\n");
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    printf("  Test complete\n");
    return passed;
}

//==============================================================================
// TEST 16x6 KERNEL
//==============================================================================

static int test_kernel_16x6(void)
{
    printf("\n=== Testing 16x6 Kernel ===\n");

    const size_t M = 16, K = 24, N = 6;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 16 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    for (size_t i = 0; i < M * K; i++)
        A[i] = (i % 11) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i % 13) * 0.1f;

    // Pack with MR=16
    memset(Ap, 0, K * 16 * sizeof(float));
    for (size_t k = 0; k < K; k++)
    {
        for (size_t i = 0; i < M && i < 16; i++)
        {
            Ap[k * 16 + i] = A[i * K + k];
        }
    }

    pack_B_for_test(Bp, B, K, N);

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask;
    gemm_test_build_mask_avx2(N, &mask);

    // Test STORE
    gemm_16x6_panel_avx2fma_store(
        C_test, ldc,
        Ap, 16,
        Bp, 16,
        K,
        16, 6,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 5e-5f, "16x6 STORE");

    // Test ADD
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 0.5f;
        C_ref[i] = 0.5f;
    }

    gemm_16x6_panel_avx2fma_add(
        C_test, ldc,
        Ap, 16,
        Bp, 16,
        K,
        16, 6,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);

    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 5e-5f, "16x6 ADD");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST 8x6 KERNEL
//==============================================================================

static int test_kernel_8x6(void)
{
    printf("\n=== Testing 8x6 Kernel ===\n");

    const size_t M = 8, K = 16, N = 6;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    for (size_t i = 0; i < M * K; i++)
        A[i] = (i % 7) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (i % 9) * 0.1f;

    pack_A_for_test(Ap, A, M, K, 8);
    pack_B_for_test(Bp, B, K, N);

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    __m256i mask;
    gemm_test_build_mask_avx2(N, &mask);

    // Test STORE
    gemm_8x6_panel_avx2fma_store(
        C_test, ldc,
        Ap, 8,
        Bp, 16,
        K,
        8, 6,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x6 STORE");

    // Test ADD
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = 0.5f;
        C_ref[i] = 0.5f;
    }

    gemm_8x6_panel_avx2fma_add(
        C_test, ldc,
        Ap, 8,
        Bp, 16,
        K,
        8, 6,
        mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 1);

    passed &= compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "8x6 ADD");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// TEST EDGE CASES
//==============================================================================

static int test_edge_cases(void)
{
    printf("\n=== Testing Edge Cases ===\n");

    int passed = 1;

    // Test 1: K=0 (should produce zeros)
    printf("  Testing K=0...\n");
    {
        const size_t M = 8, K = 0, N = 8, ldc = 8;
        float *C = gemm_aligned_alloc(32, M * N * sizeof(float));
        float *Ap = gemm_aligned_alloc(32, 8 * 8 * sizeof(float));
        float *Bp = gemm_aligned_alloc(32, 16 * sizeof(float));

        for (size_t i = 0; i < M * N; i++)
            C[i] = 999.0f;

        __m256i mask = _mm256_set1_epi32(-1);
        gemm_8x8_panel_avx2fma_store(C, ldc, Ap, 8, Bp, 16, K, 8, 8, mask);

        int all_zero = 1;
        for (size_t i = 0; i < M * N; i++)
        {
            if (C[i] != 0.0f)
            {
                all_zero = 0;
                break;
            }
        }

        if (all_zero)
        {
            printf("    K=0 PASSED\n");
        }
        else
        {
            printf("    K=0 FAILED (expected all zeros)\n");
            passed = 0;
        }

        gemm_aligned_free(C);
        gemm_aligned_free(Ap);
        gemm_aligned_free(Bp);
    }

    // Test 2: Partial tiles
    printf("  Testing partial tiles (m=5, n=7)...\n");
    {
        const size_t M = 5, K = 8, N = 7, ldc = 7;

        float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
        float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
        float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
        float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));
        float *Ap = gemm_aligned_alloc(32, K * 8 * sizeof(float));
        float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

        for (size_t i = 0; i < M * K; i++)
            A[i] = 0.1f;
        for (size_t i = 0; i < K * N; i++)
            B[i] = 0.1f;

        pack_A_for_test(Ap, A, M, K, 8);
        pack_B_for_test(Bp, B, K, N);

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        __m256i mask;
        gemm_test_build_mask_avx2(N, &mask);

        gemm_8x8_panel_avx2fma_store(C_test, ldc, Ap, 8, Bp, 16, K, M, N, mask);
        ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

        if (compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-5f, "Partial 5x7"))
        {
            printf("    Partial tiles PASSED\n");
        }
        else
        {
            printf("    Partial tiles FAILED\n");
            passed = 0;
        }

        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
        gemm_aligned_free(Ap);
        gemm_aligned_free(Bp);
    }

    return passed;
}

//==============================================================================
// TEST KERNEL COMBINATIONS
//==============================================================================

static int test_kernel_combination(void)
{
    printf("\n=== Testing Kernel Combinations ===\n");

    const size_t M = 12, K = 16, N = 8;
    const size_t ldc = N;

    float *A = gemm_aligned_alloc(32, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(32, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(32, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(32, M * N * sizeof(float));

    srand(42);
    for (size_t i = 0; i < M * K; i++)
        A[i] = (float)(rand() % 10) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (float)(rand() % 10) * 0.1f;

    memset(C_test, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    float *Ap8 = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Ap4 = gemm_aligned_alloc(32, K * 8 * sizeof(float));
    float *Bp = gemm_aligned_alloc(32, K * 16 * sizeof(float));

    pack_A_for_test(Ap8, A, 8, K, 8);
    pack_A_for_test(Ap4, A + 8 * K, 4, K, 8);
    pack_B_for_test(Bp, B, K, N);

    __m256i mask = _mm256_set1_epi32(-1);

    gemm_4x8_panel_avx2fma_store(
    C_test + 8 * ldc, ldc,
    Ap4, 8,
    Bp, 16,
    K,
    4,     // m_block (4 rows for this tile)
    8,     // jb (width)
    mask);

    gemm_4x8_panel_avx2fma_store(
    C_test + 8 * ldc, ldc,
    Ap4, 8,
    Bp, 16,
    K,
    4,     // m_block - processing 4 rows
    8,     // jb - width is 8
    mask);

    ref_gemm_simple(C_ref, ldc, A, K, B, N, M, K, N, 0);

    int passed = compare_matrices_verbose(C_test, C_ref, M, N, ldc, 1e-4f, "12x8 combination");

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);
    gemm_aligned_free(Ap8);
    gemm_aligned_free(Ap4);
    gemm_aligned_free(Bp);

    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_gemm_kernel_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("         GEMM KERNEL UNIT TESTS\n");
    printf("=================================================\n");

    results->total = 0;
    results->passed = 0;
    results->failed = 0;

    // Test individual kernels
    printf("\n--- Testing Individual Kernels ---\n");

    results->total++;
    if (test_kernel_8x8())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    results->total++;
    if (test_kernel_4x8())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    results->total++;
    if (test_kernel_1x8())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    results->total++;
    if (test_kernel_16x8())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    results->total++;
    if (test_kernel_8x16())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    results->total++;
    if (test_kernel_16x6())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    results->total++;
    if (test_kernel_8x6())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    // Test edge cases
    printf("\n--- Testing Edge Cases ---\n");

    results->total++;
    if (test_edge_cases())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    // Test combinations
    printf("\n--- Testing Kernel Combinations ---\n");

    results->total++;
    if (test_kernel_combination())
    {
        results->passed++;
    }
    else
    {
        results->failed++;
    }

    printf("\n=================================================\n");
    printf("Kernel Tests: %d/%d passed\n", results->passed, results->total);

    if (results->passed == results->total)
    {
        printf("✓ All kernel tests PASSED!\n");
    }
    else
    {
        printf("✗ %d kernel tests FAILED\n", results->failed);
        printf("Debug the failing kernels before testing the full GEMM\n");
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
    return run_gemm_kernel_tests(&results);
}
#endif

