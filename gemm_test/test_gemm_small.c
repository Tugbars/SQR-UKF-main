/**
 * @file test_gemm_small.c
 * @brief Tier 1 Small GEMM Kernel Test Suite
 *
 * Tests:
 * - 4×4, 6×6, 8×8 kernel correctness
 * - Alpha/beta combination coverage (critical bug source)
 * - LDC handling (your fix)
 * - Dispatcher routing logic
 *
 * Compile:
 *   gcc -o test_small test_gemm_small.c gemm_small.c gemm_simd_ops.c \
 *       -I. -O2 -march=native -mavx2 -mfma -Wall -Wextra -lm
 *
 * @author TUGBARS
 * @date 2025
 */

#include "gemm_small.h"
#include "test_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <inttypes.h>

// Portable size_t format for old MinGW
#ifdef _WIN32
    #define SIZE_FMT "%llu"
    #define SIZE_CAST(x) ((unsigned long long)(x))
#else
    #define SIZE_FMT "%zu"
    #define SIZE_CAST(x) (x)
#endif


//==============================================================================
// REFERENCE IMPLEMENTATION - Naive Triple Loop
//==============================================================================

/**
 * @brief Dead-simple reference GEMM (obviously correct)
 *
 * C = alpha*A*B + beta*C
 *
 * This is the gold standard - optimized kernels must match this.
 */
static void gemm_naive(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc,
    float alpha, float beta)
{
    // Step 1: Apply beta to C
    if (beta == 0.0f)
    {
        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                C[i * ldc + j] = 0.0f;
            }
        }
    }
    else if (beta != 1.0f)
    {
        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // Step 2: Accumulate alpha*A*B
    for (size_t i = 0; i < M; i++)
    {
        for (size_t k = 0; k < K; k++)
        {
            float a_ik = alpha * A[i * K + k];
            for (size_t j = 0; j < N; j++)
            {
                C[i * ldc + j] += a_ik * B[k * N + j];
            }
        }
    }
}

//==============================================================================
// COMPARISON UTILITIES
//==============================================================================

/**
 * @brief Compare two matrices with relative tolerance
 *
 * Uses relative error for normal values, absolute for tiny values.
 */
static int matrices_equal(
    const float *A, const float *B,
    size_t M, size_t N, size_t ldc,
    float rel_tol)
{
    int all_match = 1;
    
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float a = A[i * ldc + j];
            float b = B[i * ldc + j];
            float diff = fabsf(a - b);

            // Use absolute tolerance for tiny values
            if (fabsf(a) < 1e-6f && fabsf(b) < 1e-6f)
            {
                if (diff > rel_tol)
                {

                    all_match = 0;
                }
            }
            else
            {
                // Use relative tolerance
                float max_val = fmaxf(fabsf(a), fabsf(b));
                float rel_err = diff / max_val;
                
                if (rel_err > rel_tol)
                {
                    all_match = 0;
                }
            }
        }
    }
    
    return all_match;
}

//==============================================================================
// MATRIX INITIALIZATION - Structured Data
//==============================================================================

static void matrix_set_identity(float *M, size_t N, size_t ld)
{
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            M[i * ld + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

static void matrix_set_zeros(float *M, size_t rows, size_t cols, size_t ld)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            M[i * ld + j] = 0.0f;
        }
    }
}

static void matrix_set_ones(float *M, size_t rows, size_t cols, size_t ld)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            M[i * ld + j] = 1.0f;
        }
    }
}

static void matrix_set_value(float *M, size_t rows, size_t cols, size_t ld, float val)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            M[i * ld + j] = val;
        }
    }
}

static void matrix_set_sequential(float *M, size_t rows, size_t cols, size_t ld)
{
    float val = 1.0f;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            M[i * ld + j] = val++;
        }
    }
}

//==============================================================================
// ALPHA/BETA TEST COMBINATIONS - Critical Bug Source
//==============================================================================

typedef struct
{
    float alpha;
    float beta;
    const char *description;
} alpha_beta_case_t;

static const alpha_beta_case_t alpha_beta_cases[] = {
    {1.0f, 0.0f, "alpha=1, beta=0 (C = A*B)"},
    {1.0f, 1.0f, "alpha=1, beta=1 (C += A*B)"},
    {2.0f, 0.0f, "alpha=2, beta=0 (C = 2*A*B)"},
    {1.0f, 0.5f, "alpha=1, beta=0.5 (C = A*B + 0.5*C)"},
    {2.0f, 0.5f, "alpha=2, beta=0.5 (C = 2*A*B + 0.5*C)"},
    {0.0f, 1.0f, "alpha=0, beta=1 (C = C, no-op)"},
    {0.0f, 0.0f, "alpha=0, beta=0 (C = 0)"},
    {-1.0f, 1.0f, "alpha=-1, beta=1 (C = C - A*B)"},
    {1.0f, -1.0f, "alpha=1, beta=-1 (C = A*B - C)"},
    {-1.0f, -1.0f, "alpha=-1, beta=-1 (negative both)"},
};

static const size_t n_alpha_beta_cases = sizeof(alpha_beta_cases) / sizeof(alpha_beta_cases[0]);

//==============================================================================
// TEST: 4×4 Kernel
//==============================================================================

static int test_4x4_identity(void)
{
    printf("  Testing: 4×4 identity matrix (I*I = I)\n");

    float A[16], B[16], C[16], C_ref[16];

    matrix_set_identity(A, 4, 4);
    matrix_set_identity(B, 4, 4);
    matrix_set_zeros(C, 4, 4, 4);
    matrix_set_zeros(C_ref, 4, 4, 4);

    gemm_4x4_inline(C, A, B, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 4, 4, 4, 4, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 4, 4, 4, 1e-5f);
}

static int test_4x4_zeros(void)
{
    printf("  Testing: 4×4 zero matrices (0*0 = 0)\n");

    float A[16], B[16], C[16], C_ref[16];

    matrix_set_zeros(A, 4, 4, 4);
    matrix_set_zeros(B, 4, 4, 4);
    matrix_set_value(C, 4, 4, 4, 999.0f);      // Junk data
    matrix_set_value(C_ref, 4, 4, 4, 999.0f);

    gemm_4x4_inline(C, A, B, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 4, 4, 4, 4, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 4, 4, 4, 1e-5f);
}

static int test_4x4_ones(void)
{
    printf("  Testing: 4×4 ones matrices (all 1s)\n");

    float A[16], B[16], C[16], C_ref[16];

    matrix_set_ones(A, 4, 4, 4);
    matrix_set_ones(B, 4, 4, 4);
    matrix_set_zeros(C, 4, 4, 4);
    matrix_set_zeros(C_ref, 4, 4, 4);

    gemm_4x4_inline(C, A, B, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 4, 4, 4, 4, 1.0f, 0.0f);

    // Expected: Each element = 4 (sum of 1s)
    return matrices_equal(C, C_ref, 4, 4, 4, 1e-5f);
}

static int test_4x4_sequential(void)
{
    printf("  Testing: 4×4 sequential matrices\n");

    float A[16], B[16], C[16], C_ref[16];

    matrix_set_sequential(A, 4, 4, 4);
    matrix_set_sequential(B, 4, 4, 4);
    matrix_set_zeros(C, 4, 4, 4);
    matrix_set_zeros(C_ref, 4, 4, 4);

    gemm_4x4_inline(C, A, B, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 4, 4, 4, 4, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 4, 4, 4, 1e-5f);
}

static int test_4x4_alpha_beta_exhaustive(void)
{
    printf("  Testing: 4×4 exhaustive alpha/beta combinations\n");

    float A[16], B[16];
    matrix_set_sequential(A, 4, 4, 4);
    matrix_set_sequential(B, 4, 4, 4);

    int all_passed = 1;

    for (size_t i = 0; i < n_alpha_beta_cases; i++)
    {
        float C[16], C_ref[16];
        matrix_set_value(C, 4, 4, 4, 2.0f);     // Non-zero initial value
        matrix_set_value(C_ref, 4, 4, 4, 2.0f);

        float alpha = alpha_beta_cases[i].alpha;
        float beta = alpha_beta_cases[i].beta;

        gemm_4x4_inline(C, A, B, alpha, beta);
        gemm_naive(C_ref, A, B, 4, 4, 4, 4, alpha, beta);

        if (!matrices_equal(C, C_ref, 4, 4, 4, 1e-5f))
        {
            printf("    FAILED: %s\n", alpha_beta_cases[i].description);
            all_passed = 0;
        }
        else
        {
            printf("    ✓ %s\n", alpha_beta_cases[i].description);
        }
    }

    return all_passed;
}

//==============================================================================
// TEST: 6×6 Kernel (LDC Handling Critical)
//==============================================================================

static int test_6x6_identity(void)
{
    printf("  Testing: 6×6 identity matrix\n");

    float A[36], B[36], C[64], C_ref[64]; // 64 = 8*8 to test ldc

    matrix_set_identity(A, 6, 6);
    matrix_set_identity(B, 6, 6);
    matrix_set_zeros(C, 6, 6, 6);
    matrix_set_zeros(C_ref, 6, 6, 6);

    gemm_6x6_inline(C, A, B, 6, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 6, 6, 6, 6, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 6, 6, 6, 1e-5f);
}

static int test_6x6_ldc_contiguous(void)
{
    printf("  Testing: 6×6 with ldc=6 (contiguous)\n");

    float A[36], B[36], C[36], C_ref[36];

    matrix_set_sequential(A, 6, 6, 6);
    matrix_set_sequential(B, 6, 6, 6);
    matrix_set_zeros(C, 6, 6, 6);
    matrix_set_zeros(C_ref, 6, 6, 6);

    gemm_6x6_inline(C, A, B, 6, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 6, 6, 6, 6, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 6, 6, 6, 1e-5f);
}

static int test_6x6_ldc_noncontiguous(void)
{
    printf("  Testing: 6×6 with ldc=8 (your fix)\n");

    float A[36], B[36], C[64], C_ref[64]; // 8*8 = 64

    matrix_set_sequential(A, 6, 6, 6);
    matrix_set_sequential(B, 6, 6, 6);
    matrix_set_zeros(C, 6, 8, 8);     // ldc=8
    matrix_set_zeros(C_ref, 6, 8, 8);

    gemm_6x6_inline(C, A, B, 8, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 6, 6, 6, 8, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 6, 6, 8, 1e-5f);
}

static int test_6x6_alpha_beta_cases(void)
{
    printf("  Testing: 6×6 alpha/beta combinations\n");

    float A[36], B[36];
    matrix_set_sequential(A, 6, 6, 6);
    matrix_set_sequential(B, 6, 6, 6);

    int all_passed = 1;

    for (size_t i = 0; i < n_alpha_beta_cases; i++)
    {
        float C[64], C_ref[64]; // ldc=8
        matrix_set_value(C, 6, 8, 8, 3.0f);
        matrix_set_value(C_ref, 6, 8, 8, 3.0f);

        float alpha = alpha_beta_cases[i].alpha;
        float beta = alpha_beta_cases[i].beta;

        gemm_6x6_inline(C, A, B, 8, alpha, beta);
        gemm_naive(C_ref, A, B, 6, 6, 6, 8, alpha, beta);

        if (!matrices_equal(C, C_ref, 6, 6, 8, 1e-5f))
        {
            printf("    FAILED: %s\n", alpha_beta_cases[i].description);
            all_passed = 0;
        }
        else
        {
            printf("    ✓ %s\n", alpha_beta_cases[i].description);
        }
    }

    return all_passed;
}

//==============================================================================
// TEST: 8×8 Kernel (Gold Standard)
//==============================================================================

static int test_8x8_identity(void)
{
    printf("  Testing: 8×8 identity matrix\n");

    float A[64], B[64], C[64], C_ref[64];

    matrix_set_identity(A, 8, 8);
    matrix_set_identity(B, 8, 8);
    matrix_set_zeros(C, 8, 8, 8);
    matrix_set_zeros(C_ref, 8, 8, 8);

    gemm_8x8_inline(C, A, B, 8, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 8, 8, 8, 8, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 8, 8, 8, 1e-5f);
}

static int test_8x8_ldc_variations(void)
{
    printf("  Testing: 8×8 with ldc=16 (non-contiguous)\n");

    float A[64], B[64], C[256], C_ref[256]; // 16*16 = 256

    matrix_set_sequential(A, 8, 8, 8);
    matrix_set_sequential(B, 8, 8, 8);
    matrix_set_zeros(C, 8, 16, 16);
    matrix_set_zeros(C_ref, 8, 16, 16);

    gemm_8x8_inline(C, A, B, 16, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 8, 8, 8, 16, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 8, 8, 16, 1e-5f);
}

static int test_8x8_alpha_beta_exhaustive(void)
{
    printf("  Testing: 8×8 exhaustive alpha/beta combinations\n");

    float A[64], B[64];
    matrix_set_sequential(A, 8, 8, 8);
    matrix_set_sequential(B, 8, 8, 8);

    int all_passed = 1;

    for (size_t i = 0; i < n_alpha_beta_cases; i++)
    {
        float C[64], C_ref[64];
        matrix_set_value(C, 8, 8, 8, 5.0f);
        matrix_set_value(C_ref, 8, 8, 8, 5.0f);

        float alpha = alpha_beta_cases[i].alpha;
        float beta = alpha_beta_cases[i].beta;

        gemm_8x8_inline(C, A, B, 8, alpha, beta);
        gemm_naive(C_ref, A, B, 8, 8, 8, 8, alpha, beta);

        if (!matrices_equal(C, C_ref, 8, 8, 8, 1e-5f))
        {
            printf("    FAILED: %s\n", alpha_beta_cases[i].description);
            all_passed = 0;
        }
        else
        {
            printf("    ✓ %s\n", alpha_beta_cases[i].description);
        }
    }

    return all_passed;
}

//==============================================================================
// TEST: Dispatcher Logic
//==============================================================================

static int test_dispatcher_routes_4x4(void)
{
    printf("  Testing: Dispatcher routes 4×4 (ldc=4)\n");

    float A[16], B[16], C[16], C_ref[16];

    matrix_set_sequential(A, 4, 4, 4);
    matrix_set_sequential(B, 4, 4, 4);
    matrix_set_zeros(C, 4, 4, 4);
    matrix_set_zeros(C_ref, 4, 4, 4);

    int result = gemm_small_dispatch(C, A, B, 4, 4, 4, 4, 1.0f, 0.0f);

    if (result != 0)
    {
        printf("    FAIL: Dispatcher should handle 4×4 with ldc=4\n");
        return 0;
    }

    gemm_naive(C_ref, A, B, 4, 4, 4, 4, 1.0f, 0.0f);
    return matrices_equal(C, C_ref, 4, 4, 4, 1e-5f);
}

static int test_dispatcher_rejects_4x4_noncontiguous(void)
{
    printf("  Testing: Dispatcher rejects 4×4 with ldc=8\n");

    float A[16] = {0}, B[16] = {0}, C[64] = {0};  // ✅ Zero-initialize

    int result = gemm_small_dispatch(C, A, B, 4, 4, 4, 8, 1.0f, 0.0f);

    if (result == 0)
    {
        printf("    FAIL: Should reject 4×4 with ldc!=4\n");
        return 0;
    }

    printf("    Correctly rejected (returned %d)\n", result);
    return 1;
}
static int test_dispatcher_routes_6x6(void)
{
    printf("  Testing: Dispatcher routes 6×6 (any ldc)\n");

    float A[36], B[36], C[64], C_ref[64];

    matrix_set_sequential(A, 6, 6, 6);
    matrix_set_sequential(B, 6, 6, 6);
    matrix_set_zeros(C, 6, 8, 8);
    matrix_set_zeros(C_ref, 6, 8, 8);

    int result = gemm_small_dispatch(C, A, B, 6, 6, 6, 8, 1.0f, 0.0f);

    if (result != 0)
    {
        printf("    FAIL: Dispatcher should handle 6×6 with any ldc\n");
        return 0;
    }

    gemm_naive(C_ref, A, B, 6, 6, 6, 8, 1.0f, 0.0f);
    return matrices_equal(C, C_ref, 6, 6, 8, 1e-5f);
}

static int test_dispatcher_routes_8x8(void)
{
    printf("  Testing: Dispatcher routes 8×8\n");

    float A[64], B[64], C[64], C_ref[64];

    matrix_set_sequential(A, 8, 8, 8);
    matrix_set_sequential(B, 8, 8, 8);
    matrix_set_zeros(C, 8, 8, 8);
    matrix_set_zeros(C_ref, 8, 8, 8);

    int result = gemm_small_dispatch(C, A, B, 8, 8, 8, 8, 1.0f, 0.0f);

    if (result != 0)
    {
        printf("    FAIL: Dispatcher should handle 8×8\n");
        return 0;
    }

    gemm_naive(C_ref, A, B, 8, 8, 8, 8, 1.0f, 0.0f);
    return matrices_equal(C, C_ref, 8, 8, 8, 1e-5f);
}


static int test_dispatcher_rejects_large(void)
{
    printf("  Testing: Dispatcher rejects 9×9\n");

    float A[81] = {0}, B[81] = {0}, C[81] = {0};  // ✅ Zero-initialize

    int result = gemm_small_dispatch(C, A, B, 9, 9, 9, 9, 1.0f, 0.0f);

    if (result == 0)
    {
        printf("    FAIL: Should reject 9×9 (too large for Tier 1)\n");
        return 0;
    }

    printf("    Correctly rejected (returned %d)\n", result);
    return 1;
}

static int test_dispatcher_rejects_high_ops(void)
{
    printf("  Testing: Dispatcher rejects 4×4×100 (too many ops)\n");

    float A[400] = {0}, B[400] = {0}, C[16] = {0};  // ✅ Zero-initialize

    int result = gemm_small_dispatch(C, A, B, 4, 100, 4, 4, 1.0f, 0.0f);

    if (result == 0)
    {
        printf("    FAIL: Should reject high-op-count matrices\n");
        return 0;
    }

    printf("    Correctly rejected (returned %d)\n", result);
    return 1;
}



//==============================================================================
// TEST: 8×4 Rectangular Kernel
//==============================================================================

static int test_8x4_identity(void)
{
    printf("  Testing: 8×4 with identity-like data\n");

    float A[8*8], B[8*4], C[8*4], C_ref[8*4];

    // A = I (8×8 identity, but we only use first 8 columns for 8×K)
    matrix_set_identity(A, 8, 8);
    
    // B = Sequential (8×4)
    matrix_set_sequential(B, 8, 4, 4);
    
    // C = zeros
    matrix_set_zeros(C, 8, 4, 4);
    matrix_set_zeros(C_ref, 8, 4, 4);

    // K=8, so A is 8×8, B is 8×4
    gemm_8x4_inline(C, A, B, 8, 4, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 8, 8, 4, 4, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 8, 4, 4, 1e-5f);
}

static int test_8x4_alpha_beta_cases(void)
{
    printf("  Testing: 8×4 alpha/beta combinations\n");

    float A[8*8], B[8*4];
    matrix_set_sequential(A, 8, 8, 8);
    matrix_set_sequential(B, 8, 4, 4);

    int all_passed = 1;

    for (size_t i = 0; i < n_alpha_beta_cases; i++)
    {
        float C[8*4], C_ref[8*4];
        matrix_set_value(C, 8, 4, 4, 2.0f);
        matrix_set_value(C_ref, 8, 4, 4, 2.0f);

        float alpha = alpha_beta_cases[i].alpha;
        float beta = alpha_beta_cases[i].beta;

        gemm_8x4_inline(C, A, B, 8, 4, alpha, beta);
        gemm_naive(C_ref, A, B, 8, 8, 4, 4, alpha, beta);

        if (!matrices_equal(C, C_ref, 8, 4, 4, 1e-5f))
        {
            printf("    FAILED: %s\n", alpha_beta_cases[i].description);
            all_passed = 0;
        }
        else
        {
            printf("    ✓ %s\n", alpha_beta_cases[i].description);
        }
    }

    return all_passed;
}


static int test_8x4_arbitrary_K(void)
{
    printf("  Testing: 8×4 with K=12 (arbitrary inner dimension)\n");

    float A[8*12], B[12*4], C[8*4], C_ref[8*4];

    matrix_set_sequential(A, 8, 12, 12);
    matrix_set_sequential(B, 12, 4, 4);
    matrix_set_zeros(C, 8, 4, 4);
    matrix_set_zeros(C_ref, 8, 4, 4);

    gemm_8x4_inline(C, A, B, 12, 4, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 8, 12, 4, 4, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 8, 4, 4, 1e-5f);
}

//==============================================================================
// TEST: 4×8 Rectangular Kernel
//==============================================================================

static int test_4x8_basic(void)
{
    printf("  Testing: 4×8 basic correctness\n");

    float A[4*8], B[8*8], C[4*8], C_ref[4*8];

    matrix_set_sequential(A, 4, 8, 8);
    matrix_set_sequential(B, 8, 8, 8);
    matrix_set_zeros(C, 4, 8, 8);
    matrix_set_zeros(C_ref, 4, 8, 8);

    gemm_4x8_inline(C, A, B, 8, 8, 1.0f, 0.0f);
    gemm_naive(C_ref, A, B, 4, 8, 8, 8, 1.0f, 0.0f);

    return matrices_equal(C, C_ref, 4, 8, 8, 1e-5f);
}

//==============================================================================
// TEST SUITE RUNNER
//==============================================================================

/**
 * @brief Run all Tier 1 (small kernel) tests
 * @param results Output: test results structure
 * @return 0 if all tests passed, 1 if any failed
 */
int run_gemm_small_tests(test_results_t *results)
{
    // Initialize results
    results->total = 0;
    results->passed = 0;
    results->failed = 0;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  GEMM Small Kernels (Tier 1) - Test Suite               ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    // Group 1: 4×4 Kernel
    printf("\n═══ Test Group 1: 4×4 Kernel ═══\n");
    RUN_TEST(results, test_4x4_identity);
    RUN_TEST(results, test_4x4_zeros);
    RUN_TEST(results, test_4x4_ones);
    RUN_TEST(results, test_4x4_sequential);
    RUN_TEST(results, test_4x4_alpha_beta_exhaustive);

    // Group 2: 6×6 Kernel (LDC Handling)
    printf("\n═══ Test Group 2: 6×6 Kernel (LDC Critical) ═══\n");
    RUN_TEST(results, test_6x6_identity);
    RUN_TEST(results, test_6x6_ldc_contiguous);
    RUN_TEST(results, test_6x6_ldc_noncontiguous);
    RUN_TEST(results, test_6x6_alpha_beta_cases);

    // Group 3: 8×8 Kernel (Gold Standard)
    printf("\n═══ Test Group 3: 8×8 Kernel (Gold Standard) ═══\n");
    RUN_TEST(results, test_8x8_identity);
    RUN_TEST(results, test_8x8_ldc_variations);
    RUN_TEST(results, test_8x8_alpha_beta_exhaustive);

    // Group 4: Dispatcher Logic
    printf("\n═══ Test Group 4: Dispatcher Logic ═══\n");
    RUN_TEST(results, test_dispatcher_routes_4x4);
    RUN_TEST(results, test_dispatcher_rejects_4x4_noncontiguous);
    RUN_TEST(results, test_dispatcher_routes_6x6);
    RUN_TEST(results, test_dispatcher_routes_8x8);
    RUN_TEST(results, test_dispatcher_rejects_large);
    RUN_TEST(results, test_dispatcher_rejects_high_ops);

    printf("\n═══ Test Group 5: Rectangular Kernels (8×4, 4×8) ═══\n");
    RUN_TEST(results, test_8x4_identity);
    RUN_TEST(results, test_8x4_alpha_beta_cases);
    RUN_TEST(results, test_8x4_arbitrary_K);
    RUN_TEST(results, test_4x8_basic);

    // Print results
    print_test_results("GEMM Small Kernels - Results", results);

    return (results->failed == 0) ? 0 : 1;
}

// Optional: Standalone mode (compile with -DSTANDALONE)
#ifdef STANDALONE
int main(void)
{
    test_results_t results;
    int ret = run_gemm_small_tests(&results);
    
    if (ret == 0) {
        printf("\n🎉 " TEST_PASS " All tests passed!\n\n");
    } else {
        printf("\n❌ " TEST_FAIL " %d test(s) failed\n\n", results.failed);
    }
    
    return ret;
}
#endif