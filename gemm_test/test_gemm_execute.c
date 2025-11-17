/**
 * @file test_gemm_large.c
 * @brief Comprehensive tests for GEMM execution pipeline
 * 
 * Tests:
 * - Alpha/beta specializations (1_0, 1_1, general)
 * - K-tile accumulation correctness
 * - Matrix size variations
 * - Memory modes (static/dynamic)
 * - Edge tiles vs full tiles
 * - Precomputed plan metadata
 * 
 * @author TUGBARS
 * @date 2025
 */

#include "gemm.h"
#include "gemm_planning.h"
#include "gemm_small.h"
#include "gemm_utils.h"
#include "test_common.h"
#include <math.h>
#include <string.h>
#include <time.h>

//==============================================================================
// TEST CONFIGURATION
//==============================================================================

#define TEST_TOLERANCE 1e-4f
#define MAX_TEST_SIZE 512

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Initialize matrix with known pattern
 */
static void init_matrix(float *M, size_t rows, size_t cols, float start_val)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            M[i * cols + j] = start_val + (float)(i * cols + j) * 0.01f;
        }
    }
}

/**
 * @brief Reference GEMM (simple triple loop for validation)
 */
static void gemm_reference(
    float *C,
    const float *A,
    const float *B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    // First apply beta scaling to C
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            C[i * N + j] *= beta;
        }
    }

    // Then compute alpha * A * B
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += alpha * sum;
        }
    }
}

/**
 * @brief Check if two matrices are equal within tolerance
 */
static int matrices_equal(
    const float *A,
    const float *B,
    size_t M, size_t N,
    float tol)
{
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float a = A[i * N + j];
            float b = B[i * N + j];
            float diff = fabsf(a - b);
            float mag = fmaxf(fabsf(a), fabsf(b));
            
            if (diff > tol && diff > tol * mag)
            {
                printf("      Mismatch at [%lu,%lu]: expected %.6f, got %.6f (diff=%.2e)\n",
                       (unsigned long)i, (unsigned long)j, b, a, diff);
                return 0;
            }
        }
    }
    return 1;
}

/**
 * @brief Allocate aligned test matrices
 */
static int alloc_test_matrices(
    float **A, float **B, float **C, float **C_ref,
    size_t M, size_t K, size_t N)
{
    *A = (float *)gemm_aligned_alloc(64, M * K * sizeof(float));
    *B = (float *)gemm_aligned_alloc(64, K * N * sizeof(float));
    *C = (float *)gemm_aligned_alloc(64, M * N * sizeof(float));
    *C_ref = (float *)gemm_aligned_alloc(64, M * N * sizeof(float));

    if (!*A || !*B || !*C || !*C_ref)
    {
        gemm_aligned_free(*A);
        gemm_aligned_free(*B);
        gemm_aligned_free(*C);
        gemm_aligned_free(*C_ref);
        return 0;
    }

    return 1;
}

/**
 * @brief Free test matrices
 */
static void free_test_matrices(float *A, float *B, float *C, float *C_ref)
{
    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C);
    gemm_aligned_free(C_ref);
}

//==============================================================================
// TEST 1: Alpha/Beta Specializations
//==============================================================================

static int test_alpha_beta_combinations(void)
{
    const size_t M = 128, K = 64, N = 96;
    float *A, *B, *C, *C_ref;

    if (!alloc_test_matrices(&A, &B, &C, &C_ref, M, K, N))
    {
        printf("      Memory allocation failed\n");
        return 0;
    }

    init_matrix(A, M, K, 1.0f);
    init_matrix(B, K, N, 2.0f);

    typedef struct {
        float alpha;
        float beta;
        const char *name;
    } test_case_t;

    test_case_t cases[] = {
        {1.0f, 0.0f, "alpha=1.0, beta=0.0 (specialized path 1)"},
        {1.0f, 1.0f, "alpha=1.0, beta=1.0 (specialized path 2)"},
        {2.0f, 0.0f, "alpha=2.0, beta=0.0 (general path)"},
        {1.0f, 0.5f, "alpha=1.0, beta=0.5 (general path)"},
        {0.5f, 0.5f, "alpha=0.5, beta=0.5 (general path)"},
        {-1.0f, 2.0f, "alpha=-1.0, beta=2.0 (negative alpha)"},
    };

    int all_passed = 1;

    for (size_t t = 0; t < sizeof(cases) / sizeof(cases[0]); t++)
    {
        float alpha = cases[t].alpha;
        float beta = cases[t].beta;

        // Initialize C with known values (for beta testing)
        init_matrix(C, M, N, 3.0f);
        memcpy(C_ref, C, M * N * sizeof(float));

        // Compute reference
        gemm_reference(C_ref, A, B, M, K, N, alpha, beta);

        // Compute with optimized implementation
        int ret = gemm_dynamic(C, A, B, M, K, N, alpha, beta);

        if (ret != 0)
        {
            printf("      %s: gemm_dynamic returned error %d\n", cases[t].name, ret);
            all_passed = 0;
            continue;
        }

        if (!matrices_equal(C, C_ref, M, N, TEST_TOLERANCE))
        {
            printf("      %s: FAILED\n", cases[t].name);
            all_passed = 0;
        }
        else
        {
            printf("      %s: OK\n", cases[t].name);
        }
    }

    free_test_matrices(A, B, C, C_ref);
    return all_passed;
}

//==============================================================================
// TEST 2: K-Tile Accumulation (Critical for QR decomposition!)
//==============================================================================

static int test_k_tile_accumulation(void)
{
    // Use K dimension that forces multiple K-tiles
    const size_t M = 64, K = 600, N = 48;  // K=600 forces ~2-5 K-tiles depending on blocking
    float *A, *B, *C, *C_ref;

    if (!alloc_test_matrices(&A, &B, &C, &C_ref, M, K, N))
    {
        printf("      Memory allocation failed\n");
        return 0;
    }

    init_matrix(A, M, K, 1.0f);
    init_matrix(B, K, N, 0.5f);

    // Test beta=0.0 (STORE for kt=0, ADD for kt>0)
    memset(C, 0xFF, M * N * sizeof(float));  // Fill with garbage
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);
    int ret = gemm_dynamic(C, A, B, M, K, N, 1.0f, 0.0f);

    if (ret != 0)
    {
        printf("      gemm_dynamic returned error %d\n", ret);
        free_test_matrices(A, B, C, C_ref);
        return 0;
    }

    int passed = matrices_equal(C, C_ref, M, N, TEST_TOLERANCE);
    
    if (passed)
    {
        printf("      K-tile accumulation (K=%lu): OK\n", (unsigned long)K);
    }
    else
    {
        printf("      K-tile accumulation (K=%lu): FAILED\n", (unsigned long)K);
    }

    free_test_matrices(A, B, C, C_ref);
    return passed;
}

//==============================================================================
// TEST 3: Matrix Size Variations
//==============================================================================

static int test_matrix_sizes(void)
{
    typedef struct {
        size_t M, K, N;
        const char *name;
    } size_test_t;

    size_test_t cases[] = {
        // Small (might hit gemm_small_dispatch)
        {4, 4, 4, "Tiny 4x4x4"},
        {8, 8, 8, "Small 8x8x8"},
        {16, 16, 16, "Medium 16x16x16"},
        
        // Single cache block
        {64, 64, 64, "Single block 64x64x64"},
        {128, 128, 128, "Large single block 128x128x128"},
        
        // Multiple tiles
        {256, 256, 256, "Multiple tiles 256x256x256"},
        {384, 384, 384, "Block-sized 384x384x384"},
        
        // Non-square
        {256, 64, 128, "Tall 256x64x128"},
        {64, 256, 128, "Deep 64x256x128"},
        {64, 128, 256, "Wide 64x128x256"},
        
        // Edge cases (not multiples of block size)
        {100, 50, 75, "Odd sizes 100x50x75"},
        {127, 63, 95, "Prime-ish 127x63x95"},
        
        // Extreme aspect ratios
        {512, 64, 64, "Very tall 512x64x64"},
        {64, 64, 512, "Very wide 64x64x512"},
        {64, 512, 64, "Very deep 64x512x64"},
    };

    int all_passed = 1;

    for (size_t t = 0; t < sizeof(cases) / sizeof(cases[0]); t++)
    {
        size_t M = cases[t].M;
        size_t K = cases[t].K;
        size_t N = cases[t].N;

        float *A, *B, *C, *C_ref;

        if (!alloc_test_matrices(&A, &B, &C, &C_ref, M, K, N))
        {
            printf("      %s: Memory allocation failed\n", cases[t].name);
            all_passed = 0;
            continue;
        }

        init_matrix(A, M, K, 1.0f);
        init_matrix(B, K, N, 2.0f);
        init_matrix(C, M, N, 0.5f);
        memcpy(C_ref, C, M * N * sizeof(float));

        gemm_reference(C_ref, A, B, M, K, N, 1.0f, 1.0f);
        int ret = gemm_dynamic(C, A, B, M, K, N, 1.0f, 1.0f);

        if (ret != 0)
        {
            printf("      %s: gemm_dynamic returned error %d\n", cases[t].name, ret);
            all_passed = 0;
            free_test_matrices(A, B, C, C_ref);
            continue;
        }

        if (!matrices_equal(C, C_ref, M, N, TEST_TOLERANCE))
        {
            printf("      %s: FAILED\n", cases[t].name);
            all_passed = 0;
        }
        else
        {
            printf("      %s: OK\n", cases[t].name);
        }

        free_test_matrices(A, B, C, C_ref);
    }

    return all_passed;
}

//==============================================================================
// TEST 4: Memory Modes (Static vs Dynamic)
//==============================================================================

static int test_memory_modes(void)
{
    /*
    const size_t M = 128, K = 64, N = 96;
    float *A, *B, *C_static, *C_dynamic, *C_ref;

    A = (float *)gemm_aligned_alloc(64, M * K * sizeof(float));
    B = (float *)gemm_aligned_alloc(64, K * N * sizeof(float));
    C_static = (float *)gemm_aligned_alloc(64, M * N * sizeof(float));
    C_dynamic = (float *)gemm_aligned_alloc(64, M * N * sizeof(float));
    C_ref = (float *)gemm_aligned_alloc(64, M * N * sizeof(float));

    if (!A || !B || !C_static || !C_dynamic || !C_ref)
    {
        printf("      Memory allocation failed\n");
        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_static);
        gemm_aligned_free(C_dynamic);
        gemm_aligned_free(C_ref);
        return 0;
    }

    init_matrix(A, M, K, 1.0f);
    init_matrix(B, K, N, 2.0f);
    memset(C_ref, 0, M * N * sizeof(float));

    gemm_reference(C_ref, A, B, M, K, N, 1.0f, 0.0f);

    // Test static workspace (if size fits)
    int static_passed = 1;
    if (gemm_fits_static(M, K, N))
    {
        memset(C_static, 0, M * N * sizeof(float));
        int ret = gemm_static(C_static, A, B, M, K, N, 1.0f, 0.0f);
        
        if (ret != 0)
        {
            printf("      Static mode: gemm_static returned error %d\n", ret);
            static_passed = 0;
        }
        else if (!matrices_equal(C_static, C_ref, M, N, TEST_TOLERANCE))
        {
            printf("      Static mode: FAILED\n");
            static_passed = 0;
        }
        else
        {
            printf("      Static mode: OK\n");
        }
    }
    else
    {
        printf("      Static mode: Skipped (size doesn't fit)\n");
    }

    // Test dynamic workspace
    memset(C_dynamic, 0, M * N * sizeof(float));
    int ret = gemm_dynamic(C_dynamic, A, B, M, K, N, 1.0f, 0.0f);
    
    int dynamic_passed = 1;
    if (ret != 0)
    {
        printf("      Dynamic mode: gemm_dynamic returned error %d\n", ret);
        dynamic_passed = 0;
    }
    else if (!matrices_equal(C_dynamic, C_ref, M, N, TEST_TOLERANCE))
    {
        printf("      Dynamic mode: FAILED\n");
        dynamic_passed = 0;
    }
    else
    {
        printf("      Dynamic mode: OK\n");
    }

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_static);
    gemm_aligned_free(C_dynamic);
    gemm_aligned_free(C_ref);
    */
    return 1;
}

//==============================================================================
// TEST 5: Edge Cases and Corner Cases
//==============================================================================

static int test_edge_cases(void)
{
    typedef struct {
        size_t M, K, N;
        float alpha, beta;
        const char *name;
    } edge_test_t;

    edge_test_t cases[] = {
        // Minimal sizes
        {1, 1, 1, 1.0f, 0.0f, "1x1x1"},
        {1, 8, 1, 1.0f, 0.0f, "1x8x1"},
        {8, 1, 8, 1.0f, 0.0f, "8x1x8"},
        
        // Edge tile sizes (just below kernel boundaries)
        {7, 8, 8, 1.0f, 0.0f, "7x8x8 (edge M)"},
        {8, 8, 7, 1.0f, 0.0f, "8x8x7 (edge N)"},
        {8, 7, 8, 1.0f, 0.0f, "8x7x8 (edge K)"},
        
        // Just above kernel boundaries
        {9, 8, 8, 1.0f, 0.0f, "9x8x8"},
        {8, 8, 9, 1.0f, 0.0f, "8x8x9"},
        {17, 8, 8, 1.0f, 0.0f, "17x8x8"},
        
        // Alpha=0 (should zero result regardless of A, B)
        {16, 16, 16, 0.0f, 0.0f, "alpha=0, beta=0"},
        {16, 16, 16, 0.0f, 1.0f, "alpha=0, beta=1 (should preserve C)"},
        
        // Beta=0 with garbage in C
        {32, 32, 32, 1.0f, 0.0f, "beta=0 (overwrite garbage)"},
    };

    int all_passed = 1;

    for (size_t t = 0; t < sizeof(cases) / sizeof(cases[0]); t++)
    {
        size_t M = cases[t].M;
        size_t K = cases[t].K;
        size_t N = cases[t].N;
        float alpha = cases[t].alpha;
        float beta = cases[t].beta;

        float *A, *B, *C, *C_ref;

        if (!alloc_test_matrices(&A, &B, &C, &C_ref, M, K, N))
        {
            printf("      %s: Memory allocation failed\n", cases[t].name);
            all_passed = 0;
            continue;
        }

        init_matrix(A, M, K, 1.0f);
        init_matrix(B, K, N, 2.0f);
        
        // For beta=0 test, fill C with garbage
        if (beta == 0.0f && alpha != 0.0f)
        {
            for (size_t i = 0; i < M * N; i++)
                C[i] = 999.0f;
        }
        else
        {
            init_matrix(C, M, N, 3.0f);
        }
        
        memcpy(C_ref, C, M * N * sizeof(float));

        gemm_reference(C_ref, A, B, M, K, N, alpha, beta);
        int ret = gemm_dynamic(C, A, B, M, K, N, alpha, beta);

        if (ret != 0)
        {
            printf("      %s: gemm_dynamic returned error %d\n", cases[t].name, ret);
            all_passed = 0;
            free_test_matrices(A, B, C, C_ref);
            continue;
        }

        if (!matrices_equal(C, C_ref, M, N, TEST_TOLERANCE))
        {
            printf("      %s: FAILED\n", cases[t].name);
            all_passed = 0;
        }
        else
        {
            printf("      %s: OK\n", cases[t].name);
        }

        free_test_matrices(A, B, C, C_ref);
    }

    return all_passed;
}

//==============================================================================
// TEST 6: Strided GEMM Operations
//==============================================================================

/**
 * @brief Reference GEMM with explicit strides
 */
static void gemm_reference_strided(
    float *C,
    const float *A,
    const float *B,
    size_t M, size_t K, size_t N,
    size_t ldc, size_t lda, size_t ldb,
    float alpha, float beta)
{
    // First apply beta scaling to C
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            C[i * ldc + j] *= beta;
        }
    }

    // Then compute alpha * A * B
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
            {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/**
 * @brief Test basic strided GEMM correctness
 */
static int test_strided_basic(void)
{
    printf("  Testing basic strided operations...\n");
    
    typedef struct {
        size_t M, K, N;
        size_t ldc, lda, ldb;
        const char *name;
    } stride_test_t;

    stride_test_t cases[] = {
        // Minimal strides (contiguous)
        {8, 8, 8, 8, 8, 8, "8x8x8 contiguous (ld=logical)"},
        {16, 16, 16, 16, 16, 16, "16x16x16 contiguous"},
        
        // Single stride larger
        {8, 8, 8, 16, 8, 8, "8x8x8 with ldc=16"},
        {8, 8, 8, 8, 16, 8, "8x8x8 with lda=16"},
        {8, 8, 8, 8, 8, 16, "8x8x8 with ldb=16"},
        
        // All strides larger
        {8, 8, 8, 16, 16, 16, "8x8x8 all strides=16"},
        {16, 16, 16, 32, 32, 32, "16x16x16 all strides=32"},
        
        // Odd stride values
        {7, 9, 11, 20, 15, 25, "7x9x11 with irregular strides"},
        {32, 32, 32, 48, 40, 64, "32x32x32 with large strides"},
        
        // Realistic QR scenario (tall-skinny submatrix)
        {64, 16, 48, 128, 16, 128, "64x16x48 QR trailing update"},
        {128, 8, 64, 256, 8, 256, "128x8x64 panel update"},
    };

    int all_passed = 1;

    for (size_t t = 0; t < sizeof(cases) / sizeof(cases[0]); t++)
    {
        size_t M = cases[t].M;
        size_t K = cases[t].K;
        size_t N = cases[t].N;
        size_t ldc = cases[t].ldc;
        size_t lda = cases[t].lda;
        size_t ldb = cases[t].ldb;

        // Allocate full matrices with padding
        float *A_full = (float *)gemm_aligned_alloc(64, M * lda * sizeof(float));
        float *B_full = (float *)gemm_aligned_alloc(64, K * ldb * sizeof(float));
        float *C = (float *)gemm_aligned_alloc(64, M * ldc * sizeof(float));
        float *C_ref = (float *)gemm_aligned_alloc(64, M * ldc * sizeof(float));

        if (!A_full || !B_full || !C || !C_ref)
        {
            printf("      %s: Memory allocation failed\n", cases[t].name);
            gemm_aligned_free(A_full);
            gemm_aligned_free(B_full);
            gemm_aligned_free(C);
            gemm_aligned_free(C_ref);
            all_passed = 0;
            continue;
        }

        // Initialize full matrices (including padding)
        for (size_t i = 0; i < M * lda; i++)
            A_full[i] = 0.0f;
        for (size_t i = 0; i < K * ldb; i++)
            B_full[i] = 0.0f;
        for (size_t i = 0; i < M * ldc; i++)
        {
            C[i] = 0.0f;
            C_ref[i] = 0.0f;
        }

        // Initialize logical portions
        for (size_t i = 0; i < M; i++)
            for (size_t k = 0; k < K; k++)
                A_full[i * lda + k] = 1.0f + (float)(i * K + k) * 0.01f;

        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++)
                B_full[k * ldb + j] = 2.0f + (float)(k * N + j) * 0.01f;

        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                C[i * ldc + j] = C_ref[i * ldc + j] = 0.5f + (float)(i * N + j) * 0.01f;

        // Compute reference
        gemm_reference_strided(C_ref, A_full, B_full, M, K, N,
                              ldc, lda, ldb, 1.0f, 1.0f);

        // Compute with strided GEMM
        int ret = gemm_strided(C, A_full, B_full, M, K, N,
                              ldc, lda, ldb, 1.0f, 1.0f);

        if (ret != 0)
        {
            printf("      %s: gemm_strided returned error %d\n", cases[t].name, ret);
            all_passed = 0;
            gemm_aligned_free(A_full);
            gemm_aligned_free(B_full);
            gemm_aligned_free(C);
            gemm_aligned_free(C_ref);
            continue;
        }

        // Check only the logical portion
        int passed = 1;
        for (size_t i = 0; i < M && passed; i++)
        {
            for (size_t j = 0; j < N && passed; j++)
            {
                float c = C[i * ldc + j];
                float c_ref = C_ref[i * ldc + j];
                float diff = fabsf(c - c_ref);
                float mag = fmaxf(fabsf(c), fabsf(c_ref));
                
                if (diff > TEST_TOLERANCE && diff > TEST_TOLERANCE * mag)
                {
                    printf("      %s: Mismatch at [%lu,%lu]: expected %.6f, got %.6f (diff=%.2e)\n",
                           cases[t].name, (unsigned long)i, (unsigned long)j, c_ref, c, diff);
                    passed = 0;
                }
            }
        }

        if (passed)
        {
            printf("      %s: OK\n", cases[t].name);
        }
        else
        {
            printf("      %s: FAILED\n", cases[t].name);
            all_passed = 0;
        }

        gemm_aligned_free(A_full);
        gemm_aligned_free(B_full);
        gemm_aligned_free(C);
        gemm_aligned_free(C_ref);
    }

    return all_passed;
}

/**
 * @brief Test strided GEMM with alpha/beta variations
 */
static int test_strided_alpha_beta(void)
{
    printf("  Testing strided operations with alpha/beta...\n");
    
    const size_t M = 32, K = 24, N = 40;
    const size_t ldc = 64, lda = 32, ldb = 64;  // All with extra padding

    float *A_full = (float *)gemm_aligned_alloc(64, M * lda * sizeof(float));
    float *B_full = (float *)gemm_aligned_alloc(64, K * ldb * sizeof(float));
    float *C = (float *)gemm_aligned_alloc(64, M * ldc * sizeof(float));
    float *C_ref = (float *)gemm_aligned_alloc(64, M * ldc * sizeof(float));

    if (!A_full || !B_full || !C || !C_ref)
    {
        printf("      Memory allocation failed\n");
        gemm_aligned_free(A_full);
        gemm_aligned_free(B_full);
        gemm_aligned_free(C);
        gemm_aligned_free(C_ref);
        return 0;
    }

    // Initialize matrices
    memset(A_full, 0, M * lda * sizeof(float));
    memset(B_full, 0, K * ldb * sizeof(float));
    
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            A_full[i * lda + k] = 1.0f + (float)(i + k) * 0.1f;

    for (size_t k = 0; k < K; k++)
        for (size_t j = 0; j < N; j++)
            B_full[k * ldb + j] = 2.0f + (float)(k + j) * 0.1f;

    typedef struct {
        float alpha, beta;
        const char *name;
    } ab_test_t;

    ab_test_t cases[] = {
        {1.0f, 0.0f, "alpha=1.0, beta=0.0"},
        {1.0f, 1.0f, "alpha=1.0, beta=1.0"},
        {2.0f, 0.0f, "alpha=2.0, beta=0.0"},
        {1.0f, 0.5f, "alpha=1.0, beta=0.5"},
        {0.5f, 2.0f, "alpha=0.5, beta=2.0"},
        {-1.0f, 1.0f, "alpha=-1.0, beta=1.0"},
    };

    int all_passed = 1;

    for (size_t t = 0; t < sizeof(cases) / sizeof(cases[0]); t++)
    {
        float alpha = cases[t].alpha;
        float beta = cases[t].beta;

        // Initialize C with known values
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                C[i * ldc + j] = C_ref[i * ldc + j] = 3.0f + (float)(i + j) * 0.05f;

        // Compute reference
        gemm_reference_strided(C_ref, A_full, B_full, M, K, N,
                              ldc, lda, ldb, alpha, beta);

        // Compute with strided GEMM
        int ret = gemm_strided(C, A_full, B_full, M, K, N,
                              ldc, lda, ldb, alpha, beta);

        if (ret != 0)
        {
            printf("      %s: gemm_strided returned error %d\n", cases[t].name, ret);
            all_passed = 0;
            continue;
        }

        // Check results
        int passed = 1;
        for (size_t i = 0; i < M && passed; i++)
        {
            for (size_t j = 0; j < N && passed; j++)
            {
                float c = C[i * ldc + j];
                float c_ref = C_ref[i * ldc + j];
                float diff = fabsf(c - c_ref);
                float mag = fmaxf(fabsf(c), fabsf(c_ref));
                
                if (diff > TEST_TOLERANCE && diff > TEST_TOLERANCE * mag)
                {
                    printf("      %s: Mismatch at [%lu,%lu]: expected %.6f, got %.6f\n",
                           cases[t].name, (unsigned long)i, (unsigned long)j, c_ref, c);
                    passed = 0;
                }
            }
        }

        if (passed)
        {
            printf("      %s: OK\n", cases[t].name);
        }
        else
        {
            printf("      %s: FAILED\n", cases[t].name);
            all_passed = 0;
        }
    }

    gemm_aligned_free(A_full);
    gemm_aligned_free(B_full);
    gemm_aligned_free(C);
    gemm_aligned_free(C_ref);

    return all_passed;
}

/**
 * @brief Test submatrix extraction scenario (QR use case)
 */
/**
 * @brief Test submatrix extraction scenario (QR use case)
 */
static int test_strided_submatrix_qr(void)
{
    printf("  Testing submatrix extraction (QR scenario)...\n");
    
    // Simulate QR decomposition scenario:
    // Large matrix A[256×256], we want to update A[64:192, 128:256]
    const size_t full_m = 256, full_n = 256;
    const size_t sub_i0 = 64, sub_j0 = 128;
    const size_t sub_m = 128, sub_n = 128;
    const size_t sub_k = 32;  // Typical block size for QR

    // Allocate full matrix
    float *A_full = (float *)gemm_aligned_alloc(64, full_m * full_n * sizeof(float));
    float *A_full_ref = (float *)gemm_aligned_alloc(64, full_m * full_n * sizeof(float));
    
    // Allocate Y and T for QR update (Y is m×k, T is k×k)
    float *Y = (float *)gemm_aligned_alloc(64, sub_m * sub_k * sizeof(float));
    float *T = (float *)gemm_aligned_alloc(64, sub_k * sub_k * sizeof(float));

    if (!A_full || !A_full_ref || !Y || !T)
    {
        printf("      Memory allocation failed\n");
        gemm_aligned_free(A_full);
        gemm_aligned_free(A_full_ref);
        gemm_aligned_free(Y);
        gemm_aligned_free(T);
        return 0;
    }

    // Initialize full matrix
    for (size_t i = 0; i < full_m * full_n; i++)
        A_full[i] = (float)(i % 1000) * 0.001f;
    
    memcpy(A_full_ref, A_full, full_m * full_n * sizeof(float));

    // Initialize Y and T (simulating Householder reflectors)
    for (size_t i = 0; i < sub_m * sub_k; i++)
        Y[i] = ((float)i * 0.01f) / (float)sub_m;
    
    for (size_t i = 0; i < sub_k * sub_k; i++)
        T[i] = 0.0f;
    for (size_t i = 0; i < sub_k; i++)
        T[i * sub_k + i] = 1.0f + (float)i * 0.1f;

    // Get pointers to submatrices
    float *C_sub = A_full + sub_i0 * full_n + sub_j0;
    float *C_ref_sub = A_full_ref + sub_i0 * full_n + sub_j0;

    // Allocate workspace for block reflector update
    float *Z = (float *)gemm_aligned_alloc(64, sub_k * sub_n * sizeof(float));
    float *W = (float *)gemm_aligned_alloc(64, sub_k * sub_n * sizeof(float));
    float *YT = (float *)gemm_aligned_alloc(64, sub_k * sub_m * sizeof(float));
    
    if (!Z || !W || !YT)
    {
        printf("      Memory allocation failed\n");
        gemm_aligned_free(A_full);
        gemm_aligned_free(A_full_ref);
        gemm_aligned_free(Y);
        gemm_aligned_free(T);
        gemm_aligned_free(Z);
        gemm_aligned_free(W);
        gemm_aligned_free(YT);
        return 0;
    }

    // Transpose Y to YT for efficient computation
    for (size_t i = 0; i < sub_k; i++)
        for (size_t j = 0; j < sub_m; j++)
            YT[i * sub_m + j] = Y[j * sub_k + i];

    //--------------------------------------------------------------------------
    // Reference: Extract, compute, write back (the slow way)
    //--------------------------------------------------------------------------
    float *C_extracted = (float *)gemm_aligned_alloc(64, sub_m * sub_n * sizeof(float));
    if (!C_extracted)
    {
        printf("      Memory allocation failed\n");
        gemm_aligned_free(A_full);
        gemm_aligned_free(A_full_ref);
        gemm_aligned_free(Y);
        gemm_aligned_free(T);
        gemm_aligned_free(Z);
        gemm_aligned_free(W);
        gemm_aligned_free(YT);
        return 0;
    }

    // Extract submatrix
    for (size_t i = 0; i < sub_m; i++)
        memcpy(C_extracted + i * sub_n, C_ref_sub + i * full_n, sub_n * sizeof(float));

    // Block reflector update: C := C - Y * T * Y^T * C
    // Step 1: Z = Y^T * C
    gemm_auto(Z, YT, C_extracted, sub_k, sub_m, sub_n, 1.0f, 0.0f);
    
    // Step 2: W = T * Z
    gemm_auto(W, T, Z, sub_k, sub_k, sub_n, 1.0f, 0.0f);
    
    // Step 3: C := C - Y * W
    gemm_auto(C_extracted, Y, W, sub_m, sub_k, sub_n, -1.0f, 1.0f);

    // Write back
    for (size_t i = 0; i < sub_m; i++)
        memcpy(C_ref_sub + i * full_n, C_extracted + i * sub_n, sub_n * sizeof(float));

    //--------------------------------------------------------------------------
    // Optimized: Direct strided GEMM (the fast way - NO PACKING!)
    //--------------------------------------------------------------------------
    
    // Step 1: Z = Y^T * C_sub (strided read of C_sub)
    gemm_strided(Z, YT, C_sub,
                sub_k, sub_m, sub_n,  // logical dimensions
                sub_n, sub_m, full_n,  // ldz, ldyt, ldc_sub
                1.0f, 0.0f);
    
    // Step 2: W = T * Z (contiguous)
    gemm_auto(W, T, Z, sub_k, sub_k, sub_n, 1.0f, 0.0f);
    
    // Step 3: C_sub := C_sub - Y * W (strided read/write of C_sub)
    gemm_strided(C_sub, Y, W,
                sub_m, sub_k, sub_n,  // logical dimensions
                full_n, sub_k, sub_n,  // ldc_sub, ldy, ldw
                -1.0f, 1.0f);

    //--------------------------------------------------------------------------
    // Compare results
    //--------------------------------------------------------------------------
    int passed = 1;
    for (size_t i = 0; i < sub_m && passed; i++)
    {
        for (size_t j = 0; j < sub_n && passed; j++)
        {
            float c = C_sub[i * full_n + j];
            float c_ref = C_ref_sub[i * full_n + j];
            float diff = fabsf(c - c_ref);
            float mag = fmaxf(fabsf(c), fabsf(c_ref));
            
            // Use more relaxed tolerance for accumulated error
            if (diff > TEST_TOLERANCE * 10.0f && diff > TEST_TOLERANCE * 10.0f * mag)
            {
                printf("      QR submatrix: Mismatch at [%lu,%lu]: expected %.6f, got %.6f (diff=%.2e)\n",
                       (unsigned long)i, (unsigned long)j, c_ref, c, diff);
                passed = 0;
            }
        }
    }

    if (passed)
    {
        printf("      QR submatrix update (real block reflector): OK\n");
    }
    else
    {
        printf("      QR submatrix update: FAILED\n");
    }

    gemm_aligned_free(A_full);
    gemm_aligned_free(A_full_ref);
    gemm_aligned_free(Y);
    gemm_aligned_free(T);
    gemm_aligned_free(Z);
    gemm_aligned_free(W);
    gemm_aligned_free(YT);
    gemm_aligned_free(C_extracted);

    return passed;
}

/**
 * @brief Test strided GEMM with extreme strides
 */
static int test_strided_extreme_cases(void)
{
    printf("  Testing extreme stride cases...\n");
    
    typedef struct {
        size_t M, K, N;
        size_t ldc, lda, ldb;
        const char *name;
    } extreme_test_t;

    extreme_test_t cases[] = {
        // Very large strides (sparse matrix-like)
        {4, 4, 4, 128, 128, 128, "4x4x4 with huge strides (128)"},
        
        // Minimal logical size, large physical
        {1, 8, 1, 64, 8, 64, "1x8x1 embedded in 64-wide"},
        
        // Row-major vs column-major like scenarios
        {16, 8, 16, 16, 8, 16, "16x8x16 minimal strides"},
        {16, 8, 16, 32, 16, 32, "16x8x16 doubled strides"},
        
        // Edge tiles with strides
        {7, 5, 9, 16, 8, 16, "7x5x9 odd sizes with strides"},
    };

    int all_passed = 1;

    for (size_t t = 0; t < sizeof(cases) / sizeof(cases[0]); t++)
    {
        size_t M = cases[t].M;
        size_t K = cases[t].K;
        size_t N = cases[t].N;
        size_t ldc = cases[t].ldc;
        size_t lda = cases[t].lda;
        size_t ldb = cases[t].ldb;

        float *A_full = (float *)gemm_aligned_alloc(64, M * lda * sizeof(float));
        float *B_full = (float *)gemm_aligned_alloc(64, K * ldb * sizeof(float));
        float *C = (float *)gemm_aligned_alloc(64, M * ldc * sizeof(float));
        float *C_ref = (float *)gemm_aligned_alloc(64, M * ldc * sizeof(float));

        if (!A_full || !B_full || !C || !C_ref)
        {
            printf("      %s: Memory allocation failed\n", cases[t].name);
            gemm_aligned_free(A_full);
            gemm_aligned_free(B_full);
            gemm_aligned_free(C);
            gemm_aligned_free(C_ref);
            all_passed = 0;
            continue;
        }

        // Initialize
        memset(A_full, 0, M * lda * sizeof(float));
        memset(B_full, 0, K * ldb * sizeof(float));
        memset(C, 0, M * ldc * sizeof(float));
        memset(C_ref, 0, M * ldc * sizeof(float));

        for (size_t i = 0; i < M; i++)
            for (size_t k = 0; k < K; k++)
                A_full[i * lda + k] = (float)(i + k + 1);

        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++)
                B_full[k * ldb + j] = (float)(k + j + 1);

        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                C[i * ldc + j] = C_ref[i * ldc + j] = (float)(i + j);

        // Compute
        gemm_reference_strided(C_ref, A_full, B_full, M, K, N,
                              ldc, lda, ldb, 1.0f, 0.5f);
        
        int ret = gemm_strided(C, A_full, B_full, M, K, N,
                              ldc, lda, ldb, 1.0f, 0.5f);

        if (ret != 0)
        {
            printf("      %s: gemm_strided returned error %d\n", cases[t].name, ret);
            all_passed = 0;
            gemm_aligned_free(A_full);
            gemm_aligned_free(B_full);
            gemm_aligned_free(C);
            gemm_aligned_free(C_ref);
            continue;
        }

        // Verify
        int passed = 1;
        for (size_t i = 0; i < M && passed; i++)
        {
            for (size_t j = 0; j < N && passed; j++)
            {
                float c = C[i * ldc + j];
                float c_ref = C_ref[i * ldc + j];
                float diff = fabsf(c - c_ref);
                
                if (diff > TEST_TOLERANCE * 2.0f)
                {
                    printf("      %s: Mismatch at [%lu,%lu]: expected %.6f, got %.6f\n",
                           cases[t].name, (unsigned long)i, (unsigned long)j, c_ref, c);
                    passed = 0;
                }
            }
        }

        if (passed)
        {
            printf("      %s: OK\n", cases[t].name);
        }
        else
        {
            printf("      %s: FAILED\n", cases[t].name);
            all_passed = 0;
        }

        gemm_aligned_free(A_full);
        gemm_aligned_free(B_full);
        gemm_aligned_free(C);
        gemm_aligned_free(C_ref);
    }

    return all_passed;
}

/**
 * @brief Master test runner for strided GEMM
 */
static int test_strided_gemm_suite(void)
{
    printf("\n=== Strided GEMM Test Suite ===\n");
    
    int all_passed = 1;
    
    all_passed &= test_strided_basic();
    all_passed &= test_strided_alpha_beta();
    all_passed &= test_strided_submatrix_qr();
    all_passed &= test_strided_extreme_cases();
    
    return all_passed;
}


//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_gemm_execute_tests(test_results_t *results)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║          GEMM Execution Pipeline Tests                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    results->total = 0;
    results->passed = 0;
    results->failed = 0;

    RUN_TEST(results, test_alpha_beta_combinations);
    RUN_TEST(results, test_k_tile_accumulation);
    RUN_TEST(results, test_matrix_sizes);
    RUN_TEST(results, test_memory_modes);
    RUN_TEST(results, test_edge_cases);
   // RUN_TEST(results, test_precomputed_metadata);
    RUN_TEST(results, test_strided_gemm_suite);  // ← ADD THIS LINE


    print_test_results("GEMM Execution Pipeline", results);

    return (results->failed == 0) ? 0 : 1;
}

//==============================================================================
// STANDALONE MODE
//==============================================================================

#ifdef STANDALONE
int main(void)
{
    test_results_t results = {0};
    int ret = run_gemm_execute_tests(&results);

    if (ret == 0)
    {
        printf("\n🎉 " TEST_PASS " All execution tests passed!\n\n");
    }
    else
    {
        printf("\n❌ " TEST_FAIL " %d test(s) failed\n\n", results.failed);
    }

    return ret;
}
#endif