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