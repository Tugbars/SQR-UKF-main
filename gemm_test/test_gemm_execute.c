/**
 * @file test_gemm_execute.c
 * @brief Tests for gemm_large.c execution pipeline
 *
 * Tests:
 * - SIMD packing correctness
 * - Alpha/beta scaling
 * - Full gemm_execute_plan
 * - Different matrix sizes
 * - Edge cases (partial tiles, different blocking)
 * - Pre-computed metadata optimization
 */

#include "test_common.h"
#include "gemm.h"
#include "gemm_planning.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Portable size_t printf format
#define FMT_SIZE_T "%lu"
#define CAST_SIZE_T(x) ((unsigned long)(x))

//==============================================================================
// REFERENCE IMPLEMENTATIONS
//==============================================================================

/**
 * @brief Reference GEMM: C = alpha*A*B + beta*C
 */
static void ref_gemm(
    float *C, size_t ldc,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    // Beta scaling
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            C[i * ldc + j] *= beta;
        }
    }

    // Alpha * A * B
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

//==============================================================================
// COMPARISON HELPERS
//==============================================================================

static int compare_matrices(
    const float *test,
    const float *ref,
    size_t M, size_t N, size_t ldc,
    float tol,
    const char *test_name)
{
    int errors = 0;
    float max_error = 0.0f;
    size_t error_i = 0, error_j = 0;

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
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
                if (errors <= 3)
                {
                    printf("    ERROR at [" FMT_SIZE_T "," FMT_SIZE_T "]: test=%.6f, ref=%.6f, diff=%.6f\n",
                           CAST_SIZE_T(i), CAST_SIZE_T(j), t, r, err);
                }
            }
        }
    }

    if (errors > 0)
    {
        printf("  %s FAILED: %d errors, max error %.6e at [" FMT_SIZE_T "," FMT_SIZE_T "]\n",
               test_name, errors, max_error, CAST_SIZE_T(error_i), CAST_SIZE_T(error_j));
        return 0;
    }

    printf("  %s PASSED (max error: %.6e)\n", test_name, max_error);
    return 1;
}

//==============================================================================
// TEST: Beta Scaling
//==============================================================================

static int test_beta_scaling(void)
{
    printf("\n=== Testing Beta Pre-Scaling ===\n");

    const size_t M = 64, N = 64;
    int passed = 1;

    float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

    // Initialize C with known values
    for (size_t i = 0; i < M * N; i++)
    {
        C_test[i] = (float)(i % 10) * 0.1f;
        C_ref[i] = C_test[i];
    }

    // Test beta=0 (should zero out)
    {
        printf("  Testing beta=0...\n");

        float *A = gemm_aligned_alloc(64, M * M * sizeof(float));
        float *B = gemm_aligned_alloc(64, M * N * sizeof(float));

        for (size_t i = 0; i < M * M; i++)
            A[i] = 1.0f;
        for (size_t i = 0; i < M * N; i++)
            B[i] = 1.0f;

        gemm_auto(C_test, A, B, M, M, N, 1.0f, 0.0f);

        // Reference
        for (size_t i = 0; i < M * N; i++)
            C_ref[i] = 0.0f;
        ref_gemm(C_ref, N, A, M, B, N, M, M, N, 1.0f, 0.0f);

        passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-4f, "beta=0");

        gemm_aligned_free(A);
        gemm_aligned_free(B);
    }

    // Test beta=0.5
    {
        printf("  Testing beta=0.5...\n");

        for (size_t i = 0; i < M * N; i++)
        {
            C_test[i] = 1.0f;
            C_ref[i] = 1.0f;
        }

        float *A = gemm_aligned_alloc(64, M * M * sizeof(float));
        float *B = gemm_aligned_alloc(64, M * N * sizeof(float));

        for (size_t i = 0; i < M * M; i++)
            A[i] = 0.1f;
        for (size_t i = 0; i < M * N; i++)
            B[i] = 0.1f;

        gemm_auto(C_test, A, B, M, M, N, 1.0f, 0.5f);
        ref_gemm(C_ref, N, A, M, B, N, M, M, N, 1.0f, 0.5f);

        passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-4f, "beta=0.5");

        gemm_aligned_free(A);
        gemm_aligned_free(B);
    }

    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return passed;
}

//==============================================================================
// TEST: Alpha Scaling
//==============================================================================

static int test_alpha_scaling(void)
{
    printf("\n=== Testing Alpha Scaling ===\n");

    const size_t M = 64, K = 64, N = 64;
    int passed = 1;

    float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

    // Initialize with known pattern
    for (size_t i = 0; i < M * K; i++)
        A[i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (float)(i % 5) * 0.1f;

    // Test alpha=2.0
    {
        printf("  Testing alpha=2.0...\n");

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        gemm_auto(C_test, A, B, M, K, N, 2.0f, 0.0f);
        ref_gemm(C_ref, N, A, K, B, N, M, K, N, 2.0f, 0.0f);

        passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-4f, "alpha=2.0");
    }

    // Test alpha=0.5
    {
        printf("  Testing alpha=0.5...\n");

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        gemm_auto(C_test, A, B, M, K, N, 0.5f, 0.0f);
        ref_gemm(C_ref, N, A, K, B, N, M, K, N, 0.5f, 0.0f);

        passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-4f, "alpha=0.5");
    }

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return passed;
}

//==============================================================================
// TEST: Different Matrix Sizes
//==============================================================================

static int test_matrix_sizes(void)
{
    printf("\n=== Testing Various Matrix Sizes ===\n");

    int passed = 1;

    struct
    {
        size_t M, K, N;
        const char *name;
        float tol;
    } test_cases[] = {
        // Small (Tier 1 should handle)
        {8, 8, 8, "8x8x8 (Tier 1)", 1e-5f},
        {16, 16, 16, "16x16x16 (Tier 1)", 1e-5f},

        // Medium (Tier 2, single block)
        {64, 64, 64, "64x64x64", 1e-4f},
        {128, 128, 128, "128x128x128", 5e-4f},

        // Tall matrices
        {256, 64, 64, "256x64x64 (tall)", 5e-4f},
        {512, 32, 32, "512x32x32 (very tall)", 1e-3f},

        // Wide matrices
        {64, 64, 256, "64x64x256 (wide)", 5e-4f},
        {32, 32, 512, "32x32x512 (very wide)", 1e-3f},

        // Deep matrices (large K)
        {64, 256, 64, "64x256x64 (deep)", 5e-4f},
        {32, 512, 32, "32x512x32 (very deep)", 1e-3f},

        // Non-power-of-2
        {100, 100, 100, "100x100x100", 5e-4f},
        {123, 77, 91, "123x77x91 (irregular)", 5e-4f},

        // Partial tiles
        {67, 53, 89, "67x53x89 (many partials)", 5e-4f},
    };

    for (size_t tc = 0; tc < sizeof(test_cases) / sizeof(test_cases[0]); tc++)
    {
        size_t M = test_cases[tc].M;
        size_t K = test_cases[tc].K;
        size_t N = test_cases[tc].N;

        printf("  Testing %s...\n", test_cases[tc].name);

        float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
        float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
        float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
        float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

        // Initialize with reproducible pattern
        srand(42 + (unsigned int)tc);
        for (size_t i = 0; i < M * K; i++)
        {
            A[i] = ((float)(rand() % 1000) / 1000.0f) - 0.5f;
        }
        for (size_t i = 0; i < K * N; i++)
        {
            B[i] = ((float)(rand() % 1000) / 1000.0f) - 0.5f;
        }

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        // Test with alpha=1, beta=0
        gemm_auto(C_test, A, B, M, K, N, 1.0f, 0.0f);
        ref_gemm(C_ref, N, A, K, B, N, M, K, N, 1.0f, 0.0f);

        if (!compare_matrices(C_test, C_ref, M, N, N, test_cases[tc].tol, test_cases[tc].name))
        {
            passed = 0;
            // Continue testing other sizes
        }

        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
    }

    return passed;
}

//==============================================================================
// TEST: Alpha/Beta Combinations
//==============================================================================

static int test_alpha_beta_combinations(void)
{
    printf("\n=== Testing Alpha/Beta Combinations ===\n");

    const size_t M = 64, K = 64, N = 64;
    int passed = 1;

    float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
    float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
    float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
    float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

    // Initialize
    for (size_t i = 0; i < M * K; i++)
        A[i] = (float)(i % 11) * 0.1f;
    for (size_t i = 0; i < K * N; i++)
        B[i] = (float)(i % 13) * 0.1f;

    struct
    {
        float alpha, beta;
        const char *name;
    } cases[] = {
        {1.0f, 0.0f, "α=1, β=0"},
        {1.0f, 1.0f, "α=1, β=1"},
        {2.0f, 0.0f, "α=2, β=0"},
        {0.5f, 0.5f, "α=0.5, β=0.5"},
        {-1.0f, 1.0f, "α=-1, β=1"},
        {1.0f, -0.5f, "α=1, β=-0.5"},
        {0.0f, 1.0f, "α=0, β=1 (should zero A*B)"},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        printf("  Testing %s...\n", cases[i].name);

        // Initialize C with pattern
        for (size_t j = 0; j < M * N; j++)
        {
            C_test[j] = (float)(j % 7) * 0.1f;
            C_ref[j] = C_test[j];
        }

        gemm_auto(C_test, A, B, M, K, N, cases[i].alpha, cases[i].beta);
        ref_gemm(C_ref, N, A, K, B, N, M, K, N, cases[i].alpha, cases[i].beta);

        if (!compare_matrices(C_test, C_ref, M, N, N, 1e-4f, cases[i].name))
        {
            passed = 0;
        }
    }

    gemm_aligned_free(A);
    gemm_aligned_free(B);
    gemm_aligned_free(C_test);
    gemm_aligned_free(C_ref);

    return passed;
}

//==============================================================================
// TEST: Memory Modes (Static vs Dynamic)
//==============================================================================

static int test_memory_modes(void)
{
    printf("\n=== Testing Memory Modes ===\n");

    int passed = 1;

    // Test 1: Small matrix (should use static pool)
    {
        printf("  Testing static pool (32x32x32)...\n");

        const size_t M = 32, K = 32, N = 32;

        float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
        float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
        float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
        float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

        for (size_t i = 0; i < M * K; i++)
            A[i] = 0.1f;
        for (size_t i = 0; i < K * N; i++)
            B[i] = 0.1f;

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        // Force static mode
        int ret = gemm_static(C_test, A, B, M, K, N, 1.0f, 0.0f);
        if (ret != 0)
        {
            printf("    Static mode failed (should succeed for 32x32x32)\n");
            passed = 0;
        }
        else
        {
            ref_gemm(C_ref, N, A, K, B, N, M, K, N, 1.0f, 0.0f);
            passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-5f, "static pool");
        }

        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
    }

    // Test 2: Large matrix (must use dynamic)
    {
        printf("  Testing dynamic allocation (256x256x256)...\n");

        const size_t M = 256, K = 256, N = 256;

        float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
        float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
        float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
        float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

        for (size_t i = 0; i < M * K; i++)
            A[i] = (float)(i % 7) * 0.01f;
        for (size_t i = 0; i < K * N; i++)
            B[i] = (float)(i % 5) * 0.01f;

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        // Force dynamic mode
        int ret = gemm_dynamic(C_test, A, B, M, K, N, 1.0f, 0.0f);
        if (ret != 0)
        {
            printf("    Dynamic mode failed\n");
            passed = 0;
        }
        else
        {
            ref_gemm(C_ref, N, A, K, B, N, M, K, N, 1.0f, 0.0f);
            passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-3f, "dynamic alloc");
        }

        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
    }

    return passed;
}

//==============================================================================
// TEST: Edge Cases
//==============================================================================

static int test_edge_cases(void)
{
    printf("\n=== Testing Edge Cases ===\n");

    int passed = 1;

    // Test 1: Single element matrix
    {
        printf("  Testing 1x1x1 matrix...\n");

        float A = 2.0f;
        float B = 3.0f;
        float C_test = 0.0f;
        float C_ref = 0.0f;

        gemm_auto(&C_test, &A, &B, 1, 1, 1, 1.0f, 0.0f);
        C_ref = 2.0f * 3.0f;

        if (fabsf(C_test - C_ref) < 1e-6f)
        {
            printf("    1x1x1 PASSED\n");
        }
        else
        {
            printf("    1x1x1 FAILED: got %.6f, expected %.6f\n", C_test, C_ref);
            passed = 0;
        }
    }

    // Test 2: Prime dimensions
    {
        printf("  Testing prime dimensions (7x11x13)...\n");

        const size_t M = 7, K = 11, N = 13;

        float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
        float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
        float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
        float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

        for (size_t i = 0; i < M * K; i++)
            A[i] = 1.0f;
        for (size_t i = 0; i < K * N; i++)
            B[i] = 1.0f;

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        gemm_auto(C_test, A, B, M, K, N, 1.0f, 0.0f);
        ref_gemm(C_ref, N, A, K, B, N, M, K, N, 1.0f, 0.0f);

        passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-5f, "prime dims");

        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
    }

    // Test 3: Very thin matrices
    {
        printf("  Testing thin matrix (128x128x1)...\n");

        const size_t M = 128, K = 128, N = 1;

        float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
        float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
        float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
        float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

        for (size_t i = 0; i < M * K; i++)
            A[i] = 0.1f;
        for (size_t i = 0; i < K * N; i++)
            B[i] = 0.1f;

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        gemm_auto(C_test, A, B, M, K, N, 1.0f, 0.0f);
        ref_gemm(C_ref, N, A, K, B, N, M, K, N, 1.0f, 0.0f);

        passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-4f, "thin 128x128x1");

        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
    }

    return passed;
}

//==============================================================================
// TEST: Pre-Computed Metadata Optimization (NEW!)
//==============================================================================

static int test_precomputed_metadata(void)
{
    printf("\n=== Testing Pre-Computed Metadata Optimization ===\n");

    int passed = 1;

    // Test that plan contains pre-computed values
    {
        printf("  Verifying plan metadata for 256x256x256...\n");

        gemm_plan_t *plan = gemm_plan_create(256, 256, 256);
        if (!plan)
        {
            printf("    FAIL: Plan creation failed\n");
            return 0;
        }

        // Verify tile counts are pre-computed
        size_t expected_nc = (256 + plan->NC - 1) / plan->NC;
        size_t expected_kc = (256 + plan->KC - 1) / plan->KC;
        size_t expected_mc = (256 + plan->MC - 1) / plan->MC;

        if (plan->n_nc_tiles != expected_nc ||
            plan->n_kc_tiles != expected_kc ||
            plan->n_mc_tiles != expected_mc)
        {
            printf("    FAIL: Tile counts not pre-computed correctly\n");
            gemm_plan_destroy(plan);
            return 0;
        }

        printf("    Pre-computed tile counts: nc=" FMT_SIZE_T ", kc=" FMT_SIZE_T ", mc=" FMT_SIZE_T " ✓\n",
               CAST_SIZE_T(plan->n_nc_tiles), CAST_SIZE_T(plan->n_kc_tiles), CAST_SIZE_T(plan->n_mc_tiles));

        // Verify kernels are pre-selected
        if (plan->kern_full_add == KERN_INVALID || plan->kern_full_store == KERN_INVALID)
        {
            printf("    FAIL: Kernels not pre-selected\n");
            gemm_plan_destroy(plan);
            return 0;
        }

        printf("    Pre-selected kernels: add=%d, store=%d ✓\n",
               plan->kern_full_add, plan->kern_full_store);

        gemm_plan_destroy(plan);
    }

    // Test that execution uses pre-computed values (correctness check)
    {
        printf("  Verifying execution with pre-computed metadata...\n");

        const size_t M = 256, K = 256, N = 256;

        float *A = gemm_aligned_alloc(64, M * K * sizeof(float));
        float *B = gemm_aligned_alloc(64, K * N * sizeof(float));
        float *C_test = gemm_aligned_alloc(64, M * N * sizeof(float));
        float *C_ref = gemm_aligned_alloc(64, M * N * sizeof(float));

        for (size_t i = 0; i < M * K; i++)
            A[i] = (float)(i % 7) * 0.01f;
        for (size_t i = 0; i < K * N; i++)
            B[i] = (float)(i % 5) * 0.01f;

        memset(C_test, 0, M * N * sizeof(float));
        memset(C_ref, 0, M * N * sizeof(float));

        gemm_auto(C_test, A, B, M, K, N, 1.0f, 0.0f);
        ref_gemm(C_ref, N, A, K, B, N, M, K, N, 1.0f, 0.0f);

        passed &= compare_matrices(C_test, C_ref, M, N, N, 1e-3f, "with metadata");

        gemm_aligned_free(A);
        gemm_aligned_free(B);
        gemm_aligned_free(C_test);
        gemm_aligned_free(C_ref);
    }

    return passed;
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

    RUN_TEST(results, test_beta_scaling);
    RUN_TEST(results, test_alpha_scaling);
    RUN_TEST(results, test_matrix_sizes);
    RUN_TEST(results, test_alpha_beta_combinations);
    RUN_TEST(results, test_memory_modes);
    RUN_TEST(results, test_edge_cases);
    RUN_TEST(results, test_precomputed_metadata);

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