/**
 * @file test_main.c
 * @brief Unified test runner for all GEMM test suites
 */

#include "test_common.h"
#include <stdio.h>
#include <string.h>

// Test suite runners
extern int run_gemm_small_tests(test_results_t *results);
extern int run_gemm_planning_tests(test_results_t *results);
extern int run_gemm_kernel_tests(test_results_t *results);  // ← In test_gemm_large.c

int main(int argc, char **argv)
{
    test_results_t small_results = {0};
    test_results_t planning_results = {0};
    test_results_t kernel_results = {0};
    test_results_t total_results = {0};

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║             GEMM Library - Full Test Suite               ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    // Parse command-line arguments
    int run_small = 1;
    int run_planning = 1;
    int run_kernels = 1;

    if (argc > 1) {
        run_small = 0;
        run_planning = 0;
        run_kernels = 0;
        
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "small") == 0) {
                run_small = 1;
            } else if (strcmp(argv[i], "planning") == 0) {
                run_planning = 1;
            } else if (strcmp(argv[i], "kernels") == 0) {
                run_kernels = 1;
            } else if (strcmp(argv[i], "all") == 0) {
                run_small = 1;
                run_planning = 1;
                run_kernels = 1;
            } else {
                printf("Unknown test suite: %s\n", argv[i]);
                printf("Usage: %s [small|planning|kernels|all]\n", argv[0]);
                return 1;
            }
        }
    }

    //==========================================================================
    // Run test suites
    //==========================================================================

    if (run_small) {
        printf("\n");
        printf("════════════════════════════════════════════════════════════\n");
        printf(" Running: Tier 1 (Small Kernels) Test Suite\n");
        printf("════════════════════════════════════════════════════════════\n");
        run_gemm_small_tests(&small_results);
    }

    if (run_planning) {
        printf("\n");
        printf("════════════════════════════════════════════════════════════\n");
        printf(" Running: Planning Module Test Suite\n");
        printf("════════════════════════════════════════════════════════════\n");
        run_gemm_planning_tests(&planning_results);
    }

    if (run_kernels) {
        printf("\n");
        printf("════════════════════════════════════════════════════════════\n");
        printf(" Running: Individual Kernel Unit Tests\n");
        printf("════════════════════════════════════════════════════════════\n");
        run_gemm_kernel_tests(&kernel_results);
    }

    //==========================================================================
    // Aggregate results
    //==========================================================================

    total_results.total = small_results.total + planning_results.total + 
                         kernel_results.total;
    total_results.passed = small_results.passed + planning_results.passed + 
                          kernel_results.passed;
    total_results.failed = small_results.failed + planning_results.failed + 
                          kernel_results.failed;

    //==========================================================================
    // Final summary
    //==========================================================================

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                     Test Summary                          ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");

    if (run_small) {
        printf("║  Small Matrices: %3d/%3d passed                           ║\n",
               small_results.passed, small_results.total);
    }
    if (run_planning) {
        printf("║  Planning:       %3d/%3d passed                           ║\n",
               planning_results.passed, planning_results.total);
    }
    if (run_kernels) {
        printf("║  Kernel Units:   %3d/%3d passed                           ║\n",
               kernel_results.passed, kernel_results.total);
    }

    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  TOTAL:          %3d/%3d passed                           ║\n",
           total_results.passed, total_results.total);
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    if (total_results.failed == 0) {
        printf("\n🎉 " TEST_PASS " ALL TESTS PASSED!\n\n");
        return 0;
    } else {
        printf("\n❌ " TEST_FAIL " %d test(s) failed\n\n", total_results.failed);
        
        // Provide guidance
        if (kernel_results.failed > 0) {
            printf("💡 TIP: Fix kernel unit tests first - they're the foundation.\n");
            printf("   Run: ./test_all kernels\n\n");
        }
        
        return 1;
    }
}