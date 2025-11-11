/**
 * @file test_common.h
 * @brief Common test utilities and macros
 */

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>

//==============================================================================
// TEST UTILITIES
//==============================================================================

#define TEST_PASS "\033[0;32m[PASS]\033[0m"
#define TEST_FAIL "\033[0;31m[FAIL]\033[0m"
#define TEST_INFO "\033[0;34m[INFO]\033[0m"

// Global test counters (each test suite maintains its own)
typedef struct {
    int total;
    int passed;
    int failed;
} test_results_t;

#define RUN_TEST(suite, test_func)                               \
    do                                                           \
    {                                                            \
        printf("\n" TEST_INFO " Running: %s\n", #test_func);    \
        (suite)->total++;                                        \
        if (test_func())                                         \
        {                                                        \
            (suite)->passed++;                                   \
            printf(TEST_PASS " %s\n", #test_func);               \
        }                                                        \
        else                                                     \
        {                                                        \
            (suite)->failed++;                                   \
            printf(TEST_FAIL " %s\n", #test_func);               \
        }                                                        \
    } while (0)

/**
 * @brief Print test suite results banner
 */
static inline void print_test_results(const char *suite_name, const test_results_t *results)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  %-57s║\n", suite_name);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Total:  %3d                                               ║\n", results->total);
    printf("║  Passed: %3d                                               ║\n", results->passed);
    printf("║  Failed: %3d                                               ║\n", results->failed);
    printf("╚═══════════════════════════════════════════════════════════╝\n");
}

#endif // TEST_COMMON_H