/**
 * @file test_common.h
 * @brief Common definitions for QR test suite
 */

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Test results structure
 */
typedef struct
{
    int total;
    int passed;
    int failed;
} test_results_t;

/**
 * @brief Test function prototype
 */
typedef int (*test_function_t)(test_results_t *);

#endif /* TEST_COMMON_H */