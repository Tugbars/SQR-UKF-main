/**
 * @file test_main.c
 * @brief QR test suite driver
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// Forward declarations
int test_workspace(void);
int test_correctness(void);
int test_cpqr(void);
int test_benchmark(void);

int main(int argc, char **argv)
{
    printf("===========================================\n");
    printf("      VectorFFT QR Test Suite\n");
    printf("===========================================\n\n");

    int failures = 0;

    printf(">>> Running workspace tests...\n");
    failures += test_workspace();

    printf("\n>>> Running correctness tests...\n");
    failures += test_correctness();

   // printf("\n>>> Running CPQR tests...\n");
   // failures += test_cpqr();

    //if (argc > 1 && strcmp(argv[1], "--bench") == 0)
   // {
   //     printf("\n>>> Running benchmarks...\n");
  //      test_benchmark();
  //  }

    printf("\n===========================================\n");
    if (failures == 0)
    {
        printf("✅ ALL TESTS PASSED!\n");
    }
    else
    {
        printf("❌ %d TEST(S) FAILED!\n", failures);
    }
    printf("===========================================\n");

    return (failures == 0) ? 0 : 1;
}