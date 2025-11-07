#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include "../gemm/gemm.h"

// Test function declarations
void test_kernels_unit(void);
void test_reference_small(void);
void test_reference_large(void);
void run_benchmark_suite(void);

int main(void) {
    printf("=== GEMM Test Suite ===\n");
    printf("Testing AVX2 kernels on Intel i9-14900\n\n");
    
    srand((unsigned)time(NULL));
    
    // Phase 1: Unit tests (no OpenBLAS needed)
    printf("Phase 1: Kernel unit tests\n");
    printf("---------------------------\n");
    //test_kernels_unit();
    
    // Phase 2: Small matrix correctness
    printf("\n\nPhase 2: Small matrices vs OpenBLAS\n");
    printf("------------------------------------\n");
    test_reference_small();
    
    // Phase 3: Large matrices
    printf("\n\nPhase 3: Large matrices\n");
    printf("-----------------------\n");
    test_reference_large();

     printf("\n\nPhase 4: Benchmarks (optional)\n");
    printf("------------------------------\n");
    run_benchmark_suite();          // or bench_case(256,256,256, 15, 1);


    printf("\n\n=== All tests completed! ===\n");
    return 0;
}