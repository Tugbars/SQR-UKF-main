/**
 * @file gemm_static.c
 * @brief Static pool implementation (FIXED: 64-byte alignment)
 */

#include "gemm_static.h"
#include <string.h>

//==============================================================================
// THREAD-LOCAL STORAGE (FIXED: 64-byte aligned)
//==============================================================================

#if defined(__GNUC__) || defined(__clang__)
    __thread gemm_static_pool_t gemm_static_pool __attribute__((aligned(64))) = {0};
#elif defined(_MSC_VER)
    __declspec(align(64)) __declspec(thread) gemm_static_pool_t gemm_static_pool = {0};
#else
    #error "No thread-local storage support"
#endif

//==============================================================================
// INITIALIZATION
//==============================================================================

void gemm_static_init(void) {
    if (gemm_static_pool.initialized) {
        return;
    }
    
    // Workspace already zero-initialized by compiler
    gemm_static_pool.initialized = 1;
}