/**
 * @file gemm_static.h
 * @brief Thread-local static memory pool (FIXED: 64-byte aligned)
 */

#ifndef GEMM_STATIC_H
#define GEMM_STATIC_H

#include <stddef.h>
#include <stdint.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#ifndef GEMM_STATIC_MAX_DIM
#define GEMM_STATIC_MAX_DIM 512
#endif

#define GEMM_STATIC_POOL_SIZE (GEMM_STATIC_MAX_DIM * GEMM_STATIC_MAX_DIM * sizeof(float))

//==============================================================================
// STATIC POOL STRUCTURE (FIXED: 64-byte alignment)
//==============================================================================

typedef struct {
#if defined(_MSC_VER)
    __declspec(align(64)) float workspace[GEMM_STATIC_MAX_DIM * GEMM_STATIC_MAX_DIM];
#else
    float workspace[GEMM_STATIC_MAX_DIM * GEMM_STATIC_MAX_DIM] __attribute__((aligned(64)));
#endif
    int initialized;
} gemm_static_pool_t;

// Compile-time size check
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(
    GEMM_STATIC_POOL_SIZE == sizeof(((gemm_static_pool_t*)0)->workspace),
    "GEMM_STATIC_POOL_SIZE must match workspace array size"
);
#endif

//==============================================================================
// GLOBAL THREAD-LOCAL POOL (FIXED: 64-byte aligned)
//==============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
    extern __thread gemm_static_pool_t gemm_static_pool __attribute__((aligned(64)));
#elif defined(_MSC_VER)
    extern __declspec(align(64)) __declspec(thread) gemm_static_pool_t gemm_static_pool;
#else
    #error "No thread-local storage support"
#endif

//==============================================================================
// API FUNCTIONS
//==============================================================================

void gemm_static_init(void);

static inline int gemm_fits_static(size_t M, size_t K, size_t N) {
    return (M <= GEMM_STATIC_MAX_DIM && 
            K <= GEMM_STATIC_MAX_DIM && 
            N <= GEMM_STATIC_MAX_DIM);
}

static inline int gemm_workspace_fits_static(size_t workspace_bytes) {
    return workspace_bytes <= GEMM_STATIC_POOL_SIZE;
}

static inline size_t gemm_calc_workspace_size(size_t M, size_t K, size_t N) {
    size_t ws_a = M * K * sizeof(float);
    size_t ws_b = K * N * sizeof(float);
    return ws_a + ws_b;
}

static inline float* gemm_get_static_workspace(void) {
    if (!gemm_static_pool.initialized) {
        gemm_static_init();
    }
    return gemm_static_pool.workspace;
}

static inline int gemm_get_static_limit(void) {
    return GEMM_STATIC_MAX_DIM;
}

#ifdef __cplusplus
}
#endif

#endif // GEMM_STATIC_H