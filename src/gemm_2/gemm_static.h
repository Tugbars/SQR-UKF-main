/**
 * @file gemm_static.h
 * @brief Thread-local static memory pool (FIXED: 64-byte aligned)
 * 
 * Static pool is DISABLED by default. To enable, define:
 *   #define GEMM_ENABLE_STATIC_POOL 1
 * before including this header, or pass -DGEMM_ENABLE_STATIC_POOL=1 to compiler.
 */

#ifndef GEMM_STATIC_H
#define GEMM_STATIC_H

#include <stddef.h>
#include <stdint.h>

//==============================================================================
// BUILD CONFIGURATION
//==============================================================================

// Static pool is DISABLED by default - always use dynamic allocation
// To enable static pool, define GEMM_ENABLE_STATIC_POOL=1
#ifndef GEMM_ENABLE_STATIC_POOL
#define GEMM_ENABLE_STATIC_POOL 0
#endif

//==============================================================================
// CONFIGURATION (only relevant if static pool is enabled)
//==============================================================================

#ifndef GEMM_STATIC_MAX_DIM
#define GEMM_STATIC_MAX_DIM 64
#endif

#define GEMM_STATIC_POOL_SIZE (GEMM_STATIC_MAX_DIM * GEMM_STATIC_MAX_DIM * sizeof(float))

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// STATIC POOL IMPLEMENTATION (only when enabled)
//==============================================================================

#if GEMM_ENABLE_STATIC_POOL

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

#if defined(__GNUC__) || defined(__clang__)
    extern __thread gemm_static_pool_t gemm_static_pool __attribute__((aligned(64)));
#elif defined(_MSC_VER)
    extern __declspec(align(64)) __declspec(thread) gemm_static_pool_t gemm_static_pool;
#else
    #error "No thread-local storage support"
#endif

//==============================================================================
// API FUNCTIONS (static pool enabled)
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

#else // !GEMM_ENABLE_STATIC_POOL

//==============================================================================
// STUB IMPLEMENTATION (static pool disabled - always use dynamic)
//==============================================================================

// Dummy type for compilation compatibility
typedef struct {
    int initialized;
} gemm_static_pool_t;

// Always return false - force dynamic allocation
static inline int gemm_fits_static(size_t M, size_t K, size_t N) {
    (void)M; (void)K; (void)N;
    return 0;  // Never fits - always use dynamic
}

static inline int gemm_workspace_fits_static(size_t workspace_bytes) {
    (void)workspace_bytes;
    return 0;  // Never fits - always use dynamic
}

static inline size_t gemm_calc_workspace_size(size_t M, size_t K, size_t N) {
    size_t ws_a = M * K * sizeof(float);
    size_t ws_b = K * N * sizeof(float);
    return ws_a + ws_b;
}

static inline float* gemm_get_static_workspace(void) {
    return NULL;  // No static workspace available
}

static inline int gemm_get_static_limit(void) {
    return 0;  // No static limit
}

// No-op init when disabled
static inline void gemm_static_init(void) {}

#endif // GEMM_ENABLE_STATIC_POOL

#ifdef __cplusplus
}
#endif

#endif // GEMM_STATIC_H