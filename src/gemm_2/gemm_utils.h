/**
 * @file gemm_utils.h
 * @brief Utility functions for GEMM library (header-only)
 */

#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include <stdlib.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// TYPE-SAFE MACROS (GNU C / Clang)
//==============================================================================

#if defined(__GNUC__) || defined(__clang__)

/**
 * @brief Type-safe MIN/MAX using GNU C statement expressions
 * 
 * Advantages:
 * - No double evaluation (safe for side effects)
 * - Type preservation
 * - Compile-time type checking
 * 
 * Usage: MIN(a++, b) is safe (a is only incremented once)
 */
#define MIN(a, b) ({ \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b; \
})

#define MAX(a, b) ({ \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b; \
})

/**
 * @brief Three-way clamp
 */
#define CLAMP(x, lo, hi) ({ \
    __typeof__(x) _x = (x); \
    __typeof__(lo) _lo = (lo); \
    __typeof__(hi) _hi = (hi); \
    _x < _lo ? _lo : (_x > _hi ? _hi : _x); \
})

#else

//==============================================================================
// PORTABLE MACROS (Fallback for MSVC, etc.)
//==============================================================================

/**
 * @brief Portable MIN/MAX (careful with side effects!)
 * 
 * WARNING: Arguments evaluated twice - do NOT use with side effects
 * BAD:  MIN(a++, b)  // a is incremented twice!
 * GOOD: MIN(a, b)    // No side effects
 */
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

#endif

//==============================================================================
// ALIGNED MEMORY ALLOCATION
//==============================================================================


static inline void* gemm_aligned_alloc(size_t alignment, size_t size) 
{
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL;  // Alignment must be power of 2
    }
    
    if (size == 0) {
        return NULL;
    }
    
#if defined(_WIN32) || defined(_WIN64)
    return _aligned_malloc(size, alignment);
#elif defined(__APPLE__) || defined(__MACH__)
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#elif __STDC_VERSION__ >= 201112L
    // C11: aligned_alloc
    size_t adjusted_size = ((size + alignment - 1) / alignment) * alignment;
    return aligned_alloc(alignment, adjusted_size);
#else
    // Fallback: manual alignment
    void *raw = malloc(size + alignment + sizeof(void*));
    if (!raw) return NULL;
    
    void *aligned = (void*)(((uintptr_t)raw + sizeof(void*) + alignment - 1) & ~(alignment - 1));
    ((void**)aligned)[-1] = raw;
    return aligned;
#endif
}

static inline void gemm_aligned_free(void* ptr) 
{
    if (!ptr) return;
    
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#elif defined(__APPLE__) || defined(__MACH__)
    free(ptr);
#elif __STDC_VERSION__ >= 201112L
    free(ptr);
#else
    // Fallback: retrieve original pointer
    void *raw = ((void**)ptr)[-1];
    free(raw);
#endif
}

//==============================================================================
// ERROR HANDLING
//==============================================================================

static inline const char* gemm_strerror(int error) 
{
    switch (error) {
        case 0:  // GEMM_OK
            return "Success";
        case -1:  // GEMM_ERR_INVALID_PTR
            return "Invalid pointer (NULL)";
        case -2:  // GEMM_ERR_INVALID_DIM
            return "Invalid matrix dimensions";
        case -3:  // GEMM_ERR_NO_MEMORY
            return "Memory allocation failed";
        case -4:  // GEMM_ERR_OVERFLOW
            return "Integer overflow in size calculation";
        case -5:  // GEMM_ERR_STATIC_TOO_LARGE
            return "Matrix dimensions exceed static pool limit";
        case -6:  // GEMM_ERR_NOT_IMPLEMENTED
            return "Feature not implemented";
        default:
            return "Unknown error";
    }
}

#ifdef __cplusplus
}
#endif

#endif // GEMM_UTILS_H