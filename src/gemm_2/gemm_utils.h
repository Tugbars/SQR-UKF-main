/**
 * @file gemm_utils.h
 * @brief Utility Functions and Macros for GEMM Library
 * 
 * This header provides essential utilities used throughout the GEMM library:
 * - Type-safe MIN/MAX/CLAMP macros (GNU C extensions when available)
 * - Cross-platform aligned memory allocation
 * - Error code to string conversion
 * 
 * **Design Philosophy:**
 * - Header-only (no .c file needed)
 * - Cross-platform compatibility (Windows/Linux/macOS)
 * - Zero dependencies (only standard library)
 * 
 * @author TUGBARS
 * @date 2025
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
 * @brief Type-safe MIN using GNU C statement expressions
 * @note Evaluates arguments once (safe with side effects like MIN(x++, y))
 */
#define MIN(a, b) ({ \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b; \
})

/**
 * @brief Type-safe MAX using GNU C statement expressions
 * @note Evaluates arguments once
 */
#define MAX(a, b) ({ \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b; \
})

/**
 * @brief Type-safe CLAMP: returns x clamped to [lo, hi]
 * @note Evaluates arguments once
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
 * @brief Portable MIN (evaluates arguments twice)
 * @warning Do NOT use with side effects: MIN(x++, y) is unsafe
 */
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/**
 * @brief Portable MAX (evaluates arguments twice)
 * @warning Do NOT use with side effects
 */
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * @brief Portable CLAMP (evaluates x up to three times)
 * @warning Do NOT use with side effects
 */
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

#endif

//==============================================================================
// ALIGNED MEMORY ALLOCATION
//==============================================================================

/**
 * @brief Allocate aligned memory (cross-platform)
 * 
 * Allocates memory aligned to the specified boundary. Required for:
 * - AVX2 operations (32-byte alignment recommended)
 * - Cache line alignment (64-byte optimal)
 * - SIMD performance (unaligned loads ~2-5% slower)
 * 
 * **Platform Support:**
 * - Windows: _aligned_malloc
 * - macOS: posix_memalign
 * - Linux C11: aligned_alloc
 * - Fallback: Manual alignment via malloc
 * 
 * @param alignment Alignment boundary in bytes (must be power of 2)
 * @param size      Number of bytes to allocate
 * 
 * @return Pointer to aligned memory, or NULL on failure
 * 
 * @retval NULL if alignment is not power of 2
 * @retval NULL if size is zero
 * @retval NULL if allocation fails
 * 
 * @note Must be freed with gemm_aligned_free(), not regular free()
 * @note Returns NULL for invalid alignment (not a power of 2)
 * 
 * @see gemm_aligned_free()
 */
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
    // C11: aligned_alloc requires size to be multiple of alignment
    size_t adjusted_size = ((size + alignment - 1) / alignment) * alignment;
    return aligned_alloc(alignment, adjusted_size);
#else
    // Fallback: manual alignment using malloc
    void *raw = malloc(size + alignment + sizeof(void*));
    if (!raw) return NULL;
    
    void *aligned = (void*)(((uintptr_t)raw + sizeof(void*) + alignment - 1) & ~(alignment - 1));
    ((void**)aligned)[-1] = raw;  // Store original pointer for free
    return aligned;
#endif
}

/**
 * @brief Free aligned memory (cross-platform)
 * 
 * Frees memory allocated by gemm_aligned_alloc(). Uses the appropriate
 * platform-specific deallocation function.
 * 
 * @param ptr Pointer to aligned memory (may be NULL)
 * 
 * @note Safe to call with NULL pointer (no-op)
 * @note Must ONLY be used with pointers from gemm_aligned_alloc()
 * @note Using regular free() on aligned memory causes undefined behavior
 * 
 * @see gemm_aligned_alloc()
 */
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
    // Fallback: retrieve original pointer stored before aligned address
    void *raw = ((void**)ptr)[-1];
    free(raw);
#endif
}

//==============================================================================
// ERROR HANDLING
//==============================================================================

/**
 * @brief Convert GEMM error code to human-readable string
 * 
 * Maps integer error codes to descriptive error messages.
 * 
 * **Error Codes:**
 * - 0: Success
 * - -1: Invalid pointer (NULL)
 * - -2: Invalid matrix dimensions
 * - -3: Memory allocation failed
 * - -4: Integer overflow in size calculation
 * - -5: Matrix dimensions exceed static pool limit
 * - -6: Feature not implemented
 * 
 * @param error Error code (typically negative for errors)
 * 
 * @return Pointer to static string describing the error
 * 
 * @note Returned string is static (do not free)
 * @note Thread-safe (returns pointer to string literal)
 * @note Unknown error codes return "Unknown error"
 * 
 * @see gemm_execute_plan()
 */
static inline const char* gemm_strerror(int error) 
{
    switch (error) {
        case 0:
            return "Success";
        case -1:
            return "Invalid pointer (NULL)";
        case -2:
            return "Invalid matrix dimensions";
        case -3:
            return "Memory allocation failed";
        case -4:
            return "Integer overflow in size calculation";
        case -5:
            return "Matrix dimensions exceed static pool limit";
        case -6:
            return "Feature not implemented";
        default:
            return "Unknown error";
    }
}

#ifdef __cplusplus
}
#endif

#endif // GEMM_UTILS_H