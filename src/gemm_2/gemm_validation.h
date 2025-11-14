/**
 * @file gemm_validation.h
 * @brief UNIFIED runtime validation for GEMM kernels & packing (v2.2 - Enhanced)
 *
 * NEW IN v2.2:
 * - Packed buffer access validation
 * - Kernel pre/post validation hooks
 * - Buffer size tracking
 * - Enhanced debugging for intermittent crashes
 */

#ifndef GEMM_VALIDATION_H
#define GEMM_VALIDATION_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Platform-specific includes
#if defined(__linux__)
#include <unistd.h>
#elif defined(_MSC_VER)
#include <crtdbg.h>
#endif

//==============================================================================
// UNIFIED DIAGNOSTIC MACROS
//==============================================================================

#define VALIDATION_ASSERT(cond, msg, ...)                              \
    do                                                                 \
    {                                                                  \
        if (!(cond))                                                   \
        {                                                              \
            VALIDATION_ERROR("Assertion failed: " msg, ##__VA_ARGS__); \
        }                                                              \
    } while (0)

#if GEMM_VALIDATION_VERBOSE
#define VALIDATION_LOG(msg, ...)                                  \
    do                                                            \
    {                                                             \
        fprintf(stderr, "[VALIDATION] " msg "\n", ##__VA_ARGS__); \
        fflush(stderr);                                           \
    } while (0)
#else
#define VALIDATION_LOG(msg, ...) ((void)0)
#endif

//==============================================================================
// POINTER & ALIGNMENT VALIDATION (Level >= 1)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 1

// Platform-specific heap validation
#if defined(_MSC_VER) && defined(_DEBUG)
#define VALIDATE_HEAP_PTR(ptr)                                    \
    do                                                            \
    {                                                             \
        if (_CrtIsValidHeapPointer((const void *)(ptr)) == FALSE) \
        {                                                         \
            VALIDATION_ERROR("Invalid heap pointer: %p", (ptr));  \
        }                                                         \
    } while (0)
#elif defined(__linux__)
#define VALIDATE_HEAP_PTR(ptr)                                            \
    do                                                                    \
    {                                                                     \
        const void *p = (const void *)(ptr);                              \
        volatile char test_byte;                                          \
        __asm__ __volatile__("" ::: "memory");                            \
        test_byte = *(volatile const char *)p; /* Will SEGV if invalid */ \
        (void)test_byte;                                                  \
        __asm__ __volatile__("" ::: "memory");                            \
    } while (0)
#else
#define VALIDATE_HEAP_PTR(ptr) ((void)0)
#endif

#define VALIDATE_PTR(ptr)                               \
    do                                                  \
    {                                                   \
        if ((ptr) == NULL)                              \
        {                                               \
            VALIDATION_ERROR("NULL pointer: %s", #ptr); \
        }                                               \
        VALIDATE_HEAP_PTR(ptr);                         \
    } while (0)

// Kernel entry macro
#define GEMM_KERNEL_ENTRY(C, ldc, A, a_stride, B, b_stride, K, M, N) \
    do                                                               \
    {                                                                \
        VALIDATE_PTR(C);                                             \
        VALIDATE_PTR(A);                                             \
        VALIDATE_PTR(B);                                             \
        VALIDATE_ALIGNED(C, GEMM_ALIGNMENT);                         \
        VALIDATE_ALIGNED(A, GEMM_ALIGNMENT);                         \
        VALIDATE_ALIGNED(B, GEMM_ALIGNMENT);                         \
        VALIDATION_LOG("Kernel entry: M=%zu K=%zu N=%zu",            \
                       (size_t)(M), (size_t)(K), (size_t)(N));       \
    } while (0)

// Packing entry macros
#define PACK_VALIDATE_PTR(ptr) VALIDATE_PTR(ptr)
#define PACK_VALIDATE_ALIGNED(ptr) VALIDATE_ALIGNED(ptr, GEMM_ALIGNMENT)

#else
#define VALIDATE_PTR(ptr) ((void)0)
#define VALIDATE_ALIGNED(ptr, alignment) ((void)0)
#define GEMM_KERNEL_ENTRY(C, ldc, A, a_stride, B, b_stride, K, M, N) ((void)0)
#define PACK_VALIDATE_PTR(ptr) ((void)0)
#define PACK_VALIDATE_ALIGNED(ptr) ((void)0)
#endif

//==============================================================================
// DIMENSION VALIDATION (Level >= 1)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 1

#define VALIDATE_DIM(val, min, max, name)                                          \
    do                                                                             \
    {                                                                              \
        if ((val) < (min) || (val) > (max))                                        \
        {                                                                          \
            VALIDATION_ERROR("Invalid %s dimension: %zu (expected [%zu, %zu])",    \
                             (name), (size_t)(val), (size_t)(min), (size_t)(max)); \
        }                                                                          \
    } while (0)

#define PACK_CHECK_MR(mr)                                         \
    do                                                            \
    {                                                             \
        if ((mr) != 8 && (mr) != 16)                              \
        {                                                         \
            VALIDATION_ERROR("Invalid MR: %zu (must be 8 or 16)", \
                             (size_t)(mr));                       \
        }                                                         \
    } while (0)

#else
#define VALIDATE_DIM(val, min, max, name) ((void)0)
#define PACK_CHECK_MR(mr) ((void)0)
#endif

//==============================================================================
// BOUNDS CHECKING (Level >= 2)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 2

#define GEMM_CANARY_VALUE 0xCAFEBABE
#define GEMM_CANARY_SIZE 8 // 2 x uint32_t at each end

static inline void *gemm_validate_alloc(size_t size)
{
    size_t total = size + 2 * GEMM_CANARY_SIZE;
    char *base = (char *)malloc(total);
    if (!base)
        VALIDATION_ERROR("Allocation failed: %zu bytes", (size_t)total);

    // Prefix & suffix canaries
    *(uint32_t *)base = GEMM_CANARY_VALUE;
    *(uint32_t *)(base + 4) = GEMM_CANARY_VALUE;
    *(uint32_t *)(base + total - 8) = GEMM_CANARY_VALUE;
    *(uint32_t *)(base + total - 4) = GEMM_CANARY_VALUE;

    VALIDATION_LOG("Allocated %zu bytes at %p (user ptr: %p)",
                   total, (void *)base, (void *)(base + GEMM_CANARY_SIZE));

    return base + GEMM_CANARY_SIZE;
}

static inline void gemm_validate_bounds(const void *ptr, size_t size)
{
    const char *base = (const char *)ptr - GEMM_CANARY_SIZE;
    size_t total = size + 2 * GEMM_CANARY_SIZE;

    VALIDATION_LOG("Validating bounds for ptr=%p, size=%zu", ptr, size);

    if (*(uint32_t *)base != GEMM_CANARY_VALUE ||
        *(uint32_t *)(base + 4) != GEMM_CANARY_VALUE)
    {
        VALIDATION_ERROR("Buffer underflow at %p (prefix canary corrupted)", ptr);
    }
    if (*(uint32_t *)(base + total - 8) != GEMM_CANARY_VALUE ||
        *(uint32_t *)(base + total - 4) != GEMM_CANARY_VALUE)
    {
        VALIDATION_ERROR("Buffer overflow at %p (suffix canary corrupted)", ptr);
    }
}

#define PACK_CHECK_BOUNDS(ptr, offset_bytes, buffer_size_bytes)                              \
    do                                                                                       \
    {                                                                                        \
        if ((offset_bytes) > (buffer_size_bytes))                                            \
        {                                                                                    \
            VALIDATION_ERROR("Buffer overflow: write at offset %zu exceeds buffer size %zu", \
                             (size_t)(offset_bytes), (size_t)(buffer_size_bytes));           \
        }                                                                                    \
    } while (0)

#else
#define GEMM_CANARY_VALUE 0
#define GEMM_CANARY_SIZE 0
#define gemm_validate_alloc malloc
#define gemm_validate_bounds(ptr, size) ((void)0)
#define PACK_CHECK_BOUNDS(ptr, offset, size) ((void)0)
#endif

//==============================================================================
// PACKED BUFFER VALIDATION (Level >= 2)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 2

/**
 * @brief Validate that a K-loop index won't cause out-of-bounds access in packed A
 *
 * Packed A layout: K rows × MR columns (column-major)
 * Access pattern: Ap[k * mr + row_offset]
 * Valid range: k ∈ [0, K), row_offset ∈ [0, MR)
 */
#define VALIDATE_PACKED_A_ACCESS(Ap, k, mr, K)                                    \
    do                                                                            \
    {                                                                             \
        if ((k) >= (K))                                                           \
        {                                                                         \
            VALIDATION_ERROR("Packed A: k=%zu exceeds K=%zu",                     \
                             (size_t)(k), (size_t)(K));                           \
        }                                                                         \
        /* Worst-case access: Ap[k*mr + (mr-1)] */                                \
        size_t max_offset = (k) * (mr) + (mr);                                    \
        size_t buffer_size = (K) * (mr);                                          \
        if (max_offset > buffer_size)                                             \
        {                                                                         \
            VALIDATION_ERROR("Packed A: access at k=%zu, mr=%zu exceeds buffer "  \
                             "(offset=%zu > size=%zu)",                           \
                             (size_t)(k), (size_t)(mr), max_offset, buffer_size); \
        }                                                                         \
    } while (0)

/**
 * @brief Validate that a K-loop index won't cause out-of-bounds access in packed B
 *
 * Packed B layout: K rows × 16 columns (row-major)
 * Access pattern: Bp[k * 16 + col_offset]
 * Valid range: k ∈ [0, K), col_offset ∈ [0, 16)
 */
#define VALIDATE_PACKED_B_ACCESS(Bp, k, K)                               \
    do                                                                   \
    {                                                                    \
        if ((k) >= (K))                                                  \
        {                                                                \
            VALIDATION_ERROR("Packed B: k=%zu exceeds K=%zu",            \
                             (size_t)(k), (size_t)(K));                  \
        }                                                                \
        /* Worst-case access: Bp[k*16 + 15] */                           \
        size_t max_offset = (k) * 16 + 16;                               \
        size_t buffer_size = (K) * 16;                                   \
        if (max_offset > buffer_size)                                    \
        {                                                                \
            VALIDATION_ERROR("Packed B: access at k=%zu exceeds buffer " \
                             "(offset=%zu > size=%zu)",                  \
                             (size_t)(k), max_offset, buffer_size);      \
        }                                                                \
    } while (0)

/**
 * @brief Validate entire kernel invocation parameters
 *
 * This macro should be called at the START of each microkernel to catch
 * parameter errors before any memory access occurs.
 */
#define VALIDATE_KERNEL_PARAMS(C, ldc, Ap, a_stride, Bp, b_stride, K, M, N, mr) \
    do                                                                          \
    {                                                                           \
        VALIDATION_LOG("Validating kernel: M=%zu, K=%zu, N=%zu, MR=%zu",        \
                       (size_t)(M), (size_t)(K), (size_t)(N), (size_t)(mr));    \
        /* Validate pointers */                                                 \
        VALIDATE_PTR(C);                                                        \
        VALIDATE_PTR(Ap);                                                       \
        VALIDATE_PTR(Bp);                                                       \
        /* Validate alignment */                                                \
        VALIDATE_ALIGNED(C, 32);                                                \
        VALIDATE_ALIGNED(Ap, 32);                                               \
        VALIDATE_ALIGNED(Bp, 32);                                               \
        /* Validate dimensions */                                               \
        if ((M) == 0 || (K) == 0 || (N) == 0)                                   \
        {                                                                       \
            VALIDATION_ERROR("Zero dimension: M=%zu, K=%zu, N=%zu",             \
                             (size_t)(M), (size_t)(K), (size_t)(N));            \
        }                                                                       \
        if ((M) > (mr))                                                         \
        {                                                                       \
            VALIDATION_ERROR("M=%zu exceeds MR=%zu",                            \
                             (size_t)(M), (size_t)(mr));                        \
        }                                                                       \
        if ((N) > 16)                                                           \
        {                                                                       \
            VALIDATION_ERROR("N=%zu exceeds NR=16", (size_t)(N));               \
        }                                                                       \
        /* Validate strides */                                                  \
        if ((a_stride) != (mr))                                                 \
        {                                                                       \
            VALIDATION_ERROR("A stride mismatch: expected %zu, got %zu",        \
                             (size_t)(mr), (size_t)(a_stride));                 \
        }                                                                       \
        if ((b_stride) != 16)                                                   \
        {                                                                       \
            VALIDATION_ERROR("B stride must be 16, got %zu",                    \
                             (size_t)(b_stride));                               \
        }                                                                       \
        if ((ldc) < (N))                                                        \
        {                                                                       \
            VALIDATION_ERROR("ldc=%zu < N=%zu", (size_t)(ldc), (size_t)(N));    \
        }                                                                       \
    } while (0)

/**
 * @brief Validate kernel state after execution
 *
 * Call this AFTER each microkernel to verify buffer integrity.
 */
#define VALIDATE_KERNEL_POST(Ap, Bp, K, mr)                     \
    do                                                          \
    {                                                           \
        VALIDATION_LOG("Post-kernel validation: K=%zu, MR=%zu", \
                       (size_t)(K), (size_t)(mr));              \
        gemm_validate_bounds(Ap, (K) * (mr) * sizeof(float));   \
        gemm_validate_bounds(Bp, (K) * 16 * sizeof(float));     \
    } while (0)

#else
#define VALIDATE_PACKED_A_ACCESS(Ap, k, mr, K) ((void)0)
#define VALIDATE_PACKED_B_ACCESS(Bp, k, K) ((void)0)
#define VALIDATE_KERNEL_PARAMS(C, ldc, Ap, a_stride, Bp, b_stride, K, M, N, mr) ((void)0)
#define VALIDATE_KERNEL_POST(Ap, Bp, K, mr) ((void)0)
#endif

//==============================================================================
// BUFFER ALLOCATION WRAPPERS
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 2
#define GEMM_ALLOC(ptr, size)                   \
    do                                          \
    {                                           \
        void *_tmp = gemm_validate_alloc(size); \
        (ptr) = _tmp;                           \
    } while (0)
#define GEMM_FREE(ptr)                              \
    do                                              \
    {                                               \
        if (ptr)                                    \
        {                                           \
            free((char *)(ptr) - GEMM_CANARY_SIZE); \
            (ptr) = NULL;                           \
        }                                           \
    } while (0)
#define GEMM_VALIDATE(ptr, size) gemm_validate_bounds(ptr, size)
#else
#define GEMM_ALLOC(ptr, size)                                                 \
    do                                                                        \
    {                                                                         \
        (ptr) = malloc(size);                                                 \
        if (!(ptr) && (size) > 0)                                             \
            VALIDATION_ERROR("Allocation failed: %zu bytes", (size_t)(size)); \
    } while (0)
#define GEMM_FREE(ptr) \
    do                 \
    {                  \
        free(ptr);     \
        (ptr) = NULL;  \
    } while (0)
#define GEMM_VALIDATE(ptr, size) ((void)0)
#endif

#endif // GEMM_VALIDATION_H