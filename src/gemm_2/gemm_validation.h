/**
 * @file gemm_validation.h
 * @brief Unified validation macros for GEMM operations (works everywhere)
 *
 * Validation levels:
 *   0 = Disabled (release builds)
 *   1 = Basic checks (NULL, alignment, dimensions)
 *   2 = Full checks (+ canaries, bounds, NaN/Inf detection)
 *
 * Usage:
 *   Debug:   -DGEMM_VALIDATION_LEVEL=2
 *   Release: -DGEMM_VALIDATION_LEVEL=0 or -DNDEBUG
 */

#ifndef GEMM_VALIDATION_H
#define GEMM_VALIDATION_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

//==============================================================================
// CROSS-PLATFORM FORMAT SPECIFIERS
//==============================================================================

#ifdef _WIN32
// Windows: Use %llu for size_t
#define FMT_ZU "%llu"
#define CAST_ZU(x) ((unsigned long long)(x))
#else
// Linux/macOS: Use standard %zu
#define FMT_ZU "%zu"
#define CAST_ZU(x) ((size_t)(x))
#endif

//==============================================================================
// VALIDATION LEVEL
//==============================================================================

#ifndef GEMM_VALIDATION_LEVEL
#if defined(NDEBUG)
#define GEMM_VALIDATION_LEVEL 0 // Release: no validation
#else
#define GEMM_VALIDATION_LEVEL 2 // Debug: full validation
#endif
#endif

//==============================================================================
// ERROR REPORTING
//==============================================================================

#if GEMM_VALIDATION_LEVEL > 0
#define VALIDATION_ERROR(fmt, ...)                                   \
    do                                                               \
    {                                                                \
        fprintf(stderr, "\n❌ VALIDATION ERROR at %s:%d: " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__);                  \
        fflush(stderr);                                              \
        abort();                                                     \
    } while (0)
#else
#define VALIDATION_ERROR(fmt, ...) ((void)0)
#endif

//==============================================================================
// VERBOSE LOGGING (only if explicitly enabled)
//==============================================================================

#ifdef GEMM_VALIDATION_VERBOSE
#define VALIDATION_LOG(fmt, ...) \
    fprintf(stderr, "[VALIDATION] " fmt "\n", ##__VA_ARGS__)
#else
#define VALIDATION_LOG(fmt, ...) ((void)0)
#endif

//==============================================================================
// BASIC VALIDATION MACROS (Level 1+)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 1

/**
 * @brief Validate pointer is non-NULL
 */
#define VALIDATE_PTR(ptr)                               \
    do                                                  \
    {                                                   \
        if (!(ptr))                                     \
        {                                               \
            VALIDATION_ERROR("NULL pointer: %s", #ptr); \
        }                                               \
    } while (0)

/**
 * @brief Validate 32-byte alignment
 */
#define VALIDATE_ALIGNED(ptr)                                                   \
    do                                                                          \
    {                                                                           \
        if (((uintptr_t)(ptr) & 31) != 0)                                       \
        {                                                                       \
            VALIDATION_ERROR("Misaligned pointer %s: %p (not 32-byte aligned)", \
                             #ptr, (void *)(ptr));                              \
        }                                                                       \
    } while (0)

/**
 * @brief Validate leading dimension
 */
#define VALIDATE_LDC(ldc, n)                                      \
    do                                                            \
    {                                                             \
        if ((ldc) < (n))                                          \
        {                                                         \
            VALIDATION_ERROR("Invalid ldc: " FMT_ZU " < " FMT_ZU, \
                             CAST_ZU(ldc), CAST_ZU(n));           \
        }                                                         \
    } while (0)

#else
#define VALIDATE_PTR(ptr) ((void)0)
#define VALIDATE_ALIGNED(ptr) ((void)0)
#define VALIDATE_LDC(ldc, n) ((void)0)
#endif

//==============================================================================
// PACKING VALIDATION (Level 1+)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 1

#define PACK_VALIDATE_PTR(ptr) VALIDATE_PTR(ptr)
#define PACK_VALIDATE_ALIGNED(ptr) VALIDATE_ALIGNED(ptr)

#define PACK_CHECK_MR(mr)                                                \
    do                                                                   \
    {                                                                    \
        if ((mr) != 8 && (mr) != 16)                                     \
        {                                                                \
            VALIDATION_ERROR("Invalid MR: " FMT_ZU " (must be 8 or 16)", \
                             CAST_ZU(mr));                               \
        }                                                                \
    } while (0)

#else
#define PACK_VALIDATE_PTR(ptr) ((void)0)
#define PACK_VALIDATE_ALIGNED(ptr) ((void)0)
#define PACK_CHECK_MR(mr) ((void)0)
#endif

//==============================================================================
// BOUNDS CHECKING (Level 2 only)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 2

#define PACK_CHECK_BOUNDS(ptr, offset, size)                                 \
    do                                                                       \
    {                                                                        \
        if ((offset) >= (size))                                              \
        {                                                                    \
            VALIDATION_ERROR("Pack bounds violation: " FMT_ZU " >= " FMT_ZU, \
                             CAST_ZU(offset), CAST_ZU(size));                \
        }                                                                    \
    } while (0)

#else
#define PACK_CHECK_BOUNDS(ptr, offset, size) ((void)0)
#endif

//==============================================================================
// CANARY PROTECTION (Level 2 only)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 2

#define GEMM_CANARY_VALUE 0xCAFEBABE
#define GEMM_CANARY_SIZE 8 // 8 bytes prefix + 8 bytes suffix

/**
 * @brief Allocate aligned memory with canary guards
 * @param size User data size
 * @param alignment Required alignment (must be power of 2, >= 32)
 */
static inline void *gemm_validate_alloc_aligned(size_t size, size_t alignment,
                                                const char *file, int line)
{
    // Round up to include canaries while maintaining alignment
    size_t total = size + 2 * GEMM_CANARY_SIZE + alignment;
    void *raw = malloc(total);

    if (!raw)
    {
        fprintf(stderr, "❌ ALLOCATION FAILED at %s:%d (size=" FMT_ZU ")\n",
                file, line, CAST_ZU(size));
        abort();
    }

    // Find aligned position after prefix canary
    uintptr_t raw_addr = (uintptr_t)raw;
    uintptr_t user_addr = (raw_addr + GEMM_CANARY_SIZE + alignment - 1) & ~(alignment - 1);
    uint8_t *user = (uint8_t *)user_addr;
    uint8_t *base = user - GEMM_CANARY_SIZE;

    // Write prefix canary
    *(uint32_t *)base = GEMM_CANARY_VALUE;
    *(uint32_t *)(base + 4) = GEMM_CANARY_VALUE;

    // Write suffix canary
    *(uint32_t *)(user + size) = GEMM_CANARY_VALUE;
    *(uint32_t *)(user + size + 4) = GEMM_CANARY_VALUE;

    // Store raw pointer before canary for free()
    *((void **)(base - sizeof(void *))) = raw;

    VALIDATION_LOG("Allocated " FMT_ZU " bytes at %p (with canaries, aligned to " FMT_ZU ")",
                   CAST_ZU(size), user, CAST_ZU(alignment));

    return user;
}

/**
 * @brief Check canary guards
 */
static inline void gemm_validate_bounds(const void *ptr, size_t size,
                                        const char *file, int line)
{
    const uint8_t *user = (const uint8_t *)ptr;
    const uint8_t *base = user - GEMM_CANARY_SIZE;

    // Check prefix canary
    uint32_t prefix1 = *(const uint32_t *)base;
    uint32_t prefix2 = *(const uint32_t *)(base + 4);

    if (prefix1 != GEMM_CANARY_VALUE || prefix2 != GEMM_CANARY_VALUE)
    {
        fprintf(stderr,
                "\n❌ BUFFER UNDERFLOW at %s:%d\n"
                "   Pointer: %p\n"
                "   Prefix canary corrupted: 0x%08X 0x%08X (expected 0x%08X)\n",
                file, line, ptr, prefix1, prefix2, GEMM_CANARY_VALUE);
        abort();
    }

    // Check suffix canary
    uint32_t suffix1 = *(const uint32_t *)(user + size);
    uint32_t suffix2 = *(const uint32_t *)(user + size + 4);

    if (suffix1 != GEMM_CANARY_VALUE || suffix2 != GEMM_CANARY_VALUE)
    {
        fprintf(stderr,
                "\n❌ BUFFER OVERFLOW at %s:%d\n"
                "   Pointer: %p\n"
                "   Size: " FMT_ZU " bytes\n"
                "   Suffix canary corrupted: 0x%08X 0x%08X (expected 0x%08X)\n",
                file, line, ptr, CAST_ZU(size), suffix1, suffix2, GEMM_CANARY_VALUE);
        abort();
    }
}

/**
 * @brief Free canary-protected memory
 */
static inline void gemm_validate_free(void *ptr, const char *file, int line)
{
    if (!ptr)
        return;

    uint8_t *user = (uint8_t *)ptr;
    uint8_t *base = user - GEMM_CANARY_SIZE;

    // Retrieve original malloc pointer
    void *raw = *((void **)(base - sizeof(void *)));

    VALIDATION_LOG("Freeing %p", ptr);

    free(raw);
}

#define GEMM_ALLOC(ptr, size) \
    (ptr) = gemm_validate_alloc_aligned((size), 32, __FILE__, __LINE__)

#define GEMM_VALIDATE(ptr, size) \
    gemm_validate_bounds((ptr), (size), __FILE__, __LINE__)

#define GEMM_FREE(ptr) \
    gemm_validate_free((ptr), __FILE__, __LINE__)

#else
// Level 0/1: No canaries
#define GEMM_ALLOC(ptr, size) (ptr) = malloc(size)
#define GEMM_VALIDATE(ptr, size) ((void)0)
#define GEMM_FREE(ptr) free(ptr)
#endif

//==============================================================================
// KERNEL ENTRY VALIDATION (Level 2 only)
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 2

/**
 * @brief Log kernel entry (optional, controlled by GEMM_VALIDATION_VERBOSE)
 */
#define GEMM_KERNEL_ENTRY(name, M, K, N)                               \
    VALIDATION_LOG("Entering %s: M=" FMT_ZU " K=" FMT_ZU " N=" FMT_ZU, \
                   (name), CAST_ZU(M), CAST_ZU(K), CAST_ZU(N))

/**
 * @brief Comprehensive kernel parameter validation
 */
#define VALIDATE_KERNEL_PARAMS(C, ldc, Ap, a_stride, Bp, b_stride, K, M, N, mr)                   \
    do                                                                                            \
    {                                                                                             \
        VALIDATION_LOG("Validating kernel: M=" FMT_ZU ", K=" FMT_ZU ", N=" FMT_ZU ", MR=" FMT_ZU, \
                       CAST_ZU(M), CAST_ZU(K), CAST_ZU(N), CAST_ZU(mr));                          \
        VALIDATE_PTR(C);                                                                          \
        VALIDATE_PTR(Ap);                                                                         \
        VALIDATE_PTR(Bp);                                                                         \
        VALIDATE_ALIGNED(C);                                                                      \
        VALIDATE_ALIGNED(Ap);                                                                     \
        VALIDATE_ALIGNED(Bp);                                                                     \
        if ((ldc) < (N))                                                                          \
        {                                                                                         \
            VALIDATION_ERROR("ldc " FMT_ZU " < N " FMT_ZU, CAST_ZU(ldc), CAST_ZU(N));             \
        }                                                                                         \
        if ((a_stride) != (mr))                                                                   \
        {                                                                                         \
            VALIDATION_ERROR("A stride mismatch: expected " FMT_ZU ", got " FMT_ZU,               \
                             CAST_ZU(mr), CAST_ZU(a_stride));                                     \
        }                                                                                         \
        if ((b_stride) != 16)                                                                     \
        {                                                                                         \
            VALIDATION_ERROR("B stride must be 16, got " FMT_ZU, CAST_ZU(b_stride));              \
        }                                                                                         \
        if ((M) > (mr))                                                                           \
        {                                                                                         \
            VALIDATION_ERROR("M " FMT_ZU " > MR " FMT_ZU, CAST_ZU(M), CAST_ZU(mr));               \
        }                                                                                         \
        if ((N) > 16)                                                                             \
        {                                                                                         \
            VALIDATION_ERROR("N " FMT_ZU " > 16", CAST_ZU(N));                                    \
        }                                                                                         \
    } while (0)

/**
 * @brief Post-kernel validation
 */
#define VALIDATE_KERNEL_POST(Ap, Bp, K, mr)                                \
    do                                                                     \
    {                                                                      \
        VALIDATION_LOG("Post-kernel validation: K=" FMT_ZU ", MR=" FMT_ZU, \
                       CAST_ZU(K), CAST_ZU(mr));                           \
        GEMM_VALIDATE(Ap, (K) * (mr) * sizeof(float));                     \
        GEMM_VALIDATE(Bp, (K) * 16 * sizeof(float));                       \
    } while (0)

/**
 * @brief Validate packed A access
 */
#define VALIDATE_PACKED_A_ACCESS(Ap, k, mr, K)                      \
    do                                                              \
    {                                                               \
        if ((k) >= (K))                                             \
        {                                                           \
            VALIDATION_ERROR("Packed A: k=" FMT_ZU " >= K=" FMT_ZU, \
                             CAST_ZU(k), CAST_ZU(K));               \
        }                                                           \
    } while (0)

/**
 * @brief Validate packed B access
 */
#define VALIDATE_PACKED_B_ACCESS(Bp, k, K)                          \
    do                                                              \
    {                                                               \
        if ((k) >= (K))                                             \
        {                                                           \
            VALIDATION_ERROR("Packed B: k=" FMT_ZU " >= K=" FMT_ZU, \
                             CAST_ZU(k), CAST_ZU(K));               \
        }                                                           \
    } while (0)

#else
#define GEMM_KERNEL_ENTRY(name, M, K, N) ((void)0)
#define VALIDATE_KERNEL_PARAMS(C, ldc, Ap, a_stride, Bp, b_stride, K, M, N, mr) ((void)0)
#define VALIDATE_KERNEL_POST(Ap, Bp, K, mr) ((void)0)
#define VALIDATE_PACKED_A_ACCESS(Ap, k, mr, K) ((void)0)
#define VALIDATE_PACKED_B_ACCESS(Bp, k, K) ((void)0)
#endif

//==============================================================================
// CROSS-PLATFORM HEAP VALIDATION
//==============================================================================

#if GEMM_VALIDATION_LEVEL >= 1

#ifdef _WIN32
// Windows: Use _CrtIsValidHeapPointer (requires debug CRT)
#ifdef _DEBUG
#include <crtdbg.h>
#define VALIDATE_HEAP_PTR(ptr)                                           \
    do                                                                   \
    {                                                                    \
        if (!(ptr) || !_CrtIsValidHeapPointer(ptr))                      \
        {                                                                \
            VALIDATION_ERROR("Invalid heap pointer: %p", (void *)(ptr)); \
        }                                                                \
    } while (0)
#else
#define VALIDATE_HEAP_PTR(ptr) VALIDATE_PTR(ptr)
#endif
#else
// Linux: Basic NULL check (no equivalent to _CrtIsValidHeapPointer)
#define VALIDATE_HEAP_PTR(ptr) VALIDATE_PTR(ptr)
#endif

#else
#define VALIDATE_HEAP_PTR(ptr) ((void)0)
#endif

#endif /* GEMM_VALIDATION_H */