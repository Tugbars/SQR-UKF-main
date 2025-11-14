/**
 * @file gemm_kernels_avx2_complete.h
 * @brief Complete AVX2/FMA GEMM Micro-kernels (Refactored - Phase 1)
 */

#ifndef GEMM_KERNELS_AVX2_COMPLETE_H
#define GEMM_KERNELS_AVX2_COMPLETE_H

#include "gemm_simd_ops.h"
#include "gemm_planning.h"
#include <stdalign.h>
#include <stdint.h>
#include <string.h>

// Prefetch configuration
#ifndef GEMM_PREFETCH_MIN_K
#define GEMM_PREFETCH_MIN_K 128
#endif

#ifndef LINALG_GEMM_PREFETCH_A_LONG
#define LINALG_GEMM_PREFETCH_A_LONG 0
#endif

#ifndef LINALG_NT_STORES
#define LINALG_NT_STORES 1
#endif

// Prefetch macros
#ifdef _MSC_VER
#define PREFETCH_T0(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#define PREFETCH_T1(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T1)
#define PREFETCH_T2(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T2)
#define PREFETCH_NTA(addr) _mm_prefetch((const char *)(addr), _MM_HINT_NTA)
#else
#define PREFETCH_T0(addr) __builtin_prefetch((addr), 0, 3)
#define PREFETCH_T1(addr) __builtin_prefetch((addr), 0, 2)
#define PREFETCH_T2(addr) __builtin_prefetch((addr), 0, 1)
#define PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0)
#endif

// Restrict qualifier
#ifndef RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif
#endif

//==============================================================================
// CONFIGURATION: Aligned vs Unaligned Memory Operations
//==============================================================================

/**
 * @brief Toggle for aligned memory operations
 *
 * Set to 1: Use aligned loads/stores (GEMM_LOAD_PS / GEMM_STORE_PS)
 * Set to 0: Use unaligned loads/stores (_mm256_loadu_ps / _mm256_storeu_ps)
 *
 * NOTE: Aligned operations require 32-byte alignment and may segfault if misaligned.
 *       Unaligned operations are slightly slower but safer.
 */
#ifndef GEMM_USE_ALIGNED_OPS
#define GEMM_USE_ALIGNED_OPS 0 // Default: unaligned (safer)
#endif

// Load/Store macro system
#if GEMM_USE_ALIGNED_OPS
#define GEMM_LOAD_PS(ptr) _mm256_load_ps(ptr)
#define GEMM_STORE_PS(ptr, val) _mm256_store_ps(ptr, val)
#define GEMM_STREAM_PS(ptr, val) _mm256_stream_ps(ptr, val)
#else
#define GEMM_LOAD_PS(ptr) _mm256_loadu_ps(ptr)
#define GEMM_STORE_PS(ptr, val) _mm256_storeu_ps(ptr, val)
#define GEMM_STREAM_PS(ptr, val) _mm256_storeu_ps(ptr, val) // Fall back to storeu
#endif

// Masked operations (always available, unaligned by nature)
#define GEMM_MASKLOAD_PS(ptr, mask) _mm256_maskload_ps(ptr, mask)
#define GEMM_MASKSTORE_PS(ptr, mask, val) _mm256_maskstore_ps(ptr, mask, val)

// Alignment helper
#ifndef LINALG_ASSUME_ALIGNED
#if GEMM_USE_ALIGNED_OPS && (defined(__GNUC__) || defined(__clang__))
#define LINALG_ASSUME_ALIGNED(p, n) (p) = (__typeof__(p))__builtin_assume_aligned((p), (n))
#else
#define LINALG_ASSUME_ALIGNED(p, n) ((void)0)
#endif
#endif

//==============================================================================
// HELPER: Build mask for any width (handles > 8)
//==============================================================================

/**
 * @brief Prefetch output rows for upcoming writes
 */
static inline void __attribute__((always_inline))
gemm_prefetch_c_rows(const float *c, size_t ldc, size_t m)
{
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);
}

/**
 * @brief Prefetch B panel (short + long distance) and optionally A panel
 *
 * @param Bp B panel base pointer
 * @param Ap A panel base pointer (can be NULL if not prefetching A)
 * @param k Current k index
 * @param Kblk Total K dimension
 * @param b_stride Stride of B panel (e.g., 8 or 16)
 * @param a_stride Stride of A panel (e.g., 8 or 16)
 * @param pf_long_dist Long prefetch distance (e.g., 32)
 */
static inline void __attribute__((always_inline))
gemm_prefetch_panels(
    const float *Bp,
    const float *Ap,
    size_t k,
    size_t Kblk,
    size_t b_stride,
    size_t a_stride,
    size_t pf_long_dist)
{
    size_t kpf_s = k + 8;
    size_t kpf_l = k + pf_long_dist;

    if (kpf_s < Kblk)
        PREFETCH_T0(Bp + kpf_s * b_stride);
    if (kpf_l < Kblk)
        PREFETCH_T0(Bp + kpf_l * b_stride);

#if LINALG_GEMM_PREFETCH_A_LONG
    if (Ap && kpf_l < Kblk)
        PREFETCH_T0(Ap + kpf_l * a_stride);
#else
    (void)Ap; // Suppress unused warning
    (void)a_stride;
#endif
}

//==============================================================================
// TRANSPOSE-AND-WRITEBACK HELPERS
//==============================================================================

/**
 * @brief Transpose 8x8 tile and ADD to destination (C += result)
 *
 * @param c Output pointer
 * @param ldc Leading dimension
 * @param cols Array of 8 column vectors (will be transposed in-place)
 * @param use_nt Non-temporal hint (ignored for ADD mode)
 */
static inline void __attribute__((always_inline))
gemm_transpose_add_8x8(
    float *RESTRICT c,
    size_t ldc,
    __m256 cols[8],
    int use_nt)
{
    (void)use_nt; // NT stores not applicable for ADD mode

    gemm_transpose_8x8_avx2(cols);

    for (size_t r = 0; r < 8; ++r)
    {
        float *cr = c + r * ldc;
        __m256 old = GEMM_LOAD_PS(cr);
        __m256 sum = _mm256_add_ps(old, cols[r]);
        GEMM_STORE_PS(cr, sum);
    }
}

/**
 * @brief Transpose 8x8 tile and STORE to destination (C = result)
 *
 * @param c Output pointer
 * @param ldc Leading dimension
 * @param cols Array of 8 column vectors (will be transposed in-place)
 * @param use_nt Use non-temporal stores if enabled
 */
static inline void __attribute__((always_inline))
gemm_transpose_store_8x8(
    float *RESTRICT c,
    size_t ldc,
    __m256 cols[8],
    int use_nt)
{
    gemm_transpose_8x8_avx2(cols);

    for (size_t r = 0; r < 8; ++r)
    {
        float *cr = c + r * ldc;
        if (use_nt)
            GEMM_STREAM_PS(cr, cols[r]);
        else
            GEMM_STORE_PS(cr, cols[r]);
    }
}

/**
 * @brief Transpose 8x8 tile and ADD with mask (for n < 8)
 */
static inline void __attribute__((always_inline))
gemm_transpose_add_8x8_masked(
    float *RESTRICT c,
    size_t ldc,
    __m256 cols[8],
    __m256i mask,
    size_t m)
{
    gemm_transpose_8x8_avx2(cols);

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 old = GEMM_MASKLOAD_PS(cr, mask);
        __m256 sum = _mm256_add_ps(old, cols[r]);
        GEMM_MASKSTORE_PS(cr, mask, sum);
    }
}

/**
 * @brief Transpose 8x8 tile and STORE with mask (for n < 8)
 */
static inline void __attribute__((always_inline))
gemm_transpose_store_8x8_masked(
    float *RESTRICT c,
    size_t ldc,
    __m256 cols[8],
    __m256i mask,
    size_t m)
{
    gemm_transpose_8x8_avx2(cols);

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        GEMM_MASKSTORE_PS(cr, mask, cols[r]);
    }
}

//==============================================================================
// HELPER: Load columns from column-major temp buffer (unchanged)
//==============================================================================

static inline __m256 load_cols_from_temp(
    const float *temp,
    size_t stride,
    size_t r,
    size_t n)
{
    alignas(32) float lane[8] = {0};
    for (size_t j = 0; j < n; ++j)
        lane[j] = temp[j * stride + r];
    return GEMM_LOAD_PS(lane);
}

//==============================================================================
// 4×8 KERNELS (Tail handling for 4 rows)
//==============================================================================

/**
 * @brief 4×8 kernel (ADD): C += A*B
 */
static inline void gemm_4x8_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t jb, __m256i m)
{
    // Validate strides (disabled in release with -DNDEBUG)
    assert(a_k_stride == 8 && "4x8 kernel requires A packed with MR=8");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);

    PREFETCH_T0(c + 0 * ldc);
    PREFETCH_T0(c + 1 * ldc);
    PREFETCH_T0(c + 2 * ldc);
    PREFETCH_T0(c + 3 * ldc);
    PREFETCH_T1(c + 4 * ldc);

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * b_k_stride); // ← CHANGED: was kpf * 8
        }
        const float *bptr = Bp + k * b_k_stride; // ← CHANGED: was k * 8
        const float *aptr = Ap + k * a_k_stride; // ← CHANGED: was k * 8

        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = GEMM_LOAD_PS(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            bptr += b_k_stride; // ← CHANGED: was += 8
            aptr += a_k_stride; // ← CHANGED: was += 8
        }
    }

    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * b_k_stride); // ← CHANGED: was kpf * 8
        }
        const __m256 b = GEMM_LOAD_PS(Bp + k * b_k_stride);                          // ← CHANGED: was k * 8
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 0), b, acc0); // ← CHANGED: was k * 8
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 1), b, acc1); // ← CHANGED
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 2), b, acc2); // ← CHANGED
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 3), b, acc3); // ← CHANGED
    }

    // Writeback unchanged
    if (jb == 8)
    {
        GEMM_STORE_PS(c + 0 * ldc, _mm256_add_ps(GEMM_LOAD_PS(c + 0 * ldc), acc0));
        GEMM_STORE_PS(c + 1 * ldc, _mm256_add_ps(GEMM_LOAD_PS(c + 1 * ldc), acc1));
        GEMM_STORE_PS(c + 2 * ldc, _mm256_add_ps(GEMM_LOAD_PS(c + 2 * ldc), acc2));
        GEMM_STORE_PS(c + 3 * ldc, _mm256_add_ps(GEMM_LOAD_PS(c + 3 * ldc), acc3));
    }
    else
    {
        __m256 old, sum;
        old = GEMM_MASKLOAD_PS(c + 0 * ldc, m);
        sum = _mm256_add_ps(old, acc0);
        GEMM_MASKSTORE_PS(c + 0 * ldc, m, sum);
        old = GEMM_MASKLOAD_PS(c + 1 * ldc, m);
        sum = _mm256_add_ps(old, acc1);
        GEMM_MASKSTORE_PS(c + 1 * ldc, m, sum);
        old = GEMM_MASKLOAD_PS(c + 2 * ldc, m);
        sum = _mm256_add_ps(old, acc2);
        GEMM_MASKSTORE_PS(c + 2 * ldc, m, sum);
        old = GEMM_MASKLOAD_PS(c + 3 * ldc, m);
        sum = _mm256_add_ps(old, acc3);
        GEMM_MASKSTORE_PS(c + 3 * ldc, m, sum);
    }
}

/**
 * @brief 4×8 kernel (STORE): C = A*B
 */
static inline void gemm_4x8_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t jb, __m256i m)
{
    assert(a_k_stride == 8 && "4x8 kernel requires A packed with MR=8");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    // Same pattern as ADD variant - apply same changes
    // [rest of implementation with stride usage]
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);

    PREFETCH_T0(c + 0 * ldc);
    PREFETCH_T0(c + 1 * ldc);
    PREFETCH_T0(c + 2 * ldc);
    PREFETCH_T0(c + 3 * ldc);
    PREFETCH_T1(c + 4 * ldc);

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * b_k_stride); // ← USE STRIDE
        }
        const float *bptr = Bp + k * b_k_stride; // ← USE STRIDE
        const float *aptr = Ap + k * a_k_stride; // ← USE STRIDE

        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = GEMM_LOAD_PS(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            bptr += b_k_stride; // ← USE STRIDE
            aptr += a_k_stride; // ← USE STRIDE
        }
    }

    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * b_k_stride); // ← USE STRIDE
        }
        const __m256 b = GEMM_LOAD_PS(Bp + k * b_k_stride); // ← USE STRIDE
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 3), b, acc3);
    }

    if (jb == 8)
    {
        GEMM_STORE_PS(c + 0 * ldc, acc0);
        GEMM_STORE_PS(c + 1 * ldc, acc1);
        GEMM_STORE_PS(c + 2 * ldc, acc2);
        GEMM_STORE_PS(c + 3 * ldc, acc3);
    }
    else
    {
        GEMM_MASKSTORE_PS(c + 0 * ldc, m, acc0);
        GEMM_MASKSTORE_PS(c + 1 * ldc, m, acc1);
        GEMM_MASKSTORE_PS(c + 2 * ldc, m, acc2);
        GEMM_MASKSTORE_PS(c + 3 * ldc, m, acc3);
    }
}

//==============================================================================
// 1×8 KERNELS (Tail handling for 1 row)
//==============================================================================

/**
 * @brief 1×8 kernel (ADD): C += A*B
 */
static inline void gemm_1x8_panel_avx2fma_add(
    float *RESTRICT c,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t jb, __m256i m)
{
    assert(a_k_stride == 8 && "1x8 kernel requires A packed with MR=8");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    __m256 acc = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);

    PREFETCH_T0(c);

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * b_k_stride); // ← USE STRIDE
        }
        const float *bptr = Bp + k * b_k_stride; // ← USE STRIDE
        const float *aptr = Ap + k * a_k_stride; // ← USE STRIDE

        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = GEMM_LOAD_PS(bptr);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc);
            bptr += b_k_stride; // ← USE STRIDE
            aptr += a_k_stride; // ← USE STRIDE
        }
    }

    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * b_k_stride); // ← USE STRIDE
        }
        const __m256 b = GEMM_LOAD_PS(Bp + k * b_k_stride); // ← USE STRIDE
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 0), b, acc);
    }

    if (jb == 8)
    {
        GEMM_STORE_PS(c, _mm256_add_ps(GEMM_LOAD_PS(c), acc));
    }
    else
    {
        __m256 oldv = GEMM_MASKLOAD_PS(c, m);
        __m256 sum = _mm256_add_ps(oldv, acc);
        GEMM_MASKSTORE_PS(c, m, sum);
    }
}

/**
 * @brief 1×8 kernel (STORE): C = A*B
 */
static inline void gemm_1x8_panel_avx2fma_store(
    float *RESTRICT c,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t jb, __m256i m)
{
    assert(a_k_stride == 8 && "1x8 kernel requires A packed with MR=8");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    // Same pattern as ADD variant
    __m256 acc = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);

    PREFETCH_T0(c);

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * b_k_stride);
        }
        const float *bptr = Bp + k * b_k_stride;
        const float *aptr = Ap + k * a_k_stride;

        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = GEMM_LOAD_PS(bptr);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc);
            bptr += b_k_stride;
            aptr += a_k_stride;
        }
    }

    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * b_k_stride);
        }
        const __m256 b = GEMM_LOAD_PS(Bp + k * b_k_stride);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * a_k_stride + 0), b, acc);
    }

    if (jb == 8)
    {
        GEMM_STORE_PS(c, acc);
    }
    else
    {
        GEMM_MASKSTORE_PS(c, m, acc);
    }
}

//==============================================================================
// 8×8 KERNEL (ADD) - REFACTORED EXAMPLE
//==============================================================================

/**
 * @brief 8×8 kernel (ADD): C += A*B
 */

static inline void gemm_8x8_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    assert(a_k_stride == 8 && "8x8 kernel requires A packed with MR=8");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    gemm_prefetch_c_rows(c, ldc, m);

    if (Kblk)
    {
        size_t k = 0;
        __m256 a = GEMM_LOAD_PS(Ap + 0 * a_k_stride); // ← USE STRIDE
        const float *brow = Bp + 0 * b_k_stride;        // ← USE STRIDE

        for (; k + 1 < Kblk; ++k)
        {
            if (do_pf)
                PREFETCH_T0(Bp + (k + 8) * b_k_stride); // ← USE STRIDE

            __m256 a_next = GEMM_LOAD_PS(Ap + (k + 1) * a_k_stride); // ← USE STRIDE
            const float *b_next = Bp + (k + 1) * b_k_stride;           // ← USE STRIDE

            __m256 b;
            b = _mm256_broadcast_ss(brow + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(brow + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(brow + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(brow + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(brow + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(brow + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
            b = _mm256_broadcast_ss(brow + 6);
            acc6 = _mm256_fmadd_ps(a, b, acc6);
            b = _mm256_broadcast_ss(brow + 7);
            acc7 = _mm256_fmadd_ps(a, b, acc7);

            a = a_next;
            brow = b_next;
        }

        // Epilogue
        {
            __m256 b;
            b = _mm256_broadcast_ss(brow + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(brow + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(brow + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(brow + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(brow + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(brow + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
            b = _mm256_broadcast_ss(brow + 6);
            acc6 = _mm256_fmadd_ps(a, b, acc6);
            b = _mm256_broadcast_ss(brow + 7);
            acc7 = _mm256_fmadd_ps(a, b, acc7);
        }
    }

    // Writeback unchanged
    if (m == 8 && n == 8)
    {
        __m256 cols[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        gemm_transpose_add_8x8(c, ldc, cols, 0);
    }
    else
    {
        alignas(32) float temp[8 * 8];
        GEMM_STORE_PS(temp + 0 * 8, acc0);
        GEMM_STORE_PS(temp + 1 * 8, acc1);
        GEMM_STORE_PS(temp + 2 * 8, acc2);
        GEMM_STORE_PS(temp + 3 * 8, acc3);
        GEMM_STORE_PS(temp + 4 * 8, acc4);
        GEMM_STORE_PS(temp + 5 * 8, acc5);
        GEMM_STORE_PS(temp + 6 * 8, acc6);
        GEMM_STORE_PS(temp + 7 * 8, acc7);

        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            __m256 old = GEMM_MASKLOAD_PS(cr, mask);
            GEMM_MASKSTORE_PS(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/**
 * @brief 8×8 kernel (STORE): C = A*B
 */
static inline void gemm_8x8_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride,  // ← ADD THIS
    const float *RESTRICT Bp, size_t b_k_stride,  // ← ADD THIS
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    assert(a_k_stride == 8 && "8x8 kernel requires A packed with MR=8");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    gemm_prefetch_c_rows(c, ldc, m);

    if (Kblk)
    {
        size_t k = 0;
        __m256 a      = GEMM_LOAD_PS(Ap + 0 * a_k_stride);  // ← USE STRIDE
        const float *brow = Bp + 0 * b_k_stride;              // ← USE STRIDE

        for (; k + 1 < Kblk; ++k)
        {
            if (do_pf) PREFETCH_T0(Bp + (k + 8) * b_k_stride);  // ← USE STRIDE

            __m256 a_next        = GEMM_LOAD_PS(Ap + (k + 1) * a_k_stride);  // ← USE STRIDE
            const float *b_next  = Bp + (k + 1) * b_k_stride;                  // ← USE STRIDE

            __m256 b;
            b = _mm256_broadcast_ss(brow + 0); acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(brow + 1); acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(brow + 2); acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(brow + 3); acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(brow + 4); acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(brow + 5); acc5 = _mm256_fmadd_ps(a, b, acc5);
            b = _mm256_broadcast_ss(brow + 6); acc6 = _mm256_fmadd_ps(a, b, acc6);
            b = _mm256_broadcast_ss(brow + 7); acc7 = _mm256_fmadd_ps(a, b, acc7);

            a     = a_next;
            brow  = b_next;
        }

        // Epilogue
        {
            __m256 b;
            b = _mm256_broadcast_ss(brow + 0); acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(brow + 1); acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(brow + 2); acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(brow + 3); acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(brow + 4); acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(brow + 5); acc5 = _mm256_fmadd_ps(a, b, acc5);
            b = _mm256_broadcast_ss(brow + 6); acc6 = _mm256_fmadd_ps(a, b, acc6);
            b = _mm256_broadcast_ss(brow + 7); acc7 = _mm256_fmadd_ps(a, b, acc7);
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 8) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 8 && n == 8)
    {
        __m256 cols[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        gemm_transpose_8x8_avx2(cols);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt) GEMM_STREAM_PS(cr, cols[r]);
            else        GEMM_STORE_PS(cr, cols[r]);
        }
    }
    else
    {
        alignas(32) float temp[8 * 8];
        GEMM_STORE_PS(temp + 0 * 8, acc0);
        GEMM_STORE_PS(temp + 1 * 8, acc1);
        GEMM_STORE_PS(temp + 2 * 8, acc2);
        GEMM_STORE_PS(temp + 3 * 8, acc3);
        GEMM_STORE_PS(temp + 4 * 8, acc4);
        GEMM_STORE_PS(temp + 5 * 8, acc5);
        GEMM_STORE_PS(temp + 6 * 8, acc6);
        GEMM_STORE_PS(temp + 7 * 8, acc7);

        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            GEMM_MASKSTORE_PS(cr, mask, sum);
        }
    }
}

//==============================================================================
// 16×8 KERNELS
//==============================================================================

/**
 * @brief 16×8 kernel (ADD): C += A*B
 */
static inline void gemm_16x8_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    assert(a_k_stride == 16 && "16x8 kernel requires A packed with MR=16");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo0 = _mm256_setzero_ps(), acc_lo1 = _mm256_setzero_ps();
    __m256 acc_lo2 = _mm256_setzero_ps(), acc_lo3 = _mm256_setzero_ps();
    __m256 acc_hi0 = _mm256_setzero_ps(), acc_hi1 = _mm256_setzero_ps();
    __m256 acc_hi2 = _mm256_setzero_ps(), acc_hi3 = _mm256_setzero_ps();
    __m256 acc_lo4 = _mm256_setzero_ps(), acc_lo5 = _mm256_setzero_ps();
    __m256 acc_lo6 = _mm256_setzero_ps(), acc_lo7 = _mm256_setzero_ps();
    __m256 acc_hi4 = _mm256_setzero_ps(), acc_hi5 = _mm256_setzero_ps();
    __m256 acc_hi6 = _mm256_setzero_ps(), acc_hi7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    gemm_prefetch_c_rows(c, ldc, m);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
            gemm_prefetch_panels(Bp, Ap, k, Kblk, b_k_stride, a_k_stride, PF_LONG); // ← USE STRIDES

        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = GEMM_LOAD_PS(Ap + kk * a_k_stride);     // ← USE STRIDE
            __m256 a_hi = GEMM_LOAD_PS(Ap + kk * a_k_stride + 8); // ← USE STRIDE
            const float *b_row = Bp + kk * b_k_stride;              // ← USE STRIDE
            __m256 b0 = _mm256_broadcast_ss(b_row + 0);
            __m256 b1 = _mm256_broadcast_ss(b_row + 1);
            __m256 b2 = _mm256_broadcast_ss(b_row + 2);
            __m256 b3 = _mm256_broadcast_ss(b_row + 3);
            acc_lo0 = _mm256_fmadd_ps(a_lo, b0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(a_hi, b0, acc_hi0);
            acc_lo1 = _mm256_fmadd_ps(a_lo, b1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(a_hi, b1, acc_hi1);
            acc_lo2 = _mm256_fmadd_ps(a_lo, b2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(a_hi, b2, acc_hi2);
            acc_lo3 = _mm256_fmadd_ps(a_lo, b3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(a_hi, b3, acc_hi3);
            __m256 b4 = _mm256_broadcast_ss(b_row + 4);
            __m256 b5 = _mm256_broadcast_ss(b_row + 5);
            __m256 b6 = _mm256_broadcast_ss(b_row + 6);
            __m256 b7 = _mm256_broadcast_ss(b_row + 7);
            acc_lo4 = _mm256_fmadd_ps(a_lo, b4, acc_lo4);
            acc_hi4 = _mm256_fmadd_ps(a_hi, b4, acc_hi4);
            acc_lo5 = _mm256_fmadd_ps(a_lo, b5, acc_lo5);
            acc_hi5 = _mm256_fmadd_ps(a_hi, b5, acc_hi5);
            acc_lo6 = _mm256_fmadd_ps(a_lo, b6, acc_lo6);
            acc_hi6 = _mm256_fmadd_ps(a_hi, b6, acc_hi6);
            acc_lo7 = _mm256_fmadd_ps(a_lo, b7, acc_lo7);
            acc_hi7 = _mm256_fmadd_ps(a_hi, b7, acc_hi7);
        }
    }

    if (m == 16 && n == 8)
    {
        __m256 cols_lo[8] = {acc_lo0, acc_lo1, acc_lo2, acc_lo3, acc_lo4, acc_lo5, acc_lo6, acc_lo7};
        __m256 cols_hi[8] = {acc_hi0, acc_hi1, acc_hi2, acc_hi3, acc_hi4, acc_hi5, acc_hi6, acc_hi7};
        gemm_transpose_8x8_avx2(cols_lo);
        gemm_transpose_8x8_avx2(cols_hi);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = _mm256_add_ps(GEMM_LOAD_PS(cr), cols_lo[r]);
            GEMM_STORE_PS(cr, sum);
        }
        for (size_t r = 8; r < 16; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = _mm256_add_ps(GEMM_LOAD_PS(cr), cols_hi[r - 8]);
            GEMM_STORE_PS(cr, sum);
        }
    }
    else
    {
        alignas(32) float temp[16 * 8];
        GEMM_STORE_PS(temp + 0 * 16, acc_lo0);
        GEMM_STORE_PS(temp + 0 * 16 + 8, acc_hi0);
        GEMM_STORE_PS(temp + 1 * 16, acc_lo1);
        GEMM_STORE_PS(temp + 1 * 16 + 8, acc_hi1);
        GEMM_STORE_PS(temp + 2 * 16, acc_lo2);
        GEMM_STORE_PS(temp + 2 * 16 + 8, acc_hi2);
        GEMM_STORE_PS(temp + 3 * 16, acc_lo3);
        GEMM_STORE_PS(temp + 3 * 16 + 8, acc_hi3);
        GEMM_STORE_PS(temp + 4 * 16, acc_lo4);
        GEMM_STORE_PS(temp + 4 * 16 + 8, acc_hi4);
        GEMM_STORE_PS(temp + 5 * 16, acc_lo5);
        GEMM_STORE_PS(temp + 5 * 16 + 8, acc_hi5);
        GEMM_STORE_PS(temp + 6 * 16, acc_lo6);
        GEMM_STORE_PS(temp + 6 * 16 + 8, acc_hi6);
        GEMM_STORE_PS(temp + 7 * 16, acc_lo7);
        GEMM_STORE_PS(temp + 7 * 16 + 8, acc_hi7);

        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 16, r, n);
            __m256 old = GEMM_MASKLOAD_PS(cr, mask);
            GEMM_MASKSTORE_PS(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/**
 * @brief 16×8 kernel (STORE): C = A*B
 */
static inline void gemm_16x8_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    assert(a_k_stride == 16 && "16x8 kernel requires A packed with MR=16");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo0 = _mm256_setzero_ps(), acc_lo1 = _mm256_setzero_ps();
    __m256 acc_lo2 = _mm256_setzero_ps(), acc_lo3 = _mm256_setzero_ps();
    __m256 acc_hi0 = _mm256_setzero_ps(), acc_hi1 = _mm256_setzero_ps();
    __m256 acc_hi2 = _mm256_setzero_ps(), acc_hi3 = _mm256_setzero_ps();
    __m256 acc_lo4 = _mm256_setzero_ps(), acc_lo5 = _mm256_setzero_ps();
    __m256 acc_lo6 = _mm256_setzero_ps(), acc_lo7 = _mm256_setzero_ps();
    __m256 acc_hi4 = _mm256_setzero_ps(), acc_hi5 = _mm256_setzero_ps();
    __m256 acc_hi6 = _mm256_setzero_ps(), acc_hi7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    gemm_prefetch_c_rows(c, ldc, m);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
            gemm_prefetch_panels(Bp, Ap, k, Kblk, b_k_stride, a_k_stride, PF_LONG); // ← USE STRIDES

        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = GEMM_LOAD_PS(Ap + kk * a_k_stride);     // ← USE STRIDE
            __m256 a_hi = GEMM_LOAD_PS(Ap + kk * a_k_stride + 8); // ← USE STRIDE
            const float *b_row = Bp + kk * b_k_stride;              // ← USE STRIDE
            __m256 b0 = _mm256_broadcast_ss(b_row + 0);
            __m256 b1 = _mm256_broadcast_ss(b_row + 1);
            __m256 b2 = _mm256_broadcast_ss(b_row + 2);
            __m256 b3 = _mm256_broadcast_ss(b_row + 3);
            acc_lo0 = _mm256_fmadd_ps(a_lo, b0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(a_hi, b0, acc_hi0);
            acc_lo1 = _mm256_fmadd_ps(a_lo, b1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(a_hi, b1, acc_hi1);
            acc_lo2 = _mm256_fmadd_ps(a_lo, b2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(a_hi, b2, acc_hi2);
            acc_lo3 = _mm256_fmadd_ps(a_lo, b3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(a_hi, b3, acc_hi3);
            __m256 b4 = _mm256_broadcast_ss(b_row + 4);
            __m256 b5 = _mm256_broadcast_ss(b_row + 5);
            __m256 b6 = _mm256_broadcast_ss(b_row + 6);
            __m256 b7 = _mm256_broadcast_ss(b_row + 7);
            acc_lo4 = _mm256_fmadd_ps(a_lo, b4, acc_lo4);
            acc_hi4 = _mm256_fmadd_ps(a_hi, b4, acc_hi4);
            acc_lo5 = _mm256_fmadd_ps(a_lo, b5, acc_lo5);
            acc_hi5 = _mm256_fmadd_ps(a_hi, b5, acc_hi5);
            acc_lo6 = _mm256_fmadd_ps(a_lo, b6, acc_lo6);
            acc_hi6 = _mm256_fmadd_ps(a_hi, b6, acc_hi6);
            acc_lo7 = _mm256_fmadd_ps(a_lo, b7, acc_lo7);
            acc_hi7 = _mm256_fmadd_ps(a_hi, b7, acc_hi7);
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 8) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 16 && n == 8)
    {
        __m256 cols_lo[8] = {acc_lo0, acc_lo1, acc_lo2, acc_lo3, acc_lo4, acc_lo5, acc_lo6, acc_lo7};
        __m256 cols_hi[8] = {acc_hi0, acc_hi1, acc_hi2, acc_hi3, acc_hi4, acc_hi5, acc_hi6, acc_hi7};
        gemm_transpose_8x8_avx2(cols_lo);
        gemm_transpose_8x8_avx2(cols_hi);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt)
                GEMM_STREAM_PS(cr, cols_lo[r]);
            else
                GEMM_STORE_PS(cr, cols_lo[r]);
        }
        for (size_t r = 8; r < 16; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt)
                GEMM_STREAM_PS(cr, cols_hi[r - 8]);
            else
                GEMM_STORE_PS(cr, cols_hi[r - 8]);
        }
    }
    else
    {
        alignas(32) float temp[16 * 8];
        GEMM_STORE_PS(temp + 0 * 16, acc_lo0);
        GEMM_STORE_PS(temp + 0 * 16 + 8, acc_hi0);
        GEMM_STORE_PS(temp + 1 * 16, acc_lo1);
        GEMM_STORE_PS(temp + 1 * 16 + 8, acc_hi1);
        GEMM_STORE_PS(temp + 2 * 16, acc_lo2);
        GEMM_STORE_PS(temp + 2 * 16 + 8, acc_hi2);
        GEMM_STORE_PS(temp + 3 * 16, acc_lo3);
        GEMM_STORE_PS(temp + 3 * 16 + 8, acc_hi3);
        GEMM_STORE_PS(temp + 4 * 16, acc_lo4);
        GEMM_STORE_PS(temp + 4 * 16 + 8, acc_hi4);
        GEMM_STORE_PS(temp + 5 * 16, acc_lo5);
        GEMM_STORE_PS(temp + 5 * 16 + 8, acc_hi5);
        GEMM_STORE_PS(temp + 6 * 16, acc_lo6);
        GEMM_STORE_PS(temp + 6 * 16 + 8, acc_hi6);
        GEMM_STORE_PS(temp + 7 * 16, acc_lo7);
        GEMM_STORE_PS(temp + 7 * 16 + 8, acc_hi7);

        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 16, r, n);
            GEMM_MASKSTORE_PS(cr, mask, sum);
        }
    }
}

//==============================================================================
// 16×6 KERNELS
//==============================================================================

/**
 * @brief 16×6 kernel (ADD): C += A*B
 */
static inline void gemm_16x6_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    assert(a_k_stride == 16 && "16x6 kernel requires A packed with MR=16");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo[6], acc_hi[6];
    for (int j = 0; j < 6; ++j)
    {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    gemm_prefetch_c_rows(c, ldc, m);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
            gemm_prefetch_panels(Bp, Ap, k, Kblk, b_k_stride, a_k_stride, PF_LONG); // ← USE STRIDES

        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = GEMM_LOAD_PS(Ap + kk * a_k_stride);     // ← USE STRIDE
            __m256 a_hi = GEMM_LOAD_PS(Ap + kk * a_k_stride + 8); // ← USE STRIDE
            const float *b_row = Bp + kk * b_k_stride;              // ← USE STRIDE
            for (int j = 0; j < 6; ++j)
            {
                __m256 b = _mm256_broadcast_ss(b_row + j);
                acc_lo[j] = _mm256_fmadd_ps(a_lo, b, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(a_hi, b, acc_hi[j]);
            }
        }
    }

    if (m == 16 && n == 6)
    {
        alignas(32) float temp_lo[8 * 8] = {0}, temp_hi[8 * 8] = {0};
        for (int j = 0; j < 6; ++j)
        {
            GEMM_STORE_PS(temp_lo + j * 8, acc_lo[j]);
            GEMM_STORE_PS(temp_hi + j * 8, acc_hi[j]);
        }
        __m256 cols_lo[8], cols_hi[8];
        for (int j = 0; j < 8; ++j)
        {
            cols_lo[j] = GEMM_LOAD_PS(temp_lo + j * 8);
            cols_hi[j] = GEMM_LOAD_PS(temp_hi + j * 8);
        }
        gemm_transpose_8x8_avx2(cols_lo);
        gemm_transpose_8x8_avx2(cols_hi);

        __m256i mask6;
        gemm_build_mask_avx2(6, &mask6);

        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            __m256 old = GEMM_MASKLOAD_PS(cr, mask6);
            __m256 result = _mm256_add_ps(old, cols_lo[r]);
            GEMM_MASKSTORE_PS(cr, mask6, result);
        }
        for (size_t r = 8; r < 16; ++r)
        {
            float *cr = c + r * ldc;
            __m256 old = GEMM_MASKLOAD_PS(cr, mask6);
            __m256 result = _mm256_add_ps(old, cols_hi[r - 8]);
            GEMM_MASKSTORE_PS(cr, mask6, result);
        }
        return;
    }
    else
    {
        alignas(32) float temp[16 * 6];
        for (int j = 0; j < 6; ++j)
        {
            GEMM_STORE_PS(temp + j * 16, acc_lo[j]);
            GEMM_STORE_PS(temp + j * 16 + 8, acc_hi[j]);
        }
        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 16, r, n);
            __m256 old = GEMM_MASKLOAD_PS(cr, mask);
            GEMM_MASKSTORE_PS(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/**
 * @brief 16×6 kernel (STORE): C = A*B
 */
static inline void gemm_16x6_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    assert(a_k_stride == 16 && "16x6 kernel requires A packed with MR=16");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo[6], acc_hi[6];
    for (int j = 0; j < 6; ++j)
    {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    gemm_prefetch_c_rows(c, ldc, m);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
            gemm_prefetch_panels(Bp, Ap, k, Kblk, b_k_stride, a_k_stride, PF_LONG); // ← USE STRIDES

        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = GEMM_LOAD_PS(Ap + kk * a_k_stride);     // ← USE STRIDE
            __m256 a_hi = GEMM_LOAD_PS(Ap + kk * a_k_stride + 8); // ← USE STRIDE
            const float *b_row = Bp + kk * b_k_stride;              // ← USE STRIDE
            for (int j = 0; j < 6; ++j)
            {
                __m256 b = _mm256_broadcast_ss(b_row + j);
                acc_lo[j] = _mm256_fmadd_ps(a_lo, b, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(a_hi, b, acc_hi[j]);
            }
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 6) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 16 && n == 6)
    {
        alignas(32) float temp_lo[8 * 8] = {0}, temp_hi[8 * 8] = {0};
        for (int j = 0; j < 6; ++j)
        {
            GEMM_STORE_PS(temp_lo + j * 8, acc_lo[j]);
            GEMM_STORE_PS(temp_hi + j * 8, acc_hi[j]);
        }
        __m256 cols_lo[8], cols_hi[8];
        for (int j = 0; j < 8; ++j)
        {
            cols_lo[j] = GEMM_LOAD_PS(temp_lo + j * 8);
            cols_hi[j] = GEMM_LOAD_PS(temp_hi + j * 8);
        }
        gemm_transpose_8x8_avx2(cols_lo);
        gemm_transpose_8x8_avx2(cols_hi);

        __m256i mask6;
        gemm_build_mask_avx2(6, &mask6);

        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            GEMM_MASKSTORE_PS(cr, mask6, cols_lo[r]);
        }
        for (size_t r = 8; r < 16; ++r)
        {
            float *cr = c + r * ldc;
            GEMM_MASKSTORE_PS(cr, mask6, cols_hi[r - 8]);
        }
        return;
    }
    else
    {
        alignas(32) float temp[16 * 6];
        for (int j = 0; j < 6; ++j)
        {
            GEMM_STORE_PS(temp + j * 16, acc_lo[j]);
            GEMM_STORE_PS(temp + j * 16 + 8, acc_hi[j]);
        }
        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 16, r, n);
            if (use_nt && n == 6)
                GEMM_STREAM_PS(cr, sum);
            else
                GEMM_MASKSTORE_PS(cr, mask, sum);
        }
    }
}

//==============================================================================
// 8×6 KERNELS
//==============================================================================

/**
 * @brief 8×6 kernel (ADD): C += A*B
 */
static inline void gemm_8x6_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    assert(a_k_stride == 8 && "8x6 kernel requires A packed with MR=8");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    gemm_prefetch_c_rows(c, ldc, m);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
            gemm_prefetch_panels(Bp, Ap, k, Kblk, b_k_stride, a_k_stride, PF_LONG); // ← USE STRIDES

        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = GEMM_LOAD_PS(Ap + kk * a_k_stride); // ← USE STRIDE
            const float *b_row = Bp + kk * b_k_stride;       // ← USE STRIDE
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
        }
    }

    if (m == 8 && n == 6)
    {
        alignas(32) float temp[8 * 8] = {0};
        GEMM_STORE_PS(temp + 0 * 8, acc0);
        GEMM_STORE_PS(temp + 1 * 8, acc1);
        GEMM_STORE_PS(temp + 2 * 8, acc2);
        GEMM_STORE_PS(temp + 3 * 8, acc3);
        GEMM_STORE_PS(temp + 4 * 8, acc4);
        GEMM_STORE_PS(temp + 5 * 8, acc5);

        __m256 cols[8];
        for (int j = 0; j < 8; ++j)
            cols[j] = GEMM_LOAD_PS(temp + j * 8);

        gemm_transpose_8x8_avx2(cols);

        __m256i mask6;
        gemm_build_mask_avx2(6, &mask6);

        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            __m256 old = GEMM_MASKLOAD_PS(cr, mask6);
            __m256 result = _mm256_add_ps(old, cols[r]);
            GEMM_MASKSTORE_PS(cr, mask6, result);
        }
        return;
    }
    else
    {
        alignas(32) float temp[8 * 6];
        GEMM_STORE_PS(temp + 0 * 8, acc0);
        GEMM_STORE_PS(temp + 1 * 8, acc1);
        GEMM_STORE_PS(temp + 2 * 8, acc2);
        GEMM_STORE_PS(temp + 3 * 8, acc3);
        GEMM_STORE_PS(temp + 4 * 8, acc4);
        GEMM_STORE_PS(temp + 5 * 8, acc5);
        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            __m256 old = GEMM_MASKLOAD_PS(cr, mask);
            GEMM_MASKSTORE_PS(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/**
 * @brief 8×6 kernel (STORE): C = A*B
 */
static inline void gemm_8x6_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    assert(a_k_stride == 8 && "8x6 kernel requires A packed with MR=8");
    assert(b_k_stride == 16 && "All kernels require B stride=16");

    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    gemm_prefetch_c_rows(c, ldc, m);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
            gemm_prefetch_panels(Bp, Ap, k, Kblk, b_k_stride, a_k_stride, PF_LONG); // ← USE STRIDES

        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = GEMM_LOAD_PS(Ap + kk * a_k_stride); // ← USE STRIDE
            const float *b_row = Bp + kk * b_k_stride;       // ← USE STRIDE
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 8) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 8 && n == 6)
    {
        alignas(32) float temp[8 * 8] = {0};
        GEMM_STORE_PS(temp + 0 * 8, acc0);
        GEMM_STORE_PS(temp + 1 * 8, acc1);
        GEMM_STORE_PS(temp + 2 * 8, acc2);
        GEMM_STORE_PS(temp + 3 * 8, acc3);
        GEMM_STORE_PS(temp + 4 * 8, acc4);
        GEMM_STORE_PS(temp + 5 * 8, acc5);

        __m256 cols[8];
        for (int j = 0; j < 8; ++j)
            cols[j] = GEMM_LOAD_PS(temp + j * 8);

        gemm_transpose_8x8_avx2(cols);

        __m256i mask6;
        gemm_build_mask_avx2(6, &mask6);

        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            GEMM_MASKSTORE_PS(cr, mask6, cols[r]);
        }
        return;
    }
    else
    {
        alignas(32) float temp[8 * 6];
        GEMM_STORE_PS(temp + 0 * 8, acc0);
        GEMM_STORE_PS(temp + 1 * 8, acc1);
        GEMM_STORE_PS(temp + 2 * 8, acc2);
        GEMM_STORE_PS(temp + 3 * 8, acc3);
        GEMM_STORE_PS(temp + 4 * 8, acc4);
        GEMM_STORE_PS(temp + 5 * 8, acc5);
        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            GEMM_MASKSTORE_PS(cr, mask, sum);
        }
    }
}

//==============================================================================
// 8×16 KERNEL (ADD): C += A*B
//==============================================================================

static inline void gemm_8x16_panel_avx2fma_add(
    float *RESTRICT c,
    size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk,
    size_t m,
    size_t n,
    __m256i mask_lo,
    __m256i mask_hi)
{
    assert((a_k_stride == 8 || a_k_stride == 16) && "8x16 kernel requires stride 8 or 16");


    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    if (m == 8 && n == 16)
    {
        __m256 c00, c01, c10, c11, c20, c21, c30, c31;
        __m256 c40, c41, c50, c51, c60, c61, c70, c71;

        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();
        c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps();
        c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps();
        c31 = _mm256_setzero_ps();
        c40 = _mm256_setzero_ps();
        c41 = _mm256_setzero_ps();
        c50 = _mm256_setzero_ps();
        c51 = _mm256_setzero_ps();
        c60 = _mm256_setzero_ps();
        c61 = _mm256_setzero_ps();
        c70 = _mm256_setzero_ps();
        c71 = _mm256_setzero_ps();

        const float *a = Ap;
        const float *b = Bp;

        for (size_t k = 0; k < Kblk; ++k)
        {
            if (k + 1 < Kblk)
            {
                PREFETCH_T0(a + a_k_stride); // ← USE STRIDE
                PREFETCH_T0(b + b_k_stride); // ← USE STRIDE
            }

            __m256 b0 = GEMM_LOAD_PS(b);
            __m256 b1 = GEMM_LOAD_PS(b + 8);

            __m256 a0 = _mm256_broadcast_ss(a + 0);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            __m256 a1 = _mm256_broadcast_ss(a + 1);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            __m256 a2 = _mm256_broadcast_ss(a + 2);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);

            __m256 a3 = _mm256_broadcast_ss(a + 3);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);

            __m256 a4 = _mm256_broadcast_ss(a + 4);
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);

            __m256 a5 = _mm256_broadcast_ss(a + 5);
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);

            __m256 a6 = _mm256_broadcast_ss(a + 6);
            c60 = _mm256_fmadd_ps(a6, b0, c60);
            c61 = _mm256_fmadd_ps(a6, b1, c61);

            __m256 a7 = _mm256_broadcast_ss(a + 7);
            c70 = _mm256_fmadd_ps(a7, b0, c70);
            c71 = _mm256_fmadd_ps(a7, b1, c71);

            a += a_k_stride; // ← USE STRIDE
            b += b_k_stride; // ← USE STRIDE
        }

        float *c0 = c;
        __m256 old00 = _mm256_loadu_ps(c0);
        __m256 old01 = _mm256_loadu_ps(c0 + 8);
        _mm256_storeu_ps(c0, _mm256_add_ps(old00, c00));
        _mm256_storeu_ps(c0 + 8, _mm256_add_ps(old01, c01));

        c0 += ldc;
        __m256 old10 = _mm256_loadu_ps(c0);
        __m256 old11 = _mm256_loadu_ps(c0 + 8);
        _mm256_storeu_ps(c0, _mm256_add_ps(old10, c10));
        _mm256_storeu_ps(c0 + 8, _mm256_add_ps(old11, c11));

        c0 += ldc;
        __m256 old20 = _mm256_loadu_ps(c0);
        __m256 old21 = _mm256_loadu_ps(c0 + 8);
        _mm256_storeu_ps(c0, _mm256_add_ps(old20, c20));
        _mm256_storeu_ps(c0 + 8, _mm256_add_ps(old21, c21));

        c0 += ldc;
        __m256 old30 = _mm256_loadu_ps(c0);
        __m256 old31 = _mm256_loadu_ps(c0 + 8);
        _mm256_storeu_ps(c0, _mm256_add_ps(old30, c30));
        _mm256_storeu_ps(c0 + 8, _mm256_add_ps(old31, c31));

        c0 += ldc;
        __m256 old40 = _mm256_loadu_ps(c0);
        __m256 old41 = _mm256_loadu_ps(c0 + 8);
        _mm256_storeu_ps(c0, _mm256_add_ps(old40, c40));
        _mm256_storeu_ps(c0 + 8, _mm256_add_ps(old41, c41));

        c0 += ldc;
        __m256 old50 = _mm256_loadu_ps(c0);
        __m256 old51 = _mm256_loadu_ps(c0 + 8);
        _mm256_storeu_ps(c0, _mm256_add_ps(old50, c50));
        _mm256_storeu_ps(c0 + 8, _mm256_add_ps(old51, c51));

        c0 += ldc;
        __m256 old60 = _mm256_loadu_ps(c0);
        __m256 old61 = _mm256_loadu_ps(c0 + 8);
        _mm256_storeu_ps(c0, _mm256_add_ps(old60, c60));
        _mm256_storeu_ps(c0 + 8, _mm256_add_ps(old61, c61));

        c0 += ldc;
        __m256 old70 = _mm256_loadu_ps(c0);
        __m256 old71 = _mm256_loadu_ps(c0 + 8);
        _mm256_storeu_ps(c0, _mm256_add_ps(old70, c70));
        _mm256_storeu_ps(c0 + 8, _mm256_add_ps(old71, c71));

        return;
    }

    // SLOW PATH WITH CRITICAL FIX
    alignas(32) float temp[8 * 16];
    memset(temp, 0, sizeof(temp));

    for (size_t k = 0; k < Kblk; ++k)
    {
        const float *b = Bp + k * b_k_stride; // ← USE STRIDE
        __m256 b0 = GEMM_LOAD_PS(b);
        __m256 b1 = GEMM_LOAD_PS(b + 8);

        for (size_t r = 0; r < m; ++r)
        {
            // CRITICAL FIX: Column-major access with stride
            float a_val = Ap[k * a_k_stride + r]; // ← CORRECTED!
            __m256 a_broadcast = _mm256_broadcast_ss(&a_val);

            __m256 t0 = GEMM_LOAD_PS(temp + r * 16);
            __m256 t1 = GEMM_LOAD_PS(temp + r * 16 + 8);

            t0 = _mm256_fmadd_ps(a_broadcast, b0, t0);
            t1 = _mm256_fmadd_ps(a_broadcast, b1, t1);

            GEMM_STORE_PS(temp + r * 16, t0);
            GEMM_STORE_PS(temp + r * 16 + 8, t1);
        }
    }

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 t0 = GEMM_LOAD_PS(temp + r * 16);
        __m256 t1 = GEMM_LOAD_PS(temp + r * 16 + 8);

        if (n <= 8)
        {
            __m256 old = GEMM_MASKLOAD_PS(cr, mask_lo);
            GEMM_MASKSTORE_PS(cr, mask_lo, _mm256_add_ps(old, t0));
        }
        else if (n == 16)
        {
            __m256 old0 = _mm256_loadu_ps(cr);
            __m256 old1 = _mm256_loadu_ps(cr + 8);
            _mm256_storeu_ps(cr, _mm256_add_ps(old0, t0));
            _mm256_storeu_ps(cr + 8, _mm256_add_ps(old1, t1));
        }
        else
        {
            __m256 old0 = _mm256_loadu_ps(cr);
            _mm256_storeu_ps(cr, _mm256_add_ps(old0, t0));

            __m256 old1 = GEMM_MASKLOAD_PS(cr + 8, mask_hi);
            GEMM_MASKSTORE_PS(cr + 8, mask_hi, _mm256_add_ps(old1, t1));
        }
    }
}

//==============================================================================
// 8×16 KERNEL (STORE): C = A*B
//==============================================================================

/**
 * @brief FULLY OPTIMIZED 8x16 kernel with dual mask support - STORE variant
 * Overwrites C = A * B (no accumulation)
 */
static inline void gemm_8x16_panel_avx2fma_store(
    float *RESTRICT c,
    size_t ldc,
    const float *RESTRICT Ap, size_t a_k_stride, // ← ADDED
    const float *RESTRICT Bp, size_t b_k_stride, // ← ADDED
    size_t Kblk,
    size_t m,
    size_t n,
    __m256i mask_lo,
    __m256i mask_hi)
{
    assert((a_k_stride == 8 || a_k_stride == 16) && "8x16 kernel requires stride 8 or 16");

    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    if (m == 8 && n == 16)
    {
        __m256 c00, c01, c10, c11, c20, c21, c30, c31;
        __m256 c40, c41, c50, c51, c60, c61, c70, c71;

        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();
        c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps();
        c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps();
        c31 = _mm256_setzero_ps();
        c40 = _mm256_setzero_ps();
        c41 = _mm256_setzero_ps();
        c50 = _mm256_setzero_ps();
        c51 = _mm256_setzero_ps();
        c60 = _mm256_setzero_ps();
        c61 = _mm256_setzero_ps();
        c70 = _mm256_setzero_ps();
        c71 = _mm256_setzero_ps();

        const float *a = Ap;
        const float *b = Bp;

        size_t k = 0;
        for (; k + 1 < Kblk; k += 2)
        {
            if (k + 8 < Kblk)
            {
                PREFETCH_T0(a + 2 * a_k_stride); // ← USE STRIDE
                PREFETCH_T0(b + 2 * b_k_stride); // ← USE STRIDE
            }

            // Iteration k
            __m256 b0_k0 = GEMM_LOAD_PS(b);
            __m256 b1_k0 = GEMM_LOAD_PS(b + 8);

            __m256 a0_k0 = _mm256_broadcast_ss(a + 0);
            c00 = _mm256_fmadd_ps(a0_k0, b0_k0, c00);
            c01 = _mm256_fmadd_ps(a0_k0, b1_k0, c01);

            __m256 a1_k0 = _mm256_broadcast_ss(a + 1);
            c10 = _mm256_fmadd_ps(a1_k0, b0_k0, c10);
            c11 = _mm256_fmadd_ps(a1_k0, b1_k0, c11);

            __m256 a2_k0 = _mm256_broadcast_ss(a + 2);
            c20 = _mm256_fmadd_ps(a2_k0, b0_k0, c20);
            c21 = _mm256_fmadd_ps(a2_k0, b1_k0, c21);

            __m256 a3_k0 = _mm256_broadcast_ss(a + 3);
            c30 = _mm256_fmadd_ps(a3_k0, b0_k0, c30);
            c31 = _mm256_fmadd_ps(a3_k0, b1_k0, c31);

            __m256 a4_k0 = _mm256_broadcast_ss(a + 4);
            c40 = _mm256_fmadd_ps(a4_k0, b0_k0, c40);
            c41 = _mm256_fmadd_ps(a4_k0, b1_k0, c41);

            __m256 a5_k0 = _mm256_broadcast_ss(a + 5);
            c50 = _mm256_fmadd_ps(a5_k0, b0_k0, c50);
            c51 = _mm256_fmadd_ps(a5_k0, b1_k0, c51);

            __m256 a6_k0 = _mm256_broadcast_ss(a + 6);
            c60 = _mm256_fmadd_ps(a6_k0, b0_k0, c60);
            c61 = _mm256_fmadd_ps(a6_k0, b1_k0, c61);

            __m256 a7_k0 = _mm256_broadcast_ss(a + 7);
            c70 = _mm256_fmadd_ps(a7_k0, b0_k0, c70);
            c71 = _mm256_fmadd_ps(a7_k0, b1_k0, c71);

            // Iteration k+1
            __m256 b0_k1 = GEMM_LOAD_PS(b + b_k_stride);     // ← USE STRIDE
            __m256 b1_k1 = GEMM_LOAD_PS(b + b_k_stride + 8); // ← USE STRIDE

            __m256 a0_k1 = _mm256_broadcast_ss(a + a_k_stride + 0); // ← USE STRIDE
            c00 = _mm256_fmadd_ps(a0_k1, b0_k1, c00);
            c01 = _mm256_fmadd_ps(a0_k1, b1_k1, c01);

            __m256 a1_k1 = _mm256_broadcast_ss(a + a_k_stride + 1); // ← USE STRIDE
            c10 = _mm256_fmadd_ps(a1_k1, b0_k1, c10);
            c11 = _mm256_fmadd_ps(a1_k1, b1_k1, c11);

            __m256 a2_k1 = _mm256_broadcast_ss(a + a_k_stride + 2); // ← USE STRIDE
            c20 = _mm256_fmadd_ps(a2_k1, b0_k1, c20);
            c21 = _mm256_fmadd_ps(a2_k1, b1_k1, c21);

            __m256 a3_k1 = _mm256_broadcast_ss(a + a_k_stride + 3); // ← USE STRIDE
            c30 = _mm256_fmadd_ps(a3_k1, b0_k1, c30);
            c31 = _mm256_fmadd_ps(a3_k1, b1_k1, c31);

            __m256 a4_k1 = _mm256_broadcast_ss(a + a_k_stride + 4); // ← USE STRIDE
            c40 = _mm256_fmadd_ps(a4_k1, b0_k1, c40);
            c41 = _mm256_fmadd_ps(a4_k1, b1_k1, c41);

            __m256 a5_k1 = _mm256_broadcast_ss(a + a_k_stride + 5); // ← USE STRIDE
            c50 = _mm256_fmadd_ps(a5_k1, b0_k1, c50);
            c51 = _mm256_fmadd_ps(a5_k1, b1_k1, c51);

            __m256 a6_k1 = _mm256_broadcast_ss(a + a_k_stride + 6); // ← USE STRIDE
            c60 = _mm256_fmadd_ps(a6_k1, b0_k1, c60);
            c61 = _mm256_fmadd_ps(a6_k1, b1_k1, c61);

            __m256 a7_k1 = _mm256_broadcast_ss(a + a_k_stride + 7); // ← USE STRIDE
            c70 = _mm256_fmadd_ps(a7_k1, b0_k1, c70);
            c71 = _mm256_fmadd_ps(a7_k1, b1_k1, c71);

            a += 2 * a_k_stride; // ← USE STRIDE
            b += 2 * b_k_stride; // ← USE STRIDE
        }

        if (k < Kblk)
        {
            __m256 b0 = GEMM_LOAD_PS(b);
            __m256 b1 = GEMM_LOAD_PS(b + 8);

            __m256 a0 = _mm256_broadcast_ss(a + 0);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            __m256 a1 = _mm256_broadcast_ss(a + 1);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            __m256 a2 = _mm256_broadcast_ss(a + 2);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);

            __m256 a3 = _mm256_broadcast_ss(a + 3);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);

            __m256 a4 = _mm256_broadcast_ss(a + 4);
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);

            __m256 a5 = _mm256_broadcast_ss(a + 5);
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);

            __m256 a6 = _mm256_broadcast_ss(a + 6);
            c60 = _mm256_fmadd_ps(a6, b0, c60);
            c61 = _mm256_fmadd_ps(a6, b1, c61);

            __m256 a7 = _mm256_broadcast_ss(a + 7);
            c70 = _mm256_fmadd_ps(a7, b0, c70);
            c71 = _mm256_fmadd_ps(a7, b1, c71);
        }

        float *c0 = c;
        _mm256_storeu_ps(c0, c00);
        _mm256_storeu_ps(c0 + 8, c01);

        c0 += ldc;
        _mm256_storeu_ps(c0, c10);
        _mm256_storeu_ps(c0 + 8, c11);

        c0 += ldc;
        _mm256_storeu_ps(c0, c20);
        _mm256_storeu_ps(c0 + 8, c21);

        c0 += ldc;
        _mm256_storeu_ps(c0, c30);
        _mm256_storeu_ps(c0 + 8, c31);

        c0 += ldc;
        _mm256_storeu_ps(c0, c40);
        _mm256_storeu_ps(c0 + 8, c41);

        c0 += ldc;
        _mm256_storeu_ps(c0, c50);
        _mm256_storeu_ps(c0 + 8, c51);

        c0 += ldc;
        _mm256_storeu_ps(c0, c60);
        _mm256_storeu_ps(c0 + 8, c61);

        c0 += ldc;
        _mm256_storeu_ps(c0, c70);
        _mm256_storeu_ps(c0 + 8, c71);

        return;
    }

    // SLOW PATH WITH CRITICAL FIX
    alignas(32) float temp[8 * 16];
    memset(temp, 0, sizeof(temp));

    for (size_t k = 0; k < Kblk; ++k)
    {
        const float *b = Bp + k * b_k_stride; // ← USE STRIDE

        if (k + 1 < Kblk)
        {
            PREFETCH_T0(b + b_k_stride);            // ← USE STRIDE
            PREFETCH_T0(Ap + (k + 1) * a_k_stride); // ← USE STRIDE
        }

        __m256 b0 = GEMM_LOAD_PS(b);
        __m256 b1 = GEMM_LOAD_PS(b + 8);

        size_t r = 0;
        for (; r + 1 < m; r += 2)
        {
            // CRITICAL FIX: Column-major access with stride
            float a_val0 = Ap[k * a_k_stride + r]; // ← CORRECTED!
            __m256 a_broadcast0 = _mm256_broadcast_ss(&a_val0);
            __m256 t00 = GEMM_LOAD_PS(temp + r * 16);
            __m256 t01 = GEMM_LOAD_PS(temp + r * 16 + 8);
            t00 = _mm256_fmadd_ps(a_broadcast0, b0, t00);
            t01 = _mm256_fmadd_ps(a_broadcast0, b1, t01);
            GEMM_STORE_PS(temp + r * 16, t00);
            GEMM_STORE_PS(temp + r * 16 + 8, t01);

            // CRITICAL FIX: Column-major access with stride
            float a_val1 = Ap[k * a_k_stride + (r + 1)]; // ← CORRECTED!
            __m256 a_broadcast1 = _mm256_broadcast_ss(&a_val1);
            __m256 t10 = GEMM_LOAD_PS(temp + (r + 1) * 16);
            __m256 t11 = GEMM_LOAD_PS(temp + (r + 1) * 16 + 8);
            t10 = _mm256_fmadd_ps(a_broadcast1, b0, t10);
            t11 = _mm256_fmadd_ps(a_broadcast1, b1, t11);
            GEMM_STORE_PS(temp + (r + 1) * 16, t10);
            GEMM_STORE_PS(temp + (r + 1) * 16 + 8, t11);
        }

        if (r < m)
        {
            // CRITICAL FIX: Column-major access with stride
            float a_val = Ap[k * a_k_stride + r]; // ← CORRECTED!
            __m256 a_broadcast = _mm256_broadcast_ss(&a_val);
            __m256 t0 = GEMM_LOAD_PS(temp + r * 16);
            __m256 t1 = GEMM_LOAD_PS(temp + r * 16 + 8);
            t0 = _mm256_fmadd_ps(a_broadcast, b0, t0);
            t1 = _mm256_fmadd_ps(a_broadcast, b1, t1);
            GEMM_STORE_PS(temp + r * 16, t0);
            GEMM_STORE_PS(temp + r * 16 + 8, t1);
        }
    }

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 t0 = GEMM_LOAD_PS(temp + r * 16);
        __m256 t1 = GEMM_LOAD_PS(temp + r * 16 + 8);

        if (n <= 8)
        {
            GEMM_MASKSTORE_PS(cr, mask_lo, t0);
        }
        else if (n == 16)
        {
            _mm256_storeu_ps(cr, t0);
            _mm256_storeu_ps(cr + 8, t1);
        }
        else
        {
            _mm256_storeu_ps(cr, t0);
            GEMM_MASKSTORE_PS(cr + 8, mask_hi, t1);
        }
    }
}

#endif // GEMM_KERNELS_AVX2_COMPLETE_H