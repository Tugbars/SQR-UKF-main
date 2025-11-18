/**
 * @file gemm_large.c
 * @brief Tier 2: Blocked GEMM Execution with SIMD-Optimized Packing
 *
 * This module implements the production GEMM execution layer that handles
 * matrices too large for Tier 1 fixed-size kernels. It orchestrates:
 *
 * **Execution Pipeline:**
 * 1. Beta pre-scaling (once, before K-loop)
 * 2. NC → KC → MC blocking (maximize B panel reuse in L2)
 * 3. SIMD-optimized packing (1.5-2× faster than scalar)
 * 4. Micro-kernel dispatch with pre-selected kernels
 *
 * **Key Optimizations:**
 * - Pre-computed tile counts (no division in hot path)
 * - Pre-selected kernels for full tiles (no dispatch overhead)
 * - Kernel selection only for edge tiles (~5% of work)
 * - Alpha absorption into packed A (reduces multiply count)
 * - Beta pre-scaling (eliminates per-tile overhead)
 *
 * **Cache Strategy:**
 * The NC → KC → MC loop order is critical for performance:
 * - B panels (KC × NR) are packed once and reused across all MC tiles
 * - Maximizes L2 cache hit rate (~98% for large matrices)
 * - Minimizes memory traffic (packing overhead ~5%)
 *
 * **Memory Layout:**
 * ```
 * Packed A: [k=0: m0 m1 ... m7/15][k=1: m0 m1 ...][...]
 *           └──── MR elements ────┘
 *
 * Packed B: [k=0: n0 n1 ... n15][k=1: n0 n1 ...][...]
 *           └──── 16 elements ──┘ (fixed stride)
 * ```
 *
 * @author TUGBARS
 * @date 2025
 */

#include "gemm_kernels_avx2.h"
#include "gemm_simd_ops.h"
#include "gemm_small.h"
#include "gemm_planning.h"
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

//==============================================================================
// STRIDE DESCRIPTOR
//==============================================================================

/**
 * @brief Stride information for packed panels
 *
 * The packing functions return stride information so the execution
 * loop knows how to index into packed buffers:
 *
 * - **A stride**: Distance between consecutive K-iterations in packed A
 * - **B stride**: Distance between consecutive K-iterations in packed B
 *
 * **Usage:**
 * ```c
 * pack_strides_t s = pack_A_panel_simd(...);
 * float *a_k0 = Ap + 0 * s.a_k_stride; // K=0
 * float *a_k1 = Ap + 1 * s.a_k_stride; // K=1
 * ```
 *
 * @note Only the relevant stride is set; the other is 0
 */
typedef struct
{
    size_t a_k_stride; /**< Stride for A panel (elements between K-iterations) */
    size_t b_k_stride; /**< Stride for B panel (elements between K-iterations) */
} pack_strides_t;

//==============================================================================
// BETA PRE-SCALING
//==============================================================================

/**
 * @brief Pre-scale output matrix C by beta (before accumulation)
 *
 * Applies the beta scaling factor to the entire C matrix ONCE before
 * the K-loop begins. This is more efficient than scaling during each
 * tile update.
 *
 * **BLAS Semantics:**
 * - beta = 0.0: Zero C (does not read old values, just memset)
 * - beta = 1.0: No-op (C unchanged, accumulation mode)
 * - beta ≠ 1.0: Scale C *= beta (general case)
 *
 * **Vectorization:**
 * - Main loop: 8 elements per iteration (AVX2)
 * - Tail loop: Scalar for remaining elements
 *
 * @param C     Output matrix (M × N, row-major)
 * @param M     Number of rows in C
 * @param N     Number of columns in C
 * @param beta  Scalar multiplier for C
 *
 * @note Called once per gemm_execute_plan(), not per tile
 * @note For beta=0, uses memset (fastest path)
 * @note For beta=1, does nothing (no memory access)
 *
 * @see gemm_execute_plan()
 */
static void scale_matrix_beta(
    float *restrict C,
    size_t M, size_t N,
    float beta)
{
    if (beta == 0.0f)
    {
        // Fast path: Just zero (don't read old values)
        memset(C, 0, M * N * sizeof(float));
    }
    else if (beta != 1.0f)
    {
        // General case: C *= beta
        __m256 vbeta = _mm256_set1_ps(beta);

        for (size_t i = 0; i < M; ++i)
        {
            float *row = C + i * N;
            size_t j = 0;

            // Vectorized main loop
            for (; j + 7 < N; j += 8)
            {
                __m256 c = _mm256_loadu_ps(row + j);
                _mm256_storeu_ps(row + j, _mm256_mul_ps(c, vbeta));
            }

            // Scalar tail
            for (; j < N; ++j)
            {
                row[j] *= beta;
            }
        }
    }
    // else beta == 1.0f: no-op (accumulation mode)
}

//==============================================================================
// SIMD-OPTIMIZED PACKING
//==============================================================================

/**
 * @brief Pack A panel with SIMD gather and alpha scaling
 *
 * Packs a submatrix of A from column-major storage to row-major packed format,
 * applying alpha scaling during the pack operation.
 *
 * **Input Layout (A):** Row-major, A[i,k] = A[i*K + k]
 * ```
 * A[i0, k0]   A[i0, k0+1]   ...
 * A[i0+1, k0] A[i0+1, k0+1] ...
 * ...
 * ```
 *
 * **Output Layout (Ap):** Column-major (K-outer), Ap[k][i] format
 * ```
 * K=0: [m0 m1 m2 ... m7/15]
 * K=1: [m0 m1 m2 ... m7/15]
 * ...
 * ```
 *
 * **SIMD Optimization:**
 * - Gathers 8 rows at once using `_mm256_set_ps()`
 * - 1.5-2× faster than scalar packing
 * - Prefetches K+8 to hide memory latency
 *
 * **Alpha Scaling:**
 * - alpha = 1.0: Skip multiplication (fast path)
 * - alpha ≠ 1.0: Apply during pack (Ap = alpha * A)
 *
 * **Buffer Safety:**
 * - Always allocates requested_mr rows (physical capacity)
 * - Only fills ib rows (logical content)
 * - Remaining rows are zeroed (safe for edge tiles)
 *
 * @param[out] Ap          Packed output buffer (kb × requested_mr floats)
 * @param[in]  A           Source matrix (M × K, row-major)
 * @param[in]  M           Total rows in A (for indexing, unused)
 * @param[in]  K           Total columns in A (for indexing)
 * @param[in]  i0          Starting row index in A
 * @param[in]  ib          Number of rows to pack (≤ requested_mr)
 * @param[in]  k0          Starting column index in A
 * @param[in]  kb          Number of columns to pack
 * @param[in]  alpha       Scalar multiplier (absorbed into pack)
 * @param[in]  requested_mr Expected MR from planner (8 or 16)
 *
 * @return Stride information (a_k_stride = requested_mr)
 *
 * @pre ib <= requested_mr (planner must ensure this)
 * @pre Ap must have space for kb × requested_mr floats
 *
 * @note Prefetching distance (8) tuned for Intel 14900K
 * @note Zeroes entire buffer first (safety, ~2% overhead)
 * @note Unused rows (ib < requested_mr) are left as zero
 *
 * @see pack_B_panel_simd()
 * @see gemm_execute_plan()
 */
static pack_strides_t pack_A_panel_simd(
    float *restrict Ap,
    const float *restrict A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t k0, size_t kb,
    float alpha,
    size_t requested_mr)
{
    (void)M;

    size_t actual_mr = requested_mr;
    assert(ib <= actual_mr && "ib exceeds MR: planning error");

    // ✅ OPTIMIZATION: Only zero if we have unused rows
    if (ib < actual_mr)
    {
        memset(Ap, 0, kb * actual_mr * sizeof(float));
    }

    //--------------------------------------------------------------------------
    // Fast path: alpha = 1.0 (no scaling needed)
    //--------------------------------------------------------------------------
    if (alpha == 1.0f)
    {
        for (size_t k = 0; k < kb; ++k)
        {
            if (k + 8 < kb)
            {
                PREFETCH_T0(A + i0 * K + (k0 + k + 8));
            }

            const float *src_col = A + i0 * K + (k0 + k);
            float *dst = Ap + k * actual_mr;

            size_t i = 0;

            // SIMD gather: 8 rows at once
            for (; i + 7 < ib; i += 8)
            {
                __m256 v = _mm256_set_ps(
                    src_col[7 * K], src_col[6 * K], src_col[5 * K], src_col[4 * K],
                    src_col[3 * K], src_col[2 * K], src_col[1 * K], src_col[0 * K]);
                _mm256_storeu_ps(dst + i, v);
                src_col += 8 * K;
            }

            // Scalar tail: Remaining rows
            const float *src_tail = A + (i0 + i) * K + (k0 + k);
            for (; i < ib; ++i)
            {
                dst[i] = src_tail[0];
                src_tail += K;
            }
            // ✅ Rows [ib .. actual_mr-1] are either:
            //    - Already zeroed (if ib < actual_mr, via memset above)
            //    - Don't exist (if ib == actual_mr)
        }
    }
    //--------------------------------------------------------------------------
    // General path: alpha ≠ 1.0 (apply scaling during pack)
    //--------------------------------------------------------------------------
    else
    {
        __m256 valpha = _mm256_set1_ps(alpha);

        for (size_t k = 0; k < kb; ++k)
        {
            if (k + 8 < kb)
            {
                PREFETCH_T0(A + i0 * K + (k0 + k + 8));
            }

            const float *src_col = A + i0 * K + (k0 + k);
            float *dst = Ap + k * actual_mr;

            size_t i = 0;

            // SIMD gather + scale: 8 rows at once
            for (; i + 7 < ib; i += 8)
            {
                __m256 v = _mm256_set_ps(
                    src_col[7 * K], src_col[6 * K], src_col[5 * K], src_col[4 * K],
                    src_col[3 * K], src_col[2 * K], src_col[1 * K], src_col[0 * K]);
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, valpha));
                src_col += 8 * K;
            }

            // Scalar tail with scaling
            const float *src_tail = A + (i0 + i) * K + (k0 + k);
            for (; i < ib; ++i)
            {
                dst[i] = src_tail[0] * alpha;
                src_tail += K;
            }
        }
    }

    pack_strides_t strides;
    strides.a_k_stride = actual_mr;
    strides.b_k_stride = 0;
    return strides;
}

/**
 * @brief Pack B panel with SIMD copy
 *
 * Packs a submatrix of B from row-major storage to row-major packed format
 * with fixed stride for cache alignment.
 *
 * **Input Layout (B):** Row-major, B[k,j] = B[k*N + j]
 * ```
 * B[k0, j0]   B[k0, j0+1]   ...
 * B[k0+1, j0] B[k0+1, j0+1] ...
 * ...
 * ```
 *
 * **Output Layout (Bp):** Row-major with fixed stride=16
 * ```
 * K=0: [n0 n1 ... n7/15] [pad to 16]
 * K=1: [n0 n1 ... n7/15] [pad to 16]
 * ...
 * ```
 *
 * **Fixed Stride:**
 * - B_STRIDE = 16 (fixed, independent of NR)
 * - Allows 16-wide kernels without special handling
 * - Wastes ~50% space for NR=8, but simplifies code
 *
 * **SIMD Optimization:**
 * - Copies 8 elements per iteration (AVX2)
 * - Prefetches K+4 to hide memory latency
 * - 1.5-2× faster than scalar packing
 *
 * @param[out] Bp  Packed output buffer (kb × 16 floats)
 * @param[in]  B   Source matrix (K × N, row-major)
 * @param[in]  K   Total rows in B (for indexing, unused)
 * @param[in]  N   Total columns in B (for indexing)
 * @param[in]  k0  Starting row index in B
 * @param[in]  kb  Number of rows to pack
 * @param[in]  j0  Starting column index in B
 * @param[in]  jb  Number of columns to pack (≤ NR)
 *
 * @return Stride information (b_k_stride = 16)
 *
 * @note B_STRIDE = 16 is hardcoded (matches kernel expectations)
 * @note Zeroes entire buffer first (ensures unused columns are zero)
 * @note Planner must ensure NR ≤ 16 (validated by assertion in caller)
 *
 * @see pack_A_panel_simd()
 * @see gemm_execute_plan()
 */
static pack_strides_t pack_B_panel_simd(
    float *restrict Bp,
    const float *restrict B,
    size_t K, size_t N,
    size_t k0, size_t kb,
    size_t j0, size_t jb)
{
    (void)K;

    const size_t B_STRIDE = 16;

    // ✅ OPTIMIZATION: Only zero if we have unused columns
    if (jb < B_STRIDE)
    {
        memset(Bp, 0, kb * B_STRIDE * sizeof(float));
    }

    for (size_t k = 0; k < kb; ++k)
    {
        if (k + 4 < kb)
        {
            PREFETCH_T0(B + (k0 + k + 4) * N + j0);
        }

        const float *src_row = B + (k0 + k) * N + j0;
        float *dst = Bp + k * B_STRIDE;

        size_t j = 0;

        // SIMD copy: 8 columns at once
        for (; j + 7 < jb; j += 8)
        {
            __m256 v = _mm256_loadu_ps(src_row + j);
            _mm256_storeu_ps(dst + j, v);
        }

        // Scalar tail: Remaining columns
        for (; j < jb; ++j)
        {
            dst[j] = src_row[j];
        }
        // ✅ Columns [jb .. B_STRIDE-1] are either:
        //    - Already zeroed (if jb < B_STRIDE, via memset above)
        //    - Don't exist (if jb == B_STRIDE)
    }

    pack_strides_t strides;
    strides.a_k_stride = 0;
    strides.b_k_stride = B_STRIDE;
    return strides;
}

/**
 * @brief Pack A panel with explicit source stride
 */
static pack_strides_t pack_A_panel_simd_strided(
    float *restrict Ap,
    const float *restrict A,
    size_t M, size_t lda, // ← was K, now lda
    size_t i0, size_t ib,
    size_t k0, size_t kb,
    float alpha,
    size_t requested_mr)
{
    (void)M;

    size_t actual_mr = requested_mr;
    assert(ib <= actual_mr && "ib exceeds MR: planning error");

    if (ib < actual_mr)
    {
        memset(Ap, 0, kb * actual_mr * sizeof(float));
    }

    if (alpha == 1.0f)
    {
        for (size_t k = 0; k < kb; ++k)
        {
            if (k + 8 < kb)
                PREFETCH_T0(A + i0 * lda + (k0 + k + 8));

            const float *src_col = A + i0 * lda + (k0 + k); // ← use lda
            float *dst = Ap + k * actual_mr;

            size_t i = 0;
            for (; i + 7 < ib; i += 8)
            {
                __m256 v = _mm256_set_ps(
                    src_col[7 * lda], src_col[6 * lda], src_col[5 * lda], src_col[4 * lda],
                    src_col[3 * lda], src_col[2 * lda], src_col[1 * lda], src_col[0 * lda]);
                _mm256_storeu_ps(dst + i, v);
                src_col += 8 * lda;
            }

            const float *src_tail = A + (i0 + i) * lda + (k0 + k);
            for (; i < ib; ++i)
            {
                dst[i] = src_tail[0];
                src_tail += lda;
            }
        }
    }
    else
    {
        __m256 valpha = _mm256_set1_ps(alpha);

        for (size_t k = 0; k < kb; ++k)
        {
            if (k + 8 < kb)
                PREFETCH_T0(A + i0 * lda + (k0 + k + 8));

            const float *src_col = A + i0 * lda + (k0 + k);
            float *dst = Ap + k * actual_mr;

            size_t i = 0;
            for (; i + 7 < ib; i += 8)
            {
                __m256 v = _mm256_set_ps(
                    src_col[7 * lda], src_col[6 * lda], src_col[5 * lda], src_col[4 * lda],
                    src_col[3 * lda], src_col[2 * lda], src_col[1 * lda], src_col[0 * lda]);
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, valpha));
                src_col += 8 * lda;
            }

            const float *src_tail = A + (i0 + i) * lda + (k0 + k);
            for (; i < ib; ++i)
            {
                dst[i] = src_tail[0] * alpha;
                src_tail += lda;
            }
        }
    }

    pack_strides_t strides;
    strides.a_k_stride = actual_mr;
    strides.b_k_stride = 0;
    return strides;
}

/**
 * @brief Pack B panel with explicit source stride
 */
static pack_strides_t pack_B_panel_simd_strided(
    float *restrict Bp,
    const float *restrict B,
    size_t K, size_t ldb, // ← was N, now ldb
    size_t k0, size_t kb,
    size_t j0, size_t jb)
{
    (void)K;

    const size_t B_STRIDE = 16;

    if (jb < B_STRIDE)
    {
        memset(Bp, 0, kb * B_STRIDE * sizeof(float));
    }

    for (size_t k = 0; k < kb; ++k)
    {
        if (k + 4 < kb)
            PREFETCH_T0(B + (k0 + k + 4) * ldb + j0);

        const float *src_row = B + (k0 + k) * ldb + j0; // ← use ldb
        float *dst = Bp + k * B_STRIDE;

        size_t j = 0;
        for (; j + 7 < jb; j += 8)
        {
            __m256 v = _mm256_loadu_ps(src_row + j);
            _mm256_storeu_ps(dst + j, v);
        }

        for (; j < jb; ++j)
        {
            dst[j] = src_row[j];
        }
    }

    pack_strides_t strides;
    strides.a_k_stride = 0;
    strides.b_k_stride = B_STRIDE;
    return strides;
}

//==============================================================================
// KERNEL DISPATCH
//==============================================================================

/**
 * @brief Dispatch to appropriate micro-kernel based on kernel ID
 *
 * This function acts as a unified entry point for all micro-kernels,
 * providing a consistent interface for the execution loop.
 *
 * **Dispatch Strategy:**
 * - Simple switch statement (branch predictor friendly)
 * - Composite kernels (16×16, 16×6) implemented as multiple calls
 * - Unused mask parameter (legacy from masked store design)
 *
 * **Composite Kernels:**
 * - KERN_16x16: Calls 8×16 twice (rows 0-7, then 8-15)
 * - KERN_16x8: Implemented in kernel file as 2× 8×8 calls
 * - KERN_16x6: Implemented in kernel file as 2× 8×6 calls
 *
 * @param kernel_id     Kernel identifier (from gemm_kernel_id_t enum)
 * @param c             Output submatrix pointer
 * @param ldc           Leading dimension of C (stride between rows)
 * @param Ap            Packed A panel
 * @param a_k_stride    Stride for A panel (MR)
 * @param Bp            Packed B panel
 * @param b_k_stride    Stride for B panel (16)
 * @param Kblk          Number of K-iterations to process
 * @param m_block       Actual rows in this tile (≤ MR)
 * @param n_block       Actual columns in this tile (≤ NR)
 *
 * @note mask_unused is a legacy parameter (from masked store design)
 * @note All kernels handle partial tiles via scalar loops
 * @note Default case does nothing (should never be reached)
 *
 * @see gemm_kernel_id_t
 * @see gemm_kernels_avx2.h
 */
static inline void dispatch_kernel(
    gemm_kernel_id_t kernel_id,
    float *restrict c,
    size_t ldc,
    const float *restrict Ap,
    size_t a_k_stride,
    const float *restrict Bp,
    size_t b_k_stride,
    size_t Kblk,
    size_t m_block,
    size_t n_block)
{
    __m256i mask_unused = _mm256_setzero_si256();

    switch (kernel_id)
    {
    case KERN_16x8_ADD:
        gemm_16x8_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                    Kblk, m_block, n_block, mask_unused);
        break;
    case KERN_16x8_STORE:
        gemm_16x8_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                      Kblk, m_block, n_block, mask_unused);
        break;
    case KERN_8x8_ADD:
        gemm_8x8_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                   Kblk, m_block, n_block, mask_unused);
        break;
    case KERN_8x8_STORE:
        gemm_8x8_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, m_block, n_block, mask_unused);
        break;
    case KERN_16x6_ADD:
        gemm_16x6_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                    Kblk, m_block, n_block, mask_unused);
        break;
    case KERN_16x6_STORE:
        gemm_16x6_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                      Kblk, m_block, n_block, mask_unused);
        break;
    case KERN_8x6_ADD:
        gemm_8x6_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                   Kblk, m_block, n_block, mask_unused);
        break;
    case KERN_8x6_STORE:
        gemm_8x6_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, m_block, n_block, mask_unused);
        break;
    case KERN_4x8_ADD:
        gemm_4x8_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                   Kblk, m_block, n_block, mask_unused);
        //                                ^^^^^^^ ✅ ADD THIS!
        break;

    case KERN_4x8_STORE:
        gemm_4x8_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, m_block, n_block, mask_unused);
        //                                  ^^^^^^^ ✅ ADD THIS!
        break;
    case KERN_1x8_ADD:
        gemm_1x8_panel_avx2fma_add(c, Ap, a_k_stride, Bp, b_k_stride,
                                   Kblk, n_block, mask_unused);
        break;
    case KERN_1x8_STORE:
        gemm_1x8_panel_avx2fma_store(c, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, n_block, mask_unused);
        break;
    case KERN_8x16_ADD:
        gemm_8x16_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                    Kblk, m_block, n_block,
                                    mask_unused, mask_unused);
        break;
    case KERN_8x16_STORE:
        gemm_8x16_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                      Kblk, m_block, n_block,
                                      mask_unused, mask_unused);
        break;
    case KERN_16x16_ADD:
        gemm_8x16_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                    Kblk, 8, n_block,
                                    mask_unused, mask_unused);
        gemm_8x16_panel_avx2fma_add(c + 8 * ldc, ldc, Ap + 8, a_k_stride,
                                    Bp, b_k_stride, Kblk, m_block - 8, n_block,
                                    mask_unused, mask_unused);
        break;
    case KERN_16x16_STORE:
        gemm_8x16_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                      Kblk, 8, n_block,
                                      mask_unused, mask_unused);
        gemm_8x16_panel_avx2fma_store(c + 8 * ldc, ldc, Ap + 8, a_k_stride,
                                      Bp, b_k_stride, Kblk, m_block - 8, n_block,
                                      mask_unused, mask_unused);
        break;
    default:
        break;
    }
}

/**
 * @brief Apply beta scaling to output matrix C (contiguous layout)
 * 
 * Implements the beta component of BLAS GEMM semantics:
 * - beta = 0.0: Zero C (does NOT read old values, uses memset)
 * - beta = 1.0: No-op (C unchanged, pure accumulation mode)
 * - beta ≠ 1.0: Scale C *= beta (general case)
 * 
 * **BLAS Compliance:**
 * According to BLAS standard, when beta=0, implementations should NOT
 * read the original C values (which may contain uninitialized data or NaN).
 * This function respects that by using memset instead of multiply-by-zero.
 * 
 * **Vectorization:**
 * Uses AVX2 to process 8 elements per iteration in the main loop,
 * with scalar tail loop for remaining elements.
 * 
 * **When to use this variant:**
 * Use when C is stored contiguously (ldc = N), allowing single memset.
 * For strided layouts, use handle_beta_scaling_strided() instead.
 * 
 * @param[in,out] C     Output matrix (M × N, row-major, contiguous)
 * @param[in]     M     Number of rows in C
 * @param[in]     N     Number of columns in C (also the stride, ldc)
 * @param[in]     ldc   Leading dimension of C (must equal N for this function)
 * @param[in]     beta  Scalar multiplier for C
 * 
 * @return true if first_accumulation mode (beta=0, C was zeroed)
 * @return false if accumulation mode (beta≠0, must ADD to existing C)
 * 
 * @note Called once per gemm_execute_plan(), not per tile
 * @note For beta=0: ~200 GB/s (memset bandwidth)
 * @note For beta≠1: ~50 GB/s (vectorized read-multiply-write)
 * @note Thread-safe if different threads use different C matrices
 * 
 * @see handle_beta_scaling_strided()
 * @see gemm_execute_plan()
 */
static bool handle_beta_scaling(
    float *restrict C,
    size_t M, size_t N, size_t ldc,
    float beta)
{
    //==========================================================================
    // CASE 1: beta = 0.0 (Zero output, BLAS-compliant)
    //==========================================================================
    // BLAS standard: When beta=0, do NOT read old C values (may be NaN/uninit).
    // Implementation: Single contiguous memset (fastest possible zeroing).
    // 
    // Performance: ~200 GB/s on modern CPUs (memory bandwidth limited)
    //==========================================================================
    
    if (beta == 0.0f)
    {
        memset(C, 0, M * N * sizeof(float));
        return true; // Signal: Use STORE mode for first K-tile (don't load C)
    }
    
    //==========================================================================
    // CASE 2: beta = 1.0 (No-op, pure accumulation)
    //==========================================================================
    // No scaling needed, C remains unchanged.
    // Kernels will ADD to existing values: C += alpha*A*B
    //==========================================================================
    
    else if (beta != 1.0f)
    {
        //======================================================================
        // CASE 3: beta ≠ 1.0 (General scaling: C *= beta)
        //======================================================================
        // Process each row with AVX2 vectorization (8 floats at a time).
        // 
        // Performance: ~50 GB/s (read 4 bytes → multiply → write 4 bytes)
        // Memory traffic: 2× (read + write) vs memset's 1× (write only)
        //======================================================================
        
        __m256 vbeta = _mm256_set1_ps(beta); // Broadcast beta to all lanes
        
        for (size_t i = 0; i < M; ++i)
        {
            float *row = C + i * ldc; // Pointer to current row
            size_t j = 0;
            
            // Vectorized main loop: Process 8 elements per iteration
            for (; j + 7 < N; j += 8)
            {
                __m256 c = _mm256_loadu_ps(row + j);              // Load 8 floats
                _mm256_storeu_ps(row + j, _mm256_mul_ps(c, vbeta)); // C *= beta, store
            }
            
            // Scalar tail loop: Handle remaining elements (0-7)
            for (; j < N; ++j)
            {
                row[j] *= beta;
            }
        }
        
        return false; // Signal: Use ADD mode (accumulate into scaled C)
    }
    
    // CASE 2 continuation: beta == 1.0
    return false; // Signal: Use ADD mode (accumulate into unchanged C)
}

/**
 * @brief Apply beta scaling to output matrix C (strided layout)
 * 
 * Identical semantics to handle_beta_scaling(), but optimized for
 * non-contiguous (strided) C matrices where ldc > N.
 * 
 * **Key Difference:**
 * When beta=0, cannot use single memset because rows are not contiguous.
 * Instead, memset each row individually.
 * 
 * **Use Case:**
 * Strided GEMM operations where C is a submatrix of a larger matrix:
 * - QR decomposition trailing matrix updates
 * - Block algorithms operating on matrix views
 * - Panel updates in factorizations
 * 
 * **Example:**
 * ```
 * float large_matrix[256][256];
 * // Update submatrix [64:192, 128:256]
 * float *C = &large_matrix[64][128];
 * size_t ldc = 256;  // Stride of large matrix
 * size_t M = 128, N = 128;  // Submatrix size
 * handle_beta_scaling_strided(C, M, N, ldc, beta);
 * ```
 * 
 * @param[in,out] C     Output matrix (M × N, row-major, stride=ldc)
 * @param[in]     M     Number of rows in C
 * @param[in]     N     Number of columns in C
 * @param[in]     ldc   Leading dimension of C (stride, may be > N)
 * @param[in]     beta  Scalar multiplier for C
 * 
 * @return true if first_accumulation mode (beta=0, C was zeroed)
 * @return false if accumulation mode (beta≠0, must ADD to existing C)
 * 
 * @note Slightly slower than handle_beta_scaling() for beta=0 due to
 *       multiple memset calls vs single contiguous memset
 * @note For beta≠1: Same performance as non-strided version
 * @note Thread-safe if different threads use different C matrices
 * 
 * @see handle_beta_scaling()
 * @see gemm_execute_plan_strided()
 */
static bool handle_beta_scaling_strided(
    float *restrict C,
    size_t M, size_t N, size_t ldc,
    float beta)
{
    //==========================================================================
    // CASE 1: beta = 0.0 (Zero output, strided layout)
    //==========================================================================
    // Cannot use single memset because rows are not contiguous (ldc > N).
    // Must zero each row individually.
    // 
    // Performance: ~180 GB/s (slightly slower than contiguous due to
    //              multiple memset calls and potential TLB pressure)
    //==========================================================================
    
    if (beta == 0.0f)
    {
        for (size_t i = 0; i < M; ++i)
        {
            memset(C + i * ldc, 0, N * sizeof(float)); // Zero one row
        }
        return true; // Signal: Use STORE mode for first K-tile
    }
    
    //==========================================================================
    // CASE 2 & 3: beta = 1.0 (no-op) or beta ≠ 1.0 (scale)
    //==========================================================================
    // Logic identical to non-strided version.
    // The stride (ldc) is already used in row pointer computation.
    //==========================================================================
    
    else if (beta != 1.0f)
    {
        __m256 vbeta = _mm256_set1_ps(beta);
        
        for (size_t i = 0; i < M; ++i)
        {
            float *row = C + i * ldc; // ← Uses stride ldc
            size_t j = 0;
            
            // Vectorized main loop: 8 elements per iteration
            for (; j + 7 < N; j += 8)
            {
                __m256 c = _mm256_loadu_ps(row + j);
                _mm256_storeu_ps(row + j, _mm256_mul_ps(c, vbeta));
            }
            
            // Scalar tail: Remaining elements
            for (; j < N; ++j)
            {
                row[j] *= beta;
            }
        }
        
        return false; // Signal: Use ADD mode
    }
    
    return false; // beta == 1.0, no-op
}

/**
 * @brief Execute GEMM operation using a pre-computed plan
 *
 * This is the main execution engine that orchestrates the entire GEMM
 * computation using pre-computed blocking parameters and workspace.
 *
 * **Algorithm Overview:**
 *
 * ```
 * 1. Beta pre-scaling (C *= beta, once)
 * 2. FOR each NC tile (j0...j0+NC):
 * 3.   FOR each KC tile (k0...k0+KC):
 * 4.     Pack B panels once (reused across all MC tiles)
 * 5.     FOR each MC tile (i0...i0+MC):
 * 6.       FOR each MR tile (i...i+MR):
 * 7.         Pack A panel (alpha absorbed)
 * 8.         FOR each NR panel:
 * 9.           Select kernel (pre-selected for full tiles)
 * 10.          Dispatch kernel (C += Ap * Bp)
 * ```
 *
 * **Loop Order Rationale (NC → KC → MC):**
 *
 * This specific ordering maximizes cache efficiency:
 * - **NC outer**: Processes vertical strips of B and C
 * - **KC middle**: Packs B once, reuses across all M
 * - **MC inner**: Packs A frequently, but A is smaller
 *
 * **Cache Behavior:**
 * - B panels (KC × NR × 4 bytes) stay in L2 (~2 MB)
 * - A panels (MC × KC × 4 bytes) fit in L1 (~48 KB)
 * - C submatrix (MC × NC × 4 bytes) fits in L2
 * - Overall L2 hit rate: ~98% for large matrices
 
 *
 * **ADD vs STORE Modes:**
 *
 * - **STORE mode**: C = A*B (used when beta=0, first K-tile)
 * - **ADD mode**: C += A*B (used for K-tiles 1..last, or when beta≠0)
 *
 * This eliminates redundant load-add cycles on first K-tile when beta=0.
 *
 * **Performance:**
 * - Small overhead (~5%) from packing
 * - 169.8 GFLOPS on Intel i9-14900 (single-core)
 * - 162× faster than naive implementation
 *
 * ┌─────────────────────────────────────────────────────────────┐
 *│ NC Loop (j0 = 0, 256, 512, ...)                             │
 *│   ┌───────────────────────────────────────────────────────┐ │
 *│   │ KC Loop (k0 = 0, 256, 512, ...)                       │ │
 *│   │   ┌─────────────────────────────────────────────────┐ │ │
 *│   │   │ Pack B panels (ONCE per KC×NC tile!)            │ │ │
 *│   │   │   Bp[panel_0] = pack(B[k0:k0+KC, j0:j0+NR])     │ │ │
 *│   │   │   Bp[panel_1] = pack(B[k0:k0+KC, j0+NR:...])    │ │ │
 *│   │   │   ...                                            │ │ │
 *│   │   └─────────────────────────────────────────────────┘ │ │
 *│   │                                                         │ │
 *│   │   ┌─────────────────────────────────────────────────┐ │ │
 *│   │   │ MC Loop (i0 = 0, 128, 256, ...)                 │ │ │
 *│   │   │   ┌───────────────────────────────────────────┐ │ │ │
 *│   │   │   │ MR Loop (i = i0, i0+16, ...)              │ │ │ │
 *│   │   │   │   Pack A[i:i+MR, k0:k0+KC] → Ap           │ │ │ │
 *│   │   │   │                                            │ │ │ │
 *│   │   │   │   For each N-panel:                       │ │ │ │
 *│   │   │   │     C[i:i+MR, j:j+NR] +=                  │ │ │ │
 *│   │   │   │       kernel(Ap, Bp[panel])               │ │ │ │
 *│   │   │   └───────────────────────────────────────────┘ │ │ │
 *│   │   └─────────────────────────────────────────────────┘ │ │
 *│   └───────────────────────────────────────────────────────┘ │
 *└─────────────────────────────────────────────────────────────┘
 *
 * @param plan  Execution plan (must not be NULL)
 * @param C     Output matrix (M × N, row-major)
 * @param A     Input matrix A (M × K, row-major)
 * @param B     Input matrix B (K × N, row-major)
 * @param alpha Scalar multiplier for A*B
 * @param beta  Scalar multiplier for C
 *
 * @return 0 on success, -1 on error (NULL pointers)
 *
 * @pre plan must be created with gemm_plan_create() or gemm_plan_create_with_mode()
 * @pre plan->M, plan->K, plan->N must match actual matrix dimensions
 * @pre C, A, B must be allocated by caller with correct sizes
 * @pre C must not alias A or B (no overlap)
 *
 * @note Thread-safe if different threads use different plans/matrices
 * @note Not thread-safe if multiple threads share the same plan (workspace collision)
 *
 * @warning Plan dimensions must exactly match matrix dimensions (no validation!)
 *
 * @see gemm_plan_create()
 * @see gemm_auto()
 */
static int gemm_execute_internal(
    gemm_plan_t *plan,
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc, size_t lda, size_t ldb,
    tile_counts_t tiles,
    float alpha, float beta,
    bool first_accumulation)
{
    // Get workspace pointers from pre-allocated plan
    float *Ap = plan->workspace_a;  // Packed A panels (MR × KC)
    float *Bp = plan->workspace_b;  // Packed B panels (multiple panels × KC × 16)

    //==========================================================================
    // LEVEL 1: NC LOOP - Vertical strips of B and C
    //==========================================================================
    // Process N-dimension in blocks of NC columns at a time.
    // Each NC block covers columns [j0, j0+jb) where jb ≤ NC.
    // 
    // **Cache strategy:** B panels packed in this loop are reused across
    // all MC tiles in the inner loops → maximize L2 cache efficiency!
    //==========================================================================
    
    for (size_t jt = 0; jt < tiles.n_nc; jt++)
    {
        // Compute starting column and width of this NC block
        size_t j0 = jt * plan->NC;           // Starting column index
        size_t jb = MIN(plan->NC, N - j0);   // Actual width (handle edge case)

        //======================================================================
        // LEVEL 2: KC LOOP - Reduction dimension blocks
        //======================================================================
        // Process K-dimension in blocks of KC elements at a time.
        // Each KC block covers reduction indices [k0, k0+kb) where kb ≤ KC.
        // 
        // **Critical insight:** B panels are packed ONCE per KC×NC tile,
        // then reused for ALL MC tiles below. This is the key to performance!
        //======================================================================
        
        for (size_t kt = 0; kt < tiles.n_kc; kt++)
        {
            // Compute starting index and size of this KC block
            size_t k0 = kt * plan->KC;       // Starting K index
            size_t kb = MIN(plan->KC, K - k0); // Actual K block size (handle edge)

            //==================================================================
            // PACK B PANELS (Once per KC×NC tile)
            //==================================================================
            // Divide the jb columns into panels of width NR.
            // Each panel is KC × NR and gets packed into a contiguous buffer.
            // 
            // **Memory layout:** Bp[panel_p] = Bp + p * plan->KC * 16
            // Note: Stride is plan->KC (not kb!) because workspace is allocated
            //       for maximum KC, ensuring consistent panel offsets.
            //==================================================================
            
            size_t n_panels = (jb + plan->NR - 1) / plan->NR; // Ceiling division
            pack_strides_t b_strides; // Will store B_STRIDE=16 for kernel access

            for (size_t p = 0; p < n_panels; p++)
            {
                // Compute starting column and width for this panel
                size_t j = j0 + p * plan->NR;        // Panel start column
                size_t jw = MIN(plan->NR, j0 + jb - j); // Panel width (handle edge)

                // ✅ CRITICAL: Use plan->KC for stride, NOT kb!
                // Reason: Workspace is allocated for max KC rows. Using kb would
                //         cause panels to overlap in memory when kb < plan->KC.
                float *Bp_panel = Bp + p * plan->KC * 16;

                // Pack B[k0:k0+kb, j:j+jw] into Bp_panel
                // Result: Bp_panel has kb rows × 16 cols (stride always 16)
                b_strides = pack_B_panel_simd_strided(
                    Bp_panel, B, K, ldb,
                    k0, kb,      // Source K range to pack
                    j, jw);      // Source N range to pack
            }

            //==================================================================
            // LEVEL 3: MC LOOP - Horizontal strips of A and C
            //==================================================================
            // Process M-dimension in blocks of MC rows at a time.
            // Each MC block covers rows [i0, i0+ib) where ib ≤ MC.
            // 
            // **Frequency:** A panels are packed MULTIPLE times (once per MC
            // block), but they're small (MR × KC) so this is acceptable.
            //==================================================================
            
            for (size_t it = 0; it < tiles.n_mc; it++)
            {
                // Compute starting row and height of this MC block
                size_t i0 = it * plan->MC;           // Starting row index
                size_t ib = MIN(plan->MC, M - i0);   // Actual height (handle edge)
                
                // How many MR-sized tiles fit in this MC block?
                size_t n_mr_tiles = (ib + plan->MR - 1) / plan->MR; // Ceiling div

                //==============================================================
                // LEVEL 4: MR LOOP - Micro-kernel tiles
                //==============================================================
                // Process ib rows in chunks of MR rows at a time.
                // Each MR tile is the unit of work for a single micro-kernel.
                //==============================================================
                
                for (size_t mt = 0; mt < n_mr_tiles; mt++)
                {
                    // Compute starting row and height for this MR tile
                    size_t i = i0 + mt * plan->MR;       // Tile start row
                    size_t mh = MIN(plan->MR, M - i);    // Tile height (handle edge)
                    size_t pack_mr = plan->MR;            // Physical capacity (always MR)

                    //==========================================================
                    // PACK A PANEL (Once per MR tile)
                    //==========================================================
                    // Pack A[i:i+mh, k0:k0+kb] into Ap, applying alpha scaling.
                    // 
                    // **Layout:** Column-major (K-outer):
                    //   Ap[k][m] = Ap[k * MR + m]
                    // 
                    // **Alpha absorption:** A is multiplied by alpha during pack,
                    // so kernels just compute A*B (not alpha*A*B).
                    //==========================================================
                    
                    pack_strides_t a_strides = pack_A_panel_simd_strided(
                        Ap, A, M, lda,
                        i, mh,        // Source M range to pack
                        k0, kb,       // Source K range to pack
                        alpha,        // Alpha absorbed into A during pack
                        pack_mr);     // Physical capacity (MR)

                    //==========================================================
                    // EXECUTE KERNELS ON ALL N-PANELS
                    //==========================================================
                    // For each N-panel in this NC block, execute the micro-kernel
                    // to compute: C[i:i+mh, j:j+jw] += Ap * Bp_panel
                    //==========================================================
                    
                    for (size_t p = 0; p < n_panels; p++)
                    {
                        // Recompute panel position (same as during pack)
                        size_t j = j0 + p * plan->NR;
                        size_t jw = MIN(plan->NR, j0 + jb - j);

                        //==================================================
                        // KERNEL SELECTION
                        //==================================================
                        // Select optimal kernel based on actual tile size (mh × jw).
                        // Returns both ADD and STORE variants, plus native width.
                        // 
                        // **Why two variants?**
                        // - STORE: C = A*B (overwrite, used when beta=0 on first K-tile)
                        // - ADD: C += A*B (accumulate, used for K-tiles 1..n)
                        //==================================================
                        
                        gemm_kernel_id_t kern_add, kern_store;
                        int kernel_width; // Native width of selected kernel (6, 8, or 16)
                        
                        gemm_select_kernels(mh, jw, &kern_add, &kern_store, &kernel_width);

                        //==================================================
                        // MULTI-PASS EXECUTION
                        //==================================================
                        // If tile width jw > kernel_width, split into multiple passes.
                        // 
                        // **Example:** jw=11, kernel_width=8
                        //   Pass 1: columns [0, 8)   (8 columns)
                        //   Pass 2: columns [8, 11)  (3 columns)
                        // 
                        // Each pass processes kernel_width columns (or remainder).
                        //==================================================
                        
                        size_t j_offset = 0; // Column offset within this panel
                        
                        while (j_offset < jw)
                        {
                            // How many columns in this pass?
                            size_t jw_chunk = MIN((size_t)kernel_width, jw - j_offset);

                            //==============================================
                            // ADD vs STORE MODE SELECTION
                            //==============================================
                            // Decision tree:
                            // 1. First K-tile (kt=0) AND beta was 0 → STORE
                            //    (No need to load old C values, just overwrite)
                            // 2. Subsequent K-tiles (kt>0) → ADD
                            //    (Accumulate partial results across K)
                            // 3. First K-tile BUT beta≠0 → ADD
                            //    (Need to preserve beta*C, so accumulate)
                            //==============================================
                            
                            gemm_kernel_id_t kernel_id;
                            if (kt == 0 && first_accumulation)
                            {
                                kernel_id = kern_store; // C = A*B (overwrite)
                            }
                            else
                            {
                                kernel_id = kern_add;   // C += A*B (accumulate)
                            }

                            //==============================================
                            // POINTER ARITHMETIC
                            //==============================================
                            
                            // Output pointer: C[i, j+j_offset]
                            // - i: Current MR tile start row
                            // - j+j_offset: Current panel column + pass offset
                            float *cptr = C + i * ldc + (j + j_offset);

                            // Input B pointer: Bp[panel_p, k=0, n=j_offset]
                            // - p * plan->KC * 16: Panel offset (uses plan->KC!)
                            // - j_offset: Column offset within panel
                            // 
                            // ✅ CRITICAL: Use plan->KC (not kb) for panel stride!
                            // Reason: Workspace allocated for max KC, so all panels
                            //         have consistent offsets even when kb < plan->KC.
                            float *bptr = Bp + p * plan->KC * 16 + j_offset;

                            //==============================================
                            // KERNEL DISPATCH
                            //==============================================
                            // Execute micro-kernel: C[i:i+mh, j+j_offset:j+j_offset+jw_chunk]
                            //                       op= Ap[0:mh, 0:kb] * Bp[0:kb, j_offset:j_offset+jw_chunk]
                            // where op is either = (STORE) or += (ADD)
                            //==============================================
                            
                            dispatch_kernel(
                                kernel_id,              // Which kernel variant?
                                cptr,                   // Output: C submatrix pointer
                                ldc,                    // C stride (leading dimension)
                                Ap,                     // Input: Packed A panel
                                a_strides.a_k_stride,   // A stride (= MR)
                                bptr,                   // Input: Packed B panel (with offset)
                                b_strides.b_k_stride,   // B stride (= 16, always)
                                kb,                     // Number of K iterations
                                mh,                     // Actual tile height (≤ MR)
                                jw_chunk);              // Actual tile width for this pass

                            // Advance to next column chunk
                            j_offset += kernel_width;
                            
                        } // end while (multi-pass)
                        
                    } // end for p (N-panels)
                    
                } // end for mt (MR tiles)
                
            } // end for it (MC blocks)
            
        } // end for kt (KC blocks)
        
    } // end for jt (NC blocks)

    return 0;
}

int gemm_execute_plan(
    gemm_plan_t *plan,
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    float alpha,
    float beta)
{
    if (!plan || !C || !A || !B)
        return -1;

    // Use precomputed tile counts for max dimensions (fast path)
    bool first_acc = handle_beta_scaling(C, plan->max_M, plan->max_N,
                                         plan->max_N, beta);

    tile_counts_t tiles = {
        .n_mc = plan->n_mc_tiles_max,
        .n_kc = plan->n_kc_tiles_max,
        .n_nc = plan->n_nc_tiles_max};

    return gemm_execute_internal(plan, C, A, B,
                                 plan->max_M, plan->max_K, plan->max_N,
                                 plan->max_N, plan->max_K, plan->max_N,
                                 tiles, alpha, beta, first_acc);
}

//==============================================================================
// PUBLIC API (Unchanged)
//==============================================================================

/**
 * @brief Automatic GEMM with Tier 1/Tier 2 dispatch
 *
 * Automatically selects between Tier 1 (fixed-size kernels) and
 * Tier 2 (blocked execution) based on matrix dimensions.
 *
 * **Dispatch Logic:**
 * 1. Try Tier 1 (gemm_small_dispatch)
 *    - If M,N ≤ 16 and K ≤ 64 and FLOPs ≤ 8192: Use fixed-size kernels
 * 2. Fallback to Tier 2 (gemm_execute_plan)
 *    - Create plan, execute, destroy (one-time use)
 *
 * **Usage Note:**
 * For repeated calls with same dimensions, use gemm_plan_create() +
 * gemm_execute_plan() instead to amortize planning cost.
 *
 * @param C     Output matrix (M × N, row-major)
 * @param A     Input matrix A (M × K, row-major)
 * @param B     Input matrix B (K × N, row-major)
 * @param M     Number of rows in A and C
 * @param K     Number of columns in A, rows in B
 * @param N     Number of columns in B and C
 * @param alpha Scalar multiplier for A*B
 * @param beta  Scalar multiplier for C
 *
 * @return 0 on success, -1 on error
 *
 * @see gemm_plan_create()
 * @see gemm_execute_plan()
 * @see gemm_small_dispatch()
 */
int gemm_auto(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    int ret = gemm_small_dispatch(C, A, B, M, K, N, N, alpha, beta);
    if (ret == 0)
        return 0;

    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
        return -1;

    ret = gemm_execute_plan(plan, C, A, B, alpha, beta);
    gemm_plan_destroy(plan);
    return ret;
}

int gemm_execute_plan_strided(
    gemm_plan_t *plan,
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc, size_t lda, size_t ldb,
    float alpha,
    float beta)
{
    if (!plan || !C || !A || !B)
        return -1;
    if (M > plan->max_M || K > plan->max_K || N > plan->max_N)
        return -1;

    bool first_acc = handle_beta_scaling_strided(C, M, N, ldc, beta);

    tile_counts_t tiles = {
        .n_mc = (M + plan->MC - 1) / plan->MC,
        .n_kc = (K + plan->KC - 1) / plan->KC,
        .n_nc = (N + plan->NC - 1) / plan->NC};

    return gemm_execute_internal(plan, C, A, B,
                                 M, K, N, ldc, lda, ldb,
                                 tiles, alpha, beta, first_acc);
}

/**
 * @brief GEMM with static workspace allocation
 *
 * Uses pre-allocated global workspace (GEMM_STATIC_WORKSPACE_SIZE).
 * Faster than dynamic allocation but limited to small/medium matrices.
 *
 * **Advantages:**
 * - Zero allocation overhead (~5-10 µs saved)
 * - Deterministic execution time
 *
 * **Limitations:**
 * - Matrix size limited by GEMM_STATIC_WORKSPACE_SIZE
 * - Not thread-safe (global workspace)
 *
 * @param C     Output matrix (M × N, row-major)
 * @param A     Input matrix A (M × K, row-major)
 * @param B     Input matrix B (K × N, row-major)
 * @param M     Number of rows in A and C
 * @param K     Number of columns in A, rows in B
 * @param N     Number of columns in B and C
 * @param alpha Scalar multiplier for A*B
 * @param beta  Scalar multiplier for C
 *
 * @return 0 on success, -1 if dimensions too large or allocation fails
 *
 * @note Check with gemm_fits_static(M,K,N) before calling
 * @note Not thread-safe (uses global workspace)
 *
 * @see gemm_fits_static()
 * @see gemm_dynamic()
 */
int gemm_static(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    if (!gemm_fits_static(M, K, N))
        return -1;

    gemm_plan_t *plan = gemm_plan_create_with_mode(M, K, N, GEMM_MEM_STATIC);
    if (!plan)
        return -1;

    int ret = gemm_execute_plan(plan, C, A, B, alpha, beta);
    gemm_plan_destroy(plan);
    return ret;
}

/**
 * @brief GEMM with dynamic workspace allocation
 *
 * Allocates workspace from heap (handles any matrix size).
 * Slightly slower than static mode due to allocation overhead.
 *
 * **Advantages:**
 * - No size limitations
 * - Thread-safe (each call gets own workspace)
 *
 * **Disadvantages:**
 * - Allocation overhead (~5-10 µs)
 * - Potential memory fragmentation
 *
 * @param C     Output matrix (M × N, row-major)
 * @param A     Input matrix A (M × K, row-major)
 * @param B     Input matrix B (K × N, row-major)
 * @param M     Number of rows in A and C
 * @param K     Number of columns in A, rows in B
 * @param N     Number of columns in B and C
 * @param alpha Scalar multiplier for A*B
 * @param beta  Scalar multiplier for C
 *
 * @return 0 on success, -1 on allocation failure
 *
 * @note Thread-safe (each call allocates own workspace)
 * @note For repeated calls, use gemm_plan_create() + gemm_execute_plan()
 *
 * @see gemm_static()
 * @see gemm_plan_create()
 */
int gemm_dynamic(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    gemm_plan_t *plan = gemm_plan_create_with_mode(M, K, N, GEMM_MEM_DYNAMIC);
    if (!plan)
        return -1;

    int ret = gemm_execute_plan(plan, C, A, B, alpha, beta);
    gemm_plan_destroy(plan);
    return ret;
}

int gemm_strided(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc, size_t lda, size_t ldb,
    float alpha, float beta)
{
    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
        return -1;

    int ret = gemm_execute_plan_strided(plan, C, A, B, M, K, N,
                                        ldc, lda, ldb, alpha, beta);
    gemm_plan_destroy(plan);
    return ret;
}

/**
 * @brief Query blocking parameters for given matrix dimensions
 *
 * Returns the blocking parameters that would be selected by the planner
 * for given matrix dimensions. Useful for:
 * - Performance analysis
 * - Workspace size estimation
 * - Debugging blocking strategies
 *
 * @param[in]  M  Number of rows in A and C
 * @param[in]  K  Number of columns in A, rows in B
 * @param[in]  N  Number of columns in B and C
 * @param[out] MC M-dimension cache block size
 * @param[out] KC K-dimension cache block size
 * @param[out] NC N-dimension cache block size
 * @param[out] MR M-dimension register block size
 * @param[out] NR N-dimension register block size
 *
 * @note All output parameters must be non-NULL
 * @note Does not create a plan (lightweight query)
 *
 * @see gemm_select_blocking()
 * @see gemm_workspace_query()
 */
void gemm_get_tuning(size_t M, size_t K, size_t N,
                     size_t *MC, size_t *KC, size_t *NC,
                     size_t *MR, size_t *NR)
{
    gemm_select_blocking(M, K, N, MC, KC, NC, MR, NR);
}
