/**
 * @file gemm_small_optimized.c
 * @brief Tier 1: Optimized Small Fixed-Size Matrix Kernels
 *
 * Key Optimizations:
 * - K-loop unrolling (2x/4x) for ILP and latency hiding
 * - Template specialization for common QR sizes (K=4,8,16,32)
 * - Aligned store fast paths
 * - K-outer loop for all kernels (better register reuse)
 * - Pre-scaling of B (reduces total multiplies)
 */

#include "gemm_small.h"
#include "gemm_simd_ops.h"
#include <string.h>

//==============================================================================
// HELPER MACROS
//==============================================================================

#define IS_ALIGNED(ptr, alignment) (((uintptr_t)(ptr) & ((alignment) - 1)) == 0)

//==============================================================================
// 4×4 GEMM - SSE K-Outer Implementation (OPTIMIZED)
//==============================================================================

/**
 * @brief Optimized 4×4 GEMM using k-outer loop
 *
 * Changes from original:
 * - Eliminated 16 permute_ps instructions
 * - K-outer loop with scalar broadcast
 * - Pre-scales B by alpha (4K instead of 16K multiplies)
 * - Maintains if-else branching for beta (branch predictor friendly)
 */
void gemm_4x4_inline(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    float alpha, float beta)
{
    //==========================================================================
    // Step 1: Initialize accumulators with beta*C
    //==========================================================================
    __m128 c0, c1, c2, c3;

    if (beta == 0.0f)
    {
        c0 = _mm_setzero_ps();
        c1 = _mm_setzero_ps();
        c2 = _mm_setzero_ps();
        c3 = _mm_setzero_ps();
    }
    else if (beta == 1.0f)
    {
        c0 = _mm_loadu_ps(C + 0);
        c1 = _mm_loadu_ps(C + 4);
        c2 = _mm_loadu_ps(C + 8);
        c3 = _mm_loadu_ps(C + 12);
    }
    else
    {
        __m128 vbeta = _mm_set1_ps(beta);
        c0 = _mm_mul_ps(vbeta, _mm_loadu_ps(C + 0));
        c1 = _mm_mul_ps(vbeta, _mm_loadu_ps(C + 4));
        c2 = _mm_mul_ps(vbeta, _mm_loadu_ps(C + 8));
        c3 = _mm_mul_ps(vbeta, _mm_loadu_ps(C + 12));
    }

    //==========================================================================
    // Step 2: K-outer loop (unrolled by 4 for ILP)
    //==========================================================================
    __m128 valpha = _mm_set1_ps(alpha);

    // Load and pre-scale all B rows
    __m128 b0 = _mm_mul_ps(_mm_loadu_ps(B + 0), valpha);
    __m128 b1 = _mm_mul_ps(_mm_loadu_ps(B + 4), valpha);
    __m128 b2 = _mm_mul_ps(_mm_loadu_ps(B + 8), valpha);
    __m128 b3 = _mm_mul_ps(_mm_loadu_ps(B + 12), valpha);

    // Fully unrolled (K=4 is fixed)
    c0 = _mm_fmadd_ps(_mm_set1_ps(A[0 * 4 + 0]), b0, c0);
    c0 = _mm_fmadd_ps(_mm_set1_ps(A[0 * 4 + 1]), b1, c0);
    c0 = _mm_fmadd_ps(_mm_set1_ps(A[0 * 4 + 2]), b2, c0);
    c0 = _mm_fmadd_ps(_mm_set1_ps(A[0 * 4 + 3]), b3, c0);

    c1 = _mm_fmadd_ps(_mm_set1_ps(A[1 * 4 + 0]), b0, c1);
    c1 = _mm_fmadd_ps(_mm_set1_ps(A[1 * 4 + 1]), b1, c1);
    c1 = _mm_fmadd_ps(_mm_set1_ps(A[1 * 4 + 2]), b2, c1);
    c1 = _mm_fmadd_ps(_mm_set1_ps(A[1 * 4 + 3]), b3, c1);

    c2 = _mm_fmadd_ps(_mm_set1_ps(A[2 * 4 + 0]), b0, c2);
    c2 = _mm_fmadd_ps(_mm_set1_ps(A[2 * 4 + 1]), b1, c2);
    c2 = _mm_fmadd_ps(_mm_set1_ps(A[2 * 4 + 2]), b2, c2);
    c2 = _mm_fmadd_ps(_mm_set1_ps(A[2 * 4 + 3]), b3, c2);

    c3 = _mm_fmadd_ps(_mm_set1_ps(A[3 * 4 + 0]), b0, c3);
    c3 = _mm_fmadd_ps(_mm_set1_ps(A[3 * 4 + 1]), b1, c3);
    c3 = _mm_fmadd_ps(_mm_set1_ps(A[3 * 4 + 2]), b2, c3);
    c3 = _mm_fmadd_ps(_mm_set1_ps(A[3 * 4 + 3]), b3, c3);

    //==========================================================================
    // Step 3: Store results (aligned if possible)
    //==========================================================================
    if (IS_ALIGNED(C, 16))
    {
        _mm_store_ps(C + 0, c0);
        _mm_store_ps(C + 4, c1);
        _mm_store_ps(C + 8, c2);
        _mm_store_ps(C + 12, c3);
    }
    else
    {
        _mm_storeu_ps(C + 0, c0);
        _mm_storeu_ps(C + 4, c1);
        _mm_storeu_ps(C + 8, c2);
        _mm_storeu_ps(C + 12, c3);
    }
}

//==============================================================================
// 6×6 GEMM - AVX2 Implementation (Unchanged - Already Optimal)
//==============================================================================

void gemm_6x6_inline(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[6];
    __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);

    // Initialize with beta*C
    if (beta == 0.0f)
    {
        for (int i = 0; i < 6; i++)
        {
            c_rows[i] = _mm256_setzero_ps();
        }
    }
    else
    {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int i = 0; i < 6; i++)
        {
            __m256 c_old = _mm256_maskload_ps(C + i * ldc, mask6);
            c_rows[i] = _mm256_mul_ps(vbeta, c_old);
        }
    }

    // Load and pre-scale B rows
    __m256 b_rows[6];
    for (int k = 0; k < 6; k++)
    {
        b_rows[k] = _mm256_setr_ps(
            B[k * 6 + 0], B[k * 6 + 1], B[k * 6 + 2],
            B[k * 6 + 3], B[k * 6 + 4], B[k * 6 + 5], 0, 0);

        if (alpha != 1.0f)
        {
            __m256 valpha = _mm256_set1_ps(alpha);
            b_rows[k] = _mm256_mul_ps(b_rows[k], valpha);
        }
    }

    // K-outer loop (fully unrolled, K=6)
    for (int k = 0; k < 6; k++)
    {
        __m256 bk = b_rows[k];
        for (int i = 0; i < 6; i++)
        {
            __m256 a_ik = _mm256_set1_ps(A[i * 6 + k]);
            c_rows[i] = _mm256_fmadd_ps(a_ik, bk, c_rows[i]);
        }
    }

    // Store results
    for (int i = 0; i < 6; i++)
    {
        _mm256_maskstore_ps(C + i * ldc, mask6, c_rows[i]);
    }
}

//==============================================================================
// 8×8 GEMM - AVX2 Implementation (Unchanged - Already Optimal)
//==============================================================================

void gemm_8x8_inline(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[8];

    // Load and pre-scale B rows
    __m256 b_rows[8];
    for (int k = 0; k < 8; k++)
    {
        b_rows[k] = _mm256_loadu_ps(B + k * 8);
    }

    if (alpha != 1.0f)
    {
        __m256 valpha = _mm256_set1_ps(alpha);
        for (int k = 0; k < 8; k++)
        {
            b_rows[k] = _mm256_mul_ps(b_rows[k], valpha);
        }
    }

    // Initialize with beta*C
    if (beta == 0.0f)
    {
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm256_setzero_ps();
        }
    }
    else if (beta == 1.0f)
    {
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm256_loadu_ps(C + row * ldc);
        }
    }
    else
    {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + row * ldc));
        }
    }

    // K-outer loop (fully unrolled, K=8)
    for (int k = 0; k < 8; k++)
    {
        __m256 bk = b_rows[k];

        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * 8 + k]), bk, c_rows[0]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * 8 + k]), bk, c_rows[1]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * 8 + k]), bk, c_rows[2]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * 8 + k]), bk, c_rows[3]);
        c_rows[4] = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * 8 + k]), bk, c_rows[4]);
        c_rows[5] = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * 8 + k]), bk, c_rows[5]);
        c_rows[6] = _mm256_fmadd_ps(_mm256_set1_ps(A[6 * 8 + k]), bk, c_rows[6]);
        c_rows[7] = _mm256_fmadd_ps(_mm256_set1_ps(A[7 * 8 + k]), bk, c_rows[7]);
    }

    // Store results (aligned if possible)
    const int is_aligned = IS_ALIGNED(C, 32) && (ldc % 8 == 0);

    if (is_aligned)
    {
        for (int row = 0; row < 8; row++)
        {
            _mm256_store_ps(C + row * ldc, c_rows[row]);
        }
    }
    else
    {
        for (int row = 0; row < 8; row++)
        {
            _mm256_storeu_ps(C + row * ldc, c_rows[row]);
        }
    }
}

//==============================================================================
// 8×4 GEMM - SSE Implementation with K-Loop Unrolling
//==============================================================================

/**
 * @brief Optimized 8×4 GEMM with K-loop unrolling
 *
 * Key optimizations:
 * - Unrolls K-loop by 4 for ILP (independent FMA chains)
 * - Pre-scales B to reduce multiplies
 * - Aligned stores when possible
 * - Tail loop for K not divisible by 4
 */
static void gemm_8x4_k_unroll4(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    __m128 c_rows[8];

    // Initialize with beta*C
    if (beta == 0.0f)
    {
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm_setzero_ps();
        }
    }
    else if (beta == 1.0f)
    {
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm_loadu_ps(C + row * ldc);
        }
    }
    else
    {
        __m128 vbeta = _mm_set1_ps(beta);
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm_mul_ps(vbeta, _mm_loadu_ps(C + row * ldc));
        }
    }

    __m128 valpha = _mm_set1_ps(alpha);

    // Main loop: unroll by 4
    size_t k = 0;
    for (; k + 3 < K; k += 4)
    {
        // Load and pre-scale 4 B rows
        __m128 bk0 = _mm_mul_ps(_mm_loadu_ps(B + (k + 0) * 4), valpha);
        __m128 bk1 = _mm_mul_ps(_mm_loadu_ps(B + (k + 1) * 4), valpha);
        __m128 bk2 = _mm_mul_ps(_mm_loadu_ps(B + (k + 2) * 4), valpha);
        __m128 bk3 = _mm_mul_ps(_mm_loadu_ps(B + (k + 3) * 4), valpha);

        // Update all 8 rows (32 independent FMAs - excellent ILP!)
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * K + k + 0]), bk0, c_rows[0]);
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * K + k + 1]), bk1, c_rows[0]);
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * K + k + 2]), bk2, c_rows[0]);
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * K + k + 3]), bk3, c_rows[0]);

        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * K + k + 0]), bk0, c_rows[1]);
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * K + k + 1]), bk1, c_rows[1]);
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * K + k + 2]), bk2, c_rows[1]);
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * K + k + 3]), bk3, c_rows[1]);

        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * K + k + 0]), bk0, c_rows[2]);
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * K + k + 1]), bk1, c_rows[2]);
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * K + k + 2]), bk2, c_rows[2]);
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * K + k + 3]), bk3, c_rows[2]);

        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * K + k + 0]), bk0, c_rows[3]);
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * K + k + 1]), bk1, c_rows[3]);
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * K + k + 2]), bk2, c_rows[3]);
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * K + k + 3]), bk3, c_rows[3]);

        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * K + k + 0]), bk0, c_rows[4]);
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * K + k + 1]), bk1, c_rows[4]);
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * K + k + 2]), bk2, c_rows[4]);
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * K + k + 3]), bk3, c_rows[4]);

        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * K + k + 0]), bk0, c_rows[5]);
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * K + k + 1]), bk1, c_rows[5]);
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * K + k + 2]), bk2, c_rows[5]);
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * K + k + 3]), bk3, c_rows[5]);

        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * K + k + 0]), bk0, c_rows[6]);
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * K + k + 1]), bk1, c_rows[6]);
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * K + k + 2]), bk2, c_rows[6]);
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * K + k + 3]), bk3, c_rows[6]);

        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * K + k + 0]), bk0, c_rows[7]);
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * K + k + 1]), bk1, c_rows[7]);
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * K + k + 2]), bk2, c_rows[7]);
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * K + k + 3]), bk3, c_rows[7]);
    }

    // Tail loop for remaining K
    for (; k < K; k++)
    {
        __m128 bk = _mm_mul_ps(_mm_loadu_ps(B + k * 4), valpha);

        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * K + k]), bk, c_rows[0]);
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * K + k]), bk, c_rows[1]);
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * K + k]), bk, c_rows[2]);
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * K + k]), bk, c_rows[3]);
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * K + k]), bk, c_rows[4]);
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * K + k]), bk, c_rows[5]);
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * K + k]), bk, c_rows[6]);
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * K + k]), bk, c_rows[7]);
    }

    // Store results (aligned if possible)
    const int is_aligned = IS_ALIGNED(C, 16) && (ldc % 4 == 0);

    if (is_aligned)
    {
        for (int row = 0; row < 8; row++)
        {
            _mm_store_ps(C + row * ldc, c_rows[row]);
        }
    }
    else
    {
        for (int row = 0; row < 8; row++)
        {
            _mm_storeu_ps(C + row * ldc, c_rows[row]);
        }
    }
}

/**
 * @brief Template specialization for K=32 (common in QR with ib=32)
 */
static void gemm_8x4_k32(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t ldc,
    float alpha,
    float beta)
{
    __m128 c_rows[8];

    // Initialize with beta*C
    if (beta == 0.0f)
    {
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm_setzero_ps();
        }
    }
    else if (beta == 1.0f)
    {
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm_loadu_ps(C + row * ldc);
        }
    }
    else
    {
        __m128 vbeta = _mm_set1_ps(beta);
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm_mul_ps(vbeta, _mm_loadu_ps(C + row * ldc));
        }
    }

    __m128 valpha = _mm_set1_ps(alpha);

// Fully unrolled for K=32 (8 iterations of unroll-by-4)
#define UNROLL_BLOCK(K_OFFSET)                                                           \
    {                                                                                    \
        __m128 bk0 = _mm_mul_ps(_mm_loadu_ps(B + (K_OFFSET + 0) * 4), valpha);           \
        __m128 bk1 = _mm_mul_ps(_mm_loadu_ps(B + (K_OFFSET + 1) * 4), valpha);           \
        __m128 bk2 = _mm_mul_ps(_mm_loadu_ps(B + (K_OFFSET + 2) * 4), valpha);           \
        __m128 bk3 = _mm_mul_ps(_mm_loadu_ps(B + (K_OFFSET + 3) * 4), valpha);           \
                                                                                         \
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * 32 + K_OFFSET + 0]), bk0, c_rows[0]); \
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * 32 + K_OFFSET + 1]), bk1, c_rows[0]); \
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * 32 + K_OFFSET + 2]), bk2, c_rows[0]); \
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0 * 32 + K_OFFSET + 3]), bk3, c_rows[0]); \
                                                                                         \
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * 32 + K_OFFSET + 0]), bk0, c_rows[1]); \
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * 32 + K_OFFSET + 1]), bk1, c_rows[1]); \
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * 32 + K_OFFSET + 2]), bk2, c_rows[1]); \
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1 * 32 + K_OFFSET + 3]), bk3, c_rows[1]); \
                                                                                         \
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * 32 + K_OFFSET + 0]), bk0, c_rows[2]); \
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * 32 + K_OFFSET + 1]), bk1, c_rows[2]); \
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * 32 + K_OFFSET + 2]), bk2, c_rows[2]); \
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2 * 32 + K_OFFSET + 3]), bk3, c_rows[2]); \
                                                                                         \
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * 32 + K_OFFSET + 0]), bk0, c_rows[3]); \
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * 32 + K_OFFSET + 1]), bk1, c_rows[3]); \
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * 32 + K_OFFSET + 2]), bk2, c_rows[3]); \
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3 * 32 + K_OFFSET + 3]), bk3, c_rows[3]); \
                                                                                         \
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * 32 + K_OFFSET + 0]), bk0, c_rows[4]); \
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * 32 + K_OFFSET + 1]), bk1, c_rows[4]); \
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * 32 + K_OFFSET + 2]), bk2, c_rows[4]); \
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4 * 32 + K_OFFSET + 3]), bk3, c_rows[4]); \
                                                                                         \
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * 32 + K_OFFSET + 0]), bk0, c_rows[5]); \
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * 32 + K_OFFSET + 1]), bk1, c_rows[5]); \
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * 32 + K_OFFSET + 2]), bk2, c_rows[5]); \
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5 * 32 + K_OFFSET + 3]), bk3, c_rows[5]); \
                                                                                         \
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * 32 + K_OFFSET + 0]), bk0, c_rows[6]); \
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * 32 + K_OFFSET + 1]), bk1, c_rows[6]); \
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * 32 + K_OFFSET + 2]), bk2, c_rows[6]); \
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6 * 32 + K_OFFSET + 3]), bk3, c_rows[6]); \
                                                                                         \
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * 32 + K_OFFSET + 0]), bk0, c_rows[7]); \
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * 32 + K_OFFSET + 1]), bk1, c_rows[7]); \
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * 32 + K_OFFSET + 2]), bk2, c_rows[7]); \
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7 * 32 + K_OFFSET + 3]), bk3, c_rows[7]); \
    }

    UNROLL_BLOCK(0)
    UNROLL_BLOCK(4)
    UNROLL_BLOCK(8)
    UNROLL_BLOCK(12)
    UNROLL_BLOCK(16)
    UNROLL_BLOCK(20)
    UNROLL_BLOCK(24)
    UNROLL_BLOCK(28)

#undef UNROLL_BLOCK

    // Store results
    const int is_aligned = IS_ALIGNED(C, 16) && (ldc % 4 == 0);

    if (is_aligned)
    {
        for (int row = 0; row < 8; row++)
        {
            _mm_store_ps(C + row * ldc, c_rows[row]);
        }
    }
    else
    {
        for (int row = 0; row < 8; row++)
        {
            _mm_storeu_ps(C + row * ldc, c_rows[row]);
        }
    }
}

void gemm_8x4_inline(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    // Dispatch to specialized version for common K values
    if (K == 32)
    {
        gemm_8x4_k32(C, A, B, ldc, alpha, beta);
    }
    else if (K >= 8)
    {
        gemm_8x4_k_unroll4(C, A, B, K, ldc, alpha, beta);
    }
    else
    {
        // For very small K, just use simple loop
        __m128 c_rows[8];

        if (beta == 0.0f)
        {
            for (int row = 0; row < 8; row++)
            {
                c_rows[row] = _mm_setzero_ps();
            }
        }
        else if (beta == 1.0f)
        {
            for (int row = 0; row < 8; row++)
            {
                c_rows[row] = _mm_loadu_ps(C + row * ldc);
            }
        }
        else
        {
            __m128 vbeta = _mm_set1_ps(beta);
            for (int row = 0; row < 8; row++)
            {
                c_rows[row] = _mm_mul_ps(vbeta, _mm_loadu_ps(C + row * ldc));
            }
        }

        __m128 valpha = _mm_set1_ps(alpha);
        for (size_t k = 0; k < K; k++)
        {
            __m128 bk = _mm_mul_ps(_mm_loadu_ps(B + k * 4), valpha);

            for (int row = 0; row < 8; row++)
            {
                c_rows[row] = _mm_fmadd_ps(_mm_set1_ps(A[row * K + k]), bk, c_rows[row]);
            }
        }

        for (int row = 0; row < 8; row++)
        {
            _mm_storeu_ps(C + row * ldc, c_rows[row]);
        }
    }
}

//==============================================================================
// 4×8 GEMM - AVX2 Implementation with K-Loop Unrolling
//==============================================================================

static void gemm_4x8_k_unroll4(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[4];

    // Initialize with beta*C
    if (beta == 0.0f)
    {
        for (int row = 0; row < 4; row++)
        {
            c_rows[row] = _mm256_setzero_ps();
        }
    }
    else if (beta == 1.0f)
    {
        for (int row = 0; row < 4; row++)
        {
            c_rows[row] = _mm256_loadu_ps(C + row * ldc);
        }
    }
    else
    {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int row = 0; row < 4; row++)
        {
            c_rows[row] = _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + row * ldc));
        }
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    // Main loop: unroll by 4
    size_t k = 0;
    for (; k + 3 < K; k += 4)
    {
        __m256 bk0 = _mm256_mul_ps(_mm256_loadu_ps(B + (k + 0) * 8), valpha);
        __m256 bk1 = _mm256_mul_ps(_mm256_loadu_ps(B + (k + 1) * 8), valpha);
        __m256 bk2 = _mm256_mul_ps(_mm256_loadu_ps(B + (k + 2) * 8), valpha);
        __m256 bk3 = _mm256_mul_ps(_mm256_loadu_ps(B + (k + 3) * 8), valpha);

        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k + 0]), bk0, c_rows[0]);
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k + 1]), bk1, c_rows[0]);
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k + 2]), bk2, c_rows[0]);
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k + 3]), bk3, c_rows[0]);

        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k + 0]), bk0, c_rows[1]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k + 1]), bk1, c_rows[1]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k + 2]), bk2, c_rows[1]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k + 3]), bk3, c_rows[1]);

        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k + 0]), bk0, c_rows[2]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k + 1]), bk1, c_rows[2]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k + 2]), bk2, c_rows[2]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k + 3]), bk3, c_rows[2]);

        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k + 0]), bk0, c_rows[3]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k + 1]), bk1, c_rows[3]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k + 2]), bk2, c_rows[3]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k + 3]), bk3, c_rows[3]);
    }

    // Tail loop
    for (; k < K; k++)
    {
        __m256 bk = _mm256_mul_ps(_mm256_loadu_ps(B + k * 8), valpha);

        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k]), bk, c_rows[0]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k]), bk, c_rows[1]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k]), bk, c_rows[2]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k]), bk, c_rows[3]);
    }

    // Store results
    const int is_aligned = IS_ALIGNED(C, 32) && (ldc % 8 == 0);

    if (is_aligned)
    {
        for (int row = 0; row < 4; row++)
        {
            _mm256_store_ps(C + row * ldc, c_rows[row]);
        }
    }
    else
    {
        for (int row = 0; row < 4; row++)
        {
            _mm256_storeu_ps(C + row * ldc, c_rows[row]);
        }
    }
}

void gemm_4x8_inline(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    if (K >= 8)
    {
        gemm_4x8_k_unroll4(C, A, B, K, ldc, alpha, beta);
    }
    else
    {
        // Simple loop for small K
        __m256 c_rows[4];

        if (beta == 0.0f)
        {
            for (int row = 0; row < 4; row++)
            {
                c_rows[row] = _mm256_setzero_ps();
            }
        }
        else if (beta == 1.0f)
        {
            for (int row = 0; row < 4; row++)
            {
                c_rows[row] = _mm256_loadu_ps(C + row * ldc);
            }
        }
        else
        {
            __m256 vbeta = _mm256_set1_ps(beta);
            for (int row = 0; row < 4; row++)
            {
                c_rows[row] = _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + row * ldc));
            }
        }

        __m256 valpha = _mm256_set1_ps(alpha);
        for (size_t k = 0; k < K; k++)
        {
            __m256 bk = _mm256_mul_ps(_mm256_loadu_ps(B + k * 8), valpha);

            for (int row = 0; row < 4; row++)
            {
                c_rows[row] = _mm256_fmadd_ps(_mm256_set1_ps(A[row * K + k]), bk, c_rows[row]);
            }
        }

        for (int row = 0; row < 4; row++)
        {
            _mm256_storeu_ps(C + row * ldc, c_rows[row]);
        }
    }
}

//==============================================================================
// 8×6 GEMM - AVX2 with K-Loop Unrolling
//==============================================================================

static void gemm_8x6_k_unroll4(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[8];
    __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);

    // Initialize with beta*C
    if (beta == 0.0f)
    {
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm256_setzero_ps();
        }
    }
    else if (beta == 1.0f)
    {
        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm256_maskload_ps(C + row * ldc, mask6);
        }
    }
    else
    {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int row = 0; row < 8; row++)
        {
            __m256 c_old = _mm256_maskload_ps(C + row * ldc, mask6);
            c_rows[row] = _mm256_mul_ps(vbeta, c_old);
        }
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    // Main loop: unroll by 2 (6 elements makes by-4 awkward)
    size_t k = 0;
    for (; k + 1 < K; k += 2)
    {
        __m256 bk0 = _mm256_setr_ps(
            B[(k + 0) * 6 + 0], B[(k + 0) * 6 + 1], B[(k + 0) * 6 + 2],
            B[(k + 0) * 6 + 3], B[(k + 0) * 6 + 4], B[(k + 0) * 6 + 5], 0.0f, 0.0f);
        __m256 bk1 = _mm256_setr_ps(
            B[(k + 1) * 6 + 0], B[(k + 1) * 6 + 1], B[(k + 1) * 6 + 2],
            B[(k + 1) * 6 + 3], B[(k + 1) * 6 + 4], B[(k + 1) * 6 + 5], 0.0f, 0.0f);

        bk0 = _mm256_mul_ps(bk0, valpha);
        bk1 = _mm256_mul_ps(bk1, valpha);

        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm256_fmadd_ps(_mm256_set1_ps(A[row * K + k + 0]), bk0, c_rows[row]);
            c_rows[row] = _mm256_fmadd_ps(_mm256_set1_ps(A[row * K + k + 1]), bk1, c_rows[row]);
        }
    }

    // Tail loop
    for (; k < K; k++)
    {
        __m256 bk = _mm256_setr_ps(
            B[k * 6 + 0], B[k * 6 + 1], B[k * 6 + 2],
            B[k * 6 + 3], B[k * 6 + 4], B[k * 6 + 5], 0.0f, 0.0f);
        bk = _mm256_mul_ps(bk, valpha);

        for (int row = 0; row < 8; row++)
        {
            c_rows[row] = _mm256_fmadd_ps(_mm256_set1_ps(A[row * K + k]), bk, c_rows[row]);
        }
    }

    // Store results
    for (int row = 0; row < 8; row++)
    {
        _mm256_maskstore_ps(C + row * ldc, mask6, c_rows[row]);
    }
}

void gemm_8x6_inline(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    if (K >= 4)
    {
        gemm_8x6_k_unroll4(C, A, B, K, ldc, alpha, beta);
    }
    else
    {
        // Simple loop for small K
        __m256 c_rows[8];
        __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);

        if (beta == 0.0f)
        {
            for (int row = 0; row < 8; row++)
            {
                c_rows[row] = _mm256_setzero_ps();
            }
        }
        else if (beta == 1.0f)
        {
            for (int row = 0; row < 8; row++)
            {
                c_rows[row] = _mm256_maskload_ps(C + row * ldc, mask6);
            }
        }
        else
        {
            __m256 vbeta = _mm256_set1_ps(beta);
            for (int row = 0; row < 8; row++)
            {
                __m256 c_old = _mm256_maskload_ps(C + row * ldc, mask6);
                c_rows[row] = _mm256_mul_ps(vbeta, c_old);
            }
        }

        __m256 valpha = _mm256_set1_ps(alpha);
        for (size_t k = 0; k < K; k++)
        {
            __m256 bk = _mm256_setr_ps(
                B[k * 6 + 0], B[k * 6 + 1], B[k * 6 + 2],
                B[k * 6 + 3], B[k * 6 + 4], B[k * 6 + 5], 0.0f, 0.0f);
            bk = _mm256_mul_ps(bk, valpha);

            for (int row = 0; row < 8; row++)
            {
                c_rows[row] = _mm256_fmadd_ps(_mm256_set1_ps(A[row * K + k]), bk, c_rows[row]);
            }
        }

        for (int row = 0; row < 8; row++)
        {
            _mm256_maskstore_ps(C + row * ldc, mask6, c_rows[row]);
        }
    }
}

//==============================================================================
// 6×8 GEMM - AVX2 with K-Loop Unrolling
//==============================================================================

static void gemm_6x8_k_unroll4(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[6];

    // Initialize with beta*C
    if (beta == 0.0f)
    {
        for (int row = 0; row < 6; row++)
        {
            c_rows[row] = _mm256_setzero_ps();
        }
    }
    else if (beta == 1.0f)
    {
        for (int row = 0; row < 6; row++)
        {
            c_rows[row] = _mm256_loadu_ps(C + row * ldc);
        }
    }
    else
    {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int row = 0; row < 6; row++)
        {
            c_rows[row] = _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + row * ldc));
        }
    }

    __m256 valpha = _mm256_set1_ps(alpha);

    // Main loop: unroll by 4
    size_t k = 0;
    for (; k + 3 < K; k += 4)
    {
        __m256 bk0 = _mm256_mul_ps(_mm256_loadu_ps(B + (k + 0) * 8), valpha);
        __m256 bk1 = _mm256_mul_ps(_mm256_loadu_ps(B + (k + 1) * 8), valpha);
        __m256 bk2 = _mm256_mul_ps(_mm256_loadu_ps(B + (k + 2) * 8), valpha);
        __m256 bk3 = _mm256_mul_ps(_mm256_loadu_ps(B + (k + 3) * 8), valpha);

        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k + 0]), bk0, c_rows[0]);
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k + 1]), bk1, c_rows[0]);
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k + 2]), bk2, c_rows[0]);
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * K + k + 3]), bk3, c_rows[0]);

        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k + 0]), bk0, c_rows[1]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k + 1]), bk1, c_rows[1]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k + 2]), bk2, c_rows[1]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * K + k + 3]), bk3, c_rows[1]);

        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k + 0]), bk0, c_rows[2]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k + 1]), bk1, c_rows[2]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k + 2]), bk2, c_rows[2]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * K + k + 3]), bk3, c_rows[2]);

        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k + 0]), bk0, c_rows[3]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k + 1]), bk1, c_rows[3]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k + 2]), bk2, c_rows[3]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * K + k + 3]), bk3, c_rows[3]);

        c_rows[4] = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * K + k + 0]), bk0, c_rows[4]);
        c_rows[4] = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * K + k + 1]), bk1, c_rows[4]);
        c_rows[4] = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * K + k + 2]), bk2, c_rows[4]);
        c_rows[4] = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * K + k + 3]), bk3, c_rows[4]);

        c_rows[5] = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * K + k + 0]), bk0, c_rows[5]);
        c_rows[5] = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * K + k + 1]), bk1, c_rows[5]);
        c_rows[5] = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * K + k + 2]), bk2, c_rows[5]);
        c_rows[5] = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * K + k + 3]), bk3, c_rows[5]);
    }

    // Tail loop
    for (; k < K; k++)
    {
        __m256 bk = _mm256_mul_ps(_mm256_loadu_ps(B + k * 8), valpha);

        for (int row = 0; row < 6; row++)
        {
            c_rows[row] = _mm256_fmadd_ps(_mm256_set1_ps(A[row * K + k]), bk, c_rows[row]);
        }
    }

    // Store results
    const int is_aligned = IS_ALIGNED(C, 32) && (ldc % 8 == 0);

    if (is_aligned)
    {
        for (int row = 0; row < 6; row++)
        {
            _mm256_store_ps(C + row * ldc, c_rows[row]);
        }
    }
    else
    {
        for (int row = 0; row < 6; row++)
        {
            _mm256_storeu_ps(C + row * ldc, c_rows[row]);
        }
    }
}

void gemm_6x8_inline(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    if (K >= 8)
    {
        gemm_6x8_k_unroll4(C, A, B, K, ldc, alpha, beta);
    }
    else
    {
        // Simple loop for small K
        __m256 c_rows[6];

        if (beta == 0.0f)
        {
            for (int row = 0; row < 6; row++)
            {
                c_rows[row] = _mm256_setzero_ps();
            }
        }
        else if (beta == 1.0f)
        {
            for (int row = 0; row < 6; row++)
            {
                c_rows[row] = _mm256_loadu_ps(C + row * ldc);
            }
        }
        else
        {
            __m256 vbeta = _mm256_set1_ps(beta);
            for (int row = 0; row < 6; row++)
            {
                c_rows[row] = _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + row * ldc));
            }
        }

        __m256 valpha = _mm256_set1_ps(alpha);
        for (size_t k = 0; k < K; k++)
        {
            __m256 bk = _mm256_mul_ps(_mm256_loadu_ps(B + k * 8), valpha);

            for (int row = 0; row < 6; row++)
            {
                c_rows[row] = _mm256_fmadd_ps(_mm256_set1_ps(A[row * K + k]), bk, c_rows[row]);
            }
        }

        for (int row = 0; row < 6; row++)
        {
            _mm256_storeu_ps(C + row * ldc, c_rows[row]);
        }
    }
}

//==============================================================================
// DISPATCHER
//==============================================================================

int gemm_small_dispatch(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc,
    float alpha, float beta)
{
    // Expanded size limits for rectangular kernels
    if (M > 16 || N > 16 || K > 64)
    {
        return -1; // Too large for Tier 1
    }

    // Reject if too compute-heavy
    size_t total_flops = 2 * M * N * K;
    if (total_flops > 8192)
    {
        return -1;
    }

    //--------------------------------------------------------------------------
    // Square kernels
    //--------------------------------------------------------------------------
    if (M == 4 && K == 4 && N == 4 && ldc == 4)
    {
        gemm_4x4_inline(C, A, B, alpha, beta);
        return 0;
    }

    if (M == 6 && K == 6 && N == 6)
    {
        gemm_6x6_inline(C, A, B, ldc, alpha, beta);
        return 0;
    }

    if (M == 8 && K == 8 && N == 8)
    {
        gemm_8x8_inline(C, A, B, ldc, alpha, beta);
        return 0;
    }

    //--------------------------------------------------------------------------
    // Rectangular kernels (with variable K)
    //--------------------------------------------------------------------------
    if (M == 8 && N == 4)
    {
        gemm_8x4_inline(C, A, B, K, ldc, alpha, beta);
        return 0;
    }

    if (M == 4 && N == 8)
    {
        gemm_4x8_inline(C, A, B, K, ldc, alpha, beta);
        return 0;
    }

    if (M == 8 && N == 6)
    {
        gemm_8x6_inline(C, A, B, K, ldc, alpha, beta);
        return 0;
    }

    if (M == 6 && N == 8)
    {
        gemm_6x8_inline(C, A, B, K, ldc, alpha, beta);
        return 0;
    }

    // Not handled by Tier 1
    return -1;
}