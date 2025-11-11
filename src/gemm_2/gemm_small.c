/**
 * @file gemm_small.c
 * @brief Tier 1: Small Fixed-Size Matrix Kernels (Register-Only)
 */

#include "gemm_small.h"
#include "gemm_simd_ops.h"
#include <string.h>

//==============================================================================
// 4×4 GEMM - SSE Implementation
//==============================================================================

void gemm_4x4_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    float alpha, float beta)
{
    // Load entire A and B matrices
    __m128 a0 = _mm_loadu_ps(A + 0);
    __m128 a1 = _mm_loadu_ps(A + 4);
    __m128 a2 = _mm_loadu_ps(A + 8);
    __m128 a3 = _mm_loadu_ps(A + 12);
    
    __m128 b0 = _mm_loadu_ps(B + 0);
    __m128 b1 = _mm_loadu_ps(B + 4);
    __m128 b2 = _mm_loadu_ps(B + 8);
    __m128 b3 = _mm_loadu_ps(B + 12);
    
    __m128 valpha = _mm_set1_ps(alpha);
    
    // Compute rows using permute-based broadcasting
    __m128 c0 = _mm_mul_ps(_mm_permute_ps(a0, 0x00), b0);
    c0 = _mm_fmadd_ps(_mm_permute_ps(a0, 0x55), b1, c0);
    c0 = _mm_fmadd_ps(_mm_permute_ps(a0, 0xAA), b2, c0);
    c0 = _mm_fmadd_ps(_mm_permute_ps(a0, 0xFF), b3, c0);
    c0 = _mm_mul_ps(c0, valpha);
    
    __m128 c1 = _mm_mul_ps(_mm_permute_ps(a1, 0x00), b0);
    c1 = _mm_fmadd_ps(_mm_permute_ps(a1, 0x55), b1, c1);
    c1 = _mm_fmadd_ps(_mm_permute_ps(a1, 0xAA), b2, c1);
    c1 = _mm_fmadd_ps(_mm_permute_ps(a1, 0xFF), b3, c1);
    c1 = _mm_mul_ps(c1, valpha);
    
    __m128 c2 = _mm_mul_ps(_mm_permute_ps(a2, 0x00), b0);
    c2 = _mm_fmadd_ps(_mm_permute_ps(a2, 0x55), b1, c2);
    c2 = _mm_fmadd_ps(_mm_permute_ps(a2, 0xAA), b2, c2);
    c2 = _mm_fmadd_ps(_mm_permute_ps(a2, 0xFF), b3, c2);
    c2 = _mm_mul_ps(c2, valpha);
    
    __m128 c3 = _mm_mul_ps(_mm_permute_ps(a3, 0x00), b0);
    c3 = _mm_fmadd_ps(_mm_permute_ps(a3, 0x55), b1, c3);
    c3 = _mm_fmadd_ps(_mm_permute_ps(a3, 0xAA), b2, c3);
    c3 = _mm_fmadd_ps(_mm_permute_ps(a3, 0xFF), b3, c3);
    c3 = _mm_mul_ps(c3, valpha);
    
    // Apply beta
    if (beta == 0.0f) {
        _mm_storeu_ps(C + 0, c0);
        _mm_storeu_ps(C + 4, c1);
        _mm_storeu_ps(C + 8, c2);
        _mm_storeu_ps(C + 12, c3);
    } else if (beta == 1.0f) {
        _mm_storeu_ps(C + 0, _mm_add_ps(_mm_loadu_ps(C + 0), c0));
        _mm_storeu_ps(C + 4, _mm_add_ps(_mm_loadu_ps(C + 4), c1));
        _mm_storeu_ps(C + 8, _mm_add_ps(_mm_loadu_ps(C + 8), c2));
        _mm_storeu_ps(C + 12, _mm_add_ps(_mm_loadu_ps(C + 12), c3));
    } else {
        __m128 vbeta = _mm_set1_ps(beta);
        _mm_storeu_ps(C + 0, _mm_fmadd_ps(vbeta, _mm_loadu_ps(C + 0), c0));
        _mm_storeu_ps(C + 4, _mm_fmadd_ps(vbeta, _mm_loadu_ps(C + 4), c1));
        _mm_storeu_ps(C + 8, _mm_fmadd_ps(vbeta, _mm_loadu_ps(C + 8), c2));
        _mm_storeu_ps(C + 12, _mm_fmadd_ps(vbeta, _mm_loadu_ps(C + 12), c3));
    }
}

//==============================================================================
// 6×6 GEMM - AVX2 Implementation
//==============================================================================

void gemm_6x6_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[6];
    __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);
    
    // Initialize c_rows with beta*C
    if (beta == 0.0f) {
        for (int i = 0; i < 6; i++) {
            c_rows[i] = _mm256_setzero_ps();
        }
    } else {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int i = 0; i < 6; i++) {
            __m256 c_old = _mm256_maskload_ps(C + i * ldc, mask6);
            c_rows[i] = _mm256_mul_ps(vbeta, c_old);
        }
    }
    
    // Load B ROWS (not columns!) and pre-scale by alpha
    __m256 b_rows[6];
    for (int k = 0; k < 6; k++) {
        // Load row k of B
        b_rows[k] = _mm256_setr_ps(
            B[k*6 + 0], B[k*6 + 1], B[k*6 + 2],
            B[k*6 + 3], B[k*6 + 4], B[k*6 + 5], 0, 0);
        
        if (alpha != 1.0f) {
            __m256 valpha = _mm256_set1_ps(alpha);
            b_rows[k] = _mm256_mul_ps(b_rows[k], valpha);
        }
    }
    
    // K-outer loop: process one row of B at a time
    for (int k = 0; k < 6; k++) {
        __m256 bk = b_rows[k];  // Row k of B
        for (int i = 0; i < 6; i++) {
            __m256 a_ik = _mm256_set1_ps(A[i*6 + k]);
            c_rows[i] = _mm256_fmadd_ps(a_ik, bk, c_rows[i]);
        }
    }
    
    // Store results
    for (int i = 0; i < 6; i++) {
        _mm256_maskstore_ps(C + i * ldc, mask6, c_rows[i]);
    }
}


//==============================================================================
// 8×8 GEMM - AVX2 Implementation with Transpose
//==============================================================================

void gemm_8x8_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[8];
    
    // Load B rows directly (NO TRANSPOSE!)
    __m256 b_rows[8];
    for (int k = 0; k < 8; k++) {
        b_rows[k] = _mm256_loadu_ps(B + k * 8);  // Row k
    }
    
    // Pre-scale B by alpha
    if (alpha != 1.0f) {
        __m256 valpha = _mm256_set1_ps(alpha);
        for (int k = 0; k < 8; k++) {
            b_rows[k] = _mm256_mul_ps(b_rows[k], valpha);
        }
    }
    
    // Initialize accumulators with beta*C
    if (beta == 0.0f) {
        for (int row = 0; row < 8; row++) {
            c_rows[row] = _mm256_setzero_ps();
        }
    } else if (beta == 1.0f) {
        for (int row = 0; row < 8; row++) {
            c_rows[row] = _mm256_loadu_ps(C + row * ldc);
        }
    } else {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int row = 0; row < 8; row++) {
            c_rows[row] = _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + row * ldc));
        }
    }
    
    // Compute C += A*B (k-outer loop)
    for (int k = 0; k < 8; k++) {
        __m256 bk = b_rows[k];  // Row k of B
        
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0*8 + k]), bk, c_rows[0]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1*8 + k]), bk, c_rows[1]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2*8 + k]), bk, c_rows[2]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3*8 + k]), bk, c_rows[3]);
        c_rows[4] = _mm256_fmadd_ps(_mm256_set1_ps(A[4*8 + k]), bk, c_rows[4]);
        c_rows[5] = _mm256_fmadd_ps(_mm256_set1_ps(A[5*8 + k]), bk, c_rows[5]);
        c_rows[6] = _mm256_fmadd_ps(_mm256_set1_ps(A[6*8 + k]), bk, c_rows[6]);
        c_rows[7] = _mm256_fmadd_ps(_mm256_set1_ps(A[7*8 + k]), bk, c_rows[7]);
    }
    
    // Store results
    for (int row = 0; row < 8; row++) {
        _mm256_storeu_ps(C + row * ldc, c_rows[row]);
    }
}

//==============================================================================
// 8×4 GEMM - SSE Implementation (Rectangular, Tall Panel)
//==============================================================================

/**
 * @brief 8×4 GEMM kernel using SSE registers
 * 
 * Architecture:
 * - 8 row accumulators (__m128 × 8 = 8 XMM registers)
 * - 4 column vectors from B (__m128 × 4 = 4 XMM registers)
 * - K-outer loop for optimal register reuse
 * - Total: 12 XMM registers (well within 16 XMM limit)
 * 
 * Performance characteristics:
 * - ~2*8*4*K FLOPs = 64K FLOPs
 * - ~8K cycles for K=8 (assuming 8 cycles per FMA chain)
 * - Memory: Streams A (8K floats), loads B once (4K floats)
 * 
 * Optimization notes:
 * - Pre-scales B by alpha (4K multiplies instead of 32K)
 * - K-outer loop enables 8 independent FMA chains (ILP=8)
 * - SSE width (4 floats) perfectly matches column count
 */
void gemm_8x4_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    //==========================================================================
    // Step 1: Allocate accumulators (8 rows × 4 columns)
    //==========================================================================
    __m128 c_rows[8];
    
    //==========================================================================
    // Step 2: Load and transpose B (K×4 → 4 columns of length K)
    //==========================================================================
    // We need B in column-major format for efficient broadcast
    // B is stored row-major: B[k*4 + j] where k ∈ [0,K), j ∈ [0,4)
    // We want: b_col[j][k] for efficient access
    
    // For small K, we can load into registers and transpose
    // For large K, we'll stream from memory (see below)
    
    //==========================================================================
    // Step 3: Initialize accumulators with beta*C
    //==========================================================================
    if (beta == 0.0f) {
        // Zero initialization (overwrite C)
        for (int row = 0; row < 8; row++) {
            c_rows[row] = _mm_setzero_ps();
        }
    } else if (beta == 1.0f) {
        // Load C unchanged (accumulate mode)
        for (int row = 0; row < 8; row++) {
            c_rows[row] = _mm_loadu_ps(C + row * ldc);
        }
    } else {
        // Scale C by beta
        __m128 vbeta = _mm_set1_ps(beta);
        for (int row = 0; row < 8; row++) {
            c_rows[row] = _mm_mul_ps(vbeta, _mm_loadu_ps(C + row * ldc));
        }
    }
    
    //==========================================================================
    // Step 4: Compute C += alpha * A * B (K-outer loop)
    //==========================================================================
    // Strategy: Process one column of A (and one row of B) at a time
    // This keeps B column hot in register across all 8 output rows
    
    __m128 valpha = _mm_set1_ps(alpha);
    
    for (size_t k = 0; k < K; k++) {
        // Load one row of B (4 floats) and scale by alpha
        __m128 b_row = _mm_loadu_ps(B + k * 4);
        __m128 bk = _mm_mul_ps(b_row, valpha);  // Pre-scale by alpha
        
        // Update all 8 output rows with this column of A
        // These 8 FMAs are independent → excellent ILP
        c_rows[0] = _mm_fmadd_ps(_mm_set1_ps(A[0*K + k]), bk, c_rows[0]);
        c_rows[1] = _mm_fmadd_ps(_mm_set1_ps(A[1*K + k]), bk, c_rows[1]);
        c_rows[2] = _mm_fmadd_ps(_mm_set1_ps(A[2*K + k]), bk, c_rows[2]);
        c_rows[3] = _mm_fmadd_ps(_mm_set1_ps(A[3*K + k]), bk, c_rows[3]);
        c_rows[4] = _mm_fmadd_ps(_mm_set1_ps(A[4*K + k]), bk, c_rows[4]);
        c_rows[5] = _mm_fmadd_ps(_mm_set1_ps(A[5*K + k]), bk, c_rows[5]);
        c_rows[6] = _mm_fmadd_ps(_mm_set1_ps(A[6*K + k]), bk, c_rows[6]);
        c_rows[7] = _mm_fmadd_ps(_mm_set1_ps(A[7*K + k]), bk, c_rows[7]);
    }
    
    //==========================================================================
    // Step 5: Store results
    //==========================================================================
    for (int row = 0; row < 8; row++) {
        _mm_storeu_ps(C + row * ldc, c_rows[row]);
    }
}

//==============================================================================
// 4×8 GEMM - AVX2 Implementation (Rectangular, Wide Panel)
//==============================================================================

/**
 * @brief 4×8 GEMM kernel using AVX2 registers
 * 
 * Architecture:
 * - 4 row accumulators (__m256 × 4 = 4 YMM registers)
 * - 8 column vectors from B (__m256 × 8 = 8 YMM registers)
 * - K-outer loop for optimal register reuse
 * - Total: 12 YMM registers (well within 16 YMM limit)
 * 
 * Performance characteristics:
 * - ~2*4*8*K FLOPs = 64K FLOPs
 * - Memory: Streams A (4K floats), loads B once (8K floats)
 */
void gemm_4x8_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    //==========================================================================
    // Step 1: Allocate accumulators (4 rows × 8 columns)
    //==========================================================================
    __m256 c_rows[4];
    
    //==========================================================================
    // Step 2: Initialize accumulators with beta*C
    //==========================================================================
    if (beta == 0.0f) {
        for (int row = 0; row < 4; row++) {
            c_rows[row] = _mm256_setzero_ps();
        }
    } else if (beta == 1.0f) {
        for (int row = 0; row < 4; row++) {
            c_rows[row] = _mm256_loadu_ps(C + row * ldc);
        }
    } else {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int row = 0; row < 4; row++) {
            c_rows[row] = _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + row * ldc));
        }
    }
    
    //==========================================================================
    // Step 3: Compute C += alpha * A * B (K-outer loop)
    //==========================================================================
    __m256 valpha = _mm256_set1_ps(alpha);
    
    for (size_t k = 0; k < K; k++) {
        // Load one row of B (8 floats) and scale by alpha
        __m256 b_row = _mm256_loadu_ps(B + k * 8);
        __m256 bk = _mm256_mul_ps(b_row, valpha);
        
        // Update all 4 output rows
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0*K + k]), bk, c_rows[0]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1*K + k]), bk, c_rows[1]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2*K + k]), bk, c_rows[2]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3*K + k]), bk, c_rows[3]);
    }
    
    //==========================================================================
    // Step 4: Store results
    //==========================================================================
    for (int row = 0; row < 4; row++) {
        _mm256_storeu_ps(C + row * ldc, c_rows[row]);
    }
}

//==============================================================================
// 8×6 GEMM - AVX2 Implementation with Masking
//==============================================================================

/**
 * @brief 8×6 GEMM kernel
 * 
 * Architecture:
 * - 8 row accumulators (__m256 × 8 = 8 YMM registers)
 * - Streams B rows (K×6, loaded with padding)
 * - Uses masked stores for 6-wide output
 * - Total: ~10 YMM registers
 * 
 * Performance:
 * - ~2*8*6*K FLOPs = 96K FLOPs
 * - Excellent for 8-state × 6-DOF Kalman operations
 */
void gemm_8x6_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[8];
    __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);
    
    //==========================================================================
    // Step 1: Initialize accumulators with beta*C
    //==========================================================================
    if (beta == 0.0f) {
        for (int row = 0; row < 8; row++) {
            c_rows[row] = _mm256_setzero_ps();
        }
    } else if (beta == 1.0f) {
        for (int row = 0; row < 8; row++) {
            c_rows[row] = _mm256_maskload_ps(C + row * ldc, mask6);
        }
    } else {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int row = 0; row < 8; row++) {
            __m256 c_old = _mm256_maskload_ps(C + row * ldc, mask6);
            c_rows[row] = _mm256_mul_ps(vbeta, c_old);
        }
    }
    
    //==========================================================================
    // Step 2: Compute C += alpha * A * B (K-outer loop)
    //==========================================================================
    __m256 valpha = _mm256_set1_ps(alpha);
    
    for (size_t k = 0; k < K; k++) {
        // Load row k of B (6 floats, padded to 8)
        __m256 b_row = _mm256_setr_ps(
            B[k*6 + 0], B[k*6 + 1], B[k*6 + 2],
            B[k*6 + 3], B[k*6 + 4], B[k*6 + 5], 0.0f, 0.0f);
        __m256 bk = _mm256_mul_ps(b_row, valpha);
        
        // Update all 8 output rows
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0*K + k]), bk, c_rows[0]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1*K + k]), bk, c_rows[1]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2*K + k]), bk, c_rows[2]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3*K + k]), bk, c_rows[3]);
        c_rows[4] = _mm256_fmadd_ps(_mm256_set1_ps(A[4*K + k]), bk, c_rows[4]);
        c_rows[5] = _mm256_fmadd_ps(_mm256_set1_ps(A[5*K + k]), bk, c_rows[5]);
        c_rows[6] = _mm256_fmadd_ps(_mm256_set1_ps(A[6*K + k]), bk, c_rows[6]);
        c_rows[7] = _mm256_fmadd_ps(_mm256_set1_ps(A[7*K + k]), bk, c_rows[7]);
    }
    
    //==========================================================================
    // Step 3: Store results with masking
    //==========================================================================
    for (int row = 0; row < 8; row++) {
        _mm256_maskstore_ps(C + row * ldc, mask6, c_rows[row]);
    }
}

//==============================================================================
// 6×8 GEMM - AVX2 Implementation
//==============================================================================

/**
 * @brief 6×8 GEMM kernel
 * 
 * Architecture:
 * - 6 row accumulators (__m256 × 6 = 6 YMM registers)
 * - Streams B rows (K×8, full width)
 * - Total: ~8 YMM registers
 * 
 * Performance:
 * - ~2*6*8*K FLOPs = 96K FLOPs
 */
void gemm_6x8_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t K,
    size_t ldc,
    float alpha,
    float beta)
{
    __m256 c_rows[6];
    
    //==========================================================================
    // Step 1: Initialize accumulators with beta*C
    //==========================================================================
    if (beta == 0.0f) {
        for (int row = 0; row < 6; row++) {
            c_rows[row] = _mm256_setzero_ps();
        }
    } else if (beta == 1.0f) {
        for (int row = 0; row < 6; row++) {
            c_rows[row] = _mm256_loadu_ps(C + row * ldc);
        }
    } else {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (int row = 0; row < 6; row++) {
            c_rows[row] = _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + row * ldc));
        }
    }
    
    //==========================================================================
    // Step 2: Compute C += alpha * A * B (K-outer loop)
    //==========================================================================
    __m256 valpha = _mm256_set1_ps(alpha);
    
    for (size_t k = 0; k < K; k++) {
        // Load row k of B (8 floats)
        __m256 b_row = _mm256_loadu_ps(B + k * 8);
        __m256 bk = _mm256_mul_ps(b_row, valpha);
        
        // Update all 6 output rows
        c_rows[0] = _mm256_fmadd_ps(_mm256_set1_ps(A[0*K + k]), bk, c_rows[0]);
        c_rows[1] = _mm256_fmadd_ps(_mm256_set1_ps(A[1*K + k]), bk, c_rows[1]);
        c_rows[2] = _mm256_fmadd_ps(_mm256_set1_ps(A[2*K + k]), bk, c_rows[2]);
        c_rows[3] = _mm256_fmadd_ps(_mm256_set1_ps(A[3*K + k]), bk, c_rows[3]);
        c_rows[4] = _mm256_fmadd_ps(_mm256_set1_ps(A[4*K + k]), bk, c_rows[4]);
        c_rows[5] = _mm256_fmadd_ps(_mm256_set1_ps(A[5*K + k]), bk, c_rows[5]);
    }
    
    //==========================================================================
    // Step 3: Store results
    //==========================================================================
    for (int row = 0; row < 6; row++) {
        _mm256_storeu_ps(C + row * ldc, c_rows[row]);
    }
}

//==============================================================================
// DISPATCHER
//==============================================================================

int gemm_small_dispatch(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc,
    float alpha, float beta)
{
    // Expanded size limits for rectangular kernels
    if (M > 16 || N > 16 || K > 32) {
        return -1;  // Too large for Tier 1
    }
    
    // Reject if too compute-heavy (better to amortize Tier 2 overhead)
    size_t total_flops = 2 * M * N * K;
    if (total_flops > 4096) {
        return -1;
    }
    
    //--------------------------------------------------------------------------
    // Square kernels
    //--------------------------------------------------------------------------
    if (M == 4 && K == 4 && N == 4 && ldc == 4) {
        gemm_4x4_inline(C, A, B, alpha, beta);
        return 0;
    }
    
    if (M == 6 && K == 6 && N == 6) {
        gemm_6x6_inline(C, A, B, ldc, alpha, beta);
        return 0;
    }
    
    if (M == 8 && K == 8 && N == 8) {
        gemm_8x8_inline(C, A, B, ldc, alpha, beta);
        return 0;
    }
    
    //--------------------------------------------------------------------------
    // Rectangular kernels (8×4, 4×8, 8×6, 6×8)
    //--------------------------------------------------------------------------
    if (M == 8 && N == 4) {
        gemm_8x4_inline(C, A, B, K, ldc, alpha, beta);
        return 0;
    }
    
    if (M == 4 && N == 8) {
        gemm_4x8_inline(C, A, B, K, ldc, alpha, beta);
        return 0;
    }
    
    if (M == 8 && N == 6) {
        gemm_8x6_inline(C, A, B, K, ldc, alpha, beta);
        return 0;
    }
    
    if (M == 6 && N == 8) {
        gemm_6x8_inline(C, A, B, K, ldc, alpha, beta);
        return 0;
    }
    
    // Not handled by Tier 1
    return -1;
}