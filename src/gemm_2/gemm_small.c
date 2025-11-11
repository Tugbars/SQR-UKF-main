/**
 * @file gemm_small.c
 * @brief Tier 1: Small Fixed-Size Matrix Kernels (Register-Only)
 * 
 * **PRODUCTION QUALITY ONLY**
 * - Removed broken 12×12 and 16×16 kernels
 * - Fixed beta handling in all kernels
 * - Fixed ldc handling in 6×6
 * - Fixed dispatcher logic
 * 
 * @author TUGBARS
 * @date 2025
 */

#include "gemm_small.h"
#include <string.h>


//==============================================================================
// TIER 1 DISPATCHER (CORRECTED VERSION)
//==============================================================================

/**
 * @brief Route small matrices to optimized kernels
 * 
 * FIXED BUGS:
 * 1. No longer requires M == K == N (square)
 * 2. Handles ldc != N properly
 * 3. Falls back to Tier 2 for unsupported shapes
 * 
 * @return 0 if handled by Tier 1, -1 if needs Tier 2
 */
int gemm_small_dispatch(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc,
    float alpha, float beta)
{
    // Size limits for Tier 1 (small enough to fit in registers)
    if (M > 8 || N > 8 || K > 8) {
        return -1;  // Too large for Tier 1
    }
    
    // Total element count check (avoid tiny matrices with overhead)
    size_t total_ops = M * N * K;
    if (total_ops > 512) {
        return -1;  // Large enough to benefit from Tier 2 blocking
    }
    
    //--------------------------------------------------------------------------
    // 4×4 Kernel: Requires contiguous storage
    //--------------------------------------------------------------------------
    if (M == 4 && K == 4 && N == 4 && ldc == 4) {
        gemm_4x4_inline(C, A, B, alpha, beta);
        return 0;
    }
    
    //--------------------------------------------------------------------------
    // 6×6 Kernel: Handles arbitrary ldc
    //--------------------------------------------------------------------------
    if (M == 6 && K == 6 && N == 6) {
        gemm_6x6_inline(C, A, B, ldc, alpha, beta);
        return 0;
    }
    
    //--------------------------------------------------------------------------
    // 8×8 Kernel: Handles arbitrary ldc
    //--------------------------------------------------------------------------
    if (M == 8 && K == 8 && N == 8) {
        gemm_8x8_inline(C, A, B, ldc, alpha, beta);
        return 0;
    }
    
    //--------------------------------------------------------------------------
    // Rectangular variants (future expansion)
    //--------------------------------------------------------------------------
    // TODO: Add 4×8, 8×4, 6×8, etc. if profiling shows they're hot paths
    
    // Not handled by Tier 1
    return -1;
}