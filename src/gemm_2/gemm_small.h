/**
 * @file gemm_small.h
 * @brief Tier 1: Small Fixed-Size Matrix Kernels (Register-Only)
 */

#ifndef GEMM_SMALL_H
#define GEMM_SMALL_H

#include <immintrin.h>
#include <stddef.h>

//==============================================================================
// TIER 1 KERNELS - Forward Declarations
//==============================================================================

/**
 * @brief 4×4 GEMM entirely in SSE registers
 * 
 * CRITICAL: Assumes C, A, B are contiguous (row-major 4×4 blocks)
 */
void gemm_4x4_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    float alpha, float beta);

/**
 * @brief 6×6 GEMM in AVX2 registers
 * 
 * Handles arbitrary ldc for C
 */
void gemm_6x6_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t ldc,
    float alpha,
    float beta);

/**
 * @brief 8×8 GEMM with transpose optimization
 * 
 * Handles arbitrary ldc for C
 */
void gemm_8x8_inline(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t ldc,
    float alpha,
    float beta);

//==============================================================================
// DISPATCHER
//==============================================================================

/**
 * @brief Tier 1 dispatcher for small matrices
 * 
 * @return 0 if handled by Tier 1, -1 if needs Tier 2
 */
int gemm_small_dispatch(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    size_t ldc,
    float alpha, float beta);

#endif // GEMM_SMALL_H