/**
 * @file gemm_simd_ops.h
 * @brief Minimal SIMD helpers for GEMM kernels
 */

#ifndef GEMM_SIMD_OPS_H
#define GEMM_SIMD_OPS_H

#include <immintrin.h>
#include <stddef.h>

//==============================================================================
// TRANSPOSE HELPER (Only function actually used across kernels)
//==============================================================================

#ifdef __AVX2__

/**
 * @brief In-register 8×8 transpose for AVX2
 * @param[in,out] rows Array of 8 vectors (input: columns, output: rows)
 */
static inline void gemm_transpose_8x8_avx2(__m256 *rows)
{
    // Step 1: Unpack pairs
    __m256 t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    __m256 t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    __m256 t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    __m256 t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    __m256 t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    __m256 t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    __m256 t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    __m256 t7 = _mm256_unpackhi_ps(rows[6], rows[7]);

    // Step 2: Shuffle 4-element blocks
    __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    // Step 3: Permute 128-bit lanes
    rows[0] = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    rows[1] = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    rows[2] = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    rows[3] = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    rows[4] = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    rows[5] = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    rows[6] = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    rows[7] = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}

/**
 * @brief Build AVX2 mask for n active lanes (0-8)
 */
static inline __m256i gemm_build_mask_avx2(int lanes)
{
    // Portable alignment
    #if defined(_MSC_VER)
        __declspec(align(32))
    #else
        __attribute__((aligned(32)))
    #endif
    static const int mask_table[9][8] = {
        { 0,  0,  0,  0,  0,  0,  0,  0},
        {-1,  0,  0,  0,  0,  0,  0,  0},
        {-1, -1,  0,  0,  0,  0,  0,  0},
        {-1, -1, -1,  0,  0,  0,  0,  0},
        {-1, -1, -1, -1,  0,  0,  0,  0},
        {-1, -1, -1, -1, -1,  0,  0,  0},
        {-1, -1, -1, -1, -1, -1,  0,  0},
        {-1, -1, -1, -1, -1, -1, -1,  0},
        {-1, -1, -1, -1, -1, -1, -1, -1}
    };
    
    lanes = (lanes < 0) ? 0 : ((lanes > 8) ? 8 : lanes);
    return _mm256_loadu_si256((const __m256i*)mask_table[lanes]);
}

#endif /* __AVX2__ */

#endif /* GEMM_SIMD_OPS_H */