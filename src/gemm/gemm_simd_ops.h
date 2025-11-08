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
    __m256 r0 = rows[0], r1 = rows[1], r2 = rows[2], r3 = rows[3];
    __m256 r4 = rows[4], r5 = rows[5], r6 = rows[6], r7 = rows[7];

    __m256 u0 = _mm256_unpacklo_ps(r0, r1);
    __m256 u1 = _mm256_unpackhi_ps(r0, r1);
    __m256 u2 = _mm256_unpacklo_ps(r2, r3);
    __m256 u3 = _mm256_unpackhi_ps(r2, r3);

    __m256 v0 = _mm256_shuffle_ps(u0, u2, 0x44);
    __m256 v1 = _mm256_shuffle_ps(u0, u2, 0xEE);
    __m256 v2 = _mm256_shuffle_ps(u1, u3, 0x44);
    __m256 v3 = _mm256_shuffle_ps(u1, u3, 0xEE);

    __m256 w0 = _mm256_unpacklo_ps(r4, r5);
    __m256 w1 = _mm256_unpackhi_ps(r4, r5);
    __m256 w2 = _mm256_unpacklo_ps(r6, r7);
    __m256 w3 = _mm256_unpackhi_ps(r6, r7);

    __m256 x0 = _mm256_shuffle_ps(w0, w2, 0x44);
    __m256 x1 = _mm256_shuffle_ps(w0, w2, 0xEE);
    __m256 x2 = _mm256_shuffle_ps(w1, w3, 0x44);
    __m256 x3 = _mm256_shuffle_ps(w1, w3, 0xEE);

    rows[0] = _mm256_permute2f128_ps(v0, x0, 0x20);
    rows[4] = _mm256_permute2f128_ps(v0, x0, 0x31);
    rows[1] = _mm256_permute2f128_ps(v1, x1, 0x20);
    rows[5] = _mm256_permute2f128_ps(v1, x1, 0x31);
    rows[2] = _mm256_permute2f128_ps(v2, x2, 0x20);
    rows[6] = _mm256_permute2f128_ps(v2, x2, 0x31);
    rows[3] = _mm256_permute2f128_ps(v3, x3, 0x20);
    rows[7] = _mm256_permute2f128_ps(v3, x3, 0x31);
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