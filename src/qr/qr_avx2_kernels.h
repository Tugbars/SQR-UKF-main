/**
 * @file linalg_qr_avx2_kernels.h
 * @brief Highly optimized AVX2/FMA kernels for blocked compact-WY QR
 *
 * @details
 * This header provides production-quality SIMD kernels for the three BLAS-3-shaped
 * operations in blocked QR:
 *   1. Y = V^T * C    (qrw_compute_Y_avx_opt)
 *   2. Z = T * Y      (qrw_compute_Z_avx_opt)
 *   3. C = C - V * Z  (qrw_apply_VZ_avx_opt)
 *
 * **Optimization techniques:**
 *  - **Register blocking**: 6x16 tiles to maximize register reuse
 *  - **Multiple accumulator pairs**: 4-6 AVX2 accumulators to hide FMA latency (3-5 cycles)
 *  - **Software pipelining**: Interleaved loads/broadcasts with FMAs to saturate ports
 *  - **Prefetching**: Non-temporal and L1/L2 prefetch hints for large matrices
 *  - **Loop unrolling**: Strategic unroll factors for ILP and reduced loop overhead
 *
 * **Performance targets:**
 *  - Raptor Lake (14900KF): 80-90% of peak GEMM throughput
 *  - Typical: 15-25 GFLOPS/core for FP32 on AVX2
 *
 * **SIMD assumptions:**
 *  - AVX2 + FMA available (checked by caller)
 *  - 32-byte alignment for aligned loads where possible
 *  - Unaligned loads used where layout prohibits alignment
 *
 * @note These kernels are designed to match or exceed OpenBLAS Level-3 performance
 *       for the specific QR workload patterns.
 */

#ifndef LINALG_QR_AVX2_KERNELS_H
#define LINALG_QR_AVX2_KERNELS_H

#include <stdint.h>
#include <immintrin.h>

#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE (36 * 1024 * 1024) // 36MB for Intel 14900KF
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /* ===========================================================================================
     * Horizontal sum for __m256 (unchanged, highly optimized already)
     * ===========================================================================================
     */
    static inline float qrw_hsum8_opt(__m256 v)
    {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        __m128 sh = _mm_movehdup_ps(s);
        s = _mm_add_ps(s, sh);
        sh = _mm_movehl_ps(sh, s);
        s = _mm_add_ss(s, sh);
        return _mm_cvtss_f32(s);
    }

    /* ===========================================================================================
     * KERNEL 1: Y = V^T * Cpack  (ib × kc)
     * ===========================================================================================
     * Strategy:
     *  - Process 6 rows of V simultaneously (p, p+1, ..., p+5)
     *  - For each V row, compute dot products with 16 columns of C (j, j+1, ..., j+15)
     *  - Use 12 accumulators (6 rows × 2 AVX vectors for 16 cols)
     *  - Software pipeline: interleave V loads/broadcasts with FMAs
     *  - Prefetch next V rows and C data for L1/L2 reuse
     *
     * Memory access pattern:
     *  - V: broadcast single elements (vp[r*n]) - high reuse
     *  - C: streaming loads across kc (packed layout, row-major within tile)
     *  - Y: streaming stores (ib × kc, row-major)
     */
    static void qrw_compute_Y_avx_opt(const float *__restrict A, uint16_t m, uint16_t n,
                                      uint16_t k, uint16_t ib,
                                      const float *__restrict Cpack, uint16_t m_sub,
                                      uint16_t kc, float *__restrict Y)
    {
        // Check if buffers are large enough to benefit from non-temporal stores
        const size_t buffer_size = (size_t)ib * kc * sizeof(float);
        const int use_streaming = (buffer_size > L3_CACHE_SIZE / 4);

        /* Process ib rows of V in blocks of 6 for register blocking */
        uint16_t p = 0;
        for (; p + 5 < ib; p += 6)
        {
            /* Get pointers to V rows p..p+5 */
            const float *vp0 = A + (size_t)(k + p + 0) * n + (k + p + 0);
            const float *vp1 = A + (size_t)(k + p + 1) * n + (k + p + 1);
            const float *vp2 = A + (size_t)(k + p + 2) * n + (k + p + 2);
            const float *vp3 = A + (size_t)(k + p + 3) * n + (k + p + 3);
            const float *vp4 = A + (size_t)(k + p + 4) * n + (k + p + 4);
            const float *vp5 = A + (size_t)(k + p + 5) * n + (k + p + 5);

            const uint16_t len0 = (uint16_t)(m - (k + p + 0));
            const uint16_t len1 = (uint16_t)(m - (k + p + 1));
            const uint16_t len2 = (uint16_t)(m - (k + p + 2));
            const uint16_t len3 = (uint16_t)(m - (k + p + 3));
            const uint16_t len4 = (uint16_t)(m - (k + p + 4));
            const uint16_t len5 = (uint16_t)(m - (k + p + 5));

            uint16_t j = 0;
            /* Main loop: process 16 columns at a time (2 AVX vectors) */
            for (; j + 15 < kc; j += 16)
            {
                /* 12 accumulators: 6 V-rows × 2 AVX vectors (16 cols total) */
                __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
                __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
                __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
                __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();
                __m256 acc40 = _mm256_setzero_ps(), acc41 = _mm256_setzero_ps();
                __m256 acc50 = _mm256_setzero_ps(), acc51 = _mm256_setzero_ps();

                /* Prefetch next V rows if we're going to process them */
                if (p + 6 + 5 < ib)
                {
                    _mm_prefetch((const char *)(A + (size_t)(k + p + 6) * n + (k + p + 6)), _MM_HINT_T0);
                    _mm_prefetch((const char *)(A + (size_t)(k + p + 7) * n + (k + p + 7)), _MM_HINT_T0);
                }

                /* Accumulate over r for all 6 V rows simultaneously
                 * Software pipelining: load V[r], broadcast, load C[r], FMA
                 * Process 4 iterations at a time for better ILP */
                uint16_t r = 0;
                uint16_t r_end = len0; /* Use minimum length for safety, but len0 is typically shortest */

                /* Unroll by 4 for better pipelining */
                for (; r + 3 < r_end; r += 4)
                {
                    /* Iteration 0 */
                    const __m256 v0_r0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                    const __m256 v1_r0 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                    const __m256 v2_r0 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                    const __m256 v3_r0 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);
                    const float *cptr0 = Cpack + (size_t)(r + p + 0) * kc + j;
                    const __m256 c0_0 = _mm256_loadu_ps(cptr0 + 0);
                    const __m256 c0_1 = _mm256_loadu_ps(cptr0 + 8);

                    acc00 = _mm256_fmadd_ps(v0_r0, c0_0, acc00);
                    acc01 = _mm256_fmadd_ps(v0_r0, c0_1, acc01);

                    const __m256 v4_r0 = _mm256_set1_ps((r < len4) ? vp4[(size_t)r * n] : 0.0f);
                    const __m256 v5_r0 = _mm256_set1_ps((r < len5) ? vp5[(size_t)r * n] : 0.0f);

                    acc10 = _mm256_fmadd_ps(v1_r0, c0_0, acc10);
                    acc11 = _mm256_fmadd_ps(v1_r0, c0_1, acc11);
                    acc20 = _mm256_fmadd_ps(v2_r0, c0_0, acc20);
                    acc21 = _mm256_fmadd_ps(v2_r0, c0_1, acc21);
                    acc30 = _mm256_fmadd_ps(v3_r0, c0_0, acc30);
                    acc31 = _mm256_fmadd_ps(v3_r0, c0_1, acc31);
                    acc40 = _mm256_fmadd_ps(v4_r0, c0_0, acc40);
                    acc41 = _mm256_fmadd_ps(v4_r0, c0_1, acc41);
                    acc50 = _mm256_fmadd_ps(v5_r0, c0_0, acc50);
                    acc51 = _mm256_fmadd_ps(v5_r0, c0_1, acc51);

                    /* Iteration 1 */
                    const __m256 v0_r1 = _mm256_set1_ps(vp0[(size_t)(r + 1) * n]);
                    const __m256 v1_r1 = _mm256_set1_ps((r + 1 < len1) ? vp1[(size_t)(r + 1) * n] : 0.0f);
                    const __m256 v2_r1 = _mm256_set1_ps((r + 1 < len2) ? vp2[(size_t)(r + 1) * n] : 0.0f);
                    const float *cptr1 = Cpack + (size_t)(r + 1 + p + 0) * kc + j;
                    const __m256 c1_0 = _mm256_loadu_ps(cptr1 + 0);
                    const __m256 c1_1 = _mm256_loadu_ps(cptr1 + 8);

                    acc00 = _mm256_fmadd_ps(v0_r1, c1_0, acc00);
                    acc01 = _mm256_fmadd_ps(v0_r1, c1_1, acc01);

                    const __m256 v3_r1 = _mm256_set1_ps((r + 1 < len3) ? vp3[(size_t)(r + 1) * n] : 0.0f);
                    const __m256 v4_r1 = _mm256_set1_ps((r + 1 < len4) ? vp4[(size_t)(r + 1) * n] : 0.0f);
                    const __m256 v5_r1 = _mm256_set1_ps((r + 1 < len5) ? vp5[(size_t)(r + 1) * n] : 0.0f);

                    acc10 = _mm256_fmadd_ps(v1_r1, c1_0, acc10);
                    acc11 = _mm256_fmadd_ps(v1_r1, c1_1, acc11);
                    acc20 = _mm256_fmadd_ps(v2_r1, c1_0, acc20);
                    acc21 = _mm256_fmadd_ps(v2_r1, c1_1, acc21);
                    acc30 = _mm256_fmadd_ps(v3_r1, c1_0, acc30);
                    acc31 = _mm256_fmadd_ps(v3_r1, c1_1, acc31);
                    acc40 = _mm256_fmadd_ps(v4_r1, c1_0, acc40);
                    acc41 = _mm256_fmadd_ps(v4_r1, c1_1, acc41);
                    acc50 = _mm256_fmadd_ps(v5_r1, c1_0, acc50);
                    acc51 = _mm256_fmadd_ps(v5_r1, c1_1, acc51);

                    /* Iteration 2 */
                    const __m256 v0_r2 = _mm256_set1_ps(vp0[(size_t)(r + 2) * n]);
                    const __m256 v1_r2 = _mm256_set1_ps((r + 2 < len1) ? vp1[(size_t)(r + 2) * n] : 0.0f);
                    const __m256 v2_r2 = _mm256_set1_ps((r + 2 < len2) ? vp2[(size_t)(r + 2) * n] : 0.0f);
                    const float *cptr2 = Cpack + (size_t)(r + 2 + p + 0) * kc + j;
                    const __m256 c2_0 = _mm256_loadu_ps(cptr2 + 0);
                    const __m256 c2_1 = _mm256_loadu_ps(cptr2 + 8);

                    acc00 = _mm256_fmadd_ps(v0_r2, c2_0, acc00);
                    acc01 = _mm256_fmadd_ps(v0_r2, c2_1, acc01);

                    const __m256 v3_r2 = _mm256_set1_ps((r + 2 < len3) ? vp3[(size_t)(r + 2) * n] : 0.0f);
                    const __m256 v4_r2 = _mm256_set1_ps((r + 2 < len4) ? vp4[(size_t)(r + 2) * n] : 0.0f);
                    const __m256 v5_r2 = _mm256_set1_ps((r + 2 < len5) ? vp5[(size_t)(r + 2) * n] : 0.0f);

                    acc10 = _mm256_fmadd_ps(v1_r2, c2_0, acc10);
                    acc11 = _mm256_fmadd_ps(v1_r2, c2_1, acc11);
                    acc20 = _mm256_fmadd_ps(v2_r2, c2_0, acc20);
                    acc21 = _mm256_fmadd_ps(v2_r2, c2_1, acc21);
                    acc30 = _mm256_fmadd_ps(v3_r2, c2_0, acc30);
                    acc31 = _mm256_fmadd_ps(v3_r2, c2_1, acc31);
                    acc40 = _mm256_fmadd_ps(v4_r2, c2_0, acc40);
                    acc41 = _mm256_fmadd_ps(v4_r2, c2_1, acc41);
                    acc50 = _mm256_fmadd_ps(v5_r2, c2_0, acc50);
                    acc51 = _mm256_fmadd_ps(v5_r2, c2_1, acc51);

                    /* Iteration 3 */
                    const __m256 v0_r3 = _mm256_set1_ps(vp0[(size_t)(r + 3) * n]);
                    const __m256 v1_r3 = _mm256_set1_ps((r + 3 < len1) ? vp1[(size_t)(r + 3) * n] : 0.0f);
                    const __m256 v2_r3 = _mm256_set1_ps((r + 3 < len2) ? vp2[(size_t)(r + 3) * n] : 0.0f);
                    const float *cptr3 = Cpack + (size_t)(r + 3 + p + 0) * kc + j;
                    const __m256 c3_0 = _mm256_loadu_ps(cptr3 + 0);
                    const __m256 c3_1 = _mm256_loadu_ps(cptr3 + 8);

                    /* Prefetch ahead for next outer iteration */
                    if (r + 4 + 3 < r_end)
                    {
                        _mm_prefetch((const char *)(Cpack + (size_t)(r + 8 + p + 0) * kc + j), _MM_HINT_T0);
                    }

                    acc00 = _mm256_fmadd_ps(v0_r3, c3_0, acc00);
                    acc01 = _mm256_fmadd_ps(v0_r3, c3_1, acc01);

                    const __m256 v3_r3 = _mm256_set1_ps((r + 3 < len3) ? vp3[(size_t)(r + 3) * n] : 0.0f);
                    const __m256 v4_r3 = _mm256_set1_ps((r + 3 < len4) ? vp4[(size_t)(r + 3) * n] : 0.0f);
                    const __m256 v5_r3 = _mm256_set1_ps((r + 3 < len5) ? vp5[(size_t)(r + 3) * n] : 0.0f);

                    acc10 = _mm256_fmadd_ps(v1_r3, c3_0, acc10);
                    acc11 = _mm256_fmadd_ps(v1_r3, c3_1, acc11);
                    acc20 = _mm256_fmadd_ps(v2_r3, c3_0, acc20);
                    acc21 = _mm256_fmadd_ps(v2_r3, c3_1, acc21);
                    acc30 = _mm256_fmadd_ps(v3_r3, c3_0, acc30);
                    acc31 = _mm256_fmadd_ps(v3_r3, c3_1, acc31);
                    acc40 = _mm256_fmadd_ps(v4_r3, c3_0, acc40);
                    acc41 = _mm256_fmadd_ps(v4_r3, c3_1, acc41);
                    acc50 = _mm256_fmadd_ps(v5_r3, c3_0, acc50);
                    acc51 = _mm256_fmadd_ps(v5_r3, c3_1, acc51);
                }

                /* Remainder loop (1-3 iterations) */
                for (; r < r_end; ++r)
                {
                    const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                    const __m256 v1 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                    const __m256 v2 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                    const __m256 v3 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);
                    const __m256 v4 = _mm256_set1_ps((r < len4) ? vp4[(size_t)r * n] : 0.0f);
                    const __m256 v5 = _mm256_set1_ps((r < len5) ? vp5[(size_t)r * n] : 0.0f);

                    const float *cptr = Cpack + (size_t)(r + p + 0) * kc + j;
                    const __m256 c0 = _mm256_loadu_ps(cptr + 0);
                    const __m256 c1 = _mm256_loadu_ps(cptr + 8);

                    acc00 = _mm256_fmadd_ps(v0, c0, acc00);
                    acc01 = _mm256_fmadd_ps(v0, c1, acc01);
                    acc10 = _mm256_fmadd_ps(v1, c0, acc10);
                    acc11 = _mm256_fmadd_ps(v1, c1, acc11);
                    acc20 = _mm256_fmadd_ps(v2, c0, acc20);
                    acc21 = _mm256_fmadd_ps(v2, c1, acc21);
                    acc30 = _mm256_fmadd_ps(v3, c0, acc30);
                    acc31 = _mm256_fmadd_ps(v3, c1, acc31);
                    acc40 = _mm256_fmadd_ps(v4, c0, acc40);
                    acc41 = _mm256_fmadd_ps(v4, c1, acc41);
                    acc50 = _mm256_fmadd_ps(v5, c0, acc50);
                    acc51 = _mm256_fmadd_ps(v5, c1, acc51);
                }

                /* Store 12 accumulators to Y */
                if (use_streaming)
                {
                    _mm256_stream_ps(Y + (size_t)(p + 0) * kc + j + 0, acc00);
                    _mm256_stream_ps(Y + (size_t)(p + 0) * kc + j + 8, acc01);
                    _mm256_stream_ps(Y + (size_t)(p + 1) * kc + j + 0, acc10);
                    _mm256_stream_ps(Y + (size_t)(p + 1) * kc + j + 8, acc11);
                    _mm256_stream_ps(Y + (size_t)(p + 2) * kc + j + 0, acc20);
                    _mm256_stream_ps(Y + (size_t)(p + 2) * kc + j + 8, acc21);
                    _mm256_stream_ps(Y + (size_t)(p + 3) * kc + j + 0, acc30);
                    _mm256_stream_ps(Y + (size_t)(p + 3) * kc + j + 8, acc31);
                    _mm256_stream_ps(Y + (size_t)(p + 4) * kc + j + 0, acc40);
                    _mm256_stream_ps(Y + (size_t)(p + 4) * kc + j + 8, acc41);
                    _mm256_stream_ps(Y + (size_t)(p + 5) * kc + j + 0, acc50);
                    _mm256_stream_ps(Y + (size_t)(p + 5) * kc + j + 8, acc51);
                }
                else
                {
                    _mm256_storeu_ps(Y + (size_t)(p + 0) * kc + j + 0, acc00);
                    _mm256_storeu_ps(Y + (size_t)(p + 0) * kc + j + 8, acc01);
                    _mm256_storeu_ps(Y + (size_t)(p + 1) * kc + j + 0, acc10);
                    _mm256_storeu_ps(Y + (size_t)(p + 1) * kc + j + 8, acc11);
                    _mm256_storeu_ps(Y + (size_t)(p + 2) * kc + j + 0, acc20);
                    _mm256_storeu_ps(Y + (size_t)(p + 2) * kc + j + 8, acc21);
                    _mm256_storeu_ps(Y + (size_t)(p + 3) * kc + j + 0, acc30);
                    _mm256_storeu_ps(Y + (size_t)(p + 3) * kc + j + 8, acc31);
                    _mm256_storeu_ps(Y + (size_t)(p + 4) * kc + j + 0, acc40);
                    _mm256_storeu_ps(Y + (size_t)(p + 4) * kc + j + 8, acc41);
                    _mm256_storeu_ps(Y + (size_t)(p + 5) * kc + j + 0, acc50);
                    _mm256_storeu_ps(Y + (size_t)(p + 5) * kc + j + 8, acc51);
                }
            }

            /* Handle 8-wide remainder (j+8 <= kc) */
            for (; j + 7 < kc; j += 8)
            {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                __m256 acc4 = _mm256_setzero_ps();
                __m256 acc5 = _mm256_setzero_ps();

                for (uint16_t r = 0; r < len0; ++r)
                {
                    const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                    const __m256 v1 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                    const __m256 v2 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                    const __m256 v3 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);
                    const __m256 v4 = _mm256_set1_ps((r < len4) ? vp4[(size_t)r * n] : 0.0f);
                    const __m256 v5 = _mm256_set1_ps((r < len5) ? vp5[(size_t)r * n] : 0.0f);

                    const float *cptr = Cpack + (size_t)(r + p + 0) * kc + j;
                    const __m256 c = _mm256_loadu_ps(cptr);

                    acc0 = _mm256_fmadd_ps(v0, c, acc0);
                    acc1 = _mm256_fmadd_ps(v1, c, acc1);
                    acc2 = _mm256_fmadd_ps(v2, c, acc2);
                    acc3 = _mm256_fmadd_ps(v3, c, acc3);
                    acc4 = _mm256_fmadd_ps(v4, c, acc4);
                    acc5 = _mm256_fmadd_ps(v5, c, acc5);
                }

                if (use_streaming)
                {
                    _mm256_stream_ps(Y + (size_t)(p + 0) * kc + j, acc0);
                    _mm256_stream_ps(Y + (size_t)(p + 1) * kc + j, acc1);
                    _mm256_stream_ps(Y + (size_t)(p + 2) * kc + j, acc2);
                    _mm256_stream_ps(Y + (size_t)(p + 3) * kc + j, acc3);
                    _mm256_stream_ps(Y + (size_t)(p + 4) * kc + j, acc4);
                    _mm256_stream_ps(Y + (size_t)(p + 5) * kc + j, acc5);
                }
                else
                {
                    _mm256_storeu_ps(Y + (size_t)(p + 0) * kc + j, acc0);
                    _mm256_storeu_ps(Y + (size_t)(p + 1) * kc + j, acc1);
                    _mm256_storeu_ps(Y + (size_t)(p + 2) * kc + j, acc2);
                    _mm256_storeu_ps(Y + (size_t)(p + 3) * kc + j, acc3);
                    _mm256_storeu_ps(Y + (size_t)(p + 4) * kc + j, acc4);
                    _mm256_storeu_ps(Y + (size_t)(p + 5) * kc + j, acc5);
                }
            }

            /* Scalar remainder (j < kc) */
            for (; j < kc; ++j)
            {
                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f;
                for (uint16_t r = 0; r < len0; ++r)
                {
                    const float c_val = Cpack[(size_t)(r + p + 0) * kc + j];
                    sum0 += vp0[(size_t)r * n] * c_val;
                    if (r < len1)
                        sum1 += vp1[(size_t)r * n] * c_val;
                    if (r < len2)
                        sum2 += vp2[(size_t)r * n] * c_val;
                    if (r < len3)
                        sum3 += vp3[(size_t)r * n] * c_val;
                    if (r < len4)
                        sum4 += vp4[(size_t)r * n] * c_val;
                    if (r < len5)
                        sum5 += vp5[(size_t)r * n] * c_val;
                }
                Y[(size_t)(p + 0) * kc + j] = sum0;
                Y[(size_t)(p + 1) * kc + j] = sum1;
                Y[(size_t)(p + 2) * kc + j] = sum2;
                Y[(size_t)(p + 3) * kc + j] = sum3;
                Y[(size_t)(p + 4) * kc + j] = sum4;
                Y[(size_t)(p + 5) * kc + j] = sum5;
            }

            // Fence after streaming stores for this 6-row block
            if (use_streaming)
            {
                _mm_sfence();
            }
        }

        /* Handle remainder rows (1-5 rows) - use original dual-accumulator approach */
        for (; p < ib; ++p)
        {
            const float *vp = A + (size_t)(k + p) * n + (k + p);
            const uint16_t len = (uint16_t)(m - (k + p));

            uint16_t j = 0;
            for (; j + 15 < kc; j += 16)
            {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (uint16_t r = 0; r < len; ++r)
                {
                    const __m256 vv = _mm256_set1_ps(vp[(size_t)r * n]);
                    const float *cptr = Cpack + (size_t)(r + p) * kc + j;
                    acc0 = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr + 0), acc0);
                    acc1 = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr + 8), acc1);
                }
                if (use_streaming)
                {
                    _mm256_stream_ps(Y + (size_t)p * kc + j + 0, acc0);
                    _mm256_stream_ps(Y + (size_t)p * kc + j + 8, acc1);
                }
                else
                {
                    _mm256_storeu_ps(Y + (size_t)p * kc + j + 0, acc0);
                    _mm256_storeu_ps(Y + (size_t)p * kc + j + 8, acc1);
                }
            }
            for (; j + 7 < kc; j += 8)
            {
                __m256 acc = _mm256_setzero_ps();
                for (uint16_t r = 0; r < len; ++r)
                {
                    const __m256 vv = _mm256_set1_ps(vp[(size_t)r * n]);
                    const float *cptr = Cpack + (size_t)(r + p) * kc + j;
                    acc = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr), acc);
                }
                if (use_streaming)
                {
                    _mm256_stream_ps(Y + (size_t)p * kc + j, acc);
                }
                else
                {
                    _mm256_storeu_ps(Y + (size_t)p * kc + j, acc);
                }
            }
            for (; j < kc; ++j)
            {
                float sum = 0.0f;
                for (uint16_t r = 0; r < len; ++r)
                    sum += vp[(size_t)r * n] * Cpack[(size_t)(r + p) * kc + j];
                Y[(size_t)p * kc + j] = sum;
            }
        }

        // Final fence if using streaming stores
        if (use_streaming)
        {
            _mm_sfence();
        }
    }

    /* ===========================================================================================
     * KERNEL 2: Z = T * Y  (ib × kc)
     * ===========================================================================================
     * Strategy:
     *  - Process 4 rows of T and Y simultaneously
     *  - For each T row, compute dot products with kc columns
     *  - Use 8 accumulators (4 rows × 2 AVX vectors for 16 cols)
     *  - Software pipeline: interleave T broadcasts with Y loads and FMAs
     *
     * Memory access pattern:
     *  - T: broadcast elements (upper triangular, ib × ib)
     *  - Y: streaming loads (ib × kc, row-major)
     *  - Z: streaming stores (ib × kc, row-major)
     */
    static void qrw_compute_Z_avx_opt(const float *__restrict T, uint16_t ib,
                                      const float *__restrict Y, uint16_t kc,
                                      float *__restrict Z)
    {
        // Check if buffers are large enough to benefit from non-temporal stores
        const size_t buffer_size = (size_t)ib * kc * sizeof(float);
        const int use_streaming = (buffer_size > L3_CACHE_SIZE / 4);

        /* Process ib rows in blocks of 4 for register blocking */
        uint16_t i = 0;
        for (; i + 3 < ib; i += 4)
        {
            uint16_t j = 0;
            for (; j + 15 < kc; j += 16)
            {
                /* 8 accumulators: 4 T-rows × 2 AVX vectors (16 cols total) */
                __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
                __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
                __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
                __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();

                /* Software pipeline: unroll by 2 for better ILP */
                uint16_t p = 0;
                for (; p + 1 < ib; p += 2)
                {
                    /* Iteration p */
                    const __m256 t0_p = _mm256_set1_ps(T[(size_t)(i + 0) * ib + p]);
                    const __m256 t1_p = _mm256_set1_ps(T[(size_t)(i + 1) * ib + p]);
                    const __m256 t2_p = _mm256_set1_ps(T[(size_t)(i + 2) * ib + p]);
                    const __m256 t3_p = _mm256_set1_ps(T[(size_t)(i + 3) * ib + p]);

                    const float *yp = Y + (size_t)p * kc + j;
                    const __m256 y_p0 = _mm256_loadu_ps(yp + 0);
                    const __m256 y_p1 = _mm256_loadu_ps(yp + 8);

                    acc00 = _mm256_fmadd_ps(t0_p, y_p0, acc00);
                    acc01 = _mm256_fmadd_ps(t0_p, y_p1, acc01);
                    acc10 = _mm256_fmadd_ps(t1_p, y_p0, acc10);
                    acc11 = _mm256_fmadd_ps(t1_p, y_p1, acc11);
                    acc20 = _mm256_fmadd_ps(t2_p, y_p0, acc20);
                    acc21 = _mm256_fmadd_ps(t2_p, y_p1, acc21);
                    acc30 = _mm256_fmadd_ps(t3_p, y_p0, acc30);
                    acc31 = _mm256_fmadd_ps(t3_p, y_p1, acc31);

                    /* Iteration p+1 */
                    const __m256 t0_p1 = _mm256_set1_ps(T[(size_t)(i + 0) * ib + (p + 1)]);
                    const __m256 t1_p1 = _mm256_set1_ps(T[(size_t)(i + 1) * ib + (p + 1)]);
                    const __m256 t2_p1 = _mm256_set1_ps(T[(size_t)(i + 2) * ib + (p + 1)]);
                    const __m256 t3_p1 = _mm256_set1_ps(T[(size_t)(i + 3) * ib + (p + 1)]);

                    const float *yp1 = Y + (size_t)(p + 1) * kc + j;
                    const __m256 y_p1_0 = _mm256_loadu_ps(yp1 + 0);
                    const __m256 y_p1_1 = _mm256_loadu_ps(yp1 + 8);

                    acc00 = _mm256_fmadd_ps(t0_p1, y_p1_0, acc00);
                    acc01 = _mm256_fmadd_ps(t0_p1, y_p1_1, acc01);
                    acc10 = _mm256_fmadd_ps(t1_p1, y_p1_0, acc10);
                    acc11 = _mm256_fmadd_ps(t1_p1, y_p1_1, acc11);
                    acc20 = _mm256_fmadd_ps(t2_p1, y_p1_0, acc20);
                    acc21 = _mm256_fmadd_ps(t2_p1, y_p1_1, acc21);
                    acc30 = _mm256_fmadd_ps(t3_p1, y_p1_0, acc30);
                    acc31 = _mm256_fmadd_ps(t3_p1, y_p1_1, acc31);
                }

                /* Remainder p iterations */
                for (; p < ib; ++p)
                {
                    const __m256 t0 = _mm256_set1_ps(T[(size_t)(i + 0) * ib + p]);
                    const __m256 t1 = _mm256_set1_ps(T[(size_t)(i + 1) * ib + p]);
                    const __m256 t2 = _mm256_set1_ps(T[(size_t)(i + 2) * ib + p]);
                    const __m256 t3 = _mm256_set1_ps(T[(size_t)(i + 3) * ib + p]);

                    const float *yp = Y + (size_t)p * kc + j;
                    const __m256 y0 = _mm256_loadu_ps(yp + 0);
                    const __m256 y1 = _mm256_loadu_ps(yp + 8);

                    acc00 = _mm256_fmadd_ps(t0, y0, acc00);
                    acc01 = _mm256_fmadd_ps(t0, y1, acc01);
                    acc10 = _mm256_fmadd_ps(t1, y0, acc10);
                    acc11 = _mm256_fmadd_ps(t1, y1, acc11);
                    acc20 = _mm256_fmadd_ps(t2, y0, acc20);
                    acc21 = _mm256_fmadd_ps(t2, y1, acc21);
                    acc30 = _mm256_fmadd_ps(t3, y0, acc30);
                    acc31 = _mm256_fmadd_ps(t3, y1, acc31);
                }

                /* Store 8 accumulators */
                if (use_streaming)
                {
                    _mm256_stream_ps(Z + (size_t)(i + 0) * kc + j + 0, acc00);
                    _mm256_stream_ps(Z + (size_t)(i + 0) * kc + j + 8, acc01);
                    _mm256_stream_ps(Z + (size_t)(i + 1) * kc + j + 0, acc10);
                    _mm256_stream_ps(Z + (size_t)(i + 1) * kc + j + 8, acc11);
                    _mm256_stream_ps(Z + (size_t)(i + 2) * kc + j + 0, acc20);
                    _mm256_stream_ps(Z + (size_t)(i + 2) * kc + j + 8, acc21);
                    _mm256_stream_ps(Z + (size_t)(i + 3) * kc + j + 0, acc30);
                    _mm256_stream_ps(Z + (size_t)(i + 3) * kc + j + 8, acc31);
                }
                else
                {
                    _mm256_storeu_ps(Z + (size_t)(i + 0) * kc + j + 0, acc00);
                    _mm256_storeu_ps(Z + (size_t)(i + 0) * kc + j + 8, acc01);
                    _mm256_storeu_ps(Z + (size_t)(i + 1) * kc + j + 0, acc10);
                    _mm256_storeu_ps(Z + (size_t)(i + 1) * kc + j + 8, acc11);
                    _mm256_storeu_ps(Z + (size_t)(i + 2) * kc + j + 0, acc20);
                    _mm256_storeu_ps(Z + (size_t)(i + 2) * kc + j + 8, acc21);
                    _mm256_storeu_ps(Z + (size_t)(i + 3) * kc + j + 0, acc30);
                    _mm256_storeu_ps(Z + (size_t)(i + 3) * kc + j + 8, acc31);
                }
            }

            /* Handle 8-wide remainder */
            for (; j + 7 < kc; j += 8)
            {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (uint16_t p = 0; p < ib; ++p)
                {
                    const __m256 t0 = _mm256_set1_ps(T[(size_t)(i + 0) * ib + p]);
                    const __m256 t1 = _mm256_set1_ps(T[(size_t)(i + 1) * ib + p]);
                    const __m256 t2 = _mm256_set1_ps(T[(size_t)(i + 2) * ib + p]);
                    const __m256 t3 = _mm256_set1_ps(T[(size_t)(i + 3) * ib + p]);

                    const float *yp = Y + (size_t)p * kc + j;
                    const __m256 y = _mm256_loadu_ps(yp);

                    acc0 = _mm256_fmadd_ps(t0, y, acc0);
                    acc1 = _mm256_fmadd_ps(t1, y, acc1);
                    acc2 = _mm256_fmadd_ps(t2, y, acc2);
                    acc3 = _mm256_fmadd_ps(t3, y, acc3);
                }

                if (use_streaming)
                {
                    _mm256_stream_ps(Z + (size_t)(i + 0) * kc + j, acc0);
                    _mm256_stream_ps(Z + (size_t)(i + 1) * kc + j, acc1);
                    _mm256_stream_ps(Z + (size_t)(i + 2) * kc + j, acc2);
                    _mm256_stream_ps(Z + (size_t)(i + 3) * kc + j, acc3);
                }
                else
                {
                    _mm256_storeu_ps(Z + (size_t)(i + 0) * kc + j, acc0);
                    _mm256_storeu_ps(Z + (size_t)(i + 1) * kc + j, acc1);
                    _mm256_storeu_ps(Z + (size_t)(i + 2) * kc + j, acc2);
                    _mm256_storeu_ps(Z + (size_t)(i + 3) * kc + j, acc3);
                }
            }

            /* Scalar remainder */
            for (; j < kc; ++j)
            {
                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                for (uint16_t p = 0; p < ib; ++p)
                {
                    const float y_val = Y[(size_t)p * kc + j];
                    sum0 += T[(size_t)(i + 0) * ib + p] * y_val;
                    sum1 += T[(size_t)(i + 1) * ib + p] * y_val;
                    sum2 += T[(size_t)(i + 2) * ib + p] * y_val;
                    sum3 += T[(size_t)(i + 3) * ib + p] * y_val;
                }
                Z[(size_t)(i + 0) * kc + j] = sum0;
                Z[(size_t)(i + 1) * kc + j] = sum1;
                Z[(size_t)(i + 2) * kc + j] = sum2;
                Z[(size_t)(i + 3) * kc + j] = sum3;
            }

            // Fence after streaming stores for this 4-row block
            if (use_streaming)
            {
                _mm_sfence();
            }
        }

        /* Handle remainder rows (1-3 rows) - use original dual-accumulator approach */
        for (; i < ib; ++i)
        {
            uint16_t j = 0;
            for (; j + 15 < kc; j += 16)
            {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (uint16_t p = 0; p < ib; ++p)
                {
                    const __m256 t = _mm256_set1_ps(T[(size_t)i * ib + p]);
                    const float *y = Y + (size_t)p * kc + j;
                    acc0 = _mm256_fmadd_ps(t, _mm256_loadu_ps(y + 0), acc0);
                    acc1 = _mm256_fmadd_ps(t, _mm256_loadu_ps(y + 8), acc1);
                }
                if (use_streaming)
                {
                    _mm256_stream_ps(Z + (size_t)i * kc + j + 0, acc0);
                    _mm256_stream_ps(Z + (size_t)i * kc + j + 8, acc1);
                }
                else
                {
                    _mm256_storeu_ps(Z + (size_t)i * kc + j + 0, acc0);
                    _mm256_storeu_ps(Z + (size_t)i * kc + j + 8, acc1);
                }
            }
            for (; j + 7 < kc; j += 8)
            {
                __m256 acc = _mm256_setzero_ps();
                for (uint16_t p = 0; p < ib; ++p)
                {
                    const __m256 t = _mm256_set1_ps(T[(size_t)i * ib + p]);
                    const float *y = Y + (size_t)p * kc + j;
                    acc = _mm256_fmadd_ps(t, _mm256_loadu_ps(y), acc);
                }
                if (use_streaming)
                {
                    _mm256_stream_ps(Z + (size_t)i * kc + j, acc);
                }
                else
                {
                    _mm256_storeu_ps(Z + (size_t)i * kc + j, acc);
                }
            }
            for (; j < kc; ++j)
            {
                float sum = 0.0f;
                for (uint16_t p = 0; p < ib; ++p)
                    sum += T[(size_t)i * ib + p] * Y[(size_t)p * kc + j];
                Z[(size_t)i * kc + j] = sum;
            }
        }

        // Final fence if using streaming stores
        if (use_streaming)
        {
            _mm_sfence();
        }
    }

    /* ===========================================================================================
     * KERNEL 3: Cpack = Cpack − V * Z
     * ===========================================================================================
     * Strategy:
     *  - Process 4 rows of V and Z simultaneously
     *  - For each V row, subtract V*Z from corresponding C rows
     *  - Use 8 accumulators (4 V-rows × 2 AVX vectors for 16 cols)
     *  - Software pipeline: interleave V broadcasts, Z loads, C loads, FMAs, and stores
     *  - Critical: This is often the hottest kernel in QR, optimize aggressively
     *
     * Memory access pattern:
     *  - V: broadcast single elements (V[r*n]) - high reuse
     *  - Z: streaming loads (ib × kc, row-major)
     *  - C: read-modify-write (m_sub × kc, row-major, packed)
     */
    static void qrw_apply_VZ_avx_opt(float *__restrict Cpack, uint16_t m_sub, uint16_t kc,
                                     const float *__restrict A, uint16_t m, uint16_t n,
                                     uint16_t k, uint16_t ib,
                                     const float *__restrict Z)
    {
        /* Process ib reflectors in blocks of 4 for better ILP */
        uint16_t p = 0;
        for (; p + 3 < ib; p += 4)
        {
            const float *vp0 = A + (size_t)(k + p + 0) * n + (k + p + 0);
            const float *vp1 = A + (size_t)(k + p + 1) * n + (k + p + 1);
            const float *vp2 = A + (size_t)(k + p + 2) * n + (k + p + 2);
            const float *vp3 = A + (size_t)(k + p + 3) * n + (k + p + 3);

            const uint16_t len0 = (uint16_t)(m - (k + p + 0));
            const uint16_t len1 = (uint16_t)(m - (k + p + 1));
            const uint16_t len2 = (uint16_t)(m - (k + p + 2));
            const uint16_t len3 = (uint16_t)(m - (k + p + 3));

            /* Process columns in 16-wide chunks */
            uint16_t j = 0;
            for (; j + 15 < kc; j += 16)
            {
                /* Load Z values for this column block (4 rows × 16 cols) */
                const float *zp0 = Z + (size_t)(p + 0) * kc + j;
                const float *zp1 = Z + (size_t)(p + 1) * kc + j;
                const float *zp2 = Z + (size_t)(p + 2) * kc + j;
                const float *zp3 = Z + (size_t)(p + 3) * kc + j;

                const __m256 z0_0 = _mm256_loadu_ps(zp0 + 0);
                const __m256 z0_1 = _mm256_loadu_ps(zp0 + 8);
                const __m256 z1_0 = _mm256_loadu_ps(zp1 + 0);
                const __m256 z1_1 = _mm256_loadu_ps(zp1 + 8);
                const __m256 z2_0 = _mm256_loadu_ps(zp2 + 0);
                const __m256 z2_1 = _mm256_loadu_ps(zp2 + 8);
                const __m256 z3_0 = _mm256_loadu_ps(zp3 + 0);
                const __m256 z3_1 = _mm256_loadu_ps(zp3 + 8);

                /* Process rows in blocks of 4 for better cache reuse and ILP */
                uint16_t r = 0;
                uint16_t r_end = len0;

                /* Unroll by 4 rows for software pipelining */
                for (; r + 3 < r_end; r += 4)
                {
                    /* Row r+0: compute V*Z and subtract from C */
                    {
                        const __m256 v0_r0 = _mm256_set1_ps(vp0[(size_t)(r + 0) * n]);
                        const __m256 v1_r0 = _mm256_set1_ps((r + 0 < len1) ? vp1[(size_t)(r + 0) * n] : 0.0f);
                        const __m256 v2_r0 = _mm256_set1_ps((r + 0 < len2) ? vp2[(size_t)(r + 0) * n] : 0.0f);
                        const __m256 v3_r0 = _mm256_set1_ps((r + 0 < len3) ? vp3[(size_t)(r + 0) * n] : 0.0f);

                        float *cptr0 = Cpack + (size_t)(r + 0 + p + 0) * kc + j;
                        __m256 c0_0 = _mm256_loadu_ps(cptr0 + 0);
                        __m256 c0_1 = _mm256_loadu_ps(cptr0 + 8);

                        /* Accumulate V*Z from all 4 reflectors */
                        __m256 vz_sum_0 = _mm256_mul_ps(v0_r0, z0_0);
                        __m256 vz_sum_1 = _mm256_mul_ps(v0_r0, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r0, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r0, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r0, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r0, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r0, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r0, z3_1, vz_sum_1);

                        /* Subtract from C */
                        c0_0 = _mm256_sub_ps(c0_0, vz_sum_0);
                        c0_1 = _mm256_sub_ps(c0_1, vz_sum_1);

                        _mm256_storeu_ps(cptr0 + 0, c0_0);
                        _mm256_storeu_ps(cptr0 + 8, c0_1);
                    }

                    /* Row r+1 */
                    {
                        const __m256 v0_r1 = _mm256_set1_ps(vp0[(size_t)(r + 1) * n]);
                        const __m256 v1_r1 = _mm256_set1_ps((r + 1 < len1) ? vp1[(size_t)(r + 1) * n] : 0.0f);
                        const __m256 v2_r1 = _mm256_set1_ps((r + 1 < len2) ? vp2[(size_t)(r + 1) * n] : 0.0f);
                        const __m256 v3_r1 = _mm256_set1_ps((r + 1 < len3) ? vp3[(size_t)(r + 1) * n] : 0.0f);

                        float *cptr1 = Cpack + (size_t)(r + 1 + p + 0) * kc + j;
                        __m256 c1_0 = _mm256_loadu_ps(cptr1 + 0);
                        __m256 c1_1 = _mm256_loadu_ps(cptr1 + 8);

                        __m256 vz_sum_0 = _mm256_mul_ps(v0_r1, z0_0);
                        __m256 vz_sum_1 = _mm256_mul_ps(v0_r1, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r1, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r1, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r1, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r1, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r1, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r1, z3_1, vz_sum_1);

                        c1_0 = _mm256_sub_ps(c1_0, vz_sum_0);
                        c1_1 = _mm256_sub_ps(c1_1, vz_sum_1);

                        _mm256_storeu_ps(cptr1 + 0, c1_0);
                        _mm256_storeu_ps(cptr1 + 8, c1_1);
                    }

                    /* Row r+2 */
                    {
                        const __m256 v0_r2 = _mm256_set1_ps(vp0[(size_t)(r + 2) * n]);
                        const __m256 v1_r2 = _mm256_set1_ps((r + 2 < len1) ? vp1[(size_t)(r + 2) * n] : 0.0f);
                        const __m256 v2_r2 = _mm256_set1_ps((r + 2 < len2) ? vp2[(size_t)(r + 2) * n] : 0.0f);
                        const __m256 v3_r2 = _mm256_set1_ps((r + 2 < len3) ? vp3[(size_t)(r + 2) * n] : 0.0f);

                        float *cptr2 = Cpack + (size_t)(r + 2 + p + 0) * kc + j;
                        __m256 c2_0 = _mm256_loadu_ps(cptr2 + 0);
                        __m256 c2_1 = _mm256_loadu_ps(cptr2 + 8);

                        __m256 vz_sum_0 = _mm256_mul_ps(v0_r2, z0_0);
                        __m256 vz_sum_1 = _mm256_mul_ps(v0_r2, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r2, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r2, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r2, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r2, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r2, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r2, z3_1, vz_sum_1);

                        c2_0 = _mm256_sub_ps(c2_0, vz_sum_0);
                        c2_1 = _mm256_sub_ps(c2_1, vz_sum_1);

                        _mm256_storeu_ps(cptr2 + 0, c2_0);
                        _mm256_storeu_ps(cptr2 + 8, c2_1);
                    }

                    /* Row r+3 */
                    {
                        const __m256 v0_r3 = _mm256_set1_ps(vp0[(size_t)(r + 3) * n]);
                        const __m256 v1_r3 = _mm256_set1_ps((r + 3 < len1) ? vp1[(size_t)(r + 3) * n] : 0.0f);
                        const __m256 v2_r3 = _mm256_set1_ps((r + 3 < len2) ? vp2[(size_t)(r + 3) * n] : 0.0f);
                        const __m256 v3_r3 = _mm256_set1_ps((r + 3 < len3) ? vp3[(size_t)(r + 3) * n] : 0.0f);

                        float *cptr3 = Cpack + (size_t)(r + 3 + p + 0) * kc + j;
                        __m256 c3_0 = _mm256_loadu_ps(cptr3 + 0);
                        __m256 c3_1 = _mm256_loadu_ps(cptr3 + 8);

                        /* Prefetch ahead */
                        if (r + 4 + 3 < r_end)
                        {
                            _mm_prefetch((const char *)(Cpack + (size_t)(r + 8 + p + 0) * kc + j), _MM_HINT_T0);
                        }

                        __m256 vz_sum_0 = _mm256_mul_ps(v0_r3, z0_0);
                        __m256 vz_sum_1 = _mm256_mul_ps(v0_r3, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r3, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r3, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r3, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r3, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r3, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r3, z3_1, vz_sum_1);

                        c3_0 = _mm256_sub_ps(c3_0, vz_sum_0);
                        c3_1 = _mm256_sub_ps(c3_1, vz_sum_1);

                        _mm256_storeu_ps(cptr3 + 0, c3_0);
                        _mm256_storeu_ps(cptr3 + 8, c3_1);
                    }
                }

                /* Remainder rows (1-3 rows) */
                for (; r < r_end; ++r)
                {
                    const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                    const __m256 v1 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                    const __m256 v2 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                    const __m256 v3 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);

                    float *cptr = Cpack + (size_t)(r + p + 0) * kc + j;
                    __m256 c0 = _mm256_loadu_ps(cptr + 0);
                    __m256 c1 = _mm256_loadu_ps(cptr + 8);

                    __m256 vz_sum_0 = _mm256_mul_ps(v0, z0_0);
                    __m256 vz_sum_1 = _mm256_mul_ps(v0, z0_1);
                    vz_sum_0 = _mm256_fmadd_ps(v1, z1_0, vz_sum_0);
                    vz_sum_1 = _mm256_fmadd_ps(v1, z1_1, vz_sum_1);
                    vz_sum_0 = _mm256_fmadd_ps(v2, z2_0, vz_sum_0);
                    vz_sum_1 = _mm256_fmadd_ps(v2, z2_1, vz_sum_1);
                    vz_sum_0 = _mm256_fmadd_ps(v3, z3_0, vz_sum_0);
                    vz_sum_1 = _mm256_fmadd_ps(v3, z3_1, vz_sum_1);

                    c0 = _mm256_sub_ps(c0, vz_sum_0);
                    c1 = _mm256_sub_ps(c1, vz_sum_1);

                    _mm256_storeu_ps(cptr + 0, c0);
                    _mm256_storeu_ps(cptr + 8, c1);
                }
            }

            /* Handle 8-wide column remainder */
            for (; j + 7 < kc; j += 8)
            {
                const __m256 z0 = _mm256_loadu_ps(Z + (size_t)(p + 0) * kc + j);
                const __m256 z1 = _mm256_loadu_ps(Z + (size_t)(p + 1) * kc + j);
                const __m256 z2 = _mm256_loadu_ps(Z + (size_t)(p + 2) * kc + j);
                const __m256 z3 = _mm256_loadu_ps(Z + (size_t)(p + 3) * kc + j);

                for (uint16_t r = 0; r < len0; ++r)
                {
                    const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                    const __m256 v1 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                    const __m256 v2 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                    const __m256 v3 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);

                    float *cptr = Cpack + (size_t)(r + p + 0) * kc + j;
                    __m256 c = _mm256_loadu_ps(cptr);

                    __m256 vz_sum = _mm256_mul_ps(v0, z0);
                    vz_sum = _mm256_fmadd_ps(v1, z1, vz_sum);
                    vz_sum = _mm256_fmadd_ps(v2, z2, vz_sum);
                    vz_sum = _mm256_fmadd_ps(v3, z3, vz_sum);

                    c = _mm256_sub_ps(c, vz_sum);
                    _mm256_storeu_ps(cptr, c);
                }
            }

            /* Scalar remainder columns */
            for (; j < kc; ++j)
            {
                const float z0_val = Z[(size_t)(p + 0) * kc + j];
                const float z1_val = Z[(size_t)(p + 1) * kc + j];
                const float z2_val = Z[(size_t)(p + 2) * kc + j];
                const float z3_val = Z[(size_t)(p + 3) * kc + j];

                for (uint16_t r = 0; r < len0; ++r)
                {
                    const float v0_val = vp0[(size_t)r * n];
                    const float v1_val = (r < len1) ? vp1[(size_t)r * n] : 0.0f;
                    const float v2_val = (r < len2) ? vp2[(size_t)r * n] : 0.0f;
                    const float v3_val = (r < len3) ? vp3[(size_t)r * n] : 0.0f;

                    float vz_sum = v0_val * z0_val + v1_val * z1_val + v2_val * z2_val + v3_val * z3_val;
                    Cpack[(size_t)(r + p + 0) * kc + j] -= vz_sum;
                }
            }
        }

        /* Handle remainder reflectors (1-3) - use original implementation approach */
        for (; p < ib; ++p)
        {
            const float *vp = A + (size_t)(k + p) * n + (k + p);
            const uint16_t len = (uint16_t)(m - (k + p));

            uint16_t j = 0;
            for (; j + 15 < kc; j += 16)
            {
                const __m256 z0 = _mm256_loadu_ps(Z + (size_t)p * kc + j + 0);
                const __m256 z1 = _mm256_loadu_ps(Z + (size_t)p * kc + j + 8);

                for (uint16_t r = 0; r < len; ++r)
                {
                    const __m256 v = _mm256_set1_ps(vp[(size_t)r * n]);
                    float *cptr = Cpack + (size_t)(r + p) * kc + j;

                    __m256 vz0 = _mm256_mul_ps(v, z0);
                    __m256 vz1 = _mm256_mul_ps(v, z1);
                    __m256 c0 = _mm256_loadu_ps(cptr + 0);
                    __m256 c1 = _mm256_loadu_ps(cptr + 8);
                    c0 = _mm256_sub_ps(c0, vz0);
                    c1 = _mm256_sub_ps(c1, vz1);
                    _mm256_storeu_ps(cptr + 0, c0);
                    _mm256_storeu_ps(cptr + 8, c1);
                }
            }
            for (; j + 7 < kc; j += 8)
            {
                const __m256 z = _mm256_loadu_ps(Z + (size_t)p * kc + j);
                for (uint16_t r = 0; r < len; ++r)
                {
                    const __m256 vz = _mm256_mul_ps(_mm256_set1_ps(vp[(size_t)r * n]), z);
                    float *cptr = Cpack + (size_t)(r + p) * kc + j;
                    __m256 c = _mm256_loadu_ps(cptr);
                    c = _mm256_sub_ps(c, vz);
                    _mm256_storeu_ps(cptr, c);
                }
            }
            for (; j < kc; ++j)
            {
                const float z_val = Z[(size_t)p * kc + j];
                for (uint16_t r = 0; r < len; ++r)
                    Cpack[(size_t)(r + p) * kc + j] -= vp[(size_t)r * n] * z_val;
            }
        }
    }

#ifdef __cplusplus
}
#endif

#endif /* LINALG_QR_AVX2_KERNELS_H */