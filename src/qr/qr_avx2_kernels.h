/**
 * @file linalg_qr_avx2_kernels_optimized.h
 * @brief Drop-in optimized replacements with fast-path specialization
 *
 * Optimizations over original:
 *  1. Fast-path for equal-length case (eliminates conditionals in hot loop)
 *  2. Consistent 6-row blocking across Y and VZ kernels
 *  3. Rectangular vs square case specialization
 */

#ifndef QR_AVX2_KERNELS_H
#define QR_AVX2_KERNELS_H

#include <stdint.h>
#include <immintrin.h>
#include <stdbool.h>

#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE (36 * 1024 * 1024)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /* ===========================================================================================
     * Horizontal sum (unchanged)
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
     * KERNEL 1: Y = V^T * Cpack (OPTIMIZED with fast-path)
     * ===========================================================================================
     * Key improvements:
     *  - Fast-path when all 6 V rows have equal length (no conditionals in inner loop)
     *  - Slow-path preserves original correctness for rectangular cases
     *  - Better branch prediction and reduced instruction count in common case
     */
    static void qrw_compute_Y_avx_opt(const float *__restrict A, uint16_t m, uint16_t n,
                                      uint16_t k, uint16_t ib,
                                      const float *__restrict Cpack, uint16_t m_sub,
                                      uint16_t kc, float *__restrict Y)
    {
        const size_t buffer_size = (size_t)ib * kc * sizeof(float);
        const int use_streaming = (buffer_size > L3_CACHE_SIZE / 4);

        uint16_t p = 0;

        /* ==================== FAST PATH: 6-row blocks ==================== */
        for (; p + 5 < ib; p += 6)
        {
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

            /* Check if all lengths are equal (common case for square/tall matrices) */
            const bool equal_lengths = (len0 == len1) && (len1 == len2) &&
                                       (len2 == len3) && (len3 == len4) && (len4 == len5);

            if (equal_lengths)
            {
                /* ========== OPTIMIZED FAST PATH: No conditionals ========== */
                uint16_t j = 0;
                for (; j + 15 < kc; j += 16)
                {
                    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
                    __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
                    __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();
                    __m256 acc40 = _mm256_setzero_ps(), acc41 = _mm256_setzero_ps();
                    __m256 acc50 = _mm256_setzero_ps(), acc51 = _mm256_setzero_ps();

                    if (p + 11 < ib)
                    {
                        _mm_prefetch((const char *)(A + (size_t)(k + p + 6) * n + (k + p + 6)), _MM_HINT_T0);
                        _mm_prefetch((const char *)(A + (size_t)(k + p + 7) * n + (k + p + 7)), _MM_HINT_T0);
                    }

                    uint16_t r = 0;
                    const uint16_t r_end = len0;

                    /* Unroll by 4 - NO CONDITIONALS */
                    for (; r + 3 < r_end; r += 4)
                    {
                        /* Iteration 0 - all loads unconditional */
                        const __m256 v0_r0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1_r0 = _mm256_set1_ps(vp1[(size_t)r * n]);
                        const __m256 v2_r0 = _mm256_set1_ps(vp2[(size_t)r * n]);
                        const __m256 v3_r0 = _mm256_set1_ps(vp3[(size_t)r * n]);
                        const __m256 v4_r0 = _mm256_set1_ps(vp4[(size_t)r * n]);
                        const __m256 v5_r0 = _mm256_set1_ps(vp5[(size_t)r * n]);

                        const float *cptr0 = Cpack + (size_t)(r + p + 0) * kc + j;
                        const __m256 c0_0 = _mm256_loadu_ps(cptr0 + 0);
                        const __m256 c0_1 = _mm256_loadu_ps(cptr0 + 8);

                        acc00 = _mm256_fmadd_ps(v0_r0, c0_0, acc00);
                        acc01 = _mm256_fmadd_ps(v0_r0, c0_1, acc01);
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
                        const __m256 v1_r1 = _mm256_set1_ps(vp1[(size_t)(r + 1) * n]);
                        const __m256 v2_r1 = _mm256_set1_ps(vp2[(size_t)(r + 1) * n]);
                        const __m256 v3_r1 = _mm256_set1_ps(vp3[(size_t)(r + 1) * n]);
                        const __m256 v4_r1 = _mm256_set1_ps(vp4[(size_t)(r + 1) * n]);
                        const __m256 v5_r1 = _mm256_set1_ps(vp5[(size_t)(r + 1) * n]);

                        const float *cptr1 = Cpack + (size_t)(r + 1 + p + 0) * kc + j;
                        const __m256 c1_0 = _mm256_loadu_ps(cptr1 + 0);
                        const __m256 c1_1 = _mm256_loadu_ps(cptr1 + 8);

                        acc00 = _mm256_fmadd_ps(v0_r1, c1_0, acc00);
                        acc01 = _mm256_fmadd_ps(v0_r1, c1_1, acc01);
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
                        const __m256 v1_r2 = _mm256_set1_ps(vp1[(size_t)(r + 2) * n]);
                        const __m256 v2_r2 = _mm256_set1_ps(vp2[(size_t)(r + 2) * n]);
                        const __m256 v3_r2 = _mm256_set1_ps(vp3[(size_t)(r + 2) * n]);
                        const __m256 v4_r2 = _mm256_set1_ps(vp4[(size_t)(r + 2) * n]);
                        const __m256 v5_r2 = _mm256_set1_ps(vp5[(size_t)(r + 2) * n]);

                        const float *cptr2 = Cpack + (size_t)(r + 2 + p + 0) * kc + j;
                        const __m256 c2_0 = _mm256_loadu_ps(cptr2 + 0);
                        const __m256 c2_1 = _mm256_loadu_ps(cptr2 + 8);

                        acc00 = _mm256_fmadd_ps(v0_r2, c2_0, acc00);
                        acc01 = _mm256_fmadd_ps(v0_r2, c2_1, acc01);
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
                        const __m256 v1_r3 = _mm256_set1_ps(vp1[(size_t)(r + 3) * n]);
                        const __m256 v2_r3 = _mm256_set1_ps(vp2[(size_t)(r + 3) * n]);
                        const __m256 v3_r3 = _mm256_set1_ps(vp3[(size_t)(r + 3) * n]);
                        const __m256 v4_r3 = _mm256_set1_ps(vp4[(size_t)(r + 3) * n]);
                        const __m256 v5_r3 = _mm256_set1_ps(vp5[(size_t)(r + 3) * n]);

                        const float *cptr3 = Cpack + (size_t)(r + 3 + p + 0) * kc + j;
                        const __m256 c3_0 = _mm256_loadu_ps(cptr3 + 0);
                        const __m256 c3_1 = _mm256_loadu_ps(cptr3 + 8);

                        if (r + 7 < r_end)
                        {
                            _mm_prefetch((const char *)(Cpack + (size_t)(r + 8 + p + 0) * kc + j), _MM_HINT_T0);
                        }

                        acc00 = _mm256_fmadd_ps(v0_r3, c3_0, acc00);
                        acc01 = _mm256_fmadd_ps(v0_r3, c3_1, acc01);
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

                    /* Remainder rows (no conditionals needed) */
                    for (; r < r_end; ++r)
                    {
                        const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1 = _mm256_set1_ps(vp1[(size_t)r * n]);
                        const __m256 v2 = _mm256_set1_ps(vp2[(size_t)r * n]);
                        const __m256 v3 = _mm256_set1_ps(vp3[(size_t)r * n]);
                        const __m256 v4 = _mm256_set1_ps(vp4[(size_t)r * n]);
                        const __m256 v5 = _mm256_set1_ps(vp5[(size_t)r * n]);

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

                    /* Store results */
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

                /* 8-wide remainder */
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
                        const __m256 v1 = _mm256_set1_ps(vp1[(size_t)r * n]);
                        const __m256 v2 = _mm256_set1_ps(vp2[(size_t)r * n]);
                        const __m256 v3 = _mm256_set1_ps(vp3[(size_t)r * n]);
                        const __m256 v4 = _mm256_set1_ps(vp4[(size_t)r * n]);
                        const __m256 v5 = _mm256_set1_ps(vp5[(size_t)r * n]);
                        const __m256 c = _mm256_loadu_ps(Cpack + (size_t)(r + p + 0) * kc + j);

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

                /* Scalar remainder */
                for (; j < kc; ++j)
                {
                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
                    float sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f;

                    for (uint16_t r = 0; r < len0; ++r)
                    {
                        const float c_val = Cpack[(size_t)(r + p + 0) * kc + j];
                        sum0 += vp0[(size_t)r * n] * c_val;
                        sum1 += vp1[(size_t)r * n] * c_val;
                        sum2 += vp2[(size_t)r * n] * c_val;
                        sum3 += vp3[(size_t)r * n] * c_val;
                        sum4 += vp4[(size_t)r * n] * c_val;
                        sum5 += vp5[(size_t)r * n] * c_val;
                    }

                    Y[(size_t)(p + 0) * kc + j] = sum0;
                    Y[(size_t)(p + 1) * kc + j] = sum1;
                    Y[(size_t)(p + 2) * kc + j] = sum2;
                    Y[(size_t)(p + 3) * kc + j] = sum3;
                    Y[(size_t)(p + 4) * kc + j] = sum4;
                    Y[(size_t)(p + 5) * kc + j] = sum5;
                }
            }
            else
            {
                /* ========== SLOW PATH: Rectangular case with conditionals ========== */
                /* (Use original implementation with bounds checking) */
                uint16_t j = 0;
                for (; j + 15 < kc; j += 16)
                {
                    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
                    __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
                    __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();
                    __m256 acc40 = _mm256_setzero_ps(), acc41 = _mm256_setzero_ps();
                    __m256 acc50 = _mm256_setzero_ps(), acc51 = _mm256_setzero_ps();

                    if (p + 11 < ib)
                    {
                        _mm_prefetch((const char *)(A + (size_t)(k + p + 6) * n + (k + p + 6)), _MM_HINT_T0);
                        _mm_prefetch((const char *)(A + (size_t)(k + p + 7) * n + (k + p + 7)), _MM_HINT_T0);
                    }

                    uint16_t r = 0;
                    const uint16_t r_end = len0;

                    for (; r + 3 < r_end; r += 4)
                    {
                        /* Iteration 0 - WITH conditionals */
                        const __m256 v0_r0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1_r0 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                        const __m256 v2_r0 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                        const __m256 v3_r0 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);
                        const __m256 v4_r0 = _mm256_set1_ps((r < len4) ? vp4[(size_t)r * n] : 0.0f);
                        const __m256 v5_r0 = _mm256_set1_ps((r < len5) ? vp5[(size_t)r * n] : 0.0f);

                        const float *cptr0 = Cpack + (size_t)(r + p + 0) * kc + j;
                        const __m256 c0_0 = _mm256_loadu_ps(cptr0 + 0);
                        const __m256 c0_1 = _mm256_loadu_ps(cptr0 + 8);

                        acc00 = _mm256_fmadd_ps(v0_r0, c0_0, acc00);
                        acc01 = _mm256_fmadd_ps(v0_r0, c0_1, acc01);
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
                        const __m256 v3_r1 = _mm256_set1_ps((r + 1 < len3) ? vp3[(size_t)(r + 1) * n] : 0.0f);
                        const __m256 v4_r1 = _mm256_set1_ps((r + 1 < len4) ? vp4[(size_t)(r + 1) * n] : 0.0f);
                        const __m256 v5_r1 = _mm256_set1_ps((r + 1 < len5) ? vp5[(size_t)(r + 1) * n] : 0.0f);

                        const float *cptr1 = Cpack + (size_t)(r + 1 + p + 0) * kc + j;
                        const __m256 c1_0 = _mm256_loadu_ps(cptr1 + 0);
                        const __m256 c1_1 = _mm256_loadu_ps(cptr1 + 8);

                        acc00 = _mm256_fmadd_ps(v0_r1, c1_0, acc00);
                        acc01 = _mm256_fmadd_ps(v0_r1, c1_1, acc01);
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
                        const __m256 v3_r2 = _mm256_set1_ps((r + 2 < len3) ? vp3[(size_t)(r + 2) * n] : 0.0f);
                        const __m256 v4_r2 = _mm256_set1_ps((r + 2 < len4) ? vp4[(size_t)(r + 2) * n] : 0.0f);
                        const __m256 v5_r2 = _mm256_set1_ps((r + 2 < len5) ? vp5[(size_t)(r + 2) * n] : 0.0f);

                        const float *cptr2 = Cpack + (size_t)(r + 2 + p + 0) * kc + j;
                        const __m256 c2_0 = _mm256_loadu_ps(cptr2 + 0);
                        const __m256 c2_1 = _mm256_loadu_ps(cptr2 + 8);

                        acc00 = _mm256_fmadd_ps(v0_r2, c2_0, acc00);
                        acc01 = _mm256_fmadd_ps(v0_r2, c2_1, acc01);
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
                        const __m256 v3_r3 = _mm256_set1_ps((r + 3 < len3) ? vp3[(size_t)(r + 3) * n] : 0.0f);
                        const __m256 v4_r3 = _mm256_set1_ps((r + 3 < len4) ? vp4[(size_t)(r + 3) * n] : 0.0f);
                        const __m256 v5_r3 = _mm256_set1_ps((r + 3 < len5) ? vp5[(size_t)(r + 3) * n] : 0.0f);

                        const float *cptr3 = Cpack + (size_t)(r + 3 + p + 0) * kc + j;
                        const __m256 c3_0 = _mm256_loadu_ps(cptr3 + 0);
                        const __m256 c3_1 = _mm256_loadu_ps(cptr3 + 8);

                        if (r + 7 < r_end)
                        {
                            _mm_prefetch((const char *)(Cpack + (size_t)(r + 8 + p + 0) * kc + j), _MM_HINT_T0);
                        }

                        acc00 = _mm256_fmadd_ps(v0_r3, c3_0, acc00);
                        acc01 = _mm256_fmadd_ps(v0_r3, c3_1, acc01);
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

                /* 8-wide and scalar remainders (with conditionals) */
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
                        const __m256 c = _mm256_loadu_ps(Cpack + (size_t)(r + p + 0) * kc + j);

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

                for (; j < kc; ++j)
                {
                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
                    float sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f;

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
            } /* end slow path */
        } /* end 6-row blocks */

        /* Remainder reflectors (1-5) */
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
                    const __m256 v = _mm256_set1_ps(vp[(size_t)r * n]);
                    const float *cptr = Cpack + (size_t)(r + p) * kc + j;
                    const __m256 c0 = _mm256_loadu_ps(cptr + 0);
                    const __m256 c1 = _mm256_loadu_ps(cptr + 8);

                    acc0 = _mm256_fmadd_ps(v, c0, acc0);
                    acc1 = _mm256_fmadd_ps(v, c1, acc1);
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
                    const __m256 v = _mm256_set1_ps(vp[(size_t)r * n]);
                    const __m256 c = _mm256_loadu_ps(Cpack + (size_t)(r + p) * kc + j);
                    acc = _mm256_fmadd_ps(v, c, acc);
                }
                if (use_streaming)
                    _mm256_stream_ps(Y + (size_t)p * kc + j, acc);
                else
                    _mm256_storeu_ps(Y + (size_t)p * kc + j, acc);
            }

            for (; j < kc; ++j)
            {
                float sum = 0.0f;
                for (uint16_t r = 0; r < len; ++r)
                    sum += vp[(size_t)r * n] * Cpack[(size_t)(r + p) * kc + j];
                Y[(size_t)p * kc + j] = sum;
            }
        }

        if (use_streaming)
            _mm_sfence();
    }

    /* ===========================================================================================
     * KERNEL 2: Z = T * Y (unchanged - already optimal)
     * ===========================================================================================
     */
    static void qrw_compute_Z_avx_opt(const float *__restrict T, uint16_t ib,
                                      const float *__restrict Y, uint16_t kc,
                                      float *__restrict Z)
    {
        const size_t buffer_size = (size_t)ib * kc * sizeof(float);
        const int use_streaming = (buffer_size > L3_CACHE_SIZE / 4);

        for (uint16_t i = 0; i < ib; ++i)
        {
            uint16_t j = 0;
            for (; j + 15 < kc; j += 16)
            {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                uint16_t k_max = i + 1;
                uint16_t k = 0;
                for (; k + 3 < k_max; k += 4)
                {
                    const float t0 = T[(size_t)i * ib + (k + 0)];
                    const float t1 = T[(size_t)i * ib + (k + 1)];
                    const float t2 = T[(size_t)i * ib + (k + 2)];
                    const float t3 = T[(size_t)i * ib + (k + 3)];

                    const __m256 tv0 = _mm256_set1_ps(t0);
                    const __m256 tv1 = _mm256_set1_ps(t1);
                    const __m256 tv2 = _mm256_set1_ps(t2);
                    const __m256 tv3 = _mm256_set1_ps(t3);

                    const __m256 y00 = _mm256_loadu_ps(Y + (size_t)(k + 0) * kc + j + 0);
                    const __m256 y01 = _mm256_loadu_ps(Y + (size_t)(k + 0) * kc + j + 8);
                    const __m256 y10 = _mm256_loadu_ps(Y + (size_t)(k + 1) * kc + j + 0);
                    const __m256 y11 = _mm256_loadu_ps(Y + (size_t)(k + 1) * kc + j + 8);

                    acc0 = _mm256_fmadd_ps(tv0, y00, acc0);
                    acc1 = _mm256_fmadd_ps(tv0, y01, acc1);
                    acc2 = _mm256_fmadd_ps(tv1, y10, acc2);
                    acc3 = _mm256_fmadd_ps(tv1, y11, acc3);

                    const __m256 y20 = _mm256_loadu_ps(Y + (size_t)(k + 2) * kc + j + 0);
                    const __m256 y21 = _mm256_loadu_ps(Y + (size_t)(k + 2) * kc + j + 8);
                    const __m256 y30 = _mm256_loadu_ps(Y + (size_t)(k + 3) * kc + j + 0);
                    const __m256 y31 = _mm256_loadu_ps(Y + (size_t)(k + 3) * kc + j + 8);

                    acc0 = _mm256_fmadd_ps(tv2, y20, acc0);
                    acc1 = _mm256_fmadd_ps(tv2, y21, acc1);
                    acc2 = _mm256_fmadd_ps(tv3, y30, acc2);
                    acc3 = _mm256_fmadd_ps(tv3, y31, acc3);
                }

                acc0 = _mm256_add_ps(acc0, acc2);
                acc1 = _mm256_add_ps(acc1, acc3);

                for (; k < k_max; ++k)
                {
                    const __m256 tv = _mm256_set1_ps(T[(size_t)i * ib + k]);
                    const __m256 y0 = _mm256_loadu_ps(Y + (size_t)k * kc + j + 0);
                    const __m256 y1 = _mm256_loadu_ps(Y + (size_t)k * kc + j + 8);
                    acc0 = _mm256_fmadd_ps(tv, y0, acc0);
                    acc1 = _mm256_fmadd_ps(tv, y1, acc1);
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
                for (uint16_t k = 0; k <= i; ++k)
                {
                    const __m256 tv = _mm256_set1_ps(T[(size_t)i * ib + k]);
                    const __m256 y = _mm256_loadu_ps(Y + (size_t)k * kc + j);
                    acc = _mm256_fmadd_ps(tv, y, acc);
                }
                if (use_streaming)
                    _mm256_stream_ps(Z + (size_t)i * kc + j, acc);
                else
                    _mm256_storeu_ps(Z + (size_t)i * kc + j, acc);
            }

            for (; j < kc; ++j)
            {
                float sum = 0.0f;
                for (uint16_t k = 0; k <= i; ++k)
                    sum += T[(size_t)i * ib + k] * Y[(size_t)k * kc + j];
                Z[(size_t)i * kc + j] = sum;
            }
        }

        if (use_streaming)
            _mm_sfence();
    }

    /* ===========================================================================================
     * KERNEL 3: C = C - V * Z (OPTIMIZED with 6-row blocking + fast-path)
     * ===========================================================================================
     * Key improvements over original:
     *  - Upgraded from 4-row to 6-row blocking (matches Y kernel)
     *  - Fast-path for equal-length case (no conditionals)
     *  - Better register utilization and throughput
     */
    static void qrw_apply_VZ_avx_opt(const float *__restrict A, uint16_t m, uint16_t n,
                                     uint16_t k, uint16_t ib,
                                     float *__restrict Cpack, uint16_t m_sub, uint16_t kc,
                                     const float *__restrict Z)
    {
        uint16_t p = 0;

        /* ==================== FAST PATH: 6-row blocks ==================== */
        for (; p + 5 < ib; p += 6)
        {
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

            const bool equal_lengths = (len0 == len1) && (len1 == len2) &&
                                       (len2 == len3) && (len3 == len4) && (len4 == len5);

            if (equal_lengths)
            {
                /* ========== OPTIMIZED FAST PATH: No conditionals ========== */
                uint16_t j = 0;
                for (; j + 15 < kc; j += 16)
                {
                    /* Load Z rows once */
                    const __m256 z0_0 = _mm256_loadu_ps(Z + (size_t)(p + 0) * kc + j + 0);
                    const __m256 z0_1 = _mm256_loadu_ps(Z + (size_t)(p + 0) * kc + j + 8);
                    const __m256 z1_0 = _mm256_loadu_ps(Z + (size_t)(p + 1) * kc + j + 0);
                    const __m256 z1_1 = _mm256_loadu_ps(Z + (size_t)(p + 1) * kc + j + 8);
                    const __m256 z2_0 = _mm256_loadu_ps(Z + (size_t)(p + 2) * kc + j + 0);
                    const __m256 z2_1 = _mm256_loadu_ps(Z + (size_t)(p + 2) * kc + j + 8);
                    const __m256 z3_0 = _mm256_loadu_ps(Z + (size_t)(p + 3) * kc + j + 0);
                    const __m256 z3_1 = _mm256_loadu_ps(Z + (size_t)(p + 3) * kc + j + 8);
                    const __m256 z4_0 = _mm256_loadu_ps(Z + (size_t)(p + 4) * kc + j + 0);
                    const __m256 z4_1 = _mm256_loadu_ps(Z + (size_t)(p + 4) * kc + j + 8);
                    const __m256 z5_0 = _mm256_loadu_ps(Z + (size_t)(p + 5) * kc + j + 0);
                    const __m256 z5_1 = _mm256_loadu_ps(Z + (size_t)(p + 5) * kc + j + 8);

                    uint16_t r = 0;
                    const uint16_t r_end = len0;

                    /* Unroll by 4 - NO CONDITIONALS */
                    for (; r + 3 < r_end; r += 4)
                    {
                        /* Iteration 0 - all loads unconditional */
                        const __m256 v0_r0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1_r0 = _mm256_set1_ps(vp1[(size_t)r * n]);
                        const __m256 v2_r0 = _mm256_set1_ps(vp2[(size_t)r * n]);
                        const __m256 v3_r0 = _mm256_set1_ps(vp3[(size_t)r * n]);
                        const __m256 v4_r0 = _mm256_set1_ps(vp4[(size_t)r * n]);
                        const __m256 v5_r0 = _mm256_set1_ps(vp5[(size_t)r * n]);

                        float *cptr0 = Cpack + (size_t)(r + p + 0) * kc + j;
                        __m256 c0_0 = _mm256_loadu_ps(cptr0 + 0);
                        __m256 c0_1 = _mm256_loadu_ps(cptr0 + 8);

                        __m256 vz_sum_0 = _mm256_mul_ps(v0_r0, z0_0);
                        __m256 vz_sum_1 = _mm256_mul_ps(v0_r0, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r0, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r0, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r0, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r0, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r0, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r0, z3_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v4_r0, z4_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v4_r0, z4_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v5_r0, z5_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v5_r0, z5_1, vz_sum_1);

                        c0_0 = _mm256_sub_ps(c0_0, vz_sum_0);
                        c0_1 = _mm256_sub_ps(c0_1, vz_sum_1);

                        _mm256_storeu_ps(cptr0 + 0, c0_0);
                        _mm256_storeu_ps(cptr0 + 8, c0_1);

                        /* Iteration 1 */
                        const __m256 v0_r1 = _mm256_set1_ps(vp0[(size_t)(r + 1) * n]);
                        const __m256 v1_r1 = _mm256_set1_ps(vp1[(size_t)(r + 1) * n]);
                        const __m256 v2_r1 = _mm256_set1_ps(vp2[(size_t)(r + 1) * n]);
                        const __m256 v3_r1 = _mm256_set1_ps(vp3[(size_t)(r + 1) * n]);
                        const __m256 v4_r1 = _mm256_set1_ps(vp4[(size_t)(r + 1) * n]);
                        const __m256 v5_r1 = _mm256_set1_ps(vp5[(size_t)(r + 1) * n]);

                        float *cptr1 = Cpack + (size_t)(r + 1 + p + 0) * kc + j;
                        __m256 c1_0 = _mm256_loadu_ps(cptr1 + 0);
                        __m256 c1_1 = _mm256_loadu_ps(cptr1 + 8);

                        vz_sum_0 = _mm256_mul_ps(v0_r1, z0_0);
                        vz_sum_1 = _mm256_mul_ps(v0_r1, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r1, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r1, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r1, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r1, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r1, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r1, z3_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v4_r1, z4_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v4_r1, z4_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v5_r1, z5_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v5_r1, z5_1, vz_sum_1);

                        c1_0 = _mm256_sub_ps(c1_0, vz_sum_0);
                        c1_1 = _mm256_sub_ps(c1_1, vz_sum_1);

                        _mm256_storeu_ps(cptr1 + 0, c1_0);
                        _mm256_storeu_ps(cptr1 + 8, c1_1);

                        /* Iteration 2 */
                        const __m256 v0_r2 = _mm256_set1_ps(vp0[(size_t)(r + 2) * n]);
                        const __m256 v1_r2 = _mm256_set1_ps(vp1[(size_t)(r + 2) * n]);
                        const __m256 v2_r2 = _mm256_set1_ps(vp2[(size_t)(r + 2) * n]);
                        const __m256 v3_r2 = _mm256_set1_ps(vp3[(size_t)(r + 2) * n]);
                        const __m256 v4_r2 = _mm256_set1_ps(vp4[(size_t)(r + 2) * n]);
                        const __m256 v5_r2 = _mm256_set1_ps(vp5[(size_t)(r + 2) * n]);

                        float *cptr2 = Cpack + (size_t)(r + 2 + p + 0) * kc + j;
                        __m256 c2_0 = _mm256_loadu_ps(cptr2 + 0);
                        __m256 c2_1 = _mm256_loadu_ps(cptr2 + 8);

                        vz_sum_0 = _mm256_mul_ps(v0_r2, z0_0);
                        vz_sum_1 = _mm256_mul_ps(v0_r2, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r2, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r2, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r2, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r2, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r2, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r2, z3_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v4_r2, z4_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v4_r2, z4_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v5_r2, z5_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v5_r2, z5_1, vz_sum_1);

                        c2_0 = _mm256_sub_ps(c2_0, vz_sum_0);
                        c2_1 = _mm256_sub_ps(c2_1, vz_sum_1);

                        _mm256_storeu_ps(cptr2 + 0, c2_0);
                        _mm256_storeu_ps(cptr2 + 8, c2_1);

                        /* Iteration 3 */
                        const __m256 v0_r3 = _mm256_set1_ps(vp0[(size_t)(r + 3) * n]);
                        const __m256 v1_r3 = _mm256_set1_ps(vp1[(size_t)(r + 3) * n]);
                        const __m256 v2_r3 = _mm256_set1_ps(vp2[(size_t)(r + 3) * n]);
                        const __m256 v3_r3 = _mm256_set1_ps(vp3[(size_t)(r + 3) * n]);
                        const __m256 v4_r3 = _mm256_set1_ps(vp4[(size_t)(r + 3) * n]);
                        const __m256 v5_r3 = _mm256_set1_ps(vp5[(size_t)(r + 3) * n]);

                        float *cptr3 = Cpack + (size_t)(r + 3 + p + 0) * kc + j;
                        __m256 c3_0 = _mm256_loadu_ps(cptr3 + 0);
                        __m256 c3_1 = _mm256_loadu_ps(cptr3 + 8);

                        if (r + 7 < r_end)
                        {
                            _mm_prefetch((const char *)(Cpack + (size_t)(r + 8 + p + 0) * kc + j), _MM_HINT_T0);
                        }

                        vz_sum_0 = _mm256_mul_ps(v0_r3, z0_0);
                        vz_sum_1 = _mm256_mul_ps(v0_r3, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r3, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r3, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r3, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r3, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r3, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r3, z3_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v4_r3, z4_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v4_r3, z4_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v5_r3, z5_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v5_r3, z5_1, vz_sum_1);

                        c3_0 = _mm256_sub_ps(c3_0, vz_sum_0);
                        c3_1 = _mm256_sub_ps(c3_1, vz_sum_1);

                        _mm256_storeu_ps(cptr3 + 0, c3_0);
                        _mm256_storeu_ps(cptr3 + 8, c3_1);
                    }

                    /* Remainder rows (no conditionals) */
                    for (; r < r_end; ++r)
                    {
                        const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1 = _mm256_set1_ps(vp1[(size_t)r * n]);
                        const __m256 v2 = _mm256_set1_ps(vp2[(size_t)r * n]);
                        const __m256 v3 = _mm256_set1_ps(vp3[(size_t)r * n]);
                        const __m256 v4 = _mm256_set1_ps(vp4[(size_t)r * n]);
                        const __m256 v5 = _mm256_set1_ps(vp5[(size_t)r * n]);

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
                        vz_sum_0 = _mm256_fmadd_ps(v4, z4_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v4, z4_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v5, z5_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v5, z5_1, vz_sum_1);

                        c0 = _mm256_sub_ps(c0, vz_sum_0);
                        c1 = _mm256_sub_ps(c1, vz_sum_1);

                        _mm256_storeu_ps(cptr + 0, c0);
                        _mm256_storeu_ps(cptr + 8, c1);
                    }
                }

                /* 8-wide remainder */
                for (; j + 7 < kc; j += 8)
                {
                    const __m256 z0 = _mm256_loadu_ps(Z + (size_t)(p + 0) * kc + j);
                    const __m256 z1 = _mm256_loadu_ps(Z + (size_t)(p + 1) * kc + j);
                    const __m256 z2 = _mm256_loadu_ps(Z + (size_t)(p + 2) * kc + j);
                    const __m256 z3 = _mm256_loadu_ps(Z + (size_t)(p + 3) * kc + j);
                    const __m256 z4 = _mm256_loadu_ps(Z + (size_t)(p + 4) * kc + j);
                    const __m256 z5 = _mm256_loadu_ps(Z + (size_t)(p + 5) * kc + j);

                    for (uint16_t r = 0; r < len0; ++r)
                    {
                        const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1 = _mm256_set1_ps(vp1[(size_t)r * n]);
                        const __m256 v2 = _mm256_set1_ps(vp2[(size_t)r * n]);
                        const __m256 v3 = _mm256_set1_ps(vp3[(size_t)r * n]);
                        const __m256 v4 = _mm256_set1_ps(vp4[(size_t)r * n]);
                        const __m256 v5 = _mm256_set1_ps(vp5[(size_t)r * n]);

                        float *cptr = Cpack + (size_t)(r + p + 0) * kc + j;
                        __m256 c = _mm256_loadu_ps(cptr);

                        __m256 vz_sum = _mm256_mul_ps(v0, z0);
                        vz_sum = _mm256_fmadd_ps(v1, z1, vz_sum);
                        vz_sum = _mm256_fmadd_ps(v2, z2, vz_sum);
                        vz_sum = _mm256_fmadd_ps(v3, z3, vz_sum);
                        vz_sum = _mm256_fmadd_ps(v4, z4, vz_sum);
                        vz_sum = _mm256_fmadd_ps(v5, z5, vz_sum);

                        c = _mm256_sub_ps(c, vz_sum);
                        _mm256_storeu_ps(cptr, c);
                    }
                }

                /* Scalar remainder */
                for (; j < kc; ++j)
                {
                    const float z0_val = Z[(size_t)(p + 0) * kc + j];
                    const float z1_val = Z[(size_t)(p + 1) * kc + j];
                    const float z2_val = Z[(size_t)(p + 2) * kc + j];
                    const float z3_val = Z[(size_t)(p + 3) * kc + j];
                    const float z4_val = Z[(size_t)(p + 4) * kc + j];
                    const float z5_val = Z[(size_t)(p + 5) * kc + j];

                    for (uint16_t r = 0; r < len0; ++r)
                    {
                        const float v0_val = vp0[(size_t)r * n];
                        const float v1_val = vp1[(size_t)r * n];
                        const float v2_val = vp2[(size_t)r * n];
                        const float v3_val = vp3[(size_t)r * n];
                        const float v4_val = vp4[(size_t)r * n];
                        const float v5_val = vp5[(size_t)r * n];

                        float vz_sum = v0_val * z0_val + v1_val * z1_val + v2_val * z2_val +
                                       v3_val * z3_val + v4_val * z4_val + v5_val * z5_val;
                        Cpack[(size_t)(r + p + 0) * kc + j] -= vz_sum;
                    }
                }
            }
            else
            {
                /* ========== SLOW PATH: Rectangular case with conditionals ========== */
                uint16_t j = 0;
                for (; j + 15 < kc; j += 16)
                {
                    const __m256 z0_0 = _mm256_loadu_ps(Z + (size_t)(p + 0) * kc + j + 0);
                    const __m256 z0_1 = _mm256_loadu_ps(Z + (size_t)(p + 0) * kc + j + 8);
                    const __m256 z1_0 = _mm256_loadu_ps(Z + (size_t)(p + 1) * kc + j + 0);
                    const __m256 z1_1 = _mm256_loadu_ps(Z + (size_t)(p + 1) * kc + j + 8);
                    const __m256 z2_0 = _mm256_loadu_ps(Z + (size_t)(p + 2) * kc + j + 0);
                    const __m256 z2_1 = _mm256_loadu_ps(Z + (size_t)(p + 2) * kc + j + 8);
                    const __m256 z3_0 = _mm256_loadu_ps(Z + (size_t)(p + 3) * kc + j + 0);
                    const __m256 z3_1 = _mm256_loadu_ps(Z + (size_t)(p + 3) * kc + j + 8);
                    const __m256 z4_0 = _mm256_loadu_ps(Z + (size_t)(p + 4) * kc + j + 0);
                    const __m256 z4_1 = _mm256_loadu_ps(Z + (size_t)(p + 4) * kc + j + 8);
                    const __m256 z5_0 = _mm256_loadu_ps(Z + (size_t)(p + 5) * kc + j + 0);
                    const __m256 z5_1 = _mm256_loadu_ps(Z + (size_t)(p + 5) * kc + j + 8);

                    uint16_t r = 0;
                    const uint16_t r_end = len0;

                    for (; r + 3 < r_end; r += 4)
                    {
                        /* Iteration 0 - WITH conditionals */
                        const __m256 v0_r0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1_r0 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                        const __m256 v2_r0 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                        const __m256 v3_r0 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);
                        const __m256 v4_r0 = _mm256_set1_ps((r < len4) ? vp4[(size_t)r * n] : 0.0f);
                        const __m256 v5_r0 = _mm256_set1_ps((r < len5) ? vp5[(size_t)r * n] : 0.0f);

                        float *cptr0 = Cpack + (size_t)(r + p + 0) * kc + j;
                        __m256 c0_0 = _mm256_loadu_ps(cptr0 + 0);
                        __m256 c0_1 = _mm256_loadu_ps(cptr0 + 8);

                        __m256 vz_sum_0 = _mm256_mul_ps(v0_r0, z0_0);
                        __m256 vz_sum_1 = _mm256_mul_ps(v0_r0, z0_1);
                        vz_sum_0 = _mm256_fmadd_ps(v1_r0, z1_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v1_r0, z1_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v2_r0, z2_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v2_r0, z2_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v3_r0, z3_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v3_r0, z3_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v4_r0, z4_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v4_r0, z4_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v5_r0, z5_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v5_r0, z5_1, vz_sum_1);

                        c0_0 = _mm256_sub_ps(c0_0, vz_sum_0);
                        c0_1 = _mm256_sub_ps(c0_1, vz_sum_1);

                        _mm256_storeu_ps(cptr0 + 0, c0_0);
                        _mm256_storeu_ps(cptr0 + 8, c0_1);

                        /* Iterations 1, 2, 3 follow same pattern... */
                        /* (Similar to iteration 0, with conditionals for boundary checking) */

                        /* Iteration 1 */
                        if (r + 1 < r_end)
                        {
                            const __m256 v0_r1 = _mm256_set1_ps(vp0[(size_t)(r + 1) * n]);
                            const __m256 v1_r1 = _mm256_set1_ps((r + 1 < len1) ? vp1[(size_t)(r + 1) * n] : 0.0f);
                            const __m256 v2_r1 = _mm256_set1_ps((r + 1 < len2) ? vp2[(size_t)(r + 1) * n] : 0.0f);
                            const __m256 v3_r1 = _mm256_set1_ps((r + 1 < len3) ? vp3[(size_t)(r + 1) * n] : 0.0f);
                            const __m256 v4_r1 = _mm256_set1_ps((r + 1 < len4) ? vp4[(size_t)(r + 1) * n] : 0.0f);
                            const __m256 v5_r1 = _mm256_set1_ps((r + 1 < len5) ? vp5[(size_t)(r + 1) * n] : 0.0f);

                            float *cptr1 = Cpack + (size_t)(r + 1 + p + 0) * kc + j;
                            __m256 c1_0 = _mm256_loadu_ps(cptr1 + 0);
                            __m256 c1_1 = _mm256_loadu_ps(cptr1 + 8);

                            vz_sum_0 = _mm256_mul_ps(v0_r1, z0_0);
                            vz_sum_1 = _mm256_mul_ps(v0_r1, z0_1);
                            vz_sum_0 = _mm256_fmadd_ps(v1_r1, z1_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v1_r1, z1_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v2_r1, z2_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v2_r1, z2_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v3_r1, z3_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v3_r1, z3_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v4_r1, z4_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v4_r1, z4_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v5_r1, z5_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v5_r1, z5_1, vz_sum_1);

                            c1_0 = _mm256_sub_ps(c1_0, vz_sum_0);
                            c1_1 = _mm256_sub_ps(c1_1, vz_sum_1);

                            _mm256_storeu_ps(cptr1 + 0, c1_0);
                            _mm256_storeu_ps(cptr1 + 8, c1_1);
                        }

                        /* Iteration 2 */
                        if (r + 2 < r_end)
                        {
                            const __m256 v0_r2 = _mm256_set1_ps(vp0[(size_t)(r + 2) * n]);
                            const __m256 v1_r2 = _mm256_set1_ps((r + 2 < len1) ? vp1[(size_t)(r + 2) * n] : 0.0f);
                            const __m256 v2_r2 = _mm256_set1_ps((r + 2 < len2) ? vp2[(size_t)(r + 2) * n] : 0.0f);
                            const __m256 v3_r2 = _mm256_set1_ps((r + 2 < len3) ? vp3[(size_t)(r + 2) * n] : 0.0f);
                            const __m256 v4_r2 = _mm256_set1_ps((r + 2 < len4) ? vp4[(size_t)(r + 2) * n] : 0.0f);
                            const __m256 v5_r2 = _mm256_set1_ps((r + 2 < len5) ? vp5[(size_t)(r + 2) * n] : 0.0f);

                            float *cptr2 = Cpack + (size_t)(r + 2 + p + 0) * kc + j;
                            __m256 c2_0 = _mm256_loadu_ps(cptr2 + 0);
                            __m256 c2_1 = _mm256_loadu_ps(cptr2 + 8);

                            vz_sum_0 = _mm256_mul_ps(v0_r2, z0_0);
                            vz_sum_1 = _mm256_mul_ps(v0_r2, z0_1);
                            vz_sum_0 = _mm256_fmadd_ps(v1_r2, z1_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v1_r2, z1_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v2_r2, z2_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v2_r2, z2_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v3_r2, z3_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v3_r2, z3_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v4_r2, z4_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v4_r2, z4_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v5_r2, z5_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v5_r2, z5_1, vz_sum_1);

                            c2_0 = _mm256_sub_ps(c2_0, vz_sum_0);
                            c2_1 = _mm256_sub_ps(c2_1, vz_sum_1);

                            _mm256_storeu_ps(cptr2 + 0, c2_0);
                            _mm256_storeu_ps(cptr2 + 8, c2_1);
                        }

                        /* Iteration 3 */
                        if (r + 3 < r_end)
                        {
                            const __m256 v0_r3 = _mm256_set1_ps(vp0[(size_t)(r + 3) * n]);
                            const __m256 v1_r3 = _mm256_set1_ps((r + 3 < len1) ? vp1[(size_t)(r + 3) * n] : 0.0f);
                            const __m256 v2_r3 = _mm256_set1_ps((r + 3 < len2) ? vp2[(size_t)(r + 3) * n] : 0.0f);
                            const __m256 v3_r3 = _mm256_set1_ps((r + 3 < len3) ? vp3[(size_t)(r + 3) * n] : 0.0f);
                            const __m256 v4_r3 = _mm256_set1_ps((r + 3 < len4) ? vp4[(size_t)(r + 3) * n] : 0.0f);
                            const __m256 v5_r3 = _mm256_set1_ps((r + 3 < len5) ? vp5[(size_t)(r + 3) * n] : 0.0f);

                            float *cptr3 = Cpack + (size_t)(r + 3 + p + 0) * kc + j;
                            __m256 c3_0 = _mm256_loadu_ps(cptr3 + 0);
                            __m256 c3_1 = _mm256_loadu_ps(cptr3 + 8);

                            if (r + 7 < r_end)
                            {
                                _mm_prefetch((const char *)(Cpack + (size_t)(r + 8 + p + 0) * kc + j), _MM_HINT_T0);
                            }

                            vz_sum_0 = _mm256_mul_ps(v0_r3, z0_0);
                            vz_sum_1 = _mm256_mul_ps(v0_r3, z0_1);
                            vz_sum_0 = _mm256_fmadd_ps(v1_r3, z1_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v1_r3, z1_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v2_r3, z2_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v2_r3, z2_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v3_r3, z3_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v3_r3, z3_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v4_r3, z4_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v4_r3, z4_1, vz_sum_1);
                            vz_sum_0 = _mm256_fmadd_ps(v5_r3, z5_0, vz_sum_0);
                            vz_sum_1 = _mm256_fmadd_ps(v5_r3, z5_1, vz_sum_1);

                            c3_0 = _mm256_sub_ps(c3_0, vz_sum_0);
                            c3_1 = _mm256_sub_ps(c3_1, vz_sum_1);

                            _mm256_storeu_ps(cptr3 + 0, c3_0);
                            _mm256_storeu_ps(cptr3 + 8, c3_1);
                        }
                    }

                    /* Remainder rows */
                    for (; r < r_end; ++r)
                    {
                        const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                        const __m256 v2 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                        const __m256 v3 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);
                        const __m256 v4 = _mm256_set1_ps((r < len4) ? vp4[(size_t)r * n] : 0.0f);
                        const __m256 v5 = _mm256_set1_ps((r < len5) ? vp5[(size_t)r * n] : 0.0f);

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
                        vz_sum_0 = _mm256_fmadd_ps(v4, z4_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v4, z4_1, vz_sum_1);
                        vz_sum_0 = _mm256_fmadd_ps(v5, z5_0, vz_sum_0);
                        vz_sum_1 = _mm256_fmadd_ps(v5, z5_1, vz_sum_1);

                        c0 = _mm256_sub_ps(c0, vz_sum_0);
                        c1 = _mm256_sub_ps(c1, vz_sum_1);

                        _mm256_storeu_ps(cptr + 0, c0);
                        _mm256_storeu_ps(cptr + 8, c1);
                    }
                }

                /* 8-wide and scalar remainders (with conditionals) */
                for (; j + 7 < kc; j += 8)
                {
                    const __m256 z0 = _mm256_loadu_ps(Z + (size_t)(p + 0) * kc + j);
                    const __m256 z1 = _mm256_loadu_ps(Z + (size_t)(p + 1) * kc + j);
                    const __m256 z2 = _mm256_loadu_ps(Z + (size_t)(p + 2) * kc + j);
                    const __m256 z3 = _mm256_loadu_ps(Z + (size_t)(p + 3) * kc + j);
                    const __m256 z4 = _mm256_loadu_ps(Z + (size_t)(p + 4) * kc + j);
                    const __m256 z5 = _mm256_loadu_ps(Z + (size_t)(p + 5) * kc + j);

                    for (uint16_t r = 0; r < len0; ++r)
                    {
                        const __m256 v0 = _mm256_set1_ps(vp0[(size_t)r * n]);
                        const __m256 v1 = _mm256_set1_ps((r < len1) ? vp1[(size_t)r * n] : 0.0f);
                        const __m256 v2 = _mm256_set1_ps((r < len2) ? vp2[(size_t)r * n] : 0.0f);
                        const __m256 v3 = _mm256_set1_ps((r < len3) ? vp3[(size_t)r * n] : 0.0f);
                        const __m256 v4 = _mm256_set1_ps((r < len4) ? vp4[(size_t)r * n] : 0.0f);
                        const __m256 v5 = _mm256_set1_ps((r < len5) ? vp5[(size_t)r * n] : 0.0f);

                        float *cptr = Cpack + (size_t)(r + p + 0) * kc + j;
                        __m256 c = _mm256_loadu_ps(cptr);

                        __m256 vz_sum = _mm256_mul_ps(v0, z0);
                        vz_sum = _mm256_fmadd_ps(v1, z1, vz_sum);
                        vz_sum = _mm256_fmadd_ps(v2, z2, vz_sum);
                        vz_sum = _mm256_fmadd_ps(v3, z3, vz_sum);
                        vz_sum = _mm256_fmadd_ps(v4, z4, vz_sum);
                        vz_sum = _mm256_fmadd_ps(v5, z5, vz_sum);

                        c = _mm256_sub_ps(c, vz_sum);
                        _mm256_storeu_ps(cptr, c);
                    }
                }

                for (; j < kc; ++j)
                {
                    const float z0_val = Z[(size_t)(p + 0) * kc + j];
                    const float z1_val = Z[(size_t)(p + 1) * kc + j];
                    const float z2_val = Z[(size_t)(p + 2) * kc + j];
                    const float z3_val = Z[(size_t)(p + 3) * kc + j];
                    const float z4_val = Z[(size_t)(p + 4) * kc + j];
                    const float z5_val = Z[(size_t)(p + 5) * kc + j];

                    for (uint16_t r = 0; r < len0; ++r)
                    {
                        const float v0_val = vp0[(size_t)r * n];
                        const float v1_val = (r < len1) ? vp1[(size_t)r * n] : 0.0f;
                        const float v2_val = (r < len2) ? vp2[(size_t)r * n] : 0.0f;
                        const float v3_val = (r < len3) ? vp3[(size_t)r * n] : 0.0f;
                        const float v4_val = (r < len4) ? vp4[(size_t)r * n] : 0.0f;
                        const float v5_val = (r < len5) ? vp5[(size_t)r * n] : 0.0f;

                        float vz_sum = v0_val * z0_val + v1_val * z1_val + v2_val * z2_val +
                                       v3_val * z3_val + v4_val * z4_val + v5_val * z5_val;
                        Cpack[(size_t)(r + p + 0) * kc + j] -= vz_sum;
                    }
                }
            } /* end slow path */
        } /* end 6-row blocks */

        /* Remainder reflectors (1-5) */
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

#endif /* QR_AVX2_KERNELS_H */