/**
 * @file gemm_large.c
 * @brief Tier 2: Planned Execution for Large Matrices (CORRECTED)
 *
 * CHANGES FROM ORIGINAL:
 * 1. Fixed n_ntiles → n_npanels (compile error)
 * 2. Dynamic kernel selection (architectural mismatch fix)
 * 3. Alpha scaling moved to packing (performance + correctness)
 * 4. Beta pre-scaling added (missing feature)
 * 5. Corrected loop structure (cache optimization)
 * 6. Fixed workspace indexing (silent corruption fix)
 *
 * @author TUGBARS (corrected implementation)
 * @date 2025
 */

#include "gemm_kernels_avx2.h"
#include "gemm_simd_ops.h"
#include "gemm_small.h"
#include <string.h>
#include <stdbool.h>

//==============================================================================
// UTILITIES
//==============================================================================

#define MIN(a, b) ((a) < (b) ? (a) : (b))

//==============================================================================
// STRIDE DESCRIPTOR
//==============================================================================

/**
 * @brief Describes strides used in packed buffers
 *
 * Used to make stride contracts explicit between packing and kernel execution.
 * Helps catch packing/kernel mismatches at runtime via assertions.
 */
typedef struct
{
    size_t a_k_stride; // Stride between K iterations in packed A (MR: 8 or 16)
    size_t b_k_stride; // Stride between K iterations in packed B (always 16)
} pack_strides_t;

/**
 * @brief Pre-scale entire C matrix by beta (called once)
 */
static void scale_matrix_beta(
    float *restrict C,
    size_t M, size_t N,
    float beta)
{
    if (beta == 0.0f)
    {
        // Zero out C
        memset(C, 0, M * N * sizeof(float));
    }
    else if (beta != 1.0f)
    {
        __m256 vbeta = _mm256_set1_ps(beta);
        for (size_t i = 0; i < M; ++i)
        {
            float *row = C + i * N;
            size_t j = 0;

            // Vectorized
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
}

//==============================================================================
// PACKING WITH INTEGRATED ALPHA SCALING
//==============================================================================

/**
 * @brief Pack A panel with optional alpha scaling (CORRECTED)
 *
 * Scales during packing to avoid:
 * - Overwriting packed buffer
 * - Scaling zeros in padding
 * - Redundant scaling across M-tiles
 */
/**
 * @brief Pack A panel with alpha scaling (FIXED)
 * CHANGE: Replace hardcoded 16 with plan->MR
 */
static pack_strides_t pack_A_panel_scaled(
    float *restrict Ap,
    const float *restrict A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t k0, size_t kb,
    float alpha,
    size_t requested_mr) // Explicit request from caller
{
    (void)M;

    // Determine actual MR based on tile height
    size_t actual_mr = (ib >= 16) ? 16 : 8;

    // Validate: catch planning/packing mismatches
    assert(requested_mr == actual_mr && "MR mismatch: planning error");

    memset(Ap, 0, kb * actual_mr * sizeof(float));

    if (alpha == 1.0f)
    {
        for (size_t k = 0; k < kb; ++k)
        {
            if (k + 8 < kb)
            {
                PREFETCH_T0(A + i0 * K + (k0 + k + 8));
            }

            const float *src_col = A + i0 * K + (k0 + k);
            float *dst = Ap + k * actual_mr; // Use actual_mr

            for (size_t i = 0; i < ib; ++i)
            {
                dst[i] = src_col[i * K];
            }
        }
    }
    else
    {
        for (size_t k = 0; k < kb; ++k)
        {
            if (k + 8 < kb)
            {
                PREFETCH_T0(A + i0 * K + (k0 + k + 8));
            }

            const float *src_col = A + i0 * K + (k0 + k);
            float *dst = Ap + k * actual_mr; // Use actual_mr

            for (size_t i = 0; i < ib; ++i)
            {
                dst[i] = src_col[i * K] * alpha;
            }
        }
    }

    // Return actual stride used
    pack_strides_t strides;
    strides.a_k_stride = actual_mr;
    strides.b_k_stride = 0; // N/A for A
    return strides;
}

/**
 * @brief Pack B panel (no alpha - handled in A)
 */
static pack_strides_t pack_B_panel(
    float *restrict Bp,
    const float *restrict B,
    size_t K, size_t N,
    size_t k0, size_t kb,
    size_t j0, size_t jb)
{
    (void)K;

    const size_t B_STRIDE = 16; // Always pad to 16

    memset(Bp, 0, kb * B_STRIDE * sizeof(float));

    for (size_t k = 0; k < kb; ++k)
    {
        if (k + 4 < kb)
        {
            PREFETCH_T0(B + (k0 + k + 4) * N + j0);
        }

        const float *src_row = B + (k0 + k) * N + j0;
        float *dst = Bp + k * B_STRIDE;

        // Copy jb elements (rest already zeroed)
        for (size_t j = 0; j < jb; ++j)
        {
            dst[j] = src_row[j];
        }
    }

    // Return actual stride used
    pack_strides_t strides;
    strides.a_k_stride = 0; // N/A for B
    strides.b_k_stride = B_STRIDE;
    return strides;
}

//==============================================================================
// KERNEL DISPATCH (Unchanged but included for completeness)
//==============================================================================

static inline void dispatch_kernel(
    gemm_kernel_id_t kernel_id,
    float *restrict c,
    size_t ldc,
    const float *restrict Ap,
    size_t a_k_stride, // ← ADDED
    const float *restrict Bp,
    size_t b_k_stride, // ← ADDED
    size_t Kblk,
    size_t m_block,
    size_t n_block,
    __m256i mask_lo,
    __m256i mask_hi)
{
    switch (kernel_id)
    {
    case KERN_16x8_ADD:
        gemm_16x8_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                    Kblk, m_block, n_block, mask_lo);
        break;
    case KERN_16x8_STORE:
        gemm_16x8_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                      Kblk, m_block, n_block, mask_lo);
        break;
    case KERN_8x8_ADD:
        gemm_8x8_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                   Kblk, m_block, n_block, mask_lo);
        break;
    case KERN_8x8_STORE:
        gemm_8x8_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, m_block, n_block, mask_lo);
        break;
    case KERN_16x6_ADD:
        gemm_16x6_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                    Kblk, m_block, n_block, mask_lo);
        break;
    case KERN_16x6_STORE:
        gemm_16x6_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                      Kblk, m_block, n_block, mask_lo);
        break;
    case KERN_8x6_ADD:
        gemm_8x6_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                   Kblk, m_block, n_block, mask_lo);
        break;
    case KERN_8x6_STORE:
        gemm_8x6_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, m_block, n_block, mask_lo);
        break;
    case KERN_4x8_ADD:
        gemm_4x8_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                   Kblk, n_block, mask_lo);
        break;
    case KERN_4x8_STORE:
        gemm_4x8_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, n_block, mask_lo);
        break;
    case KERN_1x8_ADD:
        gemm_1x8_panel_avx2fma_add(c, Ap, a_k_stride, Bp, b_k_stride,
                                   Kblk, n_block, mask_lo);
        break;
    case KERN_1x8_STORE:
        gemm_1x8_panel_avx2fma_store(c, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, n_block, mask_lo);
        break;
    case KERN_8x16_ADD:
        gemm_8x16_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                    Kblk, m_block, n_block, mask_lo, mask_hi);
        break;
    case KERN_8x16_STORE:
        gemm_8x16_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                      Kblk, m_block, n_block, mask_lo, mask_hi);
        break;
    case KERN_16x16_ADD:
        // First 8 rows
        gemm_8x16_panel_avx2fma_add(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                    Kblk, 8, n_block, mask_lo, mask_hi);
        // Next 8 rows (offset by 8 elements in packed A)
        gemm_8x16_panel_avx2fma_add(c + 8 * ldc, ldc, Ap + 8, a_k_stride,
                                    Bp, b_k_stride, Kblk, m_block - 8, n_block,
                                    mask_lo, mask_hi);
        break;

    case KERN_16x16_STORE:
        gemm_8x16_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                      Kblk, 8, n_block, mask_lo, mask_hi);
        gemm_8x16_panel_avx2fma_store(c + 8 * ldc, ldc, Ap + 8, a_k_stride,
                                      Bp, b_k_stride, Kblk, m_block - 8, n_block,
                                      mask_lo, mask_hi);
        break;
    default:
        break;
    }
}

//==============================================================================
// MAIN EXECUTION LOOP (FULLY CORRECTED)
//==============================================================================

/**
 * @brief Execute planned GEMM: C = alpha*A*B + beta*C (CORRECTED)
 *
 * FIXES APPLIED:
 * 1. n_ntiles → n_npanels (compile error fix)
 * 2. Dynamic kernel selection per tile (architectural fix)
 * 3. Alpha scaling integrated into packing (performance + correctness)
 * 4. Beta pre-scaled before K-loop (missing feature)
 * 5. Corrected loop structure: NC → KC → MC (cache optimization)
 * 6. Fixed workspace indexing (corruption fix)
 *
 * Loop structure:
 * NC-level: Split N into cache-friendly panels
 *   KC-level: Split K into cache-friendly blocks
 *     Pack B once per KC×NC tile (reused across all MC tiles)
 *     MC-level: Split M into cache-friendly blocks
 *       MR-level: Micro-tiles (register blocks)
 *         Pack A once per MR×KC tile
 *         Execute kernels on all NR-panels
 */
/**
 * @brief Execute planned GEMM: C = alpha*A*B + beta*C (FIXED)
 */
int gemm_execute_plan(
    gemm_plan_t *plan,
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    float alpha,
    float beta)
{
    if (!plan || !C || !A || !B)
    {
        return -1;
    }

    // Beta pre-scaling
    bool first_accumulation;
    if (beta == 0.0f)
    {
        memset(C, 0, plan->M * plan->N * sizeof(float));
        first_accumulation = true;
    }
    else if (beta != 1.0f)
    {
        scale_matrix_beta(C, plan->M, plan->N, beta);
        first_accumulation = false;
    }
    else
    {
        first_accumulation = false;
    }

    float *Ap = plan->workspace_a;
    float *Bp = plan->workspace_b;

    size_t n_nc_tiles = (plan->N + plan->NC - 1) / plan->NC;
    size_t n_kc_tiles = (plan->K + plan->KC - 1) / plan->KC;
    size_t n_mc_tiles = (plan->M + plan->MC - 1) / plan->MC;

    for (size_t jt = 0; jt < n_nc_tiles; jt++)
    {
        size_t j0 = jt * plan->NC;
        size_t jb = MIN(plan->NC, plan->N - j0);

        for (size_t kt = 0; kt < n_kc_tiles; kt++)
        {
            size_t k0 = kt * plan->KC;
            size_t kb = MIN(plan->KC, plan->K - k0);

            size_t n_panels = (jb + plan->NR - 1) / plan->NR;

            // PACK B once per KC×NC tile and CAPTURE strides
            pack_strides_t b_strides;
            for (size_t p = 0; p < n_panels; p++)
            {
                size_t j = j0 + p * plan->NR;
                size_t jw = MIN(plan->NR, j0 + jb - j);

                float *Bp_panel = Bp + p * plan->KC * 16; // B always padded to stride 16

                // CAPTURE B stride from packing
                b_strides = pack_B_panel(Bp_panel, B, plan->K, plan->N,
                                         k0, kb, j, jw);
            }

            for (size_t it = 0; it < n_mc_tiles; it++)
            {
                size_t i0 = it * plan->MC;
                size_t ib = MIN(plan->MC, plan->M - i0);
                size_t n_mr_tiles = (ib + plan->MR - 1) / plan->MR;

                for (size_t mt = 0; mt < n_mr_tiles; mt++)
                {
                    size_t i = i0 + mt * plan->MR;
                    size_t mh = MIN(plan->MR, plan->M - i);

                    // Determine expected packing MR
                    size_t pack_mr = (mh >= 16) ? 16 : 8;

                    // PACK A and CAPTURE stride
                    pack_strides_t a_strides = pack_A_panel_scaled(
                        Ap, A, plan->M, plan->K, i, mh, k0, kb, alpha, pack_mr);

                    // Execute kernels on all N-panels with EXPLICIT strides
                    for (size_t p = 0; p < n_panels; p++)
                    {
                        size_t j = j0 + p * plan->NR;
                        size_t jw = MIN(plan->NR, j0 + jb - j);

                        // Dynamic kernel selection
                        gemm_kernel_id_t kern_add, kern_store;
                        int kernel_width;
                        gemm_select_kernels(mh, jw, &kern_add, &kern_store, &kernel_width);

                        // Choose ADD or STORE mode
                        gemm_kernel_id_t kernel_id;
                        if (kt == 0 && first_accumulation)
                        {
                            kernel_id = kern_store;
                        }
                        else
                        {
                            kernel_id = kern_add;
                        }

                        // Get pre-computed masks
                        size_t global_panel_idx = j / plan->NR;
                        panel_info_t *panel = &plan->npanels[global_panel_idx];

                        // Dispatch with EXPLICIT strides
                        float *cptr = C + i * plan->N + j;
                        float *bptr = Bp + p * plan->KC * 16;

                        dispatch_kernel(
                            kernel_id,
                            cptr,
                            plan->N,
                            Ap,
                            a_strides.a_k_stride, // ← EXPLICIT A stride
                            bptr,
                            b_strides.b_k_stride, // ← EXPLICIT B stride
                            kb,
                            mh,
                            jw,
                            panel->mask_lo,
                            panel->mask_hi);
                    }
                }
            }
        }
    }

    return 0;
}

//==============================================================================
// PUBLIC API (Unchanged)
//==============================================================================

int gemm_auto(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    // Try Tier 1 first
    int ret = gemm_small_dispatch(C, A, B, M, K, N, N, alpha, beta);
    if (ret == 0)
    {
        return 0;
    }

    // Fall back to Tier 2
    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan)
    {
        return -1;
    }

    ret = gemm_execute_plan(plan, C, A, B, alpha, beta);

    gemm_plan_destroy(plan);
    return ret;
}

int gemm_static(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    if (!gemm_fits_static(M, K, N))
    {
        return -1;
    }

    gemm_plan_t *plan = gemm_plan_create_with_mode(M, K, N, GEMM_MEM_STATIC);
    if (!plan)
    {
        return -1;
    }

    int ret = gemm_execute_plan(plan, C, A, B, alpha, beta);

    gemm_plan_destroy(plan);
    return ret;
}

int gemm_dynamic(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    gemm_plan_t *plan = gemm_plan_create_with_mode(M, K, N, GEMM_MEM_DYNAMIC);
    if (!plan)
    {
        return -1;
    }

    int ret = gemm_execute_plan(plan, C, A, B, alpha, beta);

    gemm_plan_destroy(plan);
    return ret;
}