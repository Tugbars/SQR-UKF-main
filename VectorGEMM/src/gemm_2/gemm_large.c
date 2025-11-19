/**
 * @file gemm_large.c
 * @brief Tier 2: Planned Execution with SIMD Packing
 *
 * IMPROVEMENTS:
 * - Uses pre-computed tile counts (no division in hot path)
 * - Uses pre-selected kernels for full tiles (no selection overhead)
 * - Only calls gemm_select_kernels() for edge tiles
 * - SIMD-optimized packing (1.5-2x faster)
 * - Fixed alpha/beta handling
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
#include <assert.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

//==============================================================================
// STRIDE DESCRIPTOR
//==============================================================================

typedef struct
{
    size_t a_k_stride;
    size_t b_k_stride;
} pack_strides_t;

//==============================================================================
// BETA PRE-SCALING
//==============================================================================

static void scale_matrix_beta(
    float *restrict C,
    size_t M, size_t N,
    float beta)
{
    if (beta == 0.0f)
    {
        memset(C, 0, M * N * sizeof(float));
    }
    else if (beta != 1.0f)
    {
        __m256 vbeta = _mm256_set1_ps(beta);

        for (size_t i = 0; i < M; ++i)
        {
            float *row = C + i * N;
            size_t j = 0;

            for (; j + 7 < N; j += 8)
            {
                __m256 c = _mm256_loadu_ps(row + j);
                _mm256_storeu_ps(row + j, _mm256_mul_ps(c, vbeta));
            }

            for (; j < N; ++j)
            {
                row[j] *= beta;
            }
        }
    }
}

//==============================================================================
// SIMD-OPTIMIZED PACKING
//==============================================================================

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

    size_t actual_mr = (ib >= 16) ? 16 : 8;
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
            float *dst = Ap + k * actual_mr;

            size_t i = 0;

            for (; i + 7 < ib; i += 8)
            {
                __m256 v = _mm256_set_ps(
                    src_col[7 * K], src_col[6 * K], src_col[5 * K], src_col[4 * K],
                    src_col[3 * K], src_col[2 * K], src_col[1 * K], src_col[0 * K]);
                _mm256_storeu_ps(dst + i, v);
                src_col += 8 * K;
            }

            const float *src_tail = A + (i0 + i) * K + (k0 + k);
            for (; i < ib; ++i)
            {
                dst[i] = src_tail[0];
                src_tail += K;
            }
        }
    }
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

            for (; i + 7 < ib; i += 8)
            {
                __m256 v = _mm256_set_ps(
                    src_col[7 * K], src_col[6 * K], src_col[5 * K], src_col[4 * K],
                    src_col[3 * K], src_col[2 * K], src_col[1 * K], src_col[0 * K]);
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, valpha));
                src_col += 8 * K;
            }

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

static pack_strides_t pack_B_panel_simd(
    float *restrict Bp,
    const float *restrict B,
    size_t K, size_t N,
    size_t k0, size_t kb,
    size_t j0, size_t jb)
{
    (void)K;

    const size_t B_STRIDE = 16;

    memset(Bp, 0, kb * B_STRIDE * sizeof(float));

    for (size_t k = 0; k < kb; ++k)
    {
        if (k + 4 < kb)
        {
            PREFETCH_T0(B + (k0 + k + 4) * N + j0);
        }

        const float *src_row = B + (k0 + k) * N + j0;
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
                                   Kblk, n_block, mask_unused);
        break;
    case KERN_4x8_STORE:
        gemm_4x8_panel_avx2fma_store(c, ldc, Ap, a_k_stride, Bp, b_k_stride,
                                     Kblk, n_block, mask_unused);
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

//==============================================================================
// MAIN EXECUTION LOOP (OPTIMIZED!)
//==============================================================================

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

    //--------------------------------------------------------------------------
    // Beta pre-scaling (once, before K-loop)
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    // USE PRE-COMPUTED TILE COUNTS (no division!)
    //--------------------------------------------------------------------------
    const size_t n_nc_tiles = plan->n_nc_tiles;
    const size_t n_kc_tiles = plan->n_kc_tiles;
    const size_t n_mc_tiles = plan->n_mc_tiles;

    //--------------------------------------------------------------------------
    // NC → KC → MC loop structure (maximize B reuse in L2 cache)
    //--------------------------------------------------------------------------
    for (size_t jt = 0; jt < n_nc_tiles; jt++)
    {
        size_t j0 = jt * plan->NC;
        size_t jb = MIN(plan->NC, plan->N - j0);

        for (size_t kt = 0; kt < n_kc_tiles; kt++)
        {
            size_t k0 = kt * plan->KC;
            size_t kb = MIN(plan->KC, plan->K - k0);

            size_t n_panels = (jb + plan->NR - 1) / plan->NR;

            //------------------------------------------------------------------
            // Pack B once per KC×NC tile
            //------------------------------------------------------------------
            pack_strides_t b_strides;
            for (size_t p = 0; p < n_panels; p++)
            {
                size_t j = j0 + p * plan->NR;
                size_t jw = MIN(plan->NR, j0 + jb - j);

                float *Bp_panel = Bp + p * plan->KC * 16;
                b_strides = pack_B_panel_simd(Bp_panel, B, plan->K, plan->N,
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

                    size_t pack_mr = (mh >= 16) ? 16 : 8;

                    //----------------------------------------------------------
                    // Pack A with alpha scaling
                    //----------------------------------------------------------
                    pack_strides_t a_strides = pack_A_panel_simd(
                        Ap, A, plan->M, plan->K, i, mh, k0, kb, alpha, pack_mr);

                    //----------------------------------------------------------
                    // Execute kernels on all N-panels
                    //----------------------------------------------------------
                    for (size_t p = 0; p < n_panels; p++)
                    {
                        size_t j = j0 + p * plan->NR;
                        size_t jw = MIN(plan->NR, j0 + jb - j);

                        //------------------------------------------------------
                        // FAST PATH: Full tiles (most common case)
                        //------------------------------------------------------
                        gemm_kernel_id_t kernel_id;

                        if (mh == plan->MR && jw == plan->NR)
                        {
                            // USE PRE-SELECTED KERNELS (no selection overhead!)
                            kernel_id = (kt == 0 && first_accumulation)
                                            ? plan->kern_full_store
                                            : plan->kern_full_add;
                        }
                        else
                        {
                            //--------------------------------------------------
                            // SLOW PATH: Edge tiles (rare, only at boundaries)
                            //--------------------------------------------------
                            gemm_kernel_id_t kern_add, kern_store;
                            int dummy_width;
                            gemm_select_kernels(mh, jw, &kern_add, &kern_store, &dummy_width);

                            kernel_id = (kt == 0 && first_accumulation)
                                            ? kern_store
                                            : kern_add;
                        }

                        //------------------------------------------------------
                        // Dispatch kernel
                        //------------------------------------------------------
                        float *cptr = C + i * plan->N + j;
                        float *bptr = Bp + p * plan->KC * 16;

                        dispatch_kernel(
                            kernel_id,
                            cptr,
                            plan->N,
                            Ap,
                            a_strides.a_k_stride,
                            bptr,
                            b_strides.b_k_stride,
                            kb,
                            mh,
                            jw);
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

void gemm_get_tuning(size_t M, size_t K, size_t N,
                     size_t *MC, size_t *KC, size_t *NC,
                     size_t *MR, size_t *NR)
{
    gemm_select_blocking(M, K, N, MC, KC, NC, MR, NR);
}

/*
3. Big correctness landmine: pack_A_panel_simd

This one deserves a red flag.

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

    size_t actual_mr = (ib >= 16) ? 16 : 8;
    assert(requested_mr == actual_mr && "MR mismatch: planning error");

    memset(Ap, 0, kb * actual_mr * sizeof(float));
    ...
    float *dst = Ap + k * actual_mr;

    size_t i = 0;
    for (; i + 7 < ib; i += 8) {
        ...
        _mm256_storeu_ps(dst + i, v);
        src_col += 8 * K;
    }

    const float *src_tail = A + (i0 + i) * K + (k0 + k);
    for (; i < ib; ++i) {
        dst[i] = src_tail[0];
        src_tail += K;
    }


Key facts:

actual_mr is chosen from ib, not from some fixed MR:

actual_mr = 16 if ib >= 16

actual_mr = 8 otherwise.

Ap is assumed to be kb * actual_mr floats.

The inner scalar tail writes up to dst[i] with i < ib.

Now imagine:

MR = 16 (in the plan).

You’re on the last MR-tile with ib = mh = 12.
Then:

actual_mr = (ib >= 16) ? 16 : 8  →  8
memset(Ap, 0, kb * 8 * sizeof(float));


But the for loop uses i < ib:

For k = kb-1, dst = Ap + k * 8.

i runs up to ib-1 = 11.
Access dst[11] = Ap[k*8 + 11] → index = 11 into an 8-wide row.

For the last k, that becomes:

max index = k * 8 + 11.

Valid range is [0, kb*8 - 1], but k = kb - 1 gives index = 8*(kb-1) + 11 = 8*kb + 3 which is > 8*kb - 1. That’s a straight OOB write.

So unless your planner guarantees that ib <= 8 whenever ib < 16, this will absolutely walk past the end of the Ap panel.

Fix options

Pick one of:

Lock actual_mr to requested_mr and require the planner to guarantee ib <= requested_mr:

size_t actual_mr = requested_mr;
assert(ib <= actual_mr);
memset(Ap, 0, kb * actual_mr * sizeof(float));


Then make sure your tiling logic never creates an MR tile with mh > MR.

Separate logical_m and physical_mr:

Keep actual_mr as 8 or 16 (physical stride).

Use a second parameter for logical count ib (already present).

But never use ib as a bound that exceeds actual_mr. If you want to support ib > 8, actual_mr must be ≥ ib.

In practice: if you really want 16-row kernels, make actual_mr = 16 whenever MR = 16, even if ib < 16, and just zero the unused rows.

Force MR = 8 globally.

If the planner never uses 16-height kernels in practice, you could simplify:

MR = 8 in the plan.

actual_mr = 8 always.

The 16xK kernels become internal composites only and always receive m <= 8 per call.

But right now you do have explicit 16x* kernels, and pack_A_panel_simd clearly tries to support 8 and 16, so I’d go with fix (1) or (2).

*/

/*
4. pack_B_panel_simd sanity
const size_t B_STRIDE = 16;
...
memset(Bp, 0, kb * B_STRIDE * sizeof(float));
...
float *dst = Bp + k * B_STRIDE;
...
for (; j + 7 < jb; j += 8) {
    __m256 v = _mm256_loadu_ps(src_row + j);
    _mm256_storeu_ps(dst + j, v);
}
for (; j < jb; ++j) {
    dst[j] = src_row[j];
}
...
strides.b_k_stride = B_STRIDE;


This is fine as long as:

plan->NR <= 16 (which is implied by your kernels: 8- and 16-wide).

For narrow edge tiles (jw < NR), you just leave the rest of the 16-wide row zero, which is fine.

I’d just drop a debug assertion in the packing call site:

assert(plan->NR <= 16);


for sanity.

Also note you recompute b_strides in the loop but only use the last:

for (size_t p = 0; p < n_panels; p++) {
    ...
    b_strides = pack_B_panel_simd(...);
}
...
b_strides.b_k_stride // same for every panel, so fine


That’s logically correct, but a bit misleading: you could just call pack_B_panel_simd once, return the stride, and use the known B_STRIDE constant everywhere. Not a correctness problem, just clarity.
*/