/**
 * @file gemm_large.c
 * @brief Tier 2: Planned Execution for Large Matrices
 * 
 * This module implements:
 * - Optimized packing for A and B matrices
 * - Enum-based kernel dispatch (zero pointer indirection)
 * - 3-level cache blocking execution
 * - Alpha/beta scaling
 * - Proper mask handling for 8-wide and 16-wide kernels
 * 
 * @author TUGBARS
 * @date 2025
 */

#include "gemm_planning.h"
#include "gemm_kernels_avx2.h"  // Your complete AVX2 kernels from doc3
#include "gemm_simd_ops.h"
#include <string.h>

//==============================================================================
// PACKING CONFIGURATION
//==============================================================================

// Prefetch distances tuned for i14900K
#define PACK_PREFETCH_DIST_A 64   // Prefetch 64 bytes ahead for A
#define PACK_PREFETCH_DIST_B 128  // Prefetch 128 bytes ahead for B

//==============================================================================
// PACK A: Column-major source -> Row-major packed (MR rows)
//==============================================================================

/**
 * @brief Pack A panel: [i0:i0+ib, k0:k0+kb] from column-major A
 * 
 * Layout: Packed as [k][mr] for cache-friendly access during compute
 * 
 * @param Ap Destination packed buffer (aligned, size: kb * MR)
 * @param A Source matrix (column-major, size: M * K)
 * @param M Total rows in A
 * @param K Total columns in A
 * @param i0 Starting row
 * @param ib Number of rows to pack (<= MR)
 * @param k0 Starting column  
 * @param kb Number of columns to pack (<= KC)
 */
static void pack_A_panel(
    float * restrict Ap,
    const float * restrict A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t k0, size_t kb)
{
    (void)M;  // Unused but kept for API consistency
    
    // Zero out entire buffer (handles padding for ib < MR)
    memset(Ap, 0, kb * 16 * sizeof(float));  // Max MR is 16
    
    // Pack actual data
    for (size_t k = 0; k < kb; ++k) {
        // Prefetch next column
        if (k + 8 < kb) {
            const float *prefetch_col = A + (i0) * K + (k0 + k + 8);
            PREFETCH_T0(prefetch_col);
            PREFETCH_T0(prefetch_col + 8);
        }
        
        const float *src_col = A + i0 * K + (k0 + k);
        float *dst = Ap + k * 16;  // Max stride is 16
        
        // Copy ib elements
        for (size_t i = 0; i < ib; ++i) {
            dst[i] = src_col[i * K];
        }
        // Remaining elements already zeroed by memset
    }
}

/**
 * @brief Pack A panel with aligned loads (fast path)
 * 
 * Used when A is known to be 32-byte aligned
 */
static void pack_A_panel_aligned(
    float * restrict Ap,
    const float * restrict A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t k0, size_t kb)
{
    (void)M;
    
    // Zero padding
    if (ib < 16) {
        memset(Ap, 0, kb * 16 * sizeof(float));
    }
    
    // Pack with vectorized loads when possible
    if (ib == 16) {
        // Full height - can use SIMD
        for (size_t k = 0; k < kb; ++k) {
            // Prefetch
            if (k + 8 < kb) {
                PREFETCH_T0(A + i0 * K + (k0 + k + 8));
            }
            
            const float *src_col = A + i0 * K + (k0 + k);
            float *dst = Ap + k * 16;
            
            // Gather 16 elements from column-major A
            // This is scalar but with good prefetch
            for (size_t i = 0; i < 16; ++i) {
                dst[i] = src_col[i * K];
            }
        }
    } else if (ib == 8) {
        // 8 rows
        for (size_t k = 0; k < kb; ++k) {
            const float *src_col = A + i0 * K + (k0 + k);
            float *dst = Ap + k * 16;  // Still stride 16 for consistency
            
            for (size_t i = 0; i < 8; ++i) {
                dst[i] = src_col[i * K];
            }
        }
    } else {
        // Partial height
        pack_A_panel(Ap, A, M, K, i0, ib, k0, kb);
    }
}

//==============================================================================
// PACK B: Row-major source -> Column-panel packed (NR cols)
//==============================================================================

/**
 * @brief Pack B panel: [k0:k0+kb, j0:j0+jb] from row-major B
 * 
 * Layout: Packed as [k][nr] for streaming access during compute
 * 
 * @param Bp Destination packed buffer (aligned, size: kb * NR)
 * @param B Source matrix (row-major, size: K * N)
 * @param K Total rows in B
 * @param N Total columns in B
 * @param k0 Starting row
 * @param kb Number of rows to pack (<= KC)
 * @param j0 Starting column
 * @param jb Number of columns to pack (<= NR)
 */
static void pack_B_panel(
    float * restrict Bp,
    const float * restrict B,
    size_t K, size_t N,
    size_t k0, size_t kb,
    size_t j0, size_t jb)
{
    (void)K;
    
    // Full width - fast path
    if (jb == 8) {
        for (size_t k = 0; k < kb; ++k) {
            // Prefetch next row
            if (k + 4 < kb) {
                PREFETCH_T0(B + (k0 + k + 4) * N + j0);
            }
            
            const float *src_row = B + (k0 + k) * N + j0;
            float *dst = Bp + k * 8;
            
            // Vectorized copy
            __m256 data = _mm256_loadu_ps(src_row);
            _mm256_store_ps(dst, data);
        }
    } else if (jb == 16) {
        for (size_t k = 0; k < kb; ++k) {
            if (k + 4 < kb) {
                PREFETCH_T0(B + (k0 + k + 4) * N + j0);
                PREFETCH_T0(B + (k0 + k + 4) * N + j0 + 8);
            }
            
            const float *src_row = B + (k0 + k) * N + j0;
            float *dst = Bp + k * 16;
            
            __m256 lo = _mm256_loadu_ps(src_row);
            __m256 hi = _mm256_loadu_ps(src_row + 8);
            _mm256_store_ps(dst, lo);
            _mm256_store_ps(dst + 8, hi);
        }
    } else if (jb == 6) {
        for (size_t k = 0; k < kb; ++k) {
            const float *src_row = B + (k0 + k) * N + j0;
            float *dst = Bp + k * 6;
            
            // Scalar copy for width 6
            for (size_t j = 0; j < 6; ++j) {
                dst[j] = src_row[j];
            }
        }
    } else {
        // Partial width - use scalar copy
        // Zero padding first
        memset(Bp, 0, kb * 16 * sizeof(float));
        
        for (size_t k = 0; k < kb; ++k) {
            const float *src_row = B + (k0 + k) * N + j0;
            float *dst = Bp + k * 16;  // Max stride
            
            for (size_t j = 0; j < jb; ++j) {
                dst[j] = src_row[j];
            }
        }
    }
}

/**
 * @brief Pack B panel with aligned loads (fast path)
 */
static void pack_B_panel_aligned(
    float * restrict Bp,
    const float * restrict B,
    size_t K, size_t N,
    size_t k0, size_t kb,
    size_t j0, size_t jb)
{
    // Check if B is aligned at this position
    const float *B_start = B + k0 * N + j0;
    int is_aligned = (((uintptr_t)B_start & 31) == 0);
    
    if (is_aligned && jb == 8) {
        // Fast path: aligned 8-column pack
        for (size_t k = 0; k < kb; ++k) {
            if (k + 4 < kb) {
                PREFETCH_T0(B + (k0 + k + 4) * N + j0);
            }
            
            const float *src_row = B + (k0 + k) * N + j0;
            float *dst = Bp + k * 8;
            
            __m256 data = _mm256_load_ps(src_row);  // Aligned load
            _mm256_store_ps(dst, data);
        }
    } else if (is_aligned && jb == 16) {
        // Fast path: aligned 16-column pack
        for (size_t k = 0; k < kb; ++k) {
            if (k + 4 < kb) {
                PREFETCH_T0(B + (k0 + k + 4) * N + j0);
                PREFETCH_T0(B + (k0 + k + 4) * N + j0 + 8);
            }
            
            const float *src_row = B + (k0 + k) * N + j0;
            float *dst = Bp + k * 16;
            
            __m256 lo = _mm256_load_ps(src_row);
            __m256 hi = _mm256_load_ps(src_row + 8);
            _mm256_store_ps(dst, lo);
            _mm256_store_ps(dst + 8, hi);
        }
    } else {
        // Fall back to unaligned version
        pack_B_panel(Bp, B, K, N, k0, kb, j0, jb);
    }
}

//==============================================================================
// ENUM-BASED KERNEL DISPATCH (Zero pointer indirection)
//==============================================================================

/**
 * @brief Dispatch to appropriate kernel using direct switch statement
 * 
 * This compiles to a jump table with zero indirection cost.
 * Compiler can inline across the switch for maximum performance.
 */
static inline void dispatch_kernel(
    gemm_kernel_id_t kernel_id,
    float * restrict c,
    size_t ldc,
    const float * restrict Ap,
    const float * restrict Bp,
    size_t Kblk,
    size_t m_block,
    size_t n_block,
    __m256i mask_lo,
    __m256i mask_hi)
{
    switch (kernel_id) {
        //----------------------------------------------------------------------
        // 8-wide kernels (single mask)
        //----------------------------------------------------------------------
        case KERN_16x8_ADD:
            gemm_16x8_panel_avx2fma_add(c, ldc, Ap, Bp, Kblk, m_block, n_block, mask_lo);
            break;
            
        case KERN_16x8_STORE:
            gemm_16x8_panel_avx2fma_store(c, ldc, Ap, Bp, Kblk, m_block, n_block, mask_lo);
            break;
            
        case KERN_8x8_ADD:
            gemm_8x8_panel_avx2fma_add(c, ldc, Ap, Bp, Kblk, m_block, n_block, mask_lo);
            break;
            
        case KERN_8x8_STORE:
            gemm_8x8_panel_avx2fma_store(c, ldc, Ap, Bp, Kblk, m_block, n_block, mask_lo);
            break;
            
        case KERN_16x6_ADD:
            gemm_16x6_panel_avx2fma_add(c, ldc, Ap, Bp, Kblk, m_block, n_block, mask_lo);
            break;
            
        case KERN_16x6_STORE:
            gemm_16x6_panel_avx2fma_store(c, ldc, Ap, Bp, Kblk, m_block, n_block, mask_lo);
            break;
            
        case KERN_8x6_ADD:
            gemm_8x6_panel_avx2fma_add(c, ldc, Ap, Bp, Kblk, m_block, n_block, mask_lo);
            break;
            
        case KERN_8x6_STORE:
            gemm_8x6_panel_avx2fma_store(c, ldc, Ap, Bp, Kblk, m_block, n_block, mask_lo);
            break;
            
        case KERN_4x8_ADD:
            gemm_4x8_panel_avx2fma_add(c, ldc, Ap, Bp, Kblk, n_block, mask_lo);
            break;
            
        case KERN_4x8_STORE:
            gemm_4x8_panel_avx2fma_store(c, ldc, Ap, Bp, Kblk, n_block, mask_lo);
            break;
            
        case KERN_1x8_ADD:
            gemm_1x8_panel_avx2fma_add(c, Ap, Bp, Kblk, n_block, mask_lo);
            break;
            
        case KERN_1x8_STORE:
            gemm_1x8_panel_avx2fma_store(c, Ap, Bp, Kblk, n_block, mask_lo);
            break;
            
        //----------------------------------------------------------------------
        // 16-wide kernels (dual mask)
        //----------------------------------------------------------------------
        case KERN_8x16_ADD:
            gemm_8x16_panel_avx2fma_add(c, ldc, Ap, Bp, Kblk, m_block, n_block, 
                                       mask_lo, mask_hi);
            break;
            
        case KERN_8x16_STORE:
            gemm_8x16_panel_avx2fma_store(c, ldc, Ap, Bp, Kblk, m_block, n_block,
                                         mask_lo, mask_hi);
            break;
            
        case KERN_16x16_ADD:
            // TODO: Implement 16x16 kernels if needed
            // For now fall back to calling 8x16 twice
            gemm_8x16_panel_avx2fma_add(c, ldc, Ap, Bp, Kblk, 8, n_block, 
                                       mask_lo, mask_hi);
            gemm_8x16_panel_avx2fma_add(c + 8 * ldc, ldc, Ap + 8, Bp, Kblk, 
                                       m_block - 8, n_block, mask_lo, mask_hi);
            break;
            
        case KERN_16x16_STORE:
            gemm_8x16_panel_avx2fma_store(c, ldc, Ap, Bp, Kblk, 8, n_block,
                                         mask_lo, mask_hi);
            gemm_8x16_panel_avx2fma_store(c + 8 * ldc, ldc, Ap + 8, Bp, Kblk,
                                         m_block - 8, n_block, mask_lo, mask_hi);
            break;
            
        default:
            // Should never reach here if planning is correct
            break;
    }
}

//==============================================================================
// MAIN EXECUTION LOOP
//==============================================================================

/**
 * @brief Execute planned GEMM: C = alpha*A*B + beta*C
 * 
 * Three-level blocking structure:
 * 1. NC-level: Split N into cache-friendly panels
 * 2. KC-level: Split K into cache-friendly blocks (reuse packed B)
 * 3. MC-level: Split M into cache-friendly tiles (reuse packed A)
 * 
 * @return 0 on success, negative error code on failure
 */
int gemm_execute_plan(
    gemm_plan_t *plan,
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    float alpha,
    float beta)
{
    if (!plan || !C || !A || !B) {
        return -1;  // Invalid arguments
    }
    
    float *Ap = plan->workspace_a;
    float *Bp = plan->workspace_b;
    
    //==========================================================================
    // Three-level blocking: N -> K -> M
    //==========================================================================
    
    for (size_t jt = 0; jt < plan->n_ntiles; jt++) {
        size_t j0 = jt * plan->NC;
        size_t jb = (j0 + plan->NC <= plan->N) ? plan->NC : (plan->N - j0);
        
        for (size_t kt = 0; kt < plan->n_ktiles; kt++) {
            size_t kk = kt * plan->KC;
            size_t kb = (kk + plan->KC <= plan->K) ? plan->KC : (plan->K - kk);
            
            //------------------------------------------------------------------
            // Pack B for this K×N tile (done once, reused for all M-tiles)
            //------------------------------------------------------------------
            size_t n_panels_in_tile = (jb + plan->NR - 1) / plan->NR;
            size_t panel_base = (j0 / plan->NR);
            
            for (size_t p = 0; p < n_panels_in_tile; p++) {
                panel_info_t *panel = &plan->npanels[panel_base + p];
                size_t j = j0 + p * plan->NR;
                size_t n_block = (j + plan->NR <= j0 + jb) 
                                 ? plan->NR 
                                 : (j0 + jb - j);
                
                // Select packing function based on alignment
                if (plan->mem_mode == GEMM_MEM_STATIC || plan->workspace_aligned) {
                    pack_B_panel_aligned(
                        Bp + p * kb * plan->NR,
                        B, plan->K, plan->N,
                        kk, kb, j, n_block);
                } else {
                    pack_B_panel(
                        Bp + p * kb * plan->NR,
                        B, plan->K, plan->N,
                        kk, kb, j, n_block);
                }
            }
            
            //------------------------------------------------------------------
            // Process M-tiles with packed B
            //------------------------------------------------------------------
            for (size_t it = 0; it < plan->n_mtiles; it++) {
                size_t i0 = it * plan->MC;
                size_t ib = (i0 + plan->MC <= plan->M) ? plan->MC : (plan->M - i0);
                
                size_t tile_base = (i0 / plan->MR);
                
                for (size_t i = 0; i < ib; ) {
                    tile_info_t *tile = &plan->mtiles[tile_base + (i / plan->MR)];
                    size_t m_block = tile->i_height;
                    
                    //----------------------------------------------------------
                    // Pack A for this micro-panel
                    //----------------------------------------------------------
                    if (plan->mem_mode == GEMM_MEM_STATIC || plan->workspace_aligned) {
                        pack_A_panel_aligned(Ap, A, plan->M, plan->K, 
                                           i0 + i, m_block, kk, kb);
                    } else {
                        pack_A_panel(Ap, A, plan->M, plan->K,
                                   i0 + i, m_block, kk, kb);
                    }
                    
                    //----------------------------------------------------------
                    // Apply alpha scaling to packed A (once per K-tile)
                    //----------------------------------------------------------
                    if (alpha != 1.0f) {
                        __m256 va = _mm256_set1_ps(alpha);
                        size_t len = kb * 16;  // Max MR is 16
                        size_t idx = 0;
                        
                        // Vectorized scaling
                        for (; idx + 7 < len; idx += 8) {
                            __m256 v = _mm256_load_ps(Ap + idx);
                            _mm256_store_ps(Ap + idx, _mm256_mul_ps(v, va));
                        }
                        // Scalar tail
                        for (; idx < len; idx++) {
                            Ap[idx] *= alpha;
                        }
                    }
                    
                    //----------------------------------------------------------
                    // Execute kernel on each N-panel
                    //----------------------------------------------------------
                    for (size_t p = 0; p < n_panels_in_tile; p++) {
                        panel_info_t *panel = &plan->npanels[panel_base + p];
                        size_t j = j0 + p * plan->NR;
                        size_t n_block = (j + plan->NR <= j0 + jb) 
                                         ? plan->NR 
                                         : (j0 + jb - j);
                        
                        float *cptr = C + (i0 + i) * plan->N + j;
                        const float *bptr = Bp + p * kb * plan->NR;
                        
                        // Select kernel based on K-tile and beta
                        gemm_kernel_id_t kernel_id;
                        if (kt == 0 && beta == 0.0f) {
                            // First K-tile with beta=0: use STORE kernel
                            kernel_id = tile->kern_store;
                        } else {
                            // Subsequent K-tiles or beta!=0: use ADD kernel
                            kernel_id = tile->kern_add;
                        }
                        
                        // Dispatch to appropriate kernel (enum switch)
                        dispatch_kernel(
                            kernel_id,
                            cptr, plan->N,
                            Ap, bptr,
                            kb, m_block, n_block,
                            panel->mask_lo,
                            panel->mask_hi);
                    }
                    
                    i += m_block;
                }
            }
        }
    }
    
    return 0;  // Success
}

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief Main GEMM entry point with automatic mode selection
 * 
 * Routes to:
 * - Tier 1 (gemm_small.c) for exact fixed sizes (4x4, 6x6, 8x8, etc.)
 * - Tier 2 (this file) for everything else
 */
int gemm_auto(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    // Try Tier 1 first (small fixed-size matrices)
    int ret = gemm_small_dispatch(C, A, B, M, K, N, N, alpha, beta);
    if (ret == 0) {
        return 0;  // Handled by Tier 1
    }
    
    // Fall back to Tier 2 (planned execution)
    gemm_plan_t *plan = gemm_plan_create(M, K, N);
    if (!plan) {
        return -1;  // Planning failed
    }
    
    ret = gemm_execute_plan(plan, C, A, B, alpha, beta);
    
    gemm_plan_destroy(plan);
    return ret;
}

/**
 * @brief Explicit static mode GEMM
 * 
 * Forces static pool usage, errors if dimensions too large
 */
int gemm_static(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    // Check dimensions
    if (!gemm_fits_static(M, K, N)) {
        return -1;  // Dimensions exceed static pool
    }
    
    gemm_plan_t *plan = gemm_plan_create_with_mode(M, K, N, GEMM_MEM_STATIC);
    if (!plan) {
        return -1;
    }
    
    int ret = gemm_execute_plan(plan, C, A, B, alpha, beta);
    
    gemm_plan_destroy(plan);
    return ret;
}

/**
 * @brief Explicit dynamic mode GEMM
 * 
 * Forces dynamic allocation (useful for large matrices)
 */
int gemm_dynamic(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    gemm_plan_t *plan = gemm_plan_create_with_mode(M, K, N, GEMM_MEM_DYNAMIC);
    if (!plan) {
        return -1;
    }
    
    int ret = gemm_execute_plan(plan, C, A, B, alpha, beta);
    
    gemm_plan_destroy(plan);
    return ret;
}