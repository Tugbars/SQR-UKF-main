/**
 * @file gemm_planning.c
 * @brief GEMM Execution Planning with Static/Dynamic Memory Management
 *
 * This module handles:
 * - Automatic memory mode selection (static pool vs dynamic allocation)
 * - Cache-aware blocking parameter selection
 * - Pre-computation of tile descriptors and masks
 * - Workspace allocation and management
 *
 * @author TUGBARS
 * @date 2025
 */

#include "gemm_planning.h"
#include "gemm_static.h"
#include "gemm_utils.h"
#include "gemm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

//==============================================================================
// MASK GENERATION
//==============================================================================

/**
 * @brief Build AVX2 mask for partial vector (0-8 lanes)
 * 
 * CRITICAL: Builds mask directly into output pointer to avoid stack alignment issues.
 * Output pointer must be 32-byte aligned (struct member or aligned allocation).
 *
 * @param n Number of active lanes (0-8)
 * @param out Output pointer (32-byte aligned) to store mask
 */
inline void gemm_build_mask_avx2(size_t n, __m256i *out)
{
    if (n > 8) n = 8;

#if defined(__AVX2__)
    // Fast path: aligned LUT + AVX2 load via memcpy
    static const union { 
        __m256i v; 
        int32_t i[8]; 
    } lut[9] __attribute__((aligned(32))) = {
        { .i = {  0,  0,  0,  0,  0,  0,  0,  0 } },  // 0
        { .i = { -1,  0,  0,  0,  0,  0,  0,  0 } },  // 1
        { .i = { -1, -1,  0,  0,  0,  0,  0,  0 } },  // 2
        { .i = { -1, -1, -1,  0,  0,  0,  0,  0 } },  // 3
        { .i = { -1, -1, -1, -1,  0,  0,  0,  0 } },  // 4
        { .i = { -1, -1, -1, -1, -1,  0,  0,  0 } },  // 5
        { .i = { -1, -1, -1, -1, -1, -1,  0,  0 } },  // 6
        { .i = { -1, -1, -1, -1, -1, -1, -1,  0 } },  // 7
        { .i = { -1, -1, -1, -1, -1, -1, -1, -1 } }   // 8
    };
    
    // Safe unaligned-aware copy from LUT to output
    memcpy(out, &lut[n].v, sizeof(__m256i));
#else
    // Scalar fallback: no AVX executed
    int32_t tmp[8];
    for (size_t k = 0; k < 8; ++k) {
        tmp[k] = (k < n) ? -1 : 0;
    }
    memcpy(out, tmp, sizeof(__m256i));
#endif
}

/**
 * @brief Build mask pair for 16-wide panels with tail handling
 * 
 * For 16-wide panels (NR=16), we need two masks:
 * - mask_lo: controls lanes 0-7
 * - mask_hi: controls lanes 8-15
 * 
 * This function builds both masks directly into output pointers,
 * avoiding any intermediate stack-allocated __m256i variables.
 *
 * @param w Panel width (0-16)
 * @param lo Output pointer for low 8-lane mask
 * @param hi Output pointer for high 8-lane mask
 */
inline void gemm_build_mask_pair16(size_t w, __m256i *lo, __m256i *hi)
{
    if (w >= 16) {
        // Full 16 lanes active
        gemm_build_mask_avx2(8, lo);  // lo = 0xFFFFFFFF
        gemm_build_mask_avx2(8, hi);  // hi = 0xFFFFFFFF
    }
    else if (w > 8) {
        // Low 8 lanes full, high lanes partial (w-8)
        gemm_build_mask_avx2(8, lo);      // lo = full
        gemm_build_mask_avx2(w - 8, hi);  // hi = partial
    }
    else {
        // Only low lanes partial, high lanes zero
        gemm_build_mask_avx2(w, lo);      // lo = partial
        gemm_build_mask_avx2(0, hi);      // hi = zero
    }
}

//==============================================================================
// BLOCKING PARAMETER SELECTION
//==============================================================================

/**
 * @brief Select optimal cache blocking based on matrix shape
 *
 * Tuned for Intel 14900K cache hierarchy:
 * - L1D: 48KB per P-core (32KB per E-core)
 * - L2:  2MB per P-core (4MB per E-core cluster)
 * - L3:  36MB shared
 *
 * Strategy:
 * - Small matrices (< 64³): No blocking, direct execution
 * - Medium matrices (< 512³): Single-level L2 blocking
 * - Large matrices: Full three-level blocking (L1/L2/L3)
 */
/**
 * @brief Select optimal cache blocking based on matrix shape
 *
 * FIXED: Removed tiny matrix special case (Tier 1 handles that)
 */
void gemm_select_blocking(
    size_t M, size_t K, size_t N,
    size_t *MC, size_t *KC, size_t *NC,
    size_t *MR, size_t *NR)
{
    // Small matrices - minimal blocking
    if (M <= 512 && N <= 512 && K <= 512)
    {
        *MC = (M < 128) ? M : 128; // Fit in L2
        *KC = (K < 256) ? K : 256; // Balance L1/register pressure
        *NC = (N < 256) ? N : 256; // Good for L2 TLB
    }
    else
    {
        // Large matrices - full three-level blocking
        *MC = GEMM_BLOCK_MC; // 128
        *KC = (K < GEMM_BLOCK_KC) ? K : GEMM_BLOCK_KC;
        *NC = GEMM_BLOCK_NC; // 256
    }

    // Register blocking - prefer wider panels for better amortization
    if (N >= 16 && M >= 8)
    {
        *NR = 16; // Wide panel (enables 8x16, 16x16 kernels)
        *MR = (M >= 16) ? 16 : 8;
    }
    else if (N >= 8)
    {
        *NR = 8; // Standard panel (enables 8x8, 16x8 kernels)
        *MR = (M >= 16) ? 16 : 8;
    }
    else
    {
        // Narrow matrices - use 6-column kernels if possible
        *NR = (N >= 6) ? 6 : N;
        *MR = (M >= 8) ? 8 : M;
    }
}

//==============================================================================
// KERNEL SELECTION
//==============================================================================

/**
 * @brief Select appropriate kernel pair based on tile dimensions
 *
 * Selection priority (for best performance):
 * 1. 16x16 (largest register block)
 * 2. 16x8 or 8x16 (asymmetric for cache efficiency)
 * 3. 16x6 or 8x8 (standard sizes)
 * 4. 8x6 (narrow panels)
 * 5. 4x8 (tall tail handler)
 * 6. 1x8 (single-row tail handler)
 *
 * @param m_height Rows in this tile (1-16)
 * @param n_width Columns in this panel (1-16)
 * @param kern_add Output: kernel ID for accumulate mode
 * @param kern_store Output: kernel ID for store mode
 * @param kernel_width Output: 6, 8, or 16 (for mask selection)
 */
void gemm_select_kernels(
    size_t m_height, size_t n_width,
    gemm_kernel_id_t *kern_add,
    gemm_kernel_id_t *kern_store,
    int *kernel_width)
{
    // 16x16 kernels (largest register block)
    if (m_height >= 16 && n_width >= 16)
    {
        *kern_add = KERN_16x16_ADD;
        *kern_store = KERN_16x16_STORE;
        *kernel_width = 16;
        return;
    }

    // 16x8 kernels (tall panel)
    if (m_height >= 16 && n_width >= 8 && n_width < 16)
    {
        *kern_add = KERN_16x8_ADD;
        *kern_store = KERN_16x8_STORE;
        *kernel_width = 8;
        return;
    }

    // 8x16 kernels (wide panel)
    if (m_height >= 8 && m_height < 16 && n_width >= 16)
    {
        *kern_add = KERN_8x16_ADD;
        *kern_store = KERN_8x16_STORE;
        *kernel_width = 16;
        return;
    }

    // 16x6 kernels (tall, narrow panel)
    if (m_height >= 16 && n_width >= 6 && n_width < 8)
    {
        *kern_add = KERN_16x6_ADD;
        *kern_store = KERN_16x6_STORE;
        *kernel_width = 6;
        return;
    }

    // 8x8 kernels (standard square)
    if (m_height >= 8 && m_height < 16 && n_width >= 8 && n_width < 16)
    {
        *kern_add = KERN_8x8_ADD;
        *kern_store = KERN_8x8_STORE;
        *kernel_width = 8;
        return;
    }

    // 8x6 kernels (narrow panel)
    if (m_height >= 8 && m_height < 16 && n_width >= 6 && n_width < 8)
    {
        *kern_add = KERN_8x6_ADD;
        *kern_store = KERN_8x6_STORE;
        *kernel_width = 6;
        return;
    }

    // 4x8 kernels (4-row tail handler)
    if (m_height >= 4 && m_height < 8)
    {
        *kern_add = KERN_4x8_ADD;
        *kern_store = KERN_4x8_STORE;
        *kernel_width = 8;
        return;
    }

    // 1x8 kernels (1-3 row tail handler)
    *kern_add = KERN_1x8_ADD;
    *kern_store = KERN_1x8_STORE;
    *kernel_width = 8;
}

//==============================================================================
// MASK PRE-COMPUTATION (FIXED VERSION)
//==============================================================================

/**
 * @brief Pre-compute all masks for N-dimension panels
 *
 * This function performs three critical tasks in the GEMM planning phase:
 * 
 * 1. **Phase 1 - Mask Counting**: Determines how many AVX2 masks are needed
 *    for all partial-width panels. Full-width panels (j_width == NR) require
 *    no masking. For 8-wide kernels, one mask per partial panel. For 16-wide
 *    kernels, one mask if j_width ≤ 8, two masks if 8 < j_width < 16.
 * 
 * 2. **Phase 2 - Storage Allocation**: Allocates contiguous, 32-byte aligned
 *    storage for all masks. Contiguous storage improves cache locality and
 *    eliminates pointer chasing during kernel execution.
 * 
 * 3. **Phase 3 - Mask Generation**: Builds AVX2 masks for each partial panel
 *    and stores them in the contiguous array. Each panel descriptor stores
 *    mask values directly for fast kernel dispatch.
 * 
 * **Masking Strategy:**
 * - 8-wide panels (NR ≤ 8): One mask controls lanes [0:7]
 * - 16-wide panels (NR = 16):
 *   - j_width ≤ 8: mask_lo partial, mask_hi zero (1 mask stored)
 *   - 8 < j_width < 16: mask_lo full, mask_hi partial (2 masks stored)
 *   - j_width = 16: No masking needed (0 masks stored)
 * 
 * **Alignment Requirements:**
 * - panel_info_t struct must be 32-byte aligned
 * - npanels array must be allocated with gemm_aligned_alloc(32, ...)
 * - mask_storage must be 32-byte aligned (gemm_aligned_alloc guarantees this)
 * 
 * @param[in,out] plan GEMM execution plan with allocated panel descriptor array
 *                      and populated N, NR dimensions. On success, npanels[],
 *                      mask_storage, and n_masks fields are populated.
 * 
 * @return 0 on success, -1 on failure
 * 
 * @retval 0  Success - all masks computed and stored
 * @retval -1 Failure - mask allocation failed OR mask count mismatch
 * 
 * @pre plan != NULL
 * @pre plan->npanels != NULL (allocated with gemm_aligned_alloc)
 * @pre plan->N and plan->NR are valid and non-zero
 * 
 * @post plan->mask_storage allocated and populated (if n_masks > 0)
 * @post plan->n_masks set to number of masks stored
 * @post All panels have j_start, j_width, needs_mask, mask_lo, mask_hi set
 * 
 * @note This function is called during gemm_plan_create() after panel
 *       descriptor array allocation but before workspace setup.
 * 
 * @note For AVX2 safety, mask generation uses memcpy-based stores to
 *       handle any potential alignment issues gracefully.
 * 
 * @see gemm_build_mask_avx2() for single mask generation
 * @see gemm_build_mask_pair16() for dual mask generation (16-wide panels)
 * 
 * @author TUGBARS
 * @date 2025
 */
static int precompute_panel_masks(gemm_plan_t *plan)
{
    const size_t n_panels = (plan->N + plan->NR - 1) / plan->NR;
    
    //==========================================================================
    // Phase 1: Count masks needed for all panels
    //==========================================================================
    // Iterate through all panels to determine total mask storage requirement.
    // This two-pass approach (count then allocate) enables single contiguous
    // allocation rather than per-panel allocations, improving cache behavior.
    //
    // Mask counting rules:
    // - Full-width panels (j_width == NR): 0 masks (no masking needed)
    // - 8-wide partial (NR ≤ 8, j_width < NR): 1 mask
    // - 16-wide partial (NR = 16, j_width ≤ 8): 1 mask (lo only)
    // - 16-wide partial (NR = 16, 8 < j_width < 16): 2 masks (lo + hi)
    
    size_t n_masks_needed = 0;
    
    for (size_t p = 0; p < n_panels; ++p) {
        // Calculate this panel's starting column and actual width
        const size_t j_start = p * plan->NR;
        const size_t j_width = (j_start + plan->NR <= plan->N)
                                 ? plan->NR                    // Full panel
                                 : (plan->N - j_start);        // Tail panel
        
        // Only partial-width panels require masking
        if (j_width < plan->NR) {
            if (plan->NR <= 8) {
                // 8-wide or narrower: single mask
                n_masks_needed += 1;
            } else if (plan->NR == 16) {
                // 16-wide: one or two masks depending on width
                // j_width ∈ [1, 8]:  mask_lo partial, mask_hi zero → 1 mask
                // j_width ∈ [9, 15]: mask_lo full, mask_hi partial → 2 masks
                n_masks_needed += (j_width <= 8) ? 1 : 2;
            }
        }
    }
    
    //==========================================================================
    // Phase 2: Allocate contiguous mask storage
    //==========================================================================
    // Allocate 32-byte aligned storage for all masks in one contiguous block.
    // This improves:
    // - Cache locality (masks stored sequentially)
    // - Memory efficiency (single allocation vs per-panel allocations)
    // - Access pattern (predictable stride for prefetchers)
    //
    // If no masks are needed (all full-width panels), skip allocation entirely.
    
    plan->n_masks = n_masks_needed;
    plan->mask_storage = NULL;
    
    if (n_masks_needed > 0) {
        plan->mask_storage = (__m256i*)gemm_aligned_alloc(
            32,                              // Alignment: 32 bytes (AVX2 requirement)
            n_masks_needed * sizeof(__m256i) // Size: total masks needed
        );
        
        if (!plan->mask_storage) {
            return -1;  // Allocation failed - critical error
        }
    }
    
    //==========================================================================
    // Phase 3: Populate panel descriptors and store masks
    //==========================================================================
    // For each panel, compute:
    // 1. Starting column (j_start) and actual width (j_width)
    // 2. Whether masking is needed (needs_mask flag)
    // 3. Mask values (mask_lo, mask_hi) for partial panels
    // 4. Store masks in contiguous array for cache-friendly access
    //
    // The mask_idx counter tracks current position in mask_storage array.
    // At function end, mask_idx must equal n_masks_needed (sanity check).
    
    size_t mask_idx = 0;  // Current position in mask_storage array
    
    for (size_t p = 0; p < n_panels; p++) {
        panel_info_t *panel = &plan->npanels[p];
        
        //----------------------------------------------------------------------
        // Step 3.1: Calculate panel dimensions
        //----------------------------------------------------------------------
        panel->j_start = p * plan->NR;
        panel->j_width = (panel->j_start + plan->NR <= plan->N)
                           ? plan->NR                    // Full panel
                           : (plan->N - panel->j_start); // Tail panel
        
        //----------------------------------------------------------------------
        // Step 3.2: Check if masking is required
        //----------------------------------------------------------------------
        if (panel->j_width == plan->NR) {
            // Full-width panel - no masking needed
            // Kernels will use unmasked stores/loads (fastest path)
            panel->needs_mask = 0;
            
            // mask_lo and mask_hi remain uninitialized (never accessed when
            // needs_mask == 0). This avoids unnecessary AVX2 instructions on
            // systems without AVX2 support.
            continue;
        }
        
        //----------------------------------------------------------------------
        // Step 3.3: Partial-width panel - build and store masks
        //----------------------------------------------------------------------
        panel->needs_mask = 1;
        
        if (plan->NR <= 8) {
            //------------------------------------------------------------------
            // 8-wide or narrower panels
            //------------------------------------------------------------------
            // Only one mask needed for lanes [0:7]. The mask_hi field is set
            // to zero but never used (8-wide kernels only read mask_lo).
            
            gemm_build_mask_avx2(panel->j_width, &panel->mask_lo);  // Build partial mask
            gemm_build_mask_avx2(0, &panel->mask_hi);                // Build zero mask (unused)
            
            // Store mask in contiguous array for cache locality
            // memcpy is safe for any alignment (source and dest are both aligned)
            memcpy(&plan->mask_storage[mask_idx++], &panel->mask_lo, sizeof(__m256i));
        }
        else if (plan->NR == 16) {
            //------------------------------------------------------------------
            // 16-wide panels
            //------------------------------------------------------------------
            // Need dual masks: mask_lo controls lanes [0:7], mask_hi controls
            // lanes [8:15]. The gemm_build_mask_pair16() helper builds both
            // masks simultaneously based on panel width.
            
            gemm_build_mask_pair16(panel->j_width, &panel->mask_lo, &panel->mask_hi);
            
            if (panel->j_width <= 8) {
                //--------------------------------------------------------------
                // Width 1-8: Only low lanes active, high lanes zero
                //--------------------------------------------------------------
                // Store only mask_lo (mask_hi is all zeros, not stored)
                memcpy(&plan->mask_storage[mask_idx++], &panel->mask_lo, sizeof(__m256i));
            } else {
                //--------------------------------------------------------------
                // Width 9-15: Low lanes full, high lanes partial
                //--------------------------------------------------------------
                // Store both masks:
                // - mask_lo is full (all 8 lanes active)
                // - mask_hi is partial (width-8 lanes active)
                memcpy(&plan->mask_storage[mask_idx++], &panel->mask_lo, sizeof(__m256i));
                memcpy(&plan->mask_storage[mask_idx++], &panel->mask_hi, sizeof(__m256i));
            }
        }
    }
    
    //==========================================================================
    // Sanity check: Verify mask count consistency
    //==========================================================================
    // This catches logic errors between Phase 1 (counting) and Phase 3 (storing).
    // If mask_idx != n_masks_needed, there's a bug in the counting logic.
    //
    // Common failure modes:
    // - Incorrect width threshold in Phase 1 vs Phase 3
    // - Missing mask store in Phase 3
    // - Extra mask store in Phase 3
    //
    // This check has zero runtime cost in production (always true if code is
    // correct) but catches critical bugs during development.
    
    if (mask_idx != n_masks_needed) {
        // INTERNAL ERROR: Mask count mismatch
        // This should never happen if logic is correct
        return -1;
    }
    
    return 0;  // Success - all masks computed and stored
}

//==============================================================================
// TILE PRE-COMPUTATION
//==============================================================================

/**
 * @brief Pre-compute all M-tile descriptors
 *
 * FIXED: Removed kernel pre-selection (now done at execution time)
 * Each tile only stores dimension info, kernels selected dynamically.
 */
static void precompute_mtiles(gemm_plan_t *plan)
{
    size_t n_mtiles = (plan->M + plan->MR - 1) / plan->MR;

    for (size_t t = 0; t < n_mtiles; t++)
    {
        tile_info_t *tile = &plan->mtiles[t];

        tile->i_start = t * plan->MR;
        tile->i_height = (tile->i_start + plan->MR <= plan->M)
                             ? plan->MR
                             : (plan->M - tile->i_start);

        // FIXED: No kernel pre-selection
        // Kernels will be selected at execution time based on actual (m_block, n_block)
        tile->kern_add = KERN_INVALID;     // Placeholder
        tile->kern_store = KERN_INVALID;   // Placeholder
        tile->kernel_width = 0;            // Determined at runtime
    }
}

//==============================================================================
// WORKSPACE QUERY (FIXED VERSION)
//==============================================================================

/**
 * @brief Calculate exact workspace size needed for dynamic allocation
 *
 * Workspace consists of three buffers with DIFFERENT sizes:
 * - pack_a:  MC × KC floats (packed A tiles)
 * - pack_b:  KC × NC floats (packed B panels)
 * - temp:    MC × NC floats (for symmetric operations like A*B*A^T)
 *
 * @return Total bytes needed, rounded to 64-byte boundaries
 */
size_t gemm_workspace_query(size_t M, size_t K, size_t N)
{
    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(M, K, N, &MC, &KC, &NC, &MR, &NR);

    // Calculate exact sizes for each buffer
    size_t a_size = MC * KC * sizeof(float);
    size_t b_size = KC * NC * sizeof(float);
    size_t temp_size = MC * NC * sizeof(float);

    // Round each to 64-byte boundaries (cache line alignment)
    a_size = (a_size + 63) & ~(size_t)63;
    b_size = (b_size + 63) & ~(size_t)63;
    temp_size = (temp_size + 63) & ~(size_t)63;

    return a_size + b_size + temp_size;
}

//==============================================================================
// PLAN CREATION (FIXED VERSION)
//==============================================================================

/**
 * @brief Create execution plan with automatic memory mode selection
 *
 * Selects static mode if dimensions fit (M,K,N ≤ GEMM_STATIC_MAX_DIM),
 * otherwise falls back to dynamic allocation.
 */
gemm_plan_t *gemm_plan_create(size_t M, size_t K, size_t N)
{
    // FIXED: Validate dimensions (reject zero)
    if (M == 0 || K == 0 || N == 0) {
        return NULL;
    }

    gemm_memory_mode_t mode = gemm_fits_static(M, K, N)
                                  ? GEMM_MEM_STATIC
                                  : GEMM_MEM_DYNAMIC;

    return gemm_plan_create_with_mode(M, K, N, mode);
}

/**
 * @brief Create execution plan with explicit memory mode
 *
 * @param mode GEMM_MEM_STATIC or GEMM_MEM_DYNAMIC
 * @return Plan pointer or NULL if:
 *         - Allocation fails
 *         - Static mode requested but dimensions too large
 */
gemm_plan_t *gemm_plan_create_with_mode(
    size_t M, size_t K, size_t N,
    gemm_memory_mode_t mode)
{
    //--------------------------------------------------------------------------
    // Validate static mode request
    //--------------------------------------------------------------------------
    if (mode == GEMM_MEM_STATIC && !gemm_fits_static(M, K, N))
    {
        return NULL;
    }

    //--------------------------------------------------------------------------
    // Allocate plan structure
    //--------------------------------------------------------------------------
    gemm_plan_t *plan = (gemm_plan_t *)calloc(1, sizeof(gemm_plan_t));
    if (!plan)
    {
        printf("gemm_plan_create: Failed to allocate plan\n");
        return NULL;
    }

    //--------------------------------------------------------------------------
    // Store dimensions and memory mode
    //--------------------------------------------------------------------------
    plan->M = M;
    plan->K = K;
    plan->N = N;
    plan->mem_mode = mode;

    //--------------------------------------------------------------------------
    // Select blocking parameters (tuned for i14900K)
    //--------------------------------------------------------------------------
    gemm_select_blocking(M, K, N,
                         &plan->MC, &plan->KC, &plan->NC,
                         &plan->MR, &plan->NR);

    //--------------------------------------------------------------------------
    // Compute tile counts
    //--------------------------------------------------------------------------
    plan->n_mtiles = (M + plan->MR - 1) / plan->MR;
    plan->n_npanels = (N + plan->NR - 1) / plan->NR;
    plan->n_ktiles = (K + plan->KC - 1) / plan->KC;

    //--------------------------------------------------------------------------
    // Allocate tile and panel descriptor arrays
    //--------------------------------------------------------------------------
    plan->mtiles = (tile_info_t *)calloc(plan->n_mtiles, sizeof(tile_info_t));  // ✅ OK (no __m256i)


    plan->npanels = (panel_info_t *)gemm_aligned_alloc(
    32, plan->n_npanels * sizeof(panel_info_t));  // ✅ 32-byte aligned

    if (!plan->mtiles || !plan->npanels)
    {
        printf("gemm_plan_create: Failed to allocate tile arrays\n");
        gemm_plan_destroy(plan);
        return NULL;
    }

    memset(plan->npanels, 0, plan->n_npanels * sizeof(panel_info_t));

    //--------------------------------------------------------------------------
    // Pre-compute tile descriptors (kernels, heights)
    //--------------------------------------------------------------------------
    precompute_mtiles(plan);

    //--------------------------------------------------------------------------
    // Pre-compute panel descriptors and masks
    //--------------------------------------------------------------------------
    if (precompute_panel_masks(plan) != 0)
    {
        printf("gemm_plan_create: Failed to compute masks\n");
        gemm_plan_destroy(plan);
        return NULL;
    }

    //--------------------------------------------------------------------------
    // Setup Workspace (STATIC or DYNAMIC)
    //--------------------------------------------------------------------------

    if (mode == GEMM_MEM_STATIC)
    {
        //----------------------------------------------------------------------
        // STATIC MODE: Use thread-local pool (ZERO allocation!)
        //----------------------------------------------------------------------
        gemm_static_init(); // Ensure pool is initialized

        /*

        workspace[0]              workspace[MC*KC]            workspace[end]
        │                           │                          │
        ├───────────────────────────┼──────────────────────────┤
        │   workspace_a (MC×KC)     │  workspace_b (KC×NC)     │
        └───────────────────────────┴──────────────────────────┘
        //This is safe because MC×KC + KC×NC ≤ MAX_DIM² for static-eligible matrices.
        */

        plan->workspace_a = gemm_static_pool.workspace;
        plan->workspace_b = gemm_static_pool.workspace + (plan->MC * plan->KC);
        plan->workspace_temp = gemm_static_pool.workspace;
        plan->workspace_size = 0;    // Not allocated (points to static pool)
        plan->workspace_aligned = 1; // Always true for static
    }
    else
    {
        //----------------------------------------------------------------------
        // DYNAMIC MODE: Allocate aligned memory (CRITICAL FIX)
        //----------------------------------------------------------------------

        // Calculate EXACT sizes for each buffer (not divided by 3!)
        size_t a_size = plan->MC * plan->KC * sizeof(float);
        size_t b_size = plan->KC * plan->NC * sizeof(float);
        size_t temp_size = plan->MC * plan->NC * sizeof(float);

        // Round each to 64-byte boundaries for cache line alignment
        a_size = (a_size + 63) & ~(size_t)63;
        b_size = (b_size + 63) & ~(size_t)63;
        temp_size = (temp_size + 63) & ~(size_t)63;

        plan->workspace_size = a_size + b_size + temp_size;

        // Allocate each buffer separately with correct size
        plan->workspace_a = (float *)gemm_aligned_alloc(64, a_size);
        plan->workspace_b = (float *)gemm_aligned_alloc(64, b_size);
        plan->workspace_temp = (float *)gemm_aligned_alloc(64, temp_size);

        if (!plan->workspace_a || !plan->workspace_b || !plan->workspace_temp)
        {
            
            gemm_plan_destroy(plan);
            return NULL;
        }

        plan->workspace_aligned = 1; // gemm_aligned_alloc guarantees this
    }

    //--------------------------------------------------------------------------
    // Initialize packing function pointers (set by execution module)
    //--------------------------------------------------------------------------
    plan->pack_a_fn = NULL; // Will be set in gemm_large.c
    plan->pack_b_fn = NULL;

    return plan;
}

//==============================================================================
// PLAN DESTRUCTION (FIXED VERSION)
//==============================================================================

/**
 * @brief Destroy plan and free resources
 *
 * CRITICAL: Only free workspace for DYNAMIC mode!
 * Static mode workspace points to thread-local pool - must NOT be freed.
 */
void gemm_plan_destroy(gemm_plan_t *plan)
{
    if (!plan)
        return;

    //==========================================================================
    // REGULAR free() - Allocated with calloc()
    //==========================================================================
    free(plan->mtiles);      // ✅ calloc() → free()
    gemm_aligned_free(plan->npanels);      // ✅ FIXED: aligned_alloc → aligned_free

    //==========================================================================
    // gemm_aligned_free() - Allocated with gemm_aligned_alloc()
    //==========================================================================
    if (plan->mask_storage)
        gemm_aligned_free(plan->mask_storage);  // ✅ gemm_aligned_alloc() → gemm_aligned_free()

    //==========================================================================
    // DYNAMIC mode only: gemm_aligned_free()
    //==========================================================================
    if (plan->mem_mode == GEMM_MEM_DYNAMIC)
    {
        if (plan->workspace_a)
            gemm_aligned_free(plan->workspace_a);  // ✅ gemm_aligned_alloc() → gemm_aligned_free()
        if (plan->workspace_b)
            gemm_aligned_free(plan->workspace_b);  // ✅ gemm_aligned_alloc() → gemm_aligned_free()
        if (plan->workspace_temp)
            gemm_aligned_free(plan->workspace_temp);  // ✅ gemm_aligned_alloc() → gemm_aligned_free()
    }
    // STATIC mode: Do NOT free workspace (points to static pool)

    //==========================================================================
    // REGULAR free() - Allocated with calloc()
    //==========================================================================
    free(plan);  // ✅ calloc() → free()
}