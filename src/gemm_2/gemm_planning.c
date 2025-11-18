/**
 * @file gemm_planning.c
 * @brief GEMM Execution Planning - Adaptive Blocking and Workspace Management
 *
 * This module implements the planning layer that analyzes matrix dimensions
 * and pre-computes all metadata needed for efficient GEMM execution:
 * 
 * **Planning Phase (one-time cost):**
 * - Adaptive blocking parameter selection based on aspect ratios
 * - Tile count pre-computation (eliminates division in hot path)
 * - Kernel pre-selection for full tiles (eliminates dispatch overhead)
 * - Panel descriptor generation (just dimensions, no masks)
 * - Workspace allocation (static or dynamic)
 * 
 * **Execution Phase (amortized):**
 * - Plans can be reused for multiple executions with same dimensions
 * - All metadata lookups are O(1) array accesses
 * - No conditional branching in inner loops
 * 
 * **Cache Optimization Strategy:**
 * The blocking parameters are tuned to maximize cache reuse:
 * - MC × KC fits in L1 cache (~48 KB on Intel 14900K)
 * - KC × NC fits in L2 cache (~2 MB per core)
 * - NC → KC → MC loop order maximizes B panel reuse
 * 
 * @author TUGBARS
 * @date 2025
 */

#include "gemm_planning.h"
#include "gemm_static.h"
#include "gemm_utils.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

//==============================================================================
// BLOCKING PARAMETER SELECTION
//==============================================================================

/** @brief MC parameter for tall matrices (M >> N)
 *  @details Larger MC amortizes packing cost when M is large */
static const size_t ADAPTIVE_MC_TALL = 256;

/** @brief MC parameter for small/wide matrices (M << N)
 *  @details Smaller MC reduces memory footprint when M is small */
static const size_t ADAPTIVE_MC_SMALL = 64;

/** @brief KC parameter for tall matrices
 *  @details Moderate KC balances cache usage for tall shapes */
static const size_t ADAPTIVE_KC_TALL = 128;

/** @brief KC parameter for deep matrices (K >> M,N)
 *  @details Large KC reduces packing overhead when K is large */
static const size_t ADAPTIVE_KC_DEEP = 512;

/** @brief NC parameter for wide matrices (N >> M)
 *  @details Large NC amortizes B packing cost when N is large */
static const size_t ADAPTIVE_NC_WIDE = 512;

/** @brief NC parameter for small/tall matrices (N << M)
 *  @details Smaller NC reduces memory footprint when N is small */
static const size_t ADAPTIVE_NC_SMALL = 128;

/** @brief Minimum KC to ensure reasonable blocking
 *  @details Below this, packing overhead dominates computation */
static const size_t ADAPTIVE_KC_MIN = 32;

/**
 * @brief Select adaptive blocking parameters based on matrix dimensions
 * 
 * **Algorithm:**
 * 
 * 1. **Aspect Ratio Analysis:**
 *    - Compute M/N and K/N ratios to classify matrix shape
 *    - Tall (M >> N): Optimize for row reuse
 *    - Wide (N >> M): Optimize for column reuse
 *    - Deep (K >> M,N): Optimize for reduction dimension
 *    - Balanced: Use default tuned parameters
 * 
 * 2. **Initial Parameter Selection:**
 *    - Choose MC, KC, NC based on aspect ratio
 *    - Select MR, NR based on available kernels and dimensions
 * 
 * 3. **Dimension Clamping:**
 *    - Ensure MC ≤ M, KC ≤ K, NC ≤ N (no oversized blocks)
 * 
 * 4. **L2 Cache Fitting:**
 *    - Target: (MC×KC + KC×NC) × 4 bytes ≤ 1.8 MB
 *    - If exceeds: Scale down proportionally while maintaining alignment
 * 
 * 5. **Sanity Checks:**
 *    - Ensure MC ≥ MR, NC ≥ NR, KC ≥ 8
 *    - If violated: Reset to minimum safe values
 * 
 * **Aspect Ratio Thresholds:**
 * - Tall: M/N > 3.0
 * - Wide: M/N < 0.33
 * - Deep: K/N > 4.0
 * 
 * @param[in]  M  Number of rows in A and C
 * @param[in]  K  Number of columns in A, rows in B
 * @param[in]  N  Number of columns in B and C
 * @param[out] MC Output: M-dimension cache block size
 * @param[out] KC Output: K-dimension cache block size
 * @param[out] NC Output: N-dimension cache block size
 * @param[out] MR Output: M-dimension register block size (8 or 16)
 * @param[out] NR Output: N-dimension register block size (6, 8, or 16)
 * 
 * @note All output parameters must be non-NULL
 * @note Parameters are quantized to multiples of MR/NR for alignment
 * @note L2 cache target (1.8 MB) leaves headroom for code and stack
 * 
 * @see gemm_plan_create()
 */
void gemm_select_blocking(
    size_t M, size_t K, size_t N,
    size_t *MC, size_t *KC, size_t *NC,
    size_t *MR, size_t *NR)
{
    // Compute aspect ratios for shape classification
    double aspect_mn = (double)M / (double)N;
    double aspect_kn = (double)K / (double)N;
    
    //--------------------------------------------------------------------------
    // Case 1: Tall Matrices (M >> N)
    //--------------------------------------------------------------------------
    if (aspect_mn > 3.0) {
        *MC = ADAPTIVE_MC_TALL;      // 256 rows at a time
        *KC = ADAPTIVE_KC_TALL;      // Moderate K blocking
        *NC = MIN(N, ADAPTIVE_NC_SMALL); // Small N panels - ALREADY CLAMPED HERE
        *MR = 16;                     // Use tallest kernels
        *NR = (N >= 16) ? 16 : ((N >= 8) ? 8 : 6); // Adapt to N
    }
    //--------------------------------------------------------------------------
    // Case 2: Wide Matrices (N >> M)
    //--------------------------------------------------------------------------
    else if (aspect_mn < 0.33) {
        *MC = MIN(M, ADAPTIVE_MC_SMALL); // Don't exceed M
        *KC = ADAPTIVE_KC_TALL;
        *NC = ADAPTIVE_NC_WIDE;          // 512 columns at a time
        *MR = (M >= 16) ? 16 : ((M >= 8) ? 8 : 4); // Adapt to M
        *NR = 16;                         // Use widest kernels
    }
    //--------------------------------------------------------------------------
    // Case 3: Deep Matrices (K >> M,N)
    //--------------------------------------------------------------------------
    else if (aspect_kn > 4.0) {
        *MC = MIN(M, ADAPTIVE_MC_SMALL);
        *KC = ADAPTIVE_KC_DEEP;          // 512 K-blocks
        *NC = MIN(N, ADAPTIVE_NC_SMALL);
        *MR = (M >= 8) ? 8 : 4;
        *NR = (N >= 8) ? 8 : 6;
    }
    //--------------------------------------------------------------------------
    // Case 4: Balanced Matrices
    //--------------------------------------------------------------------------
    else {
        *MC = GEMM_BLOCK_MC; // 128
        *KC = GEMM_BLOCK_KC; // 256
        *NC = GEMM_BLOCK_NC; // 256
        
        // Select register blocking based on available dimensions
        if (N >= 16 && M >= 8) {
            *NR = 16;
            *MR = (M >= 16) ? 16 : 8;
        } else if (N >= 8) {
            *NR = 8;
            *MR = (M >= 16) ? 16 : 8;
        } else {
            *NR = (N >= 6) ? 6 : N;
            *MR = (M >= 8) ? 8 : M;
        }
    }
    
    //--------------------------------------------------------------------------
    // Clamp to actual matrix dimensions
    //--------------------------------------------------------------------------
    *MC = MIN(*MC, M);
    *KC = MIN(*KC, K);
    *NC = MIN(*NC, N);
    
    //--------------------------------------------------------------------------
    // L2 Cache Fitting (Target: 1.8 MB total workspace)
    // Workspace = (MC × KC + KC × NC) × sizeof(float)
    //--------------------------------------------------------------------------
    const size_t L2_TARGET = 1800 * 1024; // Bytes
    size_t workspace_bytes = (*MC * *KC + *KC * *NC) * sizeof(float);
    
    if (workspace_bytes > L2_TARGET) {
        // Scale down by sqrt to maintain aspect ratio
        double scale = sqrt((double)L2_TARGET / (double)workspace_bytes);
        
        // Apply scaling with alignment preservation
        *MC = ((*MC * scale) / *MR) * *MR; // Round down to MR multiple
        *KC = ((*KC * scale) / 8) * 8;     // Round down to 8 multiple
        *NC = ((*NC * scale) / *NR) * *NR; // Round down to NR multiple
        
        // Ensure minimum safe values
        *MC = MAX(*MC, *MR);
        *KC = MAX(*KC, ADAPTIVE_KC_MIN);
        *NC = MAX(*NC, MIN(*NR, N));
    }
    
    //--------------------------------------------------------------------------
    // Final Sanity Checks
    // Ensure blocks are large enough for at least one micro-kernel tile
    //--------------------------------------------------------------------------
    if (*MC < *MR || *NC < *NR || *KC < 8) {
        *MC = *MR;
        *KC = ADAPTIVE_KC_MIN;
        *NC = *NR;
    }


     // ADD THIS: Final clamp to ensure we never exceed matrix dimensions
    *MC = MIN(*MC, M);
    *KC = MIN(*KC, K);
    *NC = MIN(*NC, N);
}

//==============================================================================
// KERNEL SELECTION
//==============================================================================

/**
 * @brief Select optimal micro-kernel for given tile dimensions
 * 
 * **Selection Algorithm:**
 * 
 * Uses a decision tree to select the largest kernel that fits the tile:
 * 1. Try 16×16 (composite)
 * 2. Try 16×8
 * 3. Try 8×16
 * 4. Try 16×6 (composite)
 * 5. Try 8×8
 * 6. Try 8×6
 * 7. Try 4×8
 * 8. Fallback to 1×8 (always fits)
 * 
 * **Kernel Characteristics:**
 * 
 * | Kernel | Registers | Notes |
 * |--------|-----------|-------|
 * | 16×16  | 16 acc + 4 temp = 20 | Composite (2× 8×16 calls) |
 * | 16×8   | 16 acc + 4 temp = 20 | Composite (2× 8×8 calls) |
 * | 8×16   | 16 acc + 4 temp = 20 | Native, fast path |
 * | 8×8    | 8 acc + 3 temp = 11  | Native, most common |
 * | 8×6    | 6 acc + 2 temp = 8   | Native, for N=6 QR blocks |
 * | 4×8    | 4 acc + 2 temp = 6   | Native, for small M |
 * | 1×8    | 1 acc + 1 temp = 2   | Native, edge rows |
 * 
 * **Composite Kernels:**
 * - 16×16: Calls 8×16 twice (rows 0-7, then 8-15)
 * - 16×8: Calls 8×8 twice (rows 0-7, then 8-15)
 * - 16×6: Calls 8×6 twice (rows 0-7, then 8-15)
 * 
 * @param[in]  m_height     Tile height (number of rows)
 * @param[in]  n_width      Tile width (number of columns)
 * @param[out] kern_add     Output: Kernel ID for ADD mode (C += A*B)
 * @param[out] kern_store   Output: Kernel ID for STORE mode (C = A*B)
 * @param[out] kernel_width Output: Native width of selected kernel
 * 
 * @note All output parameters must be non-NULL
 * @note Partial tiles (m < kernel_m or n < kernel_n) handled via scalar loops
 * @note ADD vs STORE: STORE used on first K-tile when beta=0, ADD thereafter
 * 
 * @see gemm_kernel_id_t
 * @see dispatch_kernel() in gemm_large.c
 */
void gemm_select_kernels(
    size_t m_height, size_t n_width,
    gemm_kernel_id_t *kern_add,
    gemm_kernel_id_t *kern_store,
    int *kernel_width)
{
    // Priority 1: 16×16 (largest, composite)
    if (m_height >= 16 && n_width >= 16) {
        *kern_add = KERN_16x16_ADD;
        *kern_store = KERN_16x16_STORE;
        *kernel_width = 16;
        return;
    }

    // Priority 2: 16×8 (tall tile, composite)
    if (m_height >= 16 && n_width >= 8 && n_width < 16) {
        *kern_add = KERN_16x8_ADD;
        *kern_store = KERN_16x8_STORE;
        *kernel_width = 8;
        return;
    }

    // Priority 3: 8×16 (wide tile, native)
    if (m_height >= 8 && m_height < 16 && n_width >= 16) {
        *kern_add = KERN_8x16_ADD;
        *kern_store = KERN_8x16_STORE;
        *kernel_width = 16;
        return;
    }

    // Priority 4: 16×6 (for QR with N=6 blocks, composite)
    if (m_height >= 16 && n_width >= 6 && n_width < 8) {
        *kern_add = KERN_16x6_ADD;
        *kern_store = KERN_16x6_STORE;
        *kernel_width = 6;
        return;
    }

    // Priority 5: 8×8 (most common, native)
    if (m_height >= 8 && m_height < 16 && n_width >= 8 && n_width < 16) {
        *kern_add = KERN_8x8_ADD;
        *kern_store = KERN_8x8_STORE;
        *kernel_width = 8;
        return;
    }

    // Priority 6: 8×6 (for QR with N=6 blocks, native)
    if (m_height >= 8 && m_height < 16 && n_width >= 6 && n_width < 8) {
        *kern_add = KERN_8x6_ADD;
        *kern_store = KERN_8x6_STORE;
        *kernel_width = 6;
        return;
    }

    // Priority 7: 4×8 (short tile, native)
    if (m_height >= 4 && m_height < 8) {
        *kern_add = KERN_4x8_ADD;
        *kern_store = KERN_4x8_STORE;
        *kernel_width = 8;
        return;
    }

    // Fallback: 1×8 (single row, always fits)
    *kern_add = KERN_1x8_ADD;
    *kern_store = KERN_1x8_STORE;
    *kernel_width = 8;
}

//==============================================================================
// PANEL PRE-COMPUTATION (SIMPLIFIED - NO MASKS!)
//==============================================================================

/**
 * @brief Pre-compute panel descriptors for N-dimension
 * 
 * Divides the N-dimension into vertical panels of width NR.
 * Each panel is packed once per KC-tile and reused across all MC-tiles,
 * maximizing L2 cache efficiency.
 * 
 * **Panel Layout:**
 * ```
 * N-dimension: [0 ... NR-1][NR ... 2*NR-1][...][last_panel]
 *              └─ panel 0 ─┘└─── panel 1 ──┘    └─ panel p ─┘
 * ```
 * 
 * The last panel may have width < NR (edge case).
 * 
 * **No Masks Needed:**
 * - Old design: Pre-computed AVX2 masks for partial widths
 * - New design: Safe scalar loops handle edges (2-5% overhead)
 * - Benefit: No undefined behavior, portable, debuggable
 * 
 * @param plan Plan to populate with panel descriptors
 * 
 * @pre plan->npanels must be allocated with n_npanels elements
 * @pre plan->N, plan->NR must be set
 * 
 * @note This function is called during plan creation, not execution
 * @note Panel count = (N + NR - 1) / NR (ceiling division)
 * 
 * @see gemm_plan_create_with_mode()
 */
static void precompute_panels(gemm_plan_t *plan)
{
    size_t n_panels = (plan->max_N + plan->NR - 1) / plan->NR;

    for (size_t p = 0; p < n_panels; p++)
    {
        panel_info_t *panel = &plan->npanels[p];
        
        panel->j_start = p * plan->NR;
        
        panel->j_width = (panel->j_start + plan->NR <= plan->max_N)
                            ? plan->NR
                            : (plan->max_N - panel->j_start);
    }
}

//==============================================================================
// WORKSPACE QUERY
//==============================================================================

/**
 * @brief Query workspace size required for given matrix dimensions
 * 
 * Computes the total workspace needed without creating a plan.
 * Useful for:
 * - Memory budget analysis
 * - Pre-allocation strategies
 * - Checking if static workspace is sufficient
 * 
 * **Workspace Components:**
 * 1. **Packed A**: MC × KC × 4 bytes (one M-panel)
 * 2. **Packed B**: KC × NC × 4 bytes (one N-panel set)
 * 3. **Temp buffer**: MC × NC × 4 bytes (for edge tile handling)
 * 
 * Each component is 64-byte aligned for cache line alignment.
 * 
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * 
 * @return Required workspace size in bytes (including alignment padding)
 * 
 * @note Actual allocation may be slightly larger due to allocator overhead
 * @note For static workspace, check against GEMM_STATIC_WORKSPACE_SIZE
 * 
 * @see gemm_plan_create()
 * @see gemm_fits_static()
 */
size_t gemm_workspace_query(size_t M, size_t K, size_t N)
{
    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(M, K, N, &MC, &KC, &NC, &MR, &NR);

    // Component sizes (unaligned)
    size_t a_size = MC * KC * sizeof(float);
    size_t b_size = KC * NC * sizeof(float);
    size_t temp_size = MC * NC * sizeof(float);

    // Align to 64-byte boundaries (cache line size)
    a_size = (a_size + 63) & ~(size_t)63;
    b_size = (b_size + 63) & ~(size_t)63;
    temp_size = (temp_size + 63) & ~(size_t)63;

    return a_size + b_size + temp_size;
}

//==============================================================================
// PLAN CREATION
//==============================================================================

/**
 * @brief Create an execution plan with automatic memory mode selection
 * 
 * **Planning Algorithm:**
 * 
 * 1. Check if dimensions fit in static workspace
 * 2. Allocate plan structure
 * 3. Select adaptive blocking parameters (aspect-ratio based)
 * 4. Pre-compute tile counts (eliminates division in execution)
 * 5. Pre-select kernels for full tiles (eliminates dispatch overhead)
 * 6. Allocate and initialize panel descriptors
 * 7. Allocate workspace (static or dynamic)
 * 
 * **Memory Mode Selection:**
 * - Static: If MC×KC + KC×NC ≤ GEMM_STATIC_WORKSPACE_SIZE
 * - Dynamic: Otherwise (allocates from heap)
 * 
 * **Plan Reusability:**
 * Plans with the same (M, K, N) are identical, so a single plan
 * can be created and reused for thousands of executions.
 * 
 * @param M Number of rows in A and C (must be > 0)
 * @param K Number of columns in A, rows in B (must be > 0)
 * @param N Number of columns in B and C (must be > 0)
 * 
 * @return Pointer to allocated plan, or NULL on failure
 * 
 * @retval NULL if any dimension is zero
 * @retval NULL if allocation fails
 * @retval valid plan pointer on success
 * 
 * @note The returned plan must be freed with gemm_plan_destroy()
 * @note Plans are NOT thread-safe (execution is, but planning is not)
 * 
 * @see gemm_execute_plan()
 * @see gemm_plan_destroy()
 * @see gemm_plan_create_with_mode()
 */
gemm_plan_t *gemm_plan_create(size_t M, size_t K, size_t N)
{
    if (M == 0 || K == 0 || N == 0)
        return NULL;

    // Auto-select memory mode based on workspace size
    gemm_memory_mode_t mode = gemm_fits_static(M, K, N)
                                  ? GEMM_MEM_STATIC
                                  : GEMM_MEM_DYNAMIC;

    return gemm_plan_create_with_mode(M, K, N, mode);
}

/**
 * @brief Create an execution plan with explicit memory mode
 * 
 * **Workspace Allocation Details:**
 * 
 * **Static Mode:**
 * - Uses global pre-allocated buffer (GEMM_STATIC_WORKSPACE_SIZE)
 * - Zero allocation overhead (instant planning)
 * - Limited to small/medium matrices
 * - workspace_a and workspace_b share the same buffer (non-overlapping)
 * 
 * **Dynamic Mode:**
 * - Allocates 3 separate 64-byte aligned buffers:
 *   1. workspace_a: MC × KC floats
 *   2. workspace_b: (NC/NR) × KC × 16 floats (16 = B panel stride)
 *   3. workspace_temp: MC × NC floats (for edge handling)
 * - Handles arbitrary matrix sizes
 * - Allocation overhead: ~1-5 µs per plan
 * 
 * @param M Number of rows in A and C (must be > 0)
 * @param K Number of columns in A, rows in B (must be > 0)
 * @param N Number of columns in B and C (must be > 0)
 * @param mode Memory allocation mode (STATIC or DYNAMIC)
 * 
 * @return Pointer to allocated plan, or NULL on failure
 * 
 * @retval NULL if mode=STATIC but dimensions too large
 * @retval NULL if any dimension is zero
 * @retval NULL if memory allocation fails
 * @retval valid plan pointer on success
 * 
 * @note Static mode fails if gemm_fits_static(M,K,N) returns false
 * @note Dynamic mode never fails due to size (only malloc failure)
 * 
 * @see gemm_plan_create()
 * @see gemm_fits_static()
 */
gemm_plan_t *gemm_plan_create_with_mode(
    size_t M, size_t K, size_t N,
    gemm_memory_mode_t mode)
{
    if (mode == GEMM_MEM_STATIC && !gemm_fits_static(M, K, N))
        return NULL;

    gemm_plan_t *plan = (gemm_plan_t *)calloc(1, sizeof(gemm_plan_t));
    if (!plan)
        return NULL;

    // Store maximum dimensions
    plan->max_M = M;
    plan->max_K = K;
    plan->max_N = N;
    plan->mem_mode = mode;

    // Select blocking parameters
    gemm_select_blocking(M, K, N,
                         &plan->MC, &plan->KC, &plan->NC,
                         &plan->MR, &plan->NR);

    // Precompute tile counts for max dimensions
    plan->n_mc_tiles_max = (M + plan->MC - 1) / plan->MC;
    plan->n_kc_tiles_max = (K + plan->KC - 1) / plan->KC;
    plan->n_nc_tiles_max = (N + plan->NC - 1) / plan->NC;

    // Pre-select kernels for full tiles
    int dummy_width;
    gemm_select_kernels(plan->MR, plan->NR,
                       &plan->kern_full_add,
                       &plan->kern_full_store,
                       &dummy_width);

    // Allocate panel descriptors
    plan->n_npanels = (N + plan->NR - 1) / plan->NR;
    plan->npanels = (panel_info_t *)calloc(plan->n_npanels, sizeof(panel_info_t));

    if (!plan->npanels)
    {
        gemm_plan_destroy(plan);
        return NULL;
    }

    precompute_panels(plan);

    // Setup Workspace
    if (mode == GEMM_MEM_STATIC)
    {
        gemm_static_init();
        plan->workspace_a = gemm_static_pool.workspace;
        plan->workspace_b = gemm_static_pool.workspace + (plan->MC * plan->KC);
        plan->workspace_temp = gemm_static_pool.workspace;
        plan->workspace_size = 0;
        plan->workspace_aligned = 1;
    }
    else
    {
        size_t a_size = plan->MC * plan->KC * sizeof(float);
        size_t max_n_panels = (plan->NC + plan->NR - 1) / plan->NR;
        size_t b_size = max_n_panels * plan->KC * 16 * sizeof(float);
        size_t temp_size = plan->MC * plan->NC * sizeof(float);

        a_size = (a_size + 63) & ~(size_t)63;
        b_size = (b_size + 63) & ~(size_t)63;
        temp_size = (temp_size + 63) & ~(size_t)63;

        plan->workspace_size = a_size + b_size + temp_size;

        plan->workspace_a = (float *)gemm_aligned_alloc(64, a_size);
        plan->workspace_b = (float *)gemm_aligned_alloc(64, b_size);
        plan->workspace_temp = (float *)gemm_aligned_alloc(64, temp_size);

        if (!plan->workspace_a || !plan->workspace_b || !plan->workspace_temp)
        {
            gemm_plan_destroy(plan);
            return NULL;
        }

        plan->workspace_aligned = 1;
    }

    return plan;
}

//==============================================================================
// PLAN DESTRUCTION
//==============================================================================

/**
 * @brief Destroy an execution plan and free all resources
 * 
 * **Cleanup Actions:**
 * 1. Free panel descriptor array
 * 2. Free workspace buffers (if dynamic mode)
 * 3. Free plan structure
 * 
 * **Static Mode:**
 * - workspace_a, workspace_b, workspace_temp point into static pool
 * - Not freed (global buffer remains)
 * 
 * **Dynamic Mode:**
 * - All three workspace buffers are individually freed
 * - Uses gemm_aligned_free() to match gemm_aligned_alloc()
 * 
 * @param plan Pointer to plan to destroy (may be NULL)
 * 
 * @note Safe to call with NULL pointer (no-op)
 * @note After calling, the plan pointer is invalid
 * @note Does NOT free the user's A, B, C matrices (never owned by plan)
 * 
 * @see gemm_plan_create()
 * @see gemm_plan_create_with_mode()
 */
void gemm_plan_destroy(gemm_plan_t *plan)
{
    if (!plan)
        return;

    // Free panel descriptors (always owned by plan)
    free(plan->npanels);

    // Free workspace buffers (only if dynamically allocated)
    if (plan->mem_mode == GEMM_MEM_DYNAMIC)
    {
        if (plan->workspace_a)
            gemm_aligned_free(plan->workspace_a);
        if (plan->workspace_b)
            gemm_aligned_free(plan->workspace_b);
        if (plan->workspace_temp)
            gemm_aligned_free(plan->workspace_temp);
    }

    // Free plan structure itself
    free(plan);
}