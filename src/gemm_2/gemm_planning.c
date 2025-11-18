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
 * **Five-Stage Algorithm:**
 * 
 * 1. **Aspect Ratio Classification**: Analyze M/N and K/N ratios to determine
 *    matrix shape (tall/wide/deep/balanced) and select initial parameters
 * 
 * 2. **Dimension Clamping**: Ensure cache blocks don't exceed matrix dimensions
 *    (e.g., MC ≤ M, KC ≤ K, NC ≤ N)
 * 
 * 3. **L2 Cache Fitting**: Scale down parameters if workspace exceeds L2 target
 *    (~1.8 MB), maintaining aspect ratios via sqrt scaling
 * 
 * 4. **Minimum Size Enforcement**: Ensure blocks are large enough for at least
 *    one micro-kernel tile (MC ≥ MR, NC ≥ NR, KC ≥ 8)
 * 
 * 5. **Register Block Validation**: Ensure register blocks fit within cache
 *    blocks (MR ≤ MC, NR ≤ NC) to prevent buffer overflows
 * 
 * @param[in]  M  Number of rows in A and C
 * @param[in]  K  Number of columns in A, rows in B
 * @param[in]  N  Number of columns in B and C
 * @param[out] MC Output: M-dimension cache block size
 * @param[out] KC Output: K-dimension cache block size
 * @param[out] NC Output: N-dimension cache block size
 * @param[out] MR Output: M-dimension register block size
 * @param[out] NR Output: N-dimension register block size
 */
void gemm_select_blocking(
    size_t M, size_t K, size_t N,
    size_t *MC, size_t *KC, size_t *NC,
    size_t *MR, size_t *NR)
{
    //==========================================================================
    // STAGE 1: ASPECT RATIO CLASSIFICATION
    //==========================================================================
    // Analyze matrix shape to select optimal blocking strategy:
    // - Tall (M >> N): Large MC to amortize row packing overhead
    // - Wide (N >> M): Large NC to amortize column packing overhead
    // - Deep (K >> M,N): Large KC to reduce B repacking frequency
    // - Balanced: Default tuned parameters for general case
    //==========================================================================
    
    double aspect_mn = (double)M / (double)N;
    double aspect_kn = (double)K / (double)N;
    
    //--------------------------------------------------------------------------
    // Case 1: Tall Matrices (M >> N)
    // Strategy: Maximize MC to process many rows per B panel
    //--------------------------------------------------------------------------
    if (aspect_mn > 3.0) {
        *MC = ADAPTIVE_MC_TALL;      // 256 rows at a time (amortize packing)
        *KC = ADAPTIVE_KC_TALL;      // 128 (moderate K blocking)
        *NC = MIN(N, ADAPTIVE_NC_SMALL); // Small N panels (N is limited anyway)
        *MR = 16;                     // Use tallest kernels (16×N)
        *NR = (N >= 16) ? 16 : ((N >= 8) ? 8 : 6); // Adapt to available N
    }
    //--------------------------------------------------------------------------
    // Case 2: Wide Matrices (N >> M)
    // Strategy: Maximize NC to process many columns per A panel
    //--------------------------------------------------------------------------
    else if (aspect_mn < 0.33) {
        *MC = MIN(M, ADAPTIVE_MC_SMALL); // Small MC (M is limited anyway)
        *KC = ADAPTIVE_KC_TALL;           // 128 (moderate K blocking)
        *NC = ADAPTIVE_NC_WIDE;           // 512 columns at a time
        *MR = (M >= 16) ? 16 : ((M >= 8) ? 8 : 4); // Adapt to available M
        *NR = 16;                         // Use widest kernels (M×16)
    }
    //--------------------------------------------------------------------------
    // Case 3: Deep Matrices (K >> M,N)
    // Strategy: Maximize KC to reduce B repacking overhead
    //--------------------------------------------------------------------------
    else if (aspect_kn > 4.0) {
        *MC = MIN(M, ADAPTIVE_MC_SMALL); // Small MC (M is limited)
        *KC = ADAPTIVE_KC_DEEP;          // 512 K-blocks (large reduction dim)
        *NC = MIN(N, ADAPTIVE_NC_SMALL); // Small NC (N is limited)
        *MR = (M >= 8) ? 8 : 4;          // Adapt to available M
        *NR = (N >= 8) ? 8 : 6;          // Adapt to available N
    }
    //--------------------------------------------------------------------------
    // Case 4: Balanced Matrices
    // Strategy: Use default tuned parameters (128/256/256)
    //--------------------------------------------------------------------------
    else {
        *MC = GEMM_BLOCK_MC; // 128 (tuned for Intel 14900K L1)
        *KC = GEMM_BLOCK_KC; // 256 (tuned for L2 cache)
        *NC = GEMM_BLOCK_NC; // 256 (tuned for L2 cache)
        
        // Select register blocking based on available dimensions
        if (N >= 16 && M >= 8) {
            *NR = 16;                        // Use 16-wide kernels
            *MR = (M >= 16) ? 16 : 8;        // 16×16 or 8×16
        } else if (N >= 8) {
            *NR = 8;                         // Use 8-wide kernels
            *MR = (M >= 16) ? 16 : 8;        // 16×8 or 8×8
        } else {
            *NR = (N >= 6) ? 6 : N;          // Narrow matrices (6-wide or less)
            *MR = (M >= 8) ? 8 : M;          // Match available rows
        }
    }
    
    //==========================================================================
    // STAGE 2: DIMENSION CLAMPING
    //==========================================================================
    // Ensure cache blocks don't exceed actual matrix dimensions.
    // Example: If M=50 but MC=128, clamp MC=50 to avoid oversized tiles.
    // This prevents wasting workspace on unused capacity.
    //==========================================================================
    
    *MC = MIN(*MC, M);
    *KC = MIN(*KC, K);
    *NC = MIN(*NC, N);
    
    //==========================================================================
    // STAGE 3: L2 CACHE FITTING
    //==========================================================================
    // Target: (MC×KC + KC×NC) × 4 bytes ≤ 1.8 MB
    // 
    // Workspace components:
    // - workspace_a: MC × KC floats (packed A panels)
    // - workspace_b: KC × NC floats (packed B panels)
    // 
    // If workspace exceeds L2 cache, scale down ALL parameters proportionally
    // using sqrt() to maintain aspect ratios. This keeps the relative balance
    // between M/K/N blocking while fitting within cache constraints.
    //==========================================================================
    
    const size_t L2_TARGET = 1800 * 1024; // 1.8 MB (leave headroom for code/stack)
    size_t workspace_bytes = (*MC * *KC + *KC * *NC) * sizeof(float);
    
    if (workspace_bytes > L2_TARGET) {
        // Scale factor: sqrt maintains aspect ratio
        // Example: If workspace is 4× too large, scale = 0.5
        // This reduces MC, KC, NC all by 0.5, making workspace 0.25× (fits!)
        double scale = sqrt((double)L2_TARGET / (double)workspace_bytes);
        
        // Apply scaling with alignment preservation:
        // - MC must be multiple of MR (for clean micro-kernel tiling)
        // - KC must be multiple of 8 (for SIMD alignment)
        // - NC must be multiple of NR (for clean micro-kernel tiling)
        *MC = ((*MC * scale) / *MR) * *MR; // Round down to MR multiple
        *KC = ((*KC * scale) / 8) * 8;     // Round down to 8 multiple
        *NC = ((*NC * scale) / *NR) * *NR; // Round down to NR multiple
        
        // Ensure minimum safe values after rounding
        *MC = MAX(*MC, *MR);                // At least one MR tile
        *KC = MAX(*KC, ADAPTIVE_KC_MIN);    // At least 32 (min reduction block)
        *NC = MAX(*NC, MIN(*NR, N));        // At least one NR tile (or all of N)
    }
    
    //==========================================================================
    // STAGE 4: MINIMUM SIZE ENFORCEMENT
    //==========================================================================
    // Ensure cache blocks can hold at least ONE micro-kernel tile.
    // If MC < MR, we can't even pack a single tile - reset to minimum.
    // 
    // This handles edge cases where:
    // - L2 scaling made blocks too small
    // - Very small matrices (M=1, N=1, etc.)
    // - Unusual aspect ratios that broke assumptions
    //==========================================================================
    
    if (*MC < *MR || *NC < *NR || *KC < 8) {
        *MC = *MR;              // Reset to single tile height
        *KC = ADAPTIVE_KC_MIN;  // Reset to minimum K block (32)
        *NC = *NR;              // Reset to single tile width
    }
    
    // Re-clamp to matrix dimensions (in case resets exceeded them)
    *MC = MIN(*MC, M);
    *KC = MIN(*KC, K);
    *NC = MIN(*NC, N);
    
    //==========================================================================
    // STAGE 5: REGISTER BLOCK VALIDATION (CRITICAL FIX!)
    //==========================================================================
    // **THE BUG FIX**: Ensure MR ≤ MC and NR ≤ NC
    // 
    // **Why this is needed:**
    // The packing functions write MR × KC floats to workspace_a, but workspace
    // is allocated as MC × KC floats. If MR > MC, we OVERFLOW the buffer!
    // 
    // **How it happens:**
    // 1. Initial selection: MR=8 (based on available kernels)
    // 2. Dimension clamping: MC = min(128, M=7) = 7
    // 3. Result: MR=8 > MC=7 → BUFFER OVERFLOW when packing!
    // 
    // **The fix:**
    // After all scaling/clamping, ensure register blocks respect cache blocks.
    // This prevents pack functions from writing beyond allocated workspace.
    // 
    // **Example:**
    // - Before fix: MC=7, MR=8 → pack writes 8 rows, overflow by 1 row
    // - After fix:  MC=7, MR=7 → pack writes 7 rows, safe!
    //==========================================================================
    
    *MR = MIN(*MR, *MC);  // ← THE FIX: Never pack more rows than cache block
    *NR = MIN(*NR, *NC);  // ← THE FIX: Never pack more cols than cache block
    
    //==========================================================================
    // FINAL STATE SUMMARY
    //==========================================================================
    // After this function:
    // - MC, KC, NC are cache block sizes (fit in L1/L2)
    // - MR, NR are register block sizes (micro-kernel dimensions)
    // - Invariants guaranteed:
    //   * MR ≤ MC ≤ M (no M-dimension overflow)
    //   * NR ≤ NC ≤ N (no N-dimension overflow)
    //   * KC ≤ K (no K-dimension overflow)
    //   * MC ≥ MR, NC ≥ NR (at least one tile fits)
    //   * KC ≥ 8 (SIMD alignment)
    //   * (MC×KC + KC×NC) × 4 ≤ ~1.8 MB (fits in L2)
    //==========================================================================
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
/**
 * @brief Select kernel using exhaustive 2D dispatch table
 * 
 * RULES:
 * 1. Always select LARGEST kernel that fits (m_kernel ≤ m, n_kernel ≤ n)
 * 2. Handle ALL combinations systematically
 * 3. Kernels handle partial tiles via scalar loops (safe)
 */
/**
 * @brief Select kernel using refined 2D dispatch table
 * 
 * KEY FIX: Distinguish m ∈ [5,8] from m ∈ [9,15]
 */
void gemm_select_kernels(
    size_t m_height, size_t n_width,
    gemm_kernel_id_t *kern_add,
    gemm_kernel_id_t *kern_store,
    int *kernel_width)
{
    //==========================================================================
    // M-DIMENSION CLASSIFICATION (REFINED!)
    //==========================================================================
    int m_class;
    if (m_height >= 16)      m_class = 4;  // 16+ rows → use 16× kernels
    else if (m_height >= 9)  m_class = 3;  // 9-15 rows → use 16× kernels (composite)
    else if (m_height >= 5)  m_class = 2;  // 5-8 rows → use 8× kernels
    else if (m_height >= 1)  m_class = 1;  // 1-4 rows → use 4× kernels
    else                     m_class = 0;  // ERROR
    
    //==========================================================================
    // N-DIMENSION CLASSIFICATION
    //==========================================================================
    int n_class;
    if (n_width >= 16)       n_class = 4;  // 16+ cols
    else if (n_width >= 9)   n_class = 3;  // 9-15 cols (needs 16-wide kernel)
    else if (n_width >= 6)   n_class = 2;  // 6-8 cols
    else if (n_width >= 1)   n_class = 1;  // 1-5 cols
    else                     n_class = 0;  // ERROR
    
    //==========================================================================
    // 2D DISPATCH TABLE
    //==========================================================================
    // ┌─────────┬─────────┬─────────┬─────────┬─────────┐
    // │ m\n     │ 1-5     │ 6-8     │ 9-15    │ 16+     │
    // ├─────────┼─────────┼─────────┼─────────┼─────────┤
    // │ 1-4     │ 4×8     │ 4×8     │ 4×8     │ 4×8     │
    // │ 5-8     │ 8×8     │ 8×8     │ 8×16    │ 8×16    │
    // │ 9-15    │ 16×8    │ 16×8    │ 16×16   │ 16×16   │ ← CRITICAL FIX!
    // │ 16+     │ 16×8    │ 16×8    │ 16×16   │ 16×16   │
    // └─────────┴─────────┴─────────┴─────────┴─────────┘
    
    gemm_kernel_id_t selected_add, selected_store;
    int selected_width;
    
    if (m_class == 1) {
        // m ∈ [1, 4]: Always use 4×8
        selected_add = KERN_4x8_ADD;
        selected_store = KERN_4x8_STORE;
        selected_width = 8;
    }
    else if (m_class == 2) {
        // m ∈ [5, 8]: Use 8×8 or 8×16
        if (n_class <= 2) {  // n ∈ [1, 8]
            selected_add = KERN_8x8_ADD;
            selected_store = KERN_8x8_STORE;
            selected_width = 8;
        } else {  // n ∈ [9, ∞)
            selected_add = KERN_8x16_ADD;
            selected_store = KERN_8x16_STORE;
            selected_width = 16;
        }
    }
    else if (m_class == 3 || m_class == 4) {
        //  CRITICAL FIX: m ∈ [9, ∞) → Use 16× kernels (composite)
        if (n_class <= 2) {  // n ∈ [1, 8]
            selected_add = KERN_16x8_ADD;
            selected_store = KERN_16x8_STORE;
            selected_width = 8;
        } else {  // n ∈ [9, ∞)
            selected_add = KERN_16x16_ADD;  // ← Composite: 2× 8×16 calls
            selected_store = KERN_16x16_STORE;
            selected_width = 16;
        }
    }
    else {
        // Fallback (should never reach)
        selected_add = KERN_1x8_ADD;
        selected_store = KERN_1x8_STORE;
        selected_width = 8;
    }
    
    *kern_add = selected_add;
    *kern_store = selected_store;
    *kernel_width = selected_width;
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
    else  // GEMM_MEM_DYNAMIC
{
    // FIX: Allocate workspace_a based on MR, not MC!
    // 
    // Rationale: pack_A_panel_simd_strided() always writes MR rows,
    // regardless of actual ib. It writes kb × requested_mr floats.
    // So workspace_a must be sized for MR × KC, not MC × KC.

    size_t a_size = plan->MR * plan->KC * sizeof(float);
    
    // workspace_b sizing: Multiple N-panels, each KC × 16 floats
    size_t max_n_panels = (plan->NC + plan->NR - 1) / plan->NR;
    size_t b_size = max_n_panels * plan->KC * 16 * sizeof(float);
    
    // workspace_temp for edge tile handling
    size_t temp_size = plan->MC * plan->NC * sizeof(float);
    
    // Align to 64-byte boundaries (cache line size)
    a_size = (a_size + 63) & ~(size_t)63;
    b_size = (b_size + 63) & ~(size_t)63;
    temp_size = (temp_size + 63) & ~(size_t)63;
    
    plan->workspace_size = a_size + b_size + temp_size;
    
    // Allocate separate aligned buffers
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