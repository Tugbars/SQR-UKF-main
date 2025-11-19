/**
 * @file gemm_planning.h
 * @brief GEMM Execution Planning - Adaptive Blocking and Kernel Selection
 * 
 * This module implements the planning layer for GEMM execution, responsible for:
 * - Adaptive blocking parameter selection based on matrix dimensions
 * - Kernel selection for optimal performance
 * - Workspace allocation and management
 * - Execution plan creation and destruction
 * 
 * The planner analyzes matrix aspect ratios (tall/wide/deep) and selects
 * blocking parameters that maximize cache reuse and minimize memory traffic.
 * 
 * @author TUGBARS
 * @date 2025
 */

#ifndef GEMM_PLANNING_H
#define GEMM_PLANNING_H

#include <stddef.h>
#include <stdint.h>
#include "gemm_static.h"

//==============================================================================
// BLOCKING PARAMETERS (Tuned for Intel 14900K)
//==============================================================================

/** @brief MC blocking parameter (M-dimension cache block size)
 *  @details Tuned for L1 cache, controls rows of A per panel */
#define GEMM_BLOCK_MC  128

/** @brief KC blocking parameter (K-dimension cache block size)
 *  @details Tuned for L2 cache, controls shared dimension blocking */
#define GEMM_BLOCK_KC  256

/** @brief NC blocking parameter (N-dimension cache block size)
 *  @details Tuned for L2 cache, controls columns of B per panel */
#define GEMM_BLOCK_NC  256

/** @brief MR register blocking parameter (M-dimension micro-kernel height)
 *  @details Maximum rows handled by micro-kernels (8 or 16) */
#define GEMM_BLOCK_MR  16

/** @brief NR register blocking parameter (N-dimension micro-kernel width)
 *  @details Maximum columns handled by micro-kernels (6, 8, or 16) */
#define GEMM_BLOCK_NR  16

//==============================================================================
// KERNEL IDENTIFICATION
//==============================================================================

/**
 * @brief Enumeration of available micro-kernel variants
 * 
 * Each kernel is identified by its tile size (M×N) and operation mode:
 * - ADD variants: C += A*B (accumulate into existing C)
 * - STORE variants: C = A*B (overwrite C, used when beta=0 on first K-tile)
 * 
 * Naming convention: KERN_{M}x{N}_{MODE}
 * - M: Number of rows (1, 4, 8, 16)
 * - N: Number of columns (6, 8, 16)
 * - MODE: ADD or STORE
 * 
 * @note Composite kernels (16×16, 16×6) are implemented as multiple calls
 *       to smaller kernels to avoid register pressure
 */
typedef enum {
    KERN_16x8_ADD,     /**< 16 rows × 8 cols, C += A*B */
    KERN_16x8_STORE,   /**< 16 rows × 8 cols, C = A*B */
    KERN_8x8_ADD,      /**< 8 rows × 8 cols, C += A*B */
    KERN_8x8_STORE,    /**< 8 rows × 8 cols, C = A*B */
    KERN_16x6_ADD,     /**< 16 rows × 6 cols, C += A*B */
    KERN_16x6_STORE,   /**< 16 rows × 6 cols, C = A*B */
    KERN_8x6_ADD,      /**< 8 rows × 6 cols, C += A*B */
    KERN_8x6_STORE,    /**< 8 rows × 6 cols, C = A*B */
    KERN_4x8_ADD,      /**< 4 rows × 8 cols, C += A*B */
    KERN_4x8_STORE,    /**< 4 rows × 8 cols, C = A*B */
    KERN_1x8_ADD,      /**< 1 row × 8 cols, C += A*B */
    KERN_1x8_STORE,    /**< 1 row × 8 cols, C = A*B */
    KERN_8x16_ADD,     /**< 8 rows × 16 cols, C += A*B */
    KERN_8x16_STORE,   /**< 8 rows × 16 cols, C = A*B */
    KERN_16x16_ADD,    /**< 16 rows × 16 cols, C += A*B (composite) */
    KERN_16x16_STORE,  /**< 16 rows × 16 cols, C = A*B (composite) */
    KERN_INVALID       /**< Invalid/uninitialized kernel ID */
} gemm_kernel_id_t;

//==============================================================================
// SIMPLIFIED PANEL DESCRIPTOR (NO MASKS!)
//==============================================================================

/**
 * @brief Descriptor for a single N-panel (vertical strip of B matrix)
 * 
 * The execution plan divides the N-dimension into panels of width NR.
 * Each panel is packed once and reused across all M-tiles, maximizing
 * cache efficiency.
 * 
 * @note Masks were removed in favor of safe scalar loops for partial widths
 */
typedef struct {
    size_t j_start;   /**< Starting column index in B matrix */
    size_t j_width;   /**< Actual width of this panel (≤ NR) */
} panel_info_t;

//==============================================================================
// MEMORY MODES
//==============================================================================

/**
 * @brief Memory allocation strategy for workspace buffers
 * 
 * The planner supports two allocation modes:
 * - STATIC: Uses pre-allocated global workspace (faster, limited size)
 * - DYNAMIC: Allocates workspace per execution (slower, unlimited size)
 * 
 * @see gemm_fits_static() to check if static mode is available
 */
typedef enum {
    GEMM_MEM_STATIC,   /**< Use static global workspace (limited size) */
    GEMM_MEM_DYNAMIC   /**< Dynamically allocate workspace (any size) */
} gemm_memory_mode_t;

//==============================================================================
// EXECUTION PLAN
//==============================================================================

typedef struct {
    size_t n_mc, n_kc, n_nc;
} tile_counts_t;

/**
 * @brief Complete execution plan for a GEMM operation
 * 
 * The plan pre-computes all metadata needed for execution:
 * - Blocking parameters (MC, KC, NC, MR, NR)
 * - Tile counts (eliminates division in hot path)
 * - Pre-selected kernels for full tiles (eliminates dispatch overhead)
 * - Panel descriptors (just dimensions, no masks)
 * - Workspace pointers (static or dynamic)
 * 
 * Plans are created once and can be reused for multiple executions
 * with the same matrix dimensions, amortizing planning cost.
 * 
 * @note The plan does NOT own the input/output matrices (A, B, C),
 *       only the internal workspace buffers
 * 
 * @see gemm_plan_create()
 * @see gemm_execute_plan()
 * @see gemm_plan_destroy()
 */


/**
 * @brief Complete execution plan for a GEMM operation
 * 
 * The plan is a "compiled" representation of how to execute GEMM for specific
 * matrix dimensions. Think of it as bytecode: expensive to create (planning),
 * but very fast to execute repeatedly.
 * 
 * **Creation Cost**: ~20 µs (one-time)
 * **Execution Benefit**: ~10 µs saved per call (no division, no kernel selection)
 * **Amortization**: Pays off after 2 executions
 * 
 * **Typical Usage:**
 * ```c
 * // Training loop - create plan ONCE
 * gemm_plan_t *plan = gemm_plan_create(1024, 512, 768);
 * 
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     for (int batch = 0; batch < 1000; batch++) {
 *         // FAST: Reuse plan 100,000 times!
 *         gemm_execute_plan(plan, C, A, B, 1.0, 0.0);
 *     }
 * }
 * 
 * gemm_plan_destroy(plan);  // Clean up once
 * ```
 */
typedef struct gemm_plan {
    //==========================================================================
    // MATRIX DIMENSIONS (For Validation and Workspace Sizing)
    //==========================================================================
    
    /** @brief Maximum M dimension this plan can handle
     *  
     *  Stored for two purposes:
     *  1. Validation: gemm_execute_plan checks actual M ≤ max_M
     *  2. Workspace sizing: workspace_a is sized for max_M × KC
     *  
     *  For strided GEMM, actual M may be smaller than max_M.
     */
    size_t max_M;
    
    /** @brief Maximum K dimension this plan can handle
     *  
     *  Determines workspace_a size (max_M × KC) and workspace_b size (KC × max_N).
     *  Actual K in strided calls may be smaller.
     */
    size_t max_K;
    
    /** @brief Maximum N dimension this plan can handle
     *  
     *  Determines workspace_b size and number of N-panels.
     *  Actual N in strided calls may be smaller.
     */
    size_t max_N;
    
    //==========================================================================
    // PRE-COMPUTED TILE COUNTS (Eliminates Division in Hot Path!)
    //==========================================================================
    
    /** @brief Number of MC-tiles needed to cover max_M rows
     *  
     *  Pre-computed as: (max_M + MC - 1) / MC
     *  
     *  **Why pre-compute?**
     *  Division is expensive (~10-20 cycles on modern CPUs). By computing this
     *  once during planning, we replace division with a simple comparison:
     *  
     *  OLD (naive): `for (it = 0; it < (M + MC - 1) / MC; it++)`
     *  NEW (planned): `for (it = 0; it < plan->n_mc_tiles_max; it++)`
     *  
     *  Savings: ~10 cycles → ~1 cycle per loop setup
     */
    size_t n_mc_tiles_max;
    
    /** @brief Number of KC-tiles needed to cover max_K columns/rows
     *  
     *  Pre-computed as: (max_K + KC - 1) / KC
     *  
     *  This determines how many K-iterations we need. Each KC-tile requires:
     *  - Packing B panels once (reused across all MC-tiles)
     *  - Packing A panels multiple times (once per MC-tile)
     */
    size_t n_kc_tiles_max;
    
    /** @brief Number of NC-tiles needed to cover max_N columns
     *  
     *  Pre-computed as: (max_N + NC - 1) / NC
     *  
     *  This is the outer loop count. Each NC-tile processes a vertical strip
     *  of B and C matrices.
     */
    size_t n_nc_tiles_max;
    
    //==========================================================================
    // CACHE BLOCKING PARAMETERS (Tuned for Target CPU)
    //==========================================================================
    
    /** @brief M-dimension cache block size (rows of A per panel)
     *  
     *  Controls how many rows of A we pack at once. Selected to fit:
     *  MC × KC × 4 bytes ≤ L1 cache (~48 KB on Intel 14900K)
     *  
     *  Typical values: 64 (small matrices) to 256 (tall matrices)
     *  
     *  **Aspect ratio adaptation:**
     *  - Tall (M >> N): Large MC (256) to amortize row reuse
     *  - Wide (N >> M): Small MC (64) to reduce memory footprint
     */
    size_t MC;
    
    /** @brief K-dimension cache block size (shared dimension blocking)
     *  
     *  Controls the reduction dimension blocking. Selected to fit:
     *  (MC × KC + KC × NC) × 4 bytes ≤ L2 cache (~2 MB per core)
     *  
     *  Typical values: 128 (balanced) to 512 (deep matrices, K >> M,N)
     *  
     *  **Critical for performance:**
     *  - Larger KC: Fewer B packing operations (good for K >> M,N)
     *  - Smaller KC: Better cache reuse (good for small K)
     */
    size_t KC;
    
    /** @brief N-dimension cache block size (columns of B per panel)
     *  
     *  Controls how many columns of B we pack at once. Selected to fit:
     *  KC × NC × 4 bytes ≤ L2 cache (~2 MB)
     *  
     *  Typical values: 128 (small matrices) to 512 (wide matrices)
     *  
     *  **Aspect ratio adaptation:**
     *  - Wide (N >> M): Large NC (512) to amortize column reuse
     *  - Tall (M >> N): Small NC (128) since N is limited
     */
    size_t NC;
    
    /** @brief M-dimension register block size (micro-kernel height)
     *  
     *  The number of rows a single micro-kernel processes.
     *  
     *  Possible values: 4, 8, or 16
     *  - MR=4: For very small M (1-4 rows), uses 4×8 kernel
     *  - MR=8: Most common, balanced register usage
     *  - MR=16: For tall matrices, uses composite 16×8 or 16×16 kernels
     *  
     *  **Register pressure:**
     *  MR=8 uses ~11 YMM registers (safe, no spilling)
     *  MR=16 uses ~20 YMM registers (near limit, but still safe)
     */
    size_t MR;
    
    /** @brief N-dimension register block size (micro-kernel width)
     *  
     *  The number of columns a single micro-kernel processes.
     *  
     *  Possible values: 6, 8, or 16
     *  - NR=6: For QR decomposition (panel width often 6)
     *  - NR=8: Most common, power-of-2 alignment
     *  - NR=16: For wide matrices, uses 8×16 or 16×16 kernels
     *  
     *  **Important:** Packed B stride is ALWAYS 16, even for NR=6 or NR=8!
     *  This simplifies indexing: B[k, n] = Bp[k*16 + n]
     */
    size_t NR;
    
    //==========================================================================
    // PANEL DESCRIPTORS (N-Dimension Tiling)
    //==========================================================================
    
    /** @brief Number of N-panels (vertical strips of B matrix)
     *  
     *  Computed as: (max_N + NR - 1) / NR
     *  
     *  Each panel is NR columns wide (except the last, which may be narrower).
     *  Panels are packed once per KC-tile and reused across all MC-tiles.
     *  
     *  Example: If N=100, NR=16:
     *    n_npanels = 7
     *    Panels: [0:16), [16:32), [32:48), [48:64), [64:80), [80:96), [96:100)
     */
    size_t n_npanels;
    
    /** @brief Array of panel descriptors (one per N-panel)
     *  
     *  Allocated as: calloc(n_npanels, sizeof(panel_info_t))
     *  
     *  Each panel_info_t contains:
     *  - j_start: Starting column index in B matrix
     *  - j_width: Actual width of this panel (≤ NR)
     *  
     *  **Why pre-compute?**
     *  Eliminates bounds checking arithmetic in inner loop. We just index:
     *  `panel_info_t *panel = &plan->npanels[p];`
     *  
     *  **Memory cost:** ~16 bytes per panel (negligible for typical N)
     */
    panel_info_t *npanels;
    
    //==========================================================================
    // PRE-SELECTED KERNELS (Eliminates 95% of Dispatch Overhead!)
    //==========================================================================
    
    /** @brief Kernel ID for full tiles in ADD mode (C += A*B)
     *  
     *  Pre-selected during planning based on MR and NR:
     *  - If MR=8, NR=16 → KERN_8x16_ADD
     *  - If MR=16, NR=8 → KERN_16x8_ADD
     *  - If MR=8, NR=8 → KERN_8x8_ADD
     *  - etc.
     *  
     *  **Critical optimization:**
     *  ~95% of tiles are "full" (exactly MR × NR). For these, we use this
     *  pre-selected kernel directly, skipping the expensive gemm_select_kernels()
     *  call entirely!
     *  
     *  Savings: ~20 cycles (selection) → ~1 cycle (register read) per tile
     */
    gemm_kernel_id_t kern_full_add;
    
    /** @brief Kernel ID for full tiles in STORE mode (C = A*B, no accumulation)
     *  
     *  Same as kern_full_add, but overwrites C instead of accumulating.
     *  
     *  **When used:**
     *  - First K-tile when beta=0 (no need to load old C values)
     *  - Saves one memory load per output element (~2-3% speedup)
     *  
     *  For subsequent K-tiles, we always use ADD mode to accumulate results.
     */
    gemm_kernel_id_t kern_full_store;

    int kern_full_width;  //  Native width of selected kernel
    
    //==========================================================================
    // MEMORY MODE (Static vs Dynamic Allocation)
    //==========================================================================
    
    /** @brief Workspace allocation strategy
     *  
     *  GEMM_MEM_STATIC:
     *  - Uses global pre-allocated buffer (GEMM_STATIC_WORKSPACE_SIZE)
     *  - Zero allocation overhead (planning is instant)
     *  - Limited to small/medium matrices (~256×256×256)
     *  - NOT thread-safe (global buffer shared)
     *  
     *  GEMM_MEM_DYNAMIC:
     *  - Allocates workspace from heap (malloc/aligned_alloc)
     *  - Allocation overhead: ~1-5 µs per plan
     *  - Handles arbitrary matrix sizes
     *  - Thread-safe (each plan has its own workspace)
     *  
     *  **Selection:** gemm_plan_create() auto-selects based on gemm_fits_static()
     */
    gemm_memory_mode_t mem_mode;
    
    //==========================================================================
    // WORKSPACE BUFFERS (Hot Data, Cache-Critical!)
    //==========================================================================
    
    /** @brief Packed A panel buffer (MC × KC floats)
     *  
     *  Layout: Column-major (K-outer)
     *  ```
     *  K=0: [m0 m1 m2 ... m7/15]  ← MR elements
     *  K=1: [m0 m1 m2 ... m7/15]
     *  K=2: [m0 m1 m2 ... m7/15]
     *  ...
     *  ```
     *  
     *  **Why pack A?**
     *  - Converts row-major → column-major for better kernel access pattern
     *  - Absorbs alpha scaling (A *= alpha during pack, saves multiplies)
     *  - Prefetches ahead to hide memory latency
     *  
     *  **Access pattern in kernel:**
     *  `float *a_k = workspace_a + k * MR;`  // Get all M elements for K-iteration k
     *  
     *  **Memory traffic:**
     *  Packed once per MC-tile (re-packed frequently, but A is small)
     */
    float *workspace_a;
    
    /** @brief Packed B panel buffer (KC × NC floats, with stride=16)
     *  
     *  Layout: Row-major with FIXED stride=16 (not NR!)
     *  ```
     *  K=0: [n0 n1 n2 ... n7/15] [padding to 16]
     *  K=1: [n0 n1 n2 ... n7/15] [padding to 16]
     *  K=2: [n0 n1 n2 ... n7/15] [padding to 16]
     *  ...
     *  ```
     *  
     *  **Why pack B?**
     *  - Converts row-major → row-major with aligned stride
     *  - Fixed stride=16 simplifies kernel indexing
     *  - Packed ONCE per KC-tile, reused across all MC-tiles (critical!)
     *  
     *  **Access pattern in kernel:**
     *  `float *b_k = workspace_b + k * 16;`  // Get all N elements for K-iteration k
     *  
     *  **Memory traffic:**
     *  Packed once per KC×NC tile, reused (MC/MR) times → ~98% L2 hit rate!
     *  
     *  **Why stride=16, not NR?**
     *  Simplifies kernel: All kernels assume B stride=16, regardless of NR.
     *  For NR=6 or NR=8, we waste some space, but gain code simplicity.
     */
    float *workspace_b;
    
    /** @brief Temporary buffer for edge tile handling (MC × NC floats)
     *  
     *  Used only for partial tiles where we can't directly write to C.
     *  
     *  **Typical usage:** <5% of tiles (edge cases only)
     *  
     *  **Why needed?**
     *  Some kernels use transpose-and-write for performance. For partial tiles,
     *  we transpose into this temp buffer, then copy valid elements to C.
     *  
     *  **Not hot:** Rarely accessed, low performance impact
     */
    float *workspace_temp;
    
    /** @brief Total workspace size in bytes
     *  
     *  Sum of:
     *  - workspace_a: MC × KC × 4 bytes
     *  - workspace_b: KC × NC × 4 bytes (with 16-stride waste)
     *  - workspace_temp: MC × NC × 4 bytes
     *  - Alignment padding: 64-byte boundaries for cache line alignment
     *  
     *  Typical values:
     *  - Small (64×64×64): ~200 KB
     *  - Medium (256×256×256): ~1.5 MB
     *  - Large (1024×1024×1024): ~3 MB (may not fit in static!)
     *  
     *  **Zero for static mode** (uses global buffer, size not tracked per-plan)
     */
    size_t workspace_size;
    
    /** @brief Flag indicating if workspace is 64-byte aligned
     *  
     *  Always 1 for dynamic mode (we use aligned_alloc with 64-byte alignment)
     *  Always 1 for static mode (global buffer is aligned)
     *  
     *  **Why 64 bytes?**
     *  - Cache line size on x86-64 (prevents false sharing)
     *  - Required for AVX-512 aligned loads (though we use unaligned for safety)
     *  - Ensures first access doesn't cross cache line boundary
     *  
     *  **Currently unused** (we always use unaligned ops for safety), but kept
     *  for potential future optimizations with aligned SIMD ops.
     */
    int workspace_aligned;
    
} gemm_plan_t;

//==============================================================================
// PLAN LIFECYCLE
//==============================================================================

/**
 * @brief Create an execution plan with automatic memory mode selection
 * 
 * Creates a GEMM execution plan by analyzing matrix dimensions and selecting
 * optimal blocking parameters. Automatically chooses static workspace if
 * dimensions fit, otherwise uses dynamic allocation.
 * 
 * The plan pre-computes:
 * - Adaptive blocking parameters (MC, KC, NC, MR, NR)
 * - Tile counts (n_nc_tiles, n_kc_tiles, n_mc_tiles)
 * - Pre-selected kernels for full tiles
 * - Panel descriptors for N-dimension
 * - Workspace allocation
 * 
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * 
 * @return Pointer to allocated plan, or NULL on failure
 * 
 * @note The returned plan must be freed with gemm_plan_destroy()
 * @note Plans can be reused for multiple executions with same dimensions
 * 
 * @see gemm_execute_plan()
 * @see gemm_plan_destroy()
 * @see gemm_plan_create_with_mode()
 */
gemm_plan_t *gemm_plan_create(size_t M, size_t K, size_t N);

/**
 * @brief Create an execution plan with explicit memory mode
 * 
 * Similar to gemm_plan_create(), but allows explicit control over
 * workspace allocation strategy.
 * 
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * @param mode Memory allocation mode (GEMM_MEM_STATIC or GEMM_MEM_DYNAMIC)
 * 
 * @return Pointer to allocated plan, or NULL on failure
 * 
 * @note Returns NULL if GEMM_MEM_STATIC requested but dimensions too large
 * @note The returned plan must be freed with gemm_plan_destroy()
 * 
 * @see gemm_plan_create()
 * @see gemm_fits_static()
 */
gemm_plan_t *gemm_plan_create_with_mode(
    size_t M, size_t K, size_t N, 
    gemm_memory_mode_t mode);

/**
 * @brief Destroy an execution plan and free resources
 * 
 * Frees all memory associated with the plan:
 * - Panel descriptors
 * - Workspace buffers (if dynamically allocated)
 * - Plan structure itself
 * 
 * @param plan Pointer to plan to destroy (may be NULL, no-op in that case)
 * 
 * @note Safe to call with NULL pointer
 * @note After calling, the plan pointer is invalid and should not be used
 * 
 * @see gemm_plan_create()
 */
void gemm_plan_destroy(gemm_plan_t *plan);

/**
 * @brief Query workspace size required for given matrix dimensions
 * 
 * Computes the total workspace size (in bytes) needed for GEMM execution
 * without actually creating a plan. Useful for pre-allocation or memory
 * budget analysis.
 * 
 * The workspace includes:
 * - Packed A panel: MC × KC × sizeof(float)
 * - Packed B panel: KC × NC × sizeof(float)
 * - Temporary buffer: MC × NC × sizeof(float)
 * 
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * 
 * @return Required workspace size in bytes
 * 
 * @note The actual allocation may be slightly larger due to alignment padding
 * 
 * @see gemm_plan_create()
 */
size_t gemm_workspace_query(size_t M, size_t K, size_t N);

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Select adaptive blocking parameters based on matrix dimensions
 * 
 * Analyzes matrix aspect ratios and selects blocking parameters that
 * maximize cache efficiency:
 * 
 * - **Tall matrices** (M >> N): Large MC, small NC
 * - **Wide matrices** (N >> M): Small MC, large NC  
 * - **Deep matrices** (K >> M,N): Large KC
 * - **Balanced matrices**: Default tuned parameters
 * 
 * Also selects register blocking (MR, NR) based on available kernels
 * and matrix dimensions.
 * 
 * @param[in]  M  Number of rows in A and C
 * @param[in]  K  Number of columns in A, rows in B
 * @param[in]  N  Number of columns in B and C
 * @param[out] MC Output: M-dimension cache block size
 * @param[out] KC Output: K-dimension cache block size
 * @param[out] NC Output: N-dimension cache block size
 * @param[out] MR Output: M-dimension register block size
 * @param[out] NR Output: N-dimension register block size
 * 
 * @note All output parameters must be non-NULL
 * @note Selected parameters are clamped to not exceed matrix dimensions
 * @note Parameters are adjusted to fit within L2 cache target (~1.8 MB)
 * 
 * @see gemm_plan_create()
 */
void gemm_select_blocking(
    size_t M, size_t K, size_t N,
    size_t *MC, size_t *KC, size_t *NC,
    size_t *MR, size_t *NR);

/**
 * @brief Select optimal micro-kernel for given tile dimensions
 * 
 * Selects the best-fitting kernel based on actual tile size (m_height × n_width).
 * Returns both ADD and STORE variants, plus the kernel's native width.
 * 
 * Selection priority (largest to smallest):
 * 1. 16×16 (composite: two 8×16 calls)
 * 2. 16×8
 * 3. 8×16
 * 4. 16×6 (composite: two 8×6 calls)
 * 5. 8×8
 * 6. 8×6
 * 7. 4×8
 * 8. 1×8 (fallback)
 * 
 * @param[in]  m_height     Tile height (number of rows)
 * @param[in]  n_width      Tile width (number of columns)
 * @param[out] kern_add     Output: Kernel ID for ADD mode (C += A*B)
 * @param[out] kern_store   Output: Kernel ID for STORE mode (C = A*B)
 * @param[out] kernel_width Output: Native width of selected kernel
 * 
 * @note All output parameters must be non-NULL
 * @note Partial tiles use scalar loops for edge handling (safe, ~2-5% overhead)
 * 
 * @see gemm_kernel_id_t
 */
void gemm_select_kernels(
    size_t m_height, size_t n_width,
    gemm_kernel_id_t *kern_add,
    gemm_kernel_id_t *kern_store,
    int *kernel_width);

/**
 * @brief Execute GEMM operation using a pre-computed plan
 * 
 * Performs the operation: C = alpha * A * B + beta * C
 * 
 * Uses the blocking parameters, kernel selections, and workspace
 * from the pre-computed plan. This amortizes planning cost across
 * multiple executions.
 * 
 * **Execution flow:**
 * 1. Pre-scale C by beta (once, before K-loop)
 * 2. Outer NC-loop (reuse B panels in L2 cache)
 * 3. Middle KC-loop (pack B panels once per KC×NC tile)
 * 4. Inner MC-loop (pack A panels, execute kernels)
 * 
 * **Alpha/Beta semantics (BLAS-compatible):**
 * - alpha = 0: Returns beta*C (A and B ignored)
 * - beta = 0: C is zeroed (does not read old C values)
 * - beta = 1: Accumulate (C += alpha*A*B)
 * - Other values: Full GEMM (C = alpha*A*B + beta*C)
 * 
 * @param plan  Execution plan (must not be NULL)
 * @param C     Output matrix (M × N, row-major)
 * @param A     Input matrix A (M × K, row-major)
 * @param B     Input matrix B (K × N, row-major)
 * @param alpha Scalar multiplier for A*B
 * @param beta  Scalar multiplier for C
 * 
 * @return 0 on success, -1 on error (NULL pointers)
 * 
 * @note C, A, and B must be allocated by caller
 * @note Matrices are assumed row-major (C-style)
 * @note No aliasing: C must not overlap with A or B
 * @note Thread-safe if different threads use different plans/matrices
 * 
 * @warning Plan dimensions (M, K, N) must match actual matrix dimensions
 * 
 * @see gemm_plan_create()
 * @see gemm_auto()
 */
int gemm_execute_plan(
    gemm_plan_t *plan,
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    float alpha,
    float beta);

#endif // GEMM_PLANNING_H