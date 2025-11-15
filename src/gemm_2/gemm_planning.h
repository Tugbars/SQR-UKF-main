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
typedef struct gemm_plan {
    //--------------------------------------------------------------------------
    // Matrix Dimensions
    //--------------------------------------------------------------------------
    size_t M;  /**< Number of rows in A and C */
    size_t K;  /**< Number of columns in A, rows in B (shared dimension) */
    size_t N;  /**< Number of columns in B and C */
    
    //--------------------------------------------------------------------------
    // Blocking Parameters
    //--------------------------------------------------------------------------
    size_t MC;  /**< M-dimension cache block size (rows of A per panel) */
    size_t KC;  /**< K-dimension cache block size (shared dimension blocking) */
    size_t NC;  /**< N-dimension cache block size (columns of B per panel) */
    size_t MR;  /**< M-dimension register block (micro-kernel height: 8 or 16) */
    size_t NR;  /**< N-dimension register block (micro-kernel width: 6, 8, or 16) */
    
    //--------------------------------------------------------------------------
    // PRE-COMPUTED EXECUTION METADATA (OPTIMIZATION)
    //--------------------------------------------------------------------------
    size_t n_nc_tiles;   /**< Number of NC tiles: (N + NC - 1) / NC */
    size_t n_kc_tiles;   /**< Number of KC tiles: (K + KC - 1) / KC */
    size_t n_mc_tiles;   /**< Number of MC tiles: (M + MC - 1) / MC */
    
    //--------------------------------------------------------------------------
    // PRE-SELECTED KERNELS FOR FULL TILES (OPTIMIZATION)
    //--------------------------------------------------------------------------
    gemm_kernel_id_t kern_full_add;      /**< Kernel for full MR×NR tiles (ADD mode) */
    gemm_kernel_id_t kern_full_store;    /**< Kernel for full MR×NR tiles (STORE mode) */
    
    //--------------------------------------------------------------------------
    // Panel Descriptors (NO MASKS!)
    //--------------------------------------------------------------------------
    size_t n_npanels;        /**< Total number of N-panels: (N + NR - 1) / NR */
    panel_info_t *npanels;   /**< Array of panel descriptors (length: n_npanels) */
    
    //--------------------------------------------------------------------------
    // Memory Strategy
    //--------------------------------------------------------------------------
    gemm_memory_mode_t mem_mode;  /**< Workspace allocation mode (static/dynamic) */
    
    float *workspace_a;     /**< Packed A panel buffer (size: MC × KC floats) */
    float *workspace_b;     /**< Packed B panel buffer (size: KC × NC floats) */
    float *workspace_temp;  /**< Temporary computation buffer (size: MC × NC floats) */
    
    size_t workspace_size;   /**< Total workspace size in bytes (dynamic mode only) */
    int workspace_aligned;   /**< Non-zero if workspace is 64-byte aligned */
    
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