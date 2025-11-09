#ifndef GEMM_PLANNING_H
#define GEMM_PLANNING_H

#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include "gemm_static.h"

//==============================================================================
// BLOCKING PARAMETERS (Tuned for Intel 14900K)
//==============================================================================
// L1D: 48KB per P-core, 32KB per E-core
// L2:  2MB per P-core, 4MB per E-core cluster  
// L3:  36MB shared

#define GEMM_BLOCK_MC  128   // M-dimension cache block (fits in L2)
#define GEMM_BLOCK_KC  256   // K-dimension cache block (balance L1/register pressure)
#define GEMM_BLOCK_NC  256   // N-dimension cache block (good for L2 TLB)

#define GEMM_BLOCK_MR  16    // M-dimension register block
#define GEMM_BLOCK_NR  16    // N-dimension register block (adaptive: 8 or 16)

//==============================================================================
// KERNEL IDENTIFICATION
//==============================================================================

typedef enum {
    // 8-wide kernels (single __m256i mask)
    KERN_16x8_ADD,    KERN_16x8_STORE,
    KERN_8x8_ADD,     KERN_8x8_STORE,
    KERN_16x6_ADD,    KERN_16x6_STORE,
    KERN_8x6_ADD,     KERN_8x6_STORE,
    KERN_4x8_ADD,     KERN_4x8_STORE,
    KERN_1x8_ADD,     KERN_1x8_STORE,
    
    // 16-wide kernels (dual __m256i masks: lo and hi)
    KERN_8x16_ADD,    KERN_8x16_STORE,
    KERN_16x16_ADD,   KERN_16x16_STORE,
    
    KERN_INVALID
} gemm_kernel_id_t;

//==============================================================================
// TILE & PANEL DESCRIPTORS
//==============================================================================

/**
 * @brief M-dimension tile descriptor (pre-computed during planning)
 */
typedef struct {
    size_t i_start;              // Starting row index
    size_t i_height;             // Actual rows in this tile (<= MR)
    gemm_kernel_id_t kern_add;   // Kernel for accumulate mode (kt > 0)
    gemm_kernel_id_t kern_store; // Kernel for store mode (kt == 0, beta == 0)
    int kernel_width;            // 8 or 16 (for mask selection)
} tile_info_t;

/**
 * @brief N-dimension panel descriptor (pre-computed during planning)
 * 
 * CRITICAL: Aligned to 32 bytes for __m256i members
 */
typedef struct {
    size_t j_start;
    size_t j_width;
    __m256i mask_lo;    // Requires 32-byte alignment
    __m256i mask_hi;    // Requires 32-byte alignment
    int needs_mask;
} panel_info_t
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((aligned(32)))
#elif defined(_MSC_VER)
    __declspec(align(32))
#endif
;

//==============================================================================
// MEMORY MODES
//==============================================================================

typedef enum {
    GEMM_MEM_STATIC,   // Thread-local static pool (zero allocation)
    GEMM_MEM_DYNAMIC   // Aligned malloc (fallback for large matrices)
} gemm_memory_mode_t;

//==============================================================================
// EXECUTION PLAN
//==============================================================================

typedef struct gemm_plan {
    //--------------------------------------------------------------------------
    // Matrix Dimensions
    //--------------------------------------------------------------------------
    size_t M, K, N;
    
    //--------------------------------------------------------------------------
    // Blocking Parameters (Cache hierarchy optimization)
    //--------------------------------------------------------------------------
    size_t MC, KC, NC;  // L2/L3 cache blocks
    size_t MR, NR;      // Register blocks
    
    //--------------------------------------------------------------------------
    // Tile Decomposition
    //--------------------------------------------------------------------------
    size_t n_mtiles;         // Number of M-direction tiles
    size_t n_npanels;        // Number of N-direction panels
    size_t n_ktiles;         // Number of K-direction tiles
    
    tile_info_t *mtiles;     // Array[n_mtiles] of M-tile descriptors
    panel_info_t *npanels;   // Array[n_npanels] of N-panel descriptors
    
    //--------------------------------------------------------------------------
    // Pre-computed Masks (Storage owned by plan)
    //--------------------------------------------------------------------------
    __m256i *mask_storage;   // Contiguous storage for all masks
    size_t n_masks;          // Total number of masks allocated
    
    //--------------------------------------------------------------------------
    // Memory Strategy
    //--------------------------------------------------------------------------
    gemm_memory_mode_t mem_mode;
    
    float *workspace_a;      // Packed A buffer (points to static pool OR malloc)
    float *workspace_b;      // Packed B buffer (points to static pool OR malloc)
    float *workspace_temp;   // Temporary workspace (for symmetric ops, etc.)
    
    size_t workspace_size;   // Size in bytes (0 for static mode)
    int workspace_aligned;   // Always 1 (both static and dynamic are aligned)
    
    //--------------------------------------------------------------------------
    // Packing Function Pointers (Selected based on alignment/dimensions)
    //--------------------------------------------------------------------------
    void (*pack_a_fn)(float *dst, const float *src, 
                      size_t M, size_t K, size_t i0, size_t ib, 
                      size_t k0, size_t kb);
    
    void (*pack_b_fn)(float *dst, const float *src, 
                      size_t K, size_t N, size_t k0, size_t kb, 
                      size_t j0, size_t jb, panel_info_t *panel);
    
} gemm_plan_t;

//==============================================================================
// PLAN LIFECYCLE FUNCTIONS
//==============================================================================

/**
 * @brief Create execution plan with automatic memory mode selection
 * 
 * Selects static mode if dimensions fit (M,K,N ≤ GEMM_STATIC_MAX_DIM),
 * otherwise falls back to dynamic allocation.
 * 
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * @return Plan pointer or NULL on allocation failure
 */
gemm_plan_t *gemm_plan_create(size_t M, size_t K, size_t N);

/**
 * @brief Create plan with explicit memory mode
 * 
 * @param mode GEMM_MEM_STATIC or GEMM_MEM_DYNAMIC
 * @return Plan pointer or NULL if:
 *         - Allocation fails
 *         - Static mode requested but dimensions too large
 */
gemm_plan_t *gemm_plan_create_with_mode(
    size_t M, size_t K, size_t N, 
    gemm_memory_mode_t mode);

/**
 * @brief Destroy plan and free resources
 * 
 * For STATIC mode: Only frees plan structure (workspace is static pool)
 * For DYNAMIC mode: Frees plan structure AND workspace memory
 */
void gemm_plan_destroy(gemm_plan_t *plan);

/**
 * @brief Query workspace size needed for dynamic allocation
 * 
 * Useful for pre-allocating memory or checking requirements
 */
size_t gemm_workspace_query(size_t M, size_t K, size_t N);

//==============================================================================
// HELPER FUNCTIONS (Internal use, exposed for testing)
//==============================================================================

/**
 * @brief Build AVX2 mask for partial vector width
 * @param n Number of active lanes (1-8)
 * @return Mask with n lanes set to -1, rest to 0
 */
__m256i gemm_build_mask_avx2(size_t n);

/**
 * @brief Select optimal blocking parameters based on matrix shape
 */
void gemm_select_blocking(
    size_t M, size_t K, size_t N,
    size_t *MC, size_t *KC, size_t *NC,
    size_t *MR, size_t *NR);

/**
 * @brief Select kernel IDs for a given tile size
 */
void gemm_select_kernels(
    size_t m_height, size_t n_width,
    gemm_kernel_id_t *kern_add,
    gemm_kernel_id_t *kern_store,
    int *kernel_width);


    /**
 * @brief Execute planned GEMM: C = alpha*A*B + beta*C
 * 
 * This is the main execution engine called by gemm_auto/static/dynamic.
 * It performs 3-level cache blocking with optimized packing and kernel dispatch.
 * 
 * @param plan Execution plan (must be valid)
 * @param C Output matrix (M×N)
 * @param A Input matrix (M×K)
 * @param B Input matrix (K×N)
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for C
 * @return 0 on success, negative error code on failure
 */
int gemm_execute_plan(
    gemm_plan_t *plan,
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    float alpha,
    float beta);

#endif // GEMM_PLANNING_H