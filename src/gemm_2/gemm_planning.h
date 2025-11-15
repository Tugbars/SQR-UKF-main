#ifndef GEMM_PLANNING_H
#define GEMM_PLANNING_H

#include <stddef.h>
#include <stdint.h>
#include "gemm_static.h"

//==============================================================================
// BLOCKING PARAMETERS (Tuned for Intel 14900K)
//==============================================================================

#define GEMM_BLOCK_MC  128
#define GEMM_BLOCK_KC  256
#define GEMM_BLOCK_NC  256
#define GEMM_BLOCK_MR  16
#define GEMM_BLOCK_NR  16

//==============================================================================
// KERNEL IDENTIFICATION
//==============================================================================

typedef enum {
    KERN_16x8_ADD,    KERN_16x8_STORE,
    KERN_8x8_ADD,     KERN_8x8_STORE,
    KERN_16x6_ADD,    KERN_16x6_STORE,
    KERN_8x6_ADD,     KERN_8x6_STORE,
    KERN_4x8_ADD,     KERN_4x8_STORE,
    KERN_1x8_ADD,     KERN_1x8_STORE,
    KERN_8x16_ADD,    KERN_8x16_STORE,
    KERN_16x16_ADD,   KERN_16x16_STORE,
    KERN_INVALID
} gemm_kernel_id_t;

//==============================================================================
// SIMPLIFIED PANEL DESCRIPTOR (NO MASKS!)
//==============================================================================

typedef struct {
    size_t j_start;   // Starting column
    size_t j_width;   // Actual width (<= NR)
} panel_info_t;

//==============================================================================
// MEMORY MODES
//==============================================================================

typedef enum {
    GEMM_MEM_STATIC,
    GEMM_MEM_DYNAMIC
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
    // Blocking Parameters
    //--------------------------------------------------------------------------
    size_t MC, KC, NC;
    size_t MR, NR;
    
    //--------------------------------------------------------------------------
    // PRE-COMPUTED EXECUTION METADATA (NEW!)
    //--------------------------------------------------------------------------
    size_t n_nc_tiles;   // (N + NC - 1) / NC
    size_t n_kc_tiles;   // (K + KC - 1) / KC
    size_t n_mc_tiles;   // (M + MC - 1) / MC
    
    //--------------------------------------------------------------------------
    // PRE-SELECTED KERNELS FOR FULL TILES (NEW!)
    //--------------------------------------------------------------------------
    gemm_kernel_id_t kern_full_add;      // Full MR×NR tile (ADD)
    gemm_kernel_id_t kern_full_store;    // Full MR×NR tile (STORE)
    
    //--------------------------------------------------------------------------
    // Panel Descriptors (NO MASKS!)
    //--------------------------------------------------------------------------
    size_t n_npanels;        // Number of N-panels
    panel_info_t *npanels;   // Array of panel descriptors (just j_start, j_width)
    
    //--------------------------------------------------------------------------
    // Memory Strategy
    //--------------------------------------------------------------------------
    gemm_memory_mode_t mem_mode;
    
    float *workspace_a;
    float *workspace_b;
    float *workspace_temp;
    
    size_t workspace_size;
    int workspace_aligned;
    
} gemm_plan_t;

//==============================================================================
// PLAN LIFECYCLE
//==============================================================================

gemm_plan_t *gemm_plan_create(size_t M, size_t K, size_t N);

gemm_plan_t *gemm_plan_create_with_mode(
    size_t M, size_t K, size_t N, 
    gemm_memory_mode_t mode);

void gemm_plan_destroy(gemm_plan_t *plan);

size_t gemm_workspace_query(size_t M, size_t K, size_t N);

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

void gemm_select_blocking(
    size_t M, size_t K, size_t N,
    size_t *MC, size_t *KC, size_t *NC,
    size_t *MR, size_t *NR);

void gemm_select_kernels(
    size_t m_height, size_t n_width,
    gemm_kernel_id_t *kern_add,
    gemm_kernel_id_t *kern_store,
    int *kernel_width);

int gemm_execute_plan(
    gemm_plan_t *plan,
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    float alpha,
    float beta);

#endif // GEMM_PLANNING_H