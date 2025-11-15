#ifndef GEMM_PLANNING_H
#define GEMM_PLANNING_H

#include <stddef.h>
#include <stdint.h>
#include "gemm_static.h"
#include <immintrin.h>

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
// KERNEL FUNCTION POINTER TYPES
//==============================================================================

// Standard kernel signature (16x8, 8x8, 16x6, 8x6)
typedef void (*gemm_kernel_std_fn)(
    float *restrict c, size_t ldc,
    const float *restrict Ap, size_t a_k_stride,
    const float *restrict Bp, size_t b_k_stride,
    size_t Kblk, size_t m_block, size_t n_block,
    __m256i mask);

// Wide kernel signature (8x16, 16x16)
typedef void (*gemm_kernel_wide_fn)(
    float *restrict c, size_t ldc,
    const float *restrict Ap, size_t a_k_stride,
    const float *restrict Bp, size_t b_k_stride,
    size_t Kblk, size_t m_block, size_t n_block,
    __m256i mask_lo, __m256i mask_hi);

//==============================================================================
// SIMPLIFIED PANEL DESCRIPTOR (NO MASKS!)
//==============================================================================

typedef struct {
    size_t j_start;
    size_t j_width;
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
    // PRE-COMPUTED EXECUTION METADATA
    //--------------------------------------------------------------------------
    size_t n_nc_tiles;
    size_t n_kc_tiles;
    size_t n_mc_tiles;
    
    //--------------------------------------------------------------------------
    // PRE-SELECTED KERNELS FOR FULL TILES
    //--------------------------------------------------------------------------
    gemm_kernel_id_t kern_full_add;      // Kernel ID (for debugging)
    gemm_kernel_id_t kern_full_store;    // Kernel ID (for debugging)
    
    
    // FUNCTION POINTERS (for direct call - eliminates switch overhead)
    gemm_kernel_std_fn kern_full_add_fn;
    gemm_kernel_std_fn kern_full_store_fn;

     // For wide kernels (16-wide):
    gemm_kernel_wide_fn kern_full_add_wide_fn;
    gemm_kernel_wide_fn kern_full_store_wide_fn;
    
    // Flag: 1 if full kernels are 16-wide (need special handling)
    int kern_full_is_wide;
    
    //--------------------------------------------------------------------------
    // Panel Descriptors (NO MASKS!)
    //--------------------------------------------------------------------------
    size_t n_npanels;
    panel_info_t *npanels;
    
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