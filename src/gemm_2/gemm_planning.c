/**
 * @file gemm_planning.c
 * @brief GEMM Execution Planning - Simplified (No Masks)
 *
 * Pre-computes:
 * - Tile counts (eliminates division in execution)
 * - Full-tile kernel selection (eliminates selection overhead)
 * - Panel descriptors (just dimensions, no masks)
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

static const size_t ADAPTIVE_MC_TALL = 256;
static const size_t ADAPTIVE_MC_SMALL = 64;
static const size_t ADAPTIVE_KC_TALL = 128;
static const size_t ADAPTIVE_KC_DEEP = 512;
static const size_t ADAPTIVE_NC_WIDE = 512;
static const size_t ADAPTIVE_NC_SMALL = 128;
static const size_t ADAPTIVE_KC_MIN = 32;

void gemm_select_blocking(
    size_t M, size_t K, size_t N,
    size_t *MC, size_t *KC, size_t *NC,
    size_t *MR, size_t *NR)
{
    double aspect_mn = (double)M / (double)N;
    double aspect_kn = (double)K / (double)N;
    
    if (aspect_mn > 3.0) {
        *MC = ADAPTIVE_MC_TALL;
        *KC = ADAPTIVE_KC_TALL;
        *NC = MIN(N, ADAPTIVE_NC_SMALL);
        *MR = 16;
        *NR = (N >= 16) ? 16 : ((N >= 8) ? 8 : 6);
    }
    else if (aspect_mn < 0.33) {
        *MC = MIN(M, ADAPTIVE_MC_SMALL);
        *KC = ADAPTIVE_KC_TALL;
        *NC = ADAPTIVE_NC_WIDE;
        *MR = (M >= 16) ? 16 : ((M >= 8) ? 8 : 4);
        *NR = 16;
    }
    else if (aspect_kn > 4.0) {
        *MC = MIN(M, ADAPTIVE_MC_SMALL);
        *KC = ADAPTIVE_KC_DEEP;
        *NC = MIN(N, ADAPTIVE_NC_SMALL);
        *MR = (M >= 8) ? 8 : 4;
        *NR = (N >= 8) ? 8 : 6;
    }
    else {
        *MC = GEMM_BLOCK_MC;
        *KC = GEMM_BLOCK_KC;
        *NC = GEMM_BLOCK_NC;
        
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
    
    *MC = MIN(*MC, M);
    *KC = MIN(*KC, K);
    *NC = MIN(*NC, N);
    
    const size_t L2_TARGET = 1800 * 1024;
    size_t workspace_bytes = (*MC * *KC + *KC * *NC) * sizeof(float);
    
    if (workspace_bytes > L2_TARGET) {
        double scale = sqrt((double)L2_TARGET / (double)workspace_bytes);
        *MC = ((*MC * scale) / *MR) * *MR;
        *KC = ((*KC * scale) / 8) * 8;
        *NC = ((*NC * scale) / *NR) * *NR;
        *MC = MAX(*MC, *MR);
        *KC = MAX(*KC, ADAPTIVE_KC_MIN);
        *NC = MAX(*NC, *NR);
    }
    
    if (*MC < *MR || *NC < *NR || *KC < 8) {
        *MC = *MR;
        *KC = ADAPTIVE_KC_MIN;
        *NC = *NR;
    }
}

//==============================================================================
// KERNEL SELECTION
//==============================================================================

void gemm_select_kernels(
    size_t m_height, size_t n_width,
    gemm_kernel_id_t *kern_add,
    gemm_kernel_id_t *kern_store,
    int *kernel_width)
{
    if (m_height >= 16 && n_width >= 16) {
        *kern_add = KERN_16x16_ADD;
        *kern_store = KERN_16x16_STORE;
        *kernel_width = 16;
        return;
    }

    if (m_height >= 16 && n_width >= 8 && n_width < 16) {
        *kern_add = KERN_16x8_ADD;
        *kern_store = KERN_16x8_STORE;
        *kernel_width = 8;
        return;
    }

    if (m_height >= 8 && m_height < 16 && n_width >= 16) {
        *kern_add = KERN_8x16_ADD;
        *kern_store = KERN_8x16_STORE;
        *kernel_width = 16;
        return;
    }

    if (m_height >= 16 && n_width >= 6 && n_width < 8) {
        *kern_add = KERN_16x6_ADD;
        *kern_store = KERN_16x6_STORE;
        *kernel_width = 6;
        return;
    }

    if (m_height >= 8 && m_height < 16 && n_width >= 8 && n_width < 16) {
        *kern_add = KERN_8x8_ADD;
        *kern_store = KERN_8x8_STORE;
        *kernel_width = 8;
        return;
    }

    if (m_height >= 8 && m_height < 16 && n_width >= 6 && n_width < 8) {
        *kern_add = KERN_8x6_ADD;
        *kern_store = KERN_8x6_STORE;
        *kernel_width = 6;
        return;
    }

    if (m_height >= 4 && m_height < 8) {
        *kern_add = KERN_4x8_ADD;
        *kern_store = KERN_4x8_STORE;
        *kernel_width = 8;
        return;
    }

    *kern_add = KERN_1x8_ADD;
    *kern_store = KERN_1x8_STORE;
    *kernel_width = 8;
}

//==============================================================================
// PANEL PRE-COMPUTATION (SIMPLIFIED - NO MASKS!)
//==============================================================================

static void precompute_panels(gemm_plan_t *plan)
{
    size_t n_panels = (plan->N + plan->NR - 1) / plan->NR;

    for (size_t p = 0; p < n_panels; p++)
    {
        panel_info_t *panel = &plan->npanels[p];
        
        panel->j_start = p * plan->NR;
        panel->j_width = (panel->j_start + plan->NR <= plan->N)
                            ? plan->NR
                            : (plan->N - panel->j_start);
    }
}

//==============================================================================
// WORKSPACE QUERY
//==============================================================================

size_t gemm_workspace_query(size_t M, size_t K, size_t N)
{
    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(M, K, N, &MC, &KC, &NC, &MR, &NR);

    size_t a_size = MC * KC * sizeof(float);
    size_t b_size = KC * NC * sizeof(float);
    size_t temp_size = MC * NC * sizeof(float);

    a_size = (a_size + 63) & ~(size_t)63;
    b_size = (b_size + 63) & ~(size_t)63;
    temp_size = (temp_size + 63) & ~(size_t)63;

    return a_size + b_size + temp_size;
}

//==============================================================================
// PLAN CREATION
//==============================================================================

gemm_plan_t *gemm_plan_create(size_t M, size_t K, size_t N)
{
    if (M == 0 || K == 0 || N == 0)
        return NULL;

    gemm_memory_mode_t mode = gemm_fits_static(M, K, N)
                                  ? GEMM_MEM_STATIC
                                  : GEMM_MEM_DYNAMIC;

    return gemm_plan_create_with_mode(M, K, N, mode);
}

gemm_plan_t *gemm_plan_create_with_mode(
    size_t M, size_t K, size_t N,
    gemm_memory_mode_t mode)
{
    if (mode == GEMM_MEM_STATIC && !gemm_fits_static(M, K, N))
        return NULL;

    gemm_plan_t *plan = (gemm_plan_t *)calloc(1, sizeof(gemm_plan_t));
    if (!plan)
        return NULL;

    plan->M = M;
    plan->K = K;
    plan->N = N;
    plan->mem_mode = mode;

    //--------------------------------------------------------------------------
    // Select blocking parameters
    //--------------------------------------------------------------------------
    gemm_select_blocking(M, K, N,
                         &plan->MC, &plan->KC, &plan->NC,
                         &plan->MR, &plan->NR);

    //--------------------------------------------------------------------------
    // PRE-COMPUTE TILE COUNTS (NEW!)
    //--------------------------------------------------------------------------
    plan->n_nc_tiles = (N + plan->NC - 1) / plan->NC;
    plan->n_kc_tiles = (K + plan->KC - 1) / plan->KC;
    plan->n_mc_tiles = (M + plan->MC - 1) / plan->MC;

    //--------------------------------------------------------------------------
    // PRE-SELECT KERNELS FOR FULL TILES (NEW!)
    //--------------------------------------------------------------------------
    int dummy_width;
    gemm_select_kernels(plan->MR, plan->NR,
                       &plan->kern_full_add,
                       &plan->kern_full_store,
                       &dummy_width);

    //--------------------------------------------------------------------------
    // Allocate panel descriptors
    //--------------------------------------------------------------------------
    plan->n_npanels = (N + plan->NR - 1) / plan->NR;
    plan->npanels = (panel_info_t *)calloc(plan->n_npanels, sizeof(panel_info_t));

    if (!plan->npanels)
    {
        gemm_plan_destroy(plan);
        return NULL;
    }

    precompute_panels(plan);

    //--------------------------------------------------------------------------
    // Setup Workspace
    //--------------------------------------------------------------------------
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

void gemm_plan_destroy(gemm_plan_t *plan)
{
    if (!plan)
        return;

    free(plan->npanels);

    if (plan->mem_mode == GEMM_MEM_DYNAMIC)
    {
        if (plan->workspace_a)
            gemm_aligned_free(plan->workspace_a);
        if (plan->workspace_b)
            gemm_aligned_free(plan->workspace_b);
        if (plan->workspace_temp)
            gemm_aligned_free(plan->workspace_temp);
    }

    free(plan);
}