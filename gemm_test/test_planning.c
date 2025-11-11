/**
 * @file test_gemm_planning.c
 * @brief Unit tests for GEMM planning module
 *
 * Tests:
 * 1. Plan creation/destruction
 * 2. Blocking parameter selection
 * 3. Tile/panel descriptors
 * 4. Mask generation (including dual-mask for 16-wide)
 * 5. Static vs dynamic memory modes
 * 6. Edge cases (small, large, non-square matrices)
 *
 * Compile:
 *   gcc -o test_planning test_gemm_planning.c gemm_planning.c gemm_static.c \
 *       -I. -O2 -march=native -mavx2 -mfma -Wall -Wextra
 *
 * @author TUGBARS
 * @date 2025
 */

#include "gemm_planning.h"
#include "gemm_static.h"
#include "test_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

//==============================================================================
// HELPER: Verify mask correctness
//==============================================================================

static int verify_mask(const __m256i *mask, size_t expected_active_lanes)
{
    int32_t values[8];
    memcpy(values, mask, sizeof(values));  // Now copies from aligned memory
    
    for (size_t i = 0; i < 8; i++) {
        int expected = (i < expected_active_lanes) ? -1 : 0;
        if (values[i] != expected) {
            printf("      Mask mismatch at lane %zu: got %d, expected %d\n", 
                   i, values[i], expected);
            return 0;
        }
    }
    return 1;
}

//==============================================================================
// TEST 1: Plan Creation and Destruction
//==============================================================================

static int test_plan_create_destroy_basic(void)
{
    printf("  Testing: 64×64×64 matrix (should use static pool)\n");

    gemm_plan_t *plan = gemm_plan_create(64, 64, 64);

    if (!plan)
    {
        printf("    FAIL: gemm_plan_create returned NULL\n");
        return 0;
    }

    // Verify dimensions
    if (plan->M != 64 || plan->K != 64 || plan->N != 64)
    {
        printf("    FAIL: Dimensions incorrect (M=%zu, K=%zu, N=%zu)\n",
               plan->M, plan->K, plan->N);
        gemm_plan_destroy(plan);
        return 0;
    }

    // Verify memory mode
    if (plan->mem_mode != GEMM_MEM_STATIC)
    {
        printf("    FAIL: Should use static mode for 64³ matrix\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    // Verify workspace setup
    if (!plan->workspace_a || !plan->workspace_b)
    {
        printf("    FAIL: Workspace not initialized\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    printf("    Memory mode: %s\n",
           plan->mem_mode == GEMM_MEM_STATIC ? "STATIC" : "DYNAMIC");
    printf("    Blocking: MC=%zu, KC=%zu, NC=%zu\n", plan->MC, plan->KC, plan->NC);
    printf("    Register: MR=%zu, NR=%zu\n", plan->MR, plan->NR);

    gemm_plan_destroy(plan);
    return 1;
}

static int test_plan_create_destroy_large(void)
{
    printf("  Testing: 1024×1024×1024 matrix (should use dynamic allocation)\n");

    gemm_plan_t *plan = gemm_plan_create(1024, 1024, 1024);

    if (!plan)
    {
        printf("    FAIL: gemm_plan_create returned NULL\n");
        return 0;
    }

    // Verify memory mode
    if (plan->mem_mode != GEMM_MEM_DYNAMIC)
    {
        printf("    FAIL: Should use dynamic mode for 1024³ matrix\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    // Verify workspace allocated
    if (!plan->workspace_a || !plan->workspace_b)
    {
        printf("    FAIL: Workspace not allocated\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    printf("    Memory mode: DYNAMIC\n");
    printf("    Workspace size: %zu bytes\n", plan->workspace_size);
    printf("    Blocking: MC=%zu, KC=%zu, NC=%zu\n", plan->MC, plan->KC, plan->NC);

    gemm_plan_destroy(plan);
    return 1;
}

static int test_plan_explicit_static_too_large(void)
{
    printf("  Testing: Explicit static mode with too-large dimensions\n");

    // Try to force static mode with dimensions > GEMM_STATIC_MAX_DIM
    gemm_plan_t *plan = gemm_plan_create_with_mode(
        1024, 1024, 1024, GEMM_MEM_STATIC);

    if (plan != NULL)
    {
        printf("    FAIL: Should have rejected static mode for 1024³\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    printf("    Correctly rejected static mode for oversized matrix\n");
    return 1;
}

//==============================================================================
// TEST 2: Blocking Parameter Selection
//==============================================================================

static int test_blocking_small_matrix(void)
{
    printf("  Testing: Blocking for 256×256×256 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(256, 256, 256, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=%zu, KC=%zu, NC=%zu, MR=%zu, NR=%zu\n", MC, KC, NC, MR, NR);

    // Verify reasonable blocking
    if (MC > 256 || KC > 256 || NC > 256)
    {
        printf("    FAIL: Blocking params exceed matrix dimensions\n");
        return 0;
    }

    if (MR > 16 || NR > 16)
    {
        printf("    FAIL: Register blocking too large (MR=%zu, NR=%zu)\n", MR, NR);
        return 0;
    }

    return 1;
}

static int test_blocking_rectangular(void)
{
    printf("  Testing: Blocking for rectangular 100×500×200 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(100, 500, 200, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=%zu, KC=%zu, NC=%zu, MR=%zu, NR=%zu\n", MC, KC, NC, MR, NR);

    // Verify doesn't exceed dimensions
    if (MC > 100 || KC > 500 || NC > 200)
    {
        printf("    FAIL: Blocking exceeds dimensions\n");
        return 0;
    }

    return 1;
}

static int test_blocking_narrow(void)
{
    printf("  Testing: Blocking for narrow 128×128×4 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(128, 128, 4, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=%zu, KC=%zu, NC=%zu, MR=%zu, NR=%zu\n", MC, KC, NC, MR, NR);

    // Should use 6-column panels for N=4
    if (NR > 6)
    {
        printf("    FAIL: Should use narrow panel (NR <= 6) for N=4\n");
        return 0;
    }

    return 1;
}

//==============================================================================
// TEST 3: Tile Descriptor Pre-computation
//==============================================================================

static int test_mtiles_regular(void)
{
    printf("  Testing: M-tile descriptors for 128×64×64 (regular tiles)\n");

    gemm_plan_t *plan = gemm_plan_create(128, 64, 64);
    if (!plan)
        return 0;

    printf("    Number of M-tiles: %zu (MR=%zu)\n", plan->n_mtiles, plan->MR);

    // Verify tile count
    size_t expected_tiles = (128 + plan->MR - 1) / plan->MR;
    if (plan->n_mtiles != expected_tiles)
    {
        printf("    FAIL: Expected %zu tiles, got %zu\n", expected_tiles, plan->n_mtiles);
        gemm_plan_destroy(plan);
        return 0;
    }

    // Verify each tile
    size_t total_rows = 0;
    for (size_t t = 0; t < plan->n_mtiles; t++)
    {
        tile_info_t *tile = &plan->mtiles[t];

        printf("    Tile %zu: start=%zu, height=%zu\n",
               t, tile->i_start, tile->i_height);

        if (tile->i_start != t * plan->MR)
        {
            printf("    FAIL: Tile %zu has wrong start\n", t);
            gemm_plan_destroy(plan);
            return 0;
        }

        total_rows += tile->i_height;
    }

    if (total_rows != 128)
    {
        printf("    FAIL: Total rows = %zu, expected 128\n", total_rows);
        gemm_plan_destroy(plan);
        return 0;
    }

    gemm_plan_destroy(plan);
    return 1;
}

static int test_mtiles_tail(void)
{
    printf("  Testing: M-tile descriptors with tail (67×64×64)\n");

    gemm_plan_t *plan = gemm_plan_create(67, 64, 64);
    if (!plan)
        return 0;

    printf("    Number of M-tiles: %zu (MR=%zu)\n", plan->n_mtiles, plan->MR);

    // Check last tile has correct tail height
    tile_info_t *last_tile = &plan->mtiles[plan->n_mtiles - 1];
    size_t expected_tail = 67 % plan->MR;
    if (expected_tail == 0)
        expected_tail = plan->MR;

    printf("    Last tile: start=%zu, height=%zu (expected tail: %zu)\n",
           last_tile->i_start, last_tile->i_height, expected_tail);

    if (last_tile->i_height != expected_tail)
    {
        printf("    FAIL: Last tile height incorrect\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    gemm_plan_destroy(plan);
    return 1;
}

//==============================================================================
// TEST 4: Panel Descriptor and Mask Generation
//==============================================================================

static int test_npanels_full_width(void)
{
    printf("  Testing: N-panels for 64×64×64 (full width panels)\n");

    gemm_plan_t *plan = gemm_plan_create(64, 64, 64);
    if (!plan)
        return 0;

    printf("    Number of N-panels: %zu (NR=%zu)\n", plan->n_npanels, plan->NR);

    // Verify no masks needed for full-width panels
    for (size_t p = 0; p < plan->n_npanels; p++)
    {
        panel_info_t *panel = &plan->npanels[p];

        if (panel->j_width == plan->NR && panel->needs_mask)
        {
            printf("    FAIL: Panel %zu is full width but needs_mask=1\n", p);
            gemm_plan_destroy(plan);
            return 0;
        }
    }

    printf("    All panels are full width (no masks needed)\n");

    gemm_plan_destroy(plan);
    return 1;
}

static int test_npanels_partial_8wide(void)
{
    printf("  Testing: N-panels for 64×64×13 (partial 8-wide panel)\n");

    gemm_plan_t *plan = gemm_plan_create(64, 64, 13);
    if (!plan)
        return 0;

    printf("    Number of N-panels: %zu (NR=%zu)\n", plan->n_npanels, plan->NR);
    printf("    Number of masks: %zu\n", plan->n_masks);

    // Find the tail panel
    panel_info_t *tail_panel = &plan->npanels[plan->n_npanels - 1];

    printf("    Tail panel: j_start=%zu, j_width=%zu, needs_mask=%d\n",
           tail_panel->j_start, tail_panel->j_width, tail_panel->needs_mask);

    // Verify tail panel needs mask
    if (!tail_panel->needs_mask)
    {
        printf("    FAIL: Tail panel should need mask\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    // Verify mask is correct (5 active lanes for width=5 if NR=8)
    size_t expected_width = 13 % plan->NR;
    if (expected_width == 0)
        expected_width = plan->NR;

    if (tail_panel->j_width != expected_width)
    {
        printf("    FAIL: Tail panel width = %zu, expected %zu\n",
               tail_panel->j_width, expected_width);
        gemm_plan_destroy(plan);
        return 0;
    }

    // Verify mask
    if (!verify_mask(&tail_panel->mask_lo, tail_panel->j_width))
    {
        printf("    FAIL: mask_lo incorrect\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    printf("    Mask verified for %zu active lanes\n", tail_panel->j_width);

    gemm_plan_destroy(plan);
    return 1;
}

static int test_npanels_partial_16wide(void)
{
    printf("  Testing: N-panels for 64×64×21 (partial 16-wide panel)\n");

    gemm_plan_t *plan = gemm_plan_create(64, 64, 21);
    if (!plan)
        return 0;

    printf("    Number of N-panels: %zu (NR=%zu)\n", plan->n_npanels, plan->NR);
    printf("    Number of masks: %zu\n", plan->n_masks);

    // Find tail panel (should be 5 columns: 21 = 16 + 5)
    panel_info_t *tail_panel = &plan->npanels[plan->n_npanels - 1];

    printf("    Tail panel: j_start=%zu, j_width=%zu\n",
           tail_panel->j_start, tail_panel->j_width);

    size_t expected_width = 21 % plan->NR;
    if (expected_width == 0)
        expected_width = plan->NR;

    if (tail_panel->j_width != expected_width)
    {
        printf("    FAIL: Tail width incorrect\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    // For 16-wide with j_width=5, should use only mask_lo
    if (!verify_mask(&tail_panel->mask_lo, tail_panel->j_width))
    {
        printf("    FAIL: mask_lo incorrect\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    printf("    Masks verified for width=%zu\n", expected_width);

    gemm_plan_destroy(plan);
    return 1;
}

static int test_npanels_dual_mask_16wide(void)
{
    printf("  Testing: N-panels for 64×64×27 (dual mask: 16+11)\n");

    gemm_plan_t *plan = gemm_plan_create(64, 64, 27);
    if (!plan)
        return 0;

    printf("    Number of N-panels: %zu (NR=%zu)\n", plan->n_npanels, plan->NR);
    printf("    Number of masks: %zu\n", plan->n_masks);

    // Find tail panel (27 = 16 + 11)
    panel_info_t *tail_panel = &plan->npanels[plan->n_npanels - 1];

    printf("    Tail panel: j_start=%zu, j_width=%zu\n",
           tail_panel->j_start, tail_panel->j_width);

    size_t expected_width = 27 % plan->NR;
    if (expected_width == 0)
        expected_width = plan->NR;

    if (tail_panel->j_width != expected_width)
    {
        printf("    FAIL: Width = %zu, expected %zu\n",
               tail_panel->j_width, expected_width);
        gemm_plan_destroy(plan);
        return 0;
    }

    // For j_width=11 with NR=16:
    // - mask_lo should be full (8 lanes)
    // - mask_hi should have 3 lanes (11-8=3)
    if (expected_width > 8)
    {
        if (!verify_mask(&tail_panel->mask_lo, 8))
        {
            printf("    FAIL: mask_lo should be full\n");
            gemm_plan_destroy(plan);
            return 0;
        }

        if (!verify_mask(&tail_panel->mask_hi, expected_width - 8))
        {
            printf("    FAIL: mask_hi should have %zu lanes\n", expected_width - 8);
            gemm_plan_destroy(plan);
            return 0;
        }

        printf("    Dual masks verified: lo=full, hi=%zu lanes\n", expected_width - 8);
    }

    gemm_plan_destroy(plan);
    return 1;
}

//==============================================================================
// TEST 5: Workspace Size Calculation
//==============================================================================

static int test_workspace_query(void)
{
    printf("  Testing: Workspace size query for 256×256×256\n");

    size_t size = gemm_workspace_query(256, 256, 256);

    printf("    Workspace required: %zu bytes (%.2f MB)\n",
           size, size / (1024.0 * 1024.0));

    // Sanity check: should be > 0 and reasonable
    if (size == 0)
    {
        printf("    FAIL: Workspace size is zero\n");
        return 0;
    }

    if (size > 1024 * 1024 * 1024)
    {
        printf("    FAIL: Workspace unreasonably large (>1GB)\n");
        return 0;
    }

    return 1;
}

//==============================================================================
// TEST 6: Edge Cases
//==============================================================================

static int test_edge_single_tile(void)
{
    printf("  Testing: Single tile matrix (4×4×4)\n");

    gemm_plan_t *plan = gemm_plan_create(4, 4, 4);
    if (!plan)
        return 0;

    printf("    n_mtiles=%zu, n_npanels=%zu\n", plan->n_mtiles, plan->n_npanels);

    // Should have at least 1 tile
    if (plan->n_mtiles == 0 || plan->n_npanels == 0)
    {
        printf("    FAIL: Should have at least one tile/panel\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    gemm_plan_destroy(plan);
    return 1;
}

static int test_edge_very_rectangular(void)
{
    printf("  Testing: Very rectangular matrix (1000×10×1000)\n");

    gemm_plan_t *plan = gemm_plan_create(1000, 10, 1000);
    if (!plan)
        return 0;

    printf("    Blocking: MC=%zu, KC=%zu, NC=%zu\n", plan->MC, plan->KC, plan->NC);

    // Verify KC is reasonable for K=10
    if (plan->KC > 10)
    {
        printf("    FAIL: KC=%zu exceeds K=10\n", plan->KC);
        gemm_plan_destroy(plan);
        return 0;
    }

    gemm_plan_destroy(plan);
    return 1;
}

static int test_mask_generation_edge_cases(void)
{
    printf("  Testing: gemm_build_mask_avx2 edge cases\n");

    // FIXED: Explicitly align stack locals
#if defined(__GNUC__) || defined(__clang__)
    __m256i mask0 __attribute__((aligned(32)));
    __m256i mask8 __attribute__((aligned(32)));
    __m256i mask_clamp __attribute__((aligned(32)));
#elif defined(_MSC_VER)
    __declspec(align(32)) __m256i mask0;
    __declspec(align(32)) __m256i mask8;
    __declspec(align(32)) __m256i mask_clamp;
#else
    #error "Unsupported compiler - cannot guarantee __m256i alignment"
#endif

    // Test n=0 (should return all zeros)
    gemm_build_mask_avx2(0, &mask0);
    if (!verify_mask(&mask0, 0))
    {
        printf("    FAIL: n=0 mask incorrect\n");
        return 0;
    }

    // Test n=8 (full mask)
    gemm_build_mask_avx2(8, &mask8);
    if (!verify_mask(&mask8, 8))
    {
        printf("    FAIL: n=8 mask should be all -1\n");
        return 0;
    }

    // Test n>8 (should clamp to 8)
    gemm_build_mask_avx2(15, &mask_clamp);
    if (!verify_mask(&mask_clamp, 8))
    {
        printf("    FAIL: n>8 should clamp to 8\n");
        return 0;
    }

    printf("    All mask edge cases correct\n");
    return 1;
}

static int test_exact_divisibility(void)
{
    printf("  Testing: Matrix with dimensions exactly divisible by block sizes\n");

    // Use 128×128×128 which is exactly divisible by default blocks
    gemm_plan_t *plan = gemm_plan_create(128, 128, 128);
    if (!plan)
        return 0;

    // Verify every tile/panel is full-sized (no tails)
    int all_full = 1;
    for (size_t t = 0; t < plan->n_mtiles; t++)
    {
        if (plan->mtiles[t].i_height != plan->MR)
        {
            all_full = 0;
            printf("    FAIL: Tile %zu not full height\n", t);
            break;
        }
    }

    for (size_t p = 0; p < plan->n_npanels; p++)
    {
        if (plan->npanels[p].j_width != plan->NR)
        {
            all_full = 0;
            printf("    FAIL: Panel %zu not full width\n", p);
            break;
        }
    }

    if (all_full)
    {
        printf("    All tiles and panels full-sized (no tails)\n");
    }

    gemm_plan_destroy(plan);
    return all_full;
}

static int test_static_pool_reuse(void)
{
    printf("  Testing: Static pool pointer stability across multiple plans\n");

    gemm_plan_t *plan1 = gemm_plan_create(64, 64, 64);
    gemm_plan_t *plan2 = gemm_plan_create(64, 64, 64);

    if (!plan1 || !plan2)
    {
        gemm_plan_destroy(plan1);
        gemm_plan_destroy(plan2);
        return 0;
    }

    // Verify static pool pointers are identical (true reuse)
    int a_match = (plan1->workspace_a == plan2->workspace_a);
    int b_match = (plan1->workspace_b == plan2->workspace_b);
    int temp_match = (plan1->workspace_temp == plan2->workspace_temp);

    printf("    workspace_a same: %s\n", a_match ? "YES" : "NO");
    printf("    workspace_b same: %s\n", b_match ? "YES" : "NO");
    printf("    workspace_temp same: %s\n", temp_match ? "YES" : "NO");

    gemm_plan_destroy(plan1);
    gemm_plan_destroy(plan2);

    return (a_match && b_match && temp_match);
}

static int test_zero_dimension(void)
{
    printf("  Testing: Zero-dimension matrices (should fail gracefully)\n");

    gemm_plan_t *plan_M0 = gemm_plan_create(0, 64, 64);
    gemm_plan_t *plan_K0 = gemm_plan_create(64, 0, 64);
    gemm_plan_t *plan_N0 = gemm_plan_create(64, 64, 0);

    int all_null = (plan_M0 == NULL && plan_K0 == NULL && plan_N0 == NULL);

    printf("    M=0: %s\n", plan_M0 ? "CREATED (BUG)" : "REJECTED (OK)");
    printf("    K=0: %s\n", plan_K0 ? "CREATED (BUG)" : "REJECTED (OK)");
    printf("    N=0: %s\n", plan_N0 ? "CREATED (BUG)" : "REJECTED (OK)");

    gemm_plan_destroy(plan_M0);
    gemm_plan_destroy(plan_K0);
    gemm_plan_destroy(plan_N0);

    return all_null;
}

static int test_static_max_boundary(void)
{
    printf("  Testing: Maximum static dimension boundary (%d)\n", GEMM_STATIC_MAX_DIM);

    // Test at exact boundary
    gemm_plan_t *plan_exact = gemm_plan_create(
        GEMM_STATIC_MAX_DIM, GEMM_STATIC_MAX_DIM, GEMM_STATIC_MAX_DIM);

    if (!plan_exact || plan_exact->mem_mode != GEMM_MEM_STATIC)
    {
        printf("    FAIL: Should use static mode at exact boundary\n");
        gemm_plan_destroy(plan_exact);
        return 0;
    }
    printf("    Boundary: STATIC ✓\n");

    // Test at boundary+1
    int dim_over = GEMM_STATIC_MAX_DIM + 1;
    gemm_plan_t *plan_over = gemm_plan_create(dim_over, dim_over, dim_over);

    if (!plan_over || plan_over->mem_mode != GEMM_MEM_DYNAMIC)
    {
        printf("    FAIL: Should use dynamic mode when exceeding boundary\n");
        gemm_plan_destroy(plan_exact);
        gemm_plan_destroy(plan_over);
        return 0;
    }
    printf("    Boundary+1: DYNAMIC ✓\n");

    gemm_plan_destroy(plan_exact);
    gemm_plan_destroy(plan_over);
    return 1;
}

static int test_workspace_alignment(void)
{
    printf("  Testing: Workspace buffer alignment\n");

    gemm_plan_t *plan = gemm_plan_create(256, 256, 256);
    if (!plan)
        return 0;

    int a_aligned = ((uintptr_t)plan->workspace_a % 64 == 0);
    int b_aligned = ((uintptr_t)plan->workspace_b % 64 == 0);
    int temp_aligned = ((uintptr_t)plan->workspace_temp % 64 == 0);

    printf("    workspace_a: %p (aligned: %s)\n",
           (void *)plan->workspace_a, a_aligned ? "YES" : "NO");
    printf("    workspace_b: %p (aligned: %s)\n",
           (void *)plan->workspace_b, b_aligned ? "YES" : "NO");
    printf("    workspace_temp: %p (aligned: %s)\n",
           (void *)plan->workspace_temp, temp_aligned ? "YES" : "NO");

    gemm_plan_destroy(plan);

    return (a_aligned && b_aligned && temp_aligned);
}

//==============================================================================
// TEST SUITE RUNNER
//==============================================================================

/**
 * @brief Run all planning module tests
 * @param results Output: test results structure
 * @return 0 if all tests passed, 1 if any failed
 */
int run_gemm_planning_tests(test_results_t *results)
{
    // Initialize results
    results->total = 0;
    results->passed = 0;
    results->failed = 0;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  GEMM Planning Module - Comprehensive Test Suite         ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    // Initialize static pool
    gemm_static_init();
    printf("\n" TEST_INFO " Static pool initialized (max dim: %d)\n",
           GEMM_STATIC_MAX_DIM);

    // Group 1: Plan Creation/Destruction
    printf("\n═══ Test Group 1: Plan Creation/Destruction ═══\n");
    RUN_TEST(results, test_plan_create_destroy_basic);
    RUN_TEST(results, test_plan_create_destroy_large);
    RUN_TEST(results, test_plan_explicit_static_too_large);

    // Group 2: Blocking Parameters
    printf("\n═══ Test Group 2: Blocking Parameters ═══\n");
    RUN_TEST(results, test_blocking_small_matrix);
    RUN_TEST(results, test_blocking_rectangular);
    RUN_TEST(results, test_blocking_narrow);

    // Group 3: Tile Descriptors
    printf("\n═══ Test Group 3: Tile Descriptors ═══\n");
    RUN_TEST(results, test_mtiles_regular);
    RUN_TEST(results, test_mtiles_tail);

    // Group 4: Panel Descriptors and Masks
    printf("\n═══ Test Group 4: Panel Descriptors and Masks ═══\n");
    RUN_TEST(results, test_npanels_full_width);
    RUN_TEST(results, test_npanels_partial_8wide);
    RUN_TEST(results, test_npanels_partial_16wide);
    RUN_TEST(results, test_npanels_dual_mask_16wide);

    // Group 5: Workspace
    printf("\n═══ Test Group 5: Workspace ═══\n");
    RUN_TEST(results, test_workspace_query);

    // Group 6: Edge Cases
    printf("\n═══ Test Group 6: Edge Cases ═══\n");
    RUN_TEST(results, test_edge_single_tile);
    RUN_TEST(results, test_edge_very_rectangular);

    // Group 7: Edge Cases & Robustness
    printf("\n═══ Test Group 7: Edge Cases & Robustness ═══\n");
    RUN_TEST(results, test_mask_generation_edge_cases);
    RUN_TEST(results, test_exact_divisibility);
    RUN_TEST(results, test_zero_dimension);
    RUN_TEST(results, test_static_max_boundary);

    // Group 8: Performance Invariants
    printf("\n═══ Test Group 8: Performance Invariants ═══\n");
    RUN_TEST(results, test_static_pool_reuse);
    RUN_TEST(results, test_workspace_alignment);

    // Print results
    print_test_results("GEMM Planning Module - Results", results);

 
   return (results->failed == 0) ? 0 : 1;
}
// Optional: Standalone mode
#ifdef STANDALONE
int main(void)
{
    test_results_t results;
    int ret = run_gemm_planning_tests(&results);
    
    if (ret == 0) {
        printf("\n🎉 " TEST_PASS " All tests passed!\n\n");
    } else {
        printf("\n❌ " TEST_FAIL " %d test(s) failed\n\n", results.failed);
    }
    
    return ret;
}
#endif