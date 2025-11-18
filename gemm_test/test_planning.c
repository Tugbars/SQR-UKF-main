/**
 * @file test_planning.c
 * @brief Unit tests for GEMM planning module (No Masks Edition)
 *
 * Tests:
 * 1. Plan creation/destruction
 * 2. Blocking parameter selection
 * 3. Pre-computed tile counts (NEW!)
 * 4. Pre-selected kernel metadata (NEW!)
 * 5. Panel descriptors (simplified, no masks)
 * 6. Static vs dynamic memory modes
 * 7. Edge cases and robustness
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
#include <inttypes.h> // For PRIu64

// Portable size_t printf format
#define FMT_SIZE_T "%lu"
#define CAST_SIZE_T(x) ((unsigned long)(x))

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
    if (plan->max_M != 64 || plan->max_K != 64 || plan->max_N != 64)
    {
        printf("    FAIL: Dimensions incorrect (M=" FMT_SIZE_T ", K=" FMT_SIZE_T ", N=" FMT_SIZE_T ")\n",
               CAST_SIZE_T(plan->max_M), CAST_SIZE_T(plan->max_K), CAST_SIZE_T(plan->max_N));
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

    printf("    Memory mode: STATIC\n");
    printf("    Blocking: MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->MC), CAST_SIZE_T(plan->KC), CAST_SIZE_T(plan->NC));
    printf("    Register: MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->MR), CAST_SIZE_T(plan->NR));

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
    printf("    Workspace size: " FMT_SIZE_T " bytes\n", CAST_SIZE_T(plan->workspace_size));
    printf("    Blocking: MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->MC), CAST_SIZE_T(plan->KC), CAST_SIZE_T(plan->NC));

    gemm_plan_destroy(plan);
    return 1;
}

static int test_plan_explicit_static_too_large(void)
{
    printf("  Testing: Explicit static mode with too-large dimensions\n");

    gemm_plan_t *plan = gemm_plan_create_with_mode(1024, 1024, 1024, GEMM_MEM_STATIC);

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

static int test_blocking_square_medium(void)
{
    printf("  Testing: Blocking for 256×256×256 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(256, 256, 256, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T ", MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T "\n",
           CAST_SIZE_T(MC), CAST_SIZE_T(KC), CAST_SIZE_T(NC), CAST_SIZE_T(MR), CAST_SIZE_T(NR));

    // Verify reasonable blocking
    if (MC > 256 || KC > 256 || NC > 256)
    {
        printf("    FAIL: Blocking params exceed matrix dimensions\n");
        return 0;
    }

    if (MR > 16 || NR > 16)
    {
        printf("    FAIL: Register blocking too large (MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T ")\n",
               CAST_SIZE_T(MR), CAST_SIZE_T(NR));
        return 0;
    }

    // Verify alignment
    if (MC % MR != 0 || NC % NR != 0)
    {
        printf("    FAIL: Cache blocks not aligned to register blocks\n");
        return 0;
    }

    return 1;
}

static int test_blocking_tall_matrix(void)
{
    printf("  Testing: Blocking for tall 1000×100×100 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(1000, 100, 100, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T ", MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T "\n",
           CAST_SIZE_T(MC), CAST_SIZE_T(KC), CAST_SIZE_T(NC), CAST_SIZE_T(MR), CAST_SIZE_T(NR));

    // Tall matrices should favor larger MC
    if (MC < 128)
    {
        printf("    WARN: MC might be too small for tall matrix\n");
    }

    // Verify doesn't exceed dimensions
    if (MC > 1000 || KC > 100 || NC > 100)
    {
        printf("    FAIL: Blocking exceeds dimensions\n");
        return 0;
    }

    return 1;
}

static int test_blocking_wide_matrix(void)
{
    printf("  Testing: Blocking for wide 100×100×1000 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(100, 100, 1000, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T ", MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T "\n",
           CAST_SIZE_T(MC), CAST_SIZE_T(KC), CAST_SIZE_T(NC), CAST_SIZE_T(MR), CAST_SIZE_T(NR));

    // Wide matrices should favor larger NC
    if (NC < 256)
    {
        printf("    WARN: NC might be too small for wide matrix\n");
    }

    // Verify doesn't exceed dimensions
    if (MC > 100 || KC > 100 || NC > 1000)
    {
        printf("    FAIL: Blocking exceeds dimensions\n");
        return 0;
    }

    return 1;
}

static int test_blocking_deep_matrix(void)
{
    printf("  Testing: Blocking for deep 100×1000×100 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(100, 1000, 100, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T ", MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T "\n",
           CAST_SIZE_T(MC), CAST_SIZE_T(KC), CAST_SIZE_T(NC), CAST_SIZE_T(MR), CAST_SIZE_T(NR));

    // Deep matrices should favor larger KC
    if (KC < 256)
    {
        printf("    WARN: KC might be too small for deep matrix\n");
    }

    // Verify doesn't exceed dimensions
    if (MC > 100 || KC > 1000 || NC > 100)
    {
        printf("    FAIL: Blocking exceeds dimensions\n");
        return 0;
    }

    return 1;
}

static int test_blocking_narrow_width(void)
{
    printf("  Testing: Blocking for narrow 128×128×4 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(128, 128, 4, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T ", MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T "\n",
           CAST_SIZE_T(MC), CAST_SIZE_T(KC), CAST_SIZE_T(NC), CAST_SIZE_T(MR), CAST_SIZE_T(NR));

    // Should use 6-column or smaller panels for N=4
    if (NR > 6)
    {
        printf("    FAIL: Should use narrow panel (NR ≤ 6) for N=4\n");
        return 0;
    }

    if (NC > 4)
    {
        printf("    FAIL: NC should not exceed N=4\n");
        return 0;
    }

    return 1;
}

static int test_blocking_tiny_matrix(void)
{
    printf("  Testing: Blocking for tiny 8×8×8 matrix\n");

    size_t MC, KC, NC, MR, NR;
    gemm_select_blocking(8, 8, 8, &MC, &KC, &NC, &MR, &NR);

    printf("    MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T ", MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T "\n",
           CAST_SIZE_T(MC), CAST_SIZE_T(KC), CAST_SIZE_T(NC), CAST_SIZE_T(MR), CAST_SIZE_T(NR));

    // All blocks should fit within dimensions
    if (MC > 8 || KC > 8 || NC > 8)
    {
        printf("    FAIL: Blocking exceeds dimensions\n");
        return 0;
    }

    // Should use minimal blocking
    if (MR > 8 || NR > 8)
    {
        printf("    FAIL: Register blocks too large for 8×8×8\n");
        return 0;
    }

    return 1;
}

//==============================================================================
// TEST 3: Pre-Computed Tile Counts (NEW!)
//==============================================================================

static int test_precomputed_tile_counts_regular(void)
{
    printf("  Testing: Pre-computed tile counts for 256×256×256\n");

    gemm_plan_t *plan = gemm_plan_create(256, 256, 256);
    if (!plan)
        return 0;

    // Manually calculate expected counts
    size_t expected_nc = (256 + plan->NC - 1) / plan->NC;
    size_t expected_kc = (256 + plan->KC - 1) / plan->KC;
    size_t expected_mc = (256 + plan->MC - 1) / plan->MC;

    printf("    Pre-computed: nc=" FMT_SIZE_T ", kc=" FMT_SIZE_T ", mc=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->n_nc_tiles_max), CAST_SIZE_T(plan->n_kc_tiles_max), CAST_SIZE_T(plan->n_mc_tiles_max));
    printf("    Expected:     nc=" FMT_SIZE_T ", kc=" FMT_SIZE_T ", mc=" FMT_SIZE_T "\n",
           CAST_SIZE_T(expected_nc), CAST_SIZE_T(expected_kc), CAST_SIZE_T(expected_mc));

    if (plan->n_nc_tiles_max != expected_nc)
    {
        printf("    FAIL: n_nc_tiles mismatch\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    if (plan->n_kc_tiles_max != expected_kc)
    {
        printf("    FAIL: n_kc_tiles mismatch\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    if (plan->n_mc_tiles_max != expected_mc)
    {
        printf("    FAIL: n_mc_tiles mismatch\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    gemm_plan_destroy(plan);
    return 1;
}

static int test_precomputed_tile_counts_irregular(void)
{
    printf("  Testing: Pre-computed tile counts for 127×259×511 (irregular)\n");

    gemm_plan_t *plan = gemm_plan_create(127, 259, 511);
    if (!plan)
        return 0;

    size_t expected_nc = (511 + plan->NC - 1) / plan->NC;
    size_t expected_kc = (259 + plan->KC - 1) / plan->KC;
    size_t expected_mc = (127 + plan->MC - 1) / plan->MC;

    printf("    Blocking: MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->MC), CAST_SIZE_T(plan->KC), CAST_SIZE_T(plan->NC));
    printf("    Pre-computed: nc=" FMT_SIZE_T ", kc=" FMT_SIZE_T ", mc=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->n_nc_tiles_max), CAST_SIZE_T(plan->n_kc_tiles_max), CAST_SIZE_T(plan->n_mc_tiles_max));

    int match = (plan->n_nc_tiles_max == expected_nc &&
                 plan->n_kc_tiles_max == expected_kc &&
                 plan->n_mc_tiles_max == expected_mc);

    if (!match)
    {
        printf("    FAIL: Tile count mismatch\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    gemm_plan_destroy(plan);
    return 1;
}

static int test_precomputed_tile_counts_edge(void)
{
    printf("  Testing: Pre-computed tile counts for edge case (1×1×1)\n");

    gemm_plan_t *plan = gemm_plan_create(1, 1, 1);
    if (!plan)
        return 0;

    // Even 1×1×1 should have at least 1 tile in each dimension
    if (plan->n_nc_tiles_max == 0 || plan->n_kc_tiles_max == 0 || plan->n_mc_tiles_max == 0)
    {
        printf("    FAIL: Should have at least 1 tile per dimension\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    printf("    Tile counts: nc=" FMT_SIZE_T ", kc=" FMT_SIZE_T ", mc=" FMT_SIZE_T " (all ≥ 1)\n",
           CAST_SIZE_T(plan->n_nc_tiles_max), CAST_SIZE_T(plan->n_kc_tiles_max), CAST_SIZE_T(plan->n_mc_tiles_max));

    gemm_plan_destroy(plan);
    return 1;
}

//==============================================================================
// TEST 4: Pre-Selected Kernels (NEW!)
//==============================================================================

static int test_preselected_kernels_full_tiles(void)
{
    printf("  Testing: Pre-selected kernels for full MR×NR tiles\n");

    gemm_plan_t *plan = gemm_plan_create(256, 256, 256);
    if (!plan)
        return 0;

    printf("    MR=" FMT_SIZE_T ", NR=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->MR), CAST_SIZE_T(plan->NR));
    printf("    kern_full_add:   %d\n", plan->kern_full_add);
    printf("    kern_full_store: %d\n", plan->kern_full_store);

    // Verify kernels are valid (not KERN_INVALID)
    if (plan->kern_full_add == KERN_INVALID || plan->kern_full_store == KERN_INVALID)
    {
        printf("    FAIL: Kernels not initialized\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    // Verify kernels match what gemm_select_kernels would choose
    gemm_kernel_id_t expected_add, expected_store;
    int dummy_width;
    gemm_select_kernels(plan->MR, plan->NR, &expected_add, &expected_store, &dummy_width);

    if (plan->kern_full_add != expected_add || plan->kern_full_store != expected_store)
    {
        printf("    FAIL: Kernel mismatch with gemm_select_kernels\n");
        printf("    Expected: add=%d, store=%d\n", expected_add, expected_store);
        gemm_plan_destroy(plan);
        return 0;
    }

    printf("    Kernels correctly pre-selected\n");

    gemm_plan_destroy(plan);
    return 1;
}

static int test_preselected_kernels_various_sizes(void)
{
    printf("  Testing: Pre-selected kernels for various matrix sizes\n");

    struct
    {
        size_t M, K, N;
        const char *desc;
    } test_cases[] = {
        {64, 64, 64, "Small square"},
        {256, 256, 256, "Medium square"},
        {1024, 1024, 1024, "Large square"},
        {128, 128, 4, "Narrow"},
        {4, 128, 128, "Short"},
        {1000, 100, 100, "Tall"},
        {100, 100, 1000, "Wide"}};

    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++)
    {
        gemm_plan_t *plan = gemm_plan_create(test_cases[i].M, test_cases[i].K, test_cases[i].N);
        if (!plan)
            continue;

        // Verify kernels are valid
        if (plan->kern_full_add == KERN_INVALID || plan->kern_full_store == KERN_INVALID)
        {
            printf("    FAIL: %s - invalid kernels\n", test_cases[i].desc);
            gemm_plan_destroy(plan);
            return 0;
        }

        printf("    %s: MR=" FMT_SIZE_T " NR=" FMT_SIZE_T ", kernels=%d/%d ✓\n",
               test_cases[i].desc,
               CAST_SIZE_T(plan->MR), CAST_SIZE_T(plan->NR),
               plan->kern_full_add, plan->kern_full_store);

        gemm_plan_destroy(plan);
    }

    return 1;
}

//==============================================================================
// TEST 5: Panel Descriptors (Simplified - No Masks)
//==============================================================================

static int test_npanels_full_width(void)
{
    printf("  Testing: N-panels for 64×64×64 (all full width)\n");

    gemm_plan_t *plan = gemm_plan_create(64, 64, 64);
    if (!plan)
        return 0;

    printf("    Number of N-panels: " FMT_SIZE_T " (NR=" FMT_SIZE_T ")\n",
           CAST_SIZE_T(plan->n_npanels), CAST_SIZE_T(plan->NR));

    // Verify all panels are full width
    for (size_t p = 0; p < plan->n_npanels; p++)
    {
        panel_info_t *panel = &plan->npanels[p];

        if (panel->j_width != plan->NR)
        {
            printf("    FAIL: Panel " FMT_SIZE_T " has width " FMT_SIZE_T ", expected " FMT_SIZE_T "\n",
                   CAST_SIZE_T(p), CAST_SIZE_T(panel->j_width), CAST_SIZE_T(plan->NR));
            gemm_plan_destroy(plan);
            return 0;
        }

        if (panel->j_start != p * plan->NR)
        {
            printf("    FAIL: Panel " FMT_SIZE_T " has wrong start position\n", CAST_SIZE_T(p));
            gemm_plan_destroy(plan);
            return 0;
        }
    }

    printf("    All panels full width\n");

    gemm_plan_destroy(plan);
    return 1;
}

static int test_npanels_with_tail(void)
{
    printf("  Testing: N-panels for 64×64×13 (with tail panel)\n");

    gemm_plan_t *plan = gemm_plan_create(64, 64, 13);
    if (!plan)
        return 0;

    printf("    Number of N-panels: " FMT_SIZE_T " (NR=" FMT_SIZE_T ")\n",
           CAST_SIZE_T(plan->n_npanels), CAST_SIZE_T(plan->NR));

    // Find tail panel
    panel_info_t *tail = &plan->npanels[plan->n_npanels - 1];

    size_t expected_tail_width = 13 % plan->NR;
    if (expected_tail_width == 0)
        expected_tail_width = plan->NR;

    printf("    Tail panel: j_start=" FMT_SIZE_T ", j_width=" FMT_SIZE_T " (expected " FMT_SIZE_T ")\n",
           CAST_SIZE_T(tail->j_start), CAST_SIZE_T(tail->j_width), CAST_SIZE_T(expected_tail_width));

    if (tail->j_width != expected_tail_width)
    {
        printf("    FAIL: Tail panel width incorrect\n");
        gemm_plan_destroy(plan);
        return 0;
    }

    // Verify total coverage
    size_t total_width = 0;
    for (size_t p = 0; p < plan->n_npanels; p++)
    {
        total_width += plan->npanels[p].j_width;
    }

    if (total_width != 13)
    {
        printf("    FAIL: Total width = " FMT_SIZE_T ", expected 13\n", CAST_SIZE_T(total_width));
        gemm_plan_destroy(plan);
        return 0;
    }

    gemm_plan_destroy(plan);
    return 1;
}

static int test_npanels_irregular_width(void)
{
    printf("  Testing: N-panels for 64×64×127 (irregular width)\n");

    gemm_plan_t *plan = gemm_plan_create(64, 64, 127);
    if (!plan)
        return 0;

    printf("    Number of N-panels: " FMT_SIZE_T " (NR=" FMT_SIZE_T ")\n",
           CAST_SIZE_T(plan->n_npanels), CAST_SIZE_T(plan->NR));

    // Verify panel coverage and no gaps
    size_t expected_panels = (127 + plan->NR - 1) / plan->NR;

    if (plan->n_npanels != expected_panels)
    {
        printf("    FAIL: Expected " FMT_SIZE_T " panels, got " FMT_SIZE_T "\n",
               CAST_SIZE_T(expected_panels), CAST_SIZE_T(plan->n_npanels));
        gemm_plan_destroy(plan);
        return 0;
    }

    // Check for gaps or overlaps
    for (size_t p = 0; p < plan->n_npanels; p++)
    {
        panel_info_t *panel = &plan->npanels[p];
        size_t expected_start = p * plan->NR;

        if (panel->j_start != expected_start)
        {
            printf("    FAIL: Panel " FMT_SIZE_T " starts at " FMT_SIZE_T ", expected " FMT_SIZE_T "\n",
                   CAST_SIZE_T(p), CAST_SIZE_T(panel->j_start), CAST_SIZE_T(expected_start));
            gemm_plan_destroy(plan);
            return 0;
        }
    }

    printf("    All panels correctly positioned\n");

    gemm_plan_destroy(plan);
    return 1;
}

//==============================================================================
// TEST 6: Workspace
//==============================================================================

static int test_workspace_query(void)
{
    printf("  Testing: Workspace size query for 256×256×256\n");

    size_t size = gemm_workspace_query(256, 256, 256);

    printf("    Workspace required: " FMT_SIZE_T " bytes (%.2f MB)\n",
           CAST_SIZE_T(size), size / (1024.0 * 1024.0));

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

static int test_workspace_alignment(void)
{
    printf("  Testing: Workspace buffer alignment\n");

    gemm_plan_t *plan = gemm_plan_create(256, 256, 256);
    if (!plan)
        return 0;

    int a_aligned = ((uintptr_t)plan->workspace_a % 64 == 0);
    int b_aligned = ((uintptr_t)plan->workspace_b % 64 == 0);
    int temp_aligned = ((uintptr_t)plan->workspace_temp % 64 == 0);

    printf("    workspace_a: %p (%s)\n",
           (void *)plan->workspace_a, a_aligned ? "aligned" : "MISALIGNED");
    printf("    workspace_b: %p (%s)\n",
           (void *)plan->workspace_b, b_aligned ? "aligned" : "MISALIGNED");
    printf("    workspace_temp: %p (%s)\n",
           (void *)plan->workspace_temp, temp_aligned ? "aligned" : "MISALIGNED");

    gemm_plan_destroy(plan);

    return (a_aligned && b_aligned && temp_aligned);
}

static int test_workspace_static_vs_dynamic(void)
{
    printf("  Testing: Workspace pointers for static vs dynamic mode\n");

    // Static mode
    gemm_plan_t *plan_static = gemm_plan_create(64, 64, 64);
    if (!plan_static || plan_static->mem_mode != GEMM_MEM_STATIC)
    {
        printf("    FAIL: Could not create static plan\n");
        gemm_plan_destroy(plan_static);
        return 0;
    }

    printf("    Static: workspace_size=" FMT_SIZE_T " (should be 0)\n",
           CAST_SIZE_T(plan_static->workspace_size));

    if (plan_static->workspace_size != 0)
    {
        printf("    FAIL: Static mode should have workspace_size=0\n");
        gemm_plan_destroy(plan_static);
        return 0;
    }

    // Dynamic mode
    gemm_plan_t *plan_dynamic = gemm_plan_create(1024, 1024, 1024);
    if (!plan_dynamic || plan_dynamic->mem_mode != GEMM_MEM_DYNAMIC)
    {
        printf("    FAIL: Could not create dynamic plan\n");
        gemm_plan_destroy(plan_static);
        gemm_plan_destroy(plan_dynamic);
        return 0;
    }

    printf("    Dynamic: workspace_size=" FMT_SIZE_T " (should be >0)\n",
           CAST_SIZE_T(plan_dynamic->workspace_size));

    if (plan_dynamic->workspace_size == 0)
    {
        printf("    FAIL: Dynamic mode should have workspace_size>0\n");
        gemm_plan_destroy(plan_static);
        gemm_plan_destroy(plan_dynamic);
        return 0;
    }

    gemm_plan_destroy(plan_static);
    gemm_plan_destroy(plan_dynamic);
    return 1;
}

//==============================================================================
// TEST 7: Edge Cases and Robustness
//==============================================================================

static int test_edge_single_tile(void)
{
    printf("  Testing: Single tile matrix (4×4×4)\n");

    gemm_plan_t *plan = gemm_plan_create(4, 4, 4);
    if (!plan)
        return 0;

    printf("    Tile counts: nc=" FMT_SIZE_T ", kc=" FMT_SIZE_T ", mc=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->n_nc_tiles_max), CAST_SIZE_T(plan->n_kc_tiles_max), CAST_SIZE_T(plan->n_mc_tiles_max));

    if (plan->n_nc_tiles_max == 0 || plan->n_kc_tiles_max == 0 || plan->n_mc_tiles_max == 0)
    {
        printf("    FAIL: Should have at least one tile per dimension\n");
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

    printf("    Blocking: MC=" FMT_SIZE_T ", KC=" FMT_SIZE_T ", NC=" FMT_SIZE_T "\n",
           CAST_SIZE_T(plan->MC), CAST_SIZE_T(plan->KC), CAST_SIZE_T(plan->NC));

    // Verify KC doesn't exceed K=10
    if (plan->KC > 10)
    {
        printf("    FAIL: KC=" FMT_SIZE_T " exceeds K=10\n", CAST_SIZE_T(plan->KC));
        gemm_plan_destroy(plan);
        return 0;
    }

    gemm_plan_destroy(plan);
    return 1;
}

static int test_edge_zero_dimension(void)
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

static int test_exact_divisibility(void)
{
    printf("  Testing: Matrix with dimensions exactly divisible by block sizes\n");

    gemm_plan_t *plan = gemm_plan_create(128, 128, 128);
    if (!plan)
        return 0;

    // Verify all panels are full-sized (no tails)
    int all_full = 1;
    for (size_t p = 0; p < plan->n_npanels; p++)
    {
        if (plan->npanels[p].j_width != plan->NR)
        {
            all_full = 0;
            printf("    Panel " FMT_SIZE_T " not full width\n", CAST_SIZE_T(p));
            break;
        }
    }

    if (all_full)
    {
        printf("    All panels full-sized (no tails)\n");
    }

    gemm_plan_destroy(plan);
    return all_full;
}

//==============================================================================
// TEST SUITE RUNNER
//==============================================================================

int run_gemm_planning_tests(test_results_t *results)
{
    results->total = 0;
    results->passed = 0;
    results->failed = 0;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  GEMM Planning Module - Comprehensive Test Suite         ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    gemm_static_init();
    printf("\n" TEST_INFO " Static pool initialized (max dim: %d)\n", GEMM_STATIC_MAX_DIM);

    // Group 1: Plan Creation/Destruction
    printf("\n═══ Test Group 1: Plan Creation/Destruction ═══\n");
    RUN_TEST(results, test_plan_create_destroy_basic);
    RUN_TEST(results, test_plan_create_destroy_large);
    RUN_TEST(results, test_plan_explicit_static_too_large);

    // Group 2: Blocking Parameters
    printf("\n═══ Test Group 2: Blocking Parameters ═══\n");
    RUN_TEST(results, test_blocking_square_medium);
    RUN_TEST(results, test_blocking_tall_matrix);
    RUN_TEST(results, test_blocking_wide_matrix);
    RUN_TEST(results, test_blocking_deep_matrix);
    RUN_TEST(results, test_blocking_narrow_width);
    RUN_TEST(results, test_blocking_tiny_matrix);

    // Group 3: Pre-Computed Tile Counts (NEW!)
    printf("\n═══ Test Group 3: Pre-Computed Tile Counts (NEW!) ═══\n");
    RUN_TEST(results, test_precomputed_tile_counts_regular);
    RUN_TEST(results, test_precomputed_tile_counts_irregular);
    RUN_TEST(results, test_precomputed_tile_counts_edge);

    // Group 4: Pre-Selected Kernels (NEW!)
    printf("\n═══ Test Group 4: Pre-Selected Kernels (NEW!) ═══\n");
    RUN_TEST(results, test_preselected_kernels_full_tiles);
    RUN_TEST(results, test_preselected_kernels_various_sizes);

    // Group 5: Panel Descriptors (Simplified)
    printf("\n═══ Test Group 5: Panel Descriptors (Simplified) ═══\n");
    RUN_TEST(results, test_npanels_full_width);
    RUN_TEST(results, test_npanels_with_tail);
    RUN_TEST(results, test_npanels_irregular_width);

    // Group 6: Workspace
    printf("\n═══ Test Group 6: Workspace ═══\n");
    RUN_TEST(results, test_workspace_query);
    RUN_TEST(results, test_workspace_alignment);
    RUN_TEST(results, test_workspace_static_vs_dynamic);

    // Group 7: Edge Cases & Robustness
    printf("\n═══ Test Group 7: Edge Cases & Robustness ═══\n");
    RUN_TEST(results, test_edge_single_tile);
    RUN_TEST(results, test_edge_very_rectangular);
    RUN_TEST(results, test_edge_zero_dimension);
    RUN_TEST(results, test_static_max_boundary);
    RUN_TEST(results, test_static_pool_reuse);
    RUN_TEST(results, test_exact_divisibility);

    print_test_results("GEMM Planning Module", results);

    return (results->failed == 0) ? 0 : 1;
}

#ifdef STANDALONE
int main(void)
{
    test_results_t results;
    int ret = run_gemm_planning_tests(&results);

    if (ret == 0)
    {
        printf("\n🎉 " TEST_PASS " All tests passed!\n\n");
    }
    else
    {
        printf("\n❌ " TEST_FAIL " %d test(s) failed\n\n", results.failed);
    }

    return ret;
}
#endif