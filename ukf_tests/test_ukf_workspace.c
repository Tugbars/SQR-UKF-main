/**
 * @file test_ukf_workspace.c
 * @brief Unit tests for SR-UKF workspace management
 *
 * Tests:
 * - ukf_qr_ws_ensure / ukf_qr_ws_cleanup: QR workspace lifecycle
 * - ukf_upd_ws_ensure / ukf_upd_ws_cleanup: Update workspace lifecycle
 * - ukf_pxy_ws_ensure / ukf_pxy_ws_cleanup: Cross-covariance workspace lifecycle
 * - Capacity growth behavior (ensure larger → no realloc, ensure smaller → no shrink)
 * - Double-free safety
 * - NULL-safety
 * - Zero-initialization requirements
 *
 * @author TUGBARS
 * @date 2025
 */

#include "test_common.h"
#include "sqr_ukf.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//==============================================================================
// TEST: ukf_qr_ws_t lifecycle
//==============================================================================

/**
 * @brief Test QR workspace allocation and cleanup
 */
static int test_qr_ws_basic_lifecycle(void)
{
    printf("\n=== Testing ukf_qr_ws_t basic lifecycle ===\n");
    
    int passed = 1;
    
    ukf_qr_ws_t ws = {0};  /* Must be zero-initialized */
    
    /* Test initial ensure */
    printf("  Testing initial ensure (L=16)...\n");
    int rc = ukf_qr_ws_ensure(&ws, 16);
    
    if (rc != 0)
    {
        printf("    FAILED: ensure returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* Verify allocations */
    if (!ws.Aprime || !ws.R_ || !ws.b || !ws.qr_ws)
    {
        printf("    FAILED: one or more buffers not allocated\n");
        printf("    Aprime=%p, R_=%p, b=%p, qr_ws=%p\n",
               (void*)ws.Aprime, (void*)ws.R_, (void*)ws.b, (void*)ws.qr_ws);
        passed = 0;
        goto cleanup;
    }
    
    /* Verify capacity recorded */
    if (ws.capL != 16)
    {
        printf("    FAILED: capL=%zu, expected 16\n", ws.capL);
        passed = 0;
        goto cleanup;
    }
    
    printf("    Initial allocation PASSED\n");
    
    /* Test re-ensure with same size (should be fast path, no realloc) */
    printf("  Testing re-ensure same size (L=16)...\n");
    float *old_Aprime = ws.Aprime;
    
    rc = ukf_qr_ws_ensure(&ws, 16);
    
    if (rc != 0)
    {
        printf("    FAILED: re-ensure returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    if (ws.Aprime != old_Aprime)
    {
        printf("    FAILED: unnecessary reallocation on same-size ensure\n");
        passed = 0;
    }
    else
    {
        printf("    Same-size re-ensure PASSED (no realloc)\n");
    }
    
    /* Test ensure with smaller size (should NOT shrink) */
    printf("  Testing ensure smaller size (L=8)...\n");
    old_Aprime = ws.Aprime;
    
    rc = ukf_qr_ws_ensure(&ws, 8);
    
    if (rc != 0)
    {
        printf("    FAILED: smaller ensure returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    if (ws.Aprime != old_Aprime)
    {
        printf("    FAILED: unnecessary reallocation on smaller ensure\n");
        passed = 0;
    }
    
    /* capL should NOT shrink */
    if (ws.capL < 16)
    {
        printf("    FAILED: capL shrunk to %zu\n", ws.capL);
        passed = 0;
    }
    else
    {
        printf("    Smaller ensure PASSED (no shrink, capL=%zu)\n", ws.capL);
    }
    
    /* Test ensure with larger size (should reallocate) */
    printf("  Testing ensure larger size (L=32)...\n");
    
    rc = ukf_qr_ws_ensure(&ws, 32);
    
    if (rc != 0)
    {
        printf("    FAILED: larger ensure returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    if (ws.capL < 32)
    {
        printf("    FAILED: capL=%zu after ensure(32)\n", ws.capL);
        passed = 0;
    }
    else
    {
        printf("    Larger ensure PASSED (capL=%zu)\n", ws.capL);
    }
    
cleanup:
    /* Test cleanup */
    printf("  Testing cleanup...\n");
    ukf_qr_ws_cleanup(&ws);
    
    if (ws.Aprime != NULL || ws.R_ != NULL || ws.b != NULL || 
        ws.qr_ws != NULL || ws.capL != 0)
    {
        printf("    FAILED: cleanup did not reset all fields\n");
        passed = 0;
    }
    else
    {
        printf("    Cleanup PASSED\n");
    }
    
    return passed;
}

/**
 * @brief Test QR workspace double-free safety
 */
static int test_qr_ws_double_free(void)
{
    printf("\n=== Testing ukf_qr_ws_t double-free safety ===\n");
    
    ukf_qr_ws_t ws = {0};
    
    int rc = ukf_qr_ws_ensure(&ws, 8);
    if (rc != 0)
    {
        printf("  FAILED: initial ensure failed\n");
        return 0;
    }
    
    /* First cleanup */
    ukf_qr_ws_cleanup(&ws);
    
    /* Second cleanup (should be safe) */
    ukf_qr_ws_cleanup(&ws);
    
    /* Third cleanup on zero-init (should be safe) */
    ukf_qr_ws_t ws2 = {0};
    ukf_qr_ws_cleanup(&ws2);
    
    printf("  Double-free safety PASSED\n");
    return 1;
}

/**
 * @brief Test QR workspace NULL safety
 */
static int test_qr_ws_null_safety(void)
{
    printf("\n=== Testing ukf_qr_ws_t NULL safety ===\n");
    
    /* cleanup(NULL) should not crash */
    ukf_qr_ws_cleanup(NULL);
    
    printf("  NULL cleanup PASSED\n");
    return 1;
}

//==============================================================================
// TEST: ukf_upd_ws_t lifecycle
//==============================================================================

/**
 * @brief Test update workspace allocation and cleanup
 */
static int test_upd_ws_basic_lifecycle(void)
{
    printf("\n=== Testing ukf_upd_ws_t basic lifecycle ===\n");
    
    int passed = 1;
    
    ukf_upd_ws_t ws = {0};
    
    /* Test initial ensure */
    printf("  Testing initial ensure (n=16)...\n");
    int rc = ukf_upd_ws_ensure(&ws, 16);
    
    if (rc != 0)
    {
        printf("    FAILED: ensure returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    /* Verify all allocations */
    if (!ws.Z || !ws.U || !ws.Ut || !ws.Ky || !ws.yyhat || !ws.Uk)
    {
        printf("    FAILED: buffer allocation incomplete\n");
        passed = 0;
        goto cleanup;
    }
    
    if (!ws.gemm_plan || !ws.trsm_gemm_plan || !ws.chol_ws || !ws.trsm_ws)
    {
        printf("    FAILED: sub-workspace allocation incomplete\n");
        printf("    gemm_plan=%p, trsm_gemm_plan=%p, chol_ws=%p, trsm_ws=%p\n",
               (void*)ws.gemm_plan, (void*)ws.trsm_gemm_plan,
               (void*)ws.chol_ws, (void*)ws.trsm_ws);
        passed = 0;
        goto cleanup;
    }
    
    if (ws.cap != 16 * 16)
    {
        printf("    FAILED: cap=%zu, expected %d\n", ws.cap, 16*16);
        passed = 0;
        goto cleanup;
    }
    
    printf("    Initial allocation PASSED\n");
    
    /* Test re-ensure same size */
    printf("  Testing re-ensure same size (n=16)...\n");
    float *old_Z = ws.Z;
    
    rc = ukf_upd_ws_ensure(&ws, 16);
    
    if (rc != 0 || ws.Z != old_Z)
    {
        printf("    FAILED: unnecessary reallocation\n");
        passed = 0;
    }
    else
    {
        printf("    Same-size re-ensure PASSED\n");
    }
    
    /* Test ensure smaller */
    printf("  Testing ensure smaller (n=8)...\n");
    old_Z = ws.Z;
    
    rc = ukf_upd_ws_ensure(&ws, 8);
    
    if (rc != 0 || ws.Z != old_Z)
    {
        printf("    FAILED: unnecessary reallocation on smaller\n");
        passed = 0;
    }
    else
    {
        printf("    Smaller ensure PASSED (no realloc)\n");
    }
    
    /* Test ensure larger */
    printf("  Testing ensure larger (n=32)...\n");
    
    rc = ukf_upd_ws_ensure(&ws, 32);
    
    if (rc != 0)
    {
        printf("    FAILED: larger ensure returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    if (ws.cap < 32 * 32)
    {
        printf("    FAILED: cap=%zu after ensure(32)\n", ws.cap);
        passed = 0;
    }
    else
    {
        printf("    Larger ensure PASSED (cap=%zu)\n", ws.cap);
    }
    
cleanup:
    printf("  Testing cleanup...\n");
    ukf_upd_ws_cleanup(&ws);
    
    if (ws.Z != NULL || ws.U != NULL || ws.gemm_plan != NULL || 
        ws.chol_ws != NULL || ws.cap != 0)
    {
        printf("    FAILED: cleanup incomplete\n");
        passed = 0;
    }
    else
    {
        printf("    Cleanup PASSED\n");
    }
    
    return passed;
}

/**
 * @brief Test update workspace double-free safety
 */
static int test_upd_ws_double_free(void)
{
    printf("\n=== Testing ukf_upd_ws_t double-free safety ===\n");
    
    ukf_upd_ws_t ws = {0};
    
    int rc = ukf_upd_ws_ensure(&ws, 8);
    if (rc != 0)
    {
        printf("  FAILED: initial ensure failed\n");
        return 0;
    }
    
    ukf_upd_ws_cleanup(&ws);
    ukf_upd_ws_cleanup(&ws);  /* Should be safe */
    
    ukf_upd_ws_t ws2 = {0};
    ukf_upd_ws_cleanup(&ws2);  /* Zero-init cleanup should be safe */
    
    printf("  Double-free safety PASSED\n");
    return 1;
}

/**
 * @brief Test update workspace NULL safety
 */
static int test_upd_ws_null_safety(void)
{
    printf("\n=== Testing ukf_upd_ws_t NULL safety ===\n");
    
    ukf_upd_ws_cleanup(NULL);
    
    printf("  NULL cleanup PASSED\n");
    return 1;
}

//==============================================================================
// TEST: ukf_pxy_ws_t lifecycle
//==============================================================================

/**
 * @brief Test cross-covariance workspace allocation and cleanup
 */
static int test_pxy_ws_basic_lifecycle(void)
{
    printf("\n=== Testing ukf_pxy_ws_t basic lifecycle ===\n");
    
    int passed = 1;
    
    ukf_pxy_ws_t ws = {0};
    
    const size_t L = 16;
    const size_t N8 = 40;  /* Rounded up from 2*16+1=33 */
    
    /* Test initial ensure */
    printf("  Testing initial ensure (L=%zu, N8=%zu)...\n", L, N8);
    int rc = ukf_pxy_ws_ensure(&ws, L, N8);
    
    if (rc != 0)
    {
        printf("    FAILED: ensure returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    if (!ws.Xc || !ws.Y_centered || !ws.YTc || !ws.gemm_plan)
    {
        printf("    FAILED: allocation incomplete\n");
        passed = 0;
        goto cleanup;
    }
    
    if (ws.capL != L || ws.capN8 != N8)
    {
        printf("    FAILED: capacity mismatch (capL=%zu, capN8=%zu)\n",
               ws.capL, ws.capN8);
        passed = 0;
        goto cleanup;
    }
    
    printf("    Initial allocation PASSED\n");
    
    /* Test re-ensure same size */
    printf("  Testing re-ensure same size...\n");
    float *old_Xc = ws.Xc;
    
    rc = ukf_pxy_ws_ensure(&ws, L, N8);
    
    if (rc != 0 || ws.Xc != old_Xc)
    {
        printf("    FAILED: unnecessary reallocation\n");
        passed = 0;
    }
    else
    {
        printf("    Same-size PASSED\n");
    }
    
    /* Test ensure smaller */
    printf("  Testing ensure smaller (L=8, N8=24)...\n");
    old_Xc = ws.Xc;
    
    rc = ukf_pxy_ws_ensure(&ws, 8, 24);
    
    if (rc != 0 || ws.Xc != old_Xc)
    {
        printf("    FAILED: unnecessary reallocation\n");
        passed = 0;
    }
    else
    {
        printf("    Smaller PASSED (no realloc)\n");
    }
    
    /* Test ensure larger L */
    printf("  Testing ensure larger L (L=32, N8=72)...\n");
    
    rc = ukf_pxy_ws_ensure(&ws, 32, 72);
    
    if (rc != 0)
    {
        printf("    FAILED: larger ensure returned %d\n", rc);
        passed = 0;
        goto cleanup;
    }
    
    if (ws.capL < 32 || ws.capN8 < 72)
    {
        printf("    FAILED: capacity not grown\n");
        passed = 0;
    }
    else
    {
        printf("    Larger PASSED (capL=%zu, capN8=%zu)\n", ws.capL, ws.capN8);
    }
    
cleanup:
    printf("  Testing cleanup...\n");
    ukf_pxy_ws_cleanup(&ws);
    
    if (ws.Xc != NULL || ws.YTc != NULL || ws.gemm_plan != NULL ||
        ws.capL != 0 || ws.capN8 != 0)
    {
        printf("    FAILED: cleanup incomplete\n");
        passed = 0;
    }
    else
    {
        printf("    Cleanup PASSED\n");
    }
    
    return passed;
}

/**
 * @brief Test cross-covariance workspace double-free safety
 */
static int test_pxy_ws_double_free(void)
{
    printf("\n=== Testing ukf_pxy_ws_t double-free safety ===\n");
    
    ukf_pxy_ws_t ws = {0};
    
    int rc = ukf_pxy_ws_ensure(&ws, 8, 24);
    if (rc != 0)
    {
        printf("  FAILED: initial ensure failed\n");
        return 0;
    }
    
    ukf_pxy_ws_cleanup(&ws);
    ukf_pxy_ws_cleanup(&ws);
    
    ukf_pxy_ws_t ws2 = {0};
    ukf_pxy_ws_cleanup(&ws2);
    
    printf("  Double-free safety PASSED\n");
    return 1;
}

/**
 * @brief Test cross-covariance workspace NULL safety
 */
static int test_pxy_ws_null_safety(void)
{
    printf("\n=== Testing ukf_pxy_ws_t NULL safety ===\n");
    
    ukf_pxy_ws_cleanup(NULL);
    
    printf("  NULL cleanup PASSED\n");
    return 1;
}

//==============================================================================
// TEST: Workspace reuse across multiple operations
//==============================================================================

/**
 * @brief Test workspace reuse pattern (simulates multiple UKF iterations)
 */
static int test_workspace_reuse_pattern(void)
{
    printf("\n=== Testing workspace reuse pattern ===\n");
    
    int passed = 1;
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    
    /* Simulate multiple UKF iterations with same state size */
    const uint8_t L = 16;
    const uint16_t n = 16;
    
    printf("  Simulating %d UKF iterations with L=%d...\n", 100, L);
    
    for (int iter = 0; iter < 100; iter++)
    {
        /* Ensure workspaces (should be fast path after first iteration) */
        int rc1 = ukf_qr_ws_ensure(&qr_ws, L);
        int rc2 = ukf_upd_ws_ensure(&upd_ws, n);
        
        if (rc1 != 0 || rc2 != 0)
        {
            printf("    FAILED at iteration %d: ensure failed\n", iter);
            passed = 0;
            break;
        }
        
        /* Touch the buffers to verify they're valid */
        qr_ws.Aprime[0] = (float)iter;
        qr_ws.R_[0] = (float)iter;
        upd_ws.Z[0] = (float)iter;
        upd_ws.U[0] = (float)iter;
    }
    
    if (passed)
    {
        printf("    Reuse pattern PASSED\n");
    }
    
    /* Cleanup */
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    
    return passed;
}

/**
 * @brief Test workspace growth pattern (varying state sizes)
 */
static int test_workspace_growth_pattern(void)
{
    printf("\n=== Testing workspace growth pattern ===\n");
    
    int passed = 1;
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    
    /* Start small, grow progressively */
    uint8_t sizes[] = {4, 8, 16, 32, 64, 32, 16, 64};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    size_t max_qr_cap = 0;
    size_t max_upd_cap = 0;
    
    for (int i = 0; i < num_sizes; i++)
    {
        uint8_t L = sizes[i];
        printf("  Ensure L=%d...\n", L);
        
        int rc1 = ukf_qr_ws_ensure(&qr_ws, L);
        int rc2 = ukf_upd_ws_ensure(&upd_ws, L);
        
        if (rc1 != 0 || rc2 != 0)
        {
            printf("    FAILED: ensure failed for L=%d\n", L);
            passed = 0;
            break;
        }
        
        /* Track max capacity */
        if (qr_ws.capL > max_qr_cap) max_qr_cap = qr_ws.capL;
        if (upd_ws.cap > max_upd_cap) max_upd_cap = upd_ws.cap;
        
        /* Verify capacity never shrinks */
        if (qr_ws.capL < L)
        {
            printf("    FAILED: qr_ws.capL=%zu < L=%d\n", qr_ws.capL, L);
            passed = 0;
        }
        
        if (upd_ws.cap < (size_t)L * L)
        {
            printf("    FAILED: upd_ws.cap=%zu < L*L=%d\n", upd_ws.cap, L*L);
            passed = 0;
        }
    }
    
    printf("  Max capacities: qr_capL=%zu, upd_cap=%zu\n", max_qr_cap, max_upd_cap);
    
    /* After shrinking requests, capacity should still be at max */
    if (qr_ws.capL < 64 || upd_ws.cap < 64*64)
    {
        printf("  WARNING: capacity shrunk unexpectedly\n");
    }
    
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    
    if (passed)
    {
        printf("  Growth pattern PASSED\n");
    }
    
    return passed;
}

//==============================================================================
// TEST: Memory alignment verification
//==============================================================================

/**
 * @brief Verify workspace buffers are properly aligned
 */
static int test_workspace_alignment(void)
{
    printf("\n=== Testing workspace memory alignment ===\n");
    
    int passed = 1;
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    ukf_pxy_ws_t pxy_ws = {0};
    
    ukf_qr_ws_ensure(&qr_ws, 32);
    ukf_upd_ws_ensure(&upd_ws, 32);
    ukf_pxy_ws_ensure(&pxy_ws, 32, 72);
    
    /* Check 32-byte alignment (for AVX2) */
    const uintptr_t ALIGN = 32;
    
    if ((uintptr_t)qr_ws.Aprime % ALIGN != 0)
    {
        printf("  FAILED: qr_ws.Aprime not 32-byte aligned\n");
        passed = 0;
    }
    if ((uintptr_t)qr_ws.R_ % ALIGN != 0)
    {
        printf("  FAILED: qr_ws.R_ not 32-byte aligned\n");
        passed = 0;
    }
    if ((uintptr_t)upd_ws.Z % ALIGN != 0)
    {
        printf("  FAILED: upd_ws.Z not 32-byte aligned\n");
        passed = 0;
    }
    if ((uintptr_t)upd_ws.U % ALIGN != 0)
    {
        printf("  FAILED: upd_ws.U not 32-byte aligned\n");
        passed = 0;
    }
    if ((uintptr_t)pxy_ws.Xc % ALIGN != 0)
    {
        printf("  FAILED: pxy_ws.Xc not 32-byte aligned\n");
        passed = 0;
    }
    if ((uintptr_t)pxy_ws.YTc % ALIGN != 0)
    {
        printf("  FAILED: pxy_ws.YTc not 32-byte aligned\n");
        passed = 0;
    }
    
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    ukf_pxy_ws_cleanup(&pxy_ws);
    
    if (passed)
    {
        printf("  Alignment PASSED (all buffers 32-byte aligned)\n");
    }
    
    return passed;
}

//==============================================================================
// TEST: Sub-workspace capacity tracking
//==============================================================================

/**
 * @brief Test that sub-workspaces track capacity correctly
 */
static int test_subworkspace_capacity(void)
{
    printf("\n=== Testing sub-workspace capacity tracking ===\n");
    
    int passed = 1;
    
    ukf_upd_ws_t ws = {0};
    
    /* Ensure for n=16 */
    ukf_upd_ws_ensure(&ws, 16);
    
    /* Check cholupdate workspace dimensions */
    if (ws.chol_ws->n_max < 16)
    {
        printf("  FAILED: chol_ws->n_max=%u < 16\n", ws.chol_ws->n_max);
        passed = 0;
    }
    
    /* Check TRSM workspace dimensions */
    if (ws.trsm_ws->n_max < 16)
    {
        printf("  FAILED: trsm_ws->n_max=%zu < 16\n", ws.trsm_ws->n_max);
        passed = 0;
    }
    
    /* Ensure for larger n=32 - sub-workspaces should grow */
    ukf_upd_ws_ensure(&ws, 32);
    
    if (ws.chol_ws->n_max < 32)
    {
        printf("  FAILED: chol_ws->n_max=%u < 32 after growth\n", ws.chol_ws->n_max);
        passed = 0;
    }
    
    if (ws.trsm_ws->n_max < 32)
    {
        printf("  FAILED: trsm_ws->n_max=%zu < 32 after growth\n", ws.trsm_ws->n_max);
        passed = 0;
    }
    
    ukf_upd_ws_cleanup(&ws);
    
    if (passed)
    {
        printf("  Sub-workspace capacity tracking PASSED\n");
    }
    
    return passed;
}

//==============================================================================
// TEST: Edge cases
//==============================================================================

/**
 * @brief Test edge case: L=1 (minimum state size)
 */
static int test_edge_case_L1(void)
{
    printf("\n=== Testing edge case L=1 ===\n");
    
    int passed = 1;
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    ukf_pxy_ws_t pxy_ws = {0};
    
    int rc1 = ukf_qr_ws_ensure(&qr_ws, 1);
    int rc2 = ukf_upd_ws_ensure(&upd_ws, 1);
    int rc3 = ukf_pxy_ws_ensure(&pxy_ws, 1, 8);
    
    if (rc1 != 0 || rc2 != 0 || rc3 != 0)
    {
        printf("  FAILED: ensure failed for L=1\n");
        passed = 0;
    }
    
    /* Verify we can touch the buffers */
    if (qr_ws.Aprime) qr_ws.Aprime[0] = 1.0f;
    if (upd_ws.Z) upd_ws.Z[0] = 1.0f;
    if (pxy_ws.Xc) pxy_ws.Xc[0] = 1.0f;
    
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    ukf_pxy_ws_cleanup(&pxy_ws);
    
    if (passed)
    {
        printf("  Edge case L=1 PASSED\n");
    }
    
    return passed;
}

/**
 * @brief Test edge case: Large L (stress test)
 */
static int test_edge_case_large_L(void)
{
    printf("\n=== Testing edge case large L=128 ===\n");
    
    int passed = 1;
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    
    /* L=128 means M=384, N=257 - substantial memory */
    printf("  Allocating workspaces for L=128...\n");
    
    int rc1 = ukf_qr_ws_ensure(&qr_ws, 128);
    int rc2 = ukf_upd_ws_ensure(&upd_ws, 128);
    
    if (rc1 != 0 || rc2 != 0)
    {
        printf("  FAILED: ensure failed for L=128\n");
        passed = 0;
    }
    else
    {
        /* Verify capacity */
        if (qr_ws.capL < 128 || upd_ws.cap < 128*128)
        {
            printf("  FAILED: insufficient capacity\n");
            passed = 0;
        }
        else
        {
            printf("  Large L=128 PASSED\n");
        }
    }
    
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    
    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_ukf_workspace_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("    SR-UKF WORKSPACE TESTS\n");
    printf("=================================================\n");
    
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    /* QR workspace tests */
    printf("\n--- QR Workspace Tests ---\n");
    
    results->total++;
    if (test_qr_ws_basic_lifecycle())
    {
        results->passed++;
        printf("✓ QR workspace lifecycle PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ QR workspace lifecycle FAILED\n");
    }
    
    results->total++;
    if (test_qr_ws_double_free())
    {
        results->passed++;
        printf("✓ QR workspace double-free PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ QR workspace double-free FAILED\n");
    }
    
    results->total++;
    if (test_qr_ws_null_safety())
    {
        results->passed++;
        printf("✓ QR workspace NULL safety PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ QR workspace NULL safety FAILED\n");
    }
    
    /* Update workspace tests */
    printf("\n--- Update Workspace Tests ---\n");
    
    results->total++;
    if (test_upd_ws_basic_lifecycle())
    {
        results->passed++;
        printf("✓ Update workspace lifecycle PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Update workspace lifecycle FAILED\n");
    }
    
    results->total++;
    if (test_upd_ws_double_free())
    {
        results->passed++;
        printf("✓ Update workspace double-free PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Update workspace double-free FAILED\n");
    }
    
    results->total++;
    if (test_upd_ws_null_safety())
    {
        results->passed++;
        printf("✓ Update workspace NULL safety PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Update workspace NULL safety FAILED\n");
    }
    
    /* Pxy workspace tests */
    printf("\n--- Cross-Covariance Workspace Tests ---\n");
    
    results->total++;
    if (test_pxy_ws_basic_lifecycle())
    {
        results->passed++;
        printf("✓ Pxy workspace lifecycle PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Pxy workspace lifecycle FAILED\n");
    }
    
    results->total++;
    if (test_pxy_ws_double_free())
    {
        results->passed++;
        printf("✓ Pxy workspace double-free PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Pxy workspace double-free FAILED\n");
    }
    
    results->total++;
    if (test_pxy_ws_null_safety())
    {
        results->passed++;
        printf("✓ Pxy workspace NULL safety PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Pxy workspace NULL safety FAILED\n");
    }
    
    /* Usage pattern tests */
    printf("\n--- Usage Pattern Tests ---\n");
    
    results->total++;
    if (test_workspace_reuse_pattern())
    {
        results->passed++;
        printf("✓ Workspace reuse pattern PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Workspace reuse pattern FAILED\n");
    }
    
    results->total++;
    if (test_workspace_growth_pattern())
    {
        results->passed++;
        printf("✓ Workspace growth pattern PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Workspace growth pattern FAILED\n");
    }
    
    /* Memory tests */
    printf("\n--- Memory Tests ---\n");
    
    results->total++;
    if (test_workspace_alignment())
    {
        results->passed++;
        printf("✓ Workspace alignment PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Workspace alignment FAILED\n");
    }
    
    results->total++;
    if (test_subworkspace_capacity())
    {
        results->passed++;
        printf("✓ Sub-workspace capacity PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Sub-workspace capacity FAILED\n");
    }
    
    /* Edge cases */
    printf("\n--- Edge Case Tests ---\n");
    
    results->total++;
    if (test_edge_case_L1())
    {
        results->passed++;
        printf("✓ Edge case L=1 PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Edge case L=1 FAILED\n");
    }
    
    results->total++;
    if (test_edge_case_large_L())
    {
        results->passed++;
        printf("✓ Edge case large L PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Edge case large L FAILED\n");
    }
    
    /* Summary */
    printf("\n=================================================\n");
    printf("SR-UKF Workspace Tests: %d/%d passed\n", results->passed, results->total);
    
    if (results->passed == results->total)
    {
        printf("✓ ALL SR-UKF WORKSPACE TESTS PASSED!\n");
    }
    else
    {
        printf("✗ %d SR-UKF workspace tests FAILED\n", results->failed);
    }
    printf("=================================================\n");
    
    return (results->failed == 0) ? 0 : 1;
}

//==============================================================================
// STANDALONE MODE
//==============================================================================

#ifdef STANDALONE
int main(void)
{
    test_results_t results = {0};
    return run_ukf_workspace_tests(&results);
}
#endif