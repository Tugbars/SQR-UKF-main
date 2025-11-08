/**
 * @file test_qr_gemm_integrated.c
 * @brief Comprehensive QR+GEMM testing with planning
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "qr.h"
#include "../gemm/gemm.h"

//==============================================================================
// ENHANCED QR WORKSPACE WITH GEMM PLANNING
//==============================================================================

typedef struct {
    qr_workspace *base;
    
    // Pre-planned GEMM operations for block reflector
    gemm_plan_t *plan_ytc;    // Z = Y^T * C
    gemm_plan_t *plan_tz;     // Z = T * Z  
    gemm_plan_t *plan_yz;     // C -= Y * Z
    
    // Cached dimensions
    uint16_t m_max;
    uint16_t n_max;
    uint16_t ib;
} qr_workspace_planned;

/**
 * @brief Allocate QR workspace with GEMM planning
 */
qr_workspace_planned* qr_workspace_planned_alloc(uint16_t m_max, uint16_t n_max, uint16_t ib)
{
    qr_workspace_planned *ws = calloc(1, sizeof(qr_workspace_planned));
    if (!ws) return NULL;
    
    // Allocate base QR workspace
    ws->base = qr_workspace_alloc(m_max, n_max, ib);
    if (!ws->base) {
        free(ws);
        return NULL;
    }
    
    ws->m_max = m_max;
    ws->n_max = n_max;
    ws->ib = ib ? ib : 32;
    
    // Pre-plan all GEMM operations
    gemm_error_t error;
    
    // Y^T * C: [ib×m] * [m×n]
    ws->plan_ytc = gemm_plan_create_safe(ws->ib, m_max, n_max,
                                         NULL, NULL, NULL,
                                         1.0f, 0.0f, &error);
    
    // T * Z: [ib×ib] * [ib×n] - This will use tiny matrix kernels!
    ws->plan_tz = gemm_plan_create_safe(ws->ib, ws->ib, n_max,
                                        NULL, NULL, NULL,
                                        1.0f, 0.0f, &error);
    
    // Y * Z: [m×ib] * [ib×n]
    ws->plan_yz = gemm_plan_create_safe(m_max, ws->ib, n_max,
                                        NULL, NULL, NULL,
                                        -1.0f, 1.0f, &error);  // Note: alpha=-1, beta=1 for C -= Y*Z
    
    if (!ws->plan_ytc || !ws->plan_tz || !ws->plan_yz) {
        qr_workspace_planned_free(ws);
        return NULL;
    }
    
    return ws;
}

void qr_workspace_planned_free(qr_workspace_planned *ws)
{
    if (!ws) return;
    
    gemm_plan_destroy(ws->plan_ytc);
    gemm_plan_destroy(ws->plan_tz);
    gemm_plan_destroy(ws->plan_yz);
    qr_workspace_free(ws->base);
    free(ws);
}

//==============================================================================
// OPTIMIZED BLOCK REFLECTOR WITH PLANNED GEMM
//==============================================================================

static int apply_block_reflector_planned(
    float *C,
    const float *Y,
    const float *YT,
    const float *T,
    uint16_t m,
    uint16_t n,
    uint16_t ib,
    qr_workspace_planned *ws)
{
    float *Z = ws->base->Z;
    float *Z_temp = ws->base->Z_temp;
    
    // Step 1: Z = Y^T * C
    gemm_execute_plan(ws->plan_ytc, Z, YT, C, 1.0f, 0.0f);
    
    // Step 2: Z = T * Z (tiny matrix multiply - super fast!)
    memcpy(Z_temp, Z, (size_t)ib * n * sizeof(float));
    gemm_execute_plan(ws->plan_tz, Z, T, Z_temp, 1.0f, 0.0f);
    
    // Step 3: C = C - Y * Z (fused with beta=1, alpha=-1)
    gemm_execute_plan(ws->plan_yz, C, Y, Z, -1.0f, 1.0f);
    
    return 0;
}

//==============================================================================
// BLOCKED QR WITH PLANNED GEMM
//==============================================================================

int qr_blocked_planned(qr_workspace_planned *ws, const float *A, 
                       float *Q, float *R, uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R) return -EINVAL;
    
    qr_workspace *base = ws->base;
    
    // Copy A to workspace
    memcpy(base->Cpack, A, (size_t)m * n * sizeof(float));
    float *Awork = base->Cpack;
    
    const uint16_t kmax = (m < n) ? m : n;
    const uint16_t ib = ws->ib;
    
    // Main blocked loop
    for (uint16_t k = 0; k < kmax; k += ib) {
        const uint16_t block_size = (k + ib <= kmax) ? ib : (kmax - k);
        const uint16_t rows_below = m - k;
        const uint16_t cols_right = n - k - block_size;
        
        // 1. Panel factorization (dual-packs Y and YT)
        panel_qr(&Awork[k * n + k], base->Y, base->YT, &base->tau[k],
                rows_below, n, block_size);
        
        // 2. Build T matrix
        build_T_matrix(base->Y, &base->tau[k], base->T, rows_below, block_size);
        
        // 3. Apply block reflector with planned GEMM
        if (cols_right > 0) {
            int ret = apply_block_reflector_planned(
                &Awork[k * n + k + block_size],
                base->Y, base->YT, base->T,
                rows_below, cols_right, block_size,
                ws);
            if (ret != 0) return ret;
        }
    }
    
    // Extract R
    for (uint16_t i = 0; i < m; ++i) {
        for (uint16_t j = 0; j < n; ++j) {
            R[i * n + j] = (i <= j) ? Awork[i * n + j] : 0.0f;
        }
    }
    
    // Form Q if needed (could also use planned GEMM here)
    if (!only_R && Q) {
        // ... existing Q formation code ...
    }
    
    return 0;
}

//==============================================================================
// PERFORMANCE BENCHMARK
//==============================================================================

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void benchmark_qr(uint16_t m, uint16_t n, uint16_t ib, const char *desc)
{
    printf("  [BENCH] %s (%ux%u, ib=%u):\n", desc, m, n, ib);
    
    float *A = malloc((size_t)m * n * sizeof(float));
    float *Q = malloc((size_t)m * m * sizeof(float));
    float *R = malloc((size_t)m * n * sizeof(float));
    
    // Random matrix
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // Warmup
    qr_workspace *ws_basic = qr_workspace_alloc(m, n, ib);
    qr_ws_blocked(ws_basic, A, Q, R, m, n, true);
    
    // Benchmark basic blocked QR
    const int n_runs = 100;
    double t_start = get_time_sec();
    for (int i = 0; i < n_runs; i++) {
        qr_ws_blocked(ws_basic, A, Q, R, m, n, true);
    }
    double t_basic = (get_time_sec() - t_start) / n_runs;
    
    // Benchmark planned QR
    qr_workspace_planned *ws_planned = qr_workspace_planned_alloc(m, n, ib);
    
    t_start = get_time_sec();
    for (int i = 0; i < n_runs; i++) {
        qr_blocked_planned(ws_planned, A, Q, R, m, n, true);
    }
    double t_planned = (get_time_sec() - t_start) / n_runs;
    
    double gflops_basic = (2.0 * m * n * n - 2.0/3.0 * n * n * n) / (t_basic * 1e9);
    double gflops_planned = (2.0 * m * n * n - 2.0/3.0 * n * n * n) / (t_planned * 1e9);
    
    printf("    Basic:   %.3f ms (%.1f GFLOP/s)\n", t_basic * 1000, gflops_basic);
    printf("    Planned: %.3f ms (%.1f GFLOP/s) - %.1fx speedup\n", 
           t_planned * 1000, gflops_planned, t_basic / t_planned);
    
    qr_workspace_free(ws_basic);
    qr_workspace_planned_free(ws_planned);
    free(A); free(Q); free(R);
}

//==============================================================================
// CORRECTNESS TEST WITH DIFFERENT SIZE CLASSES
//==============================================================================

static int test_correctness_planned(uint16_t m, uint16_t n, uint16_t ib, const char *desc)
{
    printf("  [TEST] %s (%ux%u, ib=%u)... ", desc, m, n, ib);
    
    qr_workspace_planned *ws = qr_workspace_planned_alloc(m, n, ib);
    if (!ws) {
        printf("FAIL (workspace alloc)\n");
        return 1;
    }
    
    float *A = malloc((size_t)m * n * sizeof(float));
    float *Q = malloc((size_t)m * m * sizeof(float));
    float *R = malloc((size_t)m * n * sizeof(float));
    float *QR = malloc((size_t)m * n * sizeof(float));
    
    // Random test matrix
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // Factor with planned QR
    int ret = qr_blocked_planned(ws, A, Q, R, m, n, false);
    if (ret != 0) {
        printf("FAIL (factorization error %d)\n", ret);
        goto cleanup;
    }
    
    // Reconstruct Q*R using optimized GEMM
    gemm_auto(QR, Q, R, m, m, n, 1.0f, 0.0f);
    
    // Compute error
    float normA = frobenius_norm(A, m, n);
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        QR[i] = A[i] - QR[i];
    }
    float normErr = frobenius_norm(QR, m, n);
    float relErr = normErr / normA;
    
    const float tol = 1e-4f;
    if (relErr > tol) {
        printf("FAIL (rel_err=%.2e)\n", relErr);
        ret = 1;
    } else {
        printf("PASS (rel_err=%.2e)\n", relErr);
        ret = 0;
    }
    
cleanup:
    free(A); free(Q); free(R); free(QR);
    qr_workspace_planned_free(ws);
    return ret;
}

//==============================================================================
// MAIN TEST SUITE
//==============================================================================

int main(void)
{
    printf("QR + Planned GEMM Integration Test Suite\n");
    printf("=========================================\n\n");
    
    printf("1. CORRECTNESS TESTS\n");
    printf("--------------------\n");
    
    int failures = 0;
    
    // Test different size classes to exercise all GEMM paths
    // TINY matrices (register-only kernels)
    failures += test_correctness_planned(8, 8, 4, "Tiny 8x8");
    failures += test_correctness_planned(12, 12, 6, "Tiny 12x12");
    failures += test_correctness_planned(16, 16, 8, "Tiny 16x16");
    
    // SMALL matrices (direct kernels)
    failures += test_correctness_planned(32, 32, 16, "Small 32x32");
    failures += test_correctness_planned(64, 64, 32, "Small 64x64");
    
    // MEDIUM matrices (single-level blocking)
    failures += test_correctness_planned(128, 128, 32, "Medium 128x128");
    failures += test_correctness_planned(256, 64, 32, "Medium tall 256x64");
    
    // LARGE matrices (full blocking)
    failures += test_correctness_planned(512, 512, 64, "Large 512x512");
    failures += test_correctness_planned(1024, 256, 64, "Large tall 1024x256");
    
    printf("\nTotal failures: %d\n\n", failures);
    
    printf("2. PERFORMANCE BENCHMARKS\n");
    printf("-------------------------\n");
    
    // Benchmark different sizes
    benchmark_qr(64, 64, 32, "Small square");
    benchmark_qr(256, 256, 32, "Medium square");
    benchmark_qr(512, 256, 64, "Tall matrix");
    benchmark_qr(1024, 1024, 64, "Large square");
    
    printf("\n3. KALMAN FILTER SIZES (SPECIAL CASE)\n");
    printf("--------------------------------------\n");
    
    // These sizes are critical for Kalman filters
    benchmark_qr(4, 4, 4, "Kalman 4x4 (x,y,vx,vy)");
    benchmark_qr(6, 6, 6, "Kalman 6x6 (with acceleration)");
    benchmark_qr(12, 12, 6, "Kalman 12x12 (full 3D)");
    
    return failures;
}