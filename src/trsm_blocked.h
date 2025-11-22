#ifndef TRSM_BLOCKED_H
#define TRSM_BLOCKED_H

#include "gemm_planning.h"

/* Forward declaration */
typedef struct trsm_workspace_s trsm_workspace;

//==============================================================================
// WORKSPACE MANAGEMENT
//==============================================================================

/**
 * @brief Allocate TRSM workspace for packing buffers
 * @param n_max Maximum matrix dimension
 * @return Allocated workspace, or NULL on failure
 */
trsm_workspace *trsm_workspace_alloc(size_t n_max);

/**
 * @brief Free TRSM workspace
 */
void trsm_workspace_free(trsm_workspace *ws);

//==============================================================================
// TIER 1: CONVENIENCE API (auto-allocates workspace)
//==============================================================================

/**
 * @brief Solve L·X = B (lower triangular, convenience API)
 */
int trsm_blocked_lower(
    const float *restrict L,
    float *restrict B,
    size_t n, size_t ncols,
    size_t ldl, size_t ldb,
    gemm_plan_t *gemm_plan);

/**
 * @brief Solve U·X = B (upper triangular, convenience API)
 */
int trsm_blocked_upper(
    const float *restrict U,
    float *restrict B,
    size_t n, size_t ncols,
    size_t ldu, size_t ldb,
    gemm_plan_t *gemm_plan);

/**
 * @brief Solve U^T·X = B (upper transpose, convenience API)
 */
int trsm_blocked_upper_transpose_auto(
    const float *restrict U,
    float *restrict B,
    size_t n, size_t ncols,
    size_t ldu, size_t ldb,
    gemm_plan_t *gemm_plan);

//==============================================================================
// TIER 2: OPTIMIZED API (reuses workspace for performance)
//==============================================================================

/**
 * @brief Solve L·X = B (lower triangular, optimized with workspace reuse)
 */
int trsm_blocked_lower_optimized(
    const float *restrict L,
    float *restrict B,
    size_t n, size_t ncols,
    size_t ldl, size_t ldb,
    gemm_plan_t *gemm_plan,
    trsm_workspace *ws);

/**
 * @brief Solve U·X = B (upper triangular, optimized with workspace reuse)
 */
int trsm_blocked_upper_optimized(
    const float *restrict U,
    float *restrict B,
    size_t n, size_t ncols,
    size_t ldu, size_t ldb,
    gemm_plan_t *gemm_plan,
    trsm_workspace *ws);

/**
 * @brief Solve U^T·X = B (upper transpose, optimized with workspace reuse)
 */
int trsm_blocked_upper_transpose(
    const float *restrict U,
    float *restrict B,
    size_t n, size_t ncols,
    size_t ldu, size_t ldb,
    gemm_plan_t *gemm_plan,
    trsm_workspace *ws);

#endif /* TRSM_BLOCKED_H */