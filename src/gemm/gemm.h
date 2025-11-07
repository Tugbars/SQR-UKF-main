#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// OPAQUE WORKSPACE TYPE
//==============================================================================

/**
 * @brief Opaque workspace for GEMM operations
 * 
 * Contains pre-allocated buffers for packing A and B matrices.
 * Thread-safe: Each thread should have its own workspace.
 */
typedef struct gemm_workspace gemm_workspace_t;

//==============================================================================
// WORKSPACE API
//==============================================================================

/**
 * @brief Query workspace size for GEMM
 */
size_t gemm_workspace_query(uint16_t M, uint16_t K, uint16_t N);

/**
 * @brief Create GEMM workspace (heap allocation)
 */
gemm_workspace_t *gemm_workspace_create(size_t size);

/**
 * @brief Initialize workspace from user buffer
 */
gemm_workspace_t *gemm_workspace_init(void *buffer, size_t size);

/**
 * @brief Destroy workspace
 */
void gemm_workspace_destroy(gemm_workspace_t *ws);

//==============================================================================
// GEMM API
//==============================================================================

/**
 * @brief GEMM with workspace: C := beta*C + alpha*A*B
 */
int gemm_ws(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    uint16_t M, uint16_t K, uint16_t N,
    float alpha,
    float beta,
    gemm_workspace_t *ws);

/**
 * @brief Original mul() - C = A * B (UNCHANGED SIGNATURE)
 */
int mul(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    uint16_t row_a, uint16_t column_a,
    uint16_t row_b, uint16_t column_b);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_H */