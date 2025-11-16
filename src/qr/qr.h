#ifndef QR_H
#define QR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Forward declare gemm_plan_t (from GEMM library)
typedef struct gemm_plan gemm_plan_t;

/**
 * @brief Pre-created GEMM plans for QR operations
 */
typedef struct
{
    gemm_plan_t *plan_yt_c; // Y^T * C  [ib × m] × [m × n] = [ib × n]
    gemm_plan_t *plan_t_z;  // T * Z    [ib × ib] × [ib × n] = [ib × n]
    gemm_plan_t *plan_y_z;  // Y * Z    [m × ib] × [ib × n] = [m × n]
    uint16_t plan_m;        // Exact dimensions
    uint16_t plan_n;
    uint16_t plan_ib;
} qr_gemm_plans_t;

/**
 * @brief QR workspace with pre-allocated buffers and GEMM plans
 */
typedef struct
{
    uint16_t m_max;
    uint16_t n_max;
    uint16_t ib;

    float *tau;
    float *tmp;
    float *work;
    float *T;
    float *Cpack;
    float *Y;
    float *YT;
    float *Z;
    float *Z_temp;
    float *vn1;
    float *vn2;

    // Pre-created GEMM plans
    qr_gemm_plans_t *trailing_plans;
    qr_gemm_plans_t *q_formation_plans;

    // Stored reflectors [num_blocks][m_max][ib] and [num_blocks][ib][ib]
    float *Y_stored;
    float *T_stored;
    uint16_t num_blocks;
    size_t Y_block_stride;
    size_t T_block_stride;

    size_t total_bytes;
} qr_workspace;

/**
 * @brief Create workspace with default settings (stores reflectors)
 */
qr_workspace *qr_workspace_alloc(uint16_t m_max, uint16_t n_max, uint16_t ib);

/**
 * @brief Create workspace with reflector storage control
 */
qr_workspace *qr_workspace_alloc_ex(uint16_t m_max, uint16_t n_max,
                                    uint16_t ib, bool store_reflectors);

/**
 * @brief Free workspace
 */
void qr_workspace_free(qr_workspace *ws);

/**
 * @brief Get workspace memory footprint
 */
size_t qr_workspace_bytes(const qr_workspace *ws);

/**
 * @brief Workspace-based QR (uses pre-allocated workspace)
 */
int qr_ws_blocked(qr_workspace *ws, const float *A, float *Q, float *R,
                  uint16_t m, uint16_t n, bool only_R);

/**
 * @brief In-place blocked QR (requires 32-byte aligned A)
 */
int qr_ws_blocked_inplace(qr_workspace *ws, float *A, float *Q, float *R,
                          uint16_t m, uint16_t n, bool only_R);

/**
 * @brief Simple blocked QR (auto-allocates workspace)
 */
int qr_blocked(const float *A, float *Q, float *R,
               uint16_t m, uint16_t n, bool only_R);

#endif // QR_H