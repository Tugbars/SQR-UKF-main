#ifndef QR_H
#define QR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct qr_gemm_plans qr_gemm_plans_t;

typedef struct {
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
 * @brief Create workspace with reflector storage control
 */
qr_workspace* qr_workspace_alloc_ex(uint16_t m_max, uint16_t n_max, 
                                     uint16_t ib, bool store_reflectors);

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