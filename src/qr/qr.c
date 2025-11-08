#include "qr.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include "../gemm/gemm.h"  // Add this

#if defined(_WIN32)
    #include <malloc.h>
    static inline void* portable_aligned_alloc(size_t alignment, size_t size)
    {
        return _aligned_malloc(size, alignment);
    }
    static inline void portable_aligned_free(void* ptr)
    {
        _aligned_free(ptr);
    }
#else
    static inline void* portable_aligned_alloc(size_t alignment, size_t size)
    {
        void* ptr = NULL;
        if (posix_memalign(&ptr, alignment, size) != 0)
            ptr = NULL;
        return ptr;
    }
    static inline void portable_aligned_free(void* ptr)
    {
        free(ptr);
    }
#endif


//==============================================================================
// SIMPLE SCALAR QR IMPLEMENTATION (REFERENCE VERSION)
//==============================================================================

qr_workspace* qr_workspace_alloc(uint16_t m_max, uint16_t n_max, uint16_t ib)
{
    if (!m_max || !n_max)
        return NULL;

    qr_workspace *ws = (qr_workspace*)calloc(1, sizeof(qr_workspace));
    if (!ws)
        return NULL;

    const uint16_t mn = (m_max < n_max) ? m_max : n_max;
    ws->m_max = m_max;
    ws->n_max = n_max;
    ws->ib    = ib ? ib : 32;  // Default block size

    size_t bytes = 0;
    
    // Original buffers
    ws->tau   = (float*)malloc(mn * sizeof(float)); 
    bytes += mn * sizeof(float);
    
    ws->tmp   = (float*)malloc(m_max * sizeof(float)); 
    bytes += m_max * sizeof(float);
    
    ws->work  = (float*)malloc(m_max * sizeof(float)); 
    bytes += m_max * sizeof(float);
    
    // Blocked QR buffers
    ws->T     = (float*)malloc(ws->ib * ws->ib * sizeof(float)); 
    bytes += ws->ib * ws->ib * sizeof(float);
    
    ws->Cpack = (float*)malloc((size_t)m_max * n_max * sizeof(float)); 
    bytes += (size_t)m_max * n_max * sizeof(float);
    
    ws->Y     = (float*)malloc((size_t)m_max * ws->ib * sizeof(float)); 
    bytes += (size_t)m_max * ws->ib * sizeof(float);
    
    ws->Z     = (float*)malloc((size_t)ws->ib * n_max * sizeof(float)); 
    bytes += (size_t)ws->ib * n_max * sizeof(float);
    
    ws->vn1   = (float*)malloc(n_max * sizeof(float)); 
    bytes += n_max * sizeof(float);
    
    ws->vn2   = (float*)malloc(n_max * sizeof(float)); 
    bytes += n_max * sizeof(float);

    if (!ws->tau || !ws->tmp || !ws->work || !ws->T || !ws->Cpack || 
        !ws->Y || !ws->Z || !ws->vn1 || !ws->vn2) {
        qr_workspace_free(ws);
        return NULL;
    }

    ws->total_bytes = bytes;
    return ws;
}

void qr_workspace_free(qr_workspace *ws)
{
    if (!ws) return;
    free(ws->tau);
    free(ws->tmp);
    free(ws->work);
    free(ws->T);
    free(ws->Cpack);
    free(ws->Y);
    free(ws->Z);
    free(ws->vn1);
    free(ws->vn2);
    free(ws);
}

size_t qr_workspace_bytes(const qr_workspace *ws)
{
    return ws ? ws->total_bytes : 0;
}

//==============================================================================
// SIMPLE HOUSEHOLDER QR (SCALAR)
//==============================================================================

static void householder_reflection(float *x, uint16_t m, float *tau)
{
    // Compute v and tau for Householder vector
    double norm = 0.0;
    for (uint16_t i = 1; i < m; ++i)
        norm += (double)x[i] * (double)x[i];
    norm = sqrt(norm);

    if (norm == 0.0) {
        *tau = 0.0f;
        return;
    }

    double alpha = x[0];
    double beta = -copysign(sqrt(alpha * alpha + norm * norm), alpha);
    double inv = 1.0 / (alpha - beta);
    x[0] = 1.0f;
    for (uint16_t i = 1; i < m; ++i)
        x[i] = (float)(x[i] * inv);
    *tau = (float)((beta - alpha) / beta);
    x[0] = 1.0f; // set v0 = 1
}

static void apply_reflector(float *A, uint16_t m, uint16_t n, const float *v, float tau)
{
    if (tau == 0.0f) return;
    for (uint16_t j = 0; j < n; ++j) {
        double dot = 0.0;
        for (uint16_t i = 0; i < m; ++i)
            dot += v[i] * A[i * n + j];
        for (uint16_t i = 0; i < m; ++i)
            A[i * n + j] -= tau * v[i] * dot;
    }
}

//==============================================================================
// PANEL FACTORIZATION
//==============================================================================

static void panel_qr(float *panel, float *Y, float *tau, 
                     uint16_t m, uint16_t panel_stride, uint16_t ib)
{
    // Factor ib columns of the panel
    // panel[m×panel_stride] - row-major input
    // Y[m×ib] - row-major output (Householder vectors)
    // tau[ib] - output tau values
    
    for (uint16_t k = 0; k < ib; ++k) {
        const uint16_t rows_below = m - k;
        
        // Point to column k
        float *col = &panel[k * panel_stride + k];
        
        // Generate Householder vector (modifies col in-place)
        householder_reflection(col, rows_below, &tau[k]);
        
        // Apply to remaining columns in panel
        if (k + 1 < ib) {
            apply_reflector(&panel[k * panel_stride + (k + 1)], 
                          rows_below, ib - k - 1, col, tau[k]);
        }
        
        // Store Householder vector to Y (row-major: Y[i, k] = Y[i*ib + k])
        for (uint16_t i = k; i < m; ++i) {
            Y[i * ib + k] = (i == k) ? 1.0f : col[i - k];
        }
        
        // Zero out upper part
        for (uint16_t i = 0; i < k; ++i) {
            Y[i * ib + k] = 0.0f;
        }
    }
}

//==============================================================================
// BUILD T MATRIX (WY REPRESENTATION)
//==============================================================================

static void build_T_matrix(const float *Y, const float *tau, float *T,
                          uint16_t m, uint16_t ib)
{
    // Build upper triangular T matrix: Q = I - Y*T*Y^T
    // Y[m×ib] row-major, T[ib×ib] row-major
    
    memset(T, 0, (size_t)ib * ib * sizeof(float));
    
    // First reflector: T[0,0] = tau[0]
    T[0] = tau[0];
    
    // Build T column by column
    for (uint16_t k = 1; k < ib; ++k) {
        // T[k,k] = tau[k]
        T[k * ib + k] = tau[k];
        
        // Compute w = Y[:, 0:k]^T * Y[:, k]
        for (uint16_t j = 0; j < k; ++j) {
            double dot = 0.0;
            for (uint16_t i = 0; i < m; ++i) {
                dot += Y[i * ib + j] * Y[i * ib + k];
            }
            T[j * ib + k] = (float)dot;
        }
        
        // T[0:k, k] = -tau[k] * T[0:k, 0:k] * w
        for (int j = k - 1; j >= 0; --j) {
            double sum = 0.0;
            for (uint16_t i = j; i < k; ++i) {
                sum += T[j * ib + i] * T[i * ib + k];
            }
            T[j * ib + k] = (float)(-tau[k] * sum);
        }
    }
}

int qr_ws_scalar(qr_workspace *ws, const float *A, float *Q, float *R,
                 uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R) return -EINVAL;

    float *Awork = (float*)malloc((size_t)m * n * sizeof(float));
    memcpy(Awork, A, (size_t)m * n * sizeof(float));

    const uint16_t kmax = (m < n) ? m : n;
    float *tau = ws->tau;

    // Classical Householder QR
    for (uint16_t k = 0; k < kmax; ++k) {
        float *col = &Awork[(size_t)k * n + k];
        householder_reflection(col, m - k, &tau[k]);
        // Apply to the remaining submatrix
        if (k < n - 1)
            apply_reflector(&Awork[k * n + (k + 1)], m - k, n - k - 1, col, tau[k]);
    }

    // Extract R
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? Awork[i * n + j] : 0.0f;

    if (!only_R && Q) {
        // Form Q = I
        memset(Q, 0, (size_t)m * m * sizeof(float));
        for (uint16_t i = 0; i < m; ++i)
            Q[i * m + i] = 1.0f;

        // Apply each reflector to Q from the left
        for (int k = (int)kmax - 1; k >= 0; --k) {
            float *v = &Awork[(size_t)k * n + k];
            float tau_k = tau[k];
            if (tau_k == 0.0f) continue;
            for (uint16_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (uint16_t i = k; i < m; ++i)
                    dot += v[i - k] * Q[i * m + j];
                for (uint16_t i = k; i < m; ++i)
                    Q[i * m + j] -= tau_k * v[i - k] * dot;
            }
        }
    }

    free(Awork);
    return 0;
}

int qr_scalar_only(const float *A, float *Q, float *R,
                   uint16_t m, uint16_t n, bool only_R)
{
    qr_workspace *ws = qr_workspace_alloc(m, n, 0);
    if (!ws) return -ENOMEM;
    int ret = qr_ws_scalar(ws, A, Q, R, m, n, only_R);
    qr_workspace_free(ws);
    return ret;
}


//==============================================================================
// BLOCKED QR (NO GEMM YET - JUST PANEL)
//==============================================================================

int qr_ws_blocked(qr_workspace *ws, const float *A, float *Q, float *R,
                  uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R) return -EINVAL;
    
    // Copy A to workspace
    memcpy(ws->Cpack, A, (size_t)m * n * sizeof(float));
    float *Awork = ws->Cpack;
    
    const uint16_t kmax = (m < n) ? m : n;
    
    // For now: just do first panel to test
    if (kmax > 0) {
        uint16_t block_size = (ws->ib < kmax) ? ws->ib : kmax;
        
        // Factor first panel
        panel_qr(Awork, ws->Y, ws->tau, m, n, block_size);
        
        // Build T matrix
        build_T_matrix(ws->Y, ws->tau, ws->T, m, block_size);
        
        printf("[DEBUG] First panel factored, block_size=%d\n", block_size);
    }
    
    // Extract R
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? Awork[i * n + j] : 0.0f;
    
    // Q formation - use scalar for now
    if (!only_R && Q) {
        // Initialize Q = I
        memset(Q, 0, (size_t)m * m * sizeof(float));
        for (uint16_t i = 0; i < m; ++i)
            Q[i * m + i] = 1.0f;
        
        // Apply reflectors from Y
        for (int k = (int)kmax - 1; k >= 0; --k) {
            if (ws->tau[k] == 0.0f) continue;
            
            for (uint16_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (uint16_t i = k; i < m; ++i) {
                    float v_i = (i == k) ? 1.0f : Awork[i * n + k];
                    dot += v_i * Q[i * m + j];
                }
                for (uint16_t i = k; i < m; ++i) {
                    float v_i = (i == k) ? 1.0f : Awork[i * n + k];
                    Q[i * m + j] -= (float)(ws->tau[k] * dot) * v_i;
                }
            }
        }
    }
    
    return 0;
}