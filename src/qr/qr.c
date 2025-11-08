#include "qr.h"
#include "../gemm/gemm.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>


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
// WORKSPACE ALLOCATION
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
    ws->ib    = ib ? ib : 32;  // Default block size 32

    size_t bytes = 0;
    ws->tau   = (float*)portable_aligned_alloc(32, mn * sizeof(float)); 
    bytes += mn * sizeof(float);
    
    ws->tmp   = (float*)portable_aligned_alloc(32, m_max * sizeof(float)); 
    bytes += m_max * sizeof(float);
    
    ws->work  = (float*)portable_aligned_alloc(32, m_max * sizeof(float)); 
    bytes += m_max * sizeof(float);

    ws->work_YT = (float *)portable_aligned_alloc(32, (size_t)ws->ib * m_max * sizeof(float));
    bytes += (size_t)ws->ib * m_max * sizeof(float);

    ws->T     = (float*)portable_aligned_alloc(32, ws->ib * ws->ib * sizeof(float)); 
    bytes += ws->ib * ws->ib * sizeof(float);
    
    ws->Cpack = (float*)portable_aligned_alloc(32, (size_t)m_max * n_max * sizeof(float)); 
    bytes += (size_t)m_max * n_max * sizeof(float);
    
    ws->Y     = (float*)portable_aligned_alloc(32, (size_t)m_max * ws->ib * sizeof(float)); 
    bytes += (size_t)m_max * ws->ib * sizeof(float);
    
    ws->Z     = (float*)portable_aligned_alloc(32, (size_t)ws->ib * n_max * sizeof(float)); 
    bytes += (size_t)ws->ib * n_max * sizeof(float);
    
    ws->vn1   = (float*)portable_aligned_alloc(32, n_max * sizeof(float)); 
    bytes += n_max * sizeof(float);
    
    ws->vn2   = (float*)portable_aligned_alloc(32, n_max * sizeof(float)); 
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
    portable_aligned_free(ws->tau);
    portable_aligned_free(ws->tmp);
    portable_aligned_free(ws->work);
    portable_aligned_free(ws->work_YT);
    portable_aligned_free(ws->T);
    portable_aligned_free(ws->Cpack);
    portable_aligned_free(ws->Y);
    portable_aligned_free(ws->Z);
    portable_aligned_free(ws->vn1);
    portable_aligned_free(ws->vn2);
    free(ws);
}

size_t qr_workspace_bytes(const qr_workspace *ws)
{
    return ws ? ws->total_bytes : 0;
}

//==============================================================================
// HOUSEHOLDER UTILITIES (SCALAR - used in panel)
//==============================================================================

static void householder_reflection(float *x, uint16_t m, float *tau)
{
    // Compute Householder vector and tau
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
    
    for (uint16_t i = 1; i < m; ++i)
        x[i] = (float)(x[i] * inv);
    
    *tau = (float)((beta - alpha) / beta);
    x[0] = (float)beta;  // Store beta for R matrix
}

static void apply_reflector_to_panel(float *A, uint16_t m, uint16_t n, 
                                     const float *v, float tau)
{
    // Apply H = (I - tau*v*v^T) to matrix A (in-place)
    // A is m×n stored row-major
    if (tau == 0.0f) return;
    
    for (uint16_t j = 0; j < n; ++j) {
        double dot = 0.0;
        for (uint16_t i = 0; i < m; ++i)
            dot += v[i] * A[i * n + j];
        
        double scale = tau * dot;
        for (uint16_t i = 0; i < m; ++i)
            A[i * n + j] -= scale * v[i];
    }
}

//==============================================================================
// PANEL FACTORIZATION (Level-2, but small)
//==============================================================================

static void panel_qr(float *panel, float *Y, float *tau, 
                     uint16_t m, uint16_t panel_stride, uint16_t ib)
{
    // Factor 'ib' columns of the panel
    // panel[m×panel_stride] stores the panel in row-major
    // Y[m×ib] will store Householder vectors column-major
    // tau[ib] stores the tau values
    
    for (uint16_t k = 0; k < ib; ++k) {
        const uint16_t rows_below = m - k;
        
        // Extract column k into work buffer (for v vector)
        float *v = Y + k * m + k;  // Start at Y[k, k] in column-major
        for (uint16_t i = 0; i < rows_below; ++i)
            v[i] = panel[(k + i) * panel_stride + k];
        
        // Generate Householder vector
        householder_reflection(v, rows_below, &tau[k]);
        
        // Store diagonal element (beta) back to panel
        panel[k * panel_stride + k] = v[0];
        
        // Set v[0] = 1 for the reflection (implicitly)
        float saved = v[0];
        v[0] = 1.0f;
        
        // Apply to remaining columns in panel
        if (k + 1 < ib) {
            apply_reflector_to_panel(&panel[k * panel_stride + (k + 1)],
                                    rows_below, ib - k - 1, v, tau[k]);
        }
        
        // Restore for storage
        v[0] = saved;
    }
}

//==============================================================================
// BUILD T MATRIX (WY representation)
//==============================================================================

static void build_T_matrix(const float *Y, const float *tau, float *T,
                          uint16_t m, uint16_t ib)
{
    // Build upper triangular T matrix for WY representation
    // Q = I - Y * T * Y^T
    // Y[m×ib] is column-major, T[ib×ib] is row-major
    
    memset(T, 0, (size_t)ib * ib * sizeof(float));
    
    // T[0,0] = tau[0]
    T[0] = tau[0];
    
    for (uint16_t k = 1; k < ib; ++k) {
        // T[k,k] = tau[k]
        T[k * ib + k] = tau[k];
        
        // Compute w = Y[:, 0:k]^T * Y[:, k]
        // w[k] will store the result
        for (uint16_t j = 0; j < k; ++j) {
            double dot = 0.0;
            const float *y_j = Y + j * m;
            const float *y_k = Y + k * m;
            
            // Start from diagonal (v[i] = 1 implicitly for i < diag)
            for (uint16_t i = j; i < m; ++i) {
                float v_j = (i == j) ? 1.0f : y_j[i];
                float v_k = (i == k) ? 1.0f : y_k[i];
                dot += v_j * v_k;
            }
            T[j * ib + k] = dot;
        }
        
        // T[0:k, k] = -tau[k] * T[0:k, 0:k] * w
        // Use TRMV (triangular matrix-vector) but we'll do it manually
        for (int j = k - 1; j >= 0; --j) {
            double sum = 0.0;
            for (uint16_t i = j; i < k; ++i)
                sum += T[j * ib + i] * T[i * ib + k];
            T[j * ib + k] = -tau[k] * sum;
        }
    }
}

//==============================================================================
// BLOCK REFLECTOR APPLICATION (Level-3 GEMM!)
//==============================================================================

static int apply_block_reflector(float *C, const float *Y, const float *T,
                                 uint16_t m, uint16_t n, uint16_t ib,
                                 float *Z, float *work_YT, 
                                 gemm_workspace_t *ws_gemm)  // ← Renamed
{
    // Apply (I - Y*T*Y^T) to C
    // C[m×n], Y[m×ib] col-major, T[ib×ib] row-major, Z[ib×n] row-major
    // work_YT[ib×m] is a workspace buffer
    
    int ret;
    
    // Step 1: Prepare Y^T in row-major format [ib×m]
    for (uint16_t j = 0; j < ib; ++j) {
        for (uint16_t i = 0; i < m; ++i) {
            float val;
            if (i < j) {
                val = 0.0f;  // Upper part is zero
            } else if (i == j) {
                val = 1.0f;  // Diagonal is implicitly 1
            } else {
                val = Y[j * m + i];  // Below diagonal from Y (col-major)
            }
            work_YT[j * m + i] = val;
        }
    }
    
    // Step 2: Z = Y^T * C  [ib×n] = [ib×m] * [m×n]
    ret = gemm_ws(Z, work_YT, C, ib, m, n, 1.0f, 0.0f, ws_gemm);
    if (ret != 0) return ret;
    
    // Step 3: Z = T * Z  [ib×n] = [ib×ib] * [ib×n]
    // Save Z temporarily in work_YT (reuse first ib*n floats)
    float *Z_save = work_YT;  // Reuse work_YT buffer
    memcpy(Z_save, Z, (size_t)ib * n * sizeof(float));
    ret = gemm_ws(Z, T, Z_save, ib, ib, n, 1.0f, 0.0f, ws_gemm);
    if (ret != 0) return ret;
    
    // Step 4: Prepare Y in row-major format [m×ib] in work_YT
    for (uint16_t i = 0; i < m; ++i) {
        for (uint16_t j = 0; j < ib; ++j) {
            float val;
            if (i < j) {
                val = 0.0f;
            } else if (i == j) {
                val = 1.0f;
            } else {
                val = Y[j * m + i];
            }
            work_YT[i * ib + j] = val;
        }
    }
    
    // Step 5: C = C - Y * Z  [m×n] -= [m×ib] * [ib×n]
    ret = gemm_ws(C, work_YT, Z, m, ib, n, -1.0f, 1.0f, ws_gemm);
    
    return ret;
}

//==============================================================================
// BLOCKED QR FACTORIZATION
//==============================================================================

int qr_ws_blocked(qr_workspace *ws, const float *A, float *Q, float *R,
                  uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R) return -EINVAL;
    
    // Allocate GEMM workspace
    size_t gemm_ws_size = gemm_workspace_query(m, n, n);
    gemm_workspace_t *gemm_ws = gemm_workspace_create(gemm_ws_size);
    if (!gemm_ws) return -ENOMEM;
    
    // Copy A to working matrix
    memcpy(ws->Cpack, A, (size_t)m * n * sizeof(float));
    float *Awork = ws->Cpack;
    
    const uint16_t kmax = (m < n) ? m : n;
    
    // Main blocked loop
    for (uint16_t k = 0; k < kmax; k += ws->ib) {
        const uint16_t block_size = (k + ws->ib <= kmax) ? ws->ib : (kmax - k);
        const uint16_t rows_below = m - k;
        const uint16_t cols_right = n - k - block_size;
        
        // 1. Panel factorization
        panel_qr(&Awork[k * n + k], ws->Y, &ws->tau[k], 
                rows_below, n, block_size);
        
        // 2. Build T matrix
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size);

        // 3. Apply block reflector to trailing matrix
        if (cols_right > 0)
        {
            int ret = apply_block_reflector(&Awork[k * n + k + block_size],
                                            ws->Y, ws->T,
                                            rows_below, cols_right, block_size,
                                            ws->Z, ws->work_YT, gemm_ws); // ← This one keeps the name 'gemm_ws'
            if (ret != 0)
            {
                gemm_workspace_destroy(gemm_ws);
                return ret;
            }
        }
    }

    // Extract R (upper triangular)
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? Awork[i * n + j] : 0.0f;
    
    // Form Q if requested (TODO: implement blocked Q formation)
    if (!only_R && Q) {
        // For now, fall back to scalar version
        memset(Q, 0, (size_t)m * m * sizeof(float));
        for (uint16_t i = 0; i < m; ++i)
            Q[i * m + i] = 1.0f;
        
        // Apply reflectors in reverse order
        // (This part can be optimized later with blocked updates)
        for (int k = (int)kmax - 1; k >= 0; --k) {
            float *v = ws->work;
            
            // Extract Householder vector from Awork
            for (uint16_t i = k; i < m; ++i)
                v[i - k] = (i == k) ? 1.0f : Awork[i * n + k];
            
            // Apply to Q
            for (uint16_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (uint16_t i = k; i < m; ++i)
                    dot += v[i - k] * Q[i * m + j];
                
                double scale = ws->tau[k] * dot;
                for (uint16_t i = k; i < m; ++i)
                    Q[i * m + j] -= scale * v[i - k];
            }
        }
    }
    
    gemm_workspace_destroy(gemm_ws);
    return 0;
}

//==============================================================================
// BACKWARD COMPATIBILITY - SCALAR VERSION
//==============================================================================

int qr_ws_scalar(qr_workspace *ws, const float *A, float *Q, float *R,
                 uint16_t m, uint16_t n, bool only_R)
{
    // Keep old scalar implementation for testing/comparison
    if (!ws || !A || !R) return -EINVAL;

    float *Awork = (float*)malloc((size_t)m * n * sizeof(float));
    if (!Awork) return -ENOMEM;
    
    memcpy(Awork, A, (size_t)m * n * sizeof(float));

    const uint16_t kmax = (m < n) ? m : n;
    float *tau = ws->tau;

    // Classical Householder QR
    for (uint16_t k = 0; k < kmax; ++k) {
        float *v = &Awork[k * n + k];
        const uint16_t rows_below = m - k;
        
        householder_reflection(v, rows_below, &tau[k]);
        
        if (k < n - 1) {
            // Save diagonal
            float diag = v[0];
            v[0] = 1.0f;
            
            apply_reflector_to_panel(&Awork[k * n + (k + 1)], 
                                    rows_below, n - k - 1, v, tau[k]);
            
            v[0] = diag;
        }
    }

    // Extract R
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? Awork[i * n + j] : 0.0f;

    // Form Q if needed
    if (!only_R && Q) {
        memset(Q, 0, (size_t)m * m * sizeof(float));
        for (uint16_t i = 0; i < m; ++i)
            Q[i * m + i] = 1.0f;

        for (int k = (int)kmax - 1; k >= 0; --k) {
            float *v = ws->work;
            for (uint16_t i = k; i < m; ++i)
                v[i - k] = (i == k) ? 1.0f : Awork[i * n + k];

            for (uint16_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (uint16_t i = k; i < m; ++i)
                    dot += v[i - k] * Q[i * m + j];
                
                double scale = tau[k] * dot;
                for (uint16_t i = k; i < m; ++i)
                    Q[i * m + j] -= scale * v[i - k];
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
// DEFAULT ENTRY POINT (uses blocked algorithm)
//==============================================================================

int qr_decompose(const float *A, float *Q, float *R,
                uint16_t m, uint16_t n, bool only_R)
{
    qr_workspace *ws = qr_workspace_alloc(m, n, 32);
    if (!ws) return -ENOMEM;
    int ret = qr_ws_blocked(ws, A, Q, R, m, n, only_R);
    qr_workspace_free(ws);
    return ret;
}