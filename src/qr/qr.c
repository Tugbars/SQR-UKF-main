#include "qr.h"
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
    ws->ib    = ib ? ib : 64;

    size_t bytes = 0;
    ws->tau   = (float*)portable_aligned_alloc(32, mn * sizeof(float)); bytes += mn * sizeof(float);
    ws->tmp   = (float*)portable_aligned_alloc(32, m_max * sizeof(float)); bytes += m_max * sizeof(float);
    ws->work  = (float*)portable_aligned_alloc(32, m_max * sizeof(float)); bytes += m_max * sizeof(float);
    ws->T     = (float*)portable_aligned_alloc(32, ws->ib * ws->ib * sizeof(float)); bytes += ws->ib * ws->ib * sizeof(float);
    ws->Cpack = (float*)portable_aligned_alloc(32, m_max * n_max * sizeof(float)); bytes += (size_t)m_max * n_max * sizeof(float);
    ws->Y     = (float*)portable_aligned_alloc(32, ws->ib * n_max * sizeof(float)); bytes += ws->ib * n_max * sizeof(float);
    ws->Z     = (float*)portable_aligned_alloc(32, ws->ib * n_max * sizeof(float)); bytes += ws->ib * n_max * sizeof(float);
    ws->vn1   = (float*)portable_aligned_alloc(32, n_max * sizeof(float)); bytes += n_max * sizeof(float);
    ws->vn2   = (float*)portable_aligned_alloc(32, n_max * sizeof(float)); bytes += n_max * sizeof(float);

    if (!ws->tau || !ws->tmp || !ws->work || !ws->T || !ws->Cpack || !ws->Y || !ws->Z || !ws->vn1 || !ws->vn2) {
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
