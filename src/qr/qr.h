/**
 * @file qr.c
 * @brief Blocked compact-WY QR (single-precision) with AVX2/FMA kernels.
 *
 * @details
 * This implementation factors an m×n row-major matrix A into Q·R using
 * Householder reflections. It follows the LAPACK/BLAS pattern:
 *  - **Panel factorization (unblocked)**: GEQR2 over a panel of width ib.
 *  - **Form T (compact-WY)**: LARFT builds the ib×ib triangular T for the panel V.
 *  - **Blocked application to trailing matrix**: LARFB-style update via three BLAS-3 shaped
 *    steps: Y = Vᵀ·C, Z = T·Y, C ← C − V·Z. These are implemented with small packers and
 *    AVX2/FMA vectorized kernels (dual accumulators, contiguous loads across kc).
 *
 * **Workspace Design (Zero Hot-Path Allocations)**
 *  - All buffers pre-allocated in qr_workspace structure
 *  - Hot-path functions (_ws variants) accept workspace, no malloc/free
 *  - Legacy qr() function wraps workspace for backward compatibility
 *  - Thread-safe: each thread uses independent workspace
 *
 * **Data layout and outputs**
 *  - Input is row-major A (m×n). The routine copies A→R and factors **in-place**.
 *  - On return, the **upper triangle of R** is the R factor. The **strict lower triangle**
 *    stores the Householder reflectors V; the corresponding scalars τ are kept internally.
 *  - Q is **not** formed unless requested. When needed, ORGQR builds Q (m×m) using the same
 *    blocked machinery (no per-reflector rank-1 updates).
 *
 * **API**
 *  - Legacy: qr() - simple, allocates internally (UNCHANGED)
 *  - Workspace: qr_workspace_alloc() + qr_ws() - zero hot-path allocations
 *  - CPQR: geqp3() legacy, geqp3_ws() workspace variant
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <stdbool.h>
#include <immintrin.h>
#include "inv.h"

#include "linalg_simd.h" // RESTRICT, linalg_has_avx2(), LINALG_* knobs

#if LINALG_SIMD_ENABLE
#include "qr_avx2_kernel.h" // Highly optimized AVX2 kernels - UNCHANGED
#endif

//==============================================================================
// CONFIGURATION (UNCHANGED)
//==============================================================================

#ifndef LINALG_SMALL_N_THRESH
#define LINALG_SMALL_N_THRESH 48
#endif

#ifndef LINALG_BLOCK_KC
#define LINALG_BLOCK_KC 256
#endif

#ifndef LINALG_BLOCK_JC
#define LINALG_BLOCK_JC 64
#endif

#ifndef LINALG_BLOCK_MC
#define LINALG_BLOCK_MC 192
#endif

#ifndef LINALG_BLOCK_NC
#define LINALG_BLOCK_NC 4096
#endif

#ifndef QRW_IB_DEFAULT
#define QRW_IB_DEFAULT 64
#endif

#ifndef LINALG_DEFAULT_ALIGNMENT
#define LINALG_DEFAULT_ALIGNMENT 32
#endif

#ifndef CPQR_SMALL_N_THRESH
#define CPQR_SMALL_N_THRESH 48
#endif

_Static_assert(LINALG_DEFAULT_ALIGNMENT >= 32, "Need 32B alignment for AVX loads");

//==============================================================================
// PLATFORM-SPECIFIC ALLOCATION (32-byte aligned for AVX2)
//==============================================================================

static void* aligned_alloc32(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 32);
#else
    void* p = NULL;
    if (posix_memalign(&p, 32, size) != 0)
        return NULL;
    return p;
#endif
}

static void aligned_free32(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

//==============================================================================
// WORKSPACE STRUCTURE (OPAQUE)
//==============================================================================

typedef float qrw_t;

/**
 * @brief QR workspace - all buffers pre-allocated for zero hot-path mallocs
 * 
 * @note Thread-safe: each thread should have its own workspace
 * @note Sized for maximum dimensions (m_max, n_max) specified at creation
 */
typedef struct qr_workspace_s {
    // Dimensions
    uint16_t m_max;
    uint16_t n_max;
    uint16_t ib;
    
    // QR/GEQP3 shared buffers (32-byte aligned)
    qrw_t *tau;          // min(m,n) - Householder scalars
    qrw_t *tmp;          // m - panel factorization workspace
    qrw_t *work;         // m - CPQR Householder gather buffer
    
    // Compact-WY buffers
    qrw_t *T;            // ib×ib - triangular factor
    qrw_t *Cpack;        // mc×kc - packed trailing matrix
    qrw_t *Y;            // ib×kc - intermediate Y = Vᵀ·C
    qrw_t *Z;            // ib×kc - intermediate Z = T·Y
    
    // CPQR-specific
    qrw_t *vn1;          // n - working column norms
    qrw_t *vn2;          // n - reference column norms
    
    // Memory accounting
    size_t total_bytes;
} qr_workspace;

//==============================================================================
// WORKSPACE API
//==============================================================================

/**
 * @brief Allocate QR workspace for matrices up to m_max × n_max
 * 
 * @param m_max Maximum number of rows
 * @param n_max Maximum number of columns
 * @param ib Panel width (0 = auto, typically 64-96)
 * 
 * @return Workspace pointer on success, NULL on allocation failure
 * 
 * @note COLD PATH - call once, reuse many times
 * @note All buffers are 32-byte aligned for AVX2
 */
qr_workspace* qr_workspace_alloc(uint16_t m_max, uint16_t n_max, uint16_t ib)
{
    if (m_max == 0 || n_max == 0)
        return NULL;
    
    if (ib == 0)
        ib = QRW_IB_DEFAULT;
    
    qr_workspace *ws = (qr_workspace*)malloc(sizeof(qr_workspace));
    if (!ws)
        return NULL;
    
    memset(ws, 0, sizeof(qr_workspace));
    ws->m_max = m_max;
    ws->n_max = n_max;
    ws->ib = ib;
    ws->total_bytes = sizeof(qr_workspace);
    
    // Calculate buffer sizes
    const uint16_t mn_max = (m_max < n_max) ? m_max : n_max;
    const size_t kc = LINALG_BLOCK_KC;
    const size_t mc = LINALG_BLOCK_MC;
    
    // Allocate all buffers (fail-safe with goto cleanup)
    #define ALLOC_BUF(ptr, count) do { \
        size_t bytes = (size_t)(count) * sizeof(qrw_t); \
        ws->ptr = (qrw_t*)aligned_alloc32(bytes); \
        if (!ws->ptr) goto cleanup_fail; \
        ws->total_bytes += bytes; \
    } while(0)
    
    ALLOC_BUF(tau, mn_max);
    ALLOC_BUF(tmp, m_max);
    ALLOC_BUF(work, m_max);
    ALLOC_BUF(T, (size_t)ib * ib);
    ALLOC_BUF(Cpack, mc * kc);
    ALLOC_BUF(Y, (size_t)ib * kc);
    ALLOC_BUF(Z, (size_t)ib * kc);
    ALLOC_BUF(vn1, n_max);
    ALLOC_BUF(vn2, n_max);
    
    #undef ALLOC_BUF
    
    return ws;
    
cleanup_fail:
    // Free any successfully allocated buffers
    if (ws->tau) aligned_free32(ws->tau);
    if (ws->tmp) aligned_free32(ws->tmp);
    if (ws->work) aligned_free32(ws->work);
    if (ws->T) aligned_free32(ws->T);
    if (ws->Cpack) aligned_free32(ws->Cpack);
    if (ws->Y) aligned_free32(ws->Y);
    if (ws->Z) aligned_free32(ws->Z);
    if (ws->vn1) aligned_free32(ws->vn1);
    if (ws->vn2) aligned_free32(ws->vn2);
    free(ws);
    return NULL;
}

/**
 * @brief Free QR workspace and all associated buffers
 * 
 * @param ws Workspace to free (NULL-safe)
 */
void qr_workspace_free(qr_workspace *ws)
{
    if (!ws)
        return;
    
    aligned_free32(ws->tau);
    aligned_free32(ws->tmp);
    aligned_free32(ws->work);
    aligned_free32(ws->T);
    aligned_free32(ws->Cpack);
    aligned_free32(ws->Y);
    aligned_free32(ws->Z);
    aligned_free32(ws->vn1);
    aligned_free32(ws->vn2);
    free(ws);
}

/**
 * @brief Query workspace memory usage
 * 
 * @param ws Workspace to query
 * @return Total bytes allocated (0 if ws=NULL)
 */
size_t qr_workspace_bytes(const qr_workspace *ws)
{
    return ws ? ws->total_bytes : 0;
}

//==============================================================================
// SCALAR (REFERENCE) QR - UNCHANGED FROM ORIGINAL
//==============================================================================

//USE NAIVE MUL HERE. 

static int qr_scalar(const float *RESTRICT A, float *RESTRICT Q,
                     float *RESTRICT R, uint16_t m, uint16_t n, bool only_R)
{
    // [ORIGINAL IMPLEMENTATION - UNCHANGED]
    // This is the fallback for small matrices or non-AVX2 systems
    // Kept identical to preserve correctness
    
    const uint16_t l = (m - 1 < n) ? (m - 1) : n;

    memcpy(R, A, (size_t)m * n * sizeof(float));

    float *H = (float *)malloc((size_t)m * m * sizeof(float));
    float *W = (float *)malloc((size_t)m * sizeof(float));
    float *WW = (float *)malloc((size_t)m * m * sizeof(float));
    float *Hi = (float *)malloc((size_t)m * m * sizeof(float));
    float *HiH = (float *)malloc((size_t)m * m * sizeof(float));
    float *HiR = (float *)malloc((size_t)m * n * sizeof(float));
    if (!H || !W || !WW || !Hi || !HiH || !HiR)
    {
        free(H); free(W); free(WW); free(Hi); free(HiH); free(HiR);
        return -ENOMEM;
    }

    memset(H, 0, (size_t)m * m * sizeof(float));
    for (uint16_t i = 0; i < m; ++i)
        H[(size_t)i * m + i] = 1.0f;

    for (uint16_t k = 0; k < l; ++k)
    {
        float s = 0.0f;
        for (uint16_t i = k; i < m; ++i)
        {
            float x = R[(size_t)i * n + k];
            s += x * x;
        }
        s = sqrtf(s);
        float Rk = R[(size_t)k * n + k];
        if (s == 0.0f)
            continue;
        if (Rk < 0.0f)
            s = -s;
        float r = sqrtf(2.0f * s * (Rk + s));
        if (r == 0.0f)
            continue;

        memset(W, 0, (size_t)m * sizeof(float));
        W[k] = (Rk + s) / r;
        for (uint16_t i = k + 1; i < m; ++i)
            W[i] = R[(size_t)i * n + k] / r;

        // WW = W * Wᵀ (using external mul function)
        extern int mul(float *RESTRICT, const float *RESTRICT, const float *RESTRICT,
                      uint16_t, uint16_t, uint16_t, uint16_t);
        mul(WW, W, W, m, 1, 1, m);
        
        for (size_t i = 0; i < (size_t)m * m; ++i)
            Hi[i] = -2.0f * WW[i];
        for (uint16_t i = 0; i < m; ++i)
            Hi[(size_t)i * m + i] += 1.0f;

        if (!only_R)
        {
            mul(HiH, Hi, H, m, m, m, m);
            memcpy(H, HiH, (size_t)m * m * sizeof(float));
        }
        mul(HiR, Hi, R, m, m, m, n);
        memcpy(R, HiR, (size_t)m * n * sizeof(float));
    }

    if (!only_R)
    {
        float *Hin = (float *)malloc((size_t)m * m * sizeof(float));
        if (!Hin)
        {
            free(H); free(W); free(WW); free(Hi); free(HiH); free(HiR);
            return -ENOMEM;
        }
        memcpy(Hin, H, (size_t)m * m * sizeof(float));
        
        extern int inv(float *RESTRICT, const float *RESTRICT, uint16_t);
        int rc = inv(H, Hin, m);
        free(Hin);
        if (rc != 0)
        {
            free(H); free(W); free(WW); free(Hi); free(HiH); free(HiR);
            return -ENOTSUP;
        }
        memcpy(Q, H, (size_t)m * m * sizeof(float));
    }

    free(H); free(W); free(WW); free(Hi); free(HiH); free(HiR);
    return 0;
}