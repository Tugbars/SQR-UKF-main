#ifndef QR_H
#define QR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

//==============================================================================
// QR WORKSPACE STRUCTURE (FULL DEFINITION - needs to be visible to users)
//==============================================================================

/**
 * @brief QR workspace - all buffers pre-allocated for zero hot-path mallocs
 * 
 * @note Thread-safe: each thread should have its own workspace
 * @note Sized for maximum dimensions (m_max, n_max) specified at creation
 */
typedef struct {
    uint16_t m_max;
    uint16_t n_max;
    uint16_t ib;
    float *tau;
    float *tmp;
    float *work;
    float *T;      // Aligned
    float *Cpack;  // Aligned
    float *Y;      // Aligned
    float *YT;     // Aligned (pre-transposed Y)
    float *Z;      // Aligned
    float *Z_temp; // ✅ NEW: Aligned temp buffer
    float *YZ;     // ✅ NEW: Aligned temp buffer
    float *vn1;
    float *vn2;
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
 */
qr_workspace* qr_workspace_alloc(uint16_t m_max, uint16_t n_max, uint16_t ib);

/**
 * @brief Free QR workspace and all associated buffers
 * 
 * @param ws Workspace to free (NULL-safe)
 */
void qr_workspace_free(qr_workspace *ws);

/**
 * @brief Query workspace memory usage
 * 
 * @param ws Workspace to query
 * @return Total bytes allocated (0 if ws=NULL)
 */
size_t qr_workspace_bytes(const qr_workspace *ws);

//==============================================================================
// QR DECOMPOSITION API
//==============================================================================

/**
 * @brief Legacy QR decomposition (allocates workspace internally)
 * 
 * @param A Input matrix (m×n, row-major)
 * @param Q Output orthogonal matrix (m×m, row-major), may be NULL if only_R=true
 * @param R Output upper-triangular (m×n, row-major)
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R If true, skip Q formation
 * 
 * @return 0 on success, negative errno on failure
 */
int qr(const float *A, float *Q, float *R, uint16_t m, uint16_t n, bool only_R);

/**
 * @brief QR decomposition using pre-allocated workspace (HOT PATH - ZERO MALLOC)
 * 
 * @param ws Pre-allocated workspace
 * @param A Input matrix (m×n, row-major)
 * @param Q Output orthogonal matrix (m×m, row-major), may be NULL if only_R=true
 * @param R Output upper-triangular (m×n, row-major)
 * @param m Number of rows (must be ≤ m_max from workspace)
 * @param n Number of columns (must be ≤ n_max from workspace)
 * @param only_R If true, skip Q formation
 * 
 * @return 0 on success, negative errno on failure
 */
int qr_ws(qr_workspace *ws, const float *A, float *Q, float *R, 
          uint16_t m, uint16_t n, bool only_R);

/**
 * @brief In-place QR decomposition using workspace
 * 
 * @param ws Pre-allocated workspace
 * @param A_inout Input/output matrix (m×n, row-major) - overwritten with R
 * @param Q Output orthogonal matrix (m×m, row-major), may be NULL if only_R=true
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R If true, skip Q formation
 * 
 * @return 0 on success, negative errno on failure
 */
int qr_ws_inplace(qr_workspace *ws, float *A_inout, float *Q, 
                  uint16_t m, uint16_t n, bool only_R);

//==============================================================================
// COLUMN PIVOTING QR (CPQR) API
//==============================================================================

/**
 * @brief Legacy CPQR (allocates workspace internally)
 * 
 * @param A Input/output matrix (m×n, row-major) - factored in-place
 * @param m Number of rows
 * @param n Number of columns
 * @param tau Output Householder scalars (length ≥ min(m,n))
 * @param jpvt Output column permutation (length ≥ n)
 * @param ib Panel width
 * @param kw Look-ahead window width
 * 
 * @return 0 on success, negative errno on failure
 */
int geqp3(float *A, uint16_t m, uint16_t n, float *tau, int *jpvt, 
          uint16_t ib, uint16_t kw);

/**
 * @brief CPQR using pre-allocated workspace (HOT PATH - ZERO MALLOC)
 * 
 * @param ws Pre-allocated workspace
 * @param A Input/output matrix (m×n, row-major) - factored in-place
 * @param m Number of rows
 * @param n Number of columns
 * @param tau Output Householder scalars (length ≥ min(m,n))
 * @param jpvt Output column permutation (length ≥ n)
 * @param ib Panel width (0 = use workspace default)
 * @param kw Look-ahead window width
 * 
 * @return 0 on success, negative errno on failure
 */
int geqp3_ws(qr_workspace *ws, float *A, uint16_t m, uint16_t n, 
             float *tau, int *jpvt, uint16_t ib, uint16_t kw);

/**
 * @brief Blocked CPQR wrapper (default kw=128)
 * 
 * @param A Input/output matrix (m×n, row-major)
 * @param m Number of rows
 * @param n Number of columns
 * @param ib Panel width
 * @param tau Output Householder scalars
 * @param jpvt Output column permutation
 * 
 * @return 0 on success, negative errno on failure
 */
int geqp3_blocked(float *A, uint16_t m, uint16_t n, uint16_t ib, 
                  float *tau, int *jpvt);



/**
 * @brief QR decomposition using pre-allocated workspace (SCALAR reference path)
 *
 * @param ws Pre-allocated workspace
 * @param A  Input matrix (m×n, row-major)
 * @param Q  Output orthogonal matrix (m×m, row-major), may be NULL if only_R=true
 * @param R  Output upper-triangular (m×n, row-major)
 * @param m  Rows (≤ m_max)
 * @param n  Cols (≤ n_max)
 * @param only_R If true, skip Q formation
 *
 * @return 0 on success, negative errno on failure
 */
int qr_ws_scalar(qr_workspace *ws, const float *A, float *Q, float *R,
                 uint16_t m, uint16_t n, bool only_R);

/**
 * @brief Convenience wrapper for scalar path (allocates/free workspace internally)
 */
int qr_scalar_only(const float *A, float *Q, float *R,
                   uint16_t m, uint16_t n, bool only_R);

int qr_ws_blocked(qr_workspace *ws, const float *A, float *Q, float *R,
                  uint16_t m, uint16_t n, bool only_R);

#endif // QR_H