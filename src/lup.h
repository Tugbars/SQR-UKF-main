// SPDX-License-Identifier: MIT
/**
 * @file lup.h
 * @brief LU decomposition with partial pivoting - workspace-based API
 *
 * Performs P*A = L*U factorization where:
 * - P is a permutation matrix (stored as permutation vector)
 * - L is unit lower triangular
 * - U is upper triangular
 *
 * Uses blocked BLAS-3 algorithm with GEMM for trailing updates.
 * Workspace-based design eliminates malloc from hot paths.
 *
 * @author Your Name
 * @date 2025
 */

#ifndef LUP_H
#define LUP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// WORKSPACE API
//==============================================================================

/**
 * @brief Opaque workspace for LUP operations
 * 
 * Contains pre-allocated buffers for:
 * - GEMM operations (trailing matrix updates)
 * - Packing buffers (A and B matrices)
 * 
 * Thread-safe: Each thread should have its own workspace.
 * Reusable: Can be used for multiple LUP operations.
 */
typedef struct lup_workspace lup_workspace_t;

/**
 * @brief Query workspace size for LUP factorization
 * 
 * @param n Matrix dimension (n×n)
 * @return Required bytes for workspace
 * 
 * @note Returns 0 if n == 0
 * @note Size includes embedded GEMM workspace
 * 
 * @example
 * size_t size = lup_workspace_query(128);
 * uint8_t buffer[size] __attribute__((aligned(32)));
 * lup_workspace_t *ws = lup_workspace_init(buffer, size);
 */
size_t lup_workspace_query(uint16_t n);

/**
 * @brief Create LUP workspace (heap allocation)
 * 
 * @param size Size in bytes (from lup_workspace_query)
 * @return Workspace handle, or NULL on allocation failure
 * 
 * @note Caller must call lup_workspace_destroy() when done
 * @note Buffer is 32-byte aligned (required for AVX2/AVX-512)
 * 
 * @example
 * size_t size = lup_workspace_query(n);
 * lup_workspace_t *ws = lup_workspace_create(size);
 * if (!ws) { handle_error(); }
 * 
 * lup_ws(A, LU, P, n, ws);
 * lup_workspace_destroy(ws);
 */
lup_workspace_t *lup_workspace_create(size_t size);

/**
 * @brief Initialize workspace from user-provided buffer
 * 
 * @param buffer User memory (must be 32-byte aligned)
 * @param size Buffer size in bytes
 * @return Workspace handle, or NULL if buffer invalid/too small
 * 
 * @note Buffer ownership remains with caller
 * @note lup_workspace_destroy() will NOT free the buffer
 * @note Useful for stack allocation or embedded systems
 * 
 * @example Stack allocation:
 * uint8_t buffer[4096] __attribute__((aligned(32)));
 * lup_workspace_t *ws = lup_workspace_init(buffer, sizeof(buffer));
 * lup_ws(A, LU, P, n, ws);
 * lup_workspace_destroy(ws); // Only frees handle, not buffer
 * 
 * @example Static allocation:
 * static uint8_t lup_buffer[8192] __attribute__((aligned(32)));
 * void kalman_update(void) {
 *     lup_workspace_t *ws = lup_workspace_init(lup_buffer, sizeof(lup_buffer));
 *     lup_ws(P_matrix, LU, pivots, n, ws);
 *     lup_workspace_destroy(ws);
 * }
 */
lup_workspace_t *lup_workspace_init(void *buffer, size_t size);

/**
 * @brief Destroy workspace
 * 
 * @param ws Workspace handle (may be NULL)
 * 
 * @note If workspace was created with lup_workspace_create(), frees buffer
 * @note If workspace was created with lup_workspace_init(), does NOT free buffer
 * @note Always frees the workspace handle itself
 * @note Safe to call with NULL pointer
 */
void lup_workspace_destroy(lup_workspace_t *ws);

//==============================================================================
// LUP FACTORIZATION API
//==============================================================================

/**
 * @brief LU decomposition with partial pivoting (workspace version)
 * 
 * Computes P*A = L*U where:
 * - P is a permutation (stored as permutation vector in P)
 * - L is unit lower triangular (diagonal implicitly 1.0)
 * - U is upper triangular
 * 
 * @param A Input matrix (row-major, n×n)
 * @param LU Output factorization (row-major, n×n, may alias A)
 * @param P Output permutation vector (size n)
 *          P[i] = original row index now at position i
 * @param n Matrix dimension
 * @param ws Workspace (must be >= lup_workspace_query(n))
 * 
 * @return 0 on success
 * @return -EINVAL if n == 0 or ws == NULL
 * @return -ENOSPC if workspace too small
 * @return -ENOTSUP if matrix is singular (or near-singular)
 * 
 * @note Zero malloc - all memory comes from workspace
 * @note In-place allowed: A and LU may point to same memory
 * @note Uses blocked BLAS-3 algorithm with AVX2/AVX-512 GEMM
 * @note Relative singularity tolerance: n * FLT_EPSILON * ||row||_∞
 * 
 * @example Hot path usage (zero malloc per call):
 * // One-time setup
 * size_t ws_size = lup_workspace_query(n);
 * lup_workspace_t *ws = lup_workspace_create(ws_size);
 * 
 * // Reuse across iterations
 * for (int iter = 0; iter < 1000; iter++) {
 *     lup_ws(A, LU, P, n, ws);
 *     // ... use LU factorization ...
 * }
 * 
 * lup_workspace_destroy(ws);
 * 
 * @example Multithreading:
 * #pragma omp parallel
 * {
 *     lup_workspace_t *ws = lup_workspace_create(ws_size);
 *     
 *     #pragma omp for
 *     for (int i = 0; i < num_matrices; i++) {
 *         lup_ws(matrices[i], results[i], pivots[i], n, ws);
 *     }
 *     
 *     lup_workspace_destroy(ws);
 * }
 */
int lup_ws(
    const float *restrict A,
    float *restrict LU,
    uint8_t *restrict P,
    uint16_t n,
    lup_workspace_t *ws
);

/**
 * @brief LU decomposition with partial pivoting (convenience wrapper)
 * 
 * Same as lup_ws(), but allocates workspace internally.
 * 
 * @param A Input matrix (row-major, n×n)
 * @param LU Output factorization (row-major, n×n, may alias A)
 * @param P Output permutation vector (size n)
 * @param n Matrix dimension
 * 
 * @return 0 on success
 * @return -EINVAL if n == 0
 * @return -ENOMEM if workspace allocation fails
 * @return -ENOTSUP if matrix is singular
 * 
 * @warning Allocates workspace internally (malloc overhead)
 * @warning For hot paths or repeated calls, use lup_ws() instead
 * 
 * @note Provided for backward compatibility and convenience
 * @note Existing code using this function continues to work
 * 
 * @example Simple usage (when performance is not critical):
 * float A[64] = { ... };
 * float LU[64];
 * uint8_t P[8];
 * 
 * if (lup(A, LU, P, 8) != 0) {
 *     fprintf(stderr, "LUP failed\n");
 * }
 */
int lup(
    const float *restrict A,
    float *restrict LU,
    uint8_t *restrict P,
    uint16_t n
);

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

/**
 * @brief Apply row permutation to a matrix
 * 
 * Permutes rows of B according to permutation vector P:
 * B_out[i, :] = B_in[P[i], :]
 * 
 * @param B Input matrix (row-major, n×m)
 * @param B_out Output matrix (row-major, n×m, may alias B)
 * @param P Permutation vector (size n) from lup() or lup_ws()
 * @param n Number of rows
 * @param m Number of columns
 * 
 * @note In-place allowed: B and B_out may point to same memory
 * @note Useful for solving P*A*x = P*b systems
 * 
 * @example Solving A*x = b using LUP:
 * // 1. Factor: P*A = L*U
 * lup_ws(A, LU, P, n, ws);
 * 
 * // 2. Permute RHS: b' = P*b
 * float b_permuted[n];
 * lup_apply_permutation(b, b_permuted, P, n, 1);
 * 
 * // 3. Solve L*y = b'
 * solve_lower_unit(LU, b_permuted, y, n);
 * 
 * // 4. Solve U*x = y
 * solve_upper(LU, y, x, n);
 */
void lup_apply_permutation(
    const float *restrict B,
    float *restrict B_out,
    const uint8_t *restrict P,
    uint16_t n,
    uint16_t m
);

/**
 * @brief Apply inverse permutation to a matrix
 * 
 * Applies inverse permutation: B_out[P[i], :] = B_in[i, :]
 * 
 * @param B Input matrix (row-major, n×m)
 * @param B_out Output matrix (row-major, n×m, must NOT alias B)
 * @param P Permutation vector (size n)
 * @param n Number of rows
 * @param m Number of columns
 * 
 * @note In-place NOT allowed: B and B_out must be different
 */
void lup_apply_inverse_permutation(
    const float *restrict B,
    float *restrict B_out,
    const uint8_t *restrict P,
    uint16_t n,
    uint16_t m
);

//==============================================================================
// CONFIGURATION (compile-time tuning)
//==============================================================================

/**
 * @brief Block size for panel factorization
 * 
 * Tuning notes:
 * - Larger values: Better GEMM performance in trailing updates
 * - Smaller values: Less memory for workspace, better cache locality in panel
 * - Typical range: 64-256
 * - Default: 128
 */
#ifndef LUP_NB
#define LUP_NB 128
#endif

#ifdef __cplusplus
}
#endif

#endif /* LUP_H */