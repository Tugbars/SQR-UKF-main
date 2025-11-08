/**
 * @file gemm.h
 * @brief High-Performance GEMM (General Matrix Multiply) Library
 * 
 * Optimized for Intel Core i9-14900K with AVX2/FMA
 * Features:
 * - Planning-based architecture for amortized overhead
 * - Size-aware routing (tiny/small/medium/large)
 * - Specialized kernels for small matrices (Kalman filters)
 * - Symmetric matrix operations
 * 
 * @author TUGBARS
 * @date 2024
 */

#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// ERROR CODES
//==============================================================================

typedef enum {
    GEMM_OK = 0,
    GEMM_ERR_INVALID_DIM = -1,
    GEMM_ERR_INVALID_PTR = -2,
    GEMM_ERR_MISALIGNED = -3,
    GEMM_ERR_OVERFLOW = -4,
    GEMM_ERR_NO_MEMORY = -5,
} gemm_error_t;

//==============================================================================
// BLOCKING PARAMETERS
//==============================================================================

#define LINALG_BLOCK_MC 128  // M cache blocking
#define LINALG_BLOCK_KC 256  // K cache blocking
#define LINALG_BLOCK_JC 256  // N cache blocking (JC)

//==============================================================================
// OPAQUE TYPES
//==============================================================================

/**
 * @brief Opaque GEMM execution plan
 * 
 * Contains pre-computed tiling, packing strategies, and kernel selections.
 * Create once, execute many times for best performance.
 */
typedef struct gemm_plan gemm_plan_t;

//==============================================================================
// BASIC GEMM API (LEGACY COMPATIBILITY)
//==============================================================================

/**
 * @brief Basic matrix multiply: C = A * B
 * 
 * Simple interface for compatibility. Uses auto-routing internally.
 * 
 * @param C Output matrix [M×N] (row-major)
 * @param A Left matrix [M×K] (row-major)
 * @param B Right matrix [K×N] (row-major)
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param Arow_stride Leading dimension of A (typically K)
 * @param N Number of columns in B and C
 * @return 0 on success, negative error code on failure
 */
int mul(float *C, const float *A, const float *B,
        uint16_t M, uint16_t K, uint16_t Arow_stride, uint16_t N);

//==============================================================================
// AUTO-ROUTING GEMM API
//==============================================================================

/**
 * @brief Automatic GEMM with size-aware routing: C = alpha*A*B + beta*C
 * 
 * Automatically selects optimal implementation based on matrix dimensions:
 * - Tiny (≤16): Register-only kernels
 * - Small (≤64): Direct kernels without blocking
 * - Medium (≤256): Single-level blocking
 * - Large (>256): Full three-level blocking
 * 
 * @param C Input/output matrix [M×N]
 * @param A Input matrix [M×K]
 * @param B Input matrix [K×N]
 * @param M Rows of A and C
 * @param K Columns of A, rows of B
 * @param N Columns of B and C
 * @param alpha Scalar for A*B
 * @param beta Scalar for C (0 to overwrite, 1 to add)
 * @return 0 on success, error code on failure
 */
int gemm_auto(float *C, const float *A, const float *B,
              size_t M, size_t K, size_t N,
              float alpha, float beta);

//==============================================================================
// PLANNING-BASED GEMM API (RECOMMENDED FOR REPEATED OPERATIONS)
//==============================================================================

/**
 * @brief Create GEMM execution plan with validation
 * 
 * Pre-computes optimal tiling, selects kernels, allocates workspace.
 * For best performance, create once and reuse for many executions.
 * 
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B  
 * @param N Number of columns in B and C
 * @param A Pointer to A matrix (for alignment analysis, can be NULL)
 * @param B Pointer to B matrix (for alignment analysis, can be NULL)
 * @param C Pointer to C matrix (for alignment analysis, can be NULL)
 * @param alpha Scalar for A*B
 * @param beta Scalar for C
 * @param[out] error Optional error code output
 * @return Plan pointer on success, NULL on failure
 */
gemm_plan_t* gemm_plan_create_safe(
    size_t M, size_t K, size_t N,
    const float *A, const float *B, float *C,
    float alpha, float beta,
    gemm_error_t *error);

/**
 * @brief Execute pre-planned GEMM
 * 
 * Executes C = alpha*A*B + beta*C using pre-computed plan.
 * Zero runtime overhead - all decisions made during planning.
 * 
 * @param plan Execution plan from gemm_plan_create_safe
 * @param C Output matrix
 * @param A Input matrix A
 * @param B Input matrix B
 * @param alpha Scalar for A*B
 * @param beta Scalar for C
 * @return 0 on success, error code on failure
 */
int gemm_execute_plan(
    gemm_plan_t *plan,
    float *C,
    const float *A,
    const float *B,
    float alpha,
    float beta);

/**
 * @brief Destroy GEMM plan and free resources
 * 
 * @param plan Plan to destroy (can be NULL)
 */
void gemm_plan_destroy(gemm_plan_t *plan);

//==============================================================================
// SPECIALIZED SMALL MATRIX OPERATIONS
//==============================================================================

/**
 * @brief Optimized GEMM for small matrices (≤16×16×16)
 * 
 * Uses register-only kernels for common small sizes (4×4, 6×6, 8×8, 12×12).
 * Ideal for Kalman filters and other real-time applications.
 * 
 * @param C Output matrix [M×N]
 * @param A Input matrix [M×K]
 * @param B Input matrix [K×N]
 * @param M Rows (must be ≤16)
 * @param K Inner dimension (must be ≤16)
 * @param N Columns (must be ≤16)
 * @param alpha Scalar for A*B
 * @param beta Scalar for C
 * @return 0 on success, -1 if size not supported
 */
int gemm_small_matrix(
    float *C,
    const float *A,
    const float *B,
    size_t M, size_t K, size_t N,
    float alpha, float beta);

//==============================================================================
// SYMMETRIC MATRIX OPERATIONS (FOR KALMAN FILTERS)
//==============================================================================

/**
 * @brief Symmetric sandwich product: C = A*B*A^T where B is symmetric
 * 
 * Optimized for Kalman filter covariance update: P = F*P*F^T
 * Exploits symmetry to reduce computation by ~50%.
 * 
 * @param C Output matrix [n×n], symmetric
 * @param A Input matrix [n×n]
 * @param B Input symmetric matrix [n×n]
 * @param n Matrix dimension
 * @param workspace Temporary storage [n×n]
 */
void gemm_symmetric_sandwich(
    float *C,
    const float *A,
    const float *B,
    size_t n,
    float *workspace);

/**
 * @brief Symmetric rank-k update: C = beta*C + alpha*A*A^T
 * 
 * Used for covariance updates in Kalman filters.
 * 
 * @param C Input/output symmetric matrix [n×n]
 * @param A Input matrix [n×k]
 * @param n Number of rows in A and dimension of C
 * @param k Number of columns in A
 * @param alpha Scalar for A*A^T
 * @param beta Scalar for C
 * @param lower 0 for upper triangle, 1 for lower triangle
 */
void gemm_syrk(
    float *C,
    const float *A,
    size_t n, size_t k,
    float alpha, float beta,
    int lower);

//==============================================================================
// KALMAN FILTER SPECIFIC OPERATIONS
//==============================================================================

/**
 * @brief Kalman filter covariance prediction: P = F*P*F^T + Q
 * 
 * @param P Input/output covariance matrix [n×n], symmetric
 * @param F State transition matrix [n×n]
 * @param Q Process noise covariance [n×n], symmetric
 * @param n State dimension
 * @param workspace Temporary storage [n×n]
 */
void kalman_predict_covariance(
    float *P,
    const float *F,
    const float *Q,
    size_t n,
    float *workspace);

/**
 * @brief Kalman filter covariance update: P = (I - K*H)*P
 * 
 * @param P Input/output covariance [n×n]
 * @param K Kalman gain [n×m]
 * @param H Measurement model [m×n]
 * @param n State dimension
 * @param m Measurement dimension
 * @param workspace Temporary storage [n×n]
 */
void kalman_update_covariance(
    float *P,
    const float *K,
    const float *H,
    size_t n, size_t m,
    float *workspace);

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

/**
 * @brief Query workspace size for GEMM operations
 * 
 * @param M Number of rows
 * @param K Inner dimension
 * @param N Number of columns
 * @return Required workspace size in bytes
 */
size_t gemm_workspace_query(size_t M, size_t K, size_t N);

/**
 * @brief Aligned memory allocation
 * 
 * @param alignment Required alignment (must be power of 2)
 * @param size Number of bytes to allocate
 * @return Aligned pointer or NULL on failure
 */
void* gemm_aligned_alloc(size_t alignment, size_t size);

/**
 * @brief Free aligned memory
 * 
 * @param ptr Pointer from gemm_aligned_alloc
 */
void gemm_aligned_free(void *ptr);

//==============================================================================
// VERSION INFORMATION
//==============================================================================

/**
 * @brief Get library version string
 * @return Version string (e.g., "1.0.0-avx2")
 */
const char* gemm_version(void);

/**
 * @brief Check CPU feature support
 * @return Bitmask of supported features (AVX2, FMA, etc.)
 */
uint32_t gemm_cpu_features(void);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_H */