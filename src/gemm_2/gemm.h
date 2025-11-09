/**
 * @file gemm.h
 * @brief High-Performance GEMM Library - Public API
 * 
 * Features:
 * - Tier 1: Register-only kernels for 4×4, 6×6, 8×8
 * - Tier 2: Planned execution with cache blocking
 * - Static memory pool (512×512 default, zero allocation)
 * - Dynamic fallback for larger matrices
 * - Symmetric operations optimized for Kalman filters
 * 
 * @author TUGBARS
 * @date 2025
 */

#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// ERROR CODES
//==============================================================================

typedef enum {
    GEMM_OK = 0,                  // Success
    GEMM_ERR_INVALID_PTR = -1,    // NULL pointer passed
    GEMM_ERR_INVALID_DIM = -2,    // Invalid matrix dimensions
    GEMM_ERR_NO_MEMORY = -3,      // Memory allocation failed
    GEMM_ERR_OVERFLOW = -4,       // Integer overflow in size calculation
    GEMM_ERR_STATIC_TOO_LARGE = -5 // Matrix too large for static pool
} gemm_error_t;

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @brief Get current static pool size limit
 * @return Maximum dimension supported by static pool
 */
int gemm_get_static_limit(void);

/**
 * @brief Check if dimensions fit in static pool
 * @return 1 if fits, 0 otherwise
 */
int gemm_fits_static(size_t M, size_t K, size_t N);

/**
 * @brief Query workspace size needed for dynamic allocation
 * @return Bytes required for workspace
 */
size_t gemm_workspace_query(size_t M, size_t K, size_t N);

//==============================================================================
// CORE GEMM OPERATIONS
//==============================================================================

/**
 * @brief General matrix multiply: C = alpha*A*B + beta*C
 * 
 * Automatically selects optimal execution path:
 * - Tier 1: Small fixed sizes (4×4, 6×6, 8×8)
 * - Tier 2: Larger matrices with planning
 * - Static pool if dimensions fit, dynamic allocation otherwise
 * 
 * @param C Output matrix (M×N, row-major)
 * @param A Input matrix (M×K, row-major)
 * @param B Input matrix (K×N, row-major)
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for C
 * @return 0 on success, negative error code on failure
 */
int gemm_auto(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta);

/**
 * @brief GEMM with explicit static pool (zero allocation)
 * 
 * Forces use of thread-local static pool.
 * Returns error if dimensions exceed static pool limit.
 * 
 * @return 0 on success, GEMM_ERR_STATIC_TOO_LARGE if too large
 */
int gemm_static(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta);

/**
 * @brief GEMM with explicit dynamic allocation
 * 
 * Forces use of aligned malloc for workspace.
 * Use for matrices larger than static pool.
 * 
 * @return 0 on success, GEMM_ERR_NO_MEMORY if allocation fails
 */
int gemm_dynamic(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N,
    float alpha, float beta);

/**
 * @brief Simple matrix multiply: C = A*B (alpha=1, beta=0)
 */
static inline int gemm(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t M, size_t K, size_t N)
{
    return gemm_auto(C, A, B, M, K, N, 1.0f, 0.0f);
}

//==============================================================================
// SYMMETRIC OPERATIONS (Optimized for Kalman Filters)
//==============================================================================

/**
 * @brief Symmetric sandwich product: C = A*B*A^T
 * 
 * Optimized for B symmetric (only upper/lower triangle accessed).
 * Critical for Kalman filter covariance propagation: P = F*P*F^T
 * 
 * @param C Output matrix (n×n symmetric, only upper triangle computed)
 * @param A Input matrix (n×n)
 * @param B Input matrix (n×n symmetric)
 * @param n Matrix dimension
 * @param workspace Temporary workspace (n×n floats), can use static pool
 * @return 0 on success, negative on error
 */
int gemm_symmetric_sandwich(
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    size_t n,
    float * restrict workspace);

/**
 * @brief Symmetric rank-k update: C = beta*C + alpha*A*A^T
 * 
 * Only computes upper or lower triangle (symmetric result).
 * Used in Kalman for process noise: P = P + Q
 * 
 * @param C In/out: n×n symmetric matrix
 * @param A Input: n×k matrix
 * @param n Number of rows/cols in C
 * @param k Number of columns in A
 * @param alpha Scalar multiplier for A*A^T
 * @param beta Scalar multiplier for C
 * @param lower 0=upper triangle, 1=lower triangle
 * @return 0 on success, negative on error
 */
int gemm_syrk(
    float * restrict C,
    const float * restrict A,
    size_t n, size_t k,
    float alpha, float beta,
    int lower);

//==============================================================================
// KALMAN FILTER OPERATIONS
//==============================================================================

/**
 * @brief Kalman predict covariance: P = F*P*F^T + Q
 * 
 * Optimized for symmetric P and Q matrices.
 * Uses static pool for workspace if n ≤ static limit.
 * 
 * @param P In/out: State covariance (n×n symmetric)
 * @param F State transition matrix (n×n)
 * @param Q Process noise covariance (n×n symmetric)
 * @param n State dimension
 * @return 0 on success, negative on error
 */
int kalman_predict_covariance(
    float * restrict P,
    const float * restrict F,
    const float * restrict Q,
    size_t n);

/**
 * @brief Kalman update covariance: P = (I-K*H)*P*(I-K*H)^T + K*R*K^T
 * 
 * Joseph form for numerical stability.
 * 
 * @param P In/out: State covariance (n×n symmetric)
 * @param K Kalman gain (n×m)
 * @param H Measurement model (m×n)
 * @param R Measurement noise covariance (m×m symmetric)
 * @param n State dimension
 * @param m Measurement dimension
 * @return 0 on success, negative on error
 */
int kalman_update_covariance(
    float * restrict P,
    const float * restrict K,
    const float * restrict H,
    const float * restrict R,
    size_t n, size_t m);

/**
 * @brief Simplified Kalman update: P = (I-K*H)*P
 * 
 * Standard form (not Joseph form). Faster but less numerically stable.
 * 
 * @param P In/out: State covariance (n×n symmetric)
 * @param K Kalman gain (n×m)
 * @param H Measurement model (m×n)
 * @param n State dimension
 * @param m Measurement dimension
 * @return 0 on success, negative on error
 */
int kalman_update_simple(
    float * restrict P,
    const float * restrict K,
    const float * restrict H,
    size_t n, size_t m);

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

/**
 * @brief Allocate aligned memory
 * @param alignment Alignment in bytes (must be power of 2)
 * @param size Number of bytes to allocate
 * @return Pointer to aligned memory, or NULL on failure
 */
void* gemm_aligned_alloc(size_t alignment, size_t size);

/**
 * @brief Free aligned memory
 * @param ptr Pointer returned by gemm_aligned_alloc
 */
void gemm_aligned_free(void* ptr);

/**
 * @brief Initialize thread-local static pool
 * 
 * Called automatically on first use, but can be called explicitly
 * to avoid latency on first GEMM call.
 */
void gemm_static_init(void);

/**
 * @brief Get error string for error code
 * @param error Error code
 * @return Human-readable error message
 */
const char* gemm_strerror(gemm_error_t error);

#ifdef __cplusplus
}
#endif

#endif // GEMM_H