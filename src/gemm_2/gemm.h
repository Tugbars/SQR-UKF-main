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
// OPAQUE TYPES
//==============================================================================

/**
 * @brief Opaque handle for GEMM execution plan
 * 
 * A plan encapsulates:
 * - Matrix dimensions
 * - Cache blocking parameters
 * - Workspace allocation
 * - Kernel selection
 * 
 * Plans can be reused for multiple GEMMs with the same dimensions,
 * amortizing planning overhead across many operations.
 */
typedef struct gemm_plan gemm_plan_t;

//==============================================================================
// CONFIGURATION
//==============================================================================

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
// PLANNED EXECUTION API (for repeated GEMMs with same dimensions)
//==============================================================================

/**
 * @brief Create execution plan for GEMM
 * 
 * Pre-computes cache blocking parameters and allocates workspace.
 * Plan can be reused for multiple GEMMs with same M, K, N.
 * 
 * Use case: Kalman filters, repeated matrix operations in loops
 * 
 * @param M Number of rows in A and C
 * @param K Number of columns in A, rows in B
 * @param N Number of columns in B and C
 * @return Plan handle on success, NULL on failure
 * 
 * @note Call gemm_plan_destroy() when done to free resources
 * 
 * Example:
 * @code
 *   gemm_plan_t *plan = gemm_plan_create(100, 100, 100);
 *   for (int i = 0; i < 1000; i++) {
 *       gemm_execute_plan(plan, C, A, B, 1.0f, 0.0f);
 *   }
 *   gemm_plan_destroy(plan);
 * @endcode
 */
gemm_plan_t* gemm_plan_create(size_t M, size_t K, size_t N);

/**
 * @brief Execute GEMM using pre-created plan
 * 
 * @param plan Execution plan created by gemm_plan_create()
 * @param C Output matrix (M×N, row-major)
 * @param A Input matrix (M×K, row-major)
 * @param B Input matrix (K×N, row-major)
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for C
 * @return 0 on success, negative error code on failure
 * 
 * @note Dimensions must match those used in gemm_plan_create()
 */
int gemm_execute_plan(
    gemm_plan_t *plan,
    float * restrict C,
    const float * restrict A,
    const float * restrict B,
    float alpha, float beta);

/**
 * @brief Destroy execution plan and free resources
 * 
 * @param plan Plan to destroy (can be NULL)
 */
void gemm_plan_destroy(gemm_plan_t *plan);

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

void gemm_get_tuning(size_t M, size_t K, size_t N,
                     size_t *MC, size_t *KC, size_t *NC,
                     size_t *MR, size_t *NR);


#ifdef __cplusplus
}
#endif

#endif // GEMM_H