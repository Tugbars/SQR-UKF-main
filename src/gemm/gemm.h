/**
 * @file gemm.h
 * @brief High-Performance GEMM (General Matrix Multiply) Interface
 *
 * @details
 * Provides optimized single-precision matrix multiplication with AVX2/FMA support.
 * 
 * C = A × B
 * 
 * Where:
 * - A is M×K (row-major)
 * - B is K×N (row-major)
 * - C is M×N (row-major)
 *
 * @author VectorFFT Team
 * @date 2025
 */

#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Single-precision matrix multiplication: C = A × B
 *
 * @details
 * Computes the matrix product C = A × B where all matrices are stored
 * in row-major format. The implementation automatically selects between
 * scalar, AVX2, or AVX-512 code paths based on CPU capabilities and
 * matrix dimensions.
 *
 * **Memory Layout:**
 * - A[i][k] is accessed as A[i * column_a + k]
 * - B[k][j] is accessed as B[k * column_b + j]
 * - C[i][j] is accessed as C[i * column_b + j]
 *
 * **Performance Notes:**
 * - For best performance, ensure matrices are 32-byte aligned
 * - Small matrices (< 64×64) use scalar fallback
 * - Large matrices use blocked algorithm with AVX2/FMA
 *
 * **Requirements:**
 * - column_a must equal row_b (inner dimensions must match)
 * - All pointers must be valid and non-overlapping
 * - Matrix dimensions must fit in uint16_t (max 65535)
 *
 * @param[out] C       Output matrix (M×N), row-major, modified in-place
 * @param[in]  A       Input matrix A (M×K), row-major, read-only
 * @param[in]  B       Input matrix B (K×N), row-major, read-only
 * @param[in]  row_a   Number of rows in A (M)
 * @param[in]  column_a Number of columns in A (K) - must equal row_b
 * @param[in]  row_b   Number of rows in B (K) - must equal column_a
 * @param[in]  column_b Number of columns in B (N)
 *
 * @return 0 on success, negative error code on failure:
 *         - -EINVAL: Dimension mismatch (column_a != row_b)
 *         - -ENOMEM: Memory allocation failed for internal buffers
 *         - -ENOTSUP: SIMD disabled at compile time
 *
 * @note This function is thread-safe (no shared state)
 * @note Internal buffers are allocated on the heap for blocking
 *
 * @par Example:
 * @code
 * // Compute C = A × B where A is 100×50, B is 50×80
 * float A[100 * 50];
 * float B[50 * 80];
 * float C[100 * 80];
 * 
 * // Initialize A and B...
 * 
 * int ret = mul(C, A, B, 100, 50, 50, 80);
 * if (ret != 0) {
 *     fprintf(stderr, "GEMM failed with error %d\n", ret);
 * }
 * @endcode
 */
int mul(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    uint16_t row_a,
    uint16_t column_a,
    uint16_t row_b,
    uint16_t column_b
);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_H */