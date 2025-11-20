// SPDX-License-Identifier: MIT
/**
 * @file tran_pack.h
 * @brief Tiled matrix transpose with AVX2/SSE optimization and fused pack-transpose helpers
 * 
 * @details
 * Provides high-performance matrix transposition and fused transpose-pack operations
 * for GEMM and other BLAS-3 routines. Optimized for Intel AVX2 with cache-blocking.
 * 
 * **Key Features:**
 * - Cache-blocked 32×32 tiling with 8×8 AVX2 micro-kernels
 * - Optional non-temporal stores for large matrices (reduces cache pollution)
 * - Specialized fused pack-transpose for GEMM panel layout
 * - Single-threaded by design (parallelize at caller level)
 * 
 * **Performance Characteristics:**
 * - Bandwidth: ~15-25 GB/s on modern CPUs (Intel 14th gen, AMD Zen 4+)
 * - Best for: n ≥ 64 (smaller matrices use scalar paths efficiently)
 * - NT stores: Recommended for n ≥ 256 when output not immediately reused
 * 
 * **Build Configuration:**
 * - `TRAN_TILE`: Macro-tile size (default 32, tuned for 48KB L1 + 2MB L2)
 * - `TRAN_USE_NT_STORES`: Enable non-temporal stores (0/1, default 1)
 * 
 * **ISA Requirements:**
 * - AVX2 for 8×8 transpose kernels
 * - SSE for 8×4 tail handling
 * - Scalar fallback always available
 * 
 * @author TUGBARS
 * @date 2025
 */

#ifndef TRAN_PACK_H
#define TRAN_PACK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// BUILD CONFIGURATION
//==============================================================================

/**
 * @def TRAN_TILE
 * @brief Macro-tile size for cache-blocked transpose
 * 
 * Default: 32 (optimized for 48KB L1 cache)
 * - Larger: Better for huge matrices (n > 1024), may reduce L1 hit rate
 * - Smaller: Better for tiny matrices (n < 128), increases tiling overhead
 * 
 * @note Must be multiple of 8 for optimal AVX2 kernel utilization
 */
#ifndef TRAN_TILE
#define TRAN_TILE 32
#endif

/**
 * @def TRAN_USE_NT_STORES
 * @brief Enable non-temporal stores (_mm256_stream_ps)
 * 
 * Default: 1 (enabled)
 * - Use for large matrices (n ≥ 256) when output not immediately reused
 * - Reduces cache pollution by bypassing L1/L2
 * - Requires _mm_sfence() before reading output
 * 
 * Set to 0 if:
 * - Output immediately used in subsequent operations
 * - Matrix small enough to fit in cache (n < 256)
 * - CPU lacks fast NT store support
 */
#ifndef TRAN_USE_NT_STORES
#define TRAN_USE_NT_STORES 1
#endif

//==============================================================================
// CORE TRANSPOSE API
//==============================================================================

/**
 * @brief Cache-blocked matrix transpose with AVX2/SSE optimization
 * 
 * Transposes dense row-major matrix A into row-major At.
 * Uses 32×32 macro-tiling with 8×8 AVX2 micro-kernels for optimal cache reuse.
 * 
 * @param[out] At   Transposed output matrix [C × R], row-major
 * @param[in]  A    Input matrix [R × C], row-major
 * @param[in]  R    Number of rows in A (columns in At)
 * @param[in]  C    Number of columns in A (rows in At)
 * 
 * @pre At and A must not alias (no in-place transpose)
 * @pre At must have space for C×R elements
 * @pre R, C ≤ UINT16_MAX (65535)
 * 
 * @post If TRAN_USE_NT_STORES=1, caller must not read At immediately
 *       (NT stores bypass cache, reads will stall)
 * 
 * @note Thread-safe: Reads A, writes At with no shared state
 * @note Parallelization: Split outer i0-loop across threads for large R
 * 
 * **Performance:**
 * - Small (n < 64): ~5-8 GB/s (scalar-dominated)
 * - Medium (64 ≤ n < 256): ~10-15 GB/s (AVX2 + good cache reuse)
 * - Large (n ≥ 256): ~15-25 GB/s (NT stores + full AVX2 utilization)
 * 
 * **Example:**
 * @code
 * float A[1024*512];  // 1024 rows × 512 cols
 * float At[512*1024]; // 512 rows × 1024 cols
 * 
 * tran_tiled(At, A, 1024, 512);
 * 
 * #if TRAN_USE_NT_STORES
 * _mm_mfence();  // Ensure NT stores visible before reading At
 * #endif
 * @endcode
 */
void tran_tiled(float *restrict At, const float *restrict A, 
                uint16_t R, uint16_t C);

//==============================================================================
// FUSED PACK-TRANSPOSE HELPERS (FOR GEMM INTEGRATION)
//==============================================================================

/**
 * @brief Pack transposed 8×K micro-panel from row-major matrix
 * 
 * Extracts and transposes a vertical strip of A into contiguous packed buffer.
 * Used to prepare left operand (A) for GEMM micro-kernels with mr=8.
 * 
 * **Operation:**
 * Given A[M × Ktot] (row-major), extract columns [i, i+7] and rows [k0, k0+K),
 * store as contiguous row-major Ap[8 × K]:
 * 
 *     Ap[r*K + t] = A[k0+t, i+r]  for r ∈ [0,7], t ∈ [0,K)
 * 
 * **Use Case:**
 * GEMM computes C += A×B where A is stored as packed panels:
 * - Extract 8 columns of A (transposed to rows)
 * - Store contiguously for streaming access in inner kernel
 * 
 * @param[in]  A     Input matrix [M × Ktot], row-major
 * @param[in]  M     Rows in A (unused, kept for API consistency)
 * @param[in]  Ktot  Columns in A (leading dimension)
 * @param[in]  i     Starting column index (0 ≤ i ≤ M-8)
 * @param[in]  k0    Starting row index (0 ≤ k0 ≤ Ktot-K)
 * @param[in]  K     Depth to pack (panel height)
 * @param[out] Ap    Output packed buffer [8 × K], contiguous row-major
 * 
 * @pre i+7 < M (8 columns available)
 * @pre k0+K ≤ Ktot (K rows available)
 * @pre Ap has space for 8×K elements
 * 
 * @note Not optimized with SIMD (strided column loads difficult to vectorize)
 * @note For large K, consider tiling this operation for cache efficiency
 * 
 * **Example:**
 * @code
 * float A[256*128];     // 256×128 matrix
 * float Ap[8*64];       // Pack buffer for 8×64 panel
 * 
 * // Pack columns 0-7, rows 32-95 (K=64)
 * pack_T_8xK(A, 256, 128, 0, 32, 64, Ap);
 * 
 * // Now Ap[0:63] = A[32:95, 0], Ap[64:127] = A[32:95, 1], ...
 * @endcode
 */
void pack_T_8xK(const float *restrict A, uint16_t M, uint16_t Ktot,
                uint16_t i, uint16_t k0, uint16_t K, 
                float *restrict Ap);

/**
 * @brief Pack transposed K×16 micro-panel from row-major matrix
 * 
 * Extracts and packs a horizontal strip of B into contiguous buffer.
 * Used to prepare right operand (B) for GEMM micro-kernels with nr=16.
 * 
 * **Operation:**
 * Given B[Ktot × N] (row-major), extract rows [k0, k0+K) and columns [j, j+15],
 * store as contiguous row-major Bp[K × 16]:
 * 
 *     Bp[t*16 + c] = B[k0+t, j+c]  for t ∈ [0,K), c ∈ [0,15]
 * 
 * **Use Case:**
 * GEMM computes C += A×B where B is stored as packed panels:
 * - Extract 16 columns of B (kept as rows)
 * - Store contiguously for streaming access in inner kernel
 * 
 * @param[in]  B     Input matrix [Ktot × N], row-major
 * @param[in]  Ktot  Rows in B (leading dimension for row index)
 * @param[in]  N     Columns in B
 * @param[in]  k0    Starting row index (0 ≤ k0 ≤ Ktot-K)
 * @param[in]  j     Starting column index (0 ≤ j ≤ N-16)
 * @param[in]  K     Depth to pack (panel height)
 * @param[out] Bp    Output packed buffer [K × 16], contiguous row-major
 * 
 * @pre k0+K ≤ Ktot (K rows available)
 * @pre j+15 < N (16 columns available)
 * @pre Bp has space for K×16 elements
 * 
 * @note Uses memcpy for contiguous row extraction (very fast)
 * @note Each row is 64 bytes (16 floats) - perfect for cache lines
 * 
 * **Example:**
 * @code
 * float B[128*512];     // 128×512 matrix
 * float Bp[64*16];      // Pack buffer for 64×16 panel
 * 
 * // Pack rows 32-95 (K=64), columns 0-15
 * pack_T_Kx16(B, 128, 512, 32, 0, 64, Bp);
 * 
 * // Now Bp[0:15] = B[32, 0:15], Bp[16:31] = B[33, 0:15], ...
 * @endcode
 */
void pack_T_Kx16(const float *restrict B, uint16_t Ktot, uint16_t N,
                 uint16_t k0, uint16_t j, uint16_t K, 
                 float *restrict Bp);

//==============================================================================
// TRIANGULAR TRANSPOSE (FOR CHOLESKY/LAPACK)
//==============================================================================

/**
 * @brief Transpose lower-triangular to upper-triangular (cache-blocked)
 * 
 * Given L (stored as lower triangular), compute U = L^T (stored as upper).
 * Only lower triangle (including diagonal) is read; upper triangle of output zeroed.
 * 
 * @param[in]  L  Input lower-triangular matrix [n × n], row-major
 * @param[out] U  Output upper-triangular matrix [n × n], row-major
 * @param[in]  n  Matrix dimension
 * 
 * @pre L and U must not alias
 * @pre n ≤ 8192 (practical limit for dense matrices)
 * 
 * @post U[i,j] = 0 for i > j (strict upper triangle)
 * @post U[i,j] = L[j,i] for i ≤ j (upper triangle + diagonal)
 * 
 * **Performance:**
 * - Tiles crossing diagonal use scalar code (correctness)
 * - Tiles fully below diagonal use fast 8×8 AVX2 kernel
 * - Overall: ~70-80% of full matrix transpose speed
 * 
 * **Use Case:**
 * Cholesky factorization produces L (lower), but BLAS routines may need U (upper).
 * This avoids expensive full-matrix transpose by exploiting triangular structure.
 * 
 * @note Not needed if you store both triangles or use LAPACK layout flags
 */
void transpose_lower_to_upper_blocked(const float *restrict L,
                                      float *restrict U,
                                      uint16_t n);

/**
 * @brief Transpose upper-triangular to lower-triangular (cache-blocked)
 * 
 * Given U (stored as upper triangular), compute L = U^T (stored as lower).
 * Only upper triangle (including diagonal) is read; lower triangle of output zeroed.
 * 
 * @param[in]  U  Input upper-triangular matrix [n × n], row-major
 * @param[out] L  Output lower-triangular matrix [n × n], row-major
 * @param[in]  n  Matrix dimension
 * 
 * @pre U and L must not alias
 * @pre n ≤ 8192
 * 
 * @post L[i,j] = 0 for i < j (strict lower triangle)
 * @post L[i,j] = U[j,i] for i ≥ j (lower triangle + diagonal)
 * 
 * **Performance:**
 * Same as transpose_lower_to_upper_blocked (symmetric operation).
 * 
 * **Use Case:**
 * Convert between upper/lower storage for BLAS compatibility.
 */
void transpose_upper_to_lower_blocked(const float *restrict U,
                                      float *restrict L,
                                      uint16_t n);


#ifdef __cplusplus
}
#endif

#endif /* TRAN_PACK_H */