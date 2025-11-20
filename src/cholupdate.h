/**
 * @file cholupdate.h
 * @brief Rank-k Cholesky update/downdate with GEMM-accelerated QR
 *
 * @details
 * High-performance rank-k Cholesky factor updates:
 * \f[
 *    L L^T \leftarrow L L^T \pm X X^T
 * \f]
 *
 * **Features:**
 * - Zero-allocation hot path (workspace-based API)
 * - Automatic algorithm selection (tiled vs QR)
 * - SIMD-optimized rank-1 updates (AVX2)
 * - BLAS-3 blocked QR for large k (~20 GFLOPS on Intel 14900K)
 * - Support for both upper and lower triangular storage
 * - Numerically stable updates and downdates
 *
 * **Algorithm Selection:**
 * - k=1: Specialized rank-1 kernel (Givens rotations)
 * - k∈[2,7]: Tiled rank-1 (good cache, already SIMD)
 * - k≥8: Blocked QR (BLAS-3 efficiency dominates)
 *
 * **API Design:**
 * 1. Workspace-based (recommended for hot paths):
 *    - Allocate once: cholupdate_workspace_alloc()
 *    - Reuse many times: cholupdatek_auto_ws()
 *    - Free when done: cholupdate_workspace_free()
 *
 * 2. Legacy API (backward compatible):
 *    - Convenience wrappers that allocate internally
 *    - Suitable for cold paths or one-off updates
 *    - Not recommended for performance-critical loops
 *
 * @example
 * @code
 * // Workspace-based API (RECOMMENDED)
 * cholupdate_workspace *ws = cholupdate_workspace_alloc(64, 16);
 * 
 * for (int i = 0; i < 1000; i++) {
 *     // Zero allocations per iteration
 *     cholupdatek_auto_ws(ws, L, X, n, k, false, +1);
 * }
 * 
 * cholupdate_workspace_free(ws);
 *
 * // Legacy API (simpler but slower)
 * cholupdatek_blas3(L, X, n, k, false, +1);  // Allocates internally
 * @endcode
 *
 * @author TUGBARS
 * @date 2025
 */

#ifndef CHOLUPDATE_H
#define CHOLUPDATE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// WORKSPACE STRUCTURE (OPAQUE)
//==============================================================================

/**
 * @brief Opaque workspace for Cholesky updates
 *
 * @details Pre-allocated buffers for zero-malloc hot path:
 * - xbuf: Working buffer for rank-1 updates (n_max floats)
 * - M: Augmented matrix [U | X] (n_max × (n_max+k_max) floats)
 * - R: QR result storage (n_max × (n_max+k_max) floats)
 * - Utmp: Temporary for transpose operations (n_max × n_max floats)
 * - qr_ws: Embedded QR workspace (optimized for rank-k updates)
 *
 * @note All buffers are 32-byte aligned for optimal SIMD performance
 * @note Thread-safe: each thread should have its own workspace
 * @note Reusable: single allocation serves unlimited updates
 */
typedef struct cholupdate_workspace_s cholupdate_workspace;

//==============================================================================
// WORKSPACE MANAGEMENT API
//==============================================================================

/**
 * @brief Allocate Cholesky update workspace
 *
 * @param n_max Maximum matrix dimension (must be > 0)
 * @param k_max Maximum rank of updates (0 for rank-1 only)
 *
 * @return Workspace pointer on success, NULL on allocation failure
 *
 * @details
 * **Memory allocation:**
 * - xbuf: n_max floats (always allocated)
 * - M, R, Utmp: Only if k_max > 0 (for QR path)
 * - QR workspace: Auto-sized for [n_max × (n_max + k_max)] matrix
 *
 * **QR workspace configuration:**
 * - Block size: Auto-selected (ib=0) via adaptive strategy
 * - Reflector storage: Disabled (store_reflectors=false)
 * - Memory savings: ~O(n²k) floats vs full storage
 * - Performance: Full BLAS-3 efficiency maintained
 *
 * **Sizing guidelines:**
 * - Set n_max to maximum matrix dimension you'll encounter
 * - Set k_max to maximum rank you'll need (0 if only rank-1)
 * - Workspace can handle any n ≤ n_max, k ≤ k_max
 * - Over-sizing is safe but wastes memory
 *
 * @example
 * @code
 * // For 64×64 matrices with up to rank-16 updates
 * cholupdate_workspace *ws = cholupdate_workspace_alloc(64, 16);
 * if (!ws) {
 *     // Handle allocation failure
 *     return -1;
 * }
 * 
 * // Now can handle any update with n ≤ 64, k ≤ 16
 * cholupdatek_auto_ws(ws, L, X, 48, 8, false, +1);  // OK
 * cholupdatek_auto_ws(ws, L, X, 64, 16, false, +1); // OK
 * // cholupdatek_auto_ws(ws, L, X, 64, 20, false, +1); // ERROR: k > k_max
 * 
 * cholupdate_workspace_free(ws);
 * @endcode
 *
 * @note COLD PATH - allocate once, reuse many times
 * @note Returns NULL on failure (check errno for details)
 * @note Safe to call from any thread (allocates independent workspace)
 */
cholupdate_workspace *cholupdate_workspace_alloc(uint16_t n_max, uint16_t k_max);

/**
 * @brief Free Cholesky update workspace
 *
 * @param ws Workspace to free (NULL-safe)
 *
 * @details
 * Frees all internal buffers:
 * - xbuf, M, R, Utmp (if allocated)
 * - Embedded QR workspace
 * - Workspace structure itself
 *
 * @note NULL-safe: calling with ws=NULL is a no-op
 * @note After free, pointer is invalid (don't reuse)
 * @note Safe to call from any thread
 */
void cholupdate_workspace_free(cholupdate_workspace *ws);

/**
 * @brief Query workspace memory usage
 *
 * @param ws Workspace to query
 * @return Total bytes allocated (0 if ws=NULL)
 *
 * @details
 * Returns total memory footprint including:
 * - All float buffers (xbuf, M, R, Utmp)
 * - Embedded QR workspace
 * - Workspace structure overhead
 *
 * @example
 * @code
 * cholupdate_workspace *ws = cholupdate_workspace_alloc(256, 32);
 * size_t bytes = cholupdate_workspace_bytes(ws);
 * printf("Workspace size: %.2f MB\n", bytes / (1024.0 * 1024.0));
 * // Typical: ~8-16 MB for n=256, k=32
 * @endcode
 *
 * @note Useful for memory budgeting and diagnostics
 * @note Returns exact allocation size (includes alignment padding)
 */
size_t cholupdate_workspace_bytes(const cholupdate_workspace *ws);

//==============================================================================
// WORKSPACE-BASED API (RECOMMENDED FOR HOT PATHS)
//==============================================================================

/**
 * @brief Smart rank-k Cholesky update with automatic algorithm selection
 *
 * @param ws Pre-allocated workspace (must be sized for n, k)
 * @param L_or_U In-place Cholesky factor (n×n, row-major)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension (must be ≤ ws->n_max)
 * @param k Rank of update (must be ≤ ws->k_max)
 * @param is_upper True for upper-triangular (U^T U), false for lower (L L^T)
 * @param add +1 for update (add X X^T), -1 for downdate (subtract X X^T)
 *
 * @return 0 on success, negative errno on failure:
 *         -EINVAL: Invalid arguments (NULL pointers, n=0, invalid add)
 *         -EOVERFLOW: Dimensions exceed workspace capacity
 *         -EDOM: Downdate would make matrix indefinite
 *         -ENOMEM: Internal allocation failure (QR workspace)
 *
 * @details
 * **Algorithm selection heuristics:**
 * - k=1: Always use specialized rank-1 kernel (optimal)
 * - k<8 or n<32: Use tiled rank-1 (good cache, low overhead)
 * - k≥8 and n≥32: Use blocked QR (BLAS-3 efficiency wins)
 * - Decision considers QR's adaptive block size selection
 *
 * **Performance characteristics:**
 * - Rank-1 path: ~2-4 GFLOPS (SIMD-optimized, cache-friendly)
 * - QR path: ~15-25 GFLOPS (GEMM-dominated, BLAS-3)
 * - Crossover: k≈8 where QR overhead is compensated by GEMM speed
 *
 * **Memory:**
 * - ZERO allocations in hot path (all buffers from workspace)
 * - Workspace must be sized for max(n, k) you'll use
 * - Thread-safe if each thread has own workspace
 *
 * **Numerical stability:**
 * - Updates use hyperbolic rotations (Givens-like)
 * - QR uses Householder reflections (LAPACK-quality)
 * - Both methods maintain positive definiteness rigorously
 * - Downdates check for indefiniteness and fail safely (-EDOM)
 *
 * @example
 * @code
 * // Allocate workspace once
 * cholupdate_workspace *ws = cholupdate_workspace_alloc(64, 16);
 * 
 * // Rank-1 update: adds x*x^T to L*L^T
 * float x[64];
 * cholupdatek_auto_ws(ws, L, x, 64, 1, false, +1);
 * 
 * // Rank-8 update: adds X*X^T to L*L^T (will use QR path)
 * float X[64*8];
 * cholupdatek_auto_ws(ws, L, X, 64, 8, false, +1);
 * 
 * // Rank-4 downdate: subtracts X*X^T from L*L^T
 * int rc = cholupdatek_auto_ws(ws, L, X, 64, 4, false, -1);
 * if (rc == -EDOM) {
 *     printf("Downdate failed: matrix would become indefinite\n");
 * }
 * 
 * cholupdate_workspace_free(ws);
 * @endcode
 *
 * @note **RECOMMENDED API** for performance-critical code
 * @note HOT PATH: Zero allocations, optimal algorithm selection
 * @note Reuse workspace across calls for maximum efficiency
 * @note L_or_U is modified in-place (overwritten with new factor)
 */
int cholupdatek_auto_ws(cholupdate_workspace *ws,
                        float *restrict L_or_U,
                        const float *restrict X,
                        uint16_t n, uint16_t k,
                        bool is_upper, int add);

/**
 * @brief Tiled rank-k update (explicit algorithm selection)
 *
 * @param ws Pre-allocated workspace
 * @param L In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension (must be ≤ ws->n_max)
 * @param k Rank of update (must be ≤ ws->k_max)
 * @param is_upper True for upper-triangular, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure
 *
 * @details
 * Forces use of tiled rank-1 algorithm regardless of k.
 * Useful for:
 * - Benchmarking (compare tiled vs QR)
 * - Small k where tiling is optimal
 * - Avoiding QR overhead for specific use cases
 *
 * **When to use:**
 * - k < 8: Tiled is typically optimal
 * - Small matrices (n < 32): Lower overhead than QR
 * - Cache-sensitive applications: Better locality than QR setup
 *
 * **When NOT to use:**
 * - k ≥ 8 on reasonably sized matrices: QR is faster
 * - Most applications: Use cholupdatek_auto_ws() instead
 *
 * @note Use cholupdatek_auto_ws() unless you have specific reasons
 * @note Algorithm: k sequential rank-1 updates, tiled for cache
 * @note Cost: O(n²k), dominated by SIMD rank-1 kernel
 */
int cholupdatek_ws(cholupdate_workspace *ws,
                   float *restrict L,
                   const float *restrict X,
                   uint16_t n, uint16_t k,
                   bool is_upper, int add);

//==============================================================================
// LEGACY API (BACKWARD COMPATIBLE - ALLOCATES INTERNALLY)
//==============================================================================

/**
 * @brief Legacy tiled rank-k update (allocates workspace internally)
 *
 * @param L In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension
 * @param k Rank of update
 * @param is_upper True for upper-triangular, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure (-ENOMEM if allocation fails)
 *
 * @details
 * Convenience wrapper that allocates workspace internally.
 *
 * **Performance warning:**
 * - Allocates and frees workspace on EVERY call
 * - Overhead: ~microseconds for small n, ~milliseconds for large n
 * - Use workspace-based API for loops or hot paths
 *
 * **When to use:**
 * - One-off updates (not in loops)
 * - Prototyping or testing
 * - Applications where allocation overhead is negligible
 *
 * @deprecated For performance-critical code, use cholupdatek_auto_ws()
 * @note Uses tiled algorithm (same as cholupdatek_ws())
 */
int cholupdatek(float *restrict L,
                const float *restrict X,
                uint16_t n, uint16_t k,
                bool is_upper, int add);

/**
 * @brief Legacy blocked QR update (allocates workspace internally)
 *
 * @param L_or_U In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension
 * @param k Rank of update
 * @param is_upper True for upper-triangular, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure
 *
 * @details
 * Forces use of blocked QR algorithm, allocating workspace internally.
 *
 * **Performance warning:**
 * - Allocates large workspace on every call (~O(n²) floats)
 * - Only worthwhile if k is large enough to amortize overhead
 *
 * @deprecated For performance-critical code, use cholupdatek_auto_ws()
 * @note Useful for testing/benchmarking QR path explicitly
 */
int cholupdatek_blockqr(float *restrict L_or_U,
                        const float *restrict X,
                        uint16_t n, uint16_t k,
                        bool is_upper, int add);

/**
 * @brief Legacy auto-dispatch update (allocates workspace internally)
 *
 * @param L_or_U In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension
 * @param k Rank of update
 * @param is_upper True for upper-triangular, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure
 *
 * @details
 * Automatically selects algorithm (tiled vs QR) and allocates workspace.
 *
 * **Compared to cholupdatek_auto_ws():**
 * - Same algorithm selection heuristics
 * - Same computational performance once workspace allocated
 * - Additional overhead: workspace allocation/deallocation per call
 *
 * **When to use:**
 * - One-off updates outside hot paths
 * - Prototyping before optimizing with workspace API
 * - Applications where convenience > performance
 *
 * @deprecated For performance-critical code, use cholupdatek_auto_ws()
 * @note Recommended legacy API if you must use non-workspace interface
 */
int cholupdatek_blas3(float *restrict L_or_U,
                      const float *restrict X,
                      uint16_t n, uint16_t k,
                      bool is_upper, int add);
//==============================================================================
// WORKSPACE-BASED API (RECOMMENDED FOR HOT PATHS)
//==============================================================================

/**
 * @brief Smart rank-k Cholesky update with automatic algorithm selection
 *
 * @param ws Pre-allocated workspace (must be sized for n, k)
 * @param L_or_U In-place Cholesky factor (n×n, row-major)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension (must be ≤ ws->n_max)
 * @param k Rank of update (must be ≤ ws->k_max)
 * @param is_upper True for upper-triangular (U^T U), false for lower (L L^T)
 * @param add +1 for update (add X X^T), -1 for downdate (subtract X X^T)
 *
 * @return 0 on success, negative errno on failure:
 *         -EINVAL: Invalid arguments (NULL pointers, n=0, invalid add)
 *         -EOVERFLOW: Dimensions exceed workspace capacity
 *         -EDOM: Downdate would make matrix indefinite
 *         -ENOMEM: Internal allocation failure (QR workspace)
 *
 * @details
 * **Algorithm selection heuristics:**
 * - k=1: Always use specialized rank-1 kernel (optimal)
 * - k<8 or n<32: Use tiled rank-1 (good cache, low overhead)
 * - k≥8 and n≥32: Use blocked QR (BLAS-3 efficiency wins)
 * - Decision considers QR's adaptive block size selection
 *
 * **Performance characteristics:**
 * - Rank-1 path: ~2-4 GFLOPS (SIMD-optimized, cache-friendly)
 * - QR path: ~15-25 GFLOPS (GEMM-dominated, BLAS-3)
 * - Crossover: k≈8 where QR overhead is compensated by GEMM speed
 *
 * **Memory:**
 * - ZERO allocations in hot path (all buffers from workspace)
 * - Workspace must be sized for max(n, k) you'll use
 * - Thread-safe if each thread has own workspace
 *
 * **Numerical stability:**
 * - Updates use hyperbolic rotations (Givens-like)
 * - QR uses Householder reflections (LAPACK-quality)
 * - Both methods maintain positive definiteness rigorously
 * - Downdates check for indefiniteness and fail safely (-EDOM)
 *
 * @example
 * @code
 * // Allocate workspace once
 * cholupdate_workspace *ws = cholupdate_workspace_alloc(64, 16);
 * 
 * // Rank-1 update: adds x*x^T to L*L^T
 * float x[64];
 * cholupdatek_auto_ws(ws, L, x, 64, 1, false, +1);
 * 
 * // Rank-8 update: adds X*X^T to L*L^T (will use QR path)
 * float X[64*8];
 * cholupdatek_auto_ws(ws, L, X, 64, 8, false, +1);
 * 
 * // Rank-4 downdate: subtracts X*X^T from L*L^T
 * int rc = cholupdatek_auto_ws(ws, L, X, 64, 4, false, -1);
 * if (rc == -EDOM) {
 *     printf("Downdate failed: matrix would become indefinite\n");
 * }
 * 
 * cholupdate_workspace_free(ws);
 * @endcode
 *
 * @note **RECOMMENDED API** for performance-critical code
 * @note HOT PATH: Zero allocations, optimal algorithm selection
 * @note Reuse workspace across calls for maximum efficiency
 * @note L_or_U is modified in-place (overwritten with new factor)
 */
int cholupdatek_auto_ws(cholupdate_workspace *ws,
                        float *restrict L_or_U,
                        const float *restrict X,
                        uint16_t n, uint16_t k,
                        bool is_upper, int add);

/**
 * @brief Tiled rank-k update (explicit algorithm selection)
 *
 * @param ws Pre-allocated workspace
 * @param L In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension (must be ≤ ws->n_max)
 * @param k Rank of update (must be ≤ ws->k_max)
 * @param is_upper True for upper-triangular, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure
 *
 * @details
 * Forces use of tiled rank-1 algorithm regardless of k.
 * Useful for:
 * - Benchmarking (compare tiled vs QR)
 * - Small k where tiling is optimal
 * - Avoiding QR overhead for specific use cases
 *
 * **When to use:**
 * - k < 8: Tiled is typically optimal
 * - Small matrices (n < 32): Lower overhead than QR
 * - Cache-sensitive applications: Better locality than QR setup
 *
 * **When NOT to use:**
 * - k ≥ 8 on reasonably sized matrices: QR is faster
 * - Most applications: Use cholupdatek_auto_ws() instead
 *
 * @note Use cholupdatek_auto_ws() unless you have specific reasons
 * @note Algorithm: k sequential rank-1 updates, tiled for cache
 * @note Cost: O(n²k), dominated by SIMD rank-1 kernel
 */
int cholupdatek_ws(cholupdate_workspace *ws,
                   float *restrict L,
                   const float *restrict X,
                   uint16_t n, uint16_t k,
                   bool is_upper, int add);

/**
 * @brief Blocked QR rank-k update (explicit algorithm selection)
 *
 * @param ws Pre-allocated workspace (must include QR workspace)
 * @param L_or_U In-place Cholesky factor (n×n)
 * @param X Update matrix (n×k, row-major)
 * @param n Matrix dimension (must be ≤ ws->n_max)
 * @param k Rank of update (must be > 0 and ≤ ws->k_max)
 * @param is_upper True for upper-triangular, false for lower
 * @param add +1 for update, -1 for downdate
 *
 * @return 0 on success, negative errno on failure
 *
 * @details
 * Forces use of blocked QR algorithm regardless of k.
 * Useful for:
 * - Benchmarking (compare QR vs tiled)
 * - Large k where QR is optimal (k ≥ 8)
 * - Testing QR path explicitly
 *
 * **Algorithm:**
 * 1. Build augmented matrix M = [U | ±X]
 * 2. Compute QR decomposition M = QR (GEMM-accelerated)
 * 3. Extract new Cholesky factor from R[1:n, 1:n]
 *
 * **Performance:**
 * - Cost: O(n²(n+k)) operations, GEMM-dominated
 * - Throughput: ~15-25 GFLOPS on modern CPUs
 * - Best for: k ≥ 8 and n ≥ 32
 *
 * **When to use:**
 * - Testing/benchmarking QR path
 * - k ≥ 8: QR typically faster than tiled
 * - Known to be optimal for your problem size
 *
 * **When NOT to use:**
 * - k < 8: Tiled is usually faster
 * - Small matrices (n < 32): QR overhead dominates
 * - Most applications: Use cholupdatek_auto_ws() instead
 *
 * @note Use cholupdatek_auto_ws() unless you have specific reasons
 * @note Requires workspace with k_max > 0 (QR buffers allocated)
 * @note k=1 works but is inefficient (use tiled path instead)
 */
int cholupdatek_blockqr_ws(cholupdate_workspace *ws,
                           float *restrict L_or_U,
                           const float *restrict X,
                           uint16_t n, uint16_t k,
                           bool is_upper, int add);
                           

//==============================================================================
// USAGE RECOMMENDATIONS
//==============================================================================

/**
 * @section usage_guide Usage Guide
 *
 * **For Hot Paths (Recommended):**
 * @code
 * // 1. Allocate workspace once
 * cholupdate_workspace *ws = cholupdate_workspace_alloc(n_max, k_max);
 * 
 * // 2. Reuse in loop (zero allocations per iteration)
 * for (int i = 0; i < iterations; i++) {
 *     cholupdatek_auto_ws(ws, L, X, n, k, false, +1);
 * }
 * 
 * // 3. Free when done
 * cholupdate_workspace_free(ws);
 * @endcode
 *
 * **For Cold Paths (Convenience):**
 * @code
 * // One-off update (simpler but slower)
 * cholupdatek_blas3(L, X, n, k, false, +1);
 * @endcode
 *
 * **Threading:**
 * @code
 * // Each thread needs its own workspace
 * #pragma omp parallel
 * {
 *     cholupdate_workspace *ws = cholupdate_workspace_alloc(n, k);
 *     
 *     #pragma omp for
 *     for (int i = 0; i < N; i++) {
 *         cholupdatek_auto_ws(ws, L[i], X[i], n, k, false, +1);
 *     }
 *     
 *     cholupdate_workspace_free(ws);
 * }
 * @endcode
 *
 * **Error Handling:**
 * @code
 * int rc = cholupdatek_auto_ws(ws, L, X, n, k, false, -1);
 * switch (rc) {
 *     case 0:
 *         // Success
 *         break;
 *     case -EINVAL:
 *         // Invalid arguments
 *         break;
 *     case -EOVERFLOW:
 *         // Dimensions too large for workspace
 *         break;
 *     case -EDOM:
 *         // Downdate would make matrix indefinite
 *         break;
 *     default:
 *         // Other error
 *         break;
 * }
 * @endcode
 */

#ifdef __cplusplus
}
#endif

#endif // CHOLUPDATE_H