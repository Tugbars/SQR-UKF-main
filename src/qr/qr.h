/**
 * @file qr.h
 * @brief GEMM-Accelerated Blocked QR Decomposition with Compact WY Representation
 * 
 * @section qr_overview Overview
 * 
 * This library implements a high-performance QR decomposition using:
 * - **Blocked Householder transformations** with adaptive block size selection
 * - **Compact WY representation** for Level-3 BLAS operations (90%+ of flops)
 * - **Recursive panel factorization** for 2-3× speedup on panel operations
 * - **Pre-allocated workspace** with zero malloc overhead in hot paths
 * - **AVX2/AVX-512 SIMD** for Householder vector generation and application
 * 
 * @section qr_algorithm Algorithm
 * 
 * The decomposition A = Q·R proceeds in blocks:
 * 
 * ```
 * for k = 0:ib:min(m,n)
 *     1. Factor panel A[k:m, k:k+ib] → compute Householder reflectors
 *        - Uses recursive factorization for Level-3 BLAS speedup
 *        - Stores reflectors Y[k:m, k:k+ib] and scaling factors tau[k:k+ib]
 *     
 *     2. Build compact WY factor T[k:k+ib, k:k+ib]
 *        - T is upper triangular, enables blocked operations
 *     
 *     3. Apply block reflector to trailing matrix A[k:m, k+ib:n]
 *        - Uses GEMM: Z = Y^T·C, Z = T·Z, C -= Y·Z
 *        - This is ~90% of total flops for large matrices
 *     
 *     4. Store Y and T for later Q formation
 * 
 * Form Q by applying stored reflectors in reverse order
 * ```
 * 
 * @section qr_performance Performance
 * 
 * - **Tall matrices (m >> n)**: 85-95% of OpenBLAS performance
 * - **Square matrices**: 80-90% of OpenBLAS performance
 * - **Memory usage**: ~4× matrix size for workspace
 * - **Cache efficiency**: Blocked algorithm minimizes cache misses
 * 
 * @author TUGBARS
 * @date 2025
 */

#ifndef QR_H
#define QR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Forward declare gemm_plan_t (from GEMM library)
typedef struct gemm_plan gemm_plan_t;

/**
 * @brief Pre-created GEMM execution plans for QR operations
 * 
 * These plans contain optimized micro-kernel selections, blocking parameters,
 * and memory layouts for specific GEMM operations. Pre-planning eliminates
 * runtime overhead and enables better optimization.
 * 
 * @note Plans are dimension-specific and must match the actual GEMM call dimensions
 */
typedef struct
{
    gemm_plan_t *plan_yt_c; ///< Plan for Y^T * C: [ib × m] × [m × n] → [ib × n]
    gemm_plan_t *plan_t_z;  ///< Plan for T * Z: [ib × ib] × [ib × n] → [ib × n]
    gemm_plan_t *plan_y_z;  ///< Plan for Y * Z: [m × ib] × [ib × n] → [m × n]
    uint16_t plan_m;        ///< Number of rows (must match actual GEMM calls)
    uint16_t plan_n;        ///< Number of columns (must match actual GEMM calls)
    uint16_t plan_ib;       ///< Block size (must match actual GEMM calls)
} qr_gemm_plans_t;

/**
 * @brief Pre-allocated workspace for blocked QR decomposition
 * 
 * This structure contains all buffers needed for QR decomposition, eliminating
 * malloc overhead in hot paths. The workspace is reusable for multiple QR
 * decompositions of matrices up to size [m_max × n_max].
 * 
 * @section workspace_layout Memory Layout
 * 
 * The workspace contains several categories of buffers:
 * 
 * **1. Panel Factorization Buffers** (scalar, non-aligned):
 * ```
 * tau[min(m,n)]     : Householder scaling factors τ_i
 * tmp[m_max]        : Column gather/scatter buffer
 * work[m_max]       : General-purpose working buffer
 * ```
 * 
 * **2. WY Representation Buffers** (32-byte aligned for SIMD):
 * ```
 * T[ib × ib]        : Compact WY factor (upper triangular)
 *                     - Enables blocked operations: (I - Y·T·Y^T)
 *                     - Computed via: T[i,i] = τ_i, T[0:i,i] = -τ_i·T[0:i,0:i]·(Y[:,0:i]^T·Y[:,i])
 * 
 * Y[m_max × ib]     : Householder vectors for current block (row-major, stride = ib)
 *                     - Y[:,j] = j-th Householder vector with implicit v[0]=1
 *                     - Layout: Y[i,j] at Y[i*ib + j]
 * 
 * YT[ib × m_max]    : Transposed Y (row-major)
 *                     - Pre-transposed for efficient GEMM access
 *                     - Layout: Y^T[i,j] at YT[i*m_max + j]
 * ```
 * 
 * **3. GEMM Working Buffers** (32-byte aligned):
 * ```
 * Z[ib × n_big]     : First GEMM workspace for Z = Y^T·C
 *                     - n_big = max(m_max, n_max) to handle both:
 *                       * Trailing updates: C is [m × cols_right], cols_right ≤ n_max
 *                       * Q formation: Q is [m × m], m ≤ m_max
 * 
 * Z_temp[ib × n_big]: Second GEMM workspace for Z_temp = T·Z
 *                     - Same size as Z for pipeline efficiency
 * ```
 * 
 * **4. Recursive Panel Workspace** (32-byte aligned):
 * ```
 * panel_Y_temp[m_max × ib] : Temporary storage for Y_left/Y_right during recursion
 *                            - Eliminates malloc overhead in hot path
 *                            - Sliced into [m × ib1] and [m-ib1 × ib2] regions
 * 
 * panel_T_temp[ib × ib]    : T matrix workspace for recursive calls
 * panel_Z_temp[ib × ib]    : Z workspace for recursive calls
 * ```
 * 
 * **5. Matrix Packing Buffer** (32-byte aligned):
 * ```
 * Cpack[m_max × n_max] : Aligned copy of input matrix
 *                        - Used for in-place operation in qr_ws_blocked()
 *                        - Ensures alignment for SIMD operations
 * ```
 * 
 * **6. Column Pivoting Buffers** (for future RRQR support):
 * ```
 * vn1[n_max]  : Column norms (first pass)
 * vn2[n_max]  : Column norms (verification pass)
 * ```
 * 
 * **7. Reflector Storage** (32-byte aligned, optional):
 * ```
 * Y_stored[num_blocks][m_max][ib] : All Householder vectors (packed format)
 *     - Block k stores Y_k[rows_below_k × block_size_k]
 *     - Layout: Y_stored[k*Y_block_stride + i*block_size + j]
 *     - Required for Q formation
 * 
 * T_stored[num_blocks][ib][ib]    : All WY factors
 *     - Block k stores T_k[ib × ib]
 *     - Layout: T_stored[k*T_block_stride + i*ib + j]
 *     - Required for Q formation
 * ```
 * 
 * @section workspace_usage Usage Pattern
 * 
 * ```c
 * // 1. Allocate workspace (once)
 * qr_workspace *ws = qr_workspace_alloc(m_max, n_max, 0);  // ib=0 for auto-select
 * 
 * // 2. Reuse for multiple decompositions
 * for (int i = 0; i < num_matrices; i++) {
 *     float A[m * n], Q[m * m], R[m * n];
 *     // ... initialize A ...
 *     
 *     int ret = qr_ws_blocked(ws, A, Q, R, m, n, false);
 *     if (ret != 0) {  }
 *     
 *     // ... use Q and R ...
 * }
 * 
 * // 3. Free workspace
 * qr_workspace_free(ws);
 * 
 * 
 * @section workspace_memory Memory Requirements
 * 
 * Total memory: ~4× input matrix size
 * 
 * For a 1024×512 matrix with ib=64:
 * - Panel buffers: ~0.5 MB
 * - WY buffers: ~0.8 MB
 * - GEMM buffers: ~0.5 MB
 * - Packing buffer: ~2 MB
 * - Reflector storage: ~1 MB
 * - **Total: ~4.8 MB** (vs ~2 MB for input matrix)
 * 
 * @note All SIMD buffers are 32-byte aligned for optimal AVX2/AVX-512 performance
 * @note Workspace can be reused for any m ≤ m_max, n ≤ n_max
 * @note Pre-allocation eliminates ~50-100 malloc calls per decomposition
 */
typedef struct
{
    //==========================================================================
    // DIMENSIONS AND BLOCK SIZE
    //==========================================================================
    
    uint16_t m_max;  ///< Maximum number of rows (workspace capacity)
    uint16_t n_max;  ///< Maximum number of columns (workspace capacity)
    uint16_t ib;     ///< Block size (typically 16-64, auto-selected if 0)
                     ///< - Larger ib: Better GEMM efficiency, more memory
                     ///< - Smaller ib: Less memory, more overhead
                     ///< - Optimal: 32-64 for most matrices

    //==========================================================================
    // PANEL FACTORIZATION BUFFERS
    //==========================================================================
    
    float *tau;      ///< Householder scaling factors [min(m,n)]
                     ///< - tau[k] = (β_k - α_k) / β_k where β_k = -sign(α_k)·||x_k||
                     ///< - Used in H_k = I - tau[k]·v_k·v_k^T
                     ///< - tau[k] = 0 indicates no reflection needed
    
    float *tmp;      ///< Column gather/scatter buffer [m_max]
                     ///< - Used for extracting columns from strided matrices
                     ///< - Avoids repeated strided memory accesses
    
    float *work;     ///< General-purpose working buffer [m_max]
                     ///< - Used for temporary computations
                     ///< - Reserved for future optimizations

    //==========================================================================
    // WY REPRESENTATION BUFFERS (32-byte aligned)
    //==========================================================================
    
    float *T;        ///< Compact WY factor [ib × ib], upper triangular, row-major
                     ///< - Represents (I - Y·T·Y^T) = H_1·H_2·...·H_ib
                     ///< - Enables blocked operations using Level-3 BLAS
                     ///< - T[i,i] = tau[i], T[j,i] = -tau[i]·(T[j,j:i-1]·w[j:i-1])
                     ///< - ALIGNMENT: 32 bytes for AVX2/AVX-512
    
    float *Y;        ///< Householder vectors [m_max × ib], row-major, stride = ib
                     ///< - Y[i,j] = i-th component of j-th Householder vector
                     ///< - First component implicitly 1: v_j = [1, Y[1:m,j]^T]
                     ///< - Layout: Y[i,j] stored at Y[i*ib + j]
                     ///< - ALIGNMENT: 32 bytes for SIMD gather/scatter
    
    float *YT;       ///< Transposed Y [ib × m_max], row-major
                     ///< - YT[j,i] = Y[i,j]
                     ///< - Pre-transposed for efficient GEMM: Z = Y^T·C
                     ///< - Layout: YT[j,i] stored at YT[j*m_max + i]
                     ///< - ALIGNMENT: 32 bytes for optimal memory bandwidth

    //==========================================================================
    // GEMM WORKING BUFFERS (32-byte aligned)
    //==========================================================================
    
    float *Z;        ///< First GEMM workspace [ib × n_big], row-major
                     ///< - Used for Z = Y^T·C in block reflector application
                     ///< - n_big = max(m_max, n_max) to handle:
                     ///<   * Trailing updates: n_big ≥ n_max
                     ///<   * Q formation: n_big ≥ m_max (Q is m×m)
                     ///< - ALIGNMENT: 32 bytes for GEMM micro-kernels
    
    float *Z_temp;   ///< Second GEMM workspace [ib × n_big], row-major
                     ///< - Used for Z_temp = T·Z in block reflector application
                     ///< - Same size as Z for efficient pipelining
                     ///< - ALIGNMENT: 32 bytes for GEMM micro-kernels

    //==========================================================================
    // RECURSIVE PANEL WORKSPACE (32-byte aligned)
    //==========================================================================
    
    float *panel_Y_temp; ///< Temporary Y buffer [m_max × ib] for recursion
                         ///< - Sliced into Y_left[m × ib1] and Y_right[m-ib1 × ib2]
                         ///< - Eliminates malloc overhead (~100-200 calls per QR)
                         ///< - Reused across all recursive levels
                         ///< - ALIGNMENT: 32 bytes for SIMD operations
    
    float *panel_T_temp; ///< T workspace [ib × ib] for recursive calls
                         ///< - Used for building T matrices during recursion
                         ///< - Reused across all recursive levels
                         ///< - ALIGNMENT: 32 bytes
    
    float *panel_Z_temp; ///< Z workspace [ib × ib] for recursive calls
                         ///< - Used for GEMM operations during recursion
                         ///< - Reused across all recursive levels
                         ///< - ALIGNMENT: 32 bytes

    //==========================================================================
    // MATRIX PACKING BUFFER (32-byte aligned)
    //==========================================================================
    
    float *Cpack;    ///< Aligned matrix copy [m_max × n_max], row-major
                     ///< - Used in qr_ws_blocked() for in-place operation
                     ///< - Ensures 32-byte alignment for SIMD operations
                     ///< - Avoids modifying caller's input matrix
                     ///< - ALIGNMENT: 32 bytes

    //==========================================================================
    // COLUMN PIVOTING BUFFERS (future RRQR support)
    //==========================================================================
    
    float *vn1;      ///< Column norms, first pass [n_max]
                     ///< - vn1[j] = ||A[:,j]||_2 after k reflections
                     ///< - Updated incrementally: vn1[j]^2 -= |R[k,j]|^2
                     ///< - Used for rank-revealing QR (future feature)
    
    float *vn2;      ///< Column norms, verification pass [n_max]
                     ///< - Recomputed norms for numerical stability
                     ///< - Used when incremental update becomes inaccurate
                     ///< - Threshold: when vn1[j] < 0.8·vn2[j]

    //==========================================================================
    // PRE-CREATED GEMM EXECUTION PLANS
    //==========================================================================
    
    qr_gemm_plans_t *trailing_plans;    ///< Plans for trailing matrix updates
                                         ///< - Dimensions: [m_max × (n_max - ib)]
                                         ///< - Optimized for first panel's geometry
                                         ///< - Amortizes planning cost across decomposition
                                         ///< - NULL if n_max ≤ ib (no trailing matrix)
    
    qr_gemm_plans_t *q_formation_plans; ///< Plans for Q formation
                                         ///< - Dimensions: [m_max × m_max]
                                         ///< - Optimized for square matrix geometry
                                         ///< - Used when only_R = false
                                         ///< - NULL if m_max < ib

    //==========================================================================
    // REFLECTOR STORAGE (32-byte aligned, optional)
    //==========================================================================
    
    float *Y_stored; ///< All Householder vectors [num_blocks][m_max][ib], packed
                     ///< - Block k stores Y_k[rows_below_k × block_size_k]
                     ///< - rows_below_k = m - k·ib
                     ///< - block_size_k = min(ib, min(m,n) - k·ib)
                     ///< - Access: Y_stored[k*Y_block_stride + i*block_size + j]
                     ///< - Required for Q formation
                     ///< - NULL if store_reflectors = false
                     ///< - ALIGNMENT: 32 bytes
    
    float *T_stored; ///< All WY factors [num_blocks][ib][ib], row-major
                     ///< - Block k stores T_k[ib × ib]
                     ///< - Access: T_stored[k*T_block_stride + i*ib + j]
                     ///< - Required for Q formation
                     ///< - NULL if store_reflectors = false
                     ///< - ALIGNMENT: 32 bytes
    
    uint16_t num_blocks; ///< Number of blocks = ⌈min(m,n) / ib⌉
                         ///< - Total reflector storage panels
                         ///< - Example: 256×128 with ib=32 → num_blocks = 4
    
    size_t Y_block_stride; ///< Stride between Y blocks (in floats)
                           ///< - Y_block_stride = m_max × ib
                           ///< - Worst-case allocation (first block)
                           ///< - Block k offset: k * Y_block_stride
    
    size_t T_block_stride; ///< Stride between T blocks (in floats)
                           ///< - T_block_stride = ib × ib
                           ///< - Constant across all blocks
                           ///< - Block k offset: k * T_block_stride

    //==========================================================================
    // STATISTICS
    //==========================================================================
    
    size_t total_bytes; ///< Total workspace memory (bytes)
                        ///< - Includes all buffers, padding, and alignment
                        ///< - Typically 3-5× input matrix size
                        ///< - Use qr_workspace_bytes(ws) to query
} qr_workspace;

//==============================================================================
// WORKSPACE MANAGEMENT
//==============================================================================

/**
 * @brief Create QR workspace with default settings
 * 
 * Allocates a reusable workspace for QR decompositions of matrices up to
 * [m_max × n_max]. The workspace stores reflectors for Q formation.
 * 
 * @param m_max Maximum number of rows
 * @param n_max Maximum number of columns
 * @param ib Block size (0 = auto-select based on GEMM tuning)
 *           - Typical range: 16-64
 *           - ib=0 selects optimal size based on m_max, n_max
 *           - Larger ib: Better GEMM performance, more memory
 *           - Smaller ib: Less memory, more Householder overhead
 * 
 * @return Allocated workspace, or NULL on failure
 * @retval NULL if m_max = 0 or n_max = 0
 * @retval NULL if any allocation fails
 * 
 * @note All GEMM buffers are 32-byte aligned for AVX2/AVX-512
 * @note Reflectors are stored (store_reflectors = true)
 * @note Must be freed with qr_workspace_free()
 * 
 * @see qr_workspace_alloc_ex() for advanced control
 * @see qr_workspace_free()
 * @see qr_workspace_bytes()
 * 
 * @par Example:
 * ```c
 * // Create workspace for 1024×512 matrices with auto block size
 * qr_workspace *ws = qr_workspace_alloc(1024, 512, 0);
 * if (!ws) {
 *     fprintf(stderr, "Failed to allocate workspace\n");
 *     return -1;
 * }
 * 
 * printf("Workspace: %.2f MB, block size: %d\n",
 *        qr_workspace_bytes(ws) / (1024.0 * 1024.0), ws->ib);
 * 
 * // Use workspace...
 * qr_ws_blocked(ws, A, Q, R, m, n, false);
 * 
 * qr_workspace_free(ws);
 * ```
 */
qr_workspace *qr_workspace_alloc(uint16_t m_max, uint16_t n_max, uint16_t ib);

/**
 * @brief Create QR workspace with reflector storage control
 * 
 * Extended version that allows disabling reflector storage to save memory
 * when Q formation is not needed (only_R = true).
 * 
 * @param m_max Maximum number of rows
 * @param n_max Maximum number of columns
 * @param ib Block size (0 = auto-select)
 * @param store_reflectors If true, allocate Y_stored and T_stored
 *                         - Required for Q formation
 *                         - Can be false if only computing R
 *                         - Saves ~30-40% memory when false
 * 
 * @return Allocated workspace, or NULL on failure
 * @retval NULL if m_max = 0 or n_max = 0
 * @retval NULL if any allocation fails
 * 
 * @warning If store_reflectors = false, Q formation will fail
 * @note Use qr_workspace_alloc() for simpler interface
 * 
 * @see qr_workspace_alloc()
 * 
 * @par Example (R-only mode):
 * ```c
 * // Allocate workspace without reflector storage (saves memory)
 * qr_workspace *ws = qr_workspace_alloc_ex(1024, 512, 32, false);
 * 
 * // Can only compute R (Q formation not possible)
 * qr_ws_blocked(ws, A, NULL, R, m, n, true);  // only_R = true
 * 
 * qr_workspace_free(ws);
 * ```
 */
qr_workspace *qr_workspace_alloc_ex(uint16_t m_max, uint16_t n_max,
                                    uint16_t ib, bool store_reflectors);

/**
 * @brief Free QR workspace and all associated buffers
 * 
 * Deallocates all memory associated with the workspace, including:
 * - All SIMD-aligned buffers (Y, T, Z, Cpack, etc.)
 * - Reflector storage (Y_stored, T_stored)
 * - GEMM execution plans
 * - Scalar buffers (tau, tmp, work, etc.)
 * 
 * @param ws Workspace to free (can be NULL)
 * 
 * @note Safe to call with NULL pointer (no-op)
 * @note After calling, all pointers in ws are invalid
 * 
 * @see qr_workspace_alloc()
 * 
 * @par Example:
 * ```c
 * qr_workspace *ws = qr_workspace_alloc(m, n, 0);
 * // ... use workspace ...
 * qr_workspace_free(ws);
 * ws = NULL;  // Good practice
 * ```
 */
void qr_workspace_free(qr_workspace *ws);

/**
 * @brief Get total workspace memory footprint in bytes
 * 
 * Returns the total memory allocated by the workspace, including:
 * - All computation buffers
 * - Reflector storage (if enabled)
 * - Padding and alignment overhead
 * 
 * @param ws Workspace to query
 * @return Total bytes allocated, or 0 if ws is NULL
 * 
 * @note Includes alignment padding (typically +10-20% overhead)
 * @note Does not include sizeof(qr_workspace) structure itself
 * 
 * @par Example:
 * ```c
 * qr_workspace *ws = qr_workspace_alloc(1024, 512, 64);
 * 
 * printf("Workspace memory: %.2f MB\n",
 *        qr_workspace_bytes(ws) / (1024.0 * 1024.0));
 * 
 * printf("Input matrix: %.2f MB\n",
 *        (1024 * 512 * sizeof(float)) / (1024.0 * 1024.0));
 * 
 * // Output:
 * // Workspace memory: 8.50 MB
 * // Input matrix: 2.00 MB
 * // Ratio: ~4.25× (typical for QR)
 * ```
 */
size_t qr_workspace_bytes(const qr_workspace *ws);

//==============================================================================
// QR DECOMPOSITION FUNCTIONS
//==============================================================================

/**
 * @brief Blocked QR decomposition using pre-allocated workspace
 * 
 * Computes A = Q·R where:
 * - Q is orthogonal [m × m]: Q^T·Q = I
 * - R is upper triangular [m × n]
 * 
 * Uses blocked Householder with compact WY representation for optimal
 * performance. The input matrix A is not modified.
 * 
 * @param ws Workspace (must satisfy m ≤ ws->m_max, n ≤ ws->n_max)
 * @param A Input matrix [m × n], row-major (not modified)
 * @param Q Output orthogonal matrix [m × m], row-major (can be NULL if only_R = true)
 * @param R Output upper triangular [m × n], row-major (required)
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R If true, skip Q formation (R only)
 *               - Faster if Q not needed (~30% speedup)
 *               - Q can be NULL
 * 
 * @return 0 on success, negative error code on failure
 * @retval 0 Success
 * @retval -EINVAL Invalid parameters (NULL pointers, m > m_max, etc.)
 * @retval -ENOMEM Memory allocation failed (shouldn't occur with valid workspace)
 * 
 * @pre ws != NULL && A != NULL && R != NULL
 * @pre m ≤ ws->m_max && n ≤ ws->n_max
 * @pre Q != NULL if only_R = false
 * @pre ws->Y_stored != NULL if only_R = false (workspace must store reflectors)
 * 
 * @post R is upper triangular: R[i,j] = 0 for i > j
 * @post Q is orthogonal: ||Q^T·Q - I||_F < ε·√m (if only_R = false)
 * @post ||A - Q·R||_F / ||A||_F < ε·√min(m,n) (reconstruction error)
 * 
 * @note Thread-safe if each thread uses separate workspace
 * @note A is copied to ws->Cpack, so A is not modified
 * @note For in-place operation, use qr_ws_blocked_inplace()
 * 
 * @see qr_ws_blocked_inplace() for in-place version
 * @see qr_blocked() for auto-allocated workspace version
 * 
 * @par Performance:
 * - **Tall matrices (m >> n)**: 85-95% of OpenBLAS
 * - **Square matrices**: 80-90% of OpenBLAS
 * - **Time complexity**: O(2mn² - ⅔n³)
 * - **Space complexity**: O(m·n) (workspace pre-allocated)
 * 
 * @par Example:
 * ```c
 * const uint16_t m = 1024, n = 512;
 * float A[m * n], Q[m * m], R[m * n];
 * 
 * // Initialize A...
 * for (int i = 0; i < m * n; i++)
 *     A[i] = random_float();
 * 
 * // Allocate workspace
 * qr_workspace *ws = qr_workspace_alloc(m, n, 0);
 * 
 * // Compute full QR decomposition
 * int ret = qr_ws_blocked(ws, A, Q, R, m, n, false);
 * if (ret != 0) {
 *     fprintf(stderr, "QR failed: %d\n", ret);
 *     return ret;
 * }
 * 
 * // Verify orthogonality: Q^T·Q = I
 * float QTQ[m * m];
 * gemm_auto(QTQ, Q, Q, m, m, m, 1.0f, 0.0f);  // Q^T·Q
 * double ortho_err = frobenius_norm_from_identity(QTQ, m);
 * printf("Orthogonality error: %.2e\n", ortho_err);
 * 
 * // Verify reconstruction: A = Q·R
 * float QR[m * n];
 * gemm_auto(QR, Q, R, m, m, n, 1.0f, 0.0f);
 * double recon_err = relative_error(A, QR, m, n);
 * printf("Reconstruction error: %.2e\n", recon_err);
 * 
 * qr_workspace_free(ws);
 * ```
 */
int qr_ws_blocked(qr_workspace *ws, const float *A, float *Q, float *R,
                  uint16_t m, uint16_t n, bool only_R);

/**
 * @brief In-place blocked QR decomposition
 * 
 * Like qr_ws_blocked(), but modifies A in-place instead of copying.
 * Requires A to be 32-byte aligned.
 * 
 * @param ws Workspace (must satisfy m ≤ ws->m_max, n ≤ ws->n_max)
 * @param A Input/output matrix [m × n], row-major, 32-byte aligned
 *          - INPUT: Original matrix
 *          - OUTPUT: Overwritten with Householder vectors in lower triangle
 * @param Q Output orthogonal matrix [m × m], row-major (can be NULL if only_R = true)
 * @param R Output upper triangular [m × n], row-major (required)
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R If true, skip Q formation
 * 
 * @return 0 on success, negative error code on failure
 * @retval 0 Success
 * @retval -EINVAL Invalid parameters
 * 
 * @pre A must be 32-byte aligned (use gemm_aligned_alloc())
 * @pre ws != NULL && A != NULL && R != NULL
 * @pre m ≤ ws->m_max && n ≤ ws->n_max
 * 
 * @warning A is modified! Upper triangle becomes R, lower stores reflectors
 * @warning Alignment violation causes undefined behavior (likely crash)
 * 
 * @note ~10-15% faster than qr_ws_blocked() (avoids copy)
 * @note Use qr_ws_blocked() if A alignment uncertain
 * 
 * @see qr_ws_blocked() for non-destructive version
 * 
 * @par Example:
 * ```c
 * // Allocate aligned matrix
 * float *A = gemm_aligned_alloc(32, m * n * sizeof(float));
 * float *Q = gemm_aligned_alloc(32, m * m * sizeof(float));
 * float *R = gemm_aligned_alloc(32, m * n * sizeof(float));
 * 
 * // Initialize A...
 * for (int i = 0; i < m * n; i++)
 *     A[i] = random_float();
 * 
 * qr_workspace *ws = qr_workspace_alloc(m, n, 0);
 * 
 * // A will be overwritten!
 * int ret = qr_ws_blocked_inplace(ws, A, Q, R, m, n, false);
 * 
 * // A now contains Householder vectors (destroyed)
 * 
 * gemm_aligned_free(A);
 * gemm_aligned_free(Q);
 * gemm_aligned_free(R);
 * qr_workspace_free(ws);
 * ```
 */
int qr_ws_blocked_inplace(qr_workspace *ws, float *A, float *Q, float *R,
                          uint16_t m, uint16_t n, bool only_R);

/**
 * @brief Simple blocked QR with automatic workspace allocation
 * 
 * Convenience wrapper that allocates workspace automatically.
 * Suitable for one-time decompositions. For repeated decompositions,
 * use qr_ws_blocked() with pre-allocated workspace for better performance.
 * 
 * @param A Input matrix [m × n], row-major (not modified)
 * @param Q Output orthogonal matrix [m × m], row-major (can be NULL if only_R = true)
 * @param R Output upper triangular [m × n], row-major (required)
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R If true, skip Q formation
 * 
 * @return 0 on success, negative error code on failure
 * @retval 0 Success
 * @retval -EINVAL Invalid parameters
 * @retval -ENOMEM Workspace allocation failed
 * 
 * @pre A != NULL && R != NULL
 * @pre Q != NULL if only_R = false
 * 
 * @note Slower than qr_ws_blocked() due to allocation overhead
 * @note Use qr_ws_blocked() if calling repeatedly
 * @note Block size auto-selected based on matrix dimensions
 * 
 * @see qr_ws_blocked() for workspace-based version
 * 
 * @par Example:
 * ```c
 * float A[m * n], Q[m * m], R[m * n];
 * 
 * // Initialize A...
 * 
 * // One-time decomposition (auto workspace)
 * int ret = qr_blocked(A, Q, R, m, n, false);
 * if (ret != 0) {
 *     fprintf(stderr, "QR failed: %d\n", ret);
 *     return ret;
 * }
 * 
 * // Use Q and R...
 * ```
 * 
 * @par Performance Comparison:
 * ```c
 * // Method 1: Simple (allocates every call)
 * for (int i = 0; i < 100; i++) {
 *     qr_blocked(A, Q, R, m, n, false);  // ~10% slower due to malloc
 * }
 * 
 * // Method 2: Workspace reuse (faster)
 * qr_workspace *ws = qr_workspace_alloc(m, n, 0);
 * for (int i = 0; i < 100; i++) {
 *     qr_ws_blocked(ws, A, Q, R, m, n, false);  // No malloc overhead
 * }
 * qr_workspace_free(ws);
 * ```
 */
int qr_blocked(const float *A, float *Q, float *R,
               uint16_t m, uint16_t n, bool only_R);

#endif // QR_H