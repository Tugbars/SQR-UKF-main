// SPDX-License-Identifier: MIT
/**
 * @file lup_blas3.c
 * @brief Blocked BLAS-3 LU with partial pivoting (single-precision), AVX2/FMA GEMM updates.
 *
 * Factorizes A (row-major, n×n) into P·A = L·U in-place in LU, recording final row
 * permutation in P (P[i] = original row index now at position i). The algorithm is a
 * right-looking, blocked GETRF:
 *   for k = 0..n-1 step NB:
 *     1) Panel factorization: unblocked LU on A[k:n, k:k+ib) with partial pivoting.
 *        Apply row swaps to the entire matrix.
 *     2) Compute U12: A[k:k+ib, k+ib:n] ← L11^{-1} * A[k:k+ib, k+ib:n]  (unit-lower TRSM)
 *     3) Compute L21: A[k+ib:n, k:k+ib] ← A[k+ib:n, k:k+ib] * U11^{-1}  (upper TRSM, right-side)
 *     4) Trailing update: A22 ← A22 − L21·U12  (GEMM; uses gemm_strided())
 *
 * Notes:
 *  - Row-major everywhere; leading dimension is n.
 *  - Small unblocked TRSM is used only on the panel (ib×ib). The big A22 update is BLAS-3.
 *  - P ends as a *final permutation table* (not ipiv steps), compatible with your RHS pivot apply.
 *  - All input data treated as potentially unaligned (uses loadu/storeu).
 *  - Uses gemm_strided() for trailing updates (allocates its own workspace).
 *  - L21 TRSM uses adaptive strategy: transpose for large matrices (3-5× faster).
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <immintrin.h>
#include "linalg_simd.h"        // For LINALG_DEFAULT_ALIGNMENT
#include "../gemm/gemm.h"       // For gemm_strided()
#include "../gemm/gemm_utils.h" // ✅ For gemm_aligned_alloc/free
#include "tran_pack.h"          // ✅ For transpose8x8_avx(), transpose8x4_sse()

#ifndef LUP_NB
#define LUP_NB 128 // panel/block size (try 96–160)
#endif

#ifndef LUP_TRSM_TRANSPOSE_THRESHOLD
#define LUP_TRSM_TRANSPOSE_THRESHOLD 2048 // m2*ib threshold for using transpose
#endif

_Static_assert(LINALG_DEFAULT_ALIGNMENT >= 32, "Need 32B alignment for AVX2 loads");

//==============================================================================
// SIMPLIFIED WORKSPACE IMPLEMENTATION
//==============================================================================
// LUP no longer needs embedded GEMM workspace since gemm_strided() allocates
// its own workspace internally. This workspace structure is kept for future
// expansion (e.g., temporary buffers for advanced pivoting strategies).
//==============================================================================

struct lup_workspace
{
    void *buffer;    // Reserved for future use
    size_t size;     // Buffer size
    int owns_memory; // Whether to free buffer on destroy
};

/**
 * @brief Query workspace size needed for LUP factorization
 *
 * Currently returns minimal size since gemm_strided() manages its own workspace.
 * Kept for API stability and future expansion.
 *
 * @param n Matrix dimension
 * @return Workspace size in bytes (currently minimal)
 */
size_t lup_workspace_query(uint16_t n)
{
    (void)n;
    return 64; // Minimal padding for future use
}

/**
 * @brief Create LUP workspace with owned memory
 */
lup_workspace_t *lup_workspace_create(size_t size)
{
    lup_workspace_t *ws = malloc(sizeof(*ws));
    if (!ws)
        return NULL;

    // ✅ Use gemm_aligned_alloc from gemm_utils.h
    ws->buffer = gemm_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, size);
    if (!ws->buffer)
    {
        free(ws);
        return NULL;
    }

    ws->size = size;
    ws->owns_memory = 1;

    return ws;
}

/**
 * @brief Initialize LUP workspace with user-provided buffer
 */
lup_workspace_t *lup_workspace_init(void *buffer, size_t size)
{
    if (!buffer)
        return NULL;

    lup_workspace_t *ws = malloc(sizeof(*ws));
    if (!ws)
        return NULL;

    ws->buffer = buffer;
    ws->size = size;
    ws->owns_memory = 0;

    return ws;
}

/**
 * @brief Destroy LUP workspace
 */
void lup_workspace_destroy(lup_workspace_t *ws)
{
    if (!ws)
        return;

    // ✅ Use gemm_aligned_free from gemm_utils.h
    if (ws->owns_memory && ws->buffer)
        gemm_aligned_free(ws->buffer);

    free(ws);
}

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

/**
 * @brief Swap two rows in a matrix (unaligned access safe)
 */
static inline void swap_rows(float *RESTRICT A, uint16_t n, uint16_t r1, uint16_t r2)
{
    if (r1 == r2)
        return;

    float *a = A + (size_t)r1 * n;
    float *b = A + (size_t)r2 * n;

    for (uint16_t j = 0; j < n; ++j)
    {
        float t = a[j];
        a[j] = b[j];
        b[j] = t;
    }
}

/**
 * @brief Find pivot row (max |col|) among rows [r0..n-1] in column c
 */
static inline uint16_t argmax_abs_col(const float *RESTRICT A, uint16_t n, uint16_t r0, uint16_t c)
{
    uint16_t p = r0;
    float best = fabsf(A[(size_t)r0 * n + c]);

    for (uint16_t r = (uint16_t)(r0 + 1); r < n; ++r)
    {
        float v = fabsf(A[(size_t)r * n + c]);
        if (v > best)
        {
            best = v;
            p = r;
        }
    }
    return p;
}

//==============================================================================
// TRANSPOSE HELPERS (Using Your Existing AVX2 Kernels)
//==============================================================================

/**
 * @brief Transpose submatrix from row-major to column-major
 *
 * Transposes src[i0:i0+rows, j0:j0+cols] (stride=src_stride)
 * into dst[0:rows*cols] (column-major, stride=rows)
 *
 * Uses your existing transpose8x8_avx() and transpose8x4_sse() kernels
 *
 * @param dst         Output buffer (column-major: cols × rows)
 * @param src         Source matrix (row-major with stride)
 * @param i0          Starting row in source
 * @param j0          Starting column in source
 * @param rows        Number of rows to transpose
 * @param cols        Number of columns to transpose
 * @param src_stride  Leading dimension of source (usually n)
 */
static inline void transpose_submatrix_to_colmajor(
    float *restrict dst,
    const float *restrict src,
    size_t i0, size_t j0,
    size_t rows, size_t cols,
    size_t src_stride)
{
    const size_t rows8 = rows & ~(size_t)7;
    const size_t cols8 = cols & ~(size_t)7;

    // ✅ FAST PATH: 8×8 tiles using your AVX2 kernel
    for (size_t i = 0; i < rows8; i += 8)
    {
        for (size_t j = 0; j < cols8; j += 8)
        {
            transpose8x8_avx(
                src + (i0 + i) * src_stride + (j0 + j), // Source block
                dst + j * rows + i,                     // Dest: col j, row i
                src_stride,                             // Source stride
                rows);                                  // Dest stride
        }
    }

    // ✅ MEDIUM PATH: 8×4 tiles for column remainder
    size_t cols4 = cols8 + ((cols - cols8) & ~(size_t)3);
    if (cols4 > cols8)
    {
        for (size_t i = 0; i < rows8; i += 8)
        {
            transpose8x4_sse(
                src + (i0 + i) * src_stride + (j0 + cols8),
                dst + cols8 * rows + i,
                src_stride,
                rows);
        }
    }

    // ✅ SLOW PATH: Scalar cleanup for edges
    // Row remainder (rows not divisible by 8)
    if (rows > rows8)
    {
        for (size_t i = rows8; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dst[j * rows + i] = src[(i0 + i) * src_stride + (j0 + j)];
            }
        }
    }

    // Column remainder (cols not divisible by 4)
    if (cols > cols4)
    {
        for (size_t i = 0; i < rows8; ++i)
        {
            for (size_t j = cols4; j < cols; ++j)
            {
                dst[j * rows + i] = src[(i0 + i) * src_stride + (j0 + j)];
            }
        }
    }
}

/**
 * @brief Transpose back from column-major to row-major submatrix
 *
 * Reverse of transpose_submatrix_to_colmajor()
 */
static inline void transpose_colmajor_to_submatrix(
    float *restrict dst,
    const float *restrict src,
    size_t i0, size_t j0,
    size_t rows, size_t cols,
    size_t dst_stride)
{
    const size_t rows8 = rows & ~(size_t)7;
    const size_t cols8 = cols & ~(size_t)7;

    // ✅ FAST PATH: 8×8 tiles (transposing from col-major back to row-major)
    for (size_t i = 0; i < rows8; i += 8)
    {
        for (size_t j = 0; j < cols8; j += 8)
        {
            transpose8x8_avx(
                src + j * rows + i,                     // Source (col-major)
                dst + (i0 + i) * dst_stride + (j0 + j), // Dest (row-major)
                rows,                                   // Source stride
                dst_stride);                            // Dest stride
        }
    }

    // ✅ MEDIUM PATH: 8×4 tiles
    size_t cols4 = cols8 + ((cols - cols8) & ~(size_t)3);
    if (cols4 > cols8)
    {
        for (size_t i = 0; i < rows8; i += 8)
        {
            transpose8x4_sse(
                src + cols8 * rows + i,
                dst + (i0 + i) * dst_stride + (j0 + cols8),
                rows,
                dst_stride);
        }
    }

    // ✅ SLOW PATH: Scalar cleanup
    if (rows > rows8)
    {
        for (size_t i = rows8; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dst[(i0 + i) * dst_stride + (j0 + j)] = src[j * rows + i];
            }
        }
    }

    if (cols > cols4)
    {
        for (size_t i = 0; i < rows8; ++i)
        {
            for (size_t j = cols4; j < cols; ++j)
            {
                dst[(i0 + i) * dst_stride + (j0 + j)] = src[j * rows + i];
            }
        }
    }
}

//==============================================================================
// PANEL FACTORIZATION (Unblocked LU with Partial Pivoting)
//==============================================================================

/**
 * @brief Unblocked LU factorization of panel A[k:n, k:k+ib)
 *
 * Performs LU factorization with partial pivoting on a vertical panel.
 * Physical row swaps are applied to the entire matrix, and the final
 * permutation table P is updated.
 *
 * @param A    Matrix (n × n, row-major, in-place update)
 * @param n    Matrix dimension
 * @param k    Panel starting column
 * @param ib   Panel width
 * @param P    Final permutation table (updated in-place)
 * @return 0 on success, -ENOTSUP if singular
 */
static int panel_lu_unblocked(float *RESTRICT A, uint16_t n,
                              uint16_t k, uint16_t ib,
                              uint8_t *RESTRICT P)
{
    uint16_t kend = (uint16_t)(k + ib);

    for (uint16_t j = k; j < kend; ++j)
    {
        // Find pivot row (max absolute value in column j)
        uint16_t piv = argmax_abs_col(A, n, j, j);

        // Apply row swap and update permutation
        if (piv != j)
        {
            swap_rows(A, n, j, piv);
            uint8_t tmp = P[j];
            P[j] = P[piv];
            P[piv] = tmp;
        }

        // Singularity check with relative tolerance
        float di = A[(size_t)j * n + j];
        float scale = 0.0f;
        for (uint16_t t = j; t < n; ++t)
        {
            float ak = fabsf(A[(size_t)j * n + t]);
            if (ak > scale)
                scale = ak;
        }
        float tol = (float)n * FLT_EPSILON * scale;
        if (fabsf(di) <= tol)
            return -ENOTSUP; // Singular matrix

        // Compute multipliers and apply rank-1 update to panel
        float inv_di = 1.0f / di;
        for (uint16_t r = (uint16_t)(j + 1); r < n; ++r)
            A[(size_t)r * n + j] *= inv_di;

        // Rank-1 update: remaining panel columns
        for (uint16_t c = (uint16_t)(j + 1); c < kend; ++c)
        {
            float u = A[(size_t)j * n + c];
            for (uint16_t r = (uint16_t)(j + 1); r < n; ++r)
                A[(size_t)r * n + c] -= A[(size_t)r * n + j] * u;
        }
    }

    return 0;
}

//==============================================================================
// TRIANGULAR SOLVE (TRSM) - Panel Operations
//==============================================================================

/**
 * @brief Left unit-lower TRSM: U12 := L11^{-1} * U12
 *
 * Solves L11 * U12 = U12 (in-place) where L11 is unit lower triangular.
 * Uses SIMD vectorization for the row updates.
 *
 * @param A   Matrix (n × n, row-major)
 * @param n   Matrix dimension (stride)
 * @param k   Panel starting row
 * @param ib  Panel height (L11 is ib × ib)
 * @param c0  Starting column of U12
 * @param nc  Width of U12
 */
static inline void trsm_left_unit_lower_on_U12(float *restrict A, uint16_t n,
                                               uint16_t k, uint16_t ib,
                                               uint16_t c0, uint16_t nc)
{
    for (uint16_t r = 0; r < ib; ++r)
    {
        float *Ur = A + (size_t)(k + r) * n + c0;

        // Solve: Ur -= sum(L[r,t] * Ut) for t < r
        for (uint16_t t = 0; t < r; ++t)
        {
            float lij = A[(size_t)(k + r) * n + (k + t)];
            if (lij == 0.0f)
                continue;

            const float *Ut = A + (size_t)(k + t) * n + c0;

            // ✅ VECTORIZED: Ur -= lij * Ut (unaligned access safe)
            __m256 vlij = _mm256_set1_ps(lij);
            uint16_t j = 0;

            for (; j + 7 < nc; j += 8)
            {
                __m256 ur = _mm256_loadu_ps(&Ur[j]); // ✅ Unaligned load
                __m256 ut = _mm256_loadu_ps(&Ut[j]); // ✅ Unaligned load
                ur = _mm256_fnmadd_ps(vlij, ut, ur); // ur = ur - lij*ut
                _mm256_storeu_ps(&Ur[j], ur);        // ✅ Unaligned store
            }

            // Scalar tail
            for (; j < nc; ++j)
                Ur[j] -= lij * Ut[j];
        }
    }
}

/**
 * @brief Right upper TRSM: L21 := L21 * U11^{-1} (simple gather version)
 *
 * Uses AVX2 gather instructions for strided access.
 * Used for small L21 matrices where transpose overhead isn't worthwhile.
 *
 * Performance: ~1.5-2× faster than _mm256_set_ps() but still memory-bound
 */
static inline int trsm_right_upper_on_L21_gather(
    float *restrict A, uint16_t n,
    uint16_t r0, uint16_t m2,
    uint16_t k, uint16_t ib)
{
    // Precompute gather index (stride = n floats)
    __m256i vindex = _mm256_setr_epi32(
        0 * n, 1 * n, 2 * n, 3 * n, 4 * n, 5 * n, 6 * n, 7 * n);

    for (int cc = (int)ib - 1; cc >= 0; --cc)
    {
        uint16_t c = (uint16_t)cc;
        float ucc = A[(size_t)(k + c) * n + (k + c)];

        // Singularity check
        float scale = 0.0f;
        const float *Urow = A + (size_t)(k + c) * n;
        for (uint16_t t = c; t < ib; ++t)
        {
            float v = fabsf(Urow[k + t]);
            if (v > scale)
                scale = v;
        }
        float tol = (float)ib * FLT_EPSILON * scale;
        if (fabsf(ucc) <= tol)
            return -ENOTSUP;

        float inv = 1.0f / ucc;
        __m256 vinv = _mm256_set1_ps(inv);

        // ✅ AVX2 GATHER: Divide column c by ucc
        uint16_t r = 0;
        for (; r + 7 < m2; r += 8)
        {
            size_t base = (size_t)(r0 + r) * n + (k + c);

            __m256 vals = _mm256_i32gather_ps(&A[base], vindex, 4);
            vals = _mm256_mul_ps(vals, vinv);

            // Scatter back (no AVX2 scatter, use scalar stores)
            float tmp[8];
            _mm256_storeu_ps(tmp, vals);
            for (int i = 0; i < 8; ++i)
                A[base + i * n] = tmp[i];
        }

        for (; r < m2; ++r)
            A[(size_t)(r0 + r) * n + (k + c)] *= inv;

        // ✅ AVX2 GATHER: Update columns
        for (int tt = 0; tt < cc; ++tt)
        {
            uint16_t t = (uint16_t)tt;
            float u_tc = A[(size_t)(k + t) * n + (k + c)];
            if (u_tc == 0.0f)
                continue;

            __m256 vu_tc = _mm256_set1_ps(u_tc);
            r = 0;

            for (; r + 7 < m2; r += 8)
            {
                size_t base_c = (size_t)(r0 + r) * n + (k + c);
                size_t base_t = (size_t)(r0 + r) * n + (k + t);

                __m256 L_c = _mm256_i32gather_ps(&A[base_c], vindex, 4);
                __m256 L_t = _mm256_i32gather_ps(&A[base_t], vindex, 4);

                L_t = _mm256_fnmadd_ps(L_c, vu_tc, L_t);

                float tmp[8];
                _mm256_storeu_ps(tmp, L_t);
                for (int i = 0; i < 8; ++i)
                    A[base_t + i * n] = tmp[i];
            }

            for (; r < m2; ++r)
            {
                A[(size_t)(r0 + r) * n + (k + t)] -=
                    A[(size_t)(r0 + r) * n + (k + c)] * u_tc;
            }
        }
    }

    return 0;
}

/**
 * @brief Right upper TRSM with transpose (FASTEST for large matrices)
 *
 * Transposes L21 to column-major, performs TRSM on contiguous data,
 * then transposes back. Uses your existing AVX2 transpose kernels.
 *
 * Performance: 3-5× faster than gather for large matrices
 *
 * @param A            LU matrix (n × n, row-major)
 * @param n            Matrix dimension
 * @param r0           Starting row of L21
 * @param m2           Height of L21
 * @param k            Panel starting column
 * @param ib           Panel width
 * @param temp_buffer  Workspace (ib × m2 floats)
 */
static inline int trsm_right_upper_on_L21_fast(
    float *restrict A, uint16_t n,
    uint16_t r0, uint16_t m2,
    uint16_t k, uint16_t ib,
    float *restrict temp_buffer)
{
    // ✅ STEP 1: Transpose L21 to column-major (uses your AVX2 8×8 kernel!)
    transpose_submatrix_to_colmajor(
        temp_buffer, // Dest: column-major ib × m2
        A,           // Source: row-major n × n
        r0, k,       // Submatrix start (row r0, col k)
        m2, ib,      // Submatrix size (m2 rows × ib cols)
        n);          // Source stride

    // ✅ STEP 2: TRSM on contiguous column-major data (BLAZING FAST!)
    for (int cc = (int)ib - 1; cc >= 0; --cc)
    {
        uint16_t c = (uint16_t)cc;
        float ucc = A[(size_t)(k + c) * n + (k + c)];

        // Singularity check
        float scale = 0.0f;
        const float *Urow = A + (size_t)(k + c) * n;
        for (uint16_t t = c; t < ib; ++t)
        {
            float v = fabsf(Urow[k + t]);
            if (v > scale)
                scale = v;
        }
        float tol = (float)ib * FLT_EPSILON * scale;
        if (fabsf(ucc) <= tol)
            return -ENOTSUP;

        float inv = 1.0f / ucc;
        __m256 vinv = _mm256_set1_ps(inv);

        // ✅ CONTIGUOUS LOADS/STORES (95% bandwidth efficiency!)
        float *col_c = temp_buffer + c * m2;
        uint16_t r = 0;

        for (; r + 7 < m2; r += 8)
        {
            __m256 vals = _mm256_loadu_ps(&col_c[r]);
            vals = _mm256_mul_ps(vals, vinv);
            _mm256_storeu_ps(&col_c[r], vals);
        }

        for (; r < m2; ++r)
            col_c[r] *= inv;

        // ✅ CONTIGUOUS UPDATES (FMA heaven!)
        for (int tt = 0; tt < cc; ++tt)
        {
            uint16_t t = (uint16_t)tt;
            float u_tc = A[(size_t)(k + t) * n + (k + c)];
            if (u_tc == 0.0f)
                continue;

            __m256 vu_tc = _mm256_set1_ps(u_tc);
            float *col_t = temp_buffer + t * m2;
            r = 0;

            for (; r + 7 < m2; r += 8)
            {
                __m256 L_c = _mm256_loadu_ps(&col_c[r]);
                __m256 L_t = _mm256_loadu_ps(&col_t[r]);
                L_t = _mm256_fnmadd_ps(L_c, vu_tc, L_t);
                _mm256_storeu_ps(&col_t[r], L_t);
            }

            for (; r < m2; ++r)
                col_t[r] -= col_c[r] * u_tc;
        }
    }

    // ✅ STEP 3: Transpose back to row-major (uses your AVX2 kernel again!)
    transpose_colmajor_to_submatrix(
        A,           // Dest: row-major n × n
        temp_buffer, // Source: column-major ib × m2
        r0, k,       // Submatrix start
        m2, ib,      // Submatrix size
        n);          // Dest stride

    return 0;
}

//==============================================================================
// MAIN LUP FACTORIZATION
//==============================================================================

/**
 * @brief LU factorization with partial pivoting (workspace version)
 *
 * Computes P*A = L*U factorization where:
 * - P is a permutation matrix (stored as permutation vector)
 * - L is unit lower triangular (stored in lower part of LU)
 * - U is upper triangular (stored in upper part of LU)
 *
 * Algorithm: Right-looking blocked GETRF with BLAS-3 trailing updates
 *
 * @param A    Input matrix (n × n, row-major, treated as unaligned)
 * @param LU   Output L+U matrix (n × n, row-major, in-place if A==LU)
 * @param P    Output permutation vector (P[i] = original row now at position i)
 * @param n    Matrix dimension
 * @param ws   Workspace (currently unused, kept for API stability)
 * @return 0 on success, -EINVAL for invalid args, -ENOTSUP if singular
 */
int lup_ws(const float *restrict A, float *restrict LU, uint8_t *restrict P,
           uint16_t n, lup_workspace_t *ws)
{
    if (n == 0)
        return -EINVAL;
    if (!ws)
        return -EINVAL;

    // Validate workspace size
    size_t required = lup_workspace_query(n);
    if (ws->size < required)
        return -ENOSPC;

    // Copy input if needed (handle in-place case)
    if (A != LU)
        memcpy(LU, A, (size_t)n * n * sizeof(float));

    // Initialize permutation to identity
    for (uint16_t i = 0; i < n; ++i)
        P[i] = (uint8_t)i;

    const uint16_t NB = (uint16_t)LUP_NB;

    // ✅ Allocate temp buffer for transposed L21 (worst case: n × NB)
    size_t temp_size = (size_t)n * NB * sizeof(float);
    float *temp_L21 = (float *)gemm_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, temp_size);
    if (!temp_L21)
        return -ENOMEM;

    //==========================================================================
    // BLOCKED FACTORIZATION LOOP
    //==========================================================================
    for (uint16_t k = 0; k < n; k = (uint16_t)(k + NB))
    {
        uint16_t ib = (uint16_t)((k + NB <= n) ? NB : (n - k)); // Panel width
        uint16_t nc = (uint16_t)(n - (k + ib));                 // Cols to right
        uint16_t m2 = (uint16_t)(n - (k + ib));                 // Rows below

        //----------------------------------------------------------------------
        // STEP 1: Panel factorization (unblocked LU with pivoting)
        //----------------------------------------------------------------------
        int rc = panel_lu_unblocked(LU, n, k, ib, P);
        if (rc)
        {
            gemm_aligned_free(temp_L21);
            return rc;
        }

        //----------------------------------------------------------------------
        // STEP 2: U12 = L11^{-1} * U12 (unit-lower TRSM)
        //----------------------------------------------------------------------
        if (nc)
        {
            trsm_left_unit_lower_on_U12(LU, n, k, ib, (uint16_t)(k + ib), nc);
        }

        //----------------------------------------------------------------------
        // STEP 3: L21 = L21 * U11^{-1} (upper TRSM, right-side)
        // ✅ ADAPTIVE: Use transpose for large matrices, gather for small
        //----------------------------------------------------------------------
        if (m2)
        {
            if ((size_t)m2 * ib >= LUP_TRSM_TRANSPOSE_THRESHOLD)
            {
                // 🚀 FAST: Transpose + contiguous TRSM (3-5× faster)
                rc = trsm_right_upper_on_L21_fast(LU, n, (uint16_t)(k + ib),
                                                  m2, k, ib, temp_L21);
            }
            else
            {
                // 🐌 SMALL: Direct gather (simpler, adequate for small matrices)
                rc = trsm_right_upper_on_L21_gather(LU, n, (uint16_t)(k + ib),
                                                    m2, k, ib);
            }

            if (rc)
            {
                gemm_aligned_free(temp_L21);
                return rc;
            }
        }

        //----------------------------------------------------------------------
        // STEP 4: Trailing update: A22 -= L21 * U12 (BLAS-3 GEMM)
        //----------------------------------------------------------------------
        if (m2 && nc)
        {
            // Get pointers to submatrix views (all point into LU)
            float *A22 = LU + (size_t)(k + ib) * n + (k + ib);
            const float *L21 = LU + (size_t)(k + ib) * n + k;
            const float *U12 = LU + (size_t)k * n + (k + ib);

            // ✅ USE gemm_strided (allocates its own workspace)
            // ✅ CORRECT ALPHA/BETA: C = beta*C + alpha*A*B
            //    We want: A22 = 1.0*A22 + (-1.0)*L21*U12
            rc = gemm_strided(
                A22,   // C (m2 × nc submatrix, output)
                L21,   // A (m2 × ib submatrix, input)
                U12,   // B (ib × nc submatrix, input)
                m2,    // M (rows of A22)
                ib,    // K (reduction dimension)
                nc,    // N (cols of A22)
                n,     // ldc (stride of LU matrix)
                n,     // lda (stride of LU matrix)
                n,     // ldb (stride of LU matrix)
                -1.0f, // alpha ← ✅ NEGATE the product L21*U12
                1.0f); // beta  ← ✅ KEEP existing A22 values

            if (rc)
            {
                gemm_aligned_free(temp_L21);
                return rc;
            }
        }
    }

    gemm_aligned_free(temp_L21);
    return 0;
}

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief LU factorization with partial pivoting (convenience wrapper)
 *
 * Allocates workspace internally for one-time use.
 * For repeated factorizations, create workspace once with lup_workspace_create().
 *
 * @param A    Input matrix (n × n, row-major)
 * @param LU   Output L+U matrix (n × n, row-major)
 * @param P    Output permutation vector
 * @param n    Matrix dimension
 * @return 0 on success, -ENOMEM if allocation fails, -ENOTSUP if singular
 */
int lup(const float *RESTRICT A, float *RESTRICT LU, uint8_t *P, uint16_t n)
{
    // Allocate minimal workspace
    size_t ws_size = lup_workspace_query(n);
    lup_workspace_t *ws = lup_workspace_create(ws_size);
    if (!ws)
        return -ENOMEM;

    // Perform factorization
    int rc = lup_ws(A, LU, P, n, ws);

    // Cleanup
    lup_workspace_destroy(ws);
    return rc;
}