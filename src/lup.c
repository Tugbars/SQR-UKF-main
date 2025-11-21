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
 *     4) Trailing update: A22 ← A22 − L21·U12  (GEMM; AVX2 8×16 kernel if available)
 *
 * Notes:
 *  - Row-major everywhere; leading dimension is n.
 *  - Small unblocked TRSM is used only on the panel (ib×ib). The big A22 update is BLAS-3.
 *  - P ends as a *final permutation table* (not ipiv steps), compatible with your RHS pivot apply.
 *  - All input data treated as potentially unaligned (uses loadu/storeu).
 *  - Uses gemm_strided() for trailing updates (allocates its own workspace).
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <immintrin.h>
#include "linalg_simd.h" // linalg_has_avx2(), linalg_aligned_alloc/free, LINALG_DEFAULT_ALIGNMENT
#include "../gemm/gemm.h"

#ifndef LUP_NB
#define LUP_NB 128 // panel/block size (try 96–160)
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

    ws->buffer = linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, size);
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

    if (ws->owns_memory && ws->buffer)
        linalg_aligned_free(ws->buffer);

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
 * @brief Right upper TRSM: L21 := L21 * U11^{-1}
 *
 * Solves L21 * U11 = L21 (in-place) where U11 is upper triangular.
 * Processes columns in reverse order (c = ib-1 down to 0).
 * Uses SIMD vectorization for column operations.
 *
 * @param A   Matrix (n × n, row-major)
 * @param n   Matrix dimension (stride)
 * @param r0  Starting row of L21
 * @param m2  Height of L21
 * @param k   Panel starting column
 * @param ib  Panel width (U11 is ib × ib)
 * @return 0 on success, -ENOTSUP if singular
 */
static inline int trsm_right_upper_on_L21(float *restrict A, uint16_t n,
                                          uint16_t r0, uint16_t m2,
                                          uint16_t k, uint16_t ib)
{
    // Process columns in reverse order (back-substitution)
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
            return -ENOTSUP; // Singular

        float inv = 1.0f / ucc;
        __m256 vinv = _mm256_set1_ps(inv);

        // ✅ VECTORIZED: Divide column c by ucc (strided access)
        uint16_t r = 0;
        for (; r + 7 < m2; r += 8)
        {
            size_t base = (size_t)(r0 + r) * n + (k + c);

            // Gather 8 elements (strided)
            __m256 vals = _mm256_set_ps(
                A[base + 7 * n], A[base + 6 * n], A[base + 5 * n], A[base + 4 * n],
                A[base + 3 * n], A[base + 2 * n], A[base + 1 * n], A[base + 0 * n]);

            vals = _mm256_mul_ps(vals, vinv);

            // Scatter back (strided store)
            float tmp[8];
            _mm256_storeu_ps(tmp, vals); // ✅ Unaligned store
            for (int i = 0; i < 8; ++i)
                A[base + i * n] = tmp[i];
        }

        // Scalar tail
        for (; r < m2; ++r)
            A[(size_t)(r0 + r) * n + (k + c)] *= inv;

        // ✅ VECTORIZED: Update L21[:,t] -= L21[:,c] * U[t,c] for t < c
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

                // Gather L21[:,c] and L21[:,t] (strided)
                __m256 L_c = _mm256_set_ps(
                    A[base_c + 7 * n], A[base_c + 6 * n], A[base_c + 5 * n], A[base_c + 4 * n],
                    A[base_c + 3 * n], A[base_c + 2 * n], A[base_c + 1 * n], A[base_c + 0 * n]);
                __m256 L_t = _mm256_set_ps(
                    A[base_t + 7 * n], A[base_t + 6 * n], A[base_t + 5 * n], A[base_t + 4 * n],
                    A[base_t + 3 * n], A[base_t + 2 * n], A[base_t + 1 * n], A[base_t + 0 * n]);

                L_t = _mm256_fnmadd_ps(L_c, vu_tc, L_t); // L_t -= L_c * u_tc

                // Scatter back (strided store)
                float tmp[8];
                _mm256_storeu_ps(tmp, L_t); // ✅ Unaligned store
                for (int i = 0; i < 8; ++i)
                    A[base_t + i * n] = tmp[i];
            }

            // Scalar tail
            for (; r < m2; ++r)
            {
                A[(size_t)(r0 + r) * n + (k + t)] -=
                    A[(size_t)(r0 + r) * n + (k + c)] * u_tc;
            }
        }
    }

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
            return rc;

        //----------------------------------------------------------------------
        // STEP 2: U12 = L11^{-1} * U12 (unit-lower TRSM)
        //----------------------------------------------------------------------
        if (nc)
        {
            trsm_left_unit_lower_on_U12(LU, n, k, ib, (uint16_t)(k + ib), nc);
        }

        //----------------------------------------------------------------------
        // STEP 3: L21 = L21 * U11^{-1} (upper TRSM, right-side)
        //----------------------------------------------------------------------
        if (m2)
        {
            rc = trsm_right_upper_on_L21(LU, n, (uint16_t)(k + ib), m2, k, ib);
            if (rc)
                return rc;
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
                return rc;
        }
    }

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