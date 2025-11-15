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
// WORKSPACE IMPLEMENTATION
//==============================================================================

struct lup_workspace
{
    void *buffer;
    size_t size;
    int owns_memory;
    gemm_workspace_t *gemm_ws; // Embedded GEMM workspace
};

size_t lup_workspace_query(uint16_t n)
{
    if (n == 0)
        return 0;

    // LUP needs GEMM workspace for trailing updates
    // Worst case: n×n GEMM
    return gemm_workspace_query(n, n, n) + 64;
}

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

    // Initialize embedded GEMM workspace
    ws->gemm_ws = gemm_workspace_init(ws->buffer, size);
    if (!ws->gemm_ws)
    {
        linalg_aligned_free(ws->buffer);
        free(ws);
        return NULL;
    }

    return ws;
}

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

    // Initialize embedded GEMM workspace from same buffer
    ws->gemm_ws = gemm_workspace_init(buffer, size);
    if (!ws->gemm_ws)
    {
        free(ws);
        return NULL;
    }

    return ws;
}

void lup_workspace_destroy(lup_workspace_t *ws)
{
    if (!ws)
        return;

    if (ws->gemm_ws)
        gemm_workspace_destroy(ws->gemm_ws);

    if (ws->owns_memory)
        linalg_aligned_free(ws->buffer);

    free(ws);
}

/* ---------- Utilities ---------- */

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

/* Search pivot row (max |col|) among rows [r0..n-1] in column c. Returns row index. */
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

/* ---------- Small, unblocked panel LU with partial pivoting ----------
   Panel: A[k:n, k:k+ib) — physically swaps rows in A; updates P (final permutation table).
   On exit, panel contains L11 (unit-lower) and U11 (upper).
*/
static int panel_lu_unblocked(float *RESTRICT A, uint16_t n,
                              uint16_t k, uint16_t ib,
                              uint8_t *RESTRICT P)
{
    uint16_t kend = (uint16_t)(k + ib);
    for (uint16_t j = k; j < kend; ++j)
    {
        /* pivot search on column j, rows j..n-1 (global indices) */
        uint16_t piv = argmax_abs_col(A, n, j, j);

        /* swap physical rows j <-> piv across all columns, update final permutation P */
        if (piv != j)
        {
            swap_rows(A, n, j, piv);
            /* P is final permutation table: we must swap the entries that currently map to j/piv.
               The easiest: since we physically swapped rows, also swap P[j] and P[piv]. */
            uint8_t tmp = P[j];
            P[j] = P[piv];
            P[piv] = tmp;
        }

        /* singularity guard on U(j,j) with relative tol */
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
            return -ENOTSUP;

        /* form multipliers in column j (L part) and rank-1 update columns j+1..kend-1 only
           (we’ll finish the rest via BLAS-3 after panel TRSMs) */
        for (uint16_t r = (uint16_t)(j + 1); r < n; ++r)
            A[(size_t)r * n + j] /= di;

        /* Update the rest of the panel columns (up to kend) with rank-1 */
        for (uint16_t c = (uint16_t)(j + 1); c < kend; ++c)
        {
            float u = A[(size_t)j * n + c];
            for (uint16_t r = (uint16_t)(j + 1); r < n; ++r)
                A[(size_t)r * n + c] -= A[(size_t)r * n + j] * u;
        }
    }
    return 0;
}

/* ---------- Small TRSM on the panel (scalar) ---------- */

/* U12 := L11^{-1} * U12, where L11 is ib×ib unit-lower; U12 is ib×nc (rows k..k+ib-1, cols c0..). */
static inline void trsm_left_unit_lower_on_U12(float *restrict A, uint16_t n,
                                               uint16_t k, uint16_t ib,
                                               uint16_t c0, uint16_t nc)
{
    for (uint16_t r = 0; r < ib; ++r)
    {
        float *Ur = A + (size_t)(k + r) * n + c0;

        for (uint16_t t = 0; t < r; ++t)
        {
            float lij = A[(size_t)(k + r) * n + (k + t)];
            if (lij == 0.0f)
                continue;

            const float *Ut = A + (size_t)(k + t) * n + c0;

            // ✅ VECTORIZED: Ur -= lij * Ut
            __m256 vlij = _mm256_set1_ps(lij);
            uint16_t j = 0;

            for (; j + 7 < nc; j += 8)
            {
                __m256 ur = _mm256_loadu_ps(&Ur[j]);
                __m256 ut = _mm256_loadu_ps(&Ut[j]);
                ur = _mm256_fnmadd_ps(vlij, ut, ur);
                _mm256_storeu_ps(&Ur[j], ur);
            }

            for (; j < nc; ++j)
                Ur[j] -= lij * Ut[j];
        }
    }
}

/* L21 := L21 * U11^{-1}, where U11 is ib×ib upper (non-unit); L21 is m2×ib (rows r0.., cols k..k+ib-1).
   Right-side TRSM: process columns c=ib-1..0. */
static inline int trsm_right_upper_on_L21(float *restrict A, uint16_t n,
                                          uint16_t r0, uint16_t m2,
                                          uint16_t k, uint16_t ib)
{
    for (int cc = (int)ib - 1; cc >= 0; --cc)
    {
        uint16_t c = (uint16_t)cc;
        float ucc = A[(size_t)(k + c) * n + (k + c)];

        // Singularity check (unchanged)
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

        // ✅ VECTORIZED: Divide column c by ucc
        uint16_t r = 0;
        for (; r + 7 < m2; r += 8)
        {
            size_t base = (size_t)(r0 + r) * n + (k + c);
            __m256 vals = _mm256_setr_ps(
                A[base + 0 * n], A[base + 1 * n], A[base + 2 * n], A[base + 3 * n],
                A[base + 4 * n], A[base + 5 * n], A[base + 6 * n], A[base + 7 * n]);
            vals = _mm256_mul_ps(vals, vinv);

            // Scatter back
            float tmp[8];
            _mm256_storeu_ps(tmp, vals);
            for (int i = 0; i < 8; ++i)
                A[base + i * n] = tmp[i];
        }

        for (; r < m2; ++r)
            A[(size_t)(r0 + r) * n + (k + c)] *= inv;

        // Update (vectorized similarly)
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
                // Load L21[:,c] and L21[:,t] (strided)
                size_t base_c = (size_t)(r0 + r) * n + (k + c);
                size_t base_t = (size_t)(r0 + r) * n + (k + t);

                __m256 L_c = _mm256_setr_ps(
                    A[base_c + 0 * n], A[base_c + 1 * n], A[base_c + 2 * n], A[base_c + 3 * n],
                    A[base_c + 4 * n], A[base_c + 5 * n], A[base_c + 6 * n], A[base_c + 7 * n]);
                __m256 L_t = _mm256_setr_ps(
                    A[base_t + 0 * n], A[base_t + 1 * n], A[base_t + 2 * n], A[base_t + 3 * n],
                    A[base_t + 4 * n], A[base_t + 5 * n], A[base_t + 6 * n], A[base_t + 7 * n]);

                L_t = _mm256_fnmadd_ps(L_c, vu_tc, L_t);

                // Scatter back
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

int lup_ws(const float *restrict A, float *restrict LU, uint8_t *restrict P,
           uint16_t n, lup_workspace_t *ws)
{
    if (n == 0)
        return -EINVAL;
    if (!ws || !ws->gemm_ws)
        return -EINVAL;

    // Validate workspace size
    size_t required = lup_workspace_query(n);
    if (ws->size < required)
        return -ENOSPC;

    // Copy input if needed
    if (A != LU)
        memcpy(LU, A, (size_t)n * n * sizeof(float));

    // Initialize permutation
    for (uint16_t i = 0; i < n; ++i)
        P[i] = (uint8_t)i;

    const uint16_t NB = (uint16_t)LUP_NB;

    for (uint16_t k = 0; k < n; k = (uint16_t)(k + NB))
    {
        uint16_t ib = (uint16_t)((k + NB <= n) ? NB : (n - k));
        uint16_t nc = (uint16_t)(n - (k + ib)); // columns to right
        uint16_t m2 = (uint16_t)(n - (k + ib)); // rows below

        // 1) Panel factorization (unblocked LU with pivoting)
        int rc = panel_lu_unblocked(LU, n, k, ib, P);
        if (rc)
            return rc;

        // 2) U12 = L11^{-1} * U12 (unit-lower TRSM)
        if (nc)
        {
            trsm_left_unit_lower_on_U12(LU, n, k, ib, (uint16_t)(k + ib), nc);
        }

        // 3) L21 = L21 * U11^{-1} (upper TRSM)
        if (m2)
        {
            rc = trsm_right_upper_on_L21(LU, n, (uint16_t)(k + ib), m2, k, ib);
            if (rc)
                return rc;
        }

        // 4) Trailing update: A22 -= L21 * U12 (GEMM)
        if (m2 && nc)
        {
            const float *L21 = LU + (size_t)(k + ib) * n + k;
            const float *U12 = LU + (size_t)k * n + (k + ib);
            float *A22 = LU + (size_t)(k + ib) * n + (k + ib);

            // Use embedded GEMM workspace
            rc = gemm_ws(A22, L21, U12, m2, ib, nc,
                         1.0f,  // alpha
                         -1.0f, // beta (C -= A*B)
                         ws->gemm_ws);
            if (rc)
                return rc;
        }
    }

    return 0;
}

/* ---------- Public API: blocked BLAS-3 LUP ---------- */
int lup(const float *RESTRICT A, float *RESTRICT LU, uint8_t *P, uint16_t n)
{
    // Allocate LUP workspace (includes embedded GEMM workspace)
    size_t ws_size = lup_workspace_query(n);
    lup_workspace_t *ws = lup_workspace_create(ws_size);
    if (!ws)
        return -ENOMEM;

    // Call lup_ws (does all the work)
    int rc = lup_ws(A, LU, P, n, ws);

    // Cleanup
    lup_workspace_destroy(ws);
    return rc;
}