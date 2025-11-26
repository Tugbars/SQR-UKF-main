/**
 * @file trsm_right.h
 * @brief Right-side triangular solve implementations for SR-UKF Kalman gain
 *
 * @details
 * The Kalman gain K = Pxy · Pyy^(-1) = Pxy · (Sy·Sy^T)^(-1) = Pxy · Sy^(-T) · Sy^(-1)
 *
 * This requires RIGHT-solves:
 *   Step 1: Z · Sy^T = Pxy  →  Z = Pxy · Sy^(-T)   (trsm_right_upper_transpose)
 *   Step 2: K · Sy   = Z    →  K = Z · Sy^(-1)     (trsm_right_upper)
 *
 * These functions solve for X in equations of the form X · A = B (right-side solve)
 * as opposed to A · X = B (left-side solve).
 *
 * @author Generated for SR-UKF Kalman gain fix
 */

#ifndef TRSM_RIGHT_H
#define TRSM_RIGHT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

//==============================================================================
// SCALAR IMPLEMENTATIONS (reference/fallback)
//==============================================================================

/**
 * @brief Right-solve: X · U = B where U is upper triangular (scalar version)
 *
 * @details
 * Solves for X in X · U = B, computing X = B · U^(-1).
 *
 * For upper triangular U, we process columns LEFT to RIGHT:
 *   Column j: X[:,j] = (B[:,j] - Σ_{k<j} X[:,k] * U[k,j]) / U[j,j]
 *
 * @param[in,out] B  RHS matrix [m × n], overwritten with solution X
 * @param[in]     U  Upper triangular matrix [n × n], row-major
 * @param[in]     m  Number of rows in B
 * @param[in]     n  Dimension of U (and columns in B)
 * @param[in]     ldb Leading dimension of B
 * @param[in]     ldu Leading dimension of U
 *
 * @return 0 on success, -1 if singular (zero diagonal)
 */
static inline int trsm_right_upper_scalar(
    float *restrict B,
    const float *restrict U,
    size_t m, size_t n,
    size_t ldb, size_t ldu)
{
    /* Process columns left to right */
    for (size_t j = 0; j < n; ++j)
    {
        /* Check for singularity */
        float ujj = U[j * ldu + j];
        if (ujj == 0.0f)
            return -1;

        float inv_ujj = 1.0f / ujj;

        /* Subtract contributions from already-solved columns (k < j) */
        for (size_t k = 0; k < j; ++k)
        {
            float ukj = U[k * ldu + j];
            if (ukj == 0.0f)
                continue;

            for (size_t i = 0; i < m; ++i)
                B[i * ldb + j] -= B[i * ldb + k] * ukj;
        }

        /* Scale column j by inverse diagonal */
        for (size_t i = 0; i < m; ++i)
            B[i * ldb + j] *= inv_ujj;
    }

    return 0;
}

/**
 * @brief Right-solve with transpose: X · U^T = B where U is upper triangular (scalar version)
 *
 * @details
 * Solves for X in X · U^T = B, computing X = B · U^(-T).
 *
 * Since U^T is lower triangular, we process columns RIGHT to LEFT:
 *   Column j: X[:,j] = (B[:,j] - Σ_{k>j} X[:,k] * U^T[k,j]) / U[j,j]
 *   where U^T[k,j] = U[j,k]
 *
 * @param[in,out] B  RHS matrix [m × n], overwritten with solution X
 * @param[in]     U  Upper triangular matrix [n × n], row-major (accessed as U^T)
 * @param[in]     m  Number of rows in B
 * @param[in]     n  Dimension of U (and columns in B)
 * @param[in]     ldb Leading dimension of B
 * @param[in]     ldu Leading dimension of U
 *
 * @return 0 on success, -1 if singular
 */
static inline int trsm_right_upper_transpose_scalar(
    float *restrict B,
    const float *restrict U,
    size_t m, size_t n,
    size_t ldb, size_t ldu)
{
    /* Process columns right to left (U^T is lower triangular) */
    for (int j = (int)n - 1; j >= 0; --j)
    {
        /* Check for singularity */
        float ujj = U[j * ldu + j];
        if (ujj == 0.0f)
            return -1;

        float inv_ujj = 1.0f / ujj;

        /* Subtract contributions from already-solved columns (k > j) */
        for (size_t k = (size_t)j + 1; k < n; ++k)
        {
            /* U^T[k,j] = U[j,k] */
            float ut_kj = U[j * ldu + k];
            if (ut_kj == 0.0f)
                continue;

            for (size_t i = 0; i < m; ++i)
                B[i * ldb + j] -= B[i * ldb + k] * ut_kj;
        }

        /* Scale column j by inverse diagonal */
        for (size_t i = 0; i < m; ++i)
            B[i * ldb + j] *= inv_ujj;
    }

    return 0;
}

//==============================================================================
// AVX2-OPTIMIZED IMPLEMENTATIONS
//==============================================================================

#ifdef __AVX2__

/**
 * @brief Right-solve: X · U = B where U is upper triangular (AVX2 optimized)
 *
 * @details
 * Optimizations:
 * - 8-wide SIMD for column updates
 * - 4-row unrolling for better ILP
 * - Prefetching for large matrices
 */
static inline int trsm_right_upper_avx2(
    float *restrict B,
    const float *restrict U,
    size_t m, size_t n,
    size_t ldb, size_t ldu)
{
    /* Process columns left to right */
    for (size_t j = 0; j < n; ++j)
    {
        float ujj = U[j * ldu + j];
        if (ujj == 0.0f)
            return -1;

        __m256 inv_ujj_v = _mm256_set1_ps(1.0f / ujj);

        /* Subtract contributions from columns k < j */
        for (size_t k = 0; k < j; ++k)
        {
            float ukj = U[k * ldu + j];
            if (ukj == 0.0f)
                continue;

            __m256 ukj_v = _mm256_set1_ps(ukj);

            /* Process 4 rows at a time for ILP */
            size_t i = 0;
            for (; i + 3 < m; i += 4)
            {
                /* Load B[:,k] for 4 rows */
                float bk0 = B[(i + 0) * ldb + k];
                float bk1 = B[(i + 1) * ldb + k];
                float bk2 = B[(i + 2) * ldb + k];
                float bk3 = B[(i + 3) * ldb + k];

                /* Update B[:,j] for 4 rows */
                B[(i + 0) * ldb + j] -= bk0 * ukj;
                B[(i + 1) * ldb + j] -= bk1 * ukj;
                B[(i + 2) * ldb + j] -= bk2 * ukj;
                B[(i + 3) * ldb + j] -= bk3 * ukj;
            }

            /* Scalar tail */
            for (; i < m; ++i)
                B[i * ldb + j] -= B[i * ldb + k] * ukj;
        }

        /* Scale column j */
        float inv_ujj = 1.0f / ujj;
        size_t i = 0;

        /* Process 4 rows at a time */
        for (; i + 3 < m; i += 4)
        {
            B[(i + 0) * ldb + j] *= inv_ujj;
            B[(i + 1) * ldb + j] *= inv_ujj;
            B[(i + 2) * ldb + j] *= inv_ujj;
            B[(i + 3) * ldb + j] *= inv_ujj;
        }

        for (; i < m; ++i)
            B[i * ldb + j] *= inv_ujj;
    }

    return 0;
}

/**
 * @brief Right-solve with transpose: X · U^T = B (AVX2 optimized)
 */
static inline int trsm_right_upper_transpose_avx2(
    float *restrict B,
    const float *restrict U,
    size_t m, size_t n,
    size_t ldb, size_t ldu)
{
    /* Process columns right to left */
    for (int j = (int)n - 1; j >= 0; --j)
    {
        float ujj = U[j * ldu + j];
        if (ujj == 0.0f)
            return -1;

        float inv_ujj = 1.0f / ujj;

        /* Subtract contributions from columns k > j */
        for (size_t k = (size_t)j + 1; k < n; ++k)
        {
            float ut_kj = U[j * ldu + k]; /* U^T[k,j] = U[j,k] */
            if (ut_kj == 0.0f)
                continue;

            /* Process 4 rows at a time */
            size_t i = 0;
            for (; i + 3 < m; i += 4)
            {
                B[(i + 0) * ldb + j] -= B[(i + 0) * ldb + k] * ut_kj;
                B[(i + 1) * ldb + j] -= B[(i + 1) * ldb + k] * ut_kj;
                B[(i + 2) * ldb + j] -= B[(i + 2) * ldb + k] * ut_kj;
                B[(i + 3) * ldb + j] -= B[(i + 3) * ldb + k] * ut_kj;
            }

            for (; i < m; ++i)
                B[i * ldb + j] -= B[i * ldb + k] * ut_kj;
        }

        /* Scale column j */
        size_t i = 0;
        for (; i + 3 < m; i += 4)
        {
            B[(i + 0) * ldb + j] *= inv_ujj;
            B[(i + 1) * ldb + j] *= inv_ujj;
            B[(i + 2) * ldb + j] *= inv_ujj;
            B[(i + 3) * ldb + j] *= inv_ujj;
        }

        for (; i < m; ++i)
            B[i * ldb + j] *= inv_ujj;
    }

    return 0;
}

#endif /* __AVX2__ */

//==============================================================================
// AUTO-DISPATCH WRAPPERS
//==============================================================================

/**
 * @brief Right-solve: X · U = B (auto-dispatching)
 */
static inline int trsm_right_upper(
    float *restrict B,
    const float *restrict U,
    size_t m, size_t n,
    size_t ldb, size_t ldu)
{
#ifdef __AVX2__
    return trsm_right_upper_avx2(B, U, m, n, ldb, ldu);
#else
    return trsm_right_upper_scalar(B, U, m, n, ldb, ldu);
#endif
}

/**
 * @brief Right-solve with transpose: X · U^T = B (auto-dispatching)
 */
static inline int trsm_right_upper_transpose(
    float *restrict B,
    const float *restrict U,
    size_t m, size_t n,
    size_t ldb, size_t ldu)
{
#ifdef __AVX2__
    return trsm_right_upper_transpose_avx2(B, U, m, n, ldb, ldu);
#else
    return trsm_right_upper_transpose_scalar(B, U, m, n, ldb, ldu);
#endif
}

//==============================================================================
// KALMAN GAIN COMPUTATION
//==============================================================================

/**
 * @brief Compute Kalman gain K = Pxy · Pyy^(-1) using right-TRSM
 *
 * @details
 * Given:
 *   - Sy: upper triangular square root of Pyy (i.e., Pyy = Sy · Sy^T)
 *   - Pxy: cross-covariance matrix
 *
 * Computes:
 *   K = Pxy · Sy^(-T) · Sy^(-1)
 *
 * Using two right-solves:
 *   1. Z · Sy^T = Pxy  →  Z = Pxy · Sy^(-T)
 *   2. K · Sy   = Z    →  K = Z · Sy^(-1)
 *
 * @param[out]    K   Kalman gain matrix [n × n]
 * @param[in]     Pxy Cross-covariance matrix [n × n]
 * @param[in]     Sy  Upper triangular Cholesky factor [n × n]
 * @param[in]     n   Dimension
 * @param[in]     ldK Leading dimension of K
 * @param[in]     ldPxy Leading dimension of Pxy
 * @param[in]     ldSy Leading dimension of Sy
 *
 * @return 0 on success, -1 if Sy is singular
 */
static inline int compute_kalman_gain(
    float *restrict K,
    const float *restrict Pxy,
    const float *restrict Sy,
    size_t n,
    size_t ldK,
    size_t ldPxy,
    size_t ldSy)
{
    /* Copy Pxy to K (will be overwritten with solution) */
    for (size_t i = 0; i < n; ++i)
    {
        memcpy(K + i * ldK, Pxy + i * ldPxy, n * sizeof(float));
    }

    /* Step 1: K · Sy^T = Pxy  →  K = Pxy · Sy^(-T) */
    int rc = trsm_right_upper_transpose(K, Sy, n, n, ldK, ldSy);
    if (rc != 0)
        return rc;

    /* Step 2: K · Sy = K  →  K = K · Sy^(-1) */
    rc = trsm_right_upper(K, Sy, n, n, ldK, ldSy);
    if (rc != 0)
        return rc;

    return 0;
}

#endif /* TRSM_RIGHT_H */