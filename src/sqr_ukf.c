#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* define to 1 if you want the 8-way batching; keep 0 by default */
#ifndef SQR_UKF_ENABLE_BATCH8
#define SQR_UKF_ENABLE_BATCH8 0
#endif

#ifndef UKF_PREFETCH_ROWS_AHEAD
#define UKF_PREFETCH_ROWS_AHEAD 1 /* 0..2 are sensible */
#endif
#ifndef UKF_PREFETCH_DIST_BYTES
#define UKF_PREFETCH_DIST_BYTES 128 /* 64 or 128 */
#endif
#ifndef UKF_PREFETCH_MIN_L
#define UKF_PREFETCH_MIN_L 128
#endif

#ifndef UKF_TRANS_PF_MIN_L
#define UKF_TRANS_PF_MIN_L 128 /* enable prefetch when L >= this */
#endif
#ifndef UKF_TRANS_PF_ROWS_AHEAD
#define UKF_TRANS_PF_ROWS_AHEAD 1 /* 0..2 sensible; 1 is safe default */
#endif

#ifndef UKF_MEAN_PF_MIN_ROWS
#define UKF_MEAN_PF_MIN_ROWS 128 /* enable row-ahead prefetch when L >= this */
#endif
#ifndef UKF_MEAN_PF_ROWS_AHEAD
#define UKF_MEAN_PF_ROWS_AHEAD 1 /* 0..2 are reasonable */
#endif

#ifndef UKF_APRIME_PF_MIN_L
#define UKF_APRIME_PF_MIN_L 128 /* enable prefetch when L >= this */
#endif
#ifndef UKF_APRIME_PF_ROWS_AHEAD
#define UKF_APRIME_PF_ROWS_AHEAD 1 /* 0..2 are sensible */
#endif
#ifndef UKF_APRIME_PF_DIST_BYTES
#define UKF_APRIME_PF_DIST_BYTES 128 /* cache-line distance (64 or 128) */
#endif

#ifndef UKF_PXY_PF_MIN_N
#define UKF_PXY_PF_MIN_N 256 /* enable prefetch when N >= this */
#endif
#ifndef UKF_PXY_PF_ROWS_AHEAD
#define UKF_PXY_PF_ROWS_AHEAD 1 /* prefetch this many future Y rows (0..2 sensible) */
#endif
#ifndef UKF_PXY_PF_DIST_BYTES
#define UKF_PXY_PF_DIST_BYTES 128 /* stream prefetch distance within a row: 64 or 128 */
#endif

#ifndef UKF_UPD_COLBLOCK
#define UKF_UPD_COLBLOCK 64 /* RHS column block for triangular solves */
#endif
#ifndef UKF_UPD_PF_MIN_N
#define UKF_UPD_PF_MIN_N 128 /* enable prefetch in solves when n >= this */
#endif
#ifndef UKF_UPD_PF_DIST_BYTES
#define UKF_UPD_PF_DIST_BYTES 128 /* prefetch distance along RHS rows */
#endif

#ifndef UKF_PXY_PF_MIN_L
#define UKF_PXY_PF_MIN_L 16 /* enable row-ahead prefetch when L >= this */
#endif
#ifndef UKF_PXY_PF_MIN_N
#define UKF_PXY_PF_MIN_N 32 /* enable within-row prefetch when N >= this */
#endif
#ifndef UKF_PXY_PF_ROWS_AHEAD
#define UKF_PXY_PF_ROWS_AHEAD 1 /* 0..2 sensible */
#endif
#ifndef UKF_PXY_PF_DIST_BYTES
#define UKF_PXY_PF_DIST_BYTES 128 /* 64 or 128 are typical */
#endif

/* =================== Reusable workspace for SR-UKF QR step =================== */
typedef struct
{
    float *Aprime; /* (M x L) row-major, M = 3L */
    float *R_;     /* (M x L) row-major */
    float *b;      /* (L) */
    size_t capL;   /* capacity in L */
} ukf_qr_ws_t;

typedef struct
{
    float *Z;     /* n x n, reused as K after backward solve */
    float *Ky;    /* n */
    float *U;     /* n x n */
    float *Uk;    /* n */
    float *yyhat; /* n */
    size_t cap;   /* in elements, for n*n buffers */
} ukf_upd_ws_t;

/**
 * @brief Clean up QR workspace (free internal allocations)
 */
static inline void ukf_qr_ws_cleanup(ukf_qr_ws_t *ws)
{
    if (!ws)
        return;

    if (ws->Aprime)
    {
        gemm_aligned_free(ws->Aprime);
        ws->Aprime = NULL;
    }
    if (ws->R_)
    {
        gemm_aligned_free(ws->R_);
        ws->R_ = NULL;
    }
    if (ws->b)
    {
        gemm_aligned_free(ws->b);
        ws->b = NULL;
    }
    ws->capL = 0;
}

/**
 * @brief Clean up update workspace (free internal allocations)
 */
static inline void ukf_upd_ws_cleanup(ukf_upd_ws_t *ws)
{
    if (!ws)
        return;

    if (ws->Z)
    {
        gemm_aligned_free(ws->Z);
        ws->Z = NULL;
    }
    if (ws->U)
    {
        gemm_aligned_free(ws->U);
        ws->U = NULL;
    }
    if (ws->Uk)
    {
        gemm_aligned_free(ws->Uk);
        ws->Uk = NULL;
    }
    if (ws->Ky)
    {
        gemm_aligned_free(ws->Ky);
        ws->Ky = NULL;
    }
    if (ws->yyhat)
    {
        gemm_aligned_free(ws->yyhat);
        ws->yyhat = NULL;
    }
    ws->cap = 0;
}

static inline int ukf_qr_ws_ensure(ukf_qr_ws_t *ws, size_t L)
{
    const size_t M = 3u * L;
    const size_t need_A = M * L;
    const size_t need_R = M * L;
    const size_t need_b = L;

    if (ws->capL >= L && ws->Aprime && ws->R_ && ws->b)
        return 0;

    /* Clean up old allocations */
    ukf_qr_ws_cleanup(ws);

    /* Allocate new (larger) workspace */
    ws->Aprime = (float *)gemm_aligned_alloc(32, need_A * sizeof(float));
    ws->R_ = (float *)gemm_aligned_alloc(32, need_R * sizeof(float));
    ws->b = (float *)gemm_aligned_alloc(32, need_b * sizeof(float));
    ws->capL = (ws->Aprime && ws->R_ && ws->b) ? L : 0;

    return (ws->capL ? 0 : -ENOMEM);
}

static inline int ukf_upd_ws_ensure(ukf_upd_ws_t *ws, uint16_t n)
{
    const size_t nn = (size_t)n * (size_t)n;

    if (ws->cap >= nn && ws->Z && ws->U && ws->Uk && ws->Ky && ws->yyhat)
        return 0;

    /* Clean up old allocations */
    ukf_upd_ws_cleanup(ws);

    /* Allocate new (larger) workspace */
    ws->Z = (float *)gemm_aligned_alloc(32, nn * sizeof(float));
    ws->U = (float *)gemm_aligned_alloc(32, nn * sizeof(float));
    ws->Uk = (float *)gemm_aligned_alloc(32, (size_t)n * sizeof(float));
    ws->Ky = (float *)gemm_aligned_alloc(32, (size_t)n * sizeof(float));
    ws->yyhat = (float *)gemm_aligned_alloc(32, (size_t)n * sizeof(float));
    ws->cap = (ws->Z && ws->U && ws->Uk && ws->Ky && ws->yyhat) ? nn : 0;

    return (ws->cap ? 0 : -ENOMEM);
}

#if LINALG_SIMD_ENABLE
static inline float avx2_sum_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}
#endif

/**
 * @brief Compute Unscented Transform weights for mean (Wm) and covariance (Wc).
 *
 * @details
 *  Builds the standard UKF weights from parameters \p alpha, \p beta, \p kappa and
 *  state size \p L. The weights are:
 *  \f[
 *    \lambda = \alpha^2 (L + \kappa) - L,\quad
 *    W_m^{(0)} = \frac{\lambda}{L+\lambda},\quad
 *    W_c^{(0)} = W_m^{(0)} + (1 - \alpha^2 + \beta),\quad
 *    W_m^{(i)} = W_c^{(i)} = \frac{1}{2(L+\lambda)}\ \text{for}\ i=1..2L.
 *  \f]
 *
 *  A fast AVX2 path bulk-fills the constant tail (i ≥ 1) in 8-wide chunks to reduce
 *  loop overhead and memory traffic. For small N or when AVX2 is unavailable, a
 *  scalar loop is used.
 *
 * @param[out] Wc    Covariance weights, length N = 2L + 1.
 * @param[out] Wm    Mean weights, length N = 2L + 1.
 * @param[in]  alpha Spread parameter (typ. 1e-3 ≤ α ≤ 1).
 * @param[in]  beta  Prior distribution knowledge (β=2 for Gaussian optimality).
 * @param[in]  kappa Secondary scaling parameter (often 0 or 3−L).
 * @param[in]  L     State dimension.
 *
 * @note
 *  - Requires L ≥ 1 for meaningful weights.
 *  - When \f$L+\lambda\f$ is very small, denominators can amplify round-off.
 *    Choose \p alpha/\p kappa sensibly for numerical stability.
 *  - Falls back to scalar if AVX2/FMA is not available or N < 9.
 */
static void create_weights(float Wc[],
                           float Wm[],
                           float alpha,
                           float beta,
                           float kappa,
                           uint8_t L)
{
    const size_t N = (size_t)(2u * L + 1u); //!< Number of sigma points
    const float Lf = (float)L;              //!< State size as float

    /* λ = α^2 (L + κ) − L */
    const float lam = alpha * alpha * (Lf + kappa) - Lf;

    /* Common denominator 1 / (L + λ) */
    const float den = 1.0f / (Lf + lam);

    /* First element (i = 0) */
    Wm[0] = lam * den;
    Wc[0] = Wm[0] + 1.0f - alpha * alpha + beta;

    /* Tail (i ≥ 1): 0.5 / (L + λ) */
    const float hv = 0.5f * den;

#if LINALG_SIMD_ENABLE
    /* AVX2 bulk fill of the constant tail when N >= 9 (i.e., at least one full 8-lane chunk). */
    if (ukf_has_avx2() && N >= 9)
    {
        const __m256 v = _mm256_set1_ps(hv);
        size_t i = 1; //!< Start filling from index 1
        for (; i + 7 < N; i += 8)
        {
            _mm256_storeu_ps(&Wm[i], v); //!< Store 8 identical Wm values
            _mm256_storeu_ps(&Wc[i], v); //!< Store 8 identical Wc values
        }
        /* Scalar cleanup for the remaining elements (if N is not a multiple of 8). */
        for (; i < N; ++i)
        {
            Wm[i] = hv;
            Wc[i] = hv;
        }
        return;
    }
#endif

    /* Portable scalar tail initialization */
    for (size_t i = 1; i < N; ++i)
    {
        Wm[i] = hv;
        Wc[i] = hv;
    }
}

/**
 * @brief Build the Unscented sigma-point matrix X from mean x and SR-covariance S.
 *
 * @details
 *  Constructs the (L × (2L+1)) sigma matrix in row-major order:
 *  - Column 0:            X(:,0)     = x
 *  - Columns 1..L:        X(:,j)     = x + γ S(:,j)         (j=1..L)
 *  - Columns L+1..2L:     X(:,L+j)   = x - γ S(:,j)         (j=1..L)
 *
 *  where γ = α √(L + κ). The input S is the **square-root covariance** (upper
 *  or lower is fine since we access rows of S here; we simply scale each row’s
 *  entries by ±γ).
 *
 *  SIMD path:
 *   - Uses AVX2/FMA to compute two sigma columns ( +γ and −γ ) in parallel for
 *     8 elements per iteration.
 *   - Optional row-ahead prefetch to reduce cache miss latency for large L.
 *
 *  Scalar path:
 *   - Portable, straightforward loops for all L and (2L+1) columns.
 *
 * @param[out] X      Sigma matrix, row-major, size L × (2L+1).
 * @param[in]  x      State mean vector of length L.
 * @param[in]  S      Square-root covariance (SR) matrix, row-major L × L.
 * @param[in]  alpha  UKF spread parameter (α).
 * @param[in]  kappa  Secondary scaling parameter (κ).
 * @param[in]  L8     State dimension (stored as uint8_t to match surrounding API).
 *
 * @note
 *  - X must not alias x or S.
 *  - This routine assumes S is already a valid SR factor (e.g., from QR/Cholesky).
 *  - Vectorized path requires AVX2+FMA and benefits most when L ≥ 8.
 */
static void create_sigma_point_matrix(float X[],
                                      const float x[],
                                      const float S[],
                                      float alpha,
                                      float kappa,
                                      uint8_t L8)
{
    const size_t L = (size_t)L8;
    const size_t N = 2u * L + 1u;
    const float gamma = alpha * sqrtf((float)L + kappa);

#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && L >= 16) // Higher threshold for unrolling
    {
        const __m256 g = _mm256_set1_ps(gamma);
        const __m256 ng = _mm256_set1_ps(-gamma);

        const size_t pf_elts = UKF_PREFETCH_DIST_BYTES / sizeof(float);
        const int do_pf = (L >= (size_t)UKF_PREFETCH_MIN_L);
        const int rows_ahead = UKF_PREFETCH_ROWS_AHEAD;

        size_t i = 0;

        /* Process 2 rows at a time - fits in registers */
        for (; i + 1 < L; i += 2)
        {
            float *Xi0 = X + (i + 0) * N;
            float *Xi1 = X + (i + 1) * N;
            const float *Si0 = S + (i + 0) * L;
            const float *Si1 = S + (i + 1) * L;

            const __m256 xi0 = _mm256_set1_ps(x[i + 0]);
            const __m256 xi1 = _mm256_set1_ps(x[i + 1]);

            Xi0[0] = x[i + 0];
            Xi1[0] = x[i + 1];

            /* Prefetch ahead for both rows */
            if (do_pf && rows_ahead > 0 && i + 2 < L)
            {
                _mm_prefetch((const char *)(S + (i + 2) * L), _MM_HINT_T0);
                _mm_prefetch((const char *)(X + (i + 2) * N), _MM_HINT_T0);
            }

            size_t j = 0;
            for (; j + 7 < L; j += 8)
            {
                if (do_pf && j + pf_elts + 8 < L)
                {
                    _mm_prefetch((const char *)(Si0 + j + pf_elts), _MM_HINT_T0);
                    _mm_prefetch((const char *)(Si1 + j + pf_elts), _MM_HINT_T0);
                }

                /* Row 0: load, compute, store immediately (frees registers) */
                __m256 s0 = _mm256_loadu_ps(Si0 + j);
                __m256 plus0 = _mm256_fmadd_ps(g, s0, xi0);
                __m256 minus0 = _mm256_fmadd_ps(ng, s0, xi0);
                _mm256_storeu_ps(Xi0 + 1 + j, plus0);
                _mm256_storeu_ps(Xi0 + 1 + L + j, minus0);

                /* Row 1: load, compute, store immediately */
                __m256 s1 = _mm256_loadu_ps(Si1 + j);
                __m256 plus1 = _mm256_fmadd_ps(g, s1, xi1);
                __m256 minus1 = _mm256_fmadd_ps(ng, s1, xi1);
                _mm256_storeu_ps(Xi1 + 1 + j, plus1);
                _mm256_storeu_ps(Xi1 + 1 + L + j, minus1);
            }

            /* Scalar tail for both rows */
            for (; j < L; ++j)
            {
                const float s0 = Si0[j];
                const float s1 = Si1[j];
                Xi0[1 + j] = x[i + 0] + gamma * s0;
                Xi0[1 + L + j] = x[i + 0] - gamma * s0;
                Xi1[1 + j] = x[i + 1] + gamma * s1;
                Xi1[1 + L + j] = x[i + 1] - gamma * s1;
            }
        }

        /* Handle last row if L is odd */
        if (i < L)
        {
            float *Xi = X + i * N;
            const float *Si = S + i * L;
            const __m256 xi8 = _mm256_set1_ps(x[i]);

            Xi[0] = x[i];

            size_t j = 0;
            for (; j + 7 < L; j += 8)
            {
                __m256 s8 = _mm256_loadu_ps(Si + j);
                __m256 plus = _mm256_fmadd_ps(g, s8, xi8);
                __m256 minus = _mm256_fmadd_ps(ng, s8, xi8);
                _mm256_storeu_ps(Xi + 1 + j, plus);
                _mm256_storeu_ps(Xi + 1 + L + j, minus);
            }

            for (; j < L; ++j)
            {
                const float s = Si[j];
                Xi[1 + j] = x[i] + gamma * s;
                Xi[1 + L + j] = x[i] - gamma * s;
            }
        }
        return;
    }
#endif

    /* ------------------------ Scalar fallback ------------------------ */
    for (size_t i = 0; i < L; ++i)
    {
        float *Xi = X + i * N;       //!< Row pointer of X (state i across columns)
        const float *Si = S + i * L; //!< Row pointer of S (state i across its L entries)

        Xi[0] = x[i]; //!< Mean in column 0

        for (size_t j = 0; j < L; ++j)
        {
            const float s = Si[j];
            Xi[1 + j] = x[i] + gamma * s;     //!< +γ column
            Xi[1 + L + j] = x[i] - gamma * s; //!< −γ column
        }
    }
}

/**
 * @brief Apply transition function F to all sigma points: X*[:, j] = F(X[:, j], u).
 *
 * @details
 *  Vectorized "batch-8" path packs 8 sigma columns into 8 contiguous L-length
 *  slices (SoA: k-major) so you can call F() on contiguous inputs:
 *      x[k*L + i] = X[i*N + (j+k)],  d[k*L + i] = F(x[k*L + :], u)[i]
 *  AVX2 is used only to load/store the 8 contiguous sigma entries per row i;
 *  since AVX2 lacks float scatters, we store the 8 lanes into a tiny stack
 *  buffer and perform 8 scalar lane stores to the SoA buffer.
 *
 *  Notes:
 *   - F typically dominates runtime; this path mainly reduces address
 *     arithmetic and improves cache behavior when L is large and N is big.
 *   - Falls back to scalar if allocation fails or N < 8.
 */
static void compute_transition_function(float Xstar[], const float X[], const float u[],
                                        void (*F)(float[], float[], float[]), uint8_t L8)
{
    const size_t L = (size_t)L8;
    const size_t N = 2u * L + 1u;

#if SQR_UKF_ENABLE_BATCH8
    if (ukf_has_avx2() && N >= 8)
    {
        /* SoA buffers: 8 states of length L each (k-major). */
        float *x = (float *)gemm_aligned_alloc(32, (size_t)8 * L * sizeof(float));
        float *d = (float *)gemm_aligned_alloc(32, (size_t)8 * L * sizeof(float));
        if (x && d)
        {
            const int do_pf = (L >= (size_t)UKF_TRANS_PF_MIN_L);
            const int rows_ahead = UKF_TRANS_PF_ROWS_AHEAD;

            /* process in batches of 8 sigmas */
            size_t j = 0;
            for (; j + 7 < N; j += 8)
            {
                /* pack 8 columns (j..j+7) into SoA */
                for (size_t i = 0; i < L; ++i)
                {
                    /* prefetch next row(s) of the same 8-sigma stripe */
                    if (do_pf && rows_ahead > 0)
                    {
                        for (int ra = 1; ra <= rows_ahead; ++ra)
                        {
                            const size_t ip = i + (size_t)ra;
                            if (ip < L)
                            {
                                _mm_prefetch((const char *)(&X[ip * N + j]), _MM_HINT_T0);
                                _mm_prefetch((const char *)(&Xstar[ip * N + j]), _MM_HINT_T0);
                            }
                        }
                    }

                    /* load 8 contiguous sigmas from row i */
                    __m256 v = _mm256_loadu_ps(&X[i * N + j]);

                    /* Scatter to SoA layout using UNALIGNED store to stack buffer */
                    float lanes[8];             // NO alignas - can't guarantee it!
                    _mm256_storeu_ps(lanes, v); // Use UNALIGNED store

#pragma GCC ivdep
                    for (int k = 0; k < 8; ++k)
                        x[(size_t)k * L + i] = lanes[k];
                }

                /* evaluate F on each contiguous L-vector */
                for (int k = 0; k < 8; ++k)
                    F(&d[(size_t)k * L], &x[(size_t)k * L], (float *)u);

                /* unpack back into 8 columns (j..j+7) */
                for (size_t i = 0; i < L; ++i)
                {
#pragma GCC ivdep
                    for (int k = 0; k < 8; ++k)
                        Xstar[i * N + (j + (size_t)k)] = d[(size_t)k * L + i];
                }
            }

            /* scalar tail for remaining sigmas */
            for (; j < N; ++j)
            {
                float *xk = x;
                float *dk = d;
                for (size_t i = 0; i < L; ++i)
                    xk[i] = X[i * N + j];
                F(dk, xk, (float *)u);
                for (size_t i = 0; i < L; ++i)
                    Xstar[i * N + j] = dk[i];
            }

            gemm_aligned_free(x);
            gemm_aligned_free(d);
            return;
        }
        if (x)
            gemm_aligned_free(x);
        if (d)
            gemm_aligned_free(d);
    }
#endif

    /* scalar fallback */
    float *xk = (float *)malloc(L * sizeof(float));
    float *dk = (float *)malloc(L * sizeof(float));
    if (!xk || !dk)
    {
        free(xk);
        free(dk);
        return;
    }

    for (size_t j = 0; j < N; ++j)
    {
        for (size_t i = 0; i < L; ++i)
            xk[i] = X[i * N + j];
        F(dk, xk, (float *)u);
        for (size_t i = 0; i < L; ++i)
            Xstar[i * N + j] = dk[i];
    }

    free(xk);
    free(dk);
}

/**
 * @brief Compute the weighted mean of sigma points.
 *
 * @details
 *  Given the sigma point matrix @p X (L × (2L+1)) and weight vector @p W ((2L+1) × 1),
 *  this function computes:
 *  \f[
 *      x_i = \sum_{j=0}^{2L} W_j \, X_{i,j}, \quad i = 0,\dots,L-1
 *  \f]
 *  which corresponds to the weighted mean of each state dimension.
 *
 *  - For small matrices or when AVX2 is unavailable, it falls back to a scalar loop.
 *  - For large matrices, it uses an AVX2/FMA optimized inner loop processing 8 weights at a time.
 *  - The SIMD path computes two rows (state dimensions) per iteration to maximize throughput.
 *
 * @param[out] x  Output mean vector of length L.
 * @param[in]  X  Sigma point matrix of shape (L × (2L+1)), row-major.
 * @param[in]  W  Weights vector of length (2L+1).
 * @param[in]  L  Number of state dimensions.
 *
 * @note
 *  The function automatically detects AVX2 support and falls back to portable scalar code.
 *  Prefetching hints are used to reduce memory stalls when L is large enough.
 */
#if LINALG_SIMD_ENABLE
/**
 * @brief Improved horizontal sum with better scheduling
 */
static inline float avx2_sum_ps_opt(__m256 v)
{
    /* Reduce to 128-bit */
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh); // [a+e, b+f, c+g, d+h]

    /* Horizontal add within 128-bit */
    __m128 shuf = _mm_movehdup_ps(vlow); // [b+f, b+f, d+h, d+h]
    vlow = _mm_add_ps(vlow, shuf);       // [a+b+e+f, *, c+d+g+h, *]
    shuf = _mm_movehl_ps(shuf, vlow);    // [c+d+g+h, *]
    vlow = _mm_add_ss(vlow, shuf);       // [a+b+c+d+e+f+g+h]

    return _mm_cvtss_f32(vlow);
}
#endif

static void multiply_sigma_point_matrix_to_weights(float x[],
                                                   const float X[],
                                                   const float W[],
                                                   uint8_t L)
{
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;

    /* Scalar fallback */
    if (!ukf_has_avx2() || N < 16)
    {
        for (size_t i = 0; i < Ls; ++i)
        {
            const float *row = &X[i * N];
            float acc = 0.0f;
            for (size_t j = 0; j < N; ++j)
                acc += W[j] * row[j];
            x[i] = acc;
        }
        return;
    }

#if LINALG_SIMD_ENABLE
    const int do_pf = (Ls >= (size_t)UKF_MEAN_PF_MIN_ROWS);
    const int rows_ahead = UKF_MEAN_PF_ROWS_AHEAD;

    /* Prefetch weights ONCE (reused for all rows) */
    if (do_pf)
    {
        for (size_t j = 0; j < N; j += 64)
            _mm_prefetch((const char *)(&W[j]), _MM_HINT_T0);
    }

    size_t i = 0;

    /* 4-way unrolling - processes 4 rows at a time */
    for (; i + 3 < Ls; i += 4)
    {
        const float *row0 = &X[(i + 0) * N];
        const float *row1 = &X[(i + 1) * N];
        const float *row2 = &X[(i + 2) * N];
        const float *row3 = &X[(i + 3) * N];

        /* Prefetch future rows */
        if (do_pf && rows_ahead > 0)
        {
            for (int ra = 1; ra <= rows_ahead; ++ra)
            {
                const size_t ip = i + (size_t)ra * 4;
                if (ip < Ls)
                {
                    _mm_prefetch((const char *)(&X[ip * N]), _MM_HINT_T0);
                    if (ip + 1 < Ls)
                        _mm_prefetch((const char *)(&X[(ip + 1) * N]), _MM_HINT_T0);
                    if (ip + 2 < Ls)
                        _mm_prefetch((const char *)(&X[(ip + 2) * N]), _MM_HINT_T0);
                    if (ip + 3 < Ls)
                        _mm_prefetch((const char *)(&X[(ip + 3) * N]), _MM_HINT_T0);
                }
            }
        }

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        size_t j = 0;

        /* Inner loop 2x unrolled to hide FMA latency */
        for (; j + 15 < N; j += 16)
        {
            /* First 8 elements */
            __m256 wv0 = _mm256_loadu_ps(&W[j]);
            __m256 x00 = _mm256_loadu_ps(&row0[j]);
            __m256 x10 = _mm256_loadu_ps(&row1[j]);
            __m256 x20 = _mm256_loadu_ps(&row2[j]);
            __m256 x30 = _mm256_loadu_ps(&row3[j]);

            /* Second 8 elements (start loading while FMAs execute) */
            __m256 wv1 = _mm256_loadu_ps(&W[j + 8]);
            __m256 x01 = _mm256_loadu_ps(&row0[j + 8]);
            __m256 x11 = _mm256_loadu_ps(&row1[j + 8]);

            /* First set of FMAs */
            acc0 = _mm256_fmadd_ps(wv0, x00, acc0);
            acc1 = _mm256_fmadd_ps(wv0, x10, acc1);

            /* Continue loading while FMAs execute */
            __m256 x21 = _mm256_loadu_ps(&row2[j + 8]);
            __m256 x31 = _mm256_loadu_ps(&row3[j + 8]);

            /* More FMAs */
            acc2 = _mm256_fmadd_ps(wv0, x20, acc2);
            acc3 = _mm256_fmadd_ps(wv0, x30, acc3);

            /* Second set of FMAs (different weight vector) */
            acc0 = _mm256_fmadd_ps(wv1, x01, acc0);
            acc1 = _mm256_fmadd_ps(wv1, x11, acc1);
            acc2 = _mm256_fmadd_ps(wv1, x21, acc2);
            acc3 = _mm256_fmadd_ps(wv1, x31, acc3);
        }

        /* Handle 8-element chunks */
        for (; j + 7 < N; j += 8)
        {
            __m256 wv = _mm256_loadu_ps(&W[j]);
            __m256 x0 = _mm256_loadu_ps(&row0[j]);
            __m256 x1 = _mm256_loadu_ps(&row1[j]);
            __m256 x2 = _mm256_loadu_ps(&row2[j]);
            __m256 x3 = _mm256_loadu_ps(&row3[j]);

            acc0 = _mm256_fmadd_ps(wv, x0, acc0);
            acc1 = _mm256_fmadd_ps(wv, x1, acc1);
            acc2 = _mm256_fmadd_ps(wv, x2, acc2);
            acc3 = _mm256_fmadd_ps(wv, x3, acc3);
        }

        /* Reduce accumulators */
        float sum0 = avx2_sum_ps_opt(acc0);
        float sum1 = avx2_sum_ps_opt(acc1);
        float sum2 = avx2_sum_ps_opt(acc2);
        float sum3 = avx2_sum_ps_opt(acc3);

        /* Scalar tail */
        for (; j < N; ++j)
        {
            const float w = W[j];
            sum0 += w * row0[j];
            sum1 += w * row1[j];
            sum2 += w * row2[j];
            sum3 += w * row3[j];
        }

        x[i + 0] = sum0;
        x[i + 1] = sum1;
        x[i + 2] = sum2;
        x[i + 3] = sum3;
    }

    /* 2-way unrolling for remaining rows */
    for (; i + 1 < Ls; i += 2)
    {
        const float *row0 = &X[(i + 0) * N];
        const float *row1 = &X[(i + 1) * N];

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        size_t j = 0;
        for (; j + 7 < N; j += 8)
        {
            __m256 wv = _mm256_loadu_ps(&W[j]);
            __m256 x0 = _mm256_loadu_ps(&row0[j]);
            __m256 x1 = _mm256_loadu_ps(&row1[j]);
            acc0 = _mm256_fmadd_ps(wv, x0, acc0);
            acc1 = _mm256_fmadd_ps(wv, x1, acc1);
        }

        float sum0 = avx2_sum_ps_opt(acc0);
        float sum1 = avx2_sum_ps_opt(acc1);

        for (; j < N; ++j)
        {
            const float w = W[j];
            sum0 += w * row0[j];
            sum1 += w * row1[j];
        }

        x[i + 0] = sum0;
        x[i + 1] = sum1;
    }

    /* Last row if L is odd */
    if (i < Ls)
    {
        const float *row = &X[i * N];
        __m256 acc = _mm256_setzero_ps();

        size_t j = 0;
        for (; j + 7 < N; j += 8)
        {
            __m256 wv = _mm256_loadu_ps(&W[j]);
            __m256 xv = _mm256_loadu_ps(&row[j]);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }

        float sum = avx2_sum_ps_opt(acc);
        for (; j < N; ++j)
            sum += W[j] * row[j];

        x[i] = sum;
    }
#endif
}

/**
 * @brief Build square-root state covariance S (SR-UKF) via QR of weighted deviations.
 *
 * @details
 *  Constructs the augmented matrix A′ (size M×L with M=3L) used by the SR-UKF:
 *
 *  Let X be the propagated sigma points (L × N, N = 2L+1), x the predicted mean (L),
 *  and W the covariance weights. Define:
 *   - K = 2L (number of deviation columns excluding the mean column),
 *   - w1 = sqrt(|W[1]|) (common absolute weight for all non-zero sigma columns),
 *   - w0 = sqrt(|W[0]|) (mean-deviation weight).
 *
 *  Then the columns of A′ are formed as:
 *   - Rows 0..K−1   :  w1 * (X[:, 1..N−1] − x)          (stacked by sigma index)
 *   - Rows K..M−1   :  Rsr                              (per-row copy of SR noise)
 *
 *  Next, compute R from a QR factorization (Householder) of A′:
 *     A′ = Q * R
 *  The upper L×L part of R is a square-root covariance (upper-triangular). Finally,
 *  perform a rank-1 Cholesky update/downdate with (X[:,0] − x), scaled by w0, using
 *  an **upper-triangular** routine:
 *     S ← cholupdate_upper(S,  w0*(X[:,0] − x), update=(W[0] ≥ 0))
 *
 *  Vectorization:
 *   - Deviations (X − x) are built in 8-wide chunks with AVX2 and scaled by w1.
 *   - The SR noise block uses 8-wide loads/stores (Rsr already factored).
 *   - The mean deviation vector b = w0*(X[:,0] − x) is built alongside.
 *
 *  Improvements vs. scalar:
 *   - Fewer loop-carried address computations and better cache residency.
 *   - Avoids forming identity/zero padding explicitly; builds only needed blocks.
 *   - Uses optimized @ref qr and an in-place @ref cholupdate_upper.
 *
 * @param[out] S     Output square-root covariance (L × L), **upper-triangular**.
 * @param[in]  W     Covariance weights, length N = 2L + 1.
 * @param[in]  X     Propagated sigma points (L × N), row-major.
 * @param[in]  x     Predicted state mean (L).
 * @param[in]  Rsr   Square-root of process/measurement noise (L × L), **upper-triangular**.
 * @param[in]  L8    Dimension L (stored as uint8_t to match surrounding API).
 *
 * @retval 0       Success.
 * @retval -ENOMEM Workspace allocation failed.
 * @retval -EIO    QR decomposition failed.
 * @retval -EFAULT Resulting S failed a simple PD sanity check (non-positive/NaN diagonal).
 *
 * @warning S, X, and Rsr must not alias. All matrices are row-major.
 * @note    This routine assumes an **upper** SR convention end-to-end.
 */
/**
 * @brief Build square-root state covariance S (SR-UKF) via QR of weighted deviations.
 *
 * @param[out] S       Output square-root covariance (L × L), upper-triangular.
 * @param[out] ws      Workspace structure (caller manages lifetime).
 * @param[in]  W       Covariance weights, length N = 2L + 1.
 * @param[in]  X       Propagated sigma points (L × N), row-major.
 * @param[in]  x       Predicted state mean (L).
 * @param[in]  Rsr     Square-root of process noise (L × L), upper-triangular.
 * @param[in]  L8      Dimension L.
 *
 * @retval 0       Success.
 * @retval -ENOMEM Workspace allocation failed.
 * @retval -EIO    QR decomposition failed.
 * @retval -EFAULT Resulting S has non-positive/NaN diagonal.
 */
static int create_state_estimation_error_covariance_matrix(
    float S[],
    ukf_qr_ws_t *ws,
    float W[],
    float X[],
    float x[],
    const float Rsr[],
    uint8_t L8)
{
    const size_t L = (size_t)L8;
    const size_t N = 2u * L + 1u;
    const size_t K = 2u * L; // Deviation columns
    const size_t M = 3u * L; // Augmented rows

    const float w1s = sqrtf(fabsf(W[1]));
    const float w0s = sqrtf(fabsf(W[0]));

    /* Ensure workspace is large enough */
    if (ukf_qr_ws_ensure(ws, L) != 0)
        return -ENOMEM;

    float *Aprime = ws->Aprime; // M×L column-major
    float *R_ = ws->R_;         // M×L
    float *b = ws->b;           // L

    const int do_pf = (L >= (size_t)UKF_APRIME_PF_MIN_L);
    const int rows_ahead = UKF_APRIME_PF_ROWS_AHEAD;

    /* ----------------------------------------------------------------
     * Build Aprime and b vector
     * ---------------------------------------------------------------- */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && L >= 16)
    {
        const __m256 w1v = _mm256_set1_ps(w1s);

        size_t i = 0;

        /* Process 2 state dimensions at a time */
        for (; i + 1 < L; i += 2)
        {
            const float *Xi0 = X + (i + 0) * N;
            const float *Xi1 = X + (i + 1) * N;
            const float xi0 = x[i + 0];
            const float xi1 = x[i + 1];

            /* Prefetch ahead */
            if (do_pf && rows_ahead > 0 && i + 2 < L)
            {
                _mm_prefetch((const char *)(X + (i + 2) * N), _MM_HINT_T0);
                _mm_prefetch((const char *)(Rsr + (i + 2) * L), _MM_HINT_T0);
            }

            /* Mean deviation: b[i] = w0s * (X[i,0] - x[i]) */
            b[i + 0] = w0s * (Xi0[0] - xi0);
            b[i + 1] = w0s * (Xi1[0] - xi1);

            /* Broadcast state means */
            const __m256 xi0v = _mm256_set1_ps(xi0);
            const __m256 xi1v = _mm256_set1_ps(xi1);

            /* ---- Deviations block: rows 0..K-1 ---- */
            size_t r = 0;
            for (; r + 7 < K; r += 8)
            {
                /* Load 8 sigma points for each state */
                __m256 Xv0 = _mm256_loadu_ps(&Xi0[r + 1]);
                __m256 Xv1 = _mm256_loadu_ps(&Xi1[r + 1]);

                /* Compute weighted deviations: w1 * (X - x) */
                __m256 diff0 = _mm256_sub_ps(Xv0, xi0v);
                __m256 diff1 = _mm256_sub_ps(Xv1, xi1v);
                __m256 out0 = _mm256_mul_ps(w1v, diff0);
                __m256 out1 = _mm256_mul_ps(w1v, diff1);

                /* Scatter to column-major Aprime using UNALIGNED stores */
                float lanes0[8], lanes1[8];
                _mm256_storeu_ps(lanes0, out0); // NOT aligned!
                _mm256_storeu_ps(lanes1, out1);

                /* Scalar scatter (unavoidable for column-major layout) */
                for (int k = 0; k < 8; ++k)
                {
                    Aprime[(r + k) * L + (i + 0)] = lanes0[k];
                    Aprime[(r + k) * L + (i + 1)] = lanes1[k];
                }
            }

            /* Scalar tail for deviations */
            for (; r < K; ++r)
            {
                Aprime[r * L + (i + 0)] = w1s * (Xi0[r + 1] - xi0);
                Aprime[r * L + (i + 1)] = w1s * (Xi1[r + 1] - xi1);
            }

            /* ---- SR noise block: rows K..M-1 ---- */
            const float *Rsr0 = Rsr + (i + 0) * L;
            const float *Rsr1 = Rsr + (i + 1) * L;

            size_t t = 0;
            for (; t + 7 < L; t += 8)
            {
                __m256 Sv0 = _mm256_loadu_ps(&Rsr0[t]);
                __m256 Sv1 = _mm256_loadu_ps(&Rsr1[t]);

                float lanes0[8], lanes1[8];
                _mm256_storeu_ps(lanes0, Sv0);
                _mm256_storeu_ps(lanes1, Sv1);

                for (int k = 0; k < 8; ++k)
                {
                    Aprime[(K + t + k) * L + (i + 0)] = lanes0[k];
                    Aprime[(K + t + k) * L + (i + 1)] = lanes1[k];
                }
            }

            /* Scalar tail for SR noise */
            for (; t < L; ++t)
            {
                Aprime[(K + t) * L + (i + 0)] = Rsr0[t];
                Aprime[(K + t) * L + (i + 1)] = Rsr1[t];
            }
        }

        /* Handle last state dimension if L is odd */
        if (i < L)
        {
            const float *Xi = X + i * N;
            const float xi = x[i];

            b[i] = w0s * (Xi[0] - xi);

            const __m256 xiv = _mm256_set1_ps(xi);

            /* Deviations */
            size_t r = 0;
            for (; r + 7 < K; r += 8)
            {
                __m256 Xv = _mm256_loadu_ps(&Xi[r + 1]);
                __m256 diff = _mm256_sub_ps(Xv, xiv);
                __m256 out = _mm256_mul_ps(w1v, diff);

                float lanes[8];
                _mm256_storeu_ps(lanes, out);

                for (int k = 0; k < 8; ++k)
                    Aprime[(r + k) * L + i] = lanes[k];
            }
            for (; r < K; ++r)
                Aprime[r * L + i] = w1s * (Xi[r + 1] - xi);

            /* SR noise */
            const float *Rsri = Rsr + i * L;
            size_t t = 0;
            for (; t + 7 < L; t += 8)
            {
                __m256 Sv = _mm256_loadu_ps(&Rsri[t]);

                float lanes[8];
                _mm256_storeu_ps(lanes, Sv);

                for (int k = 0; k < 8; ++k)
                    Aprime[(K + t + k) * L + i] = lanes[k];
            }
            for (; t < L; ++t)
                Aprime[(K + t) * L + i] = Rsri[t];
        }
    }
    else
#endif
    {
        /* Scalar path - cleaner and more readable */
        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X + i * N;
            const float xi = x[i];

            /* Prefetch */
            if (do_pf && rows_ahead > 0 && i + 1 < L)
            {
                _mm_prefetch((const char *)(X + (i + 1) * N), _MM_HINT_T0);
                _mm_prefetch((const char *)(Rsr + (i + 1) * L), _MM_HINT_T0);
            }

            b[i] = w0s * (Xi[0] - xi);

            /* Deviations */
            for (size_t r = 0; r < K; ++r)
                Aprime[r * L + i] = w1s * (Xi[r + 1] - xi);

            /* SR noise */
            const float *Rsri = Rsr + i * L;
            for (size_t t = 0; t < L; ++t)
                Aprime[(K + t) * L + i] = Rsri[t];
        }
    }

    /* ----------------------------------------------------------------
     * QR decomposition: Aprime = Q * R (we only need R)
     * ---------------------------------------------------------------- */
    if (qr(Aprime, NULL, R_, (uint16_t)M, (uint16_t)L, true) != 0)
        return -EIO;

    /* Extract upper L×L block of R into S */
    memcpy(S, R_, L * L * sizeof(float));

    /* ----------------------------------------------------------------
     * Cholesky rank-1 update/downdate with mean deviation
     * ---------------------------------------------------------------- */
    int rc = cholupdate(S, b, (uint16_t)L, true, (W[0] >= 0.0f));
    if (rc != 0)
        return rc;

    /* ----------------------------------------------------------------
     * Sanity check: S must be SPD (positive definite diagonal)
     * ---------------------------------------------------------------- */
    for (size_t i = 0; i < L; ++i)
    {
        const float sii = S[i * L + i];
        if (!(sii > 0.0f && isfinite(sii)))
            return -EFAULT;
    }

    return 0;
}

/**
 * @brief Identity observation model: Y = X.
 *
 * @details
 *  Copies Y := X for the sigma matrix. Implemented as a single memcpy since
 *  row-major layouts are identical.
 *
 * @param[out] Y  Observation sigma matrix [L x N], row-major.
 * @param[in]  X  State sigma matrix [L x N], row-major.
 * @param[in]  L  Dimension (N=2L+1).
 */
static void H(float Y[], float X[], uint8_t L)
{
    const uint16_t N = (uint16_t)(2 * L + 1);
    memcpy(Y, X, (size_t)L * N * sizeof(float));
}

static inline size_t ukf_round_up8(size_t n) { return (n + 7u) & ~7u; }

#if LINALG_SIMD_ENABLE
static inline void ukf_transpose8x8_ps(__m256 in[8], __m256 out[8])
{
    __m256 t0 = _mm256_unpacklo_ps(in[0], in[1]);
    __m256 t1 = _mm256_unpackhi_ps(in[0], in[1]);
    __m256 t2 = _mm256_unpacklo_ps(in[2], in[3]);
    __m256 t3 = _mm256_unpackhi_ps(in[2], in[3]);
    __m256 t4 = _mm256_unpacklo_ps(in[4], in[5]);
    __m256 t5 = _mm256_unpackhi_ps(in[4], in[5]);
    __m256 t6 = _mm256_unpacklo_ps(in[6], in[7]);
    __m256 t7 = _mm256_unpackhi_ps(in[6], in[7]);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x4E);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xB1);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x4E);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xB1);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x4E);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xB1);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x4E);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xB1);

    out[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    out[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    out[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    out[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    out[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    out[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    out[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    out[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}
#endif

/**
 * @brief Compute cross-covariance Pxy = X_c · diag(W) · Y_c^T without materializing diag(W).
 *
 * @param[out] P   Output cross-covariance (L × L), row-major.
 * @param[in]  W   Weights vector of length N = 2L + 1.
 * @param[in]  X   Sigma matrix for state (L × N), row-major.
 * @param[in]  Y   Sigma matrix for measurement (L × N), row-major.
 * @param[in]  x   Mean of X (length L).
 * @param[in]  y   Mean of Y (length L).
 * @param[in]  L8  Dimension L.
 *
 * @retval 0       Success.
 * @retval -ENOMEM Memory allocation failed.
 * @retval -EIO    Matrix multiplication failed.
 *
 * @note Non-destructive: X and Y are not modified.
 */
static int create_state_cross_covariance_matrix(float *RESTRICT P,
                                                const float *RESTRICT W,
                                                const float *RESTRICT X,
                                                const float *RESTRICT Y,
                                                const float *RESTRICT x,
                                                const float *RESTRICT y,
                                                uint8_t L8)
{
    const size_t L = (size_t)L8;
    const size_t N = 2u * L + 1u;
    const size_t N8 = ukf_round_up8(N);

    /* Zero output */
    memset(P, 0, L * L * sizeof(float));

    /* Allocate temporaries:
       - Xc:  L × N8   (weighted centered X)
       - YTc: N8 × L   (centered Y transpose)
    */
    float *Xc = (float *)gemm_aligned_alloc(32, L * N8 * sizeof(float));
    float *YTc = (float *)gemm_aligned_alloc(32, N8 * L * sizeof(float));
    if (!Xc || !YTc)
    {
        if (Xc)
            gemm_aligned_free(Xc);
        if (YTc)
            gemm_aligned_free(YTc);
        return -ENOMEM;
    }

    const int do_pf = (L >= (size_t)UKF_PXY_PF_MIN_L);

    /* ----------------------------------------------------------------
     * Step 1: Build Xc = (X - x) ⊙ W  (element-wise weighted centering)
     * ---------------------------------------------------------------- */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && N >= 16)
    {
        size_t i = 0;

        /* Process 2 rows at a time */
        for (; i + 1 < L; i += 2)
        {
            const float *Xi0 = X + (i + 0) * N;
            const float *Xi1 = X + (i + 1) * N;
            float *Xci0 = Xc + (i + 0) * N8;
            float *Xci1 = Xc + (i + 1) * N8;

            const __m256 xi0v = _mm256_set1_ps(x[i + 0]);
            const __m256 xi1v = _mm256_set1_ps(x[i + 1]);

            /* Prefetch next 2 rows */
            if (do_pf && i + 2 < L)
            {
                _mm_prefetch((const char *)(X + (i + 2) * N), _MM_HINT_T0);
                _mm_prefetch((const char *)(X + (i + 3) * N), _MM_HINT_T0);
            }

            size_t j = 0;
            for (; j + 7 < N; j += 8)
            {
                __m256 wv = _mm256_loadu_ps(W + j);

                /* Row 0: (X - x) * W */
                __m256 xv0 = _mm256_loadu_ps(Xi0 + j);
                __m256 diff0 = _mm256_sub_ps(xv0, xi0v);
                __m256 res0 = _mm256_mul_ps(diff0, wv);
                _mm256_storeu_ps(Xci0 + j, res0);

                /* Row 1: (X - x) * W */
                __m256 xv1 = _mm256_loadu_ps(Xi1 + j);
                __m256 diff1 = _mm256_sub_ps(xv1, xi1v);
                __m256 res1 = _mm256_mul_ps(diff1, wv);
                _mm256_storeu_ps(Xci1 + j, res1);
            }

            /* Scalar tail and padding */
            for (; j < N; ++j)
            {
                Xci0[j] = (Xi0[j] - x[i + 0]) * W[j];
                Xci1[j] = (Xi1[j] - x[i + 1]) * W[j];
            }
            for (; j < N8; ++j)
            {
                Xci0[j] = 0.0f;
                Xci1[j] = 0.0f;
            }
        }

        /* Handle last row if L is odd */
        if (i < L)
        {
            const float *Xi = X + i * N;
            float *Xci = Xc + i * N8;
            const __m256 xiv = _mm256_set1_ps(x[i]);

            size_t j = 0;
            for (; j + 7 < N; j += 8)
            {
                __m256 wv = _mm256_loadu_ps(W + j);
                __m256 xv = _mm256_loadu_ps(Xi + j);
                __m256 diff = _mm256_sub_ps(xv, xiv);
                _mm256_storeu_ps(Xci + j, _mm256_mul_ps(diff, wv));
            }
            for (; j < N; ++j)
                Xci[j] = (Xi[j] - x[i]) * W[j];
            for (; j < N8; ++j)
                Xci[j] = 0.0f;
        }
    }
    else
#endif
    {
        /* Scalar fallback */
        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X + i * N;
            float *Xci = Xc + i * N8;
            const float xi = x[i];

            if (do_pf && i + 1 < L)
                _mm_prefetch((const char *)(X + (i + 1) * N), _MM_HINT_T0);

            size_t j = 0;
            for (; j < N; ++j)
                Xci[j] = (Xi[j] - xi) * W[j];
            for (; j < N8; ++j)
                Xci[j] = 0.0f;
        }
    }

    /* ----------------------------------------------------------------
     * Step 2: Build YTc = (Y - y)^T with row-padding to N8
     * ---------------------------------------------------------------- */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && N >= 8 && L >= 16)
    {
        /* Zero-pad extra rows */
        for (size_t jp = N; jp < N8; ++jp)
            memset(YTc + jp * L, 0, L * sizeof(float));

        /* Process Y in 8-row panels using transpose kernel */
        size_t k0 = 0;
        for (; k0 + 7 < L; k0 += 8)
        {
            /* Prefetch next panel */
            if (do_pf && k0 + 8 < L)
                _mm_prefetch((const char *)(Y + (k0 + 8) * N), _MM_HINT_T0);

            size_t j0 = 0;
            for (; j0 + 7 < N; j0 += 8)
            {
                /* Load 8x8 tile from Y and center by y */
                __m256 row[8];
                for (int r = 0; r < 8; ++r)
                {
                    const float *Yr = Y + (k0 + (size_t)r) * N + j0;
                    __m256 yr = _mm256_set1_ps(y[k0 + (size_t)r]);
                    row[r] = _mm256_sub_ps(_mm256_loadu_ps(Yr), yr);
                }

                /* Transpose 8x8 and store to YTc */
                __m256 col[8];
                ukf_transpose8x8_ps(row, col);

                for (int c = 0; c < 8; ++c)
                    _mm256_storeu_ps(YTc + (j0 + (size_t)c) * L + k0, col[c]);
            }

            /* Scalar tail for this 8-row panel */
            for (; j0 < N; ++j0)
            {
                float *YTrow = YTc + j0 * L;
                for (int r = 0; r < 8; ++r)
                    YTrow[k0 + (size_t)r] = Y[(k0 + (size_t)r) * N + j0] - y[k0 + (size_t)r];
            }
        }

        /* Handle remaining rows (L % 8 tail) */
        for (; k0 < L; ++k0)
        {
            const float yk = y[k0];
            size_t j = 0;
            for (; j < N; ++j)
                YTc[j * L + k0] = Y[k0 * N + j] - yk;
            for (; j < N8; ++j)
                YTc[j * L + k0] = 0.0f;
        }
    }
    else
#endif
    {
        /* Scalar fallback: build Y^T directly */
        for (size_t j = 0; j < N; ++j)
        {
            float *YTrow = YTc + j * L;
            for (size_t k = 0; k < L; ++k)
                YTrow[k] = Y[k * N + j] - y[k];
        }
        /* Zero-pad extra rows */
        for (size_t j = N; j < N8; ++j)
            memset(YTc + j * L, 0, L * sizeof(float));
    }

    /* ----------------------------------------------------------------
     * Step 3: Matrix multiply P = Xc · YTc
     * ---------------------------------------------------------------- */
    int rc = mul(P, Xc, YTc, (uint16_t)L, (uint16_t)N8, (uint16_t)N8, (uint16_t)L);

    /* Cleanup */
    gemm_aligned_free(Xc);
    gemm_aligned_free(YTc);

    return rc;
}

/**
 * @brief Measurement update: compute Kalman gain, update state, and downdate SR covariance.
 *
 * @details
 *  Solves the linear system for the Kalman gain without forming any explicit inverses.
 *  With the measurement SR factor @p Sy (upper-triangular) and cross-covariance @p Pxy:
 *
 *  1) Forward solve (lower):   \( S_y^\top Z = P_{xy} \)
 *  2) Backward solve (upper):  \( S_y K     = Z \)     (in-place: Z becomes K)
 *
 *  Then:
 *   - \( \delta y = y - \hat{y} \)
 *   - \( K \delta y \) is accumulated into @p xhat
 *   - \( U = K S_y \) is formed and @p S is downdated via rank-1 Cholesky for each column of U:
 *       \( S \leftarrow \mathrm{cholupdate\_upper}(S, U_{\cdot j}, \mathrm{update}=false) \)
 *
 *  Vectorization:
 *   - AVX2/FMA AXPY-like updates inside both triangular solves, blocked over RHS columns
 *     (size controlled by UKF_UPD_COLBLOCK).
 *   - AVX2 used for building \( \delta y \) and for accumulating @p xhat.
 *   - Prefetching along RHS panels to reduce cache miss latency when n is large.
 *
 *  Conventions:
 *   - @p Sy is **upper-triangular** (SR of the measurement covariance).
 *   - @p S is **upper-triangular** (SR of the state covariance) and is downdated in-place.
 *
 * @param[in,out] S     State SR covariance (n × n), upper-triangular, updated in-place (downdated).
 * @param[in,out] xhat  State estimate (length n); on return, \( \hat{x}^+ = \hat{x} + K (y-\hat{y}) \).
 * @param[in]     yhat  Predicted measurement (length n).
 * @param[in]     y     Actual measurement (length n).
 * @param[in]     Sy    Measurement SR covariance (n × n), **upper-triangular**.
 * @param[in]     Pxy   Cross-covariance between state and measurement (n × n).
 * @param[in]     L8    Dimension n (stored as uint8_t to match surrounding API).
 *
 * @retval 0        Success.
 * @retval -ENOMEM  Workspace allocation failed.
 * @retval -EIO     Underlying GEMM (mul) failed.
 *
 * @note
 *  Uses a thread-local workspace (see ukf_upd_ws_t). All matrices are row-major.
 */
/**
 * @brief In-place matrix transpose (square matrices only)
 *
 * @param A     Square matrix (n×n), row-major, transposed in-place
 * @param n     Dimension
 *
 * @note Uses register blocking for cache efficiency
 */
static void transpose_square_inplace(float *A, uint16_t n)
{
    /* Block size for cache-friendly access */
    const uint16_t BS = 8;

#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && n >= 16)
    {
        /* Blocked transpose with AVX2 8x8 kernel */
        for (uint16_t i0 = 0; i0 < n; i0 += BS)
        {
            for (uint16_t j0 = i0; j0 < n; j0 += BS) // Note: j0 starts at i0 (upper triangle)
            {
                const uint16_t imax = MIN(i0 + BS, n);
                const uint16_t jmax = MIN(j0 + BS, n);

                if (i0 == j0)
                {
                    /* Diagonal block - transpose in-place */
                    for (uint16_t i = i0; i < imax; ++i)
                    {
                        for (uint16_t j = i + 1; j < jmax; ++j)
                        {
                            float tmp = A[i * n + j];
                            A[i * n + j] = A[j * n + i];
                            A[j * n + i] = tmp;
                        }
                    }
                }
                else
                {
                    /* Off-diagonal block - swap entire blocks */
                    for (uint16_t i = i0; i < imax; ++i)
                    {
                        for (uint16_t j = j0; j < jmax; ++j)
                        {
                            float tmp = A[i * n + j];
                            A[i * n + j] = A[j * n + i];
                            A[j * n + i] = tmp;
                        }
                    }
                }
            }
        }
        return;
    }
#endif

    /* Scalar fallback */
    for (uint16_t i = 0; i < n; ++i)
    {
        for (uint16_t j = i + 1; j < n; ++j)
        {
            float tmp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = tmp;
        }
    }
}

/**
 * @brief Measurement update: compute Kalman gain, update state, and downdate SR covariance.
 *
 * @details
 *  Solves for Kalman gain K via triangular solves (no matrix inverse):
 *    1) Forward solve:  Sy^T · Z = Pxy  (lower triangular)
 *    2) Backward solve: Sy · K = Z      (upper triangular, Z overwritten with K)
 *
 *  Then applies measurement update:
 *    - x̂ ← x̂ + K·(y − ŷ)
 *    - S ← downdate(S, U) where U = K·Sy, via n rank-1 Cholesky downdates
 *
 *  **Optimizations:**
 *   - Transpose U once using tran_tiled (32×32 blocked AVX2)
 *   - Contiguous row access eliminates strided loads (~1.5-2x faster)
 *   - Cholupdate uses workspace (no malloc overhead)
 *   - Automatic cache blocking for n ≥ 256
 *   - 16-wide AVX2 processing with register transpose for downdates
 *
 * @param[in,out] S     State SR covariance (n×n), upper-triangular, downdated in-place.
 * @param[in,out] xhat  State estimate (n); updated to x̂^+ = x̂ + K(y−ŷ).
 * @param[in]     yhat  Predicted measurement (n).
 * @param[in]     y     Actual measurement (n).
 * @param[in]     Sy    Measurement SR covariance (n×n), upper-triangular.
 * @param[in]     Pxy   Cross-covariance (n×n).
 * @param[in,out] ws    Workspace structure (caller manages).
 * @param[in]     L8    Dimension n.
 *
 * @retval 0        Success.
 * @retval -ENOMEM  Workspace allocation failed.
 * @retval -EIO     GEMM operation failed.
 * @retval -EDOM    Cholesky downdate failed (filter divergence detected).
 *
 * @note All matrices are row-major. Sy and S are upper-triangular.
 */
static int update_state_covariance_matrix_and_state_estimation_vector(
    float *RESTRICT S,
    float *RESTRICT xhat,
    const float *RESTRICT yhat,
    const float *RESTRICT y,
    const float *RESTRICT Sy,
    const float *RESTRICT Pxy,
    ukf_upd_ws_t *ws,
    uint8_t L8)
{
    const uint16_t n = (uint16_t)L8;
    const size_t nn = (size_t)n * (size_t)n;

    /* Ensure workspace is adequate */
    if (ukf_upd_ws_ensure(ws, n) != 0)
        return -ENOMEM;

    float *Z = ws->Z;         /* n×n RHS workspace → becomes K */
    float *U = ws->U;         /* n×n temporary for K·Sy */
    float *Ut = ws->Ut;       /* n×n transposed U */
    float *Ky = ws->Ky;       /* n-vector: K·(y−ŷ) */
    float *yyhat = ws->yyhat; /* n-vector: y − ŷ */

    /* Initialize Z with Pxy (will be overwritten by K after solves) */
    memcpy(Z, Pxy, nn * sizeof(float));

    /* Prefetch knobs */
    const int do_pf = (n >= (uint16_t)UKF_UPD_PF_MIN_N);
    const size_t pf_elts = (size_t)UKF_UPD_PF_DIST_BYTES / sizeof(float);

    /* ==================================================================
     * STEP 1: Forward solve  Sy^T · Z = Pxy
     *         (Sy is upper ⇒ Sy^T is lower triangular)
     * ================================================================== */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && n >= 8)
    {
        for (uint16_t i = 0; i < n; ++i)
        {
            const float sii = Sy[(size_t)i * n + i];

            for (uint16_t c0 = 0; c0 < n; c0 += UKF_UPD_COLBLOCK)
            {
                const uint16_t bc = MIN((uint16_t)UKF_UPD_COLBLOCK, n - c0);

                /* Z[i,:] -= Σ_{k<i} Sy[k,i] · Z[k,:] */
                for (uint16_t k = 0; k < i; ++k)
                {
                    const float m = Sy[(size_t)k * n + i];
                    if (m == 0.0f)
                        continue;

                    const __m256 mv = _mm256_set1_ps(m);
                    uint16_t c = 0;

                    for (; c + 7 < bc; c += 8)
                    {
                        if (do_pf && c + pf_elts + 8 < bc)
                        {
                            _mm_prefetch((const char *)(&Z[(size_t)i * n + c0 + c + pf_elts]), _MM_HINT_T0);
                            _mm_prefetch((const char *)(&Z[(size_t)k * n + c0 + c + pf_elts]), _MM_HINT_T0);
                        }

                        __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c0 + c]);
                        __m256 zk = _mm256_loadu_ps(&Z[(size_t)k * n + c0 + c]);
                        zi = _mm256_fnmadd_ps(mv, zk, zi);
                        _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], zi);
                    }

                    for (; c < bc; ++c)
                        Z[(size_t)i * n + c0 + c] -= m * Z[(size_t)k * n + c0 + c];
                }

                /* Z[i,:] /= Sy[i,i] */
                const __m256 rinv = _mm256_set1_ps(1.0f / sii);
                uint16_t c = 0;

                for (; c + 7 < bc; c += 8)
                {
                    __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c0 + c]);
                    _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], _mm256_mul_ps(zi, rinv));
                }

                for (; c < bc; ++c)
                    Z[(size_t)i * n + c0 + c] /= sii;
            }
        }
    }
    else
#endif
    {
        /* Scalar forward solve */
        for (uint16_t i = 0; i < n; ++i)
        {
            const float sii = Sy[(size_t)i * n + i];

            for (uint16_t k = 0; k < i; ++k)
            {
                const float m = Sy[(size_t)k * n + i];
                for (uint16_t c = 0; c < n; ++c)
                    Z[(size_t)i * n + c] -= m * Z[(size_t)k * n + c];
            }

            for (uint16_t c = 0; c < n; ++c)
                Z[(size_t)i * n + c] /= sii;
        }
    }

    /* ==================================================================
     * STEP 2: Backward solve  Sy · K = Z
     *         (Sy is upper triangular; Z is overwritten with K)
     * ================================================================== */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && n >= 8)
    {
        for (int i = (int)n - 1; i >= 0; --i)
        {
            const float sii = Sy[(size_t)i * n + i];

            for (uint16_t c0 = 0; c0 < n; c0 += UKF_UPD_COLBLOCK)
            {
                const uint16_t bc = MIN((uint16_t)UKF_UPD_COLBLOCK, n - c0);

                /* Z[i,:] -= Σ_{k>i} Sy[i,k] · Z[k,:] */
                for (uint16_t k = (uint16_t)(i + 1); k < n; ++k)
                {
                    const float m = Sy[(size_t)i * n + k];
                    if (m == 0.0f)
                        continue;

                    const __m256 mv = _mm256_set1_ps(m);
                    uint16_t c = 0;

                    for (; c + 7 < bc; c += 8)
                    {
                        if (do_pf && c + pf_elts + 8 < bc)
                        {
                            _mm_prefetch((const char *)(&Z[(size_t)i * n + c0 + c + pf_elts]), _MM_HINT_T0);
                            _mm_prefetch((const char *)(&Z[(size_t)k * n + c0 + c + pf_elts]), _MM_HINT_T0);
                        }

                        __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c0 + c]);
                        __m256 zk = _mm256_loadu_ps(&Z[(size_t)k * n + c0 + c]);
                        zi = _mm256_fnmadd_ps(mv, zk, zi);
                        _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], zi);
                    }

                    for (; c < bc; ++c)
                        Z[(size_t)i * n + c0 + c] -= m * Z[(size_t)k * n + c0 + c];
                }

                /* Z[i,:] /= Sy[i,i] */
                const __m256 rinv = _mm256_set1_ps(1.0f / sii);
                uint16_t c = 0;

                for (; c + 7 < bc; c += 8)
                {
                    __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c0 + c]);
                    _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], _mm256_mul_ps(zi, rinv));
                }

                for (; c < bc; ++c)
                    Z[(size_t)i * n + c0 + c] /= sii;
            }
        }
    }
    else
#endif
    {
        /* Scalar backward solve */
        for (int i = (int)n - 1; i >= 0; --i)
        {
            const float sii = Sy[(size_t)i * n + i];

            for (uint16_t k = (uint16_t)(i + 1); k < n; ++k)
            {
                const float m = Sy[(size_t)i * n + k];
                for (uint16_t c = 0; c < n; ++c)
                    Z[(size_t)i * n + c] -= m * Z[(size_t)k * n + c];
            }

            for (uint16_t c = 0; c < n; ++c)
                Z[(size_t)i * n + c] /= sii;
        }
    }
    /* Z now contains K (the Kalman gain) */

    /* ==================================================================
     * STEP 3: Compute innovation  v = y − ŷ
     * ================================================================== */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && n >= 8)
    {
        uint16_t i = 0;
        for (; i + 7 < n; i += 8)
        {
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 vyh = _mm256_loadu_ps(yhat + i);
            _mm256_storeu_ps(yyhat + i, _mm256_sub_ps(vy, vyh));
        }
        for (; i < n; ++i)
            yyhat[i] = y[i] - yhat[i];
    }
    else
#endif
    {
        for (uint16_t i = 0; i < n; ++i)
            yyhat[i] = y[i] - yhat[i];
    }

    /* ==================================================================
     * STEP 4: Compute Ky = K · (y − ŷ)
     * ================================================================== */
    if (mul(Ky, Z /*K*/, yyhat, n, n, n, 1) != 0)
        return -EIO;

    /* ==================================================================
     * STEP 5: State update  x̂ ← x̂ + Ky
     * ================================================================== */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && n >= 8)
    {
        uint16_t i = 0;
        for (; i + 7 < n; i += 8)
        {
            __m256 xv = _mm256_loadu_ps(xhat + i);
            __m256 kv = _mm256_loadu_ps(Ky + i);
            _mm256_storeu_ps(xhat + i, _mm256_add_ps(xv, kv));
        }
        for (; i < n; ++i)
            xhat[i] += Ky[i];
    }
    else
#endif
    {
        for (uint16_t i = 0; i < n; ++i)
            xhat[i] += Ky[i];
    }

    /* ==================================================================
     * STEP 6: Compute U = K · Sy
     * ================================================================== */
    if (mul(U, Z /*K*/, Sy, n, n, n, n) != 0)
        return -EIO;

    /* ==================================================================
     * STEP 7: Transpose U using optimized tiled transpose
     *         (Avoids O(n²) strided column extractions)
     * ================================================================== */
    tran_tiled(Ut, U, n, n);

    /* ==================================================================
     * STEP 8: Downdate S by each row of U^T
     *         (Uses workspace cholupdate for zero malloc overhead)
     * ================================================================== */
    for (uint16_t j = 0; j < n; ++j)
    {
        /* Prefetch next row */
        if (do_pf && j + 2 < n)
            _mm_prefetch((const char *)(Ut + (j + 2) * n), _MM_HINT_T0);

        /* Row j of U^T is now contiguous (was column j of U) */
        const float *Utj = Ut + (size_t)j * n;

        /* Copy to cholupdate workspace buffer (downdate modifies in-place) */
        memcpy(ws->chol_ws->xbuf, Utj, (size_t)n * sizeof(float));

        /* Apply optimized rank-1 downdate:
         * - 16-wide AVX2 with register transpose (lower-tri)
         * - Automatic cache blocking for n ≥ 256
         * - Zero malloc overhead (uses workspace)
         */
        int rc = cholupdate_rank1_downdate(S, ws->chol_ws->xbuf, n, /*is_upper=*/true);
        if (rc != 0)
        {
            /* Filter divergence detected - provide diagnostic info */
            /* In production, you might log which column failed:
               fprintf(stderr, "UKF divergence: cholupdate failed at column %u\n", j);
            */
            return rc;
        }
    }

    return 0;
}

/**
 * @brief Square-root Unscented Kalman Filter (SR-UKF) step (predict + update).
 *
 * @details
 *  Orchestrates the SR-UKF cycle using vectorized kernels:
 *   - Weights, sigma generation, propagation with F, weighted mean,
 *     SR covariance via QR, identity H, measurement prediction,
 *     measurement SR covariance, cross-cov, and update via triangular solves
 *     + Cholesky downdates.
 *
 *  All workspace management is explicit (no thread-local storage).
 *  Proper error propagation throughout the pipeline.
 *
 * @param[in]     y       Measurement vector [L].
 * @param[in,out] xhat    State mean [L]; on return the updated state estimate.
 * @param[in]     Rn_sr   Measurement noise SR covariance [L×L], upper-triangular.
 * @param[in]     Rv_sr   Process noise SR covariance [L×L], upper-triangular.
 * @param[in]     u       Control/input vector passed to F.
 * @param[in]     F       Transition function: F(dx, x, u).
 * @param[in,out] S       State SR covariance [L×L], upper-triangular, updated in-place.
 * @param[in]     alpha   UKF spread parameter (typically 1e-3).
 * @param[in]     beta    Prior knowledge parameter (typically 2.0 for Gaussian).
 * @param[in]     L8      State dimension.
 *
 * @retval 0        Success.
 * @retval -EINVAL  Invalid parameter (L==0).
 * @retval -ENOMEM  Memory allocation failed.
 * @retval -EIO     Internal operation (QR, GEMM) failed.
 * @retval -EDOM    Cholesky update/downdate failed (numerical issue).
 * @retval -EFAULT  Covariance matrix failed sanity check (non-PD).
 *
 * @note All matrices are row-major. S, Rn_sr, Rv_sr are upper-triangular.
 */
int sqr_ukf(float y[],
            float xhat[],
            const float Rn_sr[],
            const float Rv_sr[],
            float u[],
            void (*F)(float[], float[], float[]),
            float S[],
            float alpha,
            float beta,
            uint8_t L8)
{
    if (L8 == 0)
        return -EINVAL;

    int status = 0;
    const uint16_t L = L8;
    const uint16_t N = (uint16_t)(2 * L + 1);

    /* Allocation sizes */
    const size_t szW = (size_t)N * sizeof(float);
    const size_t szLN = (size_t)L * N * sizeof(float);
    const size_t szLL = (size_t)L * L * sizeof(float);
    const size_t szL = (size_t)L * sizeof(float);

    /* Allocate main working arrays */
    float *Wc = (float *)gemm_aligned_alloc(32, szW);
    float *Wm = (float *)gemm_aligned_alloc(32, szW);
    float *X = (float *)gemm_aligned_alloc(32, szLN);
    float *Xst = (float *)gemm_aligned_alloc(32, szLN);
    float *Y = (float *)gemm_aligned_alloc(32, szLN);
    float *yhat = (float *)gemm_aligned_alloc(32, szL);
    float *Sy = (float *)gemm_aligned_alloc(32, szLL);
    float *Pxy = (float *)gemm_aligned_alloc(32, szLL);

    if (!Wc || !Wm || !X || !Xst || !Y || !yhat || !Sy || !Pxy)
    {
        status = -ENOMEM;
        goto Cleanup;
    }

    /* Workspace structures (stack-allocated handles) */
    ukf_qr_ws_t qr_ws = {0};   /* For QR decomposition in covariance steps */
    ukf_upd_ws_t upd_ws = {0}; /* For measurement update triangular solves */

    const float kappa = 0.0f;

    /* ==================================================================
     * PREDICTION PHASE
     * ================================================================== */

    /* 1. Create UKF weights */
    create_weights(Wc, Wm, alpha, beta, kappa, (uint8_t)L);

    /* 2. Generate sigma points from current state */
    create_sigma_point_matrix(X, xhat, S, alpha, kappa, (uint8_t)L);

    /* 3. Propagate sigma points through nonlinear dynamics */
    compute_transition_function(Xst, X, u, F, (uint8_t)L);

    /* 4. Compute predicted state mean */
    multiply_sigma_point_matrix_to_weights(xhat, Xst, Wm, (uint8_t)L);

    /* 5. Compute predicted state SR covariance */
    {
        int rc = create_state_estimation_error_covariance_matrix(
            S, &qr_ws, Wc, Xst, xhat, Rv_sr, (uint8_t)L);
        if (rc != 0)
        {
            status = rc;
            goto Cleanup;
        }
    }

    /* ==================================================================
     * UPDATE PHASE
     * ================================================================== */

    /* 6. Generate new sigma points from predicted state */
    create_sigma_point_matrix(X, xhat, S, alpha, kappa, (uint8_t)L);

    /* 7. Apply measurement model (identity: Y = X) */
    H(Y, X, (uint8_t)L);

    /* 8. Compute predicted measurement mean */
    multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, (uint8_t)L);

    /* 9. Compute measurement SR covariance */
    {
        int rc = create_state_estimation_error_covariance_matrix(
            Sy, &qr_ws, Wc, Y, yhat, Rn_sr, (uint8_t)L);
        if (rc != 0)
        {
            status = rc;
            goto Cleanup;
        }
    }

    /* 10. Compute cross-covariance Pxy */
    {
        int rc = create_state_cross_covariance_matrix(
            Pxy, Wc, X, Y, xhat, yhat, (uint8_t)L);
        if (rc != 0)
        {
            status = rc;
            goto Cleanup;
        }
    }

    /* 11. Measurement update: compute gain, update state and SR covariance */
    {
        int rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y, Sy, Pxy, &upd_ws, (uint8_t)L);
        if (rc != 0)
        {
            status = rc;
            goto Cleanup;
        }
    }

Cleanup:
    /* Free main working arrays */
    gemm_aligned_free(Wc);
    gemm_aligned_free(Wm);
    gemm_aligned_free(X);
    gemm_aligned_free(Xst);
    gemm_aligned_free(Y);
    gemm_aligned_free(yhat);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);

    /* Workspace structures clean up their own internal allocations */
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);

    return status;
}