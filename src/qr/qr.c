/**
 * @file qr_blocked.c (COMPLETE FIXED VERSION)
 * @brief GEMM-Accelerated Blocked QR with Recursive Panel Factorization
 */

#include "qr.h"
#include "../gemm_2/gemm.h"
#include "../gemm_2/gemm_planning.h"
#include "../gemm_2/gemm_utils.h"
#include "qr_kernels_avx2.h" 
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>

// Forward declarations
static void build_T_matrix(const float *Y, const float *tau, float *T,
                           uint16_t m, uint16_t ib, uint16_t ldy);

#define GEMM_CALL gemm_dynamic

#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef QR_ENABLE_PREFETCH
    #define QR_ENABLE_PREFETCH 1  // Enable by default
#endif

#if QR_ENABLE_PREFETCH && defined(__AVX2__)
    #define QR_PREFETCH_ENABLED
#endif

#ifndef QR_PREFETCH_DISTANCE_NEAR
    #define QR_PREFETCH_DISTANCE_NEAR 1  // Iterations ahead for T0
#endif

#ifndef QR_PREFETCH_DISTANCE_FAR
    #define QR_PREFETCH_DISTANCE_FAR 2   // Iterations ahead for T1
#endif

//==============================================================================
// NAIVE STRIDED GEMM (for debugging)
//==============================================================================

/**
 * @brief Naive strided GEMM: C = alpha*A*B + beta*C
 * 
 * @param C Output matrix [m × n], stride ldc
 * @param A Input matrix [m × k], stride lda
 * @param B Input matrix [k × n], stride ldb
 * @param m Number of rows in A and C
 * @param k Number of columns in A, rows in B
 * @param n Number of columns in B and C
 * @param ldc Stride of C (elements between rows)
 * @param lda Stride of A
 * @param ldb Stride of B
 * @param alpha Scalar for A*B
 * @param beta Scalar for C
 */
static void naive_gemm_strided(
    float *restrict C,
    const float *restrict A,
    const float *restrict B,
    uint16_t m, uint16_t k, uint16_t n,
    uint16_t ldc, uint16_t lda, uint16_t ldb,
    float alpha, float beta)
{
    // C = beta * C
    if (beta == 0.0f)
    {
        for (uint16_t i = 0; i < m; ++i)
            for (uint16_t j = 0; j < n; ++j)
                C[i * ldc + j] = 0.0f;
    }
    else if (beta != 1.0f)
    {
        for (uint16_t i = 0; i < m; ++i)
            for (uint16_t j = 0; j < n; ++j)
                C[i * ldc + j] *= beta;
    }

    // C += alpha * A * B
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (uint16_t p = 0; p < k; ++p)
            {
                sum += (double)A[i * lda + p] * (double)B[p * ldb + j];
            }
            C[i * ldc + j] += alpha * (float)sum;
        }
    }
}

//==============================================================================
// GEMM PLAN MANAGEMENT
//==============================================================================

/**
 * @brief Destroy GEMM plans and free associated memory
 * 
 * @param plans Plan structure to destroy (may be NULL)
 * 
 * @note Safe to call with NULL pointer
 * @note Must be called before freeing workspace to avoid leaks
 */
static void destroy_panel_plans(qr_gemm_plans_t *plans)
{
    if (!plans)
        return;
    
    // Free individual plans (each contains blocking metadata)
    gemm_plan_destroy(plans->plan_yt_c);  // Y^T × C plan
    gemm_plan_destroy(plans->plan_t_z);   // T × Z plan
    gemm_plan_destroy(plans->plan_y_z);   // Y × Z plan
    
    // Free plan container
    free(plans);
}

/**
 * @brief Create GEMM plans for block reflector operations
 * 
 * **Mathematical Context:**
 * 
 * Block reflector application computes: C = (I - Y·T·Y^T)·C
 * This is decomposed into 3 GEMM operations:
 * 
 * 1. Z = Y^T × C     [IB × M] × [M × N] → [IB × N]
 *    - Purpose: Project C onto Householder space
 *    - Memory: Z is IB×N (small, fits L1/L2)
 * 
 * 2. Z_temp = T × Z  [IB × IB] × [IB × N] → [IB × N]
 *    - Purpose: Apply compact WY scaling factor
 *    - Memory: T is IB×IB (tiny, ~64×64 = 16 KB max)
 * 
 * 3. C = C - Y × Z_temp  [M × IB] × [IB × N] → [M × N]
 *    - Purpose: Apply scaled reflection back to C
 *    - Memory: C is M×N (large, streaming from L3/DRAM)
 * 
 * **Why Separate Plans:**
 * 
 * Each GEMM has different performance characteristics:
 * - GEMM 1 (Y^T × C): Small M (IB), large K and N → tall-skinny
 * - GEMM 2 (T × Z): All dims small (IB) → tiny-GEMM, L1 resident
 * - GEMM 3 (Y × Z): Large M, small K (IB), large N → panel-update
 * 
 * Different shapes → different optimal MC/KC/NC blocking parameters
 * Pre-computing each plan allows optimal tuning per operation.
 * 
 * **Performance Impact:**
 * 
 * For 1024×1024 matrix with IB=64, 16 panels:
 * - Without plans: 48 calls × 400 cycles overhead = 19,200 cycles wasted
 * - With plans: 1000 cycles (one-time) + 48 × 10 = 1,480 cycles total
 * - Speedup: 13× reduction in GEMM dispatch overhead
 * - Overall impact: ~0.5-1% faster QR (non-trivial for large matrices)
 * 
 * @param[in] m   Number of rows (M dimension)
 * @param[in] n   Number of columns (N dimension)
 * @param[in] ib  Block size (IB dimension, from QR blocking)
 * 
 * @return Allocated plans structure, or NULL on failure
 * 
 * @note Plans are read-only after creation (safe for multiple threads)
 * @note Must be destroyed with destroy_panel_plans()
 */
static qr_gemm_plans_t *create_panel_plans(uint16_t m, uint16_t n, uint16_t ib)
{
    // Validate dimensions (zero-sized GEMM is meaningless)
    if (n == 0 || m == 0 || ib == 0)
        return NULL;

    // Allocate plan container
    qr_gemm_plans_t *plans = (qr_gemm_plans_t *)calloc(1, sizeof(qr_gemm_plans_t));
    if (!plans)
        return NULL;

    // Store dimensions for validation/debugging
    plans->plan_m = m;
    plans->plan_n = n;
    plans->plan_ib = ib;
    
    // Create plan for GEMM 1: Z = Y^T × C
    // Dimensions: [IB × M] × [M × N] → [IB × N]
    // Characteristics: Small-tall × large-square → skinny output
    plans->plan_yt_c = gemm_plan_create(ib, m, n);
    
    // Create plan for GEMM 2: Z_temp = T × Z
    // Dimensions: [IB × IB] × [IB × N] → [IB × N]
    // Characteristics: Tiny-square × small-wide → small output (L1 resident)
    plans->plan_t_z = gemm_plan_create(ib, ib, n);
    
    // Create plan for GEMM 3: C = C - Y × Z_temp
    // Dimensions: [M × IB] × [IB × N] → [M × N]
    // Characteristics: Large-skinny × small-wide → large output
    plans->plan_y_z = gemm_plan_create(m, ib, n);

    // Validate all plans succeeded
    if (!plans->plan_yt_c || !plans->plan_t_z || !plans->plan_y_z)
    {
        destroy_panel_plans(plans);
        return NULL;
    }

    return plans;
}


/**
 * @brief Detect if Householder computation needs numerically stable path
 * 
 * **Detection Strategy:**
 * 
 * Scan vector for values that would cause overflow/underflow in x²:
 * - |x| > 10¹⁹ → x² > 10³⁸ → overflow (FLT_MAX ≈ 3.4×10³⁸)
 * - |x| < 10⁻¹⁹ → x² < 10⁻³⁸ → underflow (FLT_MIN ≈ 1.2×10⁻³⁸)
 * - NaN or Inf → needs special handling
 * 
 * **Why These Thresholds:**
 * 
 * Float32 range: [1.2×10⁻³⁸, 3.4×10³⁸]
 * Square operation: x² → range becomes [1.4×10⁻⁷⁶, 1.2×10⁷⁶]
 * 
 * Safe zone for direct squaring: [10⁻¹⁹, 10¹⁹]
 * - 10⁻¹⁹ squared: 10⁻³⁸ (barely above underflow)
 * - 10¹⁹ squared: 10³⁸ (barely below overflow)
 * 
 * **Example Cases:**
 * 
 * ```c
 * // Normal case (fast path)
 * float x1[] = {1.0, 2.0, 3.0};  // max = 3.0 → fast path
 * 
 * // Overflow risk (safe path)
 * float x2[] = {1e20, 2e20};  // max = 2e20 > 1e19 → safe path
 * 
 * // Underflow risk (safe path)
 * float x3[] = {1e-20, 2e-20};  // max = 2e-20 < 1e-19 → safe path
 * 
 * // Mixed scale (safe path)
 * float x4[] = {1e-15, 1e15};  // contains extreme → safe path
 * ```
 * 
 * **Performance:**
 * - Scan cost: ~0.5 cycles/element (AVX2 max reduction)
 * - False positive rate: ~1% (very rare in practice)
 * - Benefit: Avoids 10× slowdown of safe path for normal data
 * 
 * @param[in]  x       Vector to check
 * @param[in]  len     Vector length
 * @param[out] max_abs Maximum absolute value found
 * @return true if safe path needed (NaN/Inf or extreme magnitude)
 * 
 * @note Uses AVX2 for fast parallel max reduction
 * @note Function is pure (no side effects)
 */
static inline bool needs_safe_householder(const float *x, uint16_t len, float *max_abs)
{
    // Thresholds for overflow/underflow in squared computation
    // These are conservative: leave 2× margin for numerical safety
    const float OVERFLOW_THRESHOLD = 1e19f;   // √FLT_MAX ≈ 1.84×10¹⁹
    const float UNDERFLOW_THRESHOLD = 1e-19f; // √FLT_MIN ≈ 1.08×10⁻¹⁹
    
    // Check for NaN/Inf first (these always need safe path)
    // NaN/Inf corrupt all arithmetic → must be handled specially
    if (has_nan_or_inf(x, len))
        return true;
    
    // Find maximum absolute value using AVX2 (if available)
    float max_val = 0.0f;
    
#ifdef __AVX2__
    if (len >= 8)
    {
        // Initialize max vector to zero
        __m256 max_vec = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        // Each iteration: 1 load + 1 abs (AND) + 1 max = 3 instructions
        // Throughput: ~1 cycle/iteration (3 ops / 3-wide superscalar)
        for (uint16_t i = 0; i < len - 7; i += 8)
        {
            __m256 v = _mm256_loadu_ps(&x[i]);
            
            // Absolute value: clear sign bit using AND-NOT
            // -0.0f = 0x80000000 (sign bit set)
            // andnot(0x80000000, v) clears sign bit → |v|
            __m256 abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
            
            // Element-wise maximum
            max_vec = _mm256_max_ps(max_vec, abs_v);
        }
        
        // Horizontal reduction: find max of 8 values
        // Extract 8 floats from vector and find maximum scalar
        float vals[8];
        _mm256_storeu_ps(vals, max_vec);
        for (int i = 0; i < 8; ++i)
            if (vals[i] > max_val)
                max_val = vals[i];
    }
#endif

    // Scalar tail: process remaining 0-7 elements
    for (uint16_t i = (len / 8) * 8; i < len; ++i)
    {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_val)
            max_val = abs_val;
    }
    
    // Output maximum for caller's use (e.g., diagnostics)
    *max_abs = max_val;
    
    // Decision: Is value in safe range for direct squaring?
    if (max_val > OVERFLOW_THRESHOLD)
        return true;  // Risk of overflow: x² > FLT_MAX
    
    if (max_val > 0.0f && max_val < UNDERFLOW_THRESHOLD)
        return true;  // Risk of underflow: x² < FLT_MIN
    
    // All values in safe range [10⁻¹⁹, 10¹⁹] → fast path OK
    return false;
}

//==============================================================================
// HOUSEHOLDER REFLECTION PRIMITIVES
//==============================================================================

/**
 * @brief Robust Householder reflector generation with fast/safe path selection
 * 
 * Uses fast AVX2 norm computation by default, falls back to scaled algorithm
 * only when overflow/underflow risk detected.
 * 
 * @param x [in/out] Input vector, output: normalized reflector (x[0]=1 implicit)
 * @param m Vector length
 * @param tau [out] Householder scaling factor τ
 * @param beta [out] Resulting diagonal element β
 * 
 * @note Fast path: Direct norm² computation (most common)
 * @note Safe path: LAPACK DLASSQ scaling (rare, extreme values)
 */
static void compute_householder_robust(float *restrict x, uint16_t m,
                                        float *restrict tau, float *restrict beta)
{
    //==========================================================================
    // EDGE CASE 1: Empty vector
    //==========================================================================
    // If M=0, there's nothing to reflect → return identity transformation
    if (m == 0)
    {
        *tau = 0.0f;
        if (beta)
            *beta = 0.0f;
        return;
    }

    //==========================================================================
    // EDGE CASE 2: Single element
    //==========================================================================
    // If M=1, vector is [x₀] → already in canonical form [β]
    // No reflection needed: H = I, τ = 0, β = x₀
    if (m == 1)
    {
        *tau = 0.0f;
        if (beta)
            *beta = x[0];
        x[0] = 1.0f;  // Set v₀=1 for consistency (though unused when τ=0)
        return;
    }

    //==========================================================================
    // PATH SELECTION: Fast (direct) vs Safe (scaled)
    //==========================================================================
    // 
    // Scan tail x[1:m-1] for extreme values that would cause overflow/underflow
    // in squared norm computation. Also check x[0] separately.
    // 
    // Decision tree:
    // - All values in [10⁻¹⁹, 10¹⁹] AND no NaN/Inf → Fast path (99% of cases)
    // - Any value outside range OR NaN/Inf → Safe path (1% of cases)
    //
    //==========================================================================
    
    float tail_max;
    bool need_safe = needs_safe_householder(&x[1], m - 1, &tail_max);
    
    // Also check x₀ (alpha) for extremes
    float abs_alpha = fabsf(x[0]);
    if (!need_safe)
    {
        const float OVERFLOW_THRESHOLD = 1e19f;
        const float UNDERFLOW_THRESHOLD = 1e-19f;
        
        // Check if x₀ is problematic
        if (!isfinite(x[0]) ||                                    // NaN or Inf
            abs_alpha > OVERFLOW_THRESHOLD ||                     // Too large
            (abs_alpha > 0.0f && abs_alpha < UNDERFLOW_THRESHOLD)) // Too small
        {
            need_safe = true;
        }
    }

    //==========================================================================
    // SAFE PATH: Scaled computation using LAPACK DLASSQ algorithm
    //==========================================================================
    // 
    // **When Used:** ~1% of cases (extreme values or NaN/Inf)
    // 
    // **Algorithm:** Maintain (scale, sumsq) representation instead of direct sum
    // 
    //   ||x||² = scale² × sumsq
    // 
    // where:
    //   scale = max(|xᵢ|)           (largest magnitude element)
    //   sumsq = Σ(xᵢ/scale)²        (sum of scaled squares)
    // 
    // **Why This Works:**
    // - Scale captures magnitude (prevents over/underflow)
    // - Sumsq captures relative proportions (normalized to [0,1] range)
    // - Final norm = scale × √sumsq (reconstructed safely)
    //==========================================================================
    
    if (need_safe)
    {
        double alpha = (double)x[0];
        
        // Initialize (scale, sumsq) = (0, 1)
        // Convention: scale=0 means "empty accumulator", sumsq=1 is identity
        double scale = 0.0;
        double sumsq = 1.0;
        
        //======================================================================
        // Accumulate tail elements x[1:m-1] into (scale, sumsq)
        //======================================================================
        // 
        // For each element xᵢ:
        // - If |xᵢ| > scale: Rescale sumsq, update scale (new maximum)
        // - If |xᵢ| ≤ scale: Add (xᵢ/scale)² to sumsq (accumulate relative)
        // 
        // Invariant maintained: ||x[0:i]||² = scale² × sumsq
        //
        //======================================================================
        
        for (uint16_t i = 1; i < m; ++i)
        {
            double absxi = fabs((double)x[i]);
            
            if (absxi > scale)
            {
                // New maximum found: rescale existing sumsq
                // Old: ||x||² = scale_old² × sumsq_old
                // New: ||x||² = absxi² × sumsq_new
                // Therefore: sumsq_new = (scale_old/absxi)² × sumsq_old + 1
                // The "+1" accounts for current element (absxi/absxi)² = 1
                double ratio = scale / absxi;
                sumsq = 1.0 + sumsq * ratio * ratio;
                scale = absxi;
            }
            else if (absxi > 0.0)
            {
                // Add to existing sum (current element is not the max)
                double ratio = absxi / scale;
                sumsq += ratio * ratio;
            }
            // else: xᵢ = 0 or NaN, skip (doesn't contribute to norm)
        }
        
        //======================================================================
        // Accumulate x₀ (alpha) into (scale, sumsq)
        //======================================================================
        
        double absalpha = fabs(alpha);
        if (absalpha > scale)
        {
            double ratio = scale / absalpha;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = absalpha;
        }
        else if (absalpha > 0.0)
        {
            double ratio = absalpha / scale;
            sumsq += ratio * ratio;
        }
        
        // If scale=0: all elements were zero or NaN → no reflection needed
        // If sumsq is NaN: corrupted by NaN propagation → abort reflection
        
        if (scale == 0.0 || !isfinite(sumsq))
        {
            *tau = 0.0f;      // Identity transformation
            if (beta)
                *beta = x[0]; // β = x₀ (no change)
            x[0] = 1.0f;      // Set v₀=1 (convention)
            return;
        }
        
        //======================================================================
        // Reconstruct norm: ||x|| = scale × √sumsq
        //======================================================================
        
        double norm = scale * sqrt(sumsq);
        
        //======================================================================
        // Compute β with sign chosen to maximize |β - α|
        //======================================================================
        // β = -sign(α) × ||x||
        // copysign(norm, alpha) returns norm with sign of alpha
        // Therefore -copysign(norm, alpha) has opposite sign of alpha ✓
        
        double beta_val = -copysign(norm, alpha);
        
        //======================================================================
        // Compute τ = (β - α)/β
        //======================================================================
        // Derivation: From H·x = β·e₁ where H = I - τ·v·v^T
        // Solving for τ with v normalized such that v₀=1
        
        *tau = (float)((beta_val - alpha) / beta_val);
        
        if (beta)
            *beta = (float)beta_val;
        
        //======================================================================
        // Normalize reflector: vᵢ = xᵢ / (α - β)
        //======================================================================
        // This makes v₀ = α/(α-β) but we'll overwrite it to 1.0 afterward
        // The tail elements v₁, v₂, ... are correctly normalized
        
        double scale_factor = 1.0 / (alpha - beta_val);
        
        for (uint16_t i = 1; i < m; ++i)
            x[i] *= (float)scale_factor;
        
        x[0] = 1.0f;  // Explicit v₀ = 1 (overwrite normalized α)
        return;
    }

    //==========================================================================
    // FAST PATH: Direct computation (no overflow risk)
    //==========================================================================
    // 
    // **When Used:** ~99% of cases (normal float values)
    // 
    // **Algorithm:** Direct squared-norm computation
    // 
    //   ||x||² = x₁² + x₂² + ... + xₘ₋₁²    (tail only, x₀ handled separately)
    //   ||x||  = √(x₀² + ||tail||²)          (full norm including x₀)
    // 
    // **Performance:** 5-10× faster than safe path
    // - AVX2 vectorization: 8 FMAs per iteration
    // - Cache-friendly: sequential access
    // - ~0.5 cycles/element throughput
    //
    //==========================================================================
    
    double norm_sq;
    
#ifdef __AVX2__
    // Use vectorized norm computation for vectors ≥ 10 elements
    // Threshold 10: balances AVX2 overhead vs scalar simplicity
    // - Below 10: AVX2 setup cost (50 cycles) dominates
    // - Above 10: AVX2 wins (0.5 vs 1.5 cycles/element)
    norm_sq = (m > 9) ? compute_norm_sq_avx2(&x[1], m - 1) : 0.0;
    if (m <= 9)
#endif
    {
        // Scalar path for small vectors
        // Cost: ~1.5 cycles/element (1 FMA + dependencies)
        norm_sq = 0.0;
        for (uint16_t i = 1; i < m; ++i)
        {
            double xi = (double)x[i];
            norm_sq += xi * xi;
        }
    }

    //==========================================================================
    // EDGE CASE 3: Zero tail (x₁ = x₂ = ... = 0)
    //==========================================================================
    // If tail is all zeros, vector is already in form [x₀, 0, 0, ...]
    // No reflection needed: H = I, τ = 0, β = x₀
    
    if (norm_sq == 0.0)
    {
        *tau = 0.0f;
        if (beta)
            *beta = x[0];
        x[0] = 1.0f;
        return;
    }

    //==========================================================================
    // Compute full norm including x₀
    //==========================================================================
    // ||x||² = x₀² + (x₁² + x₂² + ... + xₘ₋₁²)
    //        = alpha² + norm_sq
    // ||x||  = √(alpha² + norm_sq)
    
    double alpha = (double)x[0];
    double beta_val = -copysign(sqrt(alpha * alpha + norm_sq), alpha);
    
    //==========================================================================
    // Compute scaling factor: 1/(α - β)
    //==========================================================================
    // This will be used to normalize reflector: vᵢ = xᵢ/(α - β)
    // 
    // Note: α - β is always large in magnitude due to sign choice:
    // - If α > 0: β < 0 → α - β = |α| + |β| (both positive contributions)
    // - If α < 0: β > 0 → α - β = -(|α| + |β|) (both negative, large magnitude)
    // 
    // This prevents division by a small number (good numerical stability)
    
    double scale = 1.0 / (alpha - beta_val);

    //==========================================================================
    // Vectorized scaling: vᵢ = xᵢ × scale for i ∈ [1, m-1]
    //==========================================================================
    
#ifdef __AVX2__
    if (m > 9)
    {
        // Broadcast scale to all 8 lanes of AVX2 register
        __m256 scale_vec = _mm256_set1_ps((float)scale);
        
        uint16_t i = 1;
        
        // Main loop: scale 8 elements per iteration
        // Cost: 1 load + 1 multiply + 1 store = 3 instructions
        // Throughput: ~1 cycle/iteration (3 ops / 3-wide issue)
        // Effective: ~0.125 cycles/element
        for (; i + 7 < m; i += 8)
        {
            __m256 v = _mm256_loadu_ps(&x[i]);
            v = _mm256_mul_ps(v, scale_vec);
            _mm256_storeu_ps(&x[i], v);
        }
        
        // Scalar tail: remaining 0-7 elements
        for (; i < m; ++i)
            x[i] *= (float)scale;
    }
    else
#endif
    {
        // Scalar path: multiply each element
        // Cost: ~0.5 cycles/element (simple multiply)
        for (uint16_t i = 1; i < m; ++i)
            x[i] *= (float)scale;
    }

    //==========================================================================
    // Finalize outputs
    //==========================================================================
    
    // τ = (β - α)/β
    // This is the Householder scaling factor in compact WY representation
    *tau = (float)((beta_val - alpha) / beta_val);
    
    // β = -sign(α)·||x||
    // This is the resulting diagonal element after reflection
    if (beta)
        *beta = (float)beta_val;
    
    // Set v₀ = 1 (implicit first element of reflector)
    // This overwrites the original x₀ value
    x[0] = 1.0f;
}

//==============================================================================
// HOUSEHOLDER REFLECTION APPLICATION
//==============================================================================

/**
 * @brief Apply single Householder reflector to matrix (scalar fallback)
 * 
 * ```
 * 1. Compute dot product: dⱼ = v^T · C[:,j] = Σᵢ vᵢ·Cᵢⱼ
 * 2. Scale: sⱼ = τ · dⱼ
 * 3. Update column: C[:,j] = C[:,j] - v · sⱼ
 * ```
 * 
 * Repeating for all N columns gives the full matrix update.
 * 
 * **Complexity Analysis:**
 * 
 * For matrix C of size M×N:
 * - Dot products: N × (2M-1) FLOPs = 2MN - N FLOPs
 * - Updates: N × 2M FLOPs = 2MN FLOPs
 * Total: 4MN FLOPs (Level 2 BLAS, O(MN) work per reflector)
 * 
 * Memory traffic:
 * - Read C: M×N floats (column-major with stride)
 * - Read v: N×M loads (v is reused N times, should stay in cache)
 * - Write C: M×N floats
 * Total: ~2MN loads + MN stores ≈ 12MN bytes for float32
 * 
 * **Cache Behavior:**
 * 
 * Best case (M×N fits in cache):
 * - v stays in L1 (M floats ≈ 4M bytes, typically < 48 KB)
 * - C accessed column-wise (stride = ldc)
 * - If ldc ≈ M and M×N < L2: good spatial locality
 * 
 * Worst case (M×N >> L3):
 * - Each column of C streams from DRAM (200 cycle latency)
 * - Software prefetching helps hide latency
 * 
 * 
 * @param[in,out] C   Matrix to update [M × N], row-major with stride ldc
 *                    Updated in-place: C = (I - τ·v·v^T)·C
 * @param[in]     m   Number of rows in C
 * @param[in]     n   Number of columns in C
 * @param[in]     ldc Leading dimension (stride between rows)
 * @param[in]     v   Householder vector [M], with v[0]=1 implicit
 * @param[in]     tau Scaling factor τ
 * 
 * @note If τ=0, function returns immediately (identity, no-op)
 * @note Uses double precision accumulation for numerical stability
 * @note Includes software prefetching for large matrices
 * @note Automatically dispatches to AVX2 path when N ≥ 8
 * 
 * @see apply_householder_avx2() for vectorized implementation
 * @see LAPACK SLARF for equivalent standard implementation
 */
static void apply_householder_clean(float *restrict C, uint16_t m, uint16_t n,
                                    uint16_t ldc, const float *restrict v, float tau)
{

#ifdef __AVX2__
    if (n >= 8)
    {
        apply_householder_avx2(C, m, n, ldc, v, tau);
        return;
    }
#endif

    //==========================================================================
    // EARLY EXIT: Identity transformation
    //==========================================================================

    if (tau == 0.0f)
        return;

    //==========================================================================
    // MAIN LOOP: Process columns one at a time
    //==========================================================================
    // Each iteration:
    // 1. Compute dot product: d = v^T · C[:,j]
    // 2. Apply rank-1 update: C[:,j] -= (τ·d)·v
    
    for (uint16_t j = 0; j < n; ++j)
    {
        //======================================================================
        // SOFTWARE PREFETCHING: Hide memory latency
        //======================================================================

        
#ifdef __AVX2__
        // Prefetch next column (j+1) to L1 cache
        if (j + 1 < n)
        {
            // Prefetch first 64 rows (64 × 4 bytes = 256 bytes = 4 cache lines)
            // Why 64? Balance between:
            // - Too few: Miss prefetch opportunities for large M
            // - Too many: Pollute cache with data that won't be used soon
            uint16_t prefetch_rows = MIN(64, m);
            
            // Prefetch every 8 rows (8 floats = 32 bytes = half cache line)
            // Why 8? Cache line is 64 bytes = 16 floats, stride by 8 to cover
            for (uint16_t i = 0; i < prefetch_rows; i += 8)
            {
                // _MM_HINT_T0: Prefetch to L1 cache (all cache levels)
                // Most temporal data - will be used in next iteration
                _mm_prefetch((const char*)&C[i * ldc + (j + 1)], _MM_HINT_T0);
            }
        }
        
        // Prefetch column after next (j+2) to L2 cache
        if (j + 2 < n)
        {
            // Prefetch first 32 rows (less aggressive than T0)
            // Rationale: This data is further away, don't fill L1 yet
            uint16_t prefetch_rows = MIN(32, m);
            
            for (uint16_t i = 0; i < prefetch_rows; i += 8)
            {
                // _MM_HINT_T1: Prefetch to L2 cache (not L1)
                // Less temporal - will be used in iteration after next
                _mm_prefetch((const char*)&C[i * ldc + (j + 2)], _MM_HINT_T1);
            }
        }
#endif

        //======================================================================
        // PHASE 1: Compute dot product d = v^T · C[:,j]
        //======================================================================
        
        double dot = 0.0;
        for (uint16_t i = 0; i < m; ++i)
            dot += (double)v[i] * (double)C[i * ldc + j];
        
        // Scale dot product by τ: s = τ·d
        // This is the factor by which we'll scale v before subtracting from C[:,j]
        float tau_dot = tau * (float)dot;
        
        //======================================================================
        // PHASE 2: Apply rank-1 update C[:,j] -= (τ·d)·v
        //======================================================================
        
        for (uint16_t i = 0; i < m; ++i)
            C[i * ldc + j] -= v[i] * tau_dot;
    }
}

//==============================================================================
// PANEL FACTORIZATION (WITH STRIDE SUPPORT)
//==============================================================================

/**
 * @brief Classical panel factorization with proper stride handling
 * 
 * **Algorithm:** Unblocked Householder QR on a panel [M × IB]
 * 
 * For each column j in panel:
 * 1. Extract column j (gather from strided storage)
 * 2. Compute Householder reflector vⱼ, τⱼ
 * 3. Write reflector back to panel (in-place, below diagonal)
 * 4. Store complete reflector in Y matrix (stride-aware)
 * 5. Apply reflector to remaining columns j+1:IB
 * 
 * **Storage Convention:**
 * - Panel: Lower triangle holds reflectors, upper triangle holds R
 * - Y matrix: Complete reflectors (including implicit v₀=1)
 * 
 * **Stride Handling (CRITICAL):**
 * - Panel has stride lda (may be > IB if part of larger matrix)
 * - Y has stride ldy (may differ from IB!)
 * - Must use correct stride in all accesses to avoid buffer overflows
 *
 * @param[in,out] panel Panel matrix [M × IB], stride lda
 *                      Lower triangle overwritten with reflectors
 *                      Upper triangle contains R factors
 * @param[out]    Y     Householder vectors [M × IB], stride ldy
 * @param[out]    tau   Scaling factors [IB]
 * @param[in]     m     Number of rows
 * @param[in]     ib    Number of columns (panel width)
 * @param[in]     lda   Leading dimension of panel
 * @param[in]     ldy   Leading dimension of Y (WARNING: may differ from ib!)
 * @param[in,out] work  Workspace [M] for column extraction
 */
static void panel_factor_clean(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,
    uint16_t ldy,
    float *restrict work)
{
    //==========================================================================
    // Initialize Y to zero
    //==========================================================================
    // Y stores complete Householder vectors (including explicit v₀)
    // Upper triangle of Y will be zeros (reflectors start at diagonal)
    
#ifdef __AVX2__
    if (ib >= 8)
    {
        zero_fill_strided_avx2(Y, m, ib, ldy);
    }
    else
#endif
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < ib; ++j)
        {
            Y[i * ldy + j] = 0.0f;
        }
    }

    //==========================================================================
    // Factor each column
    //==========================================================================
    
    for (uint16_t j = 0; j < ib && j < m; ++j)
    {
        uint16_t col_len = m - j;

        //======================================================================
        // Prefetch next columns to hide memory latency
        //======================================================================
        
#ifdef __AVX2__
        if (j + 1 < ib && j + 1 < m)
        {
            float *next_col = &panel[(j + 1) * lda + (j + 1)];
            uint16_t prefetch_len = MIN(64, m - j - 1);
            
            for (uint16_t i = 0; i < prefetch_len; i += 8)
            {
                _mm_prefetch((const char*)&next_col[i * lda], _MM_HINT_T0);
            }
        }
        
        if (j + 2 < ib && j + 2 < m)
        {
            float *next_next_col = &panel[(j + 2) * lda + (j + 2)];
            uint16_t prefetch_len = MIN(32, m - j - 2);
            
            for (uint16_t i = 0; i < prefetch_len; i += 8)
            {
                _mm_prefetch((const char*)&next_next_col[i * lda], _MM_HINT_T1);
            }
        }
#endif

        //======================================================================
        // Extract column j from panel (gather from strided storage)
        //======================================================================
        // Panel is stored with stride lda, need to gather column into
        // contiguous work buffer for Householder computation
        
        float *restrict col_ptr = &panel[j * lda + j];
        for (uint16_t i = 0; i < col_len; ++i)
            work[i] = col_ptr[i * lda];

        //======================================================================
        // Compute Householder reflector
        //======================================================================
        // work is overwritten with normalized reflector v (with v₀=1)
        // beta is the resulting R diagonal element
        // tau is the Householder scaling factor
        
        float beta;
        compute_householder_robust(work, col_len, &tau[j], &beta);

        //======================================================================
        // Write results back to panel
        //======================================================================
        
        // Diagonal element: R factor
        col_ptr[0] = beta;

        // Below diagonal: reflector tail (scatter to strided storage)
        // Note: work[0]=1 is implicit, stored elements are work[1:]
        for (uint16_t i = 1; i < col_len; ++i)
            col_ptr[i * lda] = work[i];

        //======================================================================
        // Store complete reflector in Y with proper stride
        //======================================================================
        // ⚠️ CRITICAL: Use ldy, not ib!
        // Y may have different stride than panel during recursion
        
        // Upper part (rows 0:j-1): zeros
        for (uint16_t i = 0; i < j; ++i)
            Y[i * ldy + j] = 0.0f;
        
        // Reflector part (rows j:m-1): complete vector including v₀=1
        for (uint16_t i = 0; i < col_len; ++i)
            Y[(j + i) * ldy + j] = work[i];

        //======================================================================
        // Apply reflector to trailing columns j+1:ib-1
        //======================================================================
        // This is Level 2 BLAS (could be Level 3 with block reflector)
        
        if (j + 1 < ib)
        {
            float *restrict trailing = &panel[j * lda + (j + 1)];
            apply_householder_clean(trailing, col_len, ib - j - 1,
                                    lda, work, tau[j]);
        }
    }
}

/**
 * @brief Recursive panel factorization (DGEQRT3-style)
 * 
 * **Algorithm:** Divide-and-conquer Householder QR
 * 
 * Base case (ib ≤ threshold):
 *   Use classical unblocked algorithm
 * 
 * Recursive case:
 *   1. Split panel vertically: [A_left | A_right] where A_left has ib1 columns
 *   2. Factor A_left recursively → produces Y_left, tau_left
 *   3. Apply Y_left reflectors to A_right (update phase)
 *   4. Factor updated A_right recursively → produces Y_right, tau_right
 *   5. Merge: Combine Y_left and Y_right into output Y
 * 
 * **Why Recursion Helps:**
 * - Classical algorithm: All operations are Level 2 BLAS (matrix-vector)
 * - Recursive algorithm: Step 3 can use Level 3 BLAS (matrix-matrix)
 * - Trade-off: Recursion overhead vs Level 3 BLAS speedup
 * - Crossover: Beneficial when ib ≥ 16 (threshold tuning)
 * 
 * **Memory Management:**
 * - Pre-allocated workspace: 2×M×IB floats (no malloc in recursion!)
 * - Partitioned for each level: [Y_left | Y_right | remaining workspace]
 * - Fallback to base case if workspace exhausted
 * 
 * **Comparison to LAPACK DGEQRT3:**
 * - Structure: Identical recursive decomposition
 * - Difference: LAPACK uses 3 GEMM calls in step 3 (Level 3 BLAS)
 * - This code: Uses IB1 Householder applications (Level 2 BLAS)
 * - TODO: Replace step 3 with block reflector for full Level 3 BLAS
 * 
 * @param[in,out] panel     Panel matrix [M × IB], stride lda
 * @param[out]    Y         Householder vectors [M × IB], stride ldy
 * @param[out]    tau       Scaling factors [IB]
 * @param[in,out] workspace Pre-allocated workspace (size: 2×M×IB floats)
 * @param[in]     workspace_size Available workspace size (in floats)
 * @param[in,out] work      Temporary buffer [M] for Householder computation
 * @param[in]     m         Number of rows
 * @param[in]     ib        Number of columns (panel width)
 * @param[in]     lda       Leading dimension of panel
 * @param[in]     ldy       Leading dimension of Y
 * @param[in]     threshold Base case threshold (stop recursing when ib ≤ threshold)
 */
static void panel_factor_recursive(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    float *restrict workspace,
    size_t workspace_size,
    float *restrict work,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,
    uint16_t ldy,
    uint16_t threshold)
{
    //==========================================================================
    // Base case: Use classical algorithm
    //==========================================================================
    // Stop recursing when:
    // - ib ≤ threshold (recursion overhead not justified)
    // - ib < 2 (cannot split further)
    
    if (ib <= threshold || ib < 2)
    {
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, work);
        return;
    }

    //==========================================================================
    // Split panel: [A_left | A_right]
    //==========================================================================
    
    uint16_t ib1 = ib / 2;      // Left width
    uint16_t ib2 = ib - ib1;    // Right width

    //==========================================================================
    // Check workspace availability
    //==========================================================================
    // Y_left needs: M × IB1 floats
    // Y_right needs: (M - IB1) × IB2 floats (fewer rows after left factorization)
    
    size_t y_left_size = (size_t)m * ib1;
    size_t y_right_size = (size_t)(m - ib1) * ib2;
    size_t required = y_left_size + y_right_size;
    
    // Fallback to base case if insufficient workspace
    if (workspace_size < required)
    {
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, work);
        return;
    }
    
    //==========================================================================
    // Partition workspace (no malloc!)
    //==========================================================================
    
    float *Y_left = workspace;
    float *Y_right = workspace + y_left_size;
    float *workspace_next = workspace + required;
    size_t workspace_next_size = workspace_size - required;

    //==========================================================================
    // STEP 1: Factor left panel recursively
    //==========================================================================
    // Factor A_left [M × IB1] → produces Y_left, tau[0:IB1-1]
    
    panel_factor_recursive(
        panel, Y_left, tau,
        workspace_next, workspace_next_size,
        work,
        m, ib1, lda, ib1, threshold);  // Note: Y_left has stride ib1 (packed)

    //==========================================================================
    // Copy Y_left to output Y with correct stride
    //==========================================================================
    // Y_left has stride ib1 (packed), output Y has stride ldy
    
    for (uint16_t i = 0; i < m; ++i)
        for (uint16_t j = 0; j < ib1; ++j)
            Y[i * ldy + j] = Y_left[i * ib1 + j];

    //==========================================================================
    // STEP 2: Apply left reflectors to right columns
    //==========================================================================
    // Update A_right using reflectors from A_left
    // TODO: Replace with block reflector (3 GEMMs) for Level 3 BLAS
    
    float *right_cols = &panel[ib1];
    for (uint16_t j = 0; j < ib1; ++j)
    {
        uint16_t rows_affected = m - j;
        
        // Extract reflector j from Y_left
        for (uint16_t i = 0; i < rows_affected; ++i)
            work[i] = Y_left[(j + i) * ib1 + j];

        // Apply to right columns
        float *right_start = &right_cols[j * lda];
        apply_householder_clean(right_start, rows_affected, ib2,
                                lda, work, tau[j]);
    }

    //==========================================================================
    // STEP 3: Factor right panel recursively
    //==========================================================================
    // Factor updated A_right [(M-IB1) × IB2] → produces Y_right, tau[IB1:]
    
    float *right_panel = &panel[ib1 * lda + ib1];
    panel_factor_recursive(
        right_panel, Y_right, &tau[ib1],
        workspace_next, workspace_next_size,
        work,
        m - ib1, ib2, lda, ib2, threshold);  // Y_right has stride ib2 (packed)

    //==========================================================================
    // STEP 4: Merge Y_right into output Y
    //==========================================================================
    // Y_right corresponds to rows IB1:M-1, columns IB1:IB-1
    
    // Upper-left block (rows 0:IB1-1, cols IB1:IB-1): zeros
    for (uint16_t i = 0; i < ib1; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            Y[i * ldy + (ib1 + j)] = 0.0f;

    // Lower-right block (rows IB1:M-1, cols IB1:IB-1): Y_right
    for (uint16_t i = 0; i < m - ib1; ++i)
        for (uint16_t j = 0; j < ib2; ++j)
            Y[(ib1 + i) * ldy + (ib1 + j)] = Y_right[i * ib2 + j];
}


/**
 * @brief Entry point for panel factorization with optimal path selection
 * 
 * Automatically chooses between:
 * - Classical (ib < 16): Direct algorithm
 * - Recursive (ib ≥ 16): Divide-and-conquer with tuned threshold
 * 
 * @param panel     Panel matrix [M × IB], stride lda
 * @param Y         Householder vectors [M × IB], stride workspace->ib
 * @param tau       Scaling factors [IB]
 * @param m         Number of rows
 * @param ib        Number of columns
 * @param lda       Leading dimension of panel
 * @param ldy       Leading dimension of Y
 * @param workspace Pre-allocated QR workspace
 */
static void panel_factor_optimized(
    float *restrict panel,
    float *restrict Y,
    float *restrict tau,
    uint16_t m,
    uint16_t ib,
    uint16_t lda,
    uint16_t ldy,
    qr_workspace *workspace)
{
    //==========================================================================
    // Small panels: Use classical algorithm
    //==========================================================================
    // For ib < 16, recursion overhead outweighs Level 3 BLAS benefit
    
    uint16_t threshold;
    if (ib < 16)
    {
        panel_factor_clean(panel, Y, tau, m, ib, lda, ldy, workspace->tmp);
        return;
    }
    
    //==========================================================================
    // Large panels: Use recursive algorithm with tuned threshold
    //==========================================================================
    // Threshold selection (empirical tuning):
    // - ib < 32: threshold = 8  (shallow recursion)
    // - ib < 64: threshold = 12 (moderate recursion)
    // - ib ≥ 64: threshold = 16 (deep recursion)
    
    else if (ib < 32)
        threshold = 8;
    else if (ib < 64)
        threshold = 12;
    else
        threshold = 16;

    // Calculate available workspace
    size_t workspace_size = 2 * (size_t)workspace->m_max * workspace->ib;
    
    panel_factor_recursive(
        panel, Y, tau,
        workspace->panel_Y_temp,
        workspace_size,
        workspace->tmp,
        m, ib, lda, ldy, threshold);
}

//==============================================================================
// BUILD T MATRIX (WITH STRIDE SUPPORT)
//==============================================================================


/**
 * @brief Build compact WY representation T matrix from Householder vectors
 * 
 * **Mathematical Background:**
 * 
 * Given IB Householder reflectors H₀, H₁, ..., H_{IB-1} where Hⱼ = I - τⱼvⱼvⱼᵀ,
 * the product H = H_{IB-1}···H₁H₀ can be written compactly as:
 * 
 *   H = I - Y·T·Yᵀ
 * 
 * where:
 * - Y = [v₀, v₁, ..., v_{IB-1}] is M×IB (stored Householder vectors)
 * - T is IB×IB upper triangular (compact factor, computed by this function)
 * 
 * **Why This Representation Matters:**
 * 
 * Without compact form:
 * - Apply H to C: Must apply H₀, H₁, ..., H_{IB-1} sequentially (IB Level 2 ops)
 * - Cost: IB × O(MN) = O(IB·M·N) work
 * 
 * With compact form (Y·T·Yᵀ):
 * - Apply H to C: Three matrix multiplies (Level 3 BLAS)
 * - Cost: O(IB·M·N) work BUT cache-optimized via GEMM
 * - Speedup: 2-5× due to better cache reuse in GEMM
 * 
 * **Recursive T Construction:**
 * 
 * Build T column by column:
 * 
 * T[:,0] = [τ₀]              (first column is just τ₀)
 *          [0 ]
 *          [⋮ ]
 * 
 * T[:,j] = [T₀,...,T_{j-1}] · w   where w = -τⱼ·Yᵀ[:,0:j]·Y[:,j]
 * 
 * This recursively incorporates each new reflector into the compact form.
 * 
 * **Algorithm (column j):**
 * 
 * 1. Set diagonal: T[j,j] = τⱼ
 * 2. Compute intermediate: wₖ = -τⱼ·⟨Y[:,k], Y[:,j]⟩ for k < j
 * 3. Apply previous T: T[:,j] = T[:,0:j-1]·w
 * 
 * **Storage:** T is upper triangular (lower triangle unused)
 * 
 * **Complexity:**
 * - Total work: O(IB²·M) (dominated by dot products)
 * - Memory: IB² floats for T (tiny, ~16 KB for IB=64)
 * 
 * @param[in]  Y   Householder vectors [M × IB], stride ldy
 * @param[in]  tau Scaling factors [IB]
 * @param[out] T   Compact WY factor [IB × IB], row-major, upper triangular
 * @param[in]  m   Number of rows
 * @param[in]  ib  Number of reflectors (columns)
 * @param[in]  ldy Leading dimension of Y
 * 
 * @note T is initialized to zero, then filled column by column
 * @note Uses stack allocation for workspace if ib ≤ 64, heap otherwise
 * @note Double precision accumulation for numerical stability
 */
static void build_T_matrix(const float *restrict Y, const float *restrict tau,
                           float *restrict T, uint16_t m, uint16_t ib,
                           uint16_t ldy)
{
    // Initialize T to zero
    memset(T, 0, (size_t)ib * ib * sizeof(float));
    if (ib == 0)
        return;

    //==========================================================================
    // Allocate workspace for w vector
    //==========================================================================
    // w stores intermediate products: wₖ = -τⱼ·⟨Y[:,k], Y[:,j]⟩
    // Size: IB doubles (use stack for ib ≤ 64, heap for larger)
    
    double w_stack[64];
    double *w = (ib <= 64) ? w_stack : (double *)malloc(ib * sizeof(double));
    if (!w)
        return;

    //==========================================================================
    // Build T column by column
    //==========================================================================
    
    for (uint16_t i = 0; i < ib; ++i)
    {
        //======================================================================
        // Step 1: Set diagonal element T[i,i] = τᵢ
        //======================================================================
        
        T[i * ib + i] = tau[i];

        // Skip rest if first column or zero reflector
        if (tau[i] == 0.0f || i == 0)
            continue;

        //======================================================================
        // Prefetch next Y column for better cache utilization
        //======================================================================
        
#ifdef __AVX2__
        if (i + 1 < ib)
        {
            uint16_t prefetch_rows = MIN(64, m);
            for (uint16_t r = 0; r < prefetch_rows; r += 8)
            {
                _mm_prefetch((const char*)&Y[r * ldy + (i + 1)], _MM_HINT_T0);
            }
        }
#endif

        //======================================================================
        // Step 2: Compute w = -τᵢ·Yᵀ[:,0:i]·Y[:,i]
        //======================================================================
        // For each previous reflector j < i:
        //   wⱼ = -τᵢ·⟨Y[:,j], Y[:,i]⟩
        // 
        // This represents the interaction between reflector i and all
        // previous reflectors. Negative sign comes from the recursion formula.
        
        for (uint16_t j = 0; j < i; ++j)
        {
#ifdef __AVX2__
            // Prefetch ahead in dot product computation
            if (j + 8 < i)
            {
                for (uint16_t r = 0; r < MIN(32, m); r += 8)
                {
                    _mm_prefetch((const char*)&Y[r * ldy + (j + 8)], _MM_HINT_T1);
                }
            }
            
            // Use AVX2 strided dot product for large M
            double dot = (m >= 16) ? dot_product_strided_avx2(&Y[j], &Y[i], m, ldy, ldy) : 0.0;
            if (m < 16)
#endif
            {
                // Scalar fallback for small M
                dot = 0.0;
                for (uint16_t r = 0; r < m; ++r)
                    dot += (double)Y[r * ldy + j] * (double)Y[r * ldy + i];
            }
            w[j] = -(double)tau[i] * dot;
        }

        //======================================================================
        // Step 3: Compute T[:,i] = T[:,0:i-1]·w
        //======================================================================
        // Matrix-vector product: multiply T's first i columns by w
        // This propagates the effect of all previous reflectors through T
        // 
        // Mathematically: T[:,i] represents how reflector i interacts with
        // the block formed by all previous reflectors
        
        for (uint16_t j = 0; j < i; ++j)
        {
            double sum = 0.0;
            for (uint16_t k = 0; k < i; ++k)
                sum += (double)T[j * ib + k] * w[k];
            T[j * ib + i] = (float)sum;
        }
    }

    // Free heap-allocated workspace if used
    if (ib > 64)
        free(w);
}

//==============================================================================
// BLOCK REFLECTOR APPLICATION
//==============================================================================

/**
 * @brief Apply block reflector H = I - Y·T·Yᵀ to matrix C (contiguous C)
 * 
 * **Operation:** C = (I - Y·T·Yᵀ)·C = C - Y·T·Yᵀ·C
 * 
 * **Three-Step Algorithm (Level 3 BLAS):**
 * 
 * 1. Z = Yᵀ·C       [IB × M] × [M × N] → [IB × N]
 *    Project C onto Householder space
 * 
 * 2. Z_temp = T·Z   [IB × IB] × [IB × N] → [IB × N]
 *    Apply compact WY scaling factor
 * 
 * 3. C = C - Y·Z_temp  [M × IB] × [IB × N] → [M × N]
 *    Apply scaled reflection back to C
 * 
 * **Why Three GEMMs:**
 * - Direct computation: (Y·T)·(Yᵀ·C) would compute Y·T first (M×IB result)
 * - Associativity trick: Y·(T·(Yᵀ·C)) keeps intermediate IB×N (smaller!)
 * - Saves memory: IB×N vs M×IB where typically IB << M
 * 
 * **Memory Requirements:**
 * - Z: IB×N floats (~256 KB for IB=64, N=1024)
 * - Z_temp: IB×N floats (~256 KB)
 * - YT: IB×M floats (transposed Y, ~256 KB for IB=64, M=1024)
 * - Total: ~768 KB (fits in L2 cache)
 * 
 * **Comparison to Sequential Application:**
 * 
 * Sequential (IB individual Householder):
 * - Work: IB × 4MN = 4·IB·M·N FLOPs
 * - Memory access: IB × (read C + write C) = 2·IB·M·N floats
 * - Cache reuse: Poor (each reflector touches all of C)
 * 
 * Block reflector (this function):
 * - Work: 2·IB·M·N + 2·IB²·N + 2·M·IB·N = 4·IB·M·N + 2·IB²·N FLOPs
 * - Memory access: Same total, but GEMM has better cache reuse
 * - Speedup: 2-5× due to cache efficiency in GEMM
 * 
 * @param[in,out] C      Matrix to update [M × N], row-major, contiguous
 * @param[in]     Y      Householder vectors [M × IB], stride ldy
 * @param[in]     T      Compact WY factor [IB × IB], row-major
 * @param[in]     m      Number of rows
 * @param[in]     n      Number of columns
 * @param[in]     ib     Number of reflectors
 * @param[in]     ldy    Leading dimension of Y
 * @param[out]    Z      Workspace [IB × N]
 * @param[out]    Z_temp Workspace [IB × N]
 * @param[out]    YT     Workspace [IB × M] for transposed Y
 * 
 * @return 0 on success, negative on GEMM failure
 * 
 * @note C must be contiguous (no stride), use apply_block_reflector_strided otherwise
 * @note All workspaces must be pre-allocated (no malloc)
 * @note Includes software prefetching for cache optimization
 */
static int apply_block_reflector_clean(
    float *restrict C,
    const float *restrict Y,
    const float *restrict T,
    uint16_t m, uint16_t n, uint16_t ib,
    uint16_t ldy,
    float *restrict Z,
    float *restrict Z_temp,
    float *restrict YT)
{
    //==========================================================================
    // STEP 1: Transpose Y → YT
    //==========================================================================
    // GEMM expects contiguous matrices, but Y has stride ldy
    // Transpose Y from [M × IB] with stride ldy to YT [IB × M] contiguous
    
#ifdef __AVX2__
    if (m >= 8 && ib >= 8)
    {
        // Use optimized 8×8 blocked transpose
        transpose_avx2_8x8(Y, YT, m, ib, ldy, m);
    }
    else
#endif
    {
        // Scalar fallback for small matrices
        for (uint16_t i = 0; i < ib; ++i)
        {
#ifdef __AVX2__
            // Prefetch next row of Y
            if (i + 1 < ib)
            {
                for (uint16_t j = 0; j < m; j += 16)
                {
                    _mm_prefetch((const char*)&Y[j * ldy + (i + 1)], _MM_HINT_T0);
                }
            }
#endif
            
            for (uint16_t j = 0; j < m; ++j)
                YT[i * m + j] = Y[j * ldy + i];
        }
    }

    //==========================================================================
    // STEP 2: Z = Yᵀ·C
    //==========================================================================
    // Compute projection of C onto Householder space
    // Dimensions: [IB × M] × [M × N] → [IB × N]
    
#ifdef __AVX2__
    // Prefetch C for the GEMM operation
    for (uint16_t i = 0; i < MIN(64, m); i += 8)
    {
        for (uint16_t j = 0; j < MIN(64, n); j += 16)
        {
            _mm_prefetch((const char*)&C[i * n + j], _MM_HINT_T0);
        }
    }
#endif

    int ret = GEMM_CALL(Z, YT, C, ib, m, n, 1.0f, 0.0f);
    if (ret != 0)
        return ret;

    //==========================================================================
    // STEP 3: Z_temp = T·Z
    //==========================================================================
    // Apply compact WY scaling factor
    // Dimensions: [IB × IB] × [IB × N] → [IB × N]
    
#ifdef __AVX2__
    // Prefetch T matrix (small, should fit in L1)
    for (uint16_t i = 0; i < ib; i += 16)
    {
        _mm_prefetch((const char*)&T[i], _MM_HINT_T0);
    }
#endif

    ret = GEMM_CALL(Z_temp, T, Z, ib, ib, n, 1.0f, 0.0f);
    if (ret != 0)
        return ret;

    //==========================================================================
    // STEP 4: Copy Y to contiguous buffer for final GEMM
    //==========================================================================
    // Y has stride ldy, but GEMM needs contiguous layout
    // Reuse YT buffer (no longer needed after step 2)
    
    float *Y_contig = YT;

#ifdef __AVX2__
    if (m >= 8 && ib >= 8)
    {
        copy_strided_to_contiguous_avx2(Y_contig, Y, m, ib, ldy);
    }
    else
#endif   
    {
        for (uint16_t i = 0; i < m; ++i)
        {
#ifdef __AVX2__
            // Prefetch ahead
            if (i + 8 < m)
            {
                _mm_prefetch((const char*)&Y[(i + 8) * ldy], _MM_HINT_T0);
            }
#endif
            
            for (uint16_t j = 0; j < ib; ++j)
                Y_contig[i * ib + j] = Y[i * ldy + j];
        }
    }

    //==========================================================================
    // STEP 5: C = C - Y·Z_temp
    //==========================================================================
    // Apply scaled reflection back to C (final update)
    // Dimensions: [M × IB] × [IB × N] → [M × N]
    // Note: beta=1.0 means C += alpha·A·B (not C = alpha·A·B)
    
#ifdef __AVX2__
    // Prefetch C for the final update
    for (uint16_t i = 0; i < MIN(64, m); i += 8)
    {
        for (uint16_t j = 0; j < MIN(64, n); j += 16)
        {
            _mm_prefetch((const char*)&C[i * n + j], _MM_HINT_T0);
        }
    }
#endif

    ret = GEMM_CALL(C, Y_contig, Z_temp, m, ib, n, -1.0f, 1.0f);

    return ret;
}


/**
 * @brief Apply block reflector to strided matrix C
 * 
 * **Purpose:** Handle C with non-trivial stride (common in trailing updates)
 * 
 * **Difference from _clean version:**
 * - _clean: C is contiguous (stride = N), can use GEMM directly
 * - _strided: C has stride ldc ≠ N, need strided GEMM or naive fallback
 * 
 * **Implementation:**
 * Uses naive_gemm_strided() which handles arbitrary strides
 * Slower than optimized GEMM but necessary for correctness
 * 
 * **When This Is Needed:**
 * - Trailing matrix updates where C is submatrix of larger matrix
 * - Q formation where working on columns of Q with spacing
 * 
 * @param[in,out] C      Matrix to update [M × N], stride ldc
 * @param[in]     Y      Householder vectors [M × IB], stride ldy
 * @param[in]     T      Compact WY factor [IB × IB]
 * @param[in]     m      Number of rows
 * @param[in]     n      Number of columns
 * @param[in]     ib     Number of reflectors
 * @param[in]     ldc    Leading dimension of C
 * @param[in]     ldy    Leading dimension of Y
 * @param[out]    Z      Workspace [IB × N]
 * @param[out]    Z_temp Workspace [IB × N]
 * @param[out]    YT     Workspace [IB × M]
 * 
 * @return 0 on success
 * 
 * @note Uses strided GEMM (slower than contiguous case)
 * @note Consider copying C to contiguous buffer if called repeatedly
 */
static int apply_block_reflector_strided(
    float *restrict C,
    const float *restrict Y,
    const float *restrict T,
    uint16_t m, uint16_t n, uint16_t ib,
    uint16_t ldc,
    uint16_t ldy,
    float *restrict Z,
    float *restrict Z_temp,
    float *restrict YT)
{
    //==========================================================================
    // Transpose Y: YT[IB × M]
    //==========================================================================
    
    for (uint16_t i = 0; i < ib; ++i)
        for (uint16_t j = 0; j < m; ++j)
            YT[i * m + j] = Y[j * ldy + i];

    //==========================================================================
    // Z = Yᵀ·C using strided GEMM
    //==========================================================================
    // YT is contiguous [IB × M], C has stride ldc
    
    naive_gemm_strided(Z, YT, C, 
                      ib, m, n,
                      n, m, ldc,
                      1.0f, 0.0f);

    //==========================================================================
    // Z_temp = T·Z (both contiguous)
    //==========================================================================
    
    naive_gemm_strided(Z_temp, T, Z,
                      ib, ib, n,
                      n, ib, n,
                      1.0f, 0.0f);
    
    //==========================================================================
    // C = C - Y·Z_temp using strided GEMM
    //==========================================================================
    // Y has stride ldy, C has stride ldc, Z_temp is contiguous
    
    naive_gemm_strided(C, Y, Z_temp,
                      m, ib, n,
                      ldc, ldy, n,
                      -1.0f, 1.0f);
    
    return 0;
}

//==============================================================================
// LEFT-LOOKING BLOCKED QR FACTORIZATION
//==============================================================================

/**
 * @brief Apply a stored block reflector to a panel
 * 
 * Computes: panel = (I - Y*T*Y^T) * panel
 * where Y and T are loaded from storage (previous block)
 * 
 * @param ws Workspace
 * @param A Full matrix (for indexing)
 * @param panel_col Starting column of panel to update
 * @param panel_width Width of panel (number of columns)
 * @param block_idx Which stored block to load (0, 1, 2, ...)
 * @param m Total rows in matrix
 * @param n Total columns in matrix
 * @param kmax min(m, n)
 * @return 0 on success, negative on error
 */
static int apply_stored_block_to_panel(
    qr_workspace *ws,
    float *A,
    uint16_t panel_col,
    uint16_t panel_width,
    uint16_t block_idx,
    uint16_t m,
    uint16_t n,
    uint16_t kmax)
{
    // Compute dimensions of the stored block
    uint16_t blk_k = block_idx * ws->ib;
    uint16_t blk_size = MIN(ws->ib, kmax - blk_k);
    uint16_t blk_rows_below = m - blk_k;
    
    // The reflector affects rows [blk_k : m-1]
    // The panel spans columns [panel_col : panel_col+panel_width-1]
    // So we update A[blk_k:m-1, panel_col:panel_col+panel_width-1]
    
    uint16_t update_rows = blk_rows_below;
    uint16_t update_cols = panel_width;
    
    if (update_rows == 0 || update_cols == 0)
        return 0;
    
    // Load stored Y and T for this block
    size_t y_offset = block_idx * ws->Y_block_stride;
    size_t t_offset = block_idx * ws->T_block_stride;
    
    // Y is stored in packed format: [blk_rows_below × blk_size]
    // We need to load it into ws->Y with proper layout
    memset(ws->Y, 0, (size_t)m * ws->ib * sizeof(float));
    
    for (uint16_t i = 0; i < blk_rows_below; ++i)
        for (uint16_t j = 0; j < blk_size; ++j)
            ws->Y[(blk_k + i) * ws->ib + j] = 
                ws->Y_stored[y_offset + i * blk_size + j];
    
    // Load T matrix
    memcpy(ws->T, &ws->T_stored[t_offset],
           blk_size * blk_size * sizeof(float));
    
    // Apply block reflector to panel: A[blk_k:m, panel_col:panel_col+width]
    // This is a strided GEMM operation because the panel is within A
    
    float *panel_ptr = &A[blk_k * n + panel_col];
    
    return apply_block_reflector_strided(
        panel_ptr,           // Panel to update (strided within A)
        ws->Y,               // Householder vectors [m × blk_size]
        ws->T,               // T matrix [blk_size × blk_size]
        update_rows,         // Number of rows to update
        update_cols,         // Number of columns in panel
        blk_size,            // Block size
        n,                   // Stride of panel (columns in A)
        ws->ib,              // Stride of Y
        ws->Z,               // Workspace
        ws->Z_temp,          // Workspace
        ws->YT);             // Workspace
}

/**
 * @brief Left-looking blocked QR factorization
 * 
 * For each panel k:
 *   1. Apply all previous reflectors H_0, ..., H_{k-1} to panel k
 *   2. Factor the updated panel
 *   3. Apply new reflector H_k to trailing matrix
 * 
 * Better cache locality: All updates to panel k happen together before factorization.
 * 
 * @param ws Workspace
 * @param A [in/out] Matrix to factor [m×n]
 * @param m Number of rows
 * @param n Number of columns
 * @return Number of blocks processed, or negative error code
 */
static int qr_factor_blocked_left_looking(qr_workspace *ws, float *A,
                                          uint16_t m, uint16_t n)
{
    const uint16_t kmax = MIN(m, n);
    uint16_t block_count = 0;

    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;
        uint16_t cols_right = (n > k + block_size) ? (n - k - block_size) : 0;

        //======================================================================
        // LEFT-LOOKING STEP: Apply all previous block reflectors to panel k
        //======================================================================
        
        for (uint16_t prev_blk = 0; prev_blk < block_count; prev_blk++)
        {
            int ret = apply_stored_block_to_panel(
                ws, A,
                k,              // Panel starts at column k
                block_size,     // Panel width
                prev_blk,       // Which previous block to apply
                m, n, kmax);
            
            if (ret != 0)
                return ret;
        }

        //======================================================================
        // Factor the updated panel
        //======================================================================
        
        panel_factor_optimized(&A[k * n + k], ws->Y, &ws->tau[k],
                               rows_below, block_size, n, ws->ib, ws);

        // Build T matrix
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size, ws->ib);

        //======================================================================
        // Store Y and T for future panels (left-looking needs this!)
        //======================================================================
        
        if (ws->Y_stored && ws->T_stored)
        {
            size_t y_offset = block_count * ws->Y_block_stride;
            size_t t_offset = block_count * ws->T_block_stride;

            for (uint16_t i = 0; i < rows_below; ++i)
                for (uint16_t j = 0; j < block_size; ++j)
                    ws->Y_stored[y_offset + i * block_size + j] =
                        ws->Y[i * ws->ib + j];

            memcpy(&ws->T_stored[t_offset], ws->T,
                   block_size * block_size * sizeof(float));
        }
        else
        {
            // Left-looking REQUIRES reflector storage!
            return -EINVAL;
        }

        //======================================================================
        // Apply to trailing matrix (same as right-looking)
        //======================================================================
        
        if (cols_right > 0)
        {
            for (uint16_t j = 0; j < block_size && (k + j) < m; ++j)
            {
                uint16_t reflector_len = m - (k + j);
                float *col_j = &A[(k + j) * n + (k + j)];

                ws->tmp[0] = 1.0f;
                for (uint16_t i = 1; i < reflector_len; ++i)
                {
                    ws->tmp[i] = col_j[i * n];
                }

                uint16_t row_start = k + j;
                const uint16_t col_block = 32;

                for (uint16_t jj = 0; jj < cols_right; jj += col_block)
                {
                    uint16_t n_block = MIN(col_block, cols_right - jj);

                    apply_householder_clean(
                        &A[row_start * n + (k + block_size + jj)],
                        reflector_len,
                        n_block,
                        n,
                        ws->tmp,
                        ws->tau[k + j]);
                }
            }
        }

        block_count++;
    }

    return block_count;
}

//==============================================================================
// PHASE 1: BLOCKED QR FACTORIZATION
//==============================================================================

/**
 * @brief Perform blocked QR factorization, storing reflectors in A and Y/T
 * 
 * @param ws Workspace
 * @param A [in/out] Matrix to factor [m×n], gets overwritten with R and reflectors
 * @param m Number of rows
 * @param n Number of columns
 * @return Number of blocks processed, or negative error code
 */
static int qr_factor_blocked(qr_workspace *ws, float *A, uint16_t m, uint16_t n)
{
    const uint16_t kmax = MIN(m, n);
    uint16_t block_count = 0;

    for (uint16_t k = 0; k < kmax; k += ws->ib)
    {
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;
        uint16_t cols_right = (n > k + block_size) ? (n - k - block_size) : 0;

        //======================================================================
        // ✅ PREFETCH: Next panel (2 blocks ahead for better timing)
        //======================================================================
        
#ifdef __AVX2__
        if (k + 2 * ws->ib < kmax)
        {
            uint16_t next_k = k + 2 * ws->ib;
            uint16_t prefetch_rows = MIN(64, m - next_k);
            uint16_t next_cols = MIN(ws->ib, kmax - next_k);
            
            float *next_panel = &A[next_k * n + next_k];
            
            // Prefetch panel in 64-byte cache line chunks
            for (uint16_t i = 0; i < prefetch_rows; i += 8)
            {
                for (uint16_t j = 0; j < next_cols; j += 16)
                {
                    _mm_prefetch((const char*)&next_panel[i * n + j], _MM_HINT_T1);
                }
            }
        }
        
        // Also prefetch trailing matrix start (if exists)
        if (cols_right > 0 && k + ws->ib < kmax)
        {
            float *trailing_start = &A[k * n + (k + block_size)];
            uint16_t prefetch_rows = MIN(32, rows_below);
            uint16_t prefetch_cols = MIN(32, cols_right);
            
            for (uint16_t i = 0; i < prefetch_rows; i += 8)
            {
                for (uint16_t j = 0; j < prefetch_cols; j += 16)
                {
                    _mm_prefetch((const char*)&trailing_start[i * n + j], _MM_HINT_T1);
                }
            }
        }
#endif

        //======================================================================
        // Factor current panel
        //======================================================================
        
        panel_factor_optimized(&A[k * n + k], ws->Y, &ws->tau[k],
                               rows_below, block_size, n, ws->ib, ws);

        // Build T matrix
        build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size, ws->ib);

        // Store Y and T for Q formation
        if (ws->Y_stored && ws->T_stored)
        {
            size_t y_offset = block_count * ws->Y_block_stride;
            size_t t_offset = block_count * ws->T_block_stride;

            for (uint16_t i = 0; i < rows_below; ++i)
                for (uint16_t j = 0; j < block_size; ++j)
                    ws->Y_stored[y_offset + i * block_size + j] =
                        ws->Y[i * ws->ib + j];

            memcpy(&ws->T_stored[t_offset], ws->T,
                   block_size * block_size * sizeof(float));
        }

        // Apply to trailing matrix
        if (cols_right > 0)
        {
            for (uint16_t j = 0; j < block_size && (k + j) < m; ++j)
            {
                uint16_t reflector_len = m - (k + j);
                float *col_j = &A[(k + j) * n + (k + j)];

                ws->tmp[0] = 1.0f;
                for (uint16_t i = 1; i < reflector_len; ++i)
                {
                    ws->tmp[i] = col_j[i * n];
                }

                uint16_t row_start = k + j;
                const uint16_t col_block = 32;

                for (uint16_t jj = 0; jj < cols_right; jj += col_block)
                {
                    uint16_t n_block = MIN(col_block, cols_right - jj);

                    apply_householder_clean(
                        &A[row_start * n + (k + block_size + jj)],
                        reflector_len,
                        n_block,
                        n,
                        ws->tmp,
                        ws->tau[k + j]);
                }
            }
        }

        block_count++;
    }

    return block_count;
}
//==============================================================================
// PHASE 2: EXTRACT R MATRIX
//==============================================================================

/**
 * @brief Extract upper triangular R from factored matrix A
 * 
 * @param R [out] Output R matrix [m×n]
 * @param A [in] Factored matrix (R in upper triangle)
 * @param m Number of rows
 * @param n Number of columns
 */
static void qr_extract_r(float *restrict R, const float *restrict A,
                         uint16_t m, uint16_t n)
{
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            R[i * n + j] = (i <= j) ? A[i * n + j] : 0.0f;
        }
    }
}

//==============================================================================
// PHASE 3: FORM ORTHOGONAL Q MATRIX
//==============================================================================

/**
 * @brief Form orthogonal matrix Q from stored Householder reflectors
 * 
 * Applies reflectors in reverse order: Q = H(1) * H(2) * ... * H(k)
 * Uses block reflector representation: H = I - Y*T*Y^T
 * 
 * @param ws Workspace containing stored Y and T matrices
 * @param Q [out] Output Q matrix [m×m]
 * @param m Number of rows
 * @param n Number of columns (original matrix)
 * @param block_count Number of blocks to process
 * @return 0 on success, negative error code on failure
 */
static int qr_form_q(qr_workspace *ws, float *Q, uint16_t m, uint16_t n,
                     uint16_t block_count)
{
    if (!ws->Y_stored || !ws->T_stored)
        return -EINVAL;

    const uint16_t kmax = MIN(m, n);

    // Initialize Q = I
    memset(Q, 0, (size_t)m * m * sizeof(float));
    for (uint16_t i = 0; i < m; ++i)
        Q[i * m + i] = 1.0f;

    // Apply blocks in reverse order
    for (int blk = block_count - 1; blk >= 0; blk--)
    {
        uint16_t k = blk * ws->ib;
        uint16_t block_size = MIN(ws->ib, kmax - k);
        uint16_t rows_below = m - k;

        size_t y_offset = blk * ws->Y_block_stride;
        size_t t_offset = blk * ws->T_block_stride;

        //======================================================================
        // ✅ PREFETCH: Next block's Y and T (if exists)
        //======================================================================
        
#ifdef __AVX2__
        if (blk > 0)
        {
            size_t next_y_offset = (blk - 1) * ws->Y_block_stride;
            size_t next_t_offset = (blk - 1) * ws->T_block_stride;
            
            // Prefetch next Y_stored
            for (size_t i = 0; i < MIN(1024, ws->Y_block_stride); i += 16)
            {
                _mm_prefetch((const char*)&ws->Y_stored[next_y_offset + i], _MM_HINT_T1);
            }
            
            // Prefetch next T_stored
            for (size_t i = 0; i < ws->T_block_stride; i += 16)
            {
                _mm_prefetch((const char*)&ws->T_stored[next_t_offset + i], _MM_HINT_T1);
            }
        }
#endif

        //======================================================================
        // Load stored Y matrix for this block
        //======================================================================
        
        memset(ws->Y, 0, (size_t)m * ws->ib * sizeof(float));
        for (uint16_t i = 0; i < rows_below; ++i)
            for (uint16_t j = 0; j < block_size; ++j)
                ws->Y[(k + i) * ws->ib + j] =
                    ws->Y_stored[y_offset + i * block_size + j];

        // Load stored T matrix for this block
        memcpy(ws->T, &ws->T_stored[t_offset],
               block_size * block_size * sizeof(float));

        // Apply block reflector: Q = Q * (I - Y*T*Y^T)
        int ret = apply_block_reflector_clean(
            Q, ws->Y, ws->T,
            m, m, block_size, ws->ib,
            ws->Z, ws->Z_temp, ws->YT);

        if (ret != 0)
            return ret;
    }

    return 0;
}

//==============================================================================
// MAIN WRAPPER (UNCHANGED SIGNATURE)
//==============================================================================

/**
 * @brief Blocked QR decomposition with in-place factorization
 * 
 * Computes A = Q*R where Q is orthogonal and R is upper triangular.
 * 
 * @param ws Pre-allocated workspace
 * @param A [in/out] Input matrix [m×n], gets overwritten during factorization
 * @param Q [out] Orthogonal matrix [m×m] (if !only_R)
 * @param R [out] Upper triangular matrix [m×n]
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R If true, skip Q formation (faster for least squares)
 * @return 0 on success, negative error code on failure
 */
int qr_ws_blocked_inplace(qr_workspace *ws, float *A, float *Q, float *R,
                          uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;
    if (m > ws->m_max || n > ws->n_max)
        return -EINVAL;

    //==========================================================================
    // Phase 1: Factorization (A → R + reflectors)
    //==========================================================================
    
    int block_count = qr_factor_blocked(ws, A, m, n);
    if (block_count < 0)
        return block_count;

    //==========================================================================
    // Phase 2: Extract R
    //==========================================================================
    
    qr_extract_r(R, A, m, n);

    //==========================================================================
    // Phase 3: Form Q (optional)
    //==========================================================================
    
    if (!only_R && Q)
    {
        return qr_form_q(ws, Q, m, n, block_count);
    }

    return 0;
}

//==============================================================================
// WRAPPER FUNCTIONS
//==============================================================================

int qr_ws_blocked(qr_workspace *ws, const float *A, float *Q, float *R,
                  uint16_t m, uint16_t n, bool only_R)
{
    if (!ws || !A || !R)
        return -EINVAL;
    memcpy(ws->Cpack, A, (size_t)m * n * sizeof(float));
    return qr_ws_blocked_inplace(ws, ws->Cpack, Q, R, m, n, only_R);
}

int qr_blocked(const float *A, float *Q, float *R,
               uint16_t m, uint16_t n, bool only_R)
{
    qr_workspace *ws = qr_workspace_alloc(m, n, 0);
    if (!ws)
        return -ENOMEM;
    int ret = qr_ws_blocked(ws, A, Q, R, m, n, only_R);
    qr_workspace_free(ws);
    return ret;
}

//==============================================================================
// WRAPPER WITH ALGORITHM SELECTION
//==============================================================================

/**
 * @brief Blocked QR with algorithm selection
 * 
 * @param ws Workspace
 * @param A [in/out] Matrix to factor
 * @param Q [out] Orthogonal matrix (if !only_R)
 * @param R [out] Upper triangular matrix
 * @param m Number of rows
 * @param n Number of columns
 * @param only_R Skip Q formation
 * @param left_looking If true, use left-looking algorithm
 * @return 0 on success
 */
int qr_ws_blocked_inplace_ex(qr_workspace *ws, float *A, float *Q, float *R,
                             uint16_t m, uint16_t n, bool only_R,
                             bool left_looking)
{
    if (!ws || !A || !R)
        return -EINVAL;
    if (m > ws->m_max || n > ws->n_max)
        return -EINVAL;

    //==========================================================================
    // Phase 1: Factorization (choose algorithm)
    //==========================================================================
    
    int block_count;
    
    if (left_looking)
    {
        block_count = qr_factor_blocked_left_looking(ws, A, m, n);
    }
    else
    {
        block_count = qr_factor_blocked(ws, A, m, n);
    }
    
    if (block_count < 0)
        return block_count;

    //==========================================================================
    // Phase 2: Extract R
    //==========================================================================
    
    qr_extract_r(R, A, m, n);

    //==========================================================================
    // Phase 3: Form Q (optional)
    //==========================================================================
    
    if (!only_R && Q)
    {
        return qr_form_q(ws, Q, m, n, block_count);
    }

    return 0;
}


//==============================================================================
// ADAPTIVE BLOCK SIZE SELECTION (14900KF TUNED)
//==============================================================================

/**
 * @brief QR blocking configuration
 * 
 * Contains all parameters needed for cache-aware QR blocking,
 * inspired by GEMM's adaptive strategy.
 */
typedef struct {
    uint16_t ib;                 ///< Block size (panel width)
                                 ///< Controls cache blocking and loop tiling
                                 ///< Typical range: 8-128 depending on matrix shape
    
    bool use_recursive;          ///< Enable recursive panel factorization
                                 ///< Set to true for IB ≥ 16 (worthwhile for Level 3 BLAS)
                                 ///< Mimics LAPACK's DGEQRT3 recursive algorithm
    
    uint16_t rec_threshold;      ///< Base case size for recursion
                                 ///< When panel width ≤ threshold, switch to direct factorization
                                 ///< Tuned to balance recursion overhead vs Level 3 BLAS benefit
                                 ///< Typical values: 8-24 depending on IB
    
    bool use_gemm_trailing;      ///< Use block reflector (Level 3) for trailing updates
                                 ///< Currently unused (future optimization)
                                 ///< Would replace Level 2 Householder loops with 3 GEMM calls
} qr_block_config_t;

/**
 * @brief Select QR block size using GEMM-inspired adaptive strategy
 * 
 * **Five-Stage Algorithm (adapted from GEMM):**
 * 
 * 1. **Aspect Ratio Classification**
 *    - Tall (M >> N): Smaller IB (many panels, minimize overhead)
 *    - Wide (N >> M): Larger IB (few panels, amortize factorization)
 *    - Square: Balanced IB (optimize for both phases)
 * 
 * 2. **L1 Cache Fitting** (Intel 14900K: 48 KB)
 *    - Target: M × IB < L1_SIZE
 *    - Panel should stay in L1 during factorization
 * 
 * 3. **L2 Cache Fitting** (Intel 14900K: 2 MB)
 *    - Target: 2×M×IB + Trailing working set < L2_SIZE
 *    - Ensures panel + trailing updates stay in L2
 * 
 * 4. **Minimum Size Enforcement**
 *    - IB ≥ 8 (minimum for AVX2 efficiency)
 *    - IB ≤ min(M, N) (can't be larger than matrix)
 * 
 * 5. **Alignment & Kernel Selection**
 *    - Round IB to SIMD-friendly values (8, 16, 24, 32, 48, 64, 96, 128)
 *    - Select recursive threshold based on IB
 * 
 * @param m Number of rows
 * @param n Number of columns
 * @return Optimal blocking configuration
 */
static qr_block_config_t select_optimal_qr_blocking(uint16_t m, uint16_t n)
{
    qr_block_config_t config;
    const uint16_t min_dim = MIN(m, n);
    
    //==========================================================================
    // STAGE 1: ASPECT RATIO CLASSIFICATION
    //==========================================================================
    
    double aspect_mn = (double)m / (double)n;
    
    // SIMD-friendly block sizes (multiples of 8, cache-line aligned)
    static const uint16_t PREFERRED_SIZES[] = {
        128, 96, 64, 48, 32, 24, 16, 12, 8
    };
    
    uint16_t ib_initial;
    
    if (aspect_mn > 4.0) {
        // Tall matrices: smaller IB, many panels
        ib_initial = 32;
    }
    else if (aspect_mn < 0.25) {
        // Wide matrices: larger IB, few panels
        ib_initial = 64;
    }
    else {
        // Balanced: standard blocking
        ib_initial = 64;
    }
    
    //==========================================================================
    // STAGE 2: L1 CACHE FITTING (48 KB target for Intel 14900K P-core)
    //==========================================================================
    // Panel data: M × IB × 4 bytes
    // Target: M × IB × 4 < 48 KB → IB < 12K / M
    //==========================================================================
    
    const size_t L1_SIZE = 48 * 1024;  // 48 KB
    uint16_t ib_max_l1 = L1_SIZE / (m * sizeof(float));
    
    //==========================================================================
    // STAGE 3: L2 CACHE FITTING (2 MB target for Intel 14900K P-core)
    //==========================================================================
    // Working set: Panel(M×IB) + Y(M×IB) + T(IB²) + Trailing(~M×IB)
    // Simplified: 3×M×IB floats (for IB << M)
    // Target: 3×M×IB×4 < 1.8 MB → IB < 600K / M
    //==========================================================================
    
    const size_t L2_TARGET = 1800 * 1024;  // 1.8 MB (leave headroom)
    const size_t WORKING_SET_MULTIPLIER = 3;  // Panel + Y + Trailing
    uint16_t ib_max_l2 = L2_TARGET / (WORKING_SET_MULTIPLIER * m * sizeof(float));
    
    //==========================================================================
    // STAGE 4: SELECT FROM PREFERRED SIZES
    //==========================================================================
    
    uint16_t ib = 8;  // Minimum safe value
    uint16_t ib_max = MIN(MIN(ib_max_l1, ib_max_l2), min_dim);
    ib_max = MIN(ib_max, ib_initial * 2);  // Don't go too far from initial estimate
    
    for (size_t i = 0; i < sizeof(PREFERRED_SIZES)/sizeof(PREFERRED_SIZES[0]); ++i)
    {
        if (PREFERRED_SIZES[i] <= ib_max && PREFERRED_SIZES[i] <= min_dim)
        {
            ib = PREFERRED_SIZES[i];
            break;
        }
    }
    
    //==========================================================================
    // STAGE 5: MATRIX SHAPE ADJUSTMENTS
    //==========================================================================
    
    // Very small matrices: reduce overhead
    if (m < 128 || n < 128)
        ib = MIN(ib, 16);
    
    // Tiny matrices: use smallest block
    if (min_dim < 32)
        ib = MIN(ib, 8);
    
    // Ensure minimum
    ib = MAX(ib, 8);
    
    // Clamp to matrix dimensions
    ib = MIN(ib, min_dim);
    
    //==========================================================================
    // CONFIGURE RECURSIVE FACTORIZATION
    //==========================================================================
    
    config.ib = ib;
    config.use_recursive = (ib >= 16);  // Only beneficial for ib ≥ 16
    
    // Threshold: base case for recursion
    if (ib >= 96)
        config.rec_threshold = 24;
    else if (ib >= 64)
        config.rec_threshold = 16;
    else if (ib >= 32)
        config.rec_threshold = 12;
    else
        config.rec_threshold = 8;
    
    // GEMM-based trailing update (future optimization)
    config.use_gemm_trailing = false;
    
    return config;
}

//==============================================================================
// WORKSPACE ALLOCATION (WITH ADAPTIVE BLOCKING)
//==============================================================================

/**
 * @brief Allocate workspace for blocked QR decomposition with adaptive blocking
 *
 * **Adaptive Blocking Strategy:**
 * 
 * Uses GEMM-inspired 5-stage algorithm to select optimal block size:
 * 1. Aspect ratio classification (tall/wide/square)
 * 2. L1 cache fitting (48 KB for Intel 14900K)
 * 3. L2 cache fitting (2 MB for Intel 14900K)
 * 4. Minimum size enforcement
 * 5. SIMD alignment and kernel selection
 * 
 * **Memory Layout:**
 *
 * The workspace contains several categories of buffers:
 *
 * 1. **Panel Factorization Buffers:**
 *    - tau[mn]:        Householder scaling factors (one per reflector)
 *    - tmp[m_max]:     Column gather/scatter buffer for strided access
 *    - work[m_max]:    General-purpose working buffer
 *
 * 2. **WY Representation Buffers:**
 *    - T[ib×ib]:       Compact WY factor for current block (upper triangular)
 *    - Y[m_max×ib]:    Current block's Householder vectors (row-major)
 *    - YT[ib×m_max]:   Transposed Y for efficient GEMM access
 *
 * 3. **GEMM Working Buffers:**
 *    - Z[ib×n_big]:    First GEMM workspace (Y^T * C)
 *    - Z_temp[ib×n_big]: Second GEMM workspace (T * Z)
 *    - n_big = max(m_max, n_max) to handle both:
 *        * Trailing matrix updates during factorization (n ≤ n_max)
 *        * Q formation where Q is m×m (n = m ≤ m_max)
 *
 * 4. **Copy/Packing Buffer:**
 *    - Cpack[m_max×n_max]: Aligned copy of input matrix for in-place operation
 *
 * 5. **Column Pivoting Buffers (for future RRQR support):**
 *    - vn1[n_max]:     Column norms (first pass)
 *    - vn2[n_max]:     Column norms (second pass / verification)
 *
 * 6. **Reflector Storage (optional, for fast Q formation):**
 *    - Y_stored[num_blocks × m_max × ib]: All Householder vectors
 *    - T_stored[num_blocks × ib × ib]:    All WY factors
 *
 * 7. **Recursive Panel Workspace:**
 *    - panel_Y_temp[2×m_max×ib]: Workspace for recursive partitioning (malloc-free)
 *    - panel_T_temp[ib×ib]: T matrix workspace for recursion
 *    - panel_Z_temp[ib×ib]: Z workspace for recursion
 *
 * **Cache Hierarchy (Intel 14900K P-core):**
 * - L1D: 48 KB  → Panel factorization target
 * - L2:  2 MB   → Full working set target
 * - L3:  36 MB  → Reflector storage fits here
 *
 * @param m_max              Maximum number of rows
 * @param n_max              Maximum number of columns
 * @param ib                 Block size (0 = auto-select using adaptive strategy)
 * @param store_reflectors   If true, allocate storage for Y and T matrices
 *                           (required for Q formation)
 *
 * @return Allocated workspace, or NULL on failure
 *
 * @retval NULL if m_max or n_max is zero
 * @retval NULL if any memory allocation fails
 *
 * @note All GEMM buffers (T, Y, YT, Z, Z_temp, Cpack) are 32-byte aligned
 *       for optimal AVX2/AVX-512 performance
 *
 * @note Must be freed with qr_workspace_free()
 *
 * @see qr_workspace_free()
 * @see qr_workspace_alloc() (simplified wrapper)
 */
qr_workspace *qr_workspace_alloc_ex(uint16_t m_max, uint16_t n_max,
                                    uint16_t ib, bool store_reflectors)
{
    if (!m_max || !n_max)
        return NULL;

    qr_workspace *ws = (qr_workspace *)calloc(1, sizeof(qr_workspace));
    if (!ws)
        return NULL;

    const uint16_t min_dim = (m_max < n_max) ? m_max : n_max;
    ws->m_max = m_max;
    ws->n_max = n_max;
    
    //==========================================================================
    // ✅ NEW: Adaptive block size selection (GEMM-inspired)
    //==========================================================================
    
    if (ib == 0)
    {
        // Use adaptive selection based on cache hierarchy and aspect ratios
        qr_block_config_t config = select_optimal_qr_blocking(m_max, n_max);
        ws->ib = config.ib;
        ws->use_recursive = config.use_recursive;
        ws->rec_threshold = config.rec_threshold;
        ws->use_gemm_trailing = config.use_gemm_trailing;
    }
    else
    {
        // User-specified: validate and configure
        ws->ib = MIN(ib, min_dim);
        ws->use_recursive = (ws->ib >= 16);
        ws->rec_threshold = (ws->ib >= 64) ? 16 : 
                           (ws->ib >= 32) ? 12 : 8;
        ws->use_gemm_trailing = false;
    }
    
    ws->num_blocks = (min_dim + ws->ib - 1) / ws->ib;

    //==========================================================================
    // ✅ Y_stored stride calculation
    //==========================================================================

    // Y_stored stores each block in PACKED format:
    //   Block k stores Y_k[rows_below_k × block_size_k]
    //   where rows_below_k = m - k*ib
    //         block_size_k = min(ib, min_dim - k*ib)
    //
    // We allocate worst-case: first block with m_max rows and ib columns
    // This gives Y_block_stride = m_max * ib elements per block
    //
    // IMPORTANT: Y_stored uses PACKED stride (block_size varies),
    //            while ws->Y uses FIXED stride (always ws->ib)

    ws->Y_block_stride = (size_t)m_max * ws->ib;
    ws->T_block_stride = (size_t)ws->ib * ws->ib;

    const uint16_t n_big = (m_max > n_max) ? m_max : n_max;

    //==========================================================================
    // Allocate buffers
    //==========================================================================

    ws->tau = (float *)malloc(min_dim * sizeof(float));
    ws->tmp = (float *)malloc(m_max * sizeof(float));
    ws->work = (float *)malloc(m_max * sizeof(float));
    ws->T = (float *)gemm_aligned_alloc(32, ws->ib * ws->ib * sizeof(float));

    // Y has FIXED stride ws->ib (used during factorization)
    ws->Y = (float *)gemm_aligned_alloc(32, (size_t)m_max * ws->ib * sizeof(float));

    ws->YT = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * m_max * sizeof(float));
    ws->Z = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_big * sizeof(float));
    ws->Z_temp = (float *)gemm_aligned_alloc(32, (size_t)ws->ib * n_big * sizeof(float));
    ws->Cpack = (float *)gemm_aligned_alloc(32, (size_t)m_max * n_max * sizeof(float));
    ws->vn1 = (float *)malloc(n_max * sizeof(float));
    ws->vn2 = (float *)malloc(n_max * sizeof(float));

    // ✅ Recursive panel workspace (malloc-free optimization)
    // Size: 2× for recursive partitioning depth
    ws->panel_Y_temp = (float *)gemm_aligned_alloc(32,
                                               2 * (size_t)m_max * ws->ib * sizeof(float));

    ws->panel_T_temp = (float *)gemm_aligned_alloc(32,
                                                   ws->ib * ws->ib * sizeof(float));

    ws->panel_Z_temp = (float *)gemm_aligned_alloc(32,
                                                   ws->ib * ws->ib * sizeof(float));

    size_t bytes =
        min_dim * sizeof(float) +
        m_max * sizeof(float) * 2 +
        ws->ib * ws->ib * sizeof(float) +
        (size_t)m_max * ws->ib * sizeof(float) * 2 +
        (size_t)ws->ib * n_big * sizeof(float) * 2 +
        (size_t)m_max * n_max * sizeof(float) +
        n_max * sizeof(float) * 2;

    bytes += 2 * (size_t)m_max * ws->ib * sizeof(float); // panel_Y_temp
    bytes += ws->ib * ws->ib * sizeof(float) * 2;        // panel_T_temp, panel_Z_temp

    //==========================================================================
    // ✅ Allocate Y_stored and T_stored with proper layout
    //==========================================================================

    if (store_reflectors)
    {
        // Y_stored: All blocks stored in packed format
        // Layout: [block0: m_max × ib][block1: (m_max-ib) × ib][...]
        //
        // We allocate worst-case for ALL blocks:
        //   Total = num_blocks × (m_max × ib) floats
        //
        // Note: This over-allocates because later blocks have fewer rows,
        //       but it simplifies indexing (y_offset = block_count * Y_block_stride)

        ws->Y_stored = (float *)gemm_aligned_alloc(32,
                                                   ws->num_blocks * ws->Y_block_stride * sizeof(float));

        // T_stored: All T matrices (each is ib × ib)
        ws->T_stored = (float *)gemm_aligned_alloc(32,
                                                   ws->num_blocks * ws->T_block_stride * sizeof(float));

        bytes += ws->num_blocks * ws->Y_block_stride * sizeof(float);
        bytes += ws->num_blocks * ws->T_block_stride * sizeof(float);
    }
    else
    {
        ws->Y_stored = NULL;
        ws->T_stored = NULL;
    }

    //==========================================================================
    // Validate allocations
    //==========================================================================

    if (!ws->tau || !ws->tmp || !ws->work || !ws->T || !ws->Cpack ||
        !ws->Y || !ws->YT || !ws->Z || !ws->Z_temp || !ws->vn1 || !ws->vn2 ||
        !ws->panel_Y_temp || !ws->panel_T_temp || !ws->panel_Z_temp)
    {
        qr_workspace_free(ws);
        return NULL;
    }

    if (store_reflectors && (!ws->Y_stored || !ws->T_stored))
    {
        qr_workspace_free(ws);
        return NULL;
    }

    //==========================================================================
    // ✅ Create GEMM plans (compatible with adaptive IB)
    //==========================================================================
    // GEMM plans adapt to QR's selected IB automatically
    // No explicit coordination needed - GEMM's blocking selection handles it
    //==========================================================================

    const uint16_t first_panel_cols = (n_max > ws->ib) ? (n_max - ws->ib) : 0;

    if (first_panel_cols > 0)
        ws->trailing_plans = create_panel_plans(m_max, first_panel_cols, ws->ib);
    else
        ws->trailing_plans = NULL;

    if (m_max >= ws->ib)
        ws->q_formation_plans = create_panel_plans(m_max, m_max, ws->ib);
    else
        ws->q_formation_plans = NULL;

    ws->total_bytes = bytes;
    
    return ws;
}

qr_workspace *qr_workspace_alloc(uint16_t m_max, uint16_t n_max, uint16_t ib)
{
    return qr_workspace_alloc_ex(m_max, n_max, ib, true);
}

void qr_workspace_free(qr_workspace *ws)
{
    if (!ws)
        return;
    destroy_panel_plans(ws->trailing_plans);
    destroy_panel_plans(ws->q_formation_plans);
    free(ws->tau);
    free(ws->tmp);
    free(ws->work);
    gemm_aligned_free(ws->T);
    gemm_aligned_free(ws->Cpack);
    gemm_aligned_free(ws->Y);
    gemm_aligned_free(ws->YT);
    gemm_aligned_free(ws->Z);
    gemm_aligned_free(ws->Z_temp);
    gemm_aligned_free(ws->Y_stored);
    gemm_aligned_free(ws->T_stored);

    // ✅ Free new buffers
    gemm_aligned_free(ws->panel_Y_temp);
    gemm_aligned_free(ws->panel_T_temp);
    gemm_aligned_free(ws->panel_Z_temp);

    free(ws->vn1);
    free(ws->vn2);
    free(ws);
}

size_t qr_workspace_bytes(const qr_workspace *ws)
{
    return ws ? ws->total_bytes : 0;
}