/**
 * @file gemm_config.h
 * @brief Configuration and missing definitions for GEMM
 */

#ifndef GEMM_CONFIG_H
#define GEMM_CONFIG_H

#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

//==============================================================================
// CACHE BLOCKING PARAMETERS (Intel i9-14900K optimized)
//==============================================================================

#ifndef LINALG_BLOCK_MC
#define LINALG_BLOCK_MC 128  // M cache blocking - fits in L2
#endif

#ifndef LINALG_BLOCK_KC  
#define LINALG_BLOCK_KC 256  // K cache blocking - balance between L1 and register pressure
#endif

#ifndef LINALG_BLOCK_JC
#define LINALG_BLOCK_JC 256  // N cache blocking (JC) - good for L2 TLB
#endif

//==============================================================================
// MEMORY MANAGEMENT
//==============================================================================

static inline void* gemm_aligned_alloc(size_t alignment, size_t size)
{
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#elif defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__)
    // GCC on Windows (MinGW)
    return __mingw_aligned_malloc(size, alignment);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    // C11 standard aligned_alloc (preferred for modern GCC)
    // Note: size must be multiple of alignment for C11
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    return aligned_alloc(alignment, aligned_size);
#else
    // POSIX systems (older GCC or non-C11)
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static inline void gemm_aligned_free(void* ptr)
{
#ifdef _MSC_VER
    _aligned_free(ptr);
#elif defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__)
    __mingw_aligned_free(ptr);
#else
    // Standard free works for both aligned_alloc and posix_memalign
    free(ptr);
#endif
}

//==============================================================================
// ADDITIONAL SIMD OPERATIONS
//==============================================================================

/**
 * @brief Horizontal sum of AVX2 vector
 */
static inline float gemm_hsum_ps_avx2(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

//==============================================================================
// FALLBACK SCALAR KERNELS (for cases not covered by AVX2)
//==============================================================================

/**
 * @brief Scalar GEMM for arbitrary small sizes
 * Used when no optimized kernel is available
 */
static inline void gemm_scalar_kernel(
    float *C, const float *A, const float *B,
    size_t M, size_t K, size_t N,
    size_t ldc, float alpha, float beta)
{
    // Apply beta to C
    if (beta == 0.0f) {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Compute alpha * A * B
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            float aik = alpha * A[i * K + k];
            for (size_t j = 0; j < N; j++) {
                C[i * ldc + j] += aik * B[k * N + j];
            }
        }
    }
}

/**
 * @brief 2x2 kernel for tiny matrices
 */
static inline void gemm_2x2_kernel(
    float *C, const float *A, const float *B,
    size_t ldc, float alpha, float beta)
{
    float c00 = beta * C[0];
    float c01 = beta * C[1];
    float c10 = beta * C[ldc];
    float c11 = beta * C[ldc + 1];
    
    c00 += alpha * (A[0]*B[0] + A[1]*B[2]);
    c01 += alpha * (A[0]*B[1] + A[1]*B[3]);
    c10 += alpha * (A[2]*B[0] + A[3]*B[2]);
    c11 += alpha * (A[2]*B[1] + A[3]*B[3]);
    
    C[0] = c00;
    C[1] = c01;
    C[ldc] = c10;
    C[ldc + 1] = c11;
}

/**
 * @brief 3x3 kernel for tiny matrices
 */
static inline void gemm_3x3_kernel(
    float *C, const float *A, const float *B,
    size_t ldc, float alpha, float beta)
{
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < 3; k++) {
                sum += A[i*3 + k] * B[k*3 + j];
            }
            C[i*ldc + j] = beta * C[i*ldc + j] + alpha * sum;
        }
    }
}

//==============================================================================
// PACKING HELPERS (simplified versions for missing cases)
//==============================================================================

/**
 * @brief Generic pack A for any MR
 */
static inline void pack_A_generic(
    float *Ap, const float *A,
    size_t M, size_t K,
    size_t i_start, size_t i_height,
    size_t k_start, size_t k_width)
{
    for (size_t k = 0; k < k_width; k++) {
        for (size_t i = 0; i < i_height; i++) {
            Ap[k * i_height + i] = A[(i_start + i) * K + k_start + k];
        }
        // Pad if necessary
        for (size_t i = i_height; i < ((i_height + 7) & ~7); i++) {
            Ap[k * ((i_height + 7) & ~7) + i] = 0.0f;
        }
    }
}

/**
 * @brief Generic pack B for any NR
 */
static inline void pack_B_generic(
    float *Bp, const float *B,
    size_t K, size_t N,
    size_t k_start, size_t k_height,
    size_t j_start, size_t j_width)
{
    for (size_t k = 0; k < k_height; k++) {
        for (size_t j = 0; j < j_width; j++) {
            Bp[k * j_width + j] = B[(k_start + k) * N + j_start + j];
        }
        // Pad if necessary
        for (size_t j = j_width; j < ((j_width + 7) & ~7); j++) {
            Bp[k * ((j_width + 7) & ~7) + j] = 0.0f;
        }
    }
}

//==============================================================================
// WORKSPACE SIZE CALCULATION
//==============================================================================

static inline size_t gemm_workspace_query(size_t M, size_t K, size_t N)
{
    // Workspace for A packing + B packing
    size_t mc = (M < LINALG_BLOCK_MC) ? M : LINALG_BLOCK_MC;
    size_t kc = (K < LINALG_BLOCK_KC) ? K : LINALG_BLOCK_KC;
    size_t nc = (N < LINALG_BLOCK_JC) ? N : LINALG_BLOCK_JC;
    
    size_t a_size = mc * kc * sizeof(float);
    size_t b_size = kc * nc * sizeof(float);
    
    return a_size + b_size + 64;  // Extra for alignment
}

//==============================================================================
// VERSION INFO
//==============================================================================

static inline const char* gemm_version(void)
{
    return "1.0.0-avx2-i14900k";
}

static inline uint32_t gemm_cpu_features(void)
{
    uint32_t features = 0;
    
#ifdef __AVX2__
    features |= (1 << 0);  // AVX2
#endif
#ifdef __FMA__
    features |= (1 << 1);  // FMA
#endif
#ifdef __AVX512F__
    features |= (1 << 2);  // AVX512
#endif
    
    return features;
}

//==============================================================================
// MISSING KERNEL WRAPPERS FOR ODD SIZES
//==============================================================================

/**
 * @brief Handle cases where M or N < 4 (not covered by AVX2 kernels)
 */
static inline int gemm_tiny_fallback(
    float *C, const float *A, const float *B,
    size_t M, size_t K, size_t N,
    float alpha, float beta)
{
    if (M == 2 && N == 2 && K == 2) {
        gemm_2x2_kernel(C, A, B, N, alpha, beta);
        return 0;
    }
    else if (M == 3 && N == 3 && K == 3) {
        gemm_3x3_kernel(C, A, B, N, alpha, beta);
        return 0;
    }
    else if (M <= 3 || N <= 3) {
        // Use scalar for very tiny matrices
        gemm_scalar_kernel(C, A, B, M, K, N, N, alpha, beta);
        return 0;
    }
    
    return -1;  // Not handled
}

#endif /* GEMM_CONFIG_H */
