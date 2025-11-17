/**
 * @file qr_debug.h
 * @brief Debug infrastructure for QR decomposition
 */

#ifndef QR_DEBUG_H
#define QR_DEBUG_H

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>

// Debug levels
#define QR_DEBUG_NONE    0
#define QR_DEBUG_ERROR   1
#define QR_DEBUG_WARN    2
#define QR_DEBUG_INFO    3
#define QR_DEBUG_TRACE   4

// Set debug level via compile flag: -DQR_DEBUG_LEVEL=3
#ifndef QR_DEBUG_LEVEL
  #ifdef DEBUG
    #define QR_DEBUG_LEVEL QR_DEBUG_INFO
  #else
    #define QR_DEBUG_LEVEL QR_DEBUG_NONE
  #endif
#endif

// Conditional compilation of debug code
#if QR_DEBUG_LEVEL > QR_DEBUG_NONE
  #define QR_DEBUG_ENABLED 1
#else
  #define QR_DEBUG_ENABLED 0
#endif

//==============================================================================
// NUMERICAL CHECKS
//==============================================================================

// Check if a value is finite (not NaN or Inf)
#define QR_ASSERT_FINITE(x, name) do { \
    if (QR_DEBUG_ENABLED && (!isfinite(x))) { \
        fprintf(stderr, "QR_ASSERT_FINITE failed: %s = %g at %s:%d\n", \
                name, (double)(x), __FILE__, __LINE__); \
        assert(0); \
    } \
} while(0)

// Check matrix has no NaN or Inf
static inline void qr_assert_matrix_finite(const float *M, uint16_t m, uint16_t n, 
                                           const char *name, const char *file, int line) 
{
#if QR_DEBUG_ENABLED
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < n; j++) {
            if (!isfinite(M[i * n + j])) {
                fprintf(stderr, "QR_ASSERT_MATRIX_FINITE: %s[%d,%d] = %g at %s:%d\n",
                        name, i, j, M[i * n + j], file, line);
                assert(0);
            }
        }
    }
#endif
}

#define QR_ASSERT_MATRIX_FINITE(M, m, n, name) \
    qr_assert_matrix_finite((M), (m), (n), (name), __FILE__, __LINE__)

// Check upper triangular structure
static inline void qr_assert_upper_triangular(const float *R, uint16_t m, uint16_t n,
                                              double tol, const char *file, int line)
{
#if QR_DEBUG_ENABLED
    for (uint16_t i = 1; i < m; i++) {
        for (uint16_t j = 0; j < MIN(i, n); j++) {
            if (fabs(R[i * n + j]) > tol) {
                fprintf(stderr, "QR_ASSERT_UPPER_TRIANGULAR: R[%d,%d] = %g (tol=%g) at %s:%d\n",
                        i, j, R[i * n + j], tol, file, line);
                assert(0);
            }
        }
    }
#endif
}

#define QR_ASSERT_UPPER_TRIANGULAR(R, m, n, tol) \
    qr_assert_upper_triangular((R), (m), (n), (tol), __FILE__, __LINE__)

//==============================================================================
// DIMENSION CHECKS
//==============================================================================

#define QR_ASSERT_DIM(condition, msg) do { \
    if (QR_DEBUG_ENABLED && !(condition)) { \
        fprintf(stderr, "QR_ASSERT_DIM failed: %s at %s:%d\n", \
                (msg), __FILE__, __LINE__); \
        assert(0); \
    } \
} while(0)

//==============================================================================
// MATRIX PRINTING
//==============================================================================

static inline void qr_debug_print_matrix(const char *name, const float *M,
                                         uint16_t m, uint16_t n, int level)
{
#if QR_DEBUG_ENABLED
    if (QR_DEBUG_LEVEL >= level) {
        printf("DEBUG: %s [%dx%d]:\n", name, m, n);
        uint16_t max_rows = (m <= 8) ? m : 4;
        uint16_t max_cols = (n <= 8) ? n : 4;
        
        for (uint16_t i = 0; i < max_rows; i++) {
            printf("  ");
            for (uint16_t j = 0; j < max_cols; j++) {
                printf("%8.4f ", M[i * n + j]);
            }
            if (n > max_cols) printf("...");
            printf("\n");
        }
        if (m > max_rows) printf("  ...\n");
    }
#endif
}

#define QR_DEBUG_PRINT(name, M, m, n, level) \
    qr_debug_print_matrix((name), (M), (m), (n), (level))

//==============================================================================
// CHECKPOINT TRACING
//==============================================================================

#if QR_DEBUG_LEVEL >= QR_DEBUG_TRACE
  #define QR_TRACE(fmt, ...) \
    printf("TRACE [%s:%d]: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
  #define QR_TRACE(fmt, ...) ((void)0)
#endif

#if QR_DEBUG_LEVEL >= QR_DEBUG_INFO
  #define QR_INFO(fmt, ...) \
    printf("INFO: " fmt "\n", ##__VA_ARGS__)
#else
  #define QR_INFO(fmt, ...) ((void)0)
#endif

//==============================================================================
// MATRIX COMPARISON
//==============================================================================

static inline double qr_debug_matrix_diff(const float *A, const float *B,
                                          uint16_t m, uint16_t n)
{
    double max_diff = 0.0;
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < n; j++) {
            double diff = fabs(A[i * n + j] - B[i * n + j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    return max_diff;
}

#define QR_ASSERT_MATRIX_EQUAL(A, B, m, n, tol, name) do { \
    if (QR_DEBUG_ENABLED) { \
        double diff = qr_debug_matrix_diff((A), (B), (m), (n)); \
        if (diff > (tol)) { \
            fprintf(stderr, "QR_ASSERT_MATRIX_EQUAL: %s max diff = %g (tol=%g) at %s:%d\n", \
                    (name), diff, (double)(tol), __FILE__, __LINE__); \
            assert(0); \
        } \
    } \
} while(0)

#endif // QR_DEBUG_H