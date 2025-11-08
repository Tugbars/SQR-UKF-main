/* SPDX-License-Identifier: MIT */
#ifndef CONTROL_LINALG_SIMD_H
#define CONTROL_LINALG_SIMD_H

/**
 * @file linalg_simd.h
 * @brief Vectorized linear algebra kernels (SIMD-accelerated) + tuning knobs.
 *
 * This header exposes the public APIs for the vectorized kernels you implemented:
 *  - tran         : matrix transpose (blocked AVX2 8x8 micro-kernel + scalar tails)
 *  - mul          : GEMV/GEMM-lite (row-major A*B, AVX2 FMA inner loops, scalar/SSE tails)
 *  - lup          : LU with partial pivoting (rank-1 updates accelerated via AVX2)
 *  - inv          : matrix inverse via LUP + triangular solves (in-place safe per API)
 *  - qr           : QR decomposition (Householder) with vectorized reflectors and blocked updates
 *  - cholupdate   : rank-one Cholesky update/downdate (vectorized axpy lanes)
 *
 * All kernels are safe to compile on non-AVX2 targets; they auto-fallback to scalar paths.
 *
 * Memory layout: row-major throughout.
 * Aliasing: unless stated otherwise, output buffers must not alias inputs.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Standard includes (kept minimal for headers) ---------- */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdalign.h>  // alignas (used later in these files)
#include <float.h>
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#  include <malloc.h>   /* _aligned_malloc / __mingw_aligned_malloc */
#endif

/* ---------- Portability helpers ---------- */
#ifndef RESTRICT
#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#    define RESTRICT restrict
#  else
#    define RESTRICT
#  endif
#endif

/* ---------- Feature gates & tuning knobs ---------- */
/**
 * @def LINALG_SIMD_ENABLE
 * Enable SIMD fast paths when the compiler target supports them.
 * Defaults to 1 if __AVX2__ and __FMA__ are defined; otherwise 0.
 */
#ifndef LINALG_SIMD_ENABLE
#  if defined(__AVX2__) && defined(__FMA__)
#    define LINALG_SIMD_ENABLE 1
#  else
#    define LINALG_SIMD_ENABLE 0
#  endif
#endif




#ifdef __cplusplus
} /* extern "C" */
#endif

/* ---------- Portable aligned allocation (32B by default) ---------- */
#ifndef LINALG_DEFAULT_ALIGNMENT
#  define LINALG_DEFAULT_ALIGNMENT 32
#endif


#endif /* CONTROL_LINALG_SIMD_H */
