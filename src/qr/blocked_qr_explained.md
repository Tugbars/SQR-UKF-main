# Blocked QR Decomposition and Applications

## What is QR Decomposition?

QR decomposition factors a matrix **A** into two components:

**A = Q × R**

Where:
- **Q** is an orthogonal matrix (Q^T·Q = I)
- **R** is an upper triangular matrix

### Why QR Decomposition?

The key advantage is **numerical stability**:

- **Q is perfectly conditioned** (condition number = 1)
- **R contains the same conditioning as A** (not amplified)
- Avoids the catastrophic squaring of condition numbers that occurs with normal equations (A^T·A)

**Example:**
- If A has condition number 10,000
- Normal equations: A^T·A has condition number ≈ 100,000,000 (squared!)
- QR approach: R has condition number ≈ 10,000 (preserved)

By separating Q and R, operations on the system don't amplify existing numerical problems.

---

## Classical vs Blocked QR

### Classical (Non-Blocked) QR Decomposition

**Algorithm:** Householder reflections applied one column at a time

```
For j = 1 to n:
    1. Compute Householder vector vⱼ for column j
    2. Apply reflection: A ← (I - 2vⱼvⱼᵀ) × A
    3. Move to next column
```

**Performance characteristics:**
- Uses **Level 2 BLAS** (matrix-vector operations)
- Each step: `y = A·v` (matrix-vector multiply)
- Memory traffic pattern: loads entire matrix, does O(n) operations, writes back
- **Poor cache reuse** - data constantly moves between cache and main memory
- Slow on modern processors

### Blocked QR Decomposition

**Algorithm:** Process multiple columns at once using compact WY representation

```
For j = 1 to n by block_size:
    1. Factor block of b columns (compute b reflectors)
    2. Accumulate reflectors: W and Y matrices
    3. Apply block update: A ← A - Y·T·(Y^T·A)
       where T is the triangular factor
    4. Move to next block
```

**Performance characteristics:**
- Uses **Level 3 BLAS** (matrix-matrix operations)
- Core operation: `C = C - Y·T·(Y^T·C)` uses GEMM (general matrix multiply)
- Memory traffic pattern: loads data into cache, does O(n³) operations, writes back
- **Excellent cache reuse** - data stays in cache while many operations are performed
- **5-10× faster** than classical approach on modern hardware

---

## Level 2 vs Level 3 BLAS Optimization

### Why BLAS Levels Matter

Modern CPU hierarchy:
```
Registers:    ~1 cycle access,  tiny capacity
L1 Cache:     ~4 cycles,        32-64 KB
L2 Cache:     ~12 cycles,       256 KB - 1 MB
L3 Cache:     ~40 cycles,       8-32 MB
Main Memory:  ~200 cycles,      GB scale
```

**The problem:** Memory is 50-200× slower than compute!

### Level 2 BLAS (Matrix-Vector)

**Example:** `y = A·x`

```
For each row i:
    y[i] = Σⱼ A[i,j] × x[j]
```

**Performance:**
- **2mn operations** (m×n matrix)
- **mn memory loads** (read entire matrix A)
- **Arithmetic intensity:** 2 flops per memory access
- Cache misses dominate - matrix doesn't fit in cache
- **Bandwidth bound** - CPU waits for data

### Level 3 BLAS (Matrix-Matrix)

**Example:** `C = A·B`

```
For blocks:
    Load A_block, B_block into cache
    C_block = A_block · B_block  (many operations!)
    Write C_block back
```

**Performance:**
- **2mnk operations** (m×n matrix times n×k matrix)
- **mn + nk + mk memory accesses** (load A, B, write C)
- **Arithmetic intensity:** ~2n flops per memory access (for n×n matrices)
- Data stays in cache for many operations
- **Compute bound** - CPU does useful work while data is hot in cache

### The Blocked QR Advantage

**Non-blocked QR:**
```
for each column j:
    v = compute_householder(A[:,j])    // Level 1
    A = apply_reflection(A, v)         // Level 2: A ← (I - 2vvᵀ)A
                                       // Matrix-vector multiply
```
- Arithmetic intensity: ~2 flops/memory access
- Poor cache reuse

**Blocked QR:**
```
for each block:
    [Y, T] = compute_block_householder(A[:,j:j+b])  // b reflectors
    A = A - Y·T·(Yᵀ·A)                              // Level 3: GEMM!
                                                     // Matrix-matrix multiply
```
- Arithmetic intensity: ~2b flops/memory access
- Excellent cache reuse
- For b=32: **32× more work per memory access**

**Real-world impact:**
- On Intel/AMD CPUs with AVX-512: 5-10× speedup
- The larger the matrix, the bigger the advantage
- Blocked algorithms scale better with matrix size

---

## Hierarchical Blocked QR

For very large matrices, use a two-level blocking strategy:

```
┌─────────────────────────────────────┐
│  Large Matrix A (split into blocks) │
│                                      │
│  Block 1  →  QR  →  R₁              │
│  Block 2  →  QR  →  R₂              │
│    ...              ...              │
│  Block n  →  QR  →  Rₙ              │
└─────────────────────────────────────┘
           ↓
    ┌─────────────┐
    │  R₁         │
    │     R₂      │  ← Stack triangular results
    │        ...  │
    │          Rₙ │
    └─────────────┘
           ↓
        QR decomposition
           ↓
    Final R (combined)
```

**Advantages:**
1. **Parallelization:** Each block can be factored independently
2. **Memory efficiency:** Don't need entire matrix in memory at once
3. **Cache optimization:** Each block fits in cache
4. **Numerical stability:** Hierarchical combining maintains accuracy

This is the approach shown in your uploaded diagram!

---

## QR Decomposition in Square Root UKF

### Why Square Root Filters?

Standard Kalman Filter tracks covariance **P**:
- **P** = E[(x - x̂)(x - x̂)ᵀ]
- Problem: Numerical errors can make P non-positive-definite
- Catastrophic failure: Filter diverges

Square Root UKF tracks **S** where **P = S·Sᵀ**:
- S is the Cholesky factor (lower triangular) or equivalently Sᵀ is upper triangular from QR
- **S automatically maintains positive-definiteness**
- More numerically stable

### QR in the Measurement Update

**Setup:**
After sigma point propagation, you have:
- Predicted state covariance square root: **Sₓ**
- Innovation covariance contributions
- Cross-covariance terms

**Stacked matrix for QR:**
```
┌──────────────────┐
│  √W₁·(Y₁ - ȳ)   │  ← Weighted innovations (measurements)
│  √W₂·(Y₂ - ȳ)   │
│       ...        │
│  √Wₙ·(Yₙ - ȳ)   │
│       Rₙ         │  ← Measurement noise square root
└──────────────────┘
        ↓
    QR decomposition
        ↓
    Upper triangle R = Sᵧ (innovation covariance square root)
```

**Why QR here?**

1. **Combines multiple covariance contributions** into a single square root form
2. **Maintains positive definiteness** automatically (R from QR is always valid)
3. **Numerically stable** - avoids forming Sᵧ·Sᵧᵀ explicitly
4. **Efficient** - triangular result is what you need for update equations

### State Covariance Update

After obtaining **Sᵧ**, update state covariance:

```
Cross-covariance: Pₓᵧ = Sₓ·U
Kalman gain: K = Pₓᵧ·Sᵧ⁻ᵀ·Sᵧ⁻¹

Updated state square root via another QR:
┌─────────────────────┐
│  Sₓ - K·Sᵧ          │
│  √R·K               │  ← Augmented with correction
└─────────────────────┘
        ↓
    QR decomposition
        ↓
    R = Sₓ₊ (updated covariance square root)
```

### Blocked QR in SR-UKF Implementation

For high-dimensional systems (n > 100):

**Benefits of blocking:**
- Sigma point matrix is large: 2n+1 sigma points × n dimensions
- Level 3 BLAS speeds up the factorization significantly
- Critical for real-time applications (robotics, navigation, finance)

**Typical block sizes:**
- 32-64 for AVX2 systems
- 64-128 for AVX-512 systems
- Tune based on cache size and vector width

**Performance impact:**
- Non-blocked: ~O(n³) with poor cache behavior
- Blocked: ~O(n³) but **5-10× faster wall-clock time**
- Essential for real-time filters with n > 50 states

---

## Summary

### QR Decomposition Benefits
- **Numerical stability:** Separates perfect conditioning (Q) from problem difficulty (R)
- **No amplification:** Condition number preserved, not squared
- **Triangular system:** R is easy to solve via back-substitution

### Blocked QR Optimization
- **Level 2 BLAS:** Matrix-vector ops, poor cache reuse, slow
- **Level 3 BLAS:** Matrix-matrix ops, excellent cache reuse, **5-10× faster**
- **Key idea:** Process blocks of columns together to maximize arithmetic intensity

### Square Root UKF Application
- **Maintains positive definiteness** automatically through square root form
- **QR combines covariance contributions** from sigma points
- **Hierarchical blocking** enables parallelization and memory efficiency
- **Critical for real-time systems** with high state dimensions

The connection: Blocked QR brings Level 3 BLAS performance to the numerically stable QR decomposition, making Square Root UKF practical for large-scale real-time systems.
