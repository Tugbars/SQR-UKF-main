# Applying QR Optimizations to GEMM

Here are **actionable optimizations** from the QR kernels that can directly improve your GEMM performance:

---

## **1. Fast-Path Specialization (10-20% gain)**

### **QR Pattern:**
```c
// Check once, then use branchless inner loop
bool equal_lengths = (len0 == len1) && (len1 == len2) && ...;

if (equal_lengths) {
    // FAST PATH: No conditionals
    for (r = 0; r + 3 < r_end; r += 4) {
        v0 = vp0[r * n];  // Always safe
        v1 = vp1[r * n];  // Always safe
    }
}
```

### **GEMM Application:**
```c
// In gemm_core(), check if tiles are full-sized
static void gemm_panel_kernel(float *C, const float *Ap, const float *Bp,
                              size_t ib_tile, size_t jb_tile, size_t Kblk,
                              size_t ib_expected, size_t jb_expected)
{
    // Check if we have a "clean" tile (no edge cases)
    const bool is_full_tile = (ib_tile == ib_expected) && (jb_tile == jb_expected);
    
    if (is_full_tile) {
        // FAST PATH: No bounds checking in inner loops
        switch (shape) {
            case K8x8:
                gemm_8x8_fast(C, Ap, Bp, Kblk, N);  // New fast kernel
                break;
            case K16x8:
                gemm_16x8_fast(C, Ap, Bp, Kblk, N);
                break;
        }
    } else {
        // SLOW PATH: Use existing edge-case kernels
        switch (shape) {
            case K8x8:
                gemm_8x8_panel_avx2fma_store(C, Ap, Bp, ib_tile, jb_tile, Kblk, N);
                break;
        }
    }
}
```

### **Fast Kernel Example (8×8):**
```c
static void gemm_8x8_fast(float *C, const float *Ap, const float *Bp,
                          size_t Kblk, size_t ldc)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    // ... acc2-acc7
    
    // NO edge checks - we know we have full 8×8 tile
    for (size_t k = 0; k < Kblk; ++k) {
        __m256 a = _mm256_load_ps(Ap + k*8);      // Always aligned
        const float *brow = Bp + k*8;
        
        // Unroll all 8 columns (no conditionals)
        acc0 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(brow+0), acc0);
        acc1 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(brow+1), acc1);
        acc2 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(brow+2), acc2);
        acc3 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(brow+3), acc3);
        acc4 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(brow+4), acc4);
        acc5 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(brow+5), acc5);
        acc6 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(brow+6), acc6);
        acc7 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(brow+7), acc7);
    }
    
    // Writeback (always full 8×8)
    for (int i = 0; i < 8; ++i) {
        __m256 c = _mm256_loadu_ps(C + i*ldc);
        c = _mm256_add_ps(c, /* extract row i from accumulators */);
        _mm256_storeu_ps(C + i*ldc, c);
    }
}
```

---

## **2. K-Loop Unrolling by 4 (5-10% gain)**

### **Current GEMM (1 K per iteration):**
```c
for (size_t k = 0; k < Kblk; ++k) {
    __m256 a = _mm256_load_ps(Ap + k*MR);
    // ... FMAs
}
```

### **QR-Style Unrolling (4 K per iteration):**
```c
for (size_t k = 0; k + 3 < Kblk; k += 4) {
    // Iteration 0
    __m256 a0 = _mm256_load_ps(Ap + (k+0)*8);
    const float *b0 = Bp + (k+0)*8;
    acc0 = _mm256_fmadd_ps(a0, _mm256_broadcast_ss(b0+0), acc0);
    acc1 = _mm256_fmadd_ps(a0, _mm256_broadcast_ss(b0+1), acc1);
    // ... acc2-acc7
    
    // Iteration 1
    __m256 a1 = _mm256_load_ps(Ap + (k+1)*8);
    const float *b1 = Bp + (k+1)*8;
    acc0 = _mm256_fmadd_ps(a1, _mm256_broadcast_ss(b1+0), acc0);
    acc1 = _mm256_fmadd_ps(a1, _mm256_broadcast_ss(b1+1), acc1);
    
    // Iterations 2, 3...
}

// Remainder
for (; k < Kblk; ++k) {
    // Single iteration
}
```

**Benefit:** Hides FMA latency (4-5 cycles), better instruction scheduling

---

## **3. Load Hoisting for B (20-30% gain for wide matrices)**

### **QR Pattern (loads Z once, reuses across rows):**
```c
// Load Z values OUTSIDE row loop
const __m256 z0 = _mm256_loadu_ps(Z + p*kc + j);
const __m256 z1 = _mm256_loadu_ps(Z + (p+1)*kc + j);
// ...

// Inner loop: only load V and C, reuse Z
for (r = 0; r < rows; ++r) {
    v0 = broadcast(V[r]);      // New load
    // ... compute using v0 and z0 (already loaded)
}
```

### **GEMM Application:**
```c
static void gemm_8x8_hoisted(float *C, const float *Ap, const float *Bp,
                             size_t Kblk, size_t ldc)
{
    __m256 acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = _mm256_setzero_ps();
    
    for (size_t k = 0; k < Kblk; ++k) {
        // Load B row once, hoist broadcasts
        const float *brow = Bp + k*8;
        __m256 b0 = _mm256_broadcast_ss(brow+0);
        __m256 b1 = _mm256_broadcast_ss(brow+1);
        __m256 b2 = _mm256_broadcast_ss(brow+2);
        __m256 b3 = _mm256_broadcast_ss(brow+3);
        __m256 b4 = _mm256_broadcast_ss(brow+4);
        __m256 b5 = _mm256_broadcast_ss(brow+5);
        __m256 b6 = _mm256_broadcast_ss(brow+6);
        __m256 b7 = _mm256_broadcast_ss(brow+7);
        
        // Now load A once, use with all 8 B broadcasts
        __m256 a = _mm256_load_ps(Ap + k*8);
        
        acc[0] = _mm256_fmadd_ps(a, b0, acc[0]);
        acc[1] = _mm256_fmadd_ps(a, b1, acc[1]);
        acc[2] = _mm256_fmadd_ps(a, b2, acc[2]);
        acc[3] = _mm256_fmadd_ps(a, b3, acc[3]);
        acc[4] = _mm256_fmadd_ps(a, b4, acc[4]);
        acc[5] = _mm256_fmadd_ps(a, b5, acc[5]);
        acc[6] = _mm256_fmadd_ps(a, b6, acc[6]);
        acc[7] = _mm256_fmadd_ps(a, b7, acc[7]);
    }
    
    // Writeback...
}
```

**Benefit:** Better register allocation, fewer memory ops

---

## **4. Non-Temporal Stores (20-30% gain for large C)**

### **QR Pattern:**
```c
const size_t buffer_size = (size_t)ib * kc * sizeof(float);
const int use_streaming = (buffer_size > L3_CACHE_SIZE / 4);

if (use_streaming) {
    _mm256_stream_ps(Y + i*kc + j, acc);
    // ...
    _mm_sfence();
}
```

### **GEMM Application:**
```c
// In gemm_core(), decide based on C size
const size_t C_size = (size_t)M * N * sizeof(float);
const bool use_streaming = (C_size > L3_CACHE_SIZE / 2);  // 18MB threshold

// Pass flag to kernels
static void gemm_8x8_panel_avx2fma_store(float *C, ..., bool stream)
{
    // ... compute accumulators ...
    
    // Writeback
    for (size_t i = 0; i < ib_tile; ++i) {
        __m256 c = _mm256_loadu_ps(C + i*ldc);
        __m256 row = /* extract from accumulators */;
        c = _mm256_add_ps(c, row);
        
        if (stream) {
            _mm256_stream_ps(C + i*ldc, c);
        } else {
            _mm256_storeu_ps(C + i*ldc, c);
        }
    }
    
    if (stream) _mm_sfence();
}
```

---

## **5. Better Prefetching (5-15% gain)**

### **QR Multi-Level Strategy:**
```c
// Panel-ahead prefetch (6 rows ahead)
if (p + 11 < ib) {
    _mm_prefetch((const char*)(A + (k+p+6)*n), _MM_HINT_T0);
}

// Row-ahead prefetch (8 rows ahead)
if (r + 7 < r_end) {
    _mm_prefetch((const char*)(Cpack + (r+8)*kc + j), _MM_HINT_T0);
}
```

### **GEMM Application:**
```c
static void gemm_8x8_fast(float *C, const float *Ap, const float *Bp,
                          size_t Kblk, size_t ldc)
{
    // ... accumulators ...
    
    for (size_t k = 0; k < Kblk; ++k) {
        // Prefetch Ap ahead (2 K-blocks = 64 bytes)
        if (k + 2 < Kblk) {
            _mm_prefetch((const char*)(Ap + (k+2)*8), _MM_HINT_T0);
        }
        
        // Prefetch Bp ahead
        if (k + 2 < Kblk) {
            _mm_prefetch((const char*)(Bp + (k+2)*8), _MM_HINT_T0);
        }
        
        __m256 a = _mm256_load_ps(Ap + k*8);
        // ... FMAs ...
    }
}
```

**Tuning macro:**
```c
#ifndef GEMM_PF_AHEAD
#define GEMM_PF_AHEAD 2  // K-blocks ahead to prefetch
#endif
```

---

## **6. Add 16-Wide Kernel (20-40% gain for wide N)**

### **Inspired by QR's 6×16 blocking:**
```c
// New kernel: 8×16 (dual 8-wide accumulators)
static void gemm_8x16_fast(float *C, const float *Ap, const float *Bp,
                           size_t Kblk, size_t ldc)
{
    __m256 acc00 = _mm256_setzero_ps();  // Cols 0-7
    __m256 acc01 = _mm256_setzero_ps();  // Cols 8-15
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();
    // ... acc20-acc71 (8 rows × 2 chunks = 16 accumulators)
    
    for (size_t k = 0; k < Kblk; ++k) {
        __m256 a = _mm256_load_ps(Ap + k*8);
        const float *brow = Bp + k*16;  // 16 columns now!
        
        // First 8 columns
        __m256 b0 = _mm256_broadcast_ss(brow+0);
        __m256 b1 = _mm256_broadcast_ss(brow+1);
        // ... b2-b7
        
        acc00 = _mm256_fmadd_ps(a, b0, acc00);
        acc10 = _mm256_fmadd_ps(a, b1, acc10);
        // ... acc20-acc70
        
        // Second 8 columns
        __m256 b8  = _mm256_broadcast_ss(brow+8);
        __m256 b9  = _mm256_broadcast_ss(brow+9);
        // ... b10-b15
        
        acc01 = _mm256_fmadd_ps(a, b8,  acc01);
        acc11 = _mm256_fmadd_ps(a, b9,  acc11);
        // ... acc21-acc71
    }
    
    // Writeback (8 rows × 16 cols)
    // ...
}
```

**When to use:**
```c
if (ib_tile >= 8 && jb_tile >= 16) return K8x16;
```

---

## **7. Improved Kernel Picker**

### **Current (broken):**
```c
static enum kernel_shape pick_kernel(size_t Mc, size_t Nc, size_t Kc)
{
    // Uses compile-time constants (always picks K16x8!)
    if (Mc >= 16 && Nc >= 8) return K16x8;
}
```

### **Fixed (QR-style):**
```c
static enum kernel_shape pick_kernel_tile(size_t ib_tile, size_t jb_tile)
{
    // Use ACTUAL tile dimensions (runtime values)
    
    // Check larger kernels first
    if (ib_tile >= 16 && jb_tile >= 16) return K16x16;  // New!
    if (ib_tile >= 16 && jb_tile >= 8)  return K16x8;
    if (ib_tile >= 16 && jb_tile >= 6)  return K16x6;
    
    // Check 8×16 BEFORE 8×8 (important!)
    if (ib_tile >= 8 && jb_tile >= 16)  return K8x16;   // New!
    if (ib_tile >= 8 && jb_tile >= 8)   return K8x8;
    if (ib_tile >= 8 && jb_tile >= 6)   return K8x6;
    
    return K8x8;  // Safe default
}
```

**Use in gemm_core():**
```c
// In M-N loop:
enum kernel_shape shape = pick_kernel_tile(ib_tile, jb_tile);  // ✅ FIXED
```

---

## **8. Summary: Priority Order**

| Optimization | Difficulty | Expected Gain | Priority |
|--------------|-----------|---------------|----------|
| Fix kernel picker | Easy | 10-30% | 🔥 **DO FIRST** |
| Fast-path specialization | Medium | 10-20% | 🔥 **HIGH** |
| Add 8×16 kernel | Medium | 20-40% | 🔥 **HIGH** |
| K-loop unroll by 4 | Easy | 5-10% | ⭐ Medium |
| Load hoisting | Medium | 20-30% | ⭐ Medium |
| Non-temporal stores | Easy | 20-30% (large) | ⭐ Medium |
| Better prefetching | Easy | 5-15% | ⚡ Low |

---

## **9. Quick Wins: What to Do Right Now**

### **Step 1: Fix Kernel Picker (5 minutes)**
```c
// In gemm_core(), replace:
enum kernel_shape shape = pick_kernel(Mc, jb_tile, Kblk);

// With:
enum kernel_shape shape = pick_kernel_tile(ib_tile, jb_tile);
```

### **Step 2: Add Fast-Path Check (30 minutes)**
```c
const bool is_full_tile = (ib_tile == ib_expected) && (jb_tile == jb_expected);
if (is_full_tile) {
    // Call new branchless kernels
}
```

### **Step 3: Add 8×16 Kernel (2 hours)**
Copy your 8×8 kernel, double the column count, test.

### **Step 4: Benchmark**
```bash
gcc -O3 -march=native benchmark_gemm.c gemm.c -o bench
./bench
```

---

**Expected total improvement: 2-4× speedup over current implementation!** 🚀