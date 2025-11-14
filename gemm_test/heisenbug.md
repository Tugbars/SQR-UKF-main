# 🔍 Understanding “Heisenbugs” in GEMM Kernel Testing  
### Why a GEMM kernel can pass all tests *sometimes* and crash *randomly*

In low-level SIMD code (especially AVX2/AVX-512 GEMM micro-kernels), it’s possible to encounter a situation where:

> **“If the program doesn’t crash, all kernel tests pass.  
> But sometimes it *does* crash, seemingly at random.”**

This type of defect is known as a **Heisenbug** — a bug whose visibility depends on unrelated factors such as memory layout, optimization level, alignment, or timing.

This document explains *why it happens*, *how to recognize it*, and *how to fix it*.

---

# 🧠 What’s Actually Happening?  
### ✔ The GEMM kernel is correct  
### ❌ The **test harness** passes invalid arguments to one of the kernels  
### ✔ That kernel corrupts memory  
### ❌ A *later* kernel crashes due to running on corrupted / misaligned data

This kind of bug gives the illusion that a kernel (e.g., the 8×16 microkernel) is faulty, when in reality:

- The earlier kernel wrote into the wrong locations (due to a wrong call signature).
- The later kernel receives *poisoned* memory or invalid stride values.
- Crash location ≠ the location where the memory was first corrupted.

This creates the misleading effect of “random” failures.

---

# 🧨 What Triggers This Behavior?  
The root cause is:

### ❗ A mismatch between the **actual function signature**  
and the **arguments passed by the test code**

For example:

```c
gemm_1x8_panel_avx2fma_store(
    C_test,         // OK
    Ap, 8,          // WRONG – Ap is interpreted as ldc
    Bp, 16,         // WRONG – Bp is interpreted as Ap
    K,              // WRONG – K is interpreted as b_stride
    8,              // WRONG – Intended jb is interpreted as K
    mask            // WRONG – Mask interpreted as jb
);
```

Because C, A, B, K, and masks all have compatible pointer/size types, the compiler cannot catch it.  
The kernel receives garbage parameters but still executes.

Depending on the heap layout, this may:

- accidentally write into valid memory → **test passes**
- overwrite heap metadata → **later crash in malloc/free**
- overwrite C/B/A buffers → **later kernel crashes**
- overwrite alignment padding → **later AVX load faults**

Hence the “works sometimes” symptom.

---

# 🎲 Why Does It Seem Random?

Memory corruption from invalid parameters depends on:

- heap placement (malloc randomness)
- alignment boundaries (32/64-byte alignment blocks)
- stack layout changes between runs
- compiler optimizations (inlining changes stack frame)
- different OS allocators

This creates **non-deterministic crash behavior**.

If the overwritten area belongs to:

- unused padding → no crash  
- another allocated region → later crash  
- a SIMD-alignment buffer → catastrophic crash  

The kernel itself appears “unstable,” but it is innocent.

---

# 🧩 Symptoms That Point to a Calling-Convention Bug

If you see this combination:

- ✔ Kernel math always matches reference  
- ✔ Crash happens before or during an unrelated later kernel  
- ✔ Running with ASan does not always catch the issue  
- ✔ Reordering test functions changes where it crashes  
- ✔ In release mode it fails more often than debug mode  
- ✔ “If it doesn’t crash, all tests pass”

…then the *first thing* to check is:

### 🔍 “Are the kernel arguments passed in the correct order?”

---

# 🌟 How to Fix It

Fix the test harness:

1. Double-check signatures against kernel prototypes  
2. Use `static inline wrappers` for each kernel to enforce argument order  
3. Add assertions on parameters inside kernels (`assert(ldc >= n)` etc.)  
4. Run under ASan, UBSan, Valgrind for confirmation  
5. Document the calling convention clearly

Once the wrong argument order is corrected:

- random crashes disappear  
- results become deterministic  
- every kernel remains stable  

---

# 🛡️ How to Prevent This Bug in the Future

### ✔ Always wrap microkernel entry points with typed helpers  
Example:

```c
static inline void call_kernel_1x8(
    float* C, size_t ldc,
    const float* Ap, size_t a_stride,
    const float* Bp, size_t b_stride,
    size_t K, size_t jb, __m256i mask)
{
    gemm_1x8_panel_avx2fma_store(C, ldc, Ap, a_stride, Bp, b_stride, K, jb, mask);
}
```

This ensures the compiler enforces the correct parameter order.

### ✔ Never call kernels directly from test code  
Always go through safe wrappers.

### ✔ Validate dimensions inside the kernel (in debug builds)

---

# 📌 Summary

- GEMM kernels appear unstable only because earlier tests corrupted memory.
- The corruption was caused by **wrong argument order** in test code.
- Later kernels (e.g., 8×16) crash only when heap layout exposes the corruption.
- If no crash happens earlier, *all mathematical results are perfect*.

This is a classic example of a **Heisenbug** caused by **silent memory corruption due to function signature mismatches** — a common hazard in hand-written SIMD code.

