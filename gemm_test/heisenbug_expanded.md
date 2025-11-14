# The Heisenbug in GEMM Kernel Testing

### Why It Sometimes Passes and Sometimes Freezes

### And Why `aligned_alloc()` *exposes* the bug

------------------------------------------------------------------------

## 1. Overview

This document explains a subtle and dangerous class of bugs often seen
in SIMD-optimized GEMM kernel development---**Heisenbugs**: failures
that only appear sometimes, disappear when instrumented, and behave
differently depending on memory layout.

In your project, all kernel tests sometimes pass cleanly, and other
times the test suite freezes before printing:

    === Testing 8x8 Kernel ===

This is not randomness---this is a specific category of memory
corruption amplified by **aligned memory allocation**.

------------------------------------------------------------------------

## 2. Key Insight

**Aligned malloc does not cause crashes.\
Your out‑of‑bounds write *causes* the crash.\
Aligned malloc only makes it visible.**

------------------------------------------------------------------------

## 3. Why This Happens

### 3.1 Normal malloc layout

Typical malloc returns memory like:

    [malloc header][user buffer][padding][unused region]

If a kernel writes beyond the end of a buffer by a few floats, the
corruption often lands in:

-   unused padding\
-   alignment slack\
-   non-critical memory

Result: **no visible crash**.

------------------------------------------------------------------------

### 3.2 Aligned malloc layout

Aligned alloc returns memory on a strict boundary, so buffers become
tightly packed:

    [aligned pointer][object A][object B][object C]

or worse:

    [aligned pointer][filesystem metadata][glibc heap struct][I/O buffers]

Now the same out‑of‑bounds write might overwrite:

-   heap linked‑list metadata\
-   `FILE` internal buffers\
-   future malloc/free bookkeeping\
-   stdout's locking structure\
-   the **call stack of the next test**

Result: **freeze**, **hang**, **segfault**, or **no output**.

This is why you sometimes see:

-   all tests pass\
-   the program freezes before first test\
-   the crash moves around if you add `printf`\
-   the error disappears under ASAN but appears under release mode

This behaviour is the **definition** of a Heisenbug.

------------------------------------------------------------------------

## 4. Why SIMD Kernels Trigger This

SIMD panel kernels often write in blocks:

-   8x8\
-   8x16\
-   16x8\
-   1x8

If **any** kernel writes:

-   1 row too many\
-   1 column too many\
-   uses the wrong ldc\
-   assumes `m = MR` when `m < MR`\
-   uses the wrong `n_tail` mask\
-   loads 32 bytes but only 6 columns exist

...then you silently corrupt memory.

When using unaligned `malloc`:

-   layout is loose\
-   corruption lands in safe padding\
-   **bug remains hidden**

When using aligned malloc:

-   layout is tight\
-   corruption hits live data\
-   **bug becomes visible**

Thus it *looks* like aligned malloc is responsible.\
It is not---it is simply exposing the truth.

------------------------------------------------------------------------

## 5. Symptoms of This Exact Bug

### ❗ Random test pass/fail

### ❗ Freeze before the first kernel runs

### ❗ Crash disappears when adding printfs

### ❗ Crash disappears when using unaligned malloc

### ❗ Crash appears with aligned alloc

### ❗ Crash location moves around

These are all textbook indicators of heap corruption.

------------------------------------------------------------------------

## 6. How to Confirm the Bug (Definitive)

### 6.1 Compile with AddressSanitizer

    CFLAGS="-O2 -fsanitize=address -fno-omit-frame-pointer -g"

Run:

    ./gemm_tests

ASAN will stop at the **exact** out-of-bounds write.

------------------------------------------------------------------------

## 7. Fix Strategy

1.  Verify each kernel writes exactly `m × n` and no more.\
2.  Confirm test harness allocates correct sizes.\
3.  Validate A and B packers use the correct leading dimensions.\
4.  Add canary guards around buffers for debugging.\
5.  Use ASAN to detect the exact failing instruction.

------------------------------------------------------------------------

## 8. Final Notes

-   This Heisenbug is **not random** --- it is reproducible based on
    heap layout.\
-   It is **not caused by aligned malloc** --- it is merely revealed by
    it.\
-   Fixing the single out‑of‑bounds write will make all behaviour
    stable.

------------------------------------------------------------------------

## 9. TL;DR

    If aligned alloc makes your program freeze,
    you **already** had memory corruption.
    The alignment simply stopped hiding it.

This is a feature, not a downside.

------------------------------------------------------------------------
