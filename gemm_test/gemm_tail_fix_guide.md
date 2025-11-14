# Preventing Out‑of‑Bounds Writes in AVX2 GEMM Kernels

### A Practical Guide to Fixing Row and Column Tails

This document explains the essential fix required to prevent Heisenbugs,
freezes, and silent memory corruption in AVX2 GEMM microkernels that
operate on blocked tiles such as 8×8, 8×16, 16×8, 16×6, and 8×6.

The bug is simple:\
**microkernels write a full MR×NR tile even when the actual tile is
smaller.**\
This leads to out‑of‑bounds writes that only appear
intermittently---especially when using aligned allocations.

This guide shows how to fix it cleanly and permanently.

------------------------------------------------------------------------

## 1. The Root Cause

GEMM microkernels assume full tiles:

-   MR = microkernel height (e.g., 8 or 16)\
-   NR = microkernel width (e.g., 8 or 16)

But in general GEMM, the last tile of a block may be:

-   fewer than MR rows (`m < MR`)\
-   fewer than NR columns (`n < NR`)

If the kernel blindly writes MR rows and NR columns, it overflows the
output buffer.

This results in:

-   random freezes\
-   hangs before printing\
-   occasional crashes\
-   all tests passing "by luck"\
-   Heisenbug behaviour caused by heap corruption

------------------------------------------------------------------------

## 2. The Fix Has Two Simple Parts

### ✔️ Part 1 --- **Row Guard**

Only store rows that actually exist.

``` c
if (rr >= m)
    break;
```

Never write all 8 or 16 rows unconditionally.

------------------------------------------------------------------------

### ✔️ Part 2 --- **Column Masking**

Use AVX2 maskstores for column tails.

Given your mask generators:

``` c
__m256i mask_lo, mask_hi;
gemm_build_mask_pair16(n, &mask_lo, &mask_hi);
```

Then:

-   columns 0--7 use `mask_lo`
-   columns 8--15 use `mask_hi` (only if needed)

``` c
_mm256_maskstore_ps(C + rr * ldc + 0, mask_lo, acc_lo[rr]);

if (n > 8)
    _mm256_maskstore_ps(C + rr * ldc + 8, mask_hi, acc_hi[rr]);
```

Never use `_mm256_storeu_ps` for partial width tiles.

------------------------------------------------------------------------

## 3. Universal Safe Store Pattern (Drop‑In Fix)

This fixes every 8×16‑style kernel:

``` c
for (size_t rr = 0; rr < 8; rr++) {

    if (rr >= m) break;          // row guard

    _mm256_maskstore_ps(         // columns 0–7
        C + rr * ldc + 0,
        mask_lo,
        acc_lo[rr]
    );

    if (n > 8)                   // columns 8–15
        _mm256_maskstore_ps(
            C + rr * ldc + 8,
            mask_hi,
            acc_hi[rr]
        );
}
```

------------------------------------------------------------------------

## 4. Why This Fix Works

Before:

-   Kernels compute correct math
-   But write too many rows/columns
-   Silent memory corruption hides until aligned malloc rearranges heap
    layout

After:

-   Every row write is validated
-   Every SIMD write is masked
-   No overflows possible
-   All tests pass consistently
-   No freezes or non‑determinism
-   GEMM becomes stable and safe

------------------------------------------------------------------------

## 5. Apply This Fix to ALL Microkernels

This pattern should be applied to:

-   8×8 store/add
-   8×16 store/add
-   16×8 store/add
-   16×6 store/add
-   8×6 store/add
-   1×8 store/add

Any kernel that writes a full MR×NR tile must use these row and column
protections.

------------------------------------------------------------------------

## 6. Summary

**Always guard rows and mask columns.**\
This eliminates every OOB write and prevents Heisenbugs permanently.

    Row guard:    if (rr >= m) break;
    Column guard: maskstore using mask_lo / mask_hi

If aligned malloc "exposes" instability, it means the kernel was already
writing OOB.

With this fix, aligned alloc becomes completely safe.

------------------------------------------------------------------------

If you want, I can also generate: - A PDF version\
- A richly formatted HTML document\
- A version ready for GitHub docs

Just ask.
