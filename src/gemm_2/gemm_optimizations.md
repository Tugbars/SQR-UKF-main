# GEMM Optimization Deep Dive

A comprehensive explanation of how modern high‑performance GEMM (General
Matrix--Matrix Multiplication) kernels are structured and optimized,
including all critical components such as α/β handling, blocking
hierarchy, packing, SIMD utilization, tail handling, and transposes.

------------------------------------------------------------------------

## 1. Divide & Conquer over K --- Why GEMM is Blocked

Matrix multiplication is defined as:

\[ C = `\alpha `{=tex}A B + `\beta `{=tex}C \]

For large matrices, the shared dimension `K` is split into smaller
chunks:

\[ C = `\alpha `{=tex}(A_0B_0 + A_1B_1 + `\dots `{=tex}+ A_TB_T) +
`\beta `{=tex}C \]

Each partial product (A_tB_t) fits into cache, allowing re‑use of data
before eviction.

**Reason:** Modern CPUs have a deep memory hierarchy (L1, L2, L3, DRAM).
A straightforward triple loop wastes bandwidth, so we *tile* the
computation into cache‑sized blocks.

**Effect:** Each Aₜ and Bₜ are loaded once into fast memory, and reused
for many FMAs (fused multiply--adds).

------------------------------------------------------------------------

## 2. α / β Handling --- Scaling Once, Accumulating Correctly

### The Mathematical Requirement

\[ C `\leftarrow `{=tex}`\alpha `{=tex}AB + `\beta `{=tex}C \]

When split over K:

\[ C = `\alpha `{=tex}(A_0B_0 + A_1B_1 + ...) + `\beta `{=tex}C \]

β must be applied **exactly once** to the original C.

### Why it matters

If you applied β inside every iteration of the K‑loop, you would shrink
your result repeatedly:

    C = α*A0*B0 + β*C
    C = α*A1*B1 + β*C   // wrong, β applied again

Instead, apply β only once.

### Two Common Strategies

**Option A -- β handled inside the first iteration:**

``` c
for (int t = 0; t < Kblocks; ++t) {
    if (t == 0)
        kernel_store(C, A_t, B_t, α, β); // first block
    else
        kernel_add(C, A_t, B_t, α);      // later blocks
}
```

**Option B -- Pre‑scale once before the loop:**

``` c
// Scale C by β once
scale_matrix(C, β);

for (int t = 0; t < Kblocks; ++t)
    kernel_add(C, A_t, B_t, α);
```

Option B simplifies the kernel (no conditional) and allows pre‑scaling
to be vectorized.

------------------------------------------------------------------------

## 3. Blocking Hierarchy --- Exploiting Cache Levels

The computation is broken into nested levels that match the CPU cache
hierarchy:

  ------------------------------------------------------------------------------
  Level            Symbol      Typical Size        Stored In       Purpose
  ---------------- ----------- ------------------- --------------- -------------
  **Micro‑tile**   Mᴿ×Nᴿ       8×8 or 16×8         CPU registers   Accumulate a
                                                                   small block
                                                                   of C

  **K‑block**      Kᴄ          128--256            L1 cache        Stream A/B
                                                                   panels for
                                                                   reuse

  **Outer tiles**  Mᴄ, Nᴄ      512--2048           L2/L3           Hold packed
                                                                   panels
  ------------------------------------------------------------------------------

Each level ensures that data is reused maximally before eviction from
cache.

**Analogy:** A three‑tier warehouse --- registers (fast shelf), L1/L2
(middle storage), main memory (warehouse).

------------------------------------------------------------------------

## 4. Packing --- Laying Out A and B for SIMD

### Why Pack?

In row‑major memory, `A[i][k]` and `B[k][j]` are strided differently.
Accessing both directly causes cache misses and unaligned loads.

**Packing** copies panels into contiguous buffers:

    A_pack: [ M_block x K_block ] rows contiguous
    B_pack: [ K_block x N_block ] columns contiguous

Packed layout ensures: - Perfectly aligned vector loads - Cache‑friendly
sequential access - Predictable prefetching behavior

### Example Layout

    Original A (row‑major):           Packed A:
    A00 A01 A02 A03 ...               A00 A10 A20 A30 ...
    A10 A11 A12 A13 ...      -->      A01 A11 A21 A31 ...
    A20 A21 A22 A23 ...               ...

Each column becomes contiguous in memory, so the micro‑kernel can
broadcast A values efficiently.

------------------------------------------------------------------------

## 5. Micro‑Kernel Execution

The micro‑kernel computes a **Mᴿ×Nᴿ** tile entirely in registers.\
Example: for AVX‑512, a 16×16 tile.

**Steps inside the kernel:**

1.  **Load/Prefetch** the next vectors of A and B.\
2.  **Broadcast** A's element across a vector register.\
3.  **FMA:** `C_col[j] += A_broadcast * B_col[j]`\
4.  Repeat for all columns of B in the tile.\
5.  After finishing Kc iterations, **C accumulators hold final partial
    sums.**

This is fully unrolled and uses vector FMAs each cycle.

**Register allocation example (AVX‑512):** - ZMM0--ZMM15 → accumulators
(16 columns of C) - ZMM16 → current A value - ZMM17--ZMM31 → B vectors
and temporaries

------------------------------------------------------------------------

## 6. In‑Register Transpose --- Efficient Writeback

Kernels often accumulate partial C blocks in **column‑major order**,
because B is streamed by columns.\
But C must be written in **row‑major order** (for contiguous stores).

**Fix:** Perform an in‑register transpose before storing.

### Example (8×8 AVX2 Transpose)

    Before transpose (column‑major accumulators):
    C00 C10 C20 C30 C40 C50 C60 C70
    C01 C11 C21 C31 C41 C51 C61 C71
    ...

    After transpose (row‑major for store):
    C00 C01 C02 C03 C04 C05 C06 C07
    C10 C11 C12 C13 C14 C15 C16 C17
    ...

This avoids scatter writes and improves store throughput.\
AVX‑512 mask stores can also be used for tails.

------------------------------------------------------------------------

## 7. Tail Handling --- Cleaning Up Remainders

Most matrix dimensions aren't exact multiples of tile sizes. The
remainders are called **tails**.

For example, with a 16×8 kernel and an M=100×N=100 matrix:

-   Full tiles: 6×12 = 72 tiles of 16×8\
-   Row tail: 4×(12×8)\
-   Column tail: (6×16)×4\
-   Corner tail: 4×4

**Approaches:**

-   Use smaller kernels (8×8, 4×8, etc.).\
-   Or (AVX‑512) use **mask registers** so the same kernel can handle
    tails without branching.

------------------------------------------------------------------------

## 8. Prefetching and Double Buffering

While one K‑block is processed, the next block's A and B panels are
**prefetched** into cache.

    prefetch(A + next_offset);
    prefetch(B + next_offset);

This hides DRAM latency and ensures continuous pipeline utilization.\
Some libraries use *double buffering* --- while one packed panel is
used, another is prepared by a background thread or loop.

------------------------------------------------------------------------

## 9. Register Blocking and FMA Scheduling

Efficient micro‑kernels carefully manage registers to maintain full FMA
throughput.

-   **Keep A values** in registers as long as possible.\
-   **Reuse B vectors** across multiple accumulators.\
-   **Unroll** the K loop to overlap loads with FMAs.\
-   **No dependencies:** ensure the CPU can issue one FMA per cycle per
    port.

Example inner loop (conceptual):

``` c
for (int k = 0; k < Kc; ++k) {
    a0 = _mm512_set1_ps(Ap[k*MR + 0]); // broadcast A
    b  = _mm512_load_ps(Bp + k*NR);    // load B
    C0 = _mm512_fmadd_ps(a0, b, C0);   // accumulate
}
```

------------------------------------------------------------------------

## 10. Writeback Policy --- Temporal vs Non‑Temporal

Once a C tile is ready:

-   **Temporal store:** `_mm512_store_ps()` keeps data in cache (used
    when C reused soon).\
-   **Non‑temporal store:** `_mm512_stream_ps()` bypasses cache (used
    for final output).

Large GEMMs often use streaming stores to avoid evicting useful data.

------------------------------------------------------------------------

## 11. Threading and Parallel Partitioning

On multi‑core systems, GEMM splits work among threads:

-   **Row partitioning (outer parallelism):** each thread computes a
    subset of M‑blocks.\
-   **Column partitioning:** threads split N‑blocks.\
-   **NUMA awareness:** threads prefer memory local to their socket.\
-   Thread barriers ensure partial results are written before the next
    K‑block.

High‑end libraries like MKL use hierarchical parallelism (OpenMP or
custom thread pools).

------------------------------------------------------------------------

## 12. Putting It All Together --- GEMM Execution Flow

    1. Scale C by β (once)
    2. For each K-block (size Kᴄ):
        a. Pack A_block and B_panel
        b. For each M- and N-block (sizes Mᴄ, Nᴄ):
            i. Loop over micro-tiles (Mᴿ×Nᴿ):
                - Load packed A and B
                - FMA accumulate in registers
                - Transpose in-register if needed
                - Store or stream results to C
    3. Handle tails (partial rows/cols)

------------------------------------------------------------------------

## 13. Glossary of Core Symbols

  Symbol            Meaning                  Notes
  ----------------- ------------------------ -----------------------
  **Mᴿ, Nᴿ**        Microkernel tile sizes   Fit in SIMD registers
  **Kᴄ**            Inner block size         Fits in L1 cache
  **Mᴄ, Nᴄ**        Outer block sizes        Fit in L2/L3 cache
  **α, β**          Scaling coefficients     β applied once
  **Tail kernel**   Small cleanup block      Handles remainders
  **Panel**         Packed submatrix         Fed into microkernel

------------------------------------------------------------------------

## 14. Optimization Summary

  ------------------------------------------------------------------------
  Optimization                   Purpose              Benefit
  ------------------------------ -------------------- --------------------
  Divide & Conquer (K blocking)  Cache efficiency     High reuse of A and
                                                      B

  α/β Handling                   Correct math         Prevents repeated
                                                      scaling

  Packing                        Contiguous aligned   Fast SIMD loads
                                 data                 

  Blocking Hierarchy             Cache reuse          Scales with cache
                                                      size

  Prefetching                    Hide latency         Keeps pipeline busy

  Register Blocking              Maximize FMA         Full SIMD
                                                      utilization

  Transpose                      Align output rows    Fast contiguous
                                                      stores

  Tail Handling                  Cover edges          Safe and correct
                                                      results

  Non‑Temporal Stores            Cache control        Avoid pollution
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 15. Closing Notes

You now have a complete overview of how industrial GEMM libraries like
BLIS, MKL, OpenBLAS, and Eigen achieve near‑theoretical peak performance
on modern CPUs.

Every piece --- from α/β handling to tail cleanup --- exists to keep the
CPU's FMA units busy **every cycle** while respecting cache and memory
behavior.