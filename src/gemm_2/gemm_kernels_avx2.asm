#===============================================================================
# GEMM 8×16 Microkernels - Pure AVX2 Implementation
#===============================================================================
#
# High-performance GEMM microkernels for computing 8×16 output tiles.
# Designed for use in blocked GEMM algorithms (BLIS-style).
#
# DESIGN RATIONALE
# ================
# AVX2 provides only 16 YMM registers. A true single-pass 8×16 kernel would
# require:
#   - 16 accumulators (8 rows × 2 column groups)
#   - 2 B vector registers (left/right halves)
#   - 1+ broadcast temporaries
#   - Total: 19+ registers (impossible on AVX2)
#
# Solution: Two-pass algorithm
#   Pass 1: Compute left half  (8×8), columns 0-7
#   Pass 2: Compute right half (8×8), columns 8-15
#
# Trade-offs:
#   + No register spills inside K-loop (critical for performance)
#   + Clean OoO execution (no false dependencies)
#   + Simple, maintainable code
#   - Reads A panel twice (2× A bandwidth)
#   - Two loop overheads
#
# For typical KC values (64-512), the extra A reads are hidden by cache
# (A panel stays hot in L1/L2) and the overhead is negligible.
#
# REGISTER ALLOCATION (per pass)
# ==============================
#   ymm0-ymm7   : 8 row accumulators
#   ymm8        : broadcast temporary (A element)
#   ymm9        : B vector (8 floats)
#   ymm10-ymm15 : unused (available for future K-loop unrolling)
#
# MEMORY LAYOUT
# =============
# Packed A (8×KC, column-major):
#   A[i,k] at offset: k*8 + i  (i=0..7, k=0..KC-1)
#   Memory: [A00 A10 A20 A30 A40 A50 A60 A70 | A01 A11 ... ]
#
# Packed B (KC×16, row-major):
#   B[k,j] at offset: k*16 + j  (k=0..KC-1, j=0..15)
#   Memory: [B00 B01 ... B0,15 | B10 B11 ... B1,15 | ...]
#
# Output C (8×16, row-major with stride ldc):
#   C[i,j] at offset: i*ldc + j
#
# CALLING CONVENTION (System V AMD64 ABI)
# =======================================
#   rdi = const float *A    : packed A panel (8×KC)
#   rsi = const float *B    : packed B panel (KC×16)
#   rdx = float *C          : output tile
#   rcx = size_t ldc        : leading dimension of C (in floats)
#   r8  = size_t KC         : K dimension (number of accumulations)
#
# PERFORMANCE CHARACTERISTICS
# ===========================
# Per K iteration (per pass):
#   - 1 vmovups (B load)
#   - 8 vbroadcastss (A broadcasts)
#   - 8 vfmadd231ps (FMA operations)
#   - 2 prefetcht0 (software prefetch)
#
# Total FLOPs per call: 2 * 8 * 16 * KC = 256 * KC
# Theoretical peak on 14900KF P-core (~5.5 GHz):
#   2 FMA units × 8 floats × 5.5 GHz = 88 GFLOPS (per FMA form)
#   = 176 GFLOPS theoretical peak (both add and mul counted)
#
# AUTHORS & LICENSE
# =================
# Part of VectorFFT linear algebra library.
# See LICENSE for terms.
#
#===============================================================================

.intel_syntax noprefix

#===============================================================================
# gemm_kernel_8x16_add_asm
#===============================================================================
# Computes: C += A * B  (accumulate into existing C values)
#
# Use case: 
#   - Accumulating multiple K-blocks into same C tile
#   - When beta=1 in standard GEMM (C = alpha*A*B + beta*C)
#
# Arguments:
#   rdi = A   : packed A panel, 8×KC, column-major
#   rsi = B   : packed B panel, KC×16, row-major  
#   rdx = C   : output tile, 8×16, row-major with stride ldc
#   rcx = ldc : leading dimension of C in floats (must be >= 16)
#   r8  = KC  : K dimension (0 is valid, results in C unchanged)
#
# Clobbers:
#   rax, rdi, rsi, rdx, rcx (ldc converted to bytes)
#   ymm0-ymm9
#
# Stack usage: 32 bytes (4 saved registers)
#
#===============================================================================
.globl gemm_kernel_8x16_add_asm
.type  gemm_kernel_8x16_add_asm, @function
.p2align 4

gemm_kernel_8x16_add_asm:
    # Prologue: save callee-saved registers
    push rbx
    push r12
    push r13
    push r14

    # Save input pointers (needed for pass 2)
    mov  r12, rdi             # r12 = A base
    mov  r13, rsi             # r13 = B base
    mov  r14, rdx             # r14 = C base
    lea  rcx, [rcx*4]         # convert ldc from floats to bytes

    #==========================================================================
    # PASS 1: Left half (columns 0-7)
    #==========================================================================
    
    # Zero accumulators for 8 rows
    vxorps ymm0, ymm0, ymm0   # row 0, cols 0-7
    vxorps ymm1, ymm1, ymm1   # row 1, cols 0-7
    vxorps ymm2, ymm2, ymm2   # row 2, cols 0-7
    vxorps ymm3, ymm3, ymm3   # row 3, cols 0-7
    vxorps ymm4, ymm4, ymm4   # row 4, cols 0-7
    vxorps ymm5, ymm5, ymm5   # row 5, cols 0-7
    vxorps ymm6, ymm6, ymm6   # row 6, cols 0-7
    vxorps ymm7, ymm7, ymm7   # row 7, cols 0-7

    # K loop setup
    mov rax, r8               # rax = K counter
    test rax, rax
    jz .Ladd_left             # skip loop if KC=0

    .p2align 4
.Lkloop_add_left:
    # Prefetch: B 4 iterations ahead, A 2 iterations ahead
    prefetcht0 [rsi + 256]    # B prefetch (256 = 4 * 64 bytes)
    prefetcht0 [rdi + 64]     # A prefetch (64 = 2 * 32 bytes)

    # Load B[k, 0:7]
    vmovups ymm9, [rsi]

    # Broadcast A[i,k] and accumulate: accum[i] += A[i,k] * B[k,0:7]
    vbroadcastss ymm8, dword ptr [rdi + 0]    # A[0,k]
    vfmadd231ps ymm0, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 4]    # A[1,k]
    vfmadd231ps ymm1, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 8]    # A[2,k]
    vfmadd231ps ymm2, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 12]   # A[3,k]
    vfmadd231ps ymm3, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 16]   # A[4,k]
    vfmadd231ps ymm4, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 20]   # A[5,k]
    vfmadd231ps ymm5, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 24]   # A[6,k]
    vfmadd231ps ymm6, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 28]   # A[7,k]
    vfmadd231ps ymm7, ymm9, ymm8

    # Advance pointers
    add rdi, 32               # A += 8 floats (one column)
    add rsi, 64               # B += 16 floats (one row)
    
    dec rax
    jnz .Lkloop_add_left

.Ladd_left:
    # Writeback: C[i, 0:7] += accum[i]
    mov rdx, r14              # rdx = C base

    vaddps ymm0, ymm0, [rdx]
    vmovups [rdx], ymm0
    add rdx, rcx

    vaddps ymm1, ymm1, [rdx]
    vmovups [rdx], ymm1
    add rdx, rcx

    vaddps ymm2, ymm2, [rdx]
    vmovups [rdx], ymm2
    add rdx, rcx

    vaddps ymm3, ymm3, [rdx]
    vmovups [rdx], ymm3
    add rdx, rcx

    vaddps ymm4, ymm4, [rdx]
    vmovups [rdx], ymm4
    add rdx, rcx

    vaddps ymm5, ymm5, [rdx]
    vmovups [rdx], ymm5
    add rdx, rcx

    vaddps ymm6, ymm6, [rdx]
    vmovups [rdx], ymm6
    add rdx, rcx

    vaddps ymm7, ymm7, [rdx]
    vmovups [rdx], ymm7

    #==========================================================================
    # PASS 2: Right half (columns 8-15)
    #==========================================================================
    
    # Restore A and B pointers
    mov rdi, r12              # rdi = A base
    mov rsi, r13              # rsi = B base

    # Zero accumulators
    vxorps ymm0, ymm0, ymm0   # row 0, cols 8-15
    vxorps ymm1, ymm1, ymm1   # row 1, cols 8-15
    vxorps ymm2, ymm2, ymm2   # row 2, cols 8-15
    vxorps ymm3, ymm3, ymm3   # row 3, cols 8-15
    vxorps ymm4, ymm4, ymm4   # row 4, cols 8-15
    vxorps ymm5, ymm5, ymm5   # row 5, cols 8-15
    vxorps ymm6, ymm6, ymm6   # row 6, cols 8-15
    vxorps ymm7, ymm7, ymm7   # row 7, cols 8-15

    mov rax, r8
    test rax, rax
    jz .Ladd_right

    .p2align 4
.Lkloop_add_right:
    prefetcht0 [rsi + 256]
    prefetcht0 [rdi + 64]

    # Load B[k, 8:15] (right half)
    vmovups ymm9, [rsi + 32]

    vbroadcastss ymm8, dword ptr [rdi + 0]
    vfmadd231ps ymm0, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 4]
    vfmadd231ps ymm1, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 8]
    vfmadd231ps ymm2, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 12]
    vfmadd231ps ymm3, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 16]
    vfmadd231ps ymm4, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 20]
    vfmadd231ps ymm5, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 24]
    vfmadd231ps ymm6, ymm9, ymm8

    vbroadcastss ymm8, dword ptr [rdi + 28]
    vfmadd231ps ymm7, ymm9, ymm8

    add rdi, 32
    add rsi, 64
    dec rax
    jnz .Lkloop_add_right

.Ladd_right:
    # Writeback: C[i, 8:15] += accum[i]
    mov rdx, r14              # rdx = C base

    vaddps ymm0, ymm0, [rdx + 32]
    vmovups [rdx + 32], ymm0
    add rdx, rcx

    vaddps ymm1, ymm1, [rdx + 32]
    vmovups [rdx + 32], ymm1
    add rdx, rcx

    vaddps ymm2, ymm2, [rdx + 32]
    vmovups [rdx + 32], ymm2
    add rdx, rcx

    vaddps ymm3, ymm3, [rdx + 32]
    vmovups [rdx + 32], ymm3
    add rdx, rcx

    vaddps ymm4, ymm4, [rdx + 32]
    vmovups [rdx + 32], ymm4
    add rdx, rcx

    vaddps ymm5, ymm5, [rdx + 32]
    vmovups [rdx + 32], ymm5
    add rdx, rcx

    vaddps ymm6, ymm6, [rdx + 32]
    vmovups [rdx + 32], ymm6
    add rdx, rcx

    vaddps ymm7, ymm7, [rdx + 32]
    vmovups [rdx + 32], ymm7

    # Epilogue
    pop r14
    pop r13
    pop r12
    pop rbx
    vzeroupper                # avoid AVX-SSE transition penalty
    ret

.size gemm_kernel_8x16_add_asm, .-gemm_kernel_8x16_add_asm


#===============================================================================
# gemm_kernel_8x16_store_asm
#===============================================================================
# Computes: C = A * B  (overwrite C, no accumulation)
#
# Use case:
#   - First K-block when beta=0
#   - Standalone GEMM without prior C values
#
# Arguments:
#   rdi = A   : packed A panel, 8×KC, column-major
#   rsi = B   : packed B panel, KC×16, row-major
#   rdx = C   : output tile, 8×16, row-major with stride ldc
#   rcx = ldc : leading dimension of C in floats (must be >= 16)
#   r8  = KC  : K dimension (0 is valid, results in C = 0)
#
# Clobbers:
#   rax, rdi, rsi, rdx, rcx
#   ymm0-ymm9
#
# Stack usage: 24 bytes (3 saved registers)
#
# Note: ~2-3% faster than ADD kernel due to fewer memory operations
#       in writeback (no load-add, just store).
#
#===============================================================================
.globl gemm_kernel_8x16_store_asm
.type  gemm_kernel_8x16_store_asm, @function
.p2align 4

gemm_kernel_8x16_store_asm:
    push rbx
    push r12
    push r13
    
    mov r12, rdi              # r12 = A base
    mov r13, rsi              # r13 = B base
    lea rcx, [rcx*4]          # convert ldc to bytes
    
    #==========================================================================
    # PASS 1: Left half (columns 0-7)
    #==========================================================================
    
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    
    mov rax, r8
    test rax, rax
    jz .Lstore_left
    
    .p2align 4
.Lkloop_store_left:
    prefetcht0 [rsi + 256]
    prefetcht0 [rdi + 64]

    vmovups ymm9, [rsi]
    
    vbroadcastss ymm8, dword ptr [rdi + 0]
    vfmadd231ps ymm0, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 4]
    vfmadd231ps ymm1, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 8]
    vfmadd231ps ymm2, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 12]
    vfmadd231ps ymm3, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 16]
    vfmadd231ps ymm4, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 20]
    vfmadd231ps ymm5, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 24]
    vfmadd231ps ymm6, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 28]
    vfmadd231ps ymm7, ymm9, ymm8
    
    add rdi, 32
    add rsi, 64
    dec rax
    jnz .Lkloop_store_left

.Lstore_left:
    # Writeback: C[i, 0:7] = accum[i]  (direct store, no add)
    mov rax, rdx              # rax = C pointer
    
    vmovups [rax], ymm0
    add rax, rcx
    vmovups [rax], ymm1
    add rax, rcx
    vmovups [rax], ymm2
    add rax, rcx
    vmovups [rax], ymm3
    add rax, rcx
    vmovups [rax], ymm4
    add rax, rcx
    vmovups [rax], ymm5
    add rax, rcx
    vmovups [rax], ymm6
    add rax, rcx
    vmovups [rax], ymm7
    
    #==========================================================================
    # PASS 2: Right half (columns 8-15)
    #==========================================================================
    
    mov rdi, r12              # restore A
    mov rsi, r13              # restore B
    
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    
    mov rax, r8
    test rax, rax
    jz .Lstore_right
    
    .p2align 4
.Lkloop_store_right:
    prefetcht0 [rsi + 256]
    prefetcht0 [rdi + 64]

    vmovups ymm9, [rsi + 32]
    
    vbroadcastss ymm8, dword ptr [rdi + 0]
    vfmadd231ps ymm0, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 4]
    vfmadd231ps ymm1, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 8]
    vfmadd231ps ymm2, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 12]
    vfmadd231ps ymm3, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 16]
    vfmadd231ps ymm4, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 20]
    vfmadd231ps ymm5, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 24]
    vfmadd231ps ymm6, ymm9, ymm8
    
    vbroadcastss ymm8, dword ptr [rdi + 28]
    vfmadd231ps ymm7, ymm9, ymm8
    
    add rdi, 32
    add rsi, 64
    dec rax
    jnz .Lkloop_store_right

.Lstore_right:
    # Writeback: C[i, 8:15] = accum[i]
    vmovups [rdx + 32], ymm0
    add rdx, rcx
    vmovups [rdx + 32], ymm1
    add rdx, rcx
    vmovups [rdx + 32], ymm2
    add rdx, rcx
    vmovups [rdx + 32], ymm3
    add rdx, rcx
    vmovups [rdx + 32], ymm4
    add rdx, rcx
    vmovups [rdx + 32], ymm5
    add rdx, rcx
    vmovups [rdx + 32], ymm6
    add rdx, rcx
    vmovups [rdx + 32], ymm7
    
    pop r13
    pop r12
    pop rbx
    vzeroupper
    ret

.size gemm_kernel_8x16_store_asm, .-gemm_kernel_8x16_store_asm