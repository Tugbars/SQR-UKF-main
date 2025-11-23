.intel_syntax noprefix
.globl gemm_kernel_8x16_add_asm
.type  gemm_kernel_8x16_add_asm, @function

# Pure AVX2 two-pass kernel (no spills, no ymm16+)
# Pass 1: left half (cols 0-7)
# Pass 2: right half (cols 8-15)
#
# Register allocation per pass:
#   ymm0-ymm7:  8 row accumulators
#   ymm8:       broadcast temp
#   ymm9:       B vector
#   ymm10-ymm15: unused (available for unrolling)

gemm_kernel_8x16_add_asm:
    push rbx
    push r12
    push r13
    push r14

    mov  r12, rdi             # save A base
    mov  r13, rsi             # save B base
    mov  r14, rdx             # save C base
    lea  rcx, [rcx*4]         # ldc in bytes

    #==========================================================================
    # PASS 1: Left half (cols 0-7)
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
    jz .Ladd_left

.Lkloop_left:
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
    jnz .Lkloop_left

.Ladd_left:
    mov rdx, r14              # C base

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
    # PASS 2: Right half (cols 8-15)
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
    jz .Ladd_right

.Lkloop_right:
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
    jnz .Lkloop_right

.Ladd_right:
    mov rdx, r14              # C base

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

    pop r14
    pop r13
    pop r12
    pop rbx
    vzeroupper
    ret

.size gemm_kernel_8x16_add_asm, .-gemm_kernel_8x16_add_asm


.intel_syntax noprefix
.globl gemm_kernel_8x16_store_asm  
.type  gemm_kernel_8x16_store_asm, @function

# Two-pass: compute left 8x8, then right 8x8
# Uses only 10 registers per pass (8 accum + 2 scratch)

gemm_kernel_8x16_store_asm:
    push rbx
    push r12
    push r13
    
    mov r12, rdi          # save A
    mov r13, rsi          # save B
    lea rcx, [rcx*4]      # ldc in bytes
    
    #=== PASS 1: Left half (cols 0-7) ===
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    
    mov rax, r8           # K counter
    test rax, rax
    jz .Lstore_left
    
.Lkloop_left:
    vmovups ymm9, [rsi]   # B[k][0:7]
    
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
    jnz .Lkloop_left

.Lstore_left:
    mov rax, rdx          # C pointer for stores
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
    
    #=== PASS 2: Right half (cols 8-15) ===
    mov rdi, r12          # restore A
    mov rsi, r13          # restore B
    
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
    
.Lkloop_right:
    vmovups ymm9, [rsi + 32]   # B[k][8:15]
    
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
    jnz .Lkloop_right

.Lstore_right:
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