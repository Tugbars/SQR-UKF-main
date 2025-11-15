# Masked Load Safety Guide: Avoiding Undefined Behavior

## TL;DR

**Masked loads ALWAYS read all lanes from memory, regardless of mask.**

```c
// ❌ DANGEROUS - Reads 8 floats even though only 6 are valid
__m256 data = _mm256_maskload_ps(array_of_6_floats, mask6);

// ✅ SAFE - Explicitly reads only valid elements
__m256 data = _mm256_setr_ps(a[0], a[1], a[2], a[3], a[4], a[5], 0, 0);
```

---

## The Common Misconception

### What Developers Think:
> "The mask tells the CPU which elements to read. If the mask bit is 0, that memory location won't be accessed."

**This is WRONG.**

### What Actually Happens:
The CPU **always** reads all elements from memory. The mask only controls which values are **written to the destination register**.

---

## Intel Intrinsic Behavior

### Masked Load (`_mm256_maskload_ps`)

```c
__m256 _mm256_maskload_ps(float const *mem_addr, __m256i mask);
```

**What it does:**
1. ✅ Reads 8 consecutive floats from `mem_addr` (32 bytes)
2. ✅ For each lane where mask[i] has high bit = 1: writes data[i]
3. ✅ For each lane where mask[i] has high bit = 0: writes 0.0

**Critical point:** Step 1 ALWAYS happens - all 8 floats are read from memory!

---

## When Undefined Behavior Occurs

### Example 1: Array Smaller Than SIMD Width

```c
float data[6] = {1, 2, 3, 4, 5, 6};  // Only 6 floats allocated

__m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);

// ❌ UNDEFINED BEHAVIOR
__m256 vec = _mm256_maskload_ps(data, mask6);
//                               ^^^^
// Reads data[0..7], but data[6] and data[7] don't exist!
// Could read garbage, trigger sanitizers, or segfault
```

**What the CPU tries to read:**
```
data[0] ✓ valid
data[1] ✓ valid
data[2] ✓ valid
data[3] ✓ valid
data[4] ✓ valid
data[5] ✓ valid
data[6] ✗ OUT OF BOUNDS (UB!)
data[7] ✗ OUT OF BOUNDS (UB!)
```

### Example 2: Page Boundary Crossing

```c
// Hypothetical memory layout:
// [Page 1: valid] [Page 2: unmapped]
// data[0..5] are at the END of Page 1
// data[6..7] would be in Page 2 → SEGFAULT!

float *data = get_data_near_page_boundary();  // Returns 6 floats

__m256 vec = _mm256_maskload_ps(data, mask6);
// Reads across page boundary → PAGE FAULT → crash!
```

This is **non-deterministic** - depends on allocator behavior.

### Example 3: Loop Over 2D Array

```c
float matrix[100][6];  // 100 rows, 6 columns each

for (int i = 0; i < 100; i++) {
    // ❌ UNDEFINED BEHAVIOR
    __m256 row = _mm256_maskload_ps(matrix[i], mask6);
    
    // matrix[i] points to 6 floats
    // Load tries to read 8 floats
    // Reads into matrix[i+1][0] and matrix[i+1][1] → data corruption risk!
}
```

---

## Safe Alternatives

### Option 1: Explicit Element Load (Recommended)

```c
// ✅ SAFE - Only reads what exists
__m256 load_6_floats(const float *data) {
    return _mm256_setr_ps(
        data[0], data[1], data[2],
        data[3], data[4], data[5],
        0.0f, 0.0f
    );
}
```

**Pros:**
- No UB, works with any allocation
- Compiler optimizes well (generates efficient scalar loads)
- Only ~2-3 cycles overhead vs. ideal

**Cons:**
- Slightly more verbose

### Option 2: Padded Allocation

```c
// ✅ SAFE - Allocate extra space
float *data = aligned_alloc(32, 8 * sizeof(float));  // Allocate 8, use 6
memset(data, 0, 8 * sizeof(float));

// Initialize only first 6
for (int i = 0; i < 6; i++) {
    data[i] = values[i];
}

// Now masked load is safe
__m256 vec = _mm256_maskload_ps(data, mask6);
```

**Pros:**
- Slightly faster (1 vector load instruction)

**Cons:**
- Wastes memory (33% overhead for 6-element case)
- Requires careful allocation management
- All code must know about padding

### Option 3: Full Load + Blend

```c
// ✅ SAFE - If you have 2 extra bytes padding
float data[8];  // Allocate 8, but only 6 are "valid"

__m256 vec_full = _mm256_loadu_ps(data);  // Load all 8
__m256 vec_masked = _mm256_and_ps(vec_full, (__m256)mask6);  // Zero upper
```

**Pros:**
- Fast (1 load + 1 AND)

**Cons:**
- Still requires 2 extra floats padding
- Upper elements contain arbitrary data (must zero)

### Option 4: Conditional Check

```c
// ✅ SAFE - Runtime safety check
__m256 safe_maskload(const float *data, size_t valid_count) {
    if (valid_count >= 8) {
        // Full SIMD load safe
        return _mm256_loadu_ps(data);
    } else {
        // Scalar fallback
        float temp[8] = {0};
        for (size_t i = 0; i < valid_count; i++) {
            temp[i] = data[i];
        }
        return _mm256_loadu_ps(temp);
    }
}
```

---

## Masked Stores vs. Masked Loads

### Key Difference

| Operation | Memory Access Behavior |
|-----------|------------------------|
| **Masked Load** | Reads ALL lanes from memory, mask controls register | 
| **Masked Store** | Writes ONLY masked lanes to memory |

### Masked Stores ARE Safe

```c
float output[6];  // Only 6 elements allocated

__m256 data = _mm256_set_ps(8, 7, 6, 5, 4, 3, 2, 1);
__m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);

// ✅ SAFE - Only writes output[0..5]
_mm256_maskstore_ps(output, mask6, data);

// Memory after:
// output[0] = 1 ✓
// output[1] = 2 ✓
// output[2] = 3 ✓
// output[3] = 4 ✓
// output[4] = 5 ✓
// output[5] = 6 ✓
// (elements 7 and 8 from data are never written)
```

**Why it's safe:** The CPU only performs memory writes for lanes where the mask bit is set.

---

## Real-World Code Review

### ❌ UNSAFE Pattern (Commonly Seen)

```c
void process_6_channel_audio(float *samples, size_t num_frames) {
    __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);
    
    for (size_t i = 0; i < num_frames; i++) {
        // samples[i*6 .. i*6+5] contains 6 valid floats
        
        // ❌ READS samples[i*6+6] and samples[i*6+7] (UB!)
        __m256 frame = _mm256_maskload_ps(&samples[i * 6], mask6);
        
        // ... process frame ...
    }
}
```

### ✅ SAFE Pattern (Corrected)

```c
void process_6_channel_audio(float *samples, size_t num_frames) {
    for (size_t i = 0; i < num_frames; i++) {
        float *frame_ptr = &samples[i * 6];
        
        // ✅ Explicitly load only valid elements
        __m256 frame = _mm256_setr_ps(
            frame_ptr[0], frame_ptr[1], frame_ptr[2],
            frame_ptr[3], frame_ptr[4], frame_ptr[5],
            0.0f, 0.0f
        );
        
        // ... process frame ...
    }
}
```

**Or with padding:**

```c
void process_6_channel_audio_padded(float *samples, size_t num_frames) {
    // Allocate with padding: 8 floats per frame instead of 6
    // samples must be allocated as: num_frames * 8 * sizeof(float)
    
    __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);
    
    for (size_t i = 0; i < num_frames; i++) {
        // ✅ SAFE - samples has padding
        __m256 frame = _mm256_maskload_ps(&samples[i * 8], mask6);
        
        // ... process frame ...
    }
}
```

---

## Detection and Debugging

### AddressSanitizer (ASan)

```bash
# Compile with sanitizer
gcc -fsanitize=address -g -O2 code.c

# Will catch:
# ==12345==ERROR: AddressSanitizer: heap-buffer-overflow
# READ of size 32 at 0x602000000018
```

### Valgrind

```bash
valgrind --tool=memcheck ./program

# Will report:
# ==12345== Invalid read of size 8
# ==12345==    at 0x4005F2: process (code.c:42)
```

### Static Analysis

```c
// Modern compilers may warn:
warning: '_mm256_maskload_ps' may read beyond end of object
```

---

## Architecture-Specific Notes

### x86-64 (AVX2/AVX-512)
- Masked loads always read full width
- Page faults can occur
- Unaligned access allowed (slower)

### ARM (SVE/NEON)
- Similar behavior - predicated loads read memory
- Check ARM SVE documentation for specifics

### RISC-V (Vector Extension)
- Vector loads with masking read full `VLEN`

**Rule of thumb:** Assume masked loads read full width on ALL architectures.

---

## Checklist for Safe Masked Loads

Before using `_mm256_maskload_ps`:

- [ ] Do I have at least 8 floats allocated (32 bytes)?
- [ ] If not, am I using `_mm256_setr_ps` instead?
- [ ] If using padding, is it documented and allocated correctly?
- [ ] Have I tested with AddressSanitizer/Valgrind?
- [ ] Is this code worth the complexity vs. explicit loads?

**When in doubt, use explicit element loading.**

---

## Summary

| Intrinsic | Memory Read | Register Write | Safe for Partial Arrays? |
|-----------|-------------|----------------|-------------------------|
| `_mm256_loadu_ps` | All 8 floats | All 8 lanes | ❌ No (needs 8 floats) |
| `_mm256_maskload_ps` | All 8 floats | Masked lanes | ❌ No (needs 8 floats) |
| `_mm256_setr_ps` | Individual scalars | All 8 lanes | ✅ Yes (safe with any count) |
| `_mm256_maskstore_ps` | None | Masked lanes only | ✅ Yes (writes only valid) |

---

## Further Reading

- [Intel Intrinsics Guide - maskload](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=maskload)
- [C++ Standard: Undefined Behavior](https://en.cppreference.com/w/cpp/language/ub)
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)

---

**Remember:** Masked loads are for masking the **destination**, not the **source**. Always ensure you have enough allocated memory for a full-width load.
