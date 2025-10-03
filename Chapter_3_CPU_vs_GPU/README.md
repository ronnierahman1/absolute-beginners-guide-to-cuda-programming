# Chapter 3 — Talk to the GPU (CUDA for Absolute Beginners)

This chapter teaches the end-to-end CUDA workflow by building a simple but canonical program: **vector addition (C = A + B)**. You’ll learn how host (CPU) and device (GPU) memory differ, how to allocate/copy/free device memory, how to launch a kernel, how to add robust error checking, and how to verify GPU results against a CPU reference.

## Folder Layout

```

Chapter_3_Talk_to_the_GPU/
│
├── device_alloc_free.cu
├── memcpy_roundtrip.cu
├── vector_add_minimal.cu
├── vector_add_workflow.cu
├── vector_add_with_checks.cu
├── vector_add_verification.cu
│
├── Makefile
└── build_ch3.bat

## Prerequisites

- **CUDA Toolkit** installed (nvcc in PATH)
- An NVIDIA GPU with a compatible driver
- C++ toolchain (gcc/clang on Linux/macOS, MSVC on Windows)

## Quick Start

### Linux/macOS
```bash
make                    # builds all programs into ./bin
make ARCH=sm_86         # pick your GPU arch (sm_61, sm_70, sm_75, sm_86, sm_89)
make MODE=debug         # debug symbols (-G), no optimization (-O0)
./bin/vector_add_verification 1000000 1e-5
````

### Windows

```bat
build_ch3.bat                   # builds all into .\bin\
build_ch3.bat ARCH=sm_86        # set architecture
build_ch3.bat MODE=debug        # debug build
build_ch3.bat vector_add_with_checks
.\bin\vector_add_verification.exe 1000000 1e-5
```

## Program Sequence (What to Run and Why)

1. **device_alloc_free.cu**
   Smoke test for `cudaMalloc`/`cudaFree`; prints device memory before/after.

2. **memcpy_roundtrip.cu**
   Verifies Host→Device→Host copies are **byte-exact** with a round-trip.

3. **vector_add_minimal.cu**
   Smallest end-to-end vector addition kernel (no macros, minimal checks).

4. **vector_add_workflow.cu**
   Same compute, but organized explicitly by the **five-step CUDA workflow**.

5. **vector_add_with_checks.cu**
   Adds `CUDA_CHECK`, `cudaGetLastError`, `cudaDeviceSynchronize` — fail fast.

6. **vector_add_verification.cu**
   Final “teaching-quality” version with **CPU reference + tolerance** check.

## Common Issues

* **`undefined reference to main`**: You built a file without a `main()`—use the provided .cu files.
* **`out of memory`**: Reduce `N` or use a GPU with more memory.
* **All zeros / wrong results**: Check launch params and ensure H2D/D2H copies are in the right direction.
* **Illegal memory access**: Ensure the kernel has `if (i < N)` bounds checks.

## Next Steps

* Try changing the kernel to `C[i] = 10*A[i] + B[i]`.
* Switch to `int` arrays and require exact equality for verification.
* Time CPU vs GPU for different `N` to see where the GPU wins.

````

---

# `device_alloc_free_README.md`

```markdown
# device_alloc_free.cu

**Purpose:** Prove your environment can allocate and free GPU memory.

**What it does:**
- Prints device info and free/total memory
- Allocates 3 float buffers (`d_A`, `d_B`, `d_C`)
- Prints memory again, then frees and prints once more

**Build/Run:**
```bash
nvcc device_alloc_free.cu -o device_alloc_free
./device_alloc_free           # default N ≈ 1,048,576
./device_alloc_free 50000000  # custom N
````

**Expected output:** Memory drop after allocation and return after free.
**You should see:** Memory drop after allocation and return after free.

````

---

# `memcpy_roundtrip_README.md`

```markdown
# memcpy_roundtrip.cu

**Purpose:** Validate host↔device copies work correctly.

**What it does:**
- Fills a host buffer with a deterministic pattern
- H2D to `d_buf`, then D2H into `h_dst`
- Confirms **byte-exact** equality (round-trip)

**Build/Run:**
```bash
nvcc memcpy_roundtrip.cu -o memcpy_roundtrip
./memcpy_roundtrip
````

**PASS means:** Transfers are correct and deterministic.

````

---

# `vector_add_minimal_README.md`

```markdown
# vector_add_minimal.cu

**Purpose:** The smallest complete GPU compute example.

**What it does:**
- Initializes A[i]=i, B[i]=2*i on host
- Copies to device, launches `C[i]=A[i]+B[i]`
- Copies C back and spot-checks results

**Build/Run:**
```bash
nvcc vector_add_minimal.cu -o vector_add_minimal
./vector_add_minimal 1000000
````

**Notes:** Minimal checks only; later files add safety nets.

````

---

# `vector_add_workflow_README.md`

```markdown
# vector_add_workflow.cu

**Purpose:** Teach the **five-step CUDA workflow** as structure.

**Five steps shown:**
1. Allocate (device)  
2. Copy (to device)  
3. Compute (kernel)  
4. Copy back (to host)  
5. Free (device)

**Build/Run:**
```bash
nvcc vector_add_workflow.cu -o vector_add_workflow
./vector_add_workflow
````

**Why this file:** It’s the same math as the minimal version, but laid out as a reusable template for any CUDA program.

````

---

# `vector_add_with_checks_README.md`

```markdown
# vector_add_with_checks.cu

**Purpose:** Add robust error handling.

**What’s new:**
- `CUDA_CHECK(...)` on every CUDA API call
- `cudaGetLastError()` after kernel launch
- `cudaDeviceSynchronize()` to surface runtime errors

**Build/Run:**
```bash
nvcc vector_add_with_checks.cu -o vector_add_with_checks
./vector_add_with_checks
````

**Outcome:** Fail fast with clear filename/line/error if anything goes wrong.

````

---

# `vector_add_verification_README.md`

```markdown
# vector_add_verification.cu

**Purpose:** Final, “teaching-quality” version with correctness proof.

**What it does:**
- GPU computes `C = A + B`
- CPU computes reference result
- Compares elementwise with tolerance (default `1e-5`)

**Build/Run:**
```bash
nvcc vector_add_verification.cu -o vector_add_verification
./vector_add_verification              # N ≈ 1,048,576, tol = 1e-5
./vector_add_verification 10000000     # bigger N
./vector_add_verification 1000000 1e-6 # stricter tolerance
````

**PASS means:** GPU matches CPU within tolerance; your pipeline is correct.

