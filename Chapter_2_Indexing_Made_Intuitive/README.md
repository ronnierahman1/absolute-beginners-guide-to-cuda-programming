Here’s a fully revised **README.md** for your Chapter 2 folder. It now includes all four programs with clear run instructions for both Linux/macOS and Windows.

````markdown
# Chapter 2 – Indexing Made Intuitive

This chapter introduces one of the most important concepts in CUDA programming: **mapping threads to data safely and correctly**.  
You will learn how to calculate thread indices, protect against out-of-bounds memory access, and compare GPU results against CPU reference implementations.  

---

## 📂 Contents

### Source Programs
- **`vector_scale_10x_plus_5.cu`**  
  Minimal kernel: each thread computes `y[i] = 10 * x[i] + 5` with a bounds check.  
  Demonstrates the correct **thread-to-data mapping**.

- **`too_many_threads_demo.cu`**  
  Demonstrates what happens when you launch more threads than data elements.  
  Runs once **without** bounds checking (undefined behavior) and once **with** the safety guard.

- **`compare_cpu_gpu_scale.cu`**  
  Runs the vector scaling on both CPU and GPU, compares results element by element, and reports **PASS/FAIL**.  
  Teaches the importance of validating GPU kernels.

- **`run_grid_block_experiments.cu`**  
  Tests different grid and block configurations (exact fit, undersubscription, oversubscription).  
  Prints coverage summaries, PASS/FAIL results, and sample values.  

### Build Scripts
- **`Makefile`** – Linux/macOS build automation.  
- **`ch2_build.bat`** – Windows batch script for building/running with `nvcc`.  

---

## ▶️ Building the Programs

### Linux / macOS (Make)
```bash
# Build all programs into ./bin
make
````

### Windows (Batch)

```bat
REM Build all programs into .\bin
ch2_build.bat
```

---

## ▶️ Running the Programs

### 1) Minimal vector scale

Each thread computes `y[i] = 10*x[i] + 5`.

```bash
make run-vec             # Linux/macOS
ch2_build.bat run-vec    # Windows
```

### 2) Oversubscription demo

Shows what happens when there are more threads than elements (unsafe vs safe).

```bash
make run-demo             # Linux/macOS
ch2_build.bat run-demo    # Windows
```

### 3) CPU vs GPU comparison

Validates GPU results against CPU results, printing **PASS/FAIL**.

```bash
make run-compare                    # Linux/macOS
ch2_build.bat run-compare           # Windows

# Optional arguments: N TPB
./bin/compare 1000000 128            # Linux/macOS
bin\compare.exe 1000000 128          # Windows
```

### 4) Grid/Block experiments

Tests multiple grid/block configurations for coverage and safety.

```bash
# Default experiments
make run-grid              # Linux/macOS
ch2_build.bat run-grid     # Windows

# Custom arguments: N TPB_CSV [fixed_blocks]
make run-grid ARGS="37 32,64,128"
ch2_build.bat run-grid 37 "32,64,128"

make run-grid ARGS="17 4,8,16 3"
ch2_build.bat run-grid 17 "4,8,16 3"
```

---

## 🧪 What You Will Learn

By completing this chapter, you will:

* Calculate the **global index** of a thread (`i = blockIdx.x * blockDim.x + threadIdx.x`).
* Map **one thread to one array element**.
* Use a **bounds check** (`if (i < N)`) to prevent out-of-bounds memory access.
* Understand **oversubscription** (too many threads) and **undersubscription** (too few threads).
* Learn why **consecutive memory access (coalescing)** improves performance.
* Compare GPU output against CPU results for correctness.
* See the consequences of omitting safety checks.

---

## 📌 Suggested Experiments

1. Change `N` in the code (e.g., 10, 17, 1000) and observe thread coverage.
2. Remove the `if (i < N)` check in `too_many_threads_demo.cu` and note the differences.
3. Run `run_grid_block_experiments.cu` with undersubscription — confirm some elements are missing.
4. Modify the kernel to compute `y[i] = 3 * x[i] - 7`.
5. Try different `threadsPerBlock` values (32, 64, 128, 256) and watch how coverage changes.

---

## ✅ Checklist

* [ ] I can explain how the global index is calculated.
* [ ] I know why `if (i < N)` is required.
* [ ] I have compared CPU and GPU results and seen **PASS/FAIL** output.
* [ ] I experimented with grid/block sizes and observed overshoot and undershoot.
* [ ] I understand why memory coalescing matters for GPU efficiency.

---
