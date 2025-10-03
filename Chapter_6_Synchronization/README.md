# Chapter 6 — Teams and Scratchpads

This chapter introduces **shared memory** in CUDA as a “team scratchpad.” You will learn how threads within a block can cooperate by writing their values into shared memory, synchronizing with `__syncthreads()`, and performing reductions to compute per-block results. The examples cover **minimum, maximum, argmin, and argmax** operations, showing both value-only and index-tracking reductions.

---

## Programs in this Chapter

* **chapter6_demo.cu**
  A self-contained CUDA program that demonstrates:

  * Block-level minimum reduction
  * Block-level argmin (value + index)
  * Correct use of `__syncthreads()`
  * Writing partial results back to global memory
  * Combining per-block results on the CPU

---

## Building

### Linux / macOS

Make sure you have the CUDA Toolkit installed and `nvcc` available in your `PATH`.

```bash
make            # builds into ./bin/chapter6_demo
```

### Windows

Use the provided batch file:

```bat
build_ch6.bat build   :: build the program
build_ch6.bat run     :: run with default size (N = 1<<20)
build_ch6.bat run 5000000  :: run with custom problem size
```

---

## Running

After building, you can run the demo:

```bash
./bin/chapter6_demo          # uses default problem size (about 1 million elements)
./bin/chapter6_demo 5000000  # run with 5 million elements
```

On Windows:

```bat
build_ch6.bat run
build_ch6.bat run 5000000
```

---

## Expected Output

The program prints both GPU-computed block reductions (combined on the CPU) and CPU reference values:

```
GPU block-reduced MIN combined on CPU  : -999.876465
CPU reference MIN                      : -999.876465

GPU block-reduced ARGMIN val/index     : -999.876465 @ 123456
CPU reference ARGMIN val/index         : -999.876465 @ 123456

Verification: MIN OK, ARGMIN OK
```

---

## Learning Goals

By studying and running this program you will:

* Understand how shared memory provides a fast cooperative scratchpad.
* Learn why synchronization (`__syncthreads()`) is necessary.
* Implement block-level reductions for min and argmin.
* Write block results safely to global memory.
* Combine partial results on the CPU to obtain a global result.
* Debug synchronization errors that lead to random wrong answers.

