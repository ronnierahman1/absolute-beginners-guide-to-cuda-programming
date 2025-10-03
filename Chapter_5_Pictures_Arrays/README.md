# Chapter 5 – Pictures Are Just Arrays

This chapter introduces **CUDA image processing** by treating grayscale pictures as 2D arrays of pixels. We implement a **negative filter** (`out = 255 – in`) that inverts brightness, then save the result to a simple **PGM file** for easy viewing. A self-test runs the kernel twice to confirm correctness.

---

## 📂 Programs in this Chapter

1. **invert_pgm.cu**  
   - Full CUDA program that:
     - Loads an 8-bit grayscale PGM (P5 format)  
     - Launches a CUDA kernel to invert all pixels  
     - Saves the result as a new PGM file  
     - Optional `--verify` mode: applies inversion twice and checks if output matches the original  

2. **gradient_pgm.cpp**  
   - Tiny helper utility to generate grayscale PGM images for testing.  
   - Supports horizontal (`h`) or vertical (`v`) gradients.  
   - Useful for debugging indexing mistakes and testing the negative filter.  

---

## ⚙️ Building

### Linux / macOS
Requirements:  
- CUDA Toolkit installed (`nvcc` in PATH)  
- g++ or clang++ for the gradient generator  

```bash
# Build both programs
make

# Build only invert_pgm
make invert

# Build only gradient_pgm
make gradient

# Clean
make clean
````

Executables will appear in `./bin`.

### Windows

Requirements:

* CUDA Toolkit installed (`nvcc` in PATH)
* MSVC `cl.exe` (use “x64 Native Tools Command Prompt”) or MinGW g++

Run the batch script:

```bat
build_ch5.bat
```

Executables will appear in `bin\`.

---

## ▶️ Usage

### Gradient Generator

```bash
# Horizontal gradient 512×256
gradient_pgm 512 256 h out_horizontal.pgm

# Vertical gradient 256×256
gradient_pgm 256 256 v out_vertical.pgm
```

### CUDA Negative Filter

```bash
# Basic usage
invert_pgm input.pgm output_negative.pgm

# Run with self-test (double inversion)
invert_pgm input.pgm output_negative.pgm --verify
```

---

## 🖼️ Viewing Results

PGM (Portable Gray Map) is supported by most tools:

* Linux: `display output.pgm` (ImageMagick) or drag into viewer
* Windows/macOS: open with GIMP, IrfanView, or ImageMagick

A gradient input should invert cleanly:

* Black→white gradient becomes white→black
* Running inversion twice restores the original

---

## ✅ Self-Test

The kernel can be validated by applying it twice:

```
255 – (255 – p) = p
```

If the double inversion equals the original, indexing and memory access are correct.

---

## 🔎 Notes

* Always use bounds checks in your kernel: `if (x < width && y < height)`
* PGM is chosen for simplicity; no external libraries are required.
* Gradient and checkerboard images are great for spotting indexing errors.

