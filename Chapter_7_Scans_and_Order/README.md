
# CUDA Scan Demos — Chapter 7 (Prefix Sum / Scan)

## Table of Contents
- [Overview](#overview)
- [Files](#files)
- [Build Instructions](#build-instructions)
- [Run Instructions](#run-instructions)
- [Quick Starts](#quick-starts)
- [Sample Outputs](#sample-outputs)
- [How it Works](#how-it-works)
- [Verifier](#verifier)
- [Troubleshooting](#troubleshooting)
- [License / Attribution](#license--attribution)
- [Support](#support)

---

## Overview

This mini-repo is the hands-on companion for Chapter 7. It shows how to build a correct **block-local scan**, “**stitch**” blocks into a **global** scan, and apply scans to real tasks (compaction, histograms, CDF). You also get a **verifier** that stress-tests both inclusive and exclusive results across edge cases.

---


## Files

* **`scan_global_stitch.cu`** — Core scan code: block-local scan (shared memory + `__syncthreads()`), block-totals collection, CPU stitching (exclusive prefix of block sums) + offset add.
* **`demo_scan_apps.cu`** — Small CLI demo app (compaction, histogram grouping, CDF).
* **`verify_scan.cu`** — §7.6 correctness harness (CPU refs, patterns, sizes, blocks).
* **`demo_scan_verify_apps.cu`** — Demo app that links the verifier and adds a `--verify` flag to run it before the chosen demo.

---


## Build Instructions

### Linux/macOS (Make)

```bash
make        # builds bin/demo and bin/demo_verify
# optional: target a specific GPU arch, e.g. sm_86
make SM=86
```

Binaries:

* `bin/demo` (demos only)
* `bin/demo_verify` (demos + `--verify`)


### Windows (Batch)

```bat
build.bat           # builds .\bin\demo.exe and .\bin\demo_verify.exe
build.bat 86        # optional: target sm_86
build.bat clean     # remove build artifacts
```


> Requires CUDA toolkit in your PATH (`nvcc`).
> Tested with CUDA 10.2 and newer. Requires a CUDA-capable GPU (Compute Capability 3.0+ recommended).

---


## Run Instructions

### Modes

* `compact` — Stream compaction (keep `> 0`) using **exclusive** positions from a scan of 0/1 flags
* `hist` — Histogram grouping: counts → **exclusive offsets** → grouped scatter
* `cdf` — Cumulative Distribution Function from **inclusive** scan of weights


### Flags (shared)

* `--mode {compact|hist|cdf}`
* `--N <int>` number of elements (default varies)
* `--BLOCK <int>` CUDA block size (e.g., 128/256)
* `--BINS <int>` number of histogram bins (for `hist` mode)


### Verifier flag (only for `demo_verify`)

* `--verify` run the full §7.6 harness first, then the selected demo

---


## Quick Starts

```bash
# Compaction demo (stable keep > 0)
./bin/demo --mode compact --N 1000 --BLOCK 256

# Histogram grouping with 64 bins
./bin/demo --mode hist --N 5000 --BINS 64

# CDF over 2048 float weights
./bin/demo --mode cdf --N 2048

# Verifier + demo (linked in bin/demo_verify)
./bin/demo_verify --verify --mode compact --N 1000 --BLOCK 256
```

Windows (PowerShell/CMD):

```bat
bin\demo.exe --mode compact --N 1000 --BLOCK 256
bin\demo_verify.exe --verify --mode hist --N 5000 --BINS 64
```

---


## Sample Outputs

> Numbers vary with randomness/GPU, but the structure should match.

**Compaction**

```
=== Stream Compaction Demo (keep > 0) ===
Kept=14  Match? YES
```

**Histogram**

```
=== Histogram Grouping Demo (values in [0,8)) ===
Sum(counts)=64 (expect 64)  Offsets OK? YES
```

**CDF**

```
=== CDF (float weights) ===
Total=511.746460  CDF[N-1]=1.000000
```

**Verifier + Demo**

```
Verifier: 630/630 cases passed
=== Stream Compaction Demo (keep > 0) ===
Kept=430  Match? YES
```

---


## How it Works

1. **Block-local scan** in shared memory with **two barriers per step** (read → update, then publish).
2. **Block sums** collected (last active element per block).
3. **CPU stitch**: exclusive scan of block sums → per-block **offsets**.
4. **Add offsets** to each block’s scanned chunk → **global** scan.
5. Demos reuse the same scan:

   * **Exclusive positions** for compaction = post-shift of inclusive results (identity at front).
   * **Histogram offsets** = exclusive scan of bin counts.
   * **CDF** = inclusive scan of weights, normalize by total.

---


## Verifier

The verifier (`--verify` flag, or `bin/demo_verify`) runs a comprehensive suite of tests on both inclusive and exclusive scan implementations. It checks a variety of input patterns, sizes, and block configurations, ensuring correctness across edge cases. If any test fails, details are printed for debugging.

---

## Troubleshooting

* **Wrong results on boundary sizes** (e.g., `N=33`, `BLOCK=32`): check tail-block handling (`active` threads) and reading `temp[active-1]` for the block total.
* **Non-deterministic mismatches**: you’re likely missing a `__syncthreads()` between stages.
* **Performance seems off**: try `BLOCK=128` or `256`, ensure coalesced global loads/stores, and avoid unnecessary host/device transfers.

---


## License / Attribution


Use freely for learning and projects. If you share, a nod to “CUDA for Absolute Beginners — Chapter 7” is appreciated.

---

## Support

For questions, bug reports, or suggestions, please open an issue on the repository or contact the maintainer at ronnierahman1@gmail.com.
