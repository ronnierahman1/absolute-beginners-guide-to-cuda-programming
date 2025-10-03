
# Chapter 9 – Time-Series Basics: Moving Averages on the GPU

This chapter introduces **time-series smoothing** using the **Simple Moving Average (SMA)** and the **Exponential Moving Average (EMA)**. These are practical techniques widely used in finance, sensors, web analytics, and many other fields. You’ll learn how to implement both methods on the GPU, test them against CPU references, and apply them to real-world datasets.

---

## What’s Inside

### Core Programs

* **`moving_average.cu`**
  Minimal kernel showing the one-thread-per-output SMA implementation.
* **`moving_average_auto.cu`**
  Optimized SMA with automatic dispatch (loop vs prefix-sum).
* **`chapter9_timeseries.cu`**
  Complete all-in-one program: CPU SMA/EMA, GPU kernels, auto-dispatch, test harness, timing, and CSV mode.
* **`moving_average_test.cu`**
  Test harness for validating GPU SMA against CPU references.

### Utilities

* **`csv_utils.hpp`** – Header-only CSV loader/saver for simple datasets.
* **`csv_head.cpp`** – Preview a CSV file (headers + first rows).
* **`csv_to_plot.cpp`** – Convert a CSV to a `timestamp,value` format for plotting.

### Documentation

* **`README.md`** – This file, explaining everything in one place.

---

## Building

You can build with either **Makefile (Linux/macOS)** or **build.bat (Windows)**. Both place binaries in `./bin`.

### Linux / macOS

```bash
# Build everything
make

# Build just the main program
make chapter9

# Build CSV utilities
make utils

# Clean
make clean
```

### Windows

```bat
REM Build everything
build all

REM Build just the main program
build chapter9

REM Build CSV utilities
build utils

REM Clean
build clean
```

---

## Running

### Random synthetic test

Run the all-in-one program with synthetic data:

```bash
./bin/chapter9_timeseries N W EDGE ALPHA THRESH
```

* **N**: number of points (default 1,048,576)
* **W**: SMA window size (default 5)
* **EDGE**: edge fill value (0.0 or `nan`)
* **ALPHA**: EMA smoothing factor (default 0.2)
* **THRESH**: switch point for auto SMA (default 64)

Example:

```bash
./bin/chapter9_timeseries 1000000 9 0.0 0.1 64
```

### CSV Mode

You can also process real datasets directly from CSV.

**SMA:**

```bash
./bin/chapter9_timeseries --csv data/prices.csv --col 2 --op sma --win 9 --edge 0.0 --out out_sma.csv
```

**EMA:**

```bash
./bin/chapter9_timeseries --csv data/sensor.csv --col 1 --op ema --alpha 0.15 --out out_ema.csv
```

CSV mode options:

* `--csv <file>` : input CSV file
* `--col <n>` : column number (1-based)
* `--op <sma|ema>` : operation type
* `--win <W>` : window size (SMA only)
* `--edge <val|nan>` : fill strategy (SMA only)
* `--alpha <float>` : smoothing factor (EMA only)
* `--out <file>` : output CSV file
* `--thresh <int>` : auto-dispatch threshold

---

## Utilities

* **Preview a CSV:**

```bash
./bin/csv_head data/prices.csv --rows 15 --sep auto
```

* **Convert for plotting:**

```bash
./bin/csv_to_plot data/prices.csv prices_plot.csv --col-t 1 --col-y 2 --to-unix
```

This produces a `timestamp,value` file suitable for gnuplot, Excel, or Python plotting.

---

## Key Lessons from This Chapter

* **SMA:** Easy to parallelize — one thread per output.
* **EMA:** Recursive, harder to parallelize — requires scan-like formulation.
* **Edge Handling:** Important to choose consistent policies for CPU vs GPU.
* **Memory Access:** Arithmetic is cheap; memory bandwidth is the real cost.
* **Performance:** Even simple kernels can smooth millions of points far faster than a CPU.

---

## Exercises

1. Implement a **centered SMA** and compare to the causal version.
2. Try **different edge policies** (fill vs shrink vs pad).
3. Measure performance for **window sizes 5 vs 500 vs 5000**.
4. Apply both SMA and EMA to a **real dataset** and plot the difference.
5. Extend the CSV utilities to support **multi-column smoothing**.

---

## Folder Structure

```
Chapter_9_TimeSeries/
│
├── moving_average.cu
├── moving_average_auto.cu
├── moving_average_test.cu
├── chapter9_timeseries.cu
│
├── csv_utils.hpp
├── csv_head.cpp
├── csv_to_plot.cpp
│
├── Makefile
├── build.bat
└── README.md
```

