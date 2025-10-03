# Makefile — Chapter 9: Time-Series Basics (CUDA + CSV utils)
# Usage:
#   make                    # build all binaries into ./bin
#   make chapter9           # build the main all-in-one program
#   make test               # build the SMA test harness
#   make utils              # build CSV utilities
#   make run N=1048576 W=5 EDGE=0.0 ALPHA=0.2 THRESH=64
#   make run_csv CSV=data.csv COL=1 OP=sma WIN=5 EDGE=0.0 OUT=out.csv THRESH=64
#   make clean              # remove ./bin

# ---- Toolchains ----
NVCC     ?= nvcc
CXX      ?= g++

# ---- GPU architecture (set to your GPU): 75=Turing, 86=Ampere, 89=Ada, 90=Hopper ----
SM       ?= 75
GENCODE   = -gencode arch=compute_$(SM),code=sm_$(SM)

# ---- Compiler flags ----
OPT      ?= -O2
CXXSTD    = -std=c++17
WARNHOST  = -Wall
NVCCFLAGS = $(OPT) $(CXXSTD) $(GENCODE) -Xcompiler -Wall -I.
CXXFLAGS  = $(OPT) $(CXXSTD) $(WARNHOST)

# ---- Paths / Binaries ----
BIN_DIR  := bin

# CUDA sources
CU_MAIN      := chapter9_timeseries.cu
CU_BASIC_SMA := moving_average.cu
CU_AUTO_SMA  := moving_average_auto.cu
CU_TEST_SMA  := moving_average_test.cu

# C++ utilities
CPP_HEAD     := csv_head.cpp
CPP_PLOT     := csv_to_plot.cpp

# Output binaries
BIN_MAIN     := $(BIN_DIR)/chapter9_timeseries
BIN_BASIC    := $(BIN_DIR)/moving_average
BIN_AUTO     := $(BIN_DIR)/moving_average_auto
BIN_TEST     := $(BIN_DIR)/moving_average_test
BIN_HEAD     := $(BIN_DIR)/csv_head
BIN_PLOT     := $(BIN_DIR)/csv_to_plot

ALL_BINS := $(BIN_MAIN) $(BIN_BASIC) $(BIN_AUTO) $(BIN_TEST) $(BIN_HEAD) $(BIN_PLOT)

# ---- Phony targets ----
.PHONY: all chapter9 test utils clean run run_csv

all: $(ALL_BINS)

chapter9: $(BIN_MAIN)

test: $(BIN_TEST)

utils: $(BIN_HEAD) $(BIN_PLOT)

# ---- Build rules ----
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(BIN_MAIN): $(CU_MAIN) csv_utils.hpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CU_MAIN)

$(BIN_BASIC): $(CU_BASIC_SMA) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CU_BASIC_SMA)

$(BIN_AUTO): $(CU_AUTO_SMA) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CU_AUTO_SMA)

$(BIN_TEST): $(CU_TEST_SMA) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CU_TEST_SMA)

$(BIN_HEAD): $(CPP_HEAD) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(CPP_HEAD)

$(BIN_PLOT): $(CPP_PLOT) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(CPP_PLOT)

# ---- Run helpers ----
# Example:
#   make run N=1048576 W=5 EDGE=0.0 ALPHA=0.2 THRESH=64
run: $(BIN_MAIN)
	$(BIN_MAIN) $(N) $(W) $(EDGE) $(ALPHA) $(THRESH)

# Example:
#   make run_csv CSV=data/series.csv COL=1 OP=sma WIN=5 EDGE=0.0 OUT=out.csv THRESH=64
# Notes:
#   OP ∈ {sma, ema}
#   For SMA: use WIN (window) and EDGE (e.g., 0.0 or nan)
#   For EMA: use ALPHA (e.g., 0.2)
run_csv: $(BIN_MAIN)
	@if [ -z "$(CSV)" ] || [ -z "$(OUT)" ]; then \
		echo "Usage: make run_csv CSV=path/to/in.csv COL=1 OP=sma WIN=5 EDGE=0.0 OUT=out.csv THRESH=64"; \
		exit 2; \
	fi; \
	if [ "$(OP)" = "ema" ]; then \
		$(BIN_MAIN) --csv "$(CSV)" --col $(COL) --op ema --alpha $(ALPHA) --out "$(OUT)"; \
	else \
		$(BIN_MAIN) --csv "$(CSV)" --col $(COL) --op sma --win $(WIN) --edge $(EDGE) --thresh $(THRESH) --out "$(OUT)"; \
	fi

# ---- Clean ----
clean:
	rm -rf $(BIN_DIR)
