# Makefile for Chapter 6 — Teams and Scratchpads
# Builds a single demo: chapter6_demo (min + argmin with shared memory)

# ----- Toolchain -----
NVCC    ?= nvcc

# ----- Configuration -----
# Set a sensible default architecture; override on the command line if needed, e.g.:
#   make ARCH=sm_86
ARCH    ?= sm_70

# Optimization and warnings
NVFLAGS ?= -O2 -std=c++14 -Xcompiler -Wall

# Where to place the binary
BIN_DIR ?= bin

# Source
SRC     := chapter6_demo.cu
EXE     := $(BIN_DIR)/chapter6_demo

# ----- Phony targets -----
.PHONY: all run clean info

all: $(EXE)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Build the demo
$(EXE): $(SRC) | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) -arch=$(ARCH) $< -o $@

# Run with optional N (problem size); defaults handled by the program
run: $(EXE)
	$(EXE) $(N)

# Show current configuration
info:
	@echo "NVCC   = $(NVCC)"
	@echo "ARCH   = $(ARCH)"
	@echo "NVFLAGS= $(NVFLAGS)"
	@echo "SRC    = $(SRC)"
	@echo "EXE    = $(EXE)"

# Remove build artifacts
clean:
	@rm -rf $(BIN_DIR)
