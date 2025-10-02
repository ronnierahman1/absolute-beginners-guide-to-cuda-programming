# Makefile for Chapter 4 sample programs
# Programs:
#   - poly_44_avoid_redundancy.cu  -> bin/poly44
#   - poly_45_precision.cu         -> bin/poly45
#   - poly_46_testing.cu           -> bin/poly46
#
# Usage examples:
#   make              # build all
#   make run44        # build + run poly44
#   make clean        # remove bin/
#
# Tunables (override on command line, e.g., `make ARCH=sm_75 FAST_MATH=1`):
#   ARCH       : target GPU architecture (sm_70, sm_75, sm_80, sm_86, sm_90, ...)
#   DEBUG      : 1 to build with -G -g -O0 (device debug); default 0
#   FAST_MATH  : 1 to enable -use_fast_math for float-heavy kernels; default 0
#   CXXSTD     : C++ standard used by nvcc's host compiler front-end (default c++17)
#   DEFINES    : extra -DNAME[=VALUE] macros you want to pass to all targets

NVCC      ?= nvcc
ARCH      ?= sm_80
DEBUG     ?= 0
FAST_MATH ?= 0
CXXSTD    ?= c++17
DEFINES   ?=

SRC_DIR := .
BIN_DIR := bin

# Targets
BIN44 := $(BIN_DIR)/poly44
BIN45 := $(BIN_DIR)/poly45
BIN46 := $(BIN_DIR)/poly46

TARGETS := $(BIN44) $(BIN45) $(BIN46)

# Base flags
NVCCFLAGS := -std=$(CXXSTD) -arch=$(ARCH) -O3 --expt-relaxed-constexpr $(DEFINES)
# Emit line info to help with profiling (kept in both opt and debug)
NVCCFLAGS += -lineinfo

# Debug toggle
ifeq ($(DEBUG),1)
  NVCCFLAGS := -std=$(CXXSTD) -arch=$(ARCH) -G -g -O0 -lineinfo $(DEFINES)
endif

# Optional fast math (float only; does not affect double throughput)
ifeq ($(FAST_MATH),1)
  NVCCFLAGS += -use_fast_math
endif

# Common host compiler hints
NVCCFLAGS += -Xcompiler -fno-exceptions

# Default rule
.PHONY: all
all: $(TARGETS)

# Create bin directory
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Pattern to build each CUDA program into bin/
$(BIN_DIR)/%: $(SRC_DIR)/%.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Explicit mappings from filenames → short binary names
$(BIN44): $(SRC_DIR)/poly_44_avoid_redundancy.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN45): $(SRC_DIR)/poly_45_precision.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN46): $(SRC_DIR)/poly_46_testing.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Convenience run targets
.PHONY: run44 run45 run46
run44: $(BIN44)
	$<

run45: $(BIN45)
	$<

run46: $(BIN46)
	$<

# Clean
.PHONY: clean
clean:
	@rm -rf $(BIN_DIR)
