# Makefile for Chapter 8 — Monte Carlo π Estimation (CUDA)
# Builds: bin/monte_carlo_pi
# Usage:
#   make                    # build
#   make run                # run with defaults
#   make run SAMPLES=100000000 TPB=256
#   make clean

# ---- Configurable knobs -----------------------------------------------------
NVCC        ?= nvcc
SM          ?= 60              # Set your GPU arch: e.g., 53 (Jetson Nano), 75 (Turing), 86 (Ampere), 90 (Ada/Lovelace)
BUILD_TYPE  ?= Release         # Release or Debug

# ---- Paths/files ------------------------------------------------------------
BIN_DIR     := bin
TARGET      := $(BIN_DIR)/monte_carlo_pi
SRC         := monte_carlo_pi.cu

# ---- Flags ------------------------------------------------------------------
COMMON_FLAGS := -std=c++14 -Xcompiler -fno-exceptions
ifeq ($(BUILD_TYPE),Debug)
  NVCCFLAGS := -O0 -g -G -lineinfo $(COMMON_FLAGS)
else
  NVCCFLAGS := -O3 -lineinfo $(COMMON_FLAGS)
endif

ARCH_FLAG   := -arch=sm_$(SM)
LIBS        := -lcurand
# Add RPATH to find CUDA libs at runtime on some systems (optional)
RPATH_FLAG  := -Xlinker -rpath,$(shell dirname `which $(NVCC)`)/../lib64

# ---- Default rule -----------------------------------------------------------
.PHONY: all
all: $(TARGET)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(NVCC) $(ARCH_FLAG) $(NVCCFLAGS) $(RPATH_FLAG) -o $@ $^ $(LIBS)

# ---- Run helpers ------------------------------------------------------------
# SAMPLES: total dart throws, TPB: threads per block
SAMPLES ?= 50000000
TPB     ?= 256

.PHONY: run
run: $(TARGET)
	$(TARGET) $(SAMPLES) $(TPB)

# ---- Housekeeping -----------------------------------------------------------
.PHONY: clean
clean:
	@rm -f $(TARGET)

.PHONY: info
info:
	@echo "NVCC       = $(NVCC)"
	@echo "SM         = $(SM)"
	@echo "BUILD_TYPE = $(BUILD_TYPE)"
	@echo "NVCCFLAGS  = $(NVCCFLAGS)"
	@echo "ARCH_FLAG  = $(ARCH_FLAG)"
	@echo "TARGET     = $(TARGET)"
