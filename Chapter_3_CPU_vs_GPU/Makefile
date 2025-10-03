# =============================================================================
# Makefile — Chapter 3: Talk to the GPU (CUDA for Absolute Beginners)
#
# Builds the step-by-step programs for vector addition and supporting demos:
#   1) device_alloc_free.cu
#   2) memcpy_roundtrip.cu
#   3) vector_add_minimal.cu
#   4) vector_add_workflow.cu
#   5) vector_add_with_checks.cu
#   6) vector_add_verification.cu
#
# Usage
# -----
#   make                # builds all targets with default (release) flags
#   make MODE=debug     # builds all targets with debug flags (-G, -O0, -lineinfo)
#   make clean          # removes bin/
#
# Architecture note
# -----------------
# Set ARCH to match your GPU’s SM version (examples below).
#   ARCH=sm_61  # Pascal (GTX 10xx)
#   ARCH=sm_70  # Volta (V100), default here
#   ARCH=sm_75  # Turing (RTX 20xx)
#   ARCH=sm_86  # Ampere (RTX 30xx)
#   ARCH=sm_89  # Ada (RTX 40xx) — adjust as needed
#
# Example:
#   make ARCH=sm_86
#
# Jetson / aarch64 tip:
#   nvcc is usually present; this Makefile does not require special flags.
#
# =============================================================================

# --- Toolchain ---------------------------------------------------------------
NVCC        ?= nvcc

# --- Directories -------------------------------------------------------------
SRC_DIR     := .
BIN_DIR     := bin

# --- Programs to build -------------------------------------------------------
TARGETS     := device_alloc_free \
               memcpy_roundtrip \
               vector_add_minimal \
               vector_add_workflow \
               vector_add_with_checks \
               vector_add_verification

# --- Files (assumes .cu filenames match TARGETS) -----------------------------
SOURCES     := $(addsuffix .cu,$(TARGETS))
BINARIES    := $(addprefix $(BIN_DIR)/,$(TARGETS))

# --- Architecture flag -------------------------------------------------------
ARCH        ?= sm_70
ARCHFLAG    := -arch=$(ARCH)

# --- Common CUDA compile flags ----------------------------------------------
# Note: We use C++14 to match the examples (std::fabs, etc. are fine either way).
STDFLAG     := -std=c++14

# MODE=release (default) or MODE=debug
MODE        ?= release

ifeq ($(MODE),debug)
  # Debug: device debug symbols, no optimization, and line info for better traces
  NVCCFLAGS := $(STDFLAG) -G -O0 -lineinfo
else
  # Release: optimize; keep line info helpful for occasional profiling
  NVCCFLAGS := $(STDFLAG) -O2 -lineinfo
endif

# --- Default target ----------------------------------------------------------
.PHONY: all
all: $(BINARIES)
	@echo "Build complete ($(MODE), $(ARCH)). Binaries in '$(BIN_DIR)'."

# --- Pattern rule: build each .cu into bin/<name> ----------------------------
$(BIN_DIR)/%: $(SRC_DIR)/%.cu | $(BIN_DIR)
	@echo "  NVCC  $<  ->  $@"
	$(NVCC) $(NVCCFLAGS) $(ARCHFLAG) $< -o $@

# --- Ensure bin/ exists ------------------------------------------------------
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# --- Convenience targets -----------------------------------------------------
.PHONY: release debug
release:
	@$(MAKE) MODE=release

debug:
	@$(MAKE) MODE=debug

# Build only a single target:
#   make vector_add_verification
$(TARGETS): %: $(BIN_DIR)/%

# --- Cleanup -----------------------------------------------------------------
.PHONY: clean
clean:
	@echo "Cleaning bin/ ..."
	@rm -rf $(BIN_DIR)
	@echo "Done."

# --- Help --------------------------------------------------------------------
.PHONY: help
help:
	@echo "CUDA Chapter 3 Makefile"
	@echo
	@echo "Targets:"
	@echo "  all (default)         Build all programs into ./bin"
	@echo "  release               Build with -O2 (default)"
	@echo "  debug                 Build with -G -O0 -lineinfo"
	@echo "  clean                 Remove ./bin"
	@echo "  <name>                Build a single program (e.g., make vector_add_verification)"
	@echo
	@echo "Variables:"
	@echo "  ARCH=sm_70            Set GPU architecture (sm_61, sm_70, sm_75, sm_86, sm_89, ...)"
	@echo "  MODE=release|debug    Choose optimization level and debug symbols"
	@echo
	@echo "Examples:"
	@echo "  make ARCH=sm_86"
	@echo "  make MODE=debug"
	@echo "  make vector_add_with_checks"
