# =================================================================================================
# Makefile — Chapter 2: Indexing Made Intuitive (CUDA)
#
# GOAL
# ----
# Build and run the Chapter 2 CUDA examples with clear, explicit rules (no ambiguous pattern rules).
# This file is heavily commented to teach:
#   • How to wire .cu sources to named executables
#   • How to set architecture flags (sm_XY) and compiler warnings
#   • How to provide convenient "run-*" targets
#
# PROJECT LAYOUT (expected)
# -------------------------
# .
# ├── vector_scale_10x_plus_5.cu        # core kernel: y = 10*x + 5
# ├── too_many_threads_demo.cu          # demo: missing bounds check vs. safe variant
# ├── compare_cpu_gpu_scale.cu          # CPU vs GPU validation harness
# ├── run_grid_block_experiments.cu     # experiments: grid/block coverage & PASS/FAIL
# └── Makefile                          # this file
#
# OUTPUTS
# -------
# All executables are placed under ./bin:
#   bin/vec_scale, bin/demo, bin/compare, bin/grid_expts
#
# COMMON COMMANDS
# ---------------
#   make                 # build all programs
#   make run-compare     # run CPU-vs-GPU validator (default args inside program)
#   make run-grid ARGS="37 32,64,128"   # pass CLI args to grid_expts
#   make clean           # remove ./bin
#
# ARCHITECTURE (VERY IMPORTANT)
# -----------------------------
# Set SM to your GPU’s compute capability:
#   • Turing (e.g., T4)              -> 75
#   • Ampere (RTX 30xx, A100)        -> 86 (RTX 30xx) / 80 (A100)
#   • Ada (RTX 40xx)                 -> 89
#   • Hopper (H100)                  -> 90
# Example: `make SM=90`
#
# TROUBLESHOOTING
# ---------------
# • nvcc not found:
#     - On Linux: ensure `which nvcc` shows your CUDA install (e.g., /usr/local/cuda/bin/nvcc)
#     - On Windows + WSL: install CUDA for WSL; or compile on Windows using the batch file.
# • Undefined reference / missing libcudart:
#     - Ensure CUDA toolkit is installed properly; nvcc links cudart by default.
# • “No rule to make target …”:
#     - Filenames must match the SRC_* variables below.
# =================================================================================================


# ----------------------------
# Toolchain & flags (customize)
# ----------------------------
NVCC        ?= nvcc                # nvcc compiler path; override with `make NVCC=/path/to/nvcc`
SM          ?= 86                  # default arch (Ampere RTX 30xx). Override: `make SM=90`
ARCH        := -arch=sm_$(SM)      # translates to e.g., -arch=sm_86
# -O2: optimize; -std=c++17: modern host C++; warnings on host via -Xcompiler
NVCCFLAGS   := -O2 -std=c++17 $(ARCH) -Xcompiler -Wall,-Wextra


# ----------------------------
# Sources (keep these in sync with your repo)
# ----------------------------
SRC_VEC     := vector_scale_10x_plus_5.cu
SRC_DEMO    := too_many_threads_demo.cu
SRC_COMP    := compare_cpu_gpu_scale.cu
SRC_GRID    := run_grid_block_experiments.cu


# ----------------------------
# Outputs
# ----------------------------
BIN_DIR     := bin
EXE_VEC     := $(BIN_DIR)/vec_scale
EXE_DEMO    := $(BIN_DIR)/demo
EXE_COMP    := $(BIN_DIR)/compare
EXE_GRID    := $(BIN_DIR)/grid_expts
ALL_EXES    := $(EXE_VEC) $(EXE_DEMO) $(EXE_COMP) $(EXE_GRID)


# ----------------------------
# Phony targets (not files)
# ----------------------------
.PHONY: all clean help \
        vec demo compare grid \
        run-vec run-demo run-compare run-grid


# ----------------------------
# Default: build everything
# ----------------------------
all: $(ALL_EXES)


# ----------------------------
# Ensure bin/ exists before linking
# ----------------------------
$(BIN_DIR):
	mkdir -p $(BIN_DIR)


# ----------------------------
# Explicit source → executable rules
# Rationale: avoids fragile stem/pattern assumptions.
# ----------------------------
$(EXE_VEC): $(SRC_VEC) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(EXE_DEMO): $(SRC_DEMO) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(EXE_COMP): $(SRC_COMP) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(EXE_GRID): $(SRC_GRID) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@


# ----------------------------
# Convenience: build single programs
# ----------------------------
vec:    $(EXE_VEC)
demo:   $(EXE_DEMO)
compare:$(EXE_COMP)
grid:   $(EXE_GRID)


# ----------------------------
# Convenience: run programs
# Notes:
#  • run-grid forwards ARGS to the executable (e.g., N and TPB list)
#  • run-compare has internal defaults; you can pass args directly if needed
# ----------------------------
run-vec: $(EXE_VEC)
	$(EXE_VEC)

run-demo: $(EXE_DEMO)
	$(EXE_DEMO)

run-compare: $(EXE_COMP)
	$(EXE_COMP)

# Example:
#   make run-grid ARGS="37 32,64,128"     # N=37, TPB list 32,64,128 (auto blocks per TPB)
#   make run-grid ARGS="17 4,8,16 3"      # N=17, TPB list, fixed blocks=3
ARGS ?=
run-grid: $(EXE_GRID)
	$(EXE_GRID) $(ARGS)


# ----------------------------
# Housekeeping
# ----------------------------
clean:
	@echo "[CLEAN] Removing $(BIN_DIR)/"
	@rm -rf $(BIN_DIR)

help:
	@echo "=== Makefile Help — Chapter 2 (CUDA) ==="
	@echo "Build targets:"
	@echo "  make            : build all programs -> bin/"
	@echo "  make vec        : build vector_scale_10x_plus_5.cu  -> bin/vec_scale"
	@echo "  make demo       : build too_many_threads_demo.cu    -> bin/demo"
	@echo "  make compare    : build compare_cpu_gpu_scale.cu    -> bin/compare"
	@echo "  make grid       : build run_grid_block_experiments.cu -> bin/grid_expts"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-vec        : execute bin/vec_scale"
	@echo "  make run-demo       : execute bin/demo"
	@echo "  make run-compare    : execute bin/compare"
	@echo "  make run-grid       : execute bin/grid_expts (forward ARGS)"
	@echo "      e.g., make run-grid ARGS=\"37 32,64,128\""
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          : remove ./bin"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  SM=<70|75|80|86|89|90>   : target GPU arch (default: $(SM))"
	@echo "  NVCC=<path/to/nvcc>      : nvcc compiler (default: $(NVCC))"
	@echo "  ARGS=\"...\"              : arguments forwarded to run-grid"
	@echo "========================================"
