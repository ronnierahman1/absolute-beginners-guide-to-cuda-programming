# Makefile — Absolute Beginners guide to CUDA programming in C/C++ (Chapter 1)
# Usage:
#   make            # build all programs
#   make run        # build and run all programs (in order)
#   make clean      # remove built executables
#   make hello      # build just one program (also works for the others)
#   make run-hello  # build and run just one program

# ---- Settings ---------------------------------------------------------------

NVCC   ?= nvcc
CFLAGS ?= -O2            # keep it simple; add flags here if needed

# List of output programs (no .cu extension)
TARGETS := hello_gpu hello_grid hello_independent cpu_reference_check

# ---- Rules -----------------------------------------------------------------

.PHONY: all clean run $(addprefix run-, $(TARGETS))
all: $(TARGETS)

# Pattern rule: how to build "prog" from "prog.cu"
%: %.cu
	$(NVCC) $(CFLAGS) $< -o $@

# Convenience: build & run everything (in a friendly order)
run: $(TARGETS)
	@echo "---- Running: hello ----"
	@./hello_gpu
	@echo
	@echo "---- Running: hello_grid ----"
	@./hello_grid
	@echo
	@echo "---- Running: hello_independent ----"
	@./hello_independent
	@echo
	@echo "---- Running: cpu_reference_check ----"
	@./cpu_reference_check

# Per-program run helpers (e.g., `make run-hello`)
$(addprefix run-, $(TARGETS)): run-%: %
	@echo "---- Running: $< ----"
	@./$<

# Clean up built executables
clean:
	@rm -f $(TARGETS)
	@echo "Cleaned."
