# Makefile — Chapter 5: Pictures Are Just Arrays
# Builds:
#   bin/invert_pgm      (CUDA grayscale negative with --verify)
#   bin/gradient_pgm    (PGM gradient generator)

# ---- Config ----
CXX      := g++
NVCC     := nvcc
CXXFLAGS := -O2 -std=c++17
NVFLAGS  := -O2 -std=c++17

SRCDIR   := .
BINDIR   := bin

# ---- Programs ----
CUDA_PROG := $(BINDIR)/invert_pgm
CPP_PROG  := $(BINDIR)/gradient_pgm

# ---- Sources ----
CUDA_SRC := $(SRCDIR)/invert_pgm.cu
CPP_SRC  := $(SRCDIR)/gradient_pgm.cpp

# ---- Default ----
.PHONY: all
all: $(BINDIR) $(CUDA_PROG) $(CPP_PROG)

# ---- Build rules ----
$(BINDIR):
	@mkdir -p $(BINDIR)

$(CUDA_PROG): $(CUDA_SRC) | $(BINDIR)
	$(NVCC) $(NVFLAGS) $< -o $@

$(CPP_PROG): $(CPP_SRC) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# ---- Convenience targets ----
.PHONY: invert gradient
invert: $(CUDA_PROG)

gradient: $(CPP_PROG)

# ---- Clean ----
.PHONY: clean
clean:
	@$(RM) -r $(BINDIR)

# ---- Help ----
.PHONY: help
help:
	@echo "Targets:"
	@echo "  all        - build both programs"
	@echo "  invert     - build bin/invert_pgm"
	@echo "  gradient   - build bin/gradient_pgm"
	@echo "  clean      - remove bin/"
