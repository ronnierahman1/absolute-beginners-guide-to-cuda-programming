# Makefile for:
#   demo_scan_apps.cu
#   demo_scan_verify_apps.cu   (with --verify flag)
#   scan_global_stitch.cu
#   verify_scan.cu
#
# Usage:
#   make                # build bin/demo and bin/demo_verify
#   make clean          # remove build/ and bin/
#   make SM=86          # (optional) set CUDA arch, e.g., sm_86
#
# Run:
#   ./bin/demo --mode compact --N 1000 --BLOCK 256
#   ./bin/demo_verify --verify --mode hist --N 5000 --BINS 64

NVCC        ?= nvcc
CUDACFLAGS  ?= -O2 -std=c++14 -lineinfo

# Optional: specify a single arch like "86" for sm_86 via: make SM=86
ifdef SM
GENCODE := -gencode arch=compute_$(SM),code=sm_$(SM)
endif

SRC_DEMO        := demo_scan_apps.cu
SRC_DEMO_VERIFY := demo_scan_verify_apps.cu
SRC_STITCH      := scan_global_stitch.cu
SRC_VERIFY      := verify_scan.cu

BUILDDIR := build
BINDIR   := bin

OBJ_DEMO        := $(BUILDDIR)/demo_scan_apps.o
OBJ_DEMO_VERIFY := $(BUILDDIR)/demo_scan_verify_apps.o
OBJ_STITCH      := $(BUILDDIR)/scan_global_stitch.o
OBJ_VERIFY      := $(BUILDDIR)/verify_scan.o

BIN_DEMO        := $(BINDIR)/demo
BIN_DEMO_VERIFY := $(BINDIR)/demo_verify

.PHONY: all clean

all: $(BIN_DEMO) $(BIN_DEMO_VERIFY)

# --- Directories ---
$(BUILDDIR):
	@mkdir -p $@

$(BINDIR):
	@mkdir -p $@

# --- Objects ---
# Shared scan implementation
$(OBJ_STITCH): $(SRC_STITCH) | $(BUILDDIR)
	$(NVCC) $(CUDACFLAGS) $(GENCODE) -c $< -o $@

# Verifier harness (used only by demo_verify)
$(OBJ_VERIFY): $(SRC_VERIFY) | $(BUILDDIR)
	$(NVCC) $(CUDACFLAGS) $(GENCODE) -c $< -o $@

# Demo without verifier
$(OBJ_DEMO): $(SRC_DEMO) | $(BUILDDIR)
	$(NVCC) $(CUDACFLAGS) $(GENCODE) -c $< -o $@

# Demo that links the verifier; define HAVE_VERIFY so code can enable the flag path.
$(OBJ_DEMO_VERIFY): $(SRC_DEMO_VERIFY) | $(BUILDDIR)
	$(NVCC) $(CUDACFLAGS) $(GENCODE) -DHAVE_VERIFY -c $< -o $@

# --- Binaries ---
$(BIN_DEMO): $(OBJ_DEMO) $(OBJ_STITCH) | $(BINDIR)
	$(NVCC) $(CUDACFLAGS) $(GENCODE) $^ -o $@

$(BIN_DEMO_VERIFY): $(OBJ_DEMO_VERIFY) $(OBJ_STITCH) $(OBJ_VERIFY) | $(BINDIR)
	$(NVCC) $(CUDACFLAGS) $(GENCODE) $^ -o $@

# --- Housekeeping ---
clean:
	@rm -rf $(BUILDDIR) $(BINDIR)
