# Makefile — Chapter 10 (Mandelbrot)
# Usage:
#   make            # build all binaries
#   make run-samples# render sample PGM/PPM outputs
#   make convert    # convert PGM/PPM to PNG via python helpers
#   make clean      # remove binaries and generated images

# ---- Config -------------------------------------------------------------------
NVCC        ?= nvcc
CUDAFLAGS   ?= -O3 -std=c++14
# Optional: set your GPU arch, e.g. -gencode arch=compute_86,code=sm_86
GENCODE     ?=
LDFLAGS     ?=
INCLUDES    ?=

# Binaries
BIN_DIR     := bin
BINS        := $(BIN_DIR)/mandelbrot \
               $(BIN_DIR)/mandelbrot_color \
               $(BIN_DIR)/mandelbrot_final \
               $(BIN_DIR)/mandelbrot_final_palette_pack

# Sources 
SRCS        := mandelbrot.cu \
               mandelbrot_color.cu \
               mandelbrot_final.cu \
               mandelbrot_final_palette_pack.cu

# Outputs
OUT_DIR     := out
PGM_OUT     := $(OUT_DIR)/mandelbrot.pgm
PPM_OUT     := $(OUT_DIR)/mandelbrot.ppm
PNG_GRAY    := $(OUT_DIR)/mandelbrot.png
PNG_COLOR   := $(OUT_DIR)/mandelbrot_color.png

# Python (for conversions)
PYTHON      ?= python3

# ---- Rules --------------------------------------------------------------------
.PHONY: all clean run-samples convert dirs

all: dirs $(BINS)

dirs:
	@mkdir -p $(BIN_DIR) $(OUT_DIR)

# Pattern rule: build each .cu into ./bin/<name>
$(BIN_DIR)/%: %.cu | dirs
	$(NVCC) $(CUDAFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)

# Convenience: individual targets
$(BIN_DIR)/mandelbrot: mandelbrot.cu
	$(NVCC) $(CUDAFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/mandelbrot_color: mandelbrot_color.cu
	$(NVCC) $(CUDAFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/mandelbrot_final: mandelbrot_final.cu
	$(NVCC) $(CUDAFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/mandelbrot_final_palette_pack: mandelbrot_final_palette_pack.cu
	$(NVCC) $(CUDAFLAGS) $(GENCODE) $(INCLUDES) $< -o $@ $(LDFLAGS)

# ---- Sample renders -----------------------------------------------------------
# Generates reference files in ./out using reasonable defaults.
run-samples: all
	@echo "[render] grayscale -> $(PGM_OUT)"
	$(BIN_DIR)/mandelbrot --width 1024 --height 768 --max-iter 200 --out $(PGM_OUT)
	@echo "[render] color (HSV) -> $(PPM_OUT)"
	$(BIN_DIR)/mandelbrot_color --width 1024 --height 768 --max-iter 500 --out $(PPM_OUT)
	@echo "[render] final (palette-pack, fire) -> $(OUT_DIR)/mandelbrot_fire.ppm"
	$(BIN_DIR)/mandelbrot_final_palette_pack --width 1920 --height 1080 --max-iter 800 \
	  --color --smooth --binary --palette fire --out $(OUT_DIR)/mandelbrot_fire.ppm

# ---- Conversions to PNG (requires pgm2png.py / ppm2png.py) --------------------
convert: $(PNG_GRAY) $(PNG_COLOR)
	@echo "[convert] done"

$(PNG_GRAY): $(PGM_OUT) pgm2png.py
	$(PYTHON) pgm2png.py $(PGM_OUT) $(PNG_GRAY) || true

$(PNG_COLOR): $(PPM_OUT) ppm2png.py
	$(PYTHON) ppm2png.py $(PPM_OUT) $(PNG_COLOR) || true

# ---- Cleanup ------------------------------------------------------------------
clean:
	@echo "[clean] removing binaries and outputs"
	@rm -rf $(BIN_DIR) $(OUT_DIR)
