/**
 * @file mandelbrot.cu
 * @brief Minimal CUDA Mandelbrot renderer (grayscale PGM) with CPU self-test.
 *
 * Usage:
 *   ./mandelbrot [width height max_iter output.pgm]
 *
 * Features:
 *   - Grayscale PGM output (ASCII P2)
 *   - Simple CUDA kernel: one thread per pixel
 *   - CPU reference for spot-checking correctness
 *   - Configurable image size, iteration cap, and output file
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

/**
 * @brief Macro for checking CUDA API calls and reporting errors.
 */
#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                     \
                   #call, __FILE__, __LINE__, cudaGetErrorString(err__));        \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

/**
 * @brief Write a grayscale image to ASCII PGM (P2) file.
 * @param path Output file path
 * @param img Image buffer (row-major, size W*H)
 * @param W Image width
 * @param H Image height
 * @return true on success, false on failure
 */
bool write_pgm(const std::string& path, const std::vector<uint8_t>& img,
               int W, int H)
{
  std::ofstream ofs(path, std::ios::out);
  if (!ofs) return false;
  ofs << "P2\n" << W << " " << H << "\n255\n";
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      ofs << int(img[y * W + x]) << (x + 1 == W ? '\n' : ' ');
    }
  }
  return true;
}

/**
 * @brief CPU reference: compute Mandelbrot escape iteration count for a point.
 * @param cr Real part of complex coordinate
 * @param ci Imaginary part of complex coordinate
 * @param max_iter Maximum number of iterations
 * @return Number of iterations before escape (or max_iter if inside set)
 */
static inline int mandelbrot_iter_cpu(double cr, double ci, int max_iter)
{
  double zr = 0.0, zi = 0.0;
  int it = 0;
  while (zr * zr + zi * zi <= 4.0 && it < max_iter) {
    double zr2 = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = zr2;
    ++it;
  }
  return it;
}

/**
 * @brief CUDA kernel: compute Mandelbrot set grayscale image (one thread per pixel).
 *
 * @param out Output image buffer (device, row-major, W*H)
 * @param W Image width
 * @param H Image height
 * @param xmin Viewport min real
 * @param xmax Viewport max real
 * @param ymin Viewport min imag
 * @param ymax Viewport max imag
 * @param max_iter Iteration cap
 */
__global__ void mandelbrot_kernel(uint8_t* out, int W, int H,
                                  double xmin, double xmax,
                                  double ymin, double ymax,
                                  int max_iter)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= W || py >= H) return;

  // Map pixel coordinates to complex plane
  double cr = xmin + (double(px) / double(W - 1)) * (xmax - xmin);
  double ci = ymin + (double(py) / double(H - 1)) * (ymax - ymin);

  // Iterate z = z^2 + c for Mandelbrot escape
  double zr = 0.0, zi = 0.0;
  int it = 0;
  while ((zr * zr + zi * zi) <= 4.0 && it < max_iter) {
    double zr2 = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = zr2;
    ++it;
  }

  // Grayscale mapping: inside set (hit cap) -> 0 (black), else scale 1..max_iter-1 to 1..255
  uint8_t shade = (it == max_iter) ? 0
                                   : static_cast<uint8_t>(std::round(255.0 * it / (max_iter - 1)));
  out[py * W + px] = shade;
}

/**
 * @brief Main entry point: parse args, launch CUDA kernel, write image, verify.
 */
int main(int argc, char** argv)
{
  // Defaults: a classic viewport and sane resolution.
  int W = 1024;
  int H = 768;
  int max_iter = 200;
  std::string out_path = "mandelbrot.pgm";

  // Parse command-line arguments (if provided)
  if (argc >= 3) {
    W = std::max(2, std::atoi(argv[1]));
    H = std::max(2, std::atoi(argv[2]));
  }
  if (argc >= 4) max_iter = std::max(2, std::atoi(argv[3]));
  if (argc >= 5) out_path = argv[4];

  // View window (standard): real in [-2.0, 1.0], imag in [-1.5, 1.5]
  // You can tweak these to explore different regions.
  const double xmin = -2.0, xmax = 1.0;
  const double ymin = -1.5, ymax = 1.5;

  std::printf("Rendering Mandelbrot %dx%d, max_iter=%d -> %s\n",
              W, H, max_iter, out_path.c_str());

  // Allocate host and device buffers for image
  std::vector<uint8_t> h_img(size_t(W) * size_t(H), 0);
  uint8_t* d_img = nullptr;
  CUDA_CHECK(cudaMalloc(&d_img, size_t(W) * size_t(H)));

  // Launch configuration: 16x16 blocks for 2D grid
  dim3 block(16, 16);
  dim3 grid((W + block.x - 1) / block.x,
            (H + block.y - 1) / block.y);

  // Kernel launch: compute Mandelbrot set on device
  mandelbrot_kernel<<<grid, block>>>(d_img, W, H, xmin, xmax, ymin, ymax, max_iter);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host and free device memory
  CUDA_CHECK(cudaMemcpy(h_img.data(), d_img, size_t(W) * size_t(H), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_img));

  // Write image to file (PGM ASCII)
  if (!write_pgm(out_path, h_img, W, H)) {
    std::fprintf(stderr, "Failed to write PGM: %s\n", out_path.c_str());
    return EXIT_FAILURE;
  }

  // ---- Self-test: compare a few representative pixels with CPU reference ----
  // We sample 6 pixels: center, four corners, and one near the Seahorse Valley.
  struct Sample { int px, py; const char* label; };
  std::vector<Sample> tests = {
      { W / 2, H / 2, "center" },
      { 0, 0, "top-left" },
      { W - 1, 0, "top-right" },
      { 0, H - 1, "bottom-left" },
      { W - 1, H - 1, "bottom-right" },
      { int(0.45 * W), int(0.5 * H), "seahorse-ish" }
  };

  bool ok = true;
  for (const auto& s : tests) {
    // Map sample pixel to complex plane
    double cr = xmin + (double(s.px) / double(W - 1)) * (xmax - xmin);
    double ci = ymin + (double(s.py) / double(H - 1)) * (ymax - ymin);
    int it = mandelbrot_iter_cpu(cr, ci, max_iter);
    uint8_t expected = (it == max_iter) ? 0
                                        : static_cast<uint8_t>(std::round(255.0 * it / (max_iter - 1)));
    uint8_t got = h_img[size_t(s.py) * size_t(W) + size_t(s.px)];
    if (expected != got) {
      ok = false;
      std::fprintf(stderr,
                   "Self-test mismatch at %s (%d,%d): expected %u, got %u (iter=%d)\n",
                   s.label, s.px, s.py, (unsigned)expected, (unsigned)got, it);
    }
  }
  if (ok) {
    std::puts("Self-test: PASS (sampled pixels match CPU reference).");
  } else {
    std::puts("Self-test: FAIL (see mismatches above).");
  }

  std::puts("Done.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
