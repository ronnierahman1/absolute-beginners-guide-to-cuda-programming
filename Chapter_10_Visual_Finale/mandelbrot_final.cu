/**
 * @file mandelbrot_final.cu
 * @brief Polished, educational CUDA Mandelbrot renderer (PGM/PPM; ASCII or binary).
 *
 * Features:
 *   - Grayscale and color (HSV palette) output
 *   - Smooth coloring for color mode
 *   - Flexible viewport and iteration controls
 *   - Binary (P5/P6) or ASCII (P2/P3) output
 *   - CPU spot-verification of a few pixels for correctness
 *
 * Build:
 *   nvcc mandelbrot_final.cu -o mandelbrot_final
 *
 * Usage (flags are optional; shown with defaults):
 *   --width W          (default 1024)
 *   --height H         (default 768)
 *   --max-iter N       (default 200)
 *   --xmin Xmin        (default -2.0)
 *   --xmax Xmax        (default  1.0)
 *   --ymin Ymin        (default -1.5)
 *   --ymax Ymax        (default  1.5)
 *   --color            (enable RGB output; default grayscale)
 *   --smooth           (enable smooth coloring for RGB)
 *   --binary           (P5 for PGM or P6 for PPM; default ASCII P2/P3)
 *   --verify K         (CPU spot-check K pixels; default 6; 0 disables)
 *   --out path         (default "mandelbrot.out")
 *
 * Notes:
 *   - Grayscale uses escape-time linear mapping; inside-set -> black.
 *   - Color uses HSV palette; with --smooth it uses normalized escape time.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

/**
 * @brief Macro for checking CUDA API calls and reporting errors.
 */
#define CUDA_CHECK(call) do {                                   \
  cudaError_t err__ = (call);                                   \
  if (err__ != cudaSuccess) {                                   \
    std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",      \
                 #call, __FILE__, __LINE__,                     \
                 cudaGetErrorString(err__));                    \
    std::exit(EXIT_FAILURE);                                    \
  }                                                             \
} while (0)

/**
 * @brief Struct to hold all command-line arguments and rendering options.
 */
struct Args {
  int    W = 1024;                ///< Image width
  int    H = 768;                 ///< Image height
  int    max_iter = 200;          ///< Maximum iterations for escape
  double xmin = -2.0, xmax = 1.0; ///< Real axis viewport
  double ymin = -1.5, ymax = 1.5; ///< Imaginary axis viewport
  bool   color = false;           ///< Enable color (PPM) output
  bool   smooth = false;          ///< Enable smooth coloring
  bool   binary = false;          ///< Output binary (P5/P6) instead of ASCII (P2/P3)
  int    verify = 6;              ///< Number of pixels to spot-check with CPU
  std::string out_path = "mandelbrot.out"; ///< Output file path
};

/**
 * @brief Print usage information for the program.
 * @param prog Program name (argv[0])
 */
static void print_usage(const char* prog) {
  std::printf(
R"(Usage: %s [options]

Options:
  --width W          Image width (default 1024)
  --height H         Image height (default 768)
  --max-iter N       Iteration cap (default 200)
  --xmin Xmin        Viewport min real (default -2.0)
  --xmax Xmax        Viewport max real (default 1.0)
  --ymin Ymin        Viewport min imag (default -1.5)
  --ymax Ymax        Viewport max imag (default 1.5)
  --color            Enable RGB color output (PPM)
  --smooth           Smooth coloring (with --color)
  --binary           Binary output (P5 PGM or P6 PPM). Default ASCII (P2/P3)
  --verify K         CPU spot-verify K pixels (default 6; 0 disables)
  --out PATH         Output file path (default mandelbrot.out)

Examples:
  %s --width 1920 --height 1080 --color --smooth --binary --out fractal.ppm
  %s --xmin -0.9 --xmax -0.6 --ymin 0.0 --ymax 0.3 --max-iter 1200 --color --out zoom.ppm
)", prog, prog, prog);
}

/**
 * @brief Parse command-line arguments into Args struct.
 * @param argc Argument count
 * @param argv Argument vector
 * @return Parsed Args struct
 */
static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1; i<argc; ++i) {
    std::string s = argv[i];
    auto need = [&](int i){ if (i+1>=argc) { print_usage(argv[0]); std::exit(1);} };
    if      (s == "--width")      { need(i); a.W = std::max(2, std::atoi(argv[++i])); }
    else if (s == "--height")     { need(i); a.H = std::max(2, std::atoi(argv[++i])); }
    else if (s == "--max-iter")   { need(i); a.max_iter = std::max(2, std::atoi(argv[++i])); }
    else if (s == "--xmin")       { need(i); a.xmin = std::atof(argv[++i]); }
    else if (s == "--xmax")       { need(i); a.xmax = std::atof(argv[++i]); }
    else if (s == "--ymin")       { need(i); a.ymin = std::atof(argv[++i]); }
    else if (s == "--ymax")       { need(i); a.ymax = std::atof(argv[++i]); }
    else if (s == "--color")      { a.color = true; }
    else if (s == "--smooth")     { a.smooth = true; }
    else if (s == "--binary")     { a.binary = true; }
    else if (s == "--verify")     { need(i); a.verify = std::max(0, std::atoi(argv[++i])); }
    else if (s == "--out")        { need(i); a.out_path = argv[++i]; }
    else if (s == "--help" || s=="-h") { print_usage(argv[0]); std::exit(0); }
    else { std::fprintf(stderr, "Unknown option: %s\n", s.c_str()); print_usage(argv[0]); std::exit(1); }
  }
  return a;
}

// ---------- CPU reference (escape iterations) ----------
/**
 * @brief Compute Mandelbrot escape iteration count for a single point (CPU reference).
 * @param cr Real part of complex coordinate
 * @param ci Imaginary part of complex coordinate
 * @param max_iter Maximum number of iterations
 * @return Number of iterations before escape (or max_iter if inside set)
 */
static inline int mandelbrot_iter_cpu(double cr, double ci, int max_iter) {
  double zr = 0.0, zi = 0.0;
  int it = 0;
  while (zr*zr + zi*zi <= 4.0 && it < max_iter) {
    double zr2 = zr*zr - zi*zi + cr;
    zi = 2.0*zr*zi + ci;
    zr = zr2;
    ++it;
  }
  return it;
}

// ---------- HSV helpers (device + host) ----------
/**
 * @brief Clamp a float value to [0,1] (device).
 */
__device__ __forceinline__ float clamp01f(float x){ return fminf(1.0f, fmaxf(0.0f, x)); }

/**
 * @brief Convert HSV color to RGB (device version).
 * @param h Hue [0,1]
 * @param s Saturation [0,1]
 * @param v Value [0,1]
 * @param R Output red channel
 * @param G Output green channel
 * @param B Output blue channel
 */
__device__ __forceinline__ void hsv_to_rgb_dev(float h, float s, float v,
                                               uint8_t& R, uint8_t& G, uint8_t& B) {
  float c = v * s;
  float hp = h * 6.0f;
  float x = c * (1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f));
  float r=0,g=0,b=0;
  if      (0.0f <= hp && hp < 1.0f) { r=c; g=x; b=0; }
  else if (1.0f <= hp && hp < 2.0f) { r=x; g=c; b=0; }
  else if (2.0f <= hp && hp < 3.0f) { r=0; g=c; b=x; }
  else if (3.0f <= hp && hp < 4.0f) { r=0; g=x; b=c; }
  else if (4.0f <= hp && hp < 5.0f) { r=x; g=0; b=c; }
  else                               { r=c; g=0; b=x; }
  float m = v - c;
  R = (uint8_t)lrintf((r+m)*255.0f);
  G = (uint8_t)lrintf((g+m)*255.0f);
  B = (uint8_t)lrintf((b+m)*255.0f);
}

/**
 * @brief Convert HSV color to RGB (host version).
 * @param h Hue [0,1]
 * @param s Saturation [0,1]
 * @param v Value [0,1]
 * @param R Output red channel
 * @param G Output green channel
 * @param B Output blue channel
 */
static inline void hsv_to_rgb_host(float h, float s, float v,
                                   uint8_t& R, uint8_t& G, uint8_t& B) {
  float c = v * s;
  float hp = h * 6.0f;
  float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
  float r=0,g=0,b=0;
  if      (0.0f <= hp && hp < 1.0f) { r=c; g=x; b=0; }
  else if (1.0f <= hp && hp < 2.0f) { r=x; g=c; b=0; }
  else if (2.0f <= hp && hp < 3.0f) { r=0; g=c; b=x; }
  else if (3.0f <= hp && hp < 4.0f) { r=0; g=x; b=c; }
  else if (4.0f <= hp && hp < 5.0f) { r=x; g=0; b=c; }
  else                               { r=c; g=0; b=x; }
  float m = v - c;
  R = (uint8_t)lround((r+m)*255.0);
  G = (uint8_t)lround((g+m)*255.0);
  B = (uint8_t)lround((b+m)*255.0);
}

/**
 * @brief Device palette: HSV rainbow mapping for Mandelbrot coloring.
 * @param t Normalized escape value [0,1]
 * @param R Output red channel
 * @param G Output green channel
 * @param B Output blue channel
 */
__device__ __forceinline__ void palette_dev(float t, uint8_t& R, uint8_t& G, uint8_t& B) {
  // Pleasant cycling hue; gentle value curve for contrast.
  float h = fmodf(0.95f + 10.0f * t, 1.0f);
  float s = 0.8f;
  float v = clamp01f(0.2f + sqrtf(t));
  hsv_to_rgb_dev(h, s, v, R, G, B);
}

// ---------- CUDA kernel ----------
/**
 * @brief CUDA kernel to compute Mandelbrot set image with color/grayscale options.
 *
 * @param out Output image buffer (device)
 * @param W Image width
 * @param H Image height
 * @param xmin Viewport min real
 * @param xmax Viewport max real
 * @param ymin Viewport min imag
 * @param ymax Viewport max imag
 * @param max_iter Iteration cap
 * @param mode 0=grayscale, 1=color
 * @param smooth Enable smooth coloring
 */
__global__ void mandelbrot_kernel(uint8_t* out, int W, int H,
                                  double xmin, double xmax,
                                  double ymin, double ymax,
                                  int max_iter, int mode, int smooth) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= W || py >= H) return;

  // Map pixel coordinates to complex plane
  double cr = xmin + (double(px) / double(W - 1)) * (xmax - xmin);
  double ci = ymin + (double(py) / double(H - 1)) * (ymax - ymin);

  // Iterate z = z^2 + c for Mandelbrot escape
  double zr = 0.0, zi = 0.0, r2 = 0.0;
  int it = 0;
  while ((r2 = zr*zr + zi*zi) <= 4.0 && it < max_iter) {
    double zr2 = zr*zr - zi*zi + cr;
    zi = 2.0*zr*zi + ci;
    zr = zr2;
    ++it;
  }

  if (mode == 0) {
    // Grayscale (PGM): inside set -> black, outside -> shade by escape
    uint8_t shade = (it == max_iter) ? 0
                                     : (uint8_t)lrintf(255.0f * float(it) / float(max_iter));
    out[py * W + px] = shade;
  } else {
    // Color (PPM): use palette and smooth coloring if enabled
    uint8_t R, G, B;
    if (it == max_iter) {
      R = G = B = 0; // inside set: black
    } else {
      float t;
      if (smooth) {
        // Smooth normalized escape time: mu = it - log2(log|z|)
        double r = sqrt(r2);
        double mu = it - log2(log(r));
        t = (float)(mu / (double)max_iter);
      } else {
        t = (float)it / (float)max_iter;
      }
      t = fminf(1.0f, fmaxf(0.0f, t));
      palette_dev(t, R, G, B);
    }
    // Write RGB triplet to output buffer
    size_t idx = (size_t(py) * (size_t)W + (size_t)px) * 3ULL;
    out[idx+0] = R; out[idx+1] = G; out[idx+2] = B;
  }
}

// ---------- Writers (ASCII + binary) ----------
/**
 * @brief Write grayscale image as ASCII PGM (P2).
 */
static bool write_pgm_ascii(const std::string& path, const uint8_t* img, int W, int H) {
  std::ofstream ofs(path);
  if (!ofs) return false;
  ofs << "P2\n" << W << " " << H << "\n255\n";
  for (int y=0; y<H; ++y) {
    for (int x=0; x<W; ++x) {
      ofs << int(img[y*W + x]) << (x+1==W ? '\n' : ' ');
    }
  }
  return true;
}
/**
 * @brief Write grayscale image as binary PGM (P5).
 */
static bool write_pgm_bin(const std::string& path, const uint8_t* img, int W, int H) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;
  ofs << "P5\n" << W << " " << H << "\n255\n";
  ofs.write(reinterpret_cast<const char*>(img), (std::streamsize)(W*H));
  return true;
}

/**
 * @brief Write color image as ASCII PPM (P3).
 */
static bool write_ppm_ascii(const std::string& path, const uint8_t* rgb, int W, int H) {
  std::ofstream ofs(path);
  if (!ofs) return false;
  ofs << "P3\n" << W << " " << H << "\n255\n";
  for (int y=0; y<H; ++y) {
    for (int x=0; x<W; ++x) {
      size_t i = (size_t(y)*W + size_t(x)) * 3ULL;
      ofs << int(rgb[i+0]) << " " << int(rgb[i+1]) << " " << int(rgb[i+2])
          << (x+1==W ? '\n' : ' ');
    }
  }
  return true;
}
/**
 * @brief Write color image as binary PPM (P6).
 */
static bool write_ppm_bin(const std::string& path, const uint8_t* rgb, int W, int H) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;
  ofs << "P6\n" << W << " " << H << "\n255\n";
  ofs.write(reinterpret_cast<const char*>(rgb), (std::streamsize)(3ULL*W*H));
  return true;
}

// ---------- CPU color for verification (matches device palette) ----------
/**
 * @brief Compute color for a single Mandelbrot point (CPU, matches device palette).
 * @param cr Real part of complex coordinate
 * @param ci Imaginary part of complex coordinate
 * @param max_iter Maximum number of iterations
 * @param smooth Enable smooth coloring
 * @param R Output red channel
 * @param G Output green channel
 * @param B Output blue channel
 */
static inline void cpu_color_for_point(double cr, double ci, int max_iter, bool smooth,
                                       uint8_t& R, uint8_t& G, uint8_t& B) {
  double zr = 0.0, zi = 0.0, r2 = 0.0;
  int it = 0;
  while ((r2 = zr*zr + zi*zi) <= 4.0 && it < max_iter) {
    double zr2 = zr*zr - zi*zi + cr;
    zi = 2.0*zr*zi + ci;
    zr = zr2;
    ++it;
  }
  if (it == max_iter) { R = G = B = 0; return; }
  float t;
  if (smooth) {
    double r = std::sqrt(r2);
    double mu = it - std::log2(std::log(r));
    t = (float)(mu / (double)max_iter);
  } else {
    t = (float)it / (float)max_iter;
  }
  if (t < 0.f) t = 0.f; else if (t > 1.f) t = 1.f;
  float h = std::fmod(0.95f + 10.0f * t, 1.0f);
  float s = 0.8f;
  float v = std::min(1.0f, std::max(0.0f, 0.2f + std::sqrt((double)t)));
  hsv_to_rgb_host(h, s, v, R, G, B);
}

/**
 * @brief Main entry point: parse args, launch CUDA kernel, write image, verify.
 */
int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  // Print configuration summary
  std::printf("Mandelbrot %dx%d, max_iter=%d, %s, %s, %s -> %s\n",
              a.W, a.H, a.max_iter,
              a.color ? (a.smooth ? "COLOR(smooth)" : "COLOR") : "GRAYSCALE",
              a.binary ? "BINARY" : "ASCII",
              "CUDA",
              a.out_path.c_str());
  std::printf("Viewport: Re[%g, %g], Im[%g, %g]\n",
              a.xmin, a.xmax, a.ymin, a.ymax);

  // Allocate host and device buffers for image
  const size_t chans = a.color ? 3ULL : 1ULL;
  const size_t bytes = chans * (size_t)a.W * (size_t)a.H;
  std::vector<uint8_t> h_img(bytes, 0);
  uint8_t* d_img = nullptr;
  CUDA_CHECK(cudaMalloc(&d_img, bytes));

  // Launch CUDA kernel for Mandelbrot computation
  dim3 block(16,16);
  dim3 grid((a.W + block.x - 1)/block.x, (a.H + block.y - 1)/block.y);
  mandelbrot_kernel<<<grid, block>>>(d_img, a.W, a.H, a.xmin, a.xmax, a.ymin, a.ymax,
                                     a.max_iter, a.color ? 1 : 0, a.smooth ? 1 : 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host and free device memory
  CUDA_CHECK(cudaMemcpy(h_img.data(), d_img, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_img));

  // Write image to file (PGM/PPM, ASCII/binary)
  bool ok_write = false;
  if (!a.color) {
    ok_write = a.binary ? write_pgm_bin(a.out_path, h_img.data(), a.W, a.H)
                        : write_pgm_ascii(a.out_path, h_img.data(), a.W, a.H);
  } else {
    ok_write = a.binary ? write_ppm_bin(a.out_path, h_img.data(), a.W, a.H)
                        : write_ppm_ascii(a.out_path, h_img.data(), a.W, a.H);
  }
  if (!ok_write) {
    std::fprintf(stderr, "Failed to write output: %s\n", a.out_path.c_str());
    return EXIT_FAILURE;
  }

  // Optional CPU spot verification of a few pixels
  if (a.verify > 0) {
    std::printf("Self-test: sampling %d pixel(s)\n", a.verify);
    // Prepare a small set: corners, center, edges, plus random-ish picks
    std::vector<std::pair<int,int>> samples;
    samples.push_back({a.W/2, a.H/2});
    samples.push_back({0,0});
    samples.push_back({a.W-1,0});
    samples.push_back({0,a.H-1});
    samples.push_back({a.W-1,a.H-1});
    // Add a few more along midlines
    samples.push_back({int(a.W*0.15), a.H/2});
    samples.push_back({int(a.W*0.45), int(a.H*0.5)});
    samples.push_back({int(a.W*0.80), int(a.H*0.33)});
    // Trim to requested verify count
    if ((int)samples.size() > a.verify) samples.resize(a.verify);

    bool pass = true;
    for (size_t i=0; i<samples.size(); ++i) {
      int px = samples[i].first;
      int py = samples[i].second;
      double cr = a.xmin + (double(px) / double(a.W - 1)) * (a.xmax - a.xmin);
      double ci = a.ymin + (double(py) / double(a.H - 1)) * (a.ymax - a.ymin);

      if (!a.color) {
        int it = mandelbrot_iter_cpu(cr, ci, a.max_iter);
        uint8_t expected = (it == a.max_iter) ? 0
          : (uint8_t)lround(255.0 * it / a.max_iter);
        uint8_t got = h_img[(size_t)py * (size_t)a.W + (size_t)px];
        if (expected != got) {
          pass = false;
          std::printf("Mismatch @(%d,%d): CPU-gray=%u, GPU=%u (iters=%d)\n",
                      px, py, (unsigned)expected, (unsigned)got, it);
        }
      } else {
        uint8_t r,g,b;
        cpu_color_for_point(cr, ci, a.max_iter, a.smooth, r,g,b);
        size_t idx = ((size_t)py * (size_t)a.W + (size_t)px) * 3ULL;
        uint8_t rr = h_img[idx+0], gg = h_img[idx+1], bb = h_img[idx+2];
        if (r!=rr || g!=gg || b!=bb) {
          pass = false;
          std::printf("Mismatch @(%d,%d): CPU=(%3u,%3u,%3u) GPU=(%3u,%3u,%3u)\n",
                      px, py, (unsigned)r,(unsigned)g,(unsigned)b,
                      (unsigned)rr,(unsigned)gg,(unsigned)bb);
        }
      }
    }
    std::puts(pass ? "Self-test: PASS." : "Self-test: FAIL (see mismatches).");
  }

  std::puts("Done.");
  return EXIT_SUCCESS;
}
