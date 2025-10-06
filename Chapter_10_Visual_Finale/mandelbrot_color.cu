
// mandelbrot_color.cu
// CUDA Mandelbrot set renderer with smooth coloring (HSV→RGB), writes PPM (P6), with CPU spot-check for correctness.
// Usage: ./mandelbrot_color [width height max_iter output.ppm]
//
// - Renders the Mandelbrot set using CUDA, with each pixel colored by a smooth palette based on escape time.
// - Output is a binary PPM image (P6 format).
// - Includes a CPU reference for spot-checking a few pixels for correctness.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>


// CUDA error checking macro: prints error and exits on failure
#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                     \
                   #call, __FILE__, __LINE__, cudaGetErrorString(err__));        \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)


// Write a binary PPM (P6) image from an RGB buffer
// path: output file path
// rgb:  vector of RGB bytes (size = 3*W*H)
// W,H: image width and height
bool write_ppm(const std::string& path, const std::vector<uint8_t>& rgb, int W, int H)
{
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;
  ofs << "P6\n" << W << " " << H << "\n255\n";
  ofs.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(3ULL * W * H));
  return true;
}

// ---------- Device utilities ----------

// Clamp a float to [0,1] (device utility)
__device__ __forceinline__ float clamp01(float x) {
  return fminf(1.0f, fmaxf(0.0f, x));
}


// Convert HSV color (h in [0,1), s,v in [0,1]) to RGB [0,255] (device)
__device__ __forceinline__ void hsv_to_rgb(float h, float s, float v,
                                           uint8_t& R, uint8_t& G, uint8_t& B)
{
  float c = v * s;
  float hp = h * 6.0f;
  float x = c * (1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f));
  float r=0, g=0, b=0;
  if      (0.0f <= hp && hp < 1.0f) { r = c; g = x; b = 0; }
  else if (1.0f <= hp && hp < 2.0f) { r = x; g = c; b = 0; }
  else if (2.0f <= hp && hp < 3.0f) { r = 0; g = c; b = x; }
  else if (3.0f <= hp && hp < 4.0f) { r = 0; g = x; b = c; }
  else if (4.0f <= hp && hp < 5.0f) { r = x; g = 0; b = c; }
  else                               { r = c; g = 0; b = x; }
  float m = v - c;
  R = (uint8_t)lrintf((r + m) * 255.0f);
  G = (uint8_t)lrintf((g + m) * 255.0f);
  B = (uint8_t)lrintf((b + m) * 255.0f);
}


// Map smooth iteration value t ∈ [0,1] to a pleasant palette via HSV.
// t: normalized smooth iteration count
// R,G,B: output color channels [0,255]
// (Feel free to tweak saturation/value curves for different looks)
__device__ __forceinline__ void palette(float t, uint8_t& R, uint8_t& G, uint8_t& B)
{
  // Cycle hue across the spectrum; boost contrast with sqrt curve.
  float h = fmodf(0.95f + 10.0f * t, 1.0f); // wrap-around hue cycles
  float s = 0.8f;
  float v = clamp01(0.2f + sqrtf(t));       // dark → bright smoothly
  hsv_to_rgb(h, s, v, R, G, B);
}

// ---------- Kernel ----------
// CUDA kernel: compute Mandelbrot set with smooth coloring for each pixel
// out_rgb: output RGB buffer (device)
// W,H: image width/height
// xmin,xmax,ymin,ymax: complex plane bounds
// max_iter: max iterations for escape
__global__ void mandelbrot_kernel_color(uint8_t* out_rgb, int W, int H,
                                        double xmin, double xmax,
                                        double ymin, double ymax,
                                        int max_iter)
{
  // Compute pixel coordinates
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= W || py >= H) return;

  // Map pixel to complex plane (cr, ci)
  double cr = xmin + (double(px) / double(W - 1)) * (xmax - xmin);
  double ci = ymin + (double(py) / double(H - 1)) * (ymax - ymin);

  // Mandelbrot iteration: z = z^2 + c
  double zr = 0.0, zi = 0.0;
  int it = 0;
  double r2 = 0.0;
  while ((r2 = zr * zr + zi * zi) <= 4.0 && it < max_iter) {
    double zr2 = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = zr2;
    ++it;
  }

  uint8_t R, G, B;
  if (it == max_iter) {
    // Inside Mandelbrot set: color black
    R = G = B = 0;
  } else {
    // Smooth coloring: use fractional escape time for smooth palette
    // mu = it - log2(log|z|), normalized to [0,1]
    double r = sqrt(r2);
    double mu = it - log2(log(r));
    float t = (float)(mu / (double)max_iter);
    t = fminf(1.0f, fmaxf(0.0f, t));
    palette(t, R, G, B);
  }

  // Write RGB to output buffer
  size_t idx = (size_t(py) * (size_t)W + (size_t)px) * 3ULL;
  out_rgb[idx + 0] = R;
  out_rgb[idx + 1] = G;
  out_rgb[idx + 2] = B;
}

// ---------- CPU reference for spot-tests (matches kernel math) ----------
// CPU reference: compute color for a single point (matches device math)
// Used for spot-checking a few pixels for correctness
static inline void cpu_color_for_point(double cr, double ci, int max_iter,
                                       uint8_t& R, uint8_t& G, uint8_t& B)
{
  double zr = 0.0, zi = 0.0, r2 = 0.0;
  int it = 0;
  while ((r2 = zr * zr + zi * zi) <= 4.0 && it < max_iter) {
    double zr2 = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = zr2;
    ++it;
  }
  if (it == max_iter) { R = G = B = 0; return; }
  double r = std::sqrt(r2);
  double mu = it - std::log2(std::log(r));
  float t = float(mu / double(max_iter));
  if (t < 0.f) t = 0.f; else if (t > 1.f) t = 1.f;

  // Same palette as device:
  auto hsv_to_rgb_host = [](float h, float s, float v, uint8_t& r8, uint8_t& g8, uint8_t& b8){
    float c = v * s;
    float hp = h * 6.0f;
    float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
    float r=0, g=0, b=0;
    if      (0.0f <= hp && hp < 1.0f) { r = c; g = x; b = 0; }
    else if (1.0f <= hp && hp < 2.0f) { r = x; g = c; b = 0; }
    else if (2.0f <= hp && hp < 3.0f) { r = 0; g = c; b = x; }
    else if (3.0f <= hp && hp < 4.0f) { r = 0; g = x; b = c; }
    else if (4.0f <= hp && hp < 5.0f) { r = x; g = 0; b = c; }
    else                               { r = c; g = 0; b = x; }
    float m = v - c;
    r8 = (uint8_t)lround((r + m) * 255.0);
    g8 = (uint8_t)lround((g + m) * 255.0);
    b8 = (uint8_t)lround((b + m) * 255.0);
  };
  float h = fmodf(0.95f + 10.0f * t, 1.0f);
  float s = 0.8f;
  float v = std::fmin(1.0f, std::fmax(0.0f, 0.2f + std::sqrt(t)));
  hsv_to_rgb_host(h, s, v, R, G, B);
}

// Main entry point: parse args, launch kernel, write image, run spot-checks
int main(int argc, char** argv)
{

  // Parse command-line arguments
  int W = 1024, H = 768, max_iter = 200;
  std::string out_path = "mandelbrot.ppm";
  if (argc >= 3) { W = std::max(2, std::atoi(argv[1])); H = std::max(2, std::atoi(argv[2])); }
  if (argc >= 4) max_iter = std::max(2, std::atoi(argv[3]));
  if (argc >= 5) out_path = argv[4];

  // Classic Mandelbrot view: real [-2.0, 1.0], imag [-1.5, 1.5]
  const double xmin = -2.0, xmax = 1.0;
  const double ymin = -1.5, ymax = 1.5;

  std::printf("Rendering (color) Mandelbrot %dx%d, max_iter=%d -> %s\n",
              W, H, max_iter, out_path.c_str());

  // Allocate host and device RGB buffers
  std::vector<uint8_t> h_rgb(3ULL * W * H, 0);
  uint8_t* d_rgb = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rgb, 3ULL * W * H));

  // Launch configuration: 16x16 blocks
  dim3 block(16,16);
  dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y);

  // Launch Mandelbrot kernel
  mandelbrot_kernel_color<<<grid, block>>>(d_rgb, W, H, xmin, xmax, ymin, ymax, max_iter);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host and write PPM image
  CUDA_CHECK(cudaMemcpy(h_rgb.data(), d_rgb, 3ULL * W * H, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_rgb));
  if (!write_ppm(out_path, h_rgb, W, H)) {
    std::fprintf(stderr, "Failed to write PPM: %s\n", out_path.c_str());
    return EXIT_FAILURE;
  }

  // -------- Self-test: CPU spot-check on 6 pixels --------
  struct Sample { int px, py; const char* label; };
  std::vector<Sample> tests = {
      { W/2, H/2, "center" },
      { 0, 0, "top-left" },
      { W-1, 0, "top-right" },
      { 0, H-1, "bottom-left" },
      { W-1, H-1, "bottom-right" },
      { int(0.45 * W), int(0.5 * H), "seahorse-ish" }
  };

  bool ok = true;
  for (const auto& s : tests) {
    // Map pixel to complex plane
    double cr = xmin + (double(s.px) / double(W - 1)) * (xmax - xmin);
    double ci = ymin + (double(s.py) / double(H - 1)) * (ymax - ymin);
    uint8_t r,g,b;
    cpu_color_for_point(cr, ci, max_iter, r, g, b);
    size_t idx = (size_t(s.py) * (size_t)W + (size_t)s.px) * 3ULL;
    uint8_t rr = h_rgb[idx+0], gg = h_rgb[idx+1], bb = h_rgb[idx+2];
    if (r != rr || g != gg || b != bb) {
      ok = false;
      std::fprintf(stderr,
        "Self-test mismatch at %-12s (%4d,%4d): expected (%3u,%3u,%3u), got (%3u,%3u,%3u)\n",
        s.label, s.px, s.py, (unsigned)r,(unsigned)g,(unsigned)b,
        (unsigned)rr,(unsigned)gg,(unsigned)bb);
    }
  }
  std::puts(ok ? "Self-test: PASS." : "Self-test: FAIL (see mismatches).");
  std::puts("Done.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
