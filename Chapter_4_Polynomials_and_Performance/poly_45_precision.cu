// poly_45_precision.cu
// Demonstrates: float vs double kernels, constant-memory for each,
// and a mixed-precision kernel (float coeffs + double accumulator).
//
// Build:
//   nvcc -O3 poly_45_precision.cu -o poly45
//
// Run:
//   ./poly45
//
// Notes:
//   * For stricter bit-reproducibility, avoid -use_fast_math.
//   * For throughput with float, you may try: nvcc -O3 -use_fast_math ...

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>

#ifndef MAX_DEGREE_PLUS_1
#define MAX_DEGREE_PLUS_1 64
#endif

#define CUDA_CHECK(call) do {                                   \
    cudaError_t _err = (call);                                  \
    if (_err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s at %s:%d\n",             \
                cudaGetErrorString(_err), __FILE__, __LINE__);  \
        std::exit(1);                                           \
    }                                                           \
} while(0)

// -------------------- Constant memory for both precisions --------------------
__constant__ float  d_coeffs_f[MAX_DEGREE_PLUS_1];
__constant__ double d_coeffs_d[MAX_DEGREE_PLUS_1];

// -------------------- CPU reference (double) --------------------
static inline double horner_ref_d(const std::vector<double>& a, int deg, double x) {
    double r = a[deg];
    for (int k = deg - 1; k >= 0; --k) r = std::fma(r, x, a[k]);
    return r;
}

static inline double rel_err(double y, double yref) {
    double denom = std::max(1.0, std::fabs(yref));
    return std::fabs(y - yref) / denom;
}

// -------------------- Kernels --------------------
// (A) Float kernel (reads from d_coeffs_f), uses fmaf
__global__ void eval_poly_const_f(const float* __restrict__ xs,
                                  float* __restrict__ ys,
                                  int n, int deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x   = xs[i];
    float acc = d_coeffs_f[deg];

    #pragma unroll
    for (int k = deg - 1; k >= 0; --k) {
        acc = fmaf(acc, x, d_coeffs_f[k]); // FMA (float)
    }
    ys[i] = acc;
}

// (B) Double kernel (reads from d_coeffs_d), uses fma
__global__ void eval_poly_const_d(const double* __restrict__ xs,
                                  double* __restrict__ ys,
                                  int n, int deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double x   = xs[i];
    double acc = d_coeffs_d[deg];

    #pragma unroll
    for (int k = deg - 1; k >= 0; --k) {
        acc = fma(acc, x, d_coeffs_d[k]); // FMA (double)
    }
    ys[i] = acc;
}

// (C) Mixed precision: xs/ys are float, coeffs in float constant memory,
// but accumulate in double for stability, cast back to float at the end.
__global__ void eval_poly_mixed_fcoeffs_double_accum(const float* __restrict__ xs,
                                                     float* __restrict__ ys,
                                                     int n, int deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double x   = static_cast<double>(xs[i]);
    double acc = static_cast<double>(d_coeffs_f[deg]);

    #pragma unroll
    for (int k = deg - 1; k >= 0; --k) {
        acc = fma(acc, x, static_cast<double>(d_coeffs_f[k]));
    }
    ys[i] = static_cast<float>(acc);
}

int main() {
    const int deg = 12;          // tweak within MAX_DEGREE_PLUS_1-1
    const int N   = 1 << 15;     // number of inputs

    assert(deg + 1 <= MAX_DEGREE_PLUS_1);

    // Random fixtures
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> Ux(-3.0, 3.0);
    std::uniform_real_distribution<double> Ua(-1.0, 1.0);

    // Host coefficients (double as reference)
    std::vector<double> h_coeffs_d(deg + 1);
    for (int k = 0; k <= deg; ++k) h_coeffs_d[k] = Ua(rng);

    // Also a float copy (for float kernels / constant memory)
    std::vector<float> h_coeffs_f(deg + 1);
    for (int k = 0; k <= deg; ++k) h_coeffs_f[k] = static_cast<float>(h_coeffs_d[k]);

    // Upload constants (once)
    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_d, h_coeffs_d.data(), (deg + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs_f, h_coeffs_f.data(), (deg + 1) * sizeof(float)));

    // Inputs
    std::vector<double> hx_d(N), hyref_d(N);
    for (int i = 0; i < N; ++i) {
        hx_d[i]   = Ux(rng);
        hyref_d[i]= horner_ref_d(h_coeffs_d, deg, hx_d[i]); // CPU double reference
    }
    // Float view of inputs for float/mixed kernels
    std::vector<float> hx_f(N);
    for (int i = 0; i < N; ++i) hx_f[i] = static_cast<float>(hx_d[i]);

    // Device buffers
    float  *d_x_f=nullptr, *d_y_f=nullptr;
    double *d_x_d=nullptr, *d_y_d=nullptr;
    float  *d_y_mixed=nullptr;

    CUDA_CHECK(cudaMalloc(&d_x_f, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_f, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_d, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_d, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_mixed, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x_f, hx_f.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_d, hx_d.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // (A) Float kernel
    eval_poly_const_f<<<blocks, threads>>>(d_x_f, d_y_f, N, deg);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> hy_f(N);
    CUDA_CHECK(cudaMemcpy(hy_f.data(), d_y_f, N * sizeof(float), cudaMemcpyDeviceToHost));

    // (B) Double kernel
    eval_poly_const_d<<<blocks, threads>>>(d_x_d, d_y_d, N, deg);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> hy_d(N);
    CUDA_CHECK(cudaMemcpy(hy_d.data(), d_y_d, N * sizeof(double), cudaMemcpyDeviceToHost));

    // (C) Mixed precision kernel
    eval_poly_mixed_fcoeffs_double_accum<<<blocks, threads>>>(d_x_f, d_y_mixed, N, deg);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> hy_mixed(N);
    CUDA_CHECK(cudaMemcpy(hy_mixed.data(), d_y_mixed, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare against CPU double reference
    double max_rel_err_float = 0.0, max_rel_err_double = 0.0, max_rel_err_mixed = 0.0;
    for (int i = 0; i < N; ++i) {
        max_rel_err_float  = std::max(max_rel_err_float,
                                      rel_err(static_cast<double>(hy_f[i]), hyref_d[i]));
        max_rel_err_double = std::max(max_rel_err_double, rel_err(hy_d[i], hyref_d[i]));
        max_rel_err_mixed  = std::max(max_rel_err_mixed,
                                      rel_err(static_cast<double>(hy_mixed[i]), hyref_d[i]));
    }

    printf("[float ] max relative error vs CPU double: %.3e\n", max_rel_err_float);
    printf("[double] max relative error vs CPU double: %.3e\n", max_rel_err_double);
    printf("[mixed ] max relative error vs CPU double: %.3e\n", max_rel_err_mixed);

    // Spot-check the first few
    printf("\nSpot-check (first 3):\n");
    for (int i = 0; i < 3; ++i) {
        printf("x=% .4f  CPU=%.8e  float=%.8e  double=%.8e  mixed=%.8e\n",
               hx_d[i], hyref_d[i],
               static_cast<double>(hy_f[i]),
               hy_d[i],
               static_cast<double>(hy_mixed[i]));
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_x_f));
    CUDA_CHECK(cudaFree(d_y_f));
    CUDA_CHECK(cudaFree(d_x_d));
    CUDA_CHECK(cudaFree(d_y_d));
    CUDA_CHECK(cudaFree(d_y_mixed));

    printf("\nDone.\n");
    return 0;
}
