// moving_average_test.cu
// CPU reference + CUDA kernel + full test harness for causal moving average.
// Policy: y[i] = mean(x[i-W+1 .. i]) for i >= W-1; else y[i] = edge_fill.

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>

// ---------------------------------------------
// Error-checking helper
// ---------------------------------------------
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(_e));            \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// ---------------------------------------------
// CPU reference (causal W-point moving average)
// ---------------------------------------------
void moving_average_causal_cpu(const float* x, float* y, int N, int W, float edge_fill)
{
    if (W <= 0) { // defensive
        std::fill(y, y + N, edge_fill);
        return;
    }
    for (int i = 0; i < N; ++i) {
        if (i < W - 1) {
            y[i] = edge_fill;
        } else {
            float sum = 0.0f;
            // window is x[i-W+1 .. i]
            int start = i - (W - 1);
            for (int k = 0; k < W; ++k) {
                sum += x[start + k];
            }
            y[i] = sum / static_cast<float>(W);
        }
    }
}

// ---------------------------------------------
// CUDA kernel (same policy as CPU)
// ---------------------------------------------
__global__ void moving_average_causal_gpu(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int N, int W, float edge_fill)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (i < W - 1) {
        y[i] = edge_fill;
        return;
    }

    float sum = 0.0f;
    int start = i - (W - 1);
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        sum += x[start + k];
    }
    y[i] = sum / static_cast<float>(W);
}

// ---------------------------------------------
// Utility: max-abs error with tolerance
// ---------------------------------------------
struct CompareResult {
    double max_abs_err;
    int    max_err_index;
    int    mismatches; // counts elements beyond tolerance
};

CompareResult compare_arrays(const float* a, const float* b, int N, float atol)
{
    CompareResult r{0.0, -1, 0};
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > r.max_abs_err) {
            r.max_abs_err = diff;
            r.max_err_index = i;
        }
        if (diff > atol) {
            r.mismatches++;
        }
    }
    return r;
}

// ---------------------------------------------
// Main: generate data, run CPU+GPU, compare
// ---------------------------------------------
int main(int argc, char** argv)
{
    int   N         = (1 << 20);   // 1,048,576
    int   W         = 5;           // window length
    float edge_fill = 0.0f;        // default fill; set to NAN if desired

    if (argc >= 2) N = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) W = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) edge_fill = std::atof(argv[3]);

    std::printf("N=%d, W=%d, edge_fill=%g\n", N, W, edge_fill);

    // Host buffers
    std::vector<float> h_x(N), h_y_cpu(N), h_y_gpu(N);

    // Seeded random input for reproducibility
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < N; ++i) h_x[i] = dist(rng);

    // CPU reference timing
    auto t0 = std::chrono::high_resolution_clock::now();
    moving_average_causal_cpu(h_x.data(), h_y_cpu.data(), N, W, edge_fill);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Device buffers
    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // GPU timing (cuda events)
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(ev_start));
    moving_average_causal_gpu<<<blocks, threads>>>(d_x, d_y, N, W, edge_fill);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError()); // catch kernel launch errors

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare CPU vs GPU
    // Note: If you set edge_fill = NAN, you should skip indices [0..W-2] when comparing.
    const float atol = 1e-5f; // absolute tolerance suitable for float here
    int start_compare = 0;
    int end_compare   = N;

    // If edge_fill is NaN, skip the first W-1 where both are NaN (or undefined for comparison).
    if (std::isnan(edge_fill)) start_compare = std::min(W - 1, N);

    CompareResult cr{0.0, -1, 0};
    for (int i = start_compare; i < end_compare; ++i) {
        float diff = std::fabs(h_y_cpu[i] - h_y_gpu[i]);
        if (diff > cr.max_abs_err) {
            cr.max_abs_err = diff;
            cr.max_err_index = i;
        }
        if (diff > atol || (!std::isnan(edge_fill) && std::isnan(h_y_cpu[i]) != std::isnan(h_y_gpu[i]))) {
            cr.mismatches++;
        }
    }

    // Print a small sample for sanity
    std::printf("\nSample (first 8 elements):\n");
    for (int i = 0; i < std::min(8, N); ++i) {
        std::printf("i=%d  x=% .4f  cpu=% .4f  gpu=% .4f\n",
                    i, h_x[i], h_y_cpu[i], h_y_gpu[i]);
    }

    std::printf("\nCPU time: %.3f ms | GPU kernel time: %.3f ms\n", cpu_ms, gpu_ms);
    std::printf("Max |CPU-GPU| error: %.6g at index %d\n", cr.max_abs_err, cr.max_err_index);
    std::printf("Mismatches (> %.1e): %d (over compared range [%d, %d))\n",
                atol, cr.mismatches, start_compare, end_compare);

    bool ok = (cr.mismatches == 0);
    std::printf("\nRESULT: %s\n", ok ? "PASS ✅" : "FAIL ❌");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return ok ? 0 : 1;
}
