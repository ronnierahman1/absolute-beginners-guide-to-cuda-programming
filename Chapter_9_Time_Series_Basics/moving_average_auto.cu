// moving_average_auto.cu
// CPU reference + two GPU paths (window-loop and prefix-sum) + auto dispatcher.

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

// Thrust for prefix-sum path
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(_e));            \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// -------------------------------
// CPU reference (causal, edge fill)
// -------------------------------
void moving_average_causal_cpu(const float* x, float* y, int N, int W, float edge_fill)
{
    if (W <= 0) { std::fill(y, y + N, edge_fill); return; }
    for (int i = 0; i < N; ++i) {
        if (i < W - 1) {
            y[i] = edge_fill;
        } else {
            float sum = 0.0f;
            int start = i - (W - 1);
            for (int k = 0; k < W; ++k) sum += x[start + k];
            y[i] = sum / static_cast<float>(W);
        }
    }
}

// -------------------------------
// GPU path A: per-thread window loop
// -------------------------------
__global__ void moving_average_causal_window(const float* __restrict__ x,
                                             float* __restrict__ y,
                                             int N, int W, float edge_fill)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (i < W - 1) { y[i] = edge_fill; return; }

    float sum = 0.0f;
    int start = i - (W - 1);
    #pragma unroll
    for (int k = 0; k < W; ++k) sum += x[start + k];
    y[i] = sum / static_cast<float>(W);
}

// -------------------------------
// GPU path B: prefix-sum + O(1) per output
// -------------------------------
__global__ void moving_average_from_prefix(const float* __restrict__ prefix,
                                           float* __restrict__ y,
                                           int N, int W, float edge_fill)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (i < W - 1) { y[i] = edge_fill; return; }

    // Inclusive prefix => sum[0..i] is prefix[i]
    // Sum of last W samples: S = prefix[i] - prefix[i-W]
    float prev = (i - W >= 0) ? prefix[i - W] : 0.0f;
    float s    = prefix[i] - prev;
    y[i]       = s / static_cast<float>(W);
}

// -------------------------------
// Dispatcher: chooses path based on W
// -------------------------------
void moving_average_causal_gpu_auto(const float* d_x, float* d_y,
                                    int N, int W, float edge_fill,
                                    int window_threshold = 64,
                                    float* d_tmp_prefix = nullptr) // optional scratch
{
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    if (W <= window_threshold) {
        // Path A: small W → per-thread loop
        moving_average_causal_window<<<blocks, threads>>>(d_x, d_y, N, W, edge_fill);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Path B: large W → prefix-sum
    // Allocate prefix buffer if not provided
    float* d_prefix = d_tmp_prefix;
    bool   owner    = false;
    if (!d_prefix) { CUDA_CHECK(cudaMalloc(&d_prefix, N * sizeof(float))); owner = true; }

    // Inclusive scan: prefix[i] = sum_{k=0..i} x[k]
    thrust::device_ptr<const float> in(d_x);
    thrust::device_ptr<float>       out(d_prefix);
    thrust::inclusive_scan(in, in + N, out);

    // Compute moving average from prefix
    moving_average_from_prefix<<<blocks, threads>>>(d_prefix, d_y, N, W, edge_fill);
    CUDA_CHECK(cudaGetLastError());

    if (owner) CUDA_CHECK(cudaFree(d_prefix));
}

// -------------------------------
// Utility: compare arrays
// -------------------------------
struct CompareResult { double max_abs_err; int max_err_index; int mismatches; };

CompareResult compare_arrays_range(const float* a, const float* b, int start, int end, float atol)
{
    CompareResult r{0.0, -1, 0};
    for (int i = start; i < end; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > r.max_abs_err) { r.max_abs_err = diff; r.max_err_index = i; }
        if (diff > atol) r.mismatches++;
    }
    return r;
}

// -------------------------------
// Main: generate data, CPU+GPU (auto), compare
// -------------------------------
int main(int argc, char** argv)
{
    int   N         = (1 << 20);
    int   W         = 5;
    float edge_fill = 0.0f;
    int   threshold = 64; // switch point between window vs prefix paths

    if (argc >= 2) N = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) W = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) edge_fill = std::atof(argv[3]);
    if (argc >= 5) threshold = std::max(1, std::atoi(argv[4]));

    std::printf("N=%d, W=%d, edge_fill=%g, switch@W<=%d → window else prefix\n",
                N, W, edge_fill, threshold);

    // Host input/output
    std::vector<float> h_x(N), h_y_cpu(N), h_y_gpu(N);
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

    // GPU timing (events)
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    moving_average_causal_gpu_auto(d_x, d_y, N, W, edge_fill, threshold);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare CPU vs GPU (skip early indices if using NaN edge fill)
    const float atol = 1e-5f;
    int start_compare = 0;
    if (std::isnan(edge_fill)) start_compare = std::min(W - 1, N);
    CompareResult cr = compare_arrays_range(h_y_cpu.data(), h_y_gpu.data(), start_compare, N, atol);

    // Sample print
    std::printf("\nSample (first 8 elements):\n");
    for (int i = 0; i < std::min(8, N); ++i) {
        std::printf("i=%d  x=% .4f  cpu=% .4f  gpu=% .4f\n",
                    i, h_x[i], h_y_cpu[i], h_y_gpu[i]);
    }

    std::printf("\nCPU time: %.3f ms | GPU time (incl. scan if used): %.3f ms\n", cpu_ms, gpu_ms);
    std::printf("Max |CPU-GPU| error: %.6g at index %d\n", cr.max_abs_err, cr.max_err_index);
    std::printf("Mismatches (> %.1e): %d (compared [%d, %d))\n",
                atol, cr.mismatches, start_compare, N);

    bool ok = (cr.mismatches == 0);
    std::printf("\nRESULT: %s\n", ok ? "PASS ✅" : "FAIL ❌");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return ok ? 0 : 1;
}
