// compare_cpu_gpu_scale.cu
//
// Purpose:
//   Compare a simple CPU reference implementation against a CUDA GPU kernel
//   for the operation: y[i] = 10 * x[i] + 5
//
// What this file demonstrates (skills from Chapter 2):
//   • Mapping threads to data using the global index i = blockIdx.x * blockDim.x + threadIdx.x
//   • Protecting against out-of-bounds with if (i < N)
//   • Verifying correctness by comparing GPU results to a simple CPU “golden” result
//   • (Optional) Experimenting with different grid/block sizes via command-line args
//
// Build:
//   nvcc -O2 compare_cpu_gpu_scale.cu -o compare
//
// Run (defaults: N=1<<20, TPB=256):
//   ./compare
//
// Run with custom N and TPB (threads per block):
//   ./compare 1000000 128
//
// Notes:
//   • This program prints PASS/FAIL and reports a few mismatches if any.
//   • It’s intentionally straightforward and heavily commented for learners.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// -----------------------------
// Error-checking convenience
// -----------------------------
#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t _err = (call);                                                     \
        if (_err != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_err));              \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

// -----------------------------
// CPU reference implementation
// -----------------------------
// Applies y[i] = 10 * x[i] + 5 for i in [0, N)
void scale10x5_cpu(const float* x, float* y, int N) {
    for (int i = 0; i < N; ++i) {
        y[i] = 10.0f * x[i] + 5.0f;
    }
}

// -----------------------------
// GPU kernel implementation
// -----------------------------
// Each thread handles exactly one element (if i < N).
__global__ void scale10x5_gpu(const float* __restrict__ x,
                              float* __restrict__ y,
                              int N)
{
    // Global index: which element this thread owns
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: prevents out-of-range memory access when the grid overshoots N
    if (i < N) {
        y[i] = 10.0f * x[i] + 5.0f;
    }
}

// -----------------------------
// Result comparison helper
// -----------------------------
// Compares y_cpu vs y_gpu element-wise with a small tolerance for floats.
// Reports up to 'max_report' mismatches.

bool compare_results(const float* a,
                     const float* b,
                     int N,
                     float tol = 1e-5f,
                     int max_report = 10)
{
    bool pass = true;
    int reported = 0;        // how many we printed
    int total_mismatches = 0; // how many we actually saw

    for (int i = 0; i < N; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tol) {
            ++total_mismatches;
            if (reported < max_report) {
                printf("Mismatch at %d: CPU=%f, GPU=%f, |diff|=%f\n",
                       i, a[i], b[i], diff);
                ++reported;
            }
            pass = false;
        }
    }

    // Only print the "additional mismatches" line if we *know*
    // there were more mismatches than we reported.
    if (!pass && total_mismatches > reported) {
        printf("... additional mismatches not shown (reported %d of %d; limit %d)\n",
               reported, total_mismatches, max_report);
    }

    return pass;
}


// -----------------------------
// Main
// -----------------------------
int main(int argc, char** argv) {
    // -----------------------------
    // 1) Parse inputs or use defaults
    // -----------------------------
    // N: number of elements. TPB: threads per block.
    int N   = (1 << 20); // ~1 million by default
    int TPB = 256;       // a common, warp-friendly default

    if (argc >= 2) N   = std::atoi(argv[1]);
    if (argc >= 3) TPB = std::atoi(argv[2]);

    if (N <= 0 || TPB <= 0) {
        fprintf(stderr, "Usage: %s [N>0] [TPB>0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("Comparing CPU vs GPU for y = 10*x + 5\n");
    printf("N=%d, ThreadsPerBlock=%d\n", N, TPB);

    // -----------------------------
    // 2) Allocate and initialize host data
    // -----------------------------
    // We use std::vector for automatic memory management on the host.
    std::vector<float> h_x(N), h_y_cpu(N), h_y_gpu(N, 0.0f);

    // Simple, predictable pattern helps sanity-check results.
    // You can replace this with random numbers if you like.
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i); // 0,1,2,3,...
    }

    // -----------------------------
    // 3) Compute CPU reference
    // -----------------------------
    scale10x5_cpu(h_x.data(), h_y_cpu.data(), N);

    // -----------------------------
    // 4) Allocate device memory
    // -----------------------------
    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));

    // -----------------------------
    // 5) Copy inputs to device
    // -----------------------------
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // -----------------------------
    // 6) Launch GPU kernel
    // -----------------------------
    // We round up the number of blocks so that blocks*TPB >= N.
    int blocks = (N + TPB - 1) / TPB;
    printf("Launching grid: blocks=%d, threadsPerBlock=%d (total threads ~ %d)\n",
           blocks, TPB, blocks * TPB);

    scale10x5_gpu<<<blocks, TPB>>>(d_x, d_y, N);

    // Always check for launch errors and synchronize before reading results.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------
    // 7) Copy results back to host
    // -----------------------------
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // -----------------------------
    // 8) Compare CPU vs GPU results
    // -----------------------------
    bool pass = compare_results(h_y_cpu.data(), h_y_gpu.data(), N, /*tol=*/1e-5f, /*max_report=*/10);
    if (pass) {
        printf("PASS: CPU and GPU results match within tolerance.\n");
    } else {
        printf("FAIL: CPU and GPU results differ.\n");
    }

    // (Optional) Print a few sample values to build intuition.
    // Comment out if you prefer quiet output for large N.
    int to_show = (N < 10) ? N : 10;
    printf("\nFirst %d elements (CPU vs GPU):\n", to_show);
    for (int i = 0; i < to_show; ++i) {
        printf("i=%d  CPU=%8.3f  GPU=%8.3f  expect=%8.3f\n",
               i, h_y_cpu[i], h_y_gpu[i], 10.0f * h_x[i] + 5.0f);
    }

    // -----------------------------
    // 9) Cleanup
    // -----------------------------
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
