/**
 * @file    vector_add_minimal.cu
 * @brief   Chapter 3 (Talk to the GPU) — Minimal end-to-end vector addition on the GPU.
 *
 * What this program shows
 * -----------------------
 * The smallest, self-contained example of running real computation on the GPU:
 *   1) Allocate host and device buffers
 *   2) Copy inputs Host → Device
 *   3) Launch a simple CUDA kernel: C[i] = A[i] + B[i]
 *   4) Copy result Device → Host
 *   5) Verify a few values and clean up
 *
 * Design goal
 * -----------
 * Keep it minimal: no error-checking macros, no timers, no abstractions.
 * This lets you focus on the essential CUDA mechanics and kernel launch syntax.
 *
 * Build
 * -----
 *   nvcc vector_add_minimal.cu -o vector_add_minimal
 *
 * Run (default N = 1,048,576 elements)
 * ------------------------------------
 *   ./vector_add_minimal
 *
 * Run with a custom size (e.g., 10 million elements)
 * --------------------------------------------------
 *   ./vector_add_minimal 10000000
 */

#include <cstdio>           // printf, fprintf
#include <cstdlib>          // atoi/atoll, malloc/free
#include <cuda_runtime.h>   // CUDA runtime API

// ============================================================================
// 1) GPU Kernel
//    Each thread computes one output element: C[i] = A[i] + B[i].
//    The bounds check protects against threads with i >= N (when grid*block > N).
// ============================================================================
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global index across the whole grid
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// 2) Minimal driver code (host side)
// ============================================================================
int main(int argc, char** argv)
{
    // ------------------------------------------------------------
    // Problem size (N elements). Default ~1M; allow override via CLI.
    // ------------------------------------------------------------
    int N = (argc > 1) ? static_cast<int>(std::atoll(argv[1])) : (1 << 20);
    if (N <= 0) {
        fprintf(stderr, "N must be positive.\n");
        return 1;
    }
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    printf("=== Minimal Vector Add (GPU) ===\n");
    printf("N elements : %d\n", N);
    printf("Bytes each : %zu (%.2f MiB)\n\n", bytes, bytes / (1024.0 * 1024.0));

    // ------------------------------------------------------------
    // Host allocations (ordinary CPU memory)
    // ------------------------------------------------------------
    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed.\n");
        std::free(h_A); std::free(h_B); std::free(h_C);
        return 1;
    }

    // Initialize inputs with a simple pattern:
    // A[i] = i, B[i] = 2*i  ⇒  Expected C[i] = 3*i (exactly representable for these ranges)
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = 2.0f * static_cast<float>(i);
    }

    // ------------------------------------------------------------
    // Device allocations (GPU memory)
    // ------------------------------------------------------------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_A), bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc(reinterpret_cast<void**>(&d_B), bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); cudaFree(d_A); return 1; }

    err = cudaMalloc(reinterpret_cast<void**>(&d_C), bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); cudaFree(d_A); cudaFree(d_B); return 1; }

    // ------------------------------------------------------------
    // Host → Device copies
    // ------------------------------------------------------------
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D A failed: %s\n", cudaGetErrorString(err)); }

    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D B failed: %s\n", cudaGetErrorString(err)); }

    // ------------------------------------------------------------
    // Kernel launch configuration
    //   - blockSize: threads per block (common starting point: 256)
    //   - gridSize : number of blocks to cover N elements (round up)
    // ------------------------------------------------------------
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    // ------------------------------------------------------------
    // Launch the kernel on the GPU
    // ------------------------------------------------------------
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // (Minimalism note) In later versions we’ll check for launch/runtime errors
    // with cudaGetLastError() and cudaDeviceSynchronize(). Here we keep it simple.
    cudaDeviceSynchronize();  // ensure kernel completed before copying back

    // ------------------------------------------------------------
    // Device → Host copy (retrieve the result)
    // ------------------------------------------------------------
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H C failed: %s\n", cudaGetErrorString(err)); }

    // ------------------------------------------------------------
    // Quick verification (spot-check a few values and a simple full pass)
    // ------------------------------------------------------------
    bool ok = true;

    // Spot-check first 5 elements
    printf("Sample results (i : A + B -> C):\n");
    for (int i = 0; i < ((N < 5) ? N : 5); ++i) {
        printf("  %d : %.1f + %.1f -> %.1f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Full check: C[i] should equal 3*i exactly with our initialization
    for (int i = 0; i < N; ++i) {
        float expected = 3.0f * static_cast<float>(i);
        if (h_C[i] != expected) {
            fprintf(stderr, "Mismatch at i=%d: got %.1f, expected %.1f\n", i, h_C[i], expected);
            ok = false;
            break;
        }
    }

    printf("\nVerification: %s\n", ok ? "PASS" : "FAIL");

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);

    std::free(h_C);
    std::free(h_B);
    std::free(h_A);

    return ok ? 0 : 1;
}
