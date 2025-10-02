/**
 * @file    vector_add_workflow.cu
 * @brief   Chapter 3 — Vector Addition organized by the Five-Step CUDA Workflow.
 *
 * Goal
 * ----
 * Show the complete GPU vector-add program *explicitly structured* by the
 * canonical five steps:
 *   (1) Allocate device memory
 *   (2) Copy inputs Host → Device
 *   (3) Compute (launch kernel)
 *   (4) Copy results Device → Host
 *   (5) Free device memory
 *
 * What’s different from vector_add_minimal.cu?
 * --------------------------------------------
 * Same computation, but the program is laid out to mirror the workflow as a
 * reusable mental template. This makes it easier to reason about where the data
 * lives at each moment and simplifies debugging.
 *
 * Intentionally *not* included here:
 *  - Error-checking macro wrappers (introduced in vector_add_with_checks.cu)
 *  - Detailed verification against a CPU reference (introduced later)
 *
 * Build
 * -----
 *   nvcc vector_add_workflow.cu -o vector_add_workflow
 *
 * Run (default N = 1,048,576)
 * ---------------------------
 *   ./vector_add_workflow
 *
 * Run with custom size (e.g., 10 million)
 * ---------------------------------------
 *   ./vector_add_workflow 10000000
 */

#include <cstdio>           // printf, fprintf
#include <cstdlib>          // atoi/atoll, malloc/free
#include <cuda_runtime.h>   // CUDA runtime

// ----------------------------------------------------------------------------------
// Kernel: one thread per element, with bounds check.
// Computes: C[i] = A[i] + B[i]
// ----------------------------------------------------------------------------------
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ----------------------------------------------------------------------------------
// Pretty-print a few elements to sanity-check data flow (not a formal verification).
// ----------------------------------------------------------------------------------
static void printSample(const float* A, const float* B, const float* C, int N, int count = 5)
{
    int show = (N < count) ? N : count;
    printf("Sample (i : A + B -> C)\n");
    for (int i = 0; i < show; ++i) {
        printf("  %d : %.1f + %.1f -> %.1f\n", i, A[i], B[i], C[i]);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    // ------------------------------------------------------------------------------
    // Problem size
    // ------------------------------------------------------------------------------
    int N = (argc > 1) ? static_cast<int>(std::atoll(argv[1])) : (1 << 20); // ~1M by default
    if (N <= 0) {
        fprintf(stderr, "N must be positive.\n");
        return 1;
    }
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    printf("=== Vector Add — Five-Step CUDA Workflow ===\n");
    printf("N elements : %d\n", N);
    printf("Bytes/vec  : %zu (%.2f MiB)\n\n", bytes, bytes / (1024.0 * 1024.0));

    // ------------------------------------------------------------------------------
    // Host allocations and initialization (lives on the CPU)
    // ------------------------------------------------------------------------------
    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed.\n");
        std::free(h_A); std::free(h_B); std::free(h_C);
        return 1;
    }

    // Initialize inputs with a simple, deterministic pattern:
    // A[i] = i, B[i] = 2*i  → expected C[i] = 3*i
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = 2.0f * static_cast<float>(i);
    }

    // ------------------------------------------------------------------------------
    // (1) Allocate memory on the device (GPU)
    // ------------------------------------------------------------------------------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaError_t err;
    err = cudaMalloc(reinterpret_cast<void**>(&d_A), bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); goto CLEANUP_HOST; }

    err = cudaMalloc(reinterpret_cast<void**>(&d_B), bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); goto CLEANUP_DA; }

    err = cudaMalloc(reinterpret_cast<void**>(&d_C), bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); goto CLEANUP_DB; }

    // ------------------------------------------------------------------------------
    // (2) Copy inputs from host to device
    // ------------------------------------------------------------------------------
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D A failed: %s\n", cudaGetErrorString(err)); goto CLEANUP_DC; }

    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D B failed: %s\n", cudaGetErrorString(err)); goto CLEANUP_DC; }

    // ------------------------------------------------------------------------------
    // (3) Compute on the device — launch the kernel
    //     Choose a reasonable configuration: 256 threads per block, enough blocks for N.
    // ------------------------------------------------------------------------------
    {
        int blockSize = 256;
        int gridSize  = (N + blockSize - 1) / blockSize;

        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        // For clarity of the workflow, we synchronize here to ensure the compute step completes
        // before proceeding to copy-back. Error-checking is introduced in the next program.
        cudaDeviceSynchronize();
    }

    // ------------------------------------------------------------------------------
    // (4) Copy results from device back to host
    // ------------------------------------------------------------------------------
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H C failed: %s\n", cudaGetErrorString(err)); goto CLEANUP_DC; }

    // Light sanity check (not a formal verification)
    printSample(h_A, h_B, h_C, N);

    // ------------------------------------------------------------------------------
    // (5) Free device memory
    // ------------------------------------------------------------------------------
CLEANUP_DC:
    cudaFree(d_C);
CLEANUP_DB:
    cudaFree(d_B);
CLEANUP_DA:
    cudaFree(d_A);

CLEANUP_HOST:
    // Free host memory
    std::free(h_C);
    std::free(h_B);
    std::free(h_A);

    // Report a final status (best-effort; if any cuda* call printed an error, exit code is non-zero)
    if (err != cudaSuccess) {
        fprintf(stderr, "Workflow completed with CUDA errors.\n");
        return 1;
    }
    printf("Workflow completed successfully.\n");
    return 0;
}
