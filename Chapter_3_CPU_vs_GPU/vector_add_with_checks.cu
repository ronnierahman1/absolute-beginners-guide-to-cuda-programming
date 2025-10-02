/**
 * @file    vector_add_with_checks.cu
 * @brief   Chapter 3 — Vector addition with robust CUDA error checking.
 *
 * Purpose
 * -------
 * This program is the *checked* version of vector addition. It performs the same
 * computation as earlier examples (C[i] = A[i] + B[i]) but now wraps every CUDA call
 * with a safety net:
 *
 *   - CUDA_CHECK(...) macro on all runtime API calls
 *   - cudaGetLastError() right after the kernel launch
 *   - cudaDeviceSynchronize() to surface runtime errors immediately
 *
 * Why this matters
 * ----------------
 * GPU bugs can be subtle. Launch failures, invalid pointers, and illegal memory
 * accesses may not crash immediately and can corrupt results later. By *failing fast*
 * with clear messages (filename, line number, error string), you save hours of
 * debugging and know exactly where a problem occurred.
 *
 * What this program does
 * ----------------------
 *  1) Allocate host arrays A, B, C and initialize A,B with a simple pattern
 *  2) Allocate device arrays d_A, d_B, d_C
 *  3) Copy A,B (host → device)
 *  4) Launch kernel C = A + B on the GPU (with error checks)
 *  5) Copy C back (device → host)
 *  6) Sanity-check results and clean up
 *
 * Build
 * -----
 *   nvcc vector_add_with_checks.cu -o vector_add_with_checks
 *
 * Run (default N = 1,048,576)
 * ---------------------------
 *   ./vector_add_with_checks
 *
 * Run with custom size (e.g., 10 million)
 * ---------------------------------------
 *   ./vector_add_with_checks 10000000
 */

#include <cstdio>               // printf, fprintf
#include <cstdlib>              // atoi/atoll, malloc/free, EXIT_*
#include <cuda_runtime.h>       // CUDA runtime API
#include <device_launch_parameters.h> // optional on some toolchains

// ============================================================================
// Error checking macro: wrap every CUDA API call to catch failures early.
// Prints filename, line number, and a human-readable error string.
// ============================================================================
#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(err__));                 \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

// ============================================================================
// Kernel: one thread per element; computes C[i] = A[i] + B[i] with bounds check.
// ============================================================================
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global linear index
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ----------------------------------------------------------------------------
// Utility: print a few sample values to confirm the data flow looks reasonable.
// ----------------------------------------------------------------------------
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
    // ------------------------------------------------------------------------
    // Problem size and memory footprint
    // ------------------------------------------------------------------------
    int N = (argc > 1) ? static_cast<int>(std::atoll(argv[1])) : (1 << 20); // default ~1M
    if (N <= 0) {
        fprintf(stderr, "N must be positive.\n");
        return EXIT_FAILURE;
    }
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    printf("=== Vector Add (with error checks) ===\n");
    printf("N elements : %d\n", N);
    printf("Bytes/vec  : %zu (%.2f MiB)\n\n", bytes, bytes / (1024.0 * 1024.0));

    // ------------------------------------------------------------------------
    // Host allocations + initialization
    //   A[i] = i, B[i] = 2*i  → expected C[i] = 3*i (exact for these values)
    // ------------------------------------------------------------------------
    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed for %zu bytes per vector.\n", bytes);
        std::free(h_A); std::free(h_B); std::free(h_C);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = 2.0f * static_cast<float>(i);
    }

    // ------------------------------------------------------------------------
    // Device allocations
    // ------------------------------------------------------------------------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes));

    // ------------------------------------------------------------------------
    // Host → Device copies
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------------
    // Kernel configuration and launch
    //   - Use 256 threads per block (a common baseline)
    //   - Compute number of blocks needed to cover N
    // ------------------------------------------------------------------------
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Immediately check for launch errors (e.g., invalid config, invalid pointers)
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to surface any runtime errors (e.g., illegal memory access)
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------------
    // Device → Host copy (retrieve results)
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // Light sanity check (not a full CPU reference yet—introduced next program)
    // We know C[i] should be exactly 3*i for our chosen initialization.
    // ------------------------------------------------------------------------
    printSample(h_A, h_B, h_C, N);
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = 3.0f * static_cast<float>(i);
        if (h_C[i] != expected) {
            fprintf(stderr, "Mismatch at i=%d: got %.1f, expected %.1f\n", i, h_C[i], expected);
            ok = false;
            break;
        }
    }
    printf("Verification (pattern 3*i): %s\n", ok ? "PASS" : "FAIL");

    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));
    std::free(h_C);
    std::free(h_B);
    std::free(h_A);

    printf("Done.\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
