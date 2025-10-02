/**
 * @file    device_alloc_free.cu
 * @brief   Chapter 3 (Talk to the GPU) — Smoke test for device memory management.
 *
 * This standalone program demonstrates ONLY the allocation and release of GPU memory.
 * It does not copy data or launch kernels. The goal is to validate that your CUDA
 * toolchain can successfully:
 *   1) Initialize a CUDA context,
 *   2) Allocate device buffers with cudaMalloc,
 *   3) Free device buffers with cudaFree,
 *   4) Report meaningful errors if anything goes wrong.
 *
 * --------------------------------------------
 * Build
 *   nvcc device_alloc_free.cu -o device_alloc_free
 *
 * Run (default: 1,048,576 float elements per buffer)
 *   ./device_alloc_free
 *
 * Run with a custom size (e.g., 50 million elements per buffer)
 *   ./device_alloc_free 50000000
 *
 * Notes
 * - We allocate three buffers (A, B, C) to mirror the later vector-add example.
 * - We report free/total memory before allocation, after allocation, and after free,
 *   so you can see how much memory is being used.
 * - No kernels are launched; this is purely a memory smoke test.
 */

#include <cstdio>              // printf, fprintf
#include <cstdlib>             // atoi, EXIT_SUCCESS/FAILURE
#include <cuda_runtime.h>      // cudaMalloc, cudaFree, cudaMemGetInfo, cudaGetDevice, etc.
#include <device_launch_parameters.h>  // optional; useful on some toolchains

// -----------------------------------------------------------------------------
// Error checking macro: wrap every CUDA API call to catch failures early.
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(err__));                \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

// -----------------------------------------------------------------------------
// Utility: print basic information about the active CUDA device.
// -----------------------------------------------------------------------------
static void printDeviceInfo()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("Active CUDA device: %d\n", dev);
    printf("  Name                  : %s\n", prop.name);
    printf("  Compute capability    : %d.%d\n", prop.major, prop.minor);
    printf("  MultiProcessor count  : %d\n", prop.multiProcessorCount);
    printf("  Global memory (bytes) : %zu\n", static_cast<size_t>(prop.totalGlobalMem));
    printf("\n");
}

// -----------------------------------------------------------------------------
// Utility: print current free/total device memory.
// -----------------------------------------------------------------------------
static void printMemInfo(const char* label)
{
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    // Convert to MiB for readability.
    const double toMiB = 1.0 / (1024.0 * 1024.0);
    printf("%s\n", label);
    printf("  Device mem free : %.2f MiB\n", free_b * toMiB);
    printf("  Device mem total: %.2f MiB\n\n", total_b * toMiB);
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // ------------------------------
    // 1) Choose element count (N)
    // ------------------------------
    // Default to ~1 million float elements per buffer; change via argv[1].
    // We use floats here to match later chapters, but the type is arbitrary for this test.
    size_t N = (argc > 1) ? static_cast<size_t>(std::atoll(argv[1])) : (1ULL << 20);
    if (N == 0) {
        fprintf(stderr, "Element count N must be positive.\n");
        return EXIT_FAILURE;
    }

    // Bytes per buffer (A, B, C each allocate this many bytes on the device).
    const size_t bytes = N * sizeof(float);

    printf("=== Device Allocation/Free Smoke Test ===\n");
    printf("Requested buffers: A, B, C\n");
    printf("Element count (N): %zu\n", N);
    printf("Bytes per buffer : %zu (%.2f MiB)\n\n", bytes, bytes / (1024.0 * 1024.0));

    // ------------------------------
    // 2) Print device info (optional)
    // ------------------------------
    printDeviceInfo();

    // ------------------------------
    // 3) Report memory before allocation
    // ------------------------------
    printMemInfo("[Before allocation]");

    // ------------------------------
    // 4) Allocate device buffers
    // ------------------------------
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    // Each allocation is wrapped in CUDA_CHECK to fail fast with a clear message.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes));

    printf("Allocated three device buffers (A, B, C), each %.2f MiB.\n\n", bytes / (1024.0 * 1024.0));

    // ------------------------------
    // 5) Report memory after allocation
    // ------------------------------
    printMemInfo("[After allocation]");

    // (Optional) You could touch the memory with cudaMemset here to validate accessibility,
    // but this example intentionally avoids any writes/reads to keep it "allocation only."
    // Example (commented out):
    // CUDA_CHECK(cudaMemset(d_A, 0, bytes));
    // CUDA_CHECK(cudaMemset(d_B, 0, bytes));
    // CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    // ------------------------------
    // 6) Free device buffers
    // ------------------------------
    // Free in the reverse order of allocation (not required, but a common habit).
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    printf("Freed device buffers A, B, C.\n\n");

    // ------------------------------
    // 7) Report memory after free
    // ------------------------------
    printMemInfo("[After free]");

    printf("Smoke test completed successfully.\n");
    return EXIT_SUCCESS;
}
