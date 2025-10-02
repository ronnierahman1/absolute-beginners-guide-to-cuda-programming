/**
 * @file    memcpy_roundtrip.cu
 * @brief   Chapter 3 (Talk to the GPU) — Host ↔ Device copy round-trip sanity test.
 *
 * Purpose
 * -------
 * This standalone program validates that memory transfers between the CPU (host) and
 * the GPU (device) work as expected. It performs a *round-trip copy*:
 *
 *   Host (h_src) --H2D--> Device (d_buf) --D2H--> Host (h_dst)
 *
 * and then verifies that h_dst is byte-for-byte identical to h_src.
 *
 * Why this matters
 * ----------------
 * Before launching any kernels, you should confirm that your environment can:
 *   1) Allocate device memory (cudaMalloc),
 *   2) Copy host → device (cudaMemcpyHostToDevice),
 *   3) Copy device → host (cudaMemcpyDeviceToHost),
 *   4) Free device memory (cudaFree),
 *   5) Detect and report errors early (CUDA_CHECK).
 *
 * What this program does (no kernels yet)
 * ---------------------------------------
 *  - Creates a host buffer h_src and fills it with a recognizable pattern.
 *  - Allocates a device buffer d_buf of the same size.
 *  - Copies h_src → d_buf (H2D).
 *  - Copies d_buf → h_dst (D2H).
 *  - Compares h_src and h_dst for exact equality.
 *
 * Build
 * -----
 *   nvcc memcpy_roundtrip.cu -o memcpy_roundtrip
 *
 * Run (default: 1,048,576 float elements)
 * ---------------------------------------
 *   ./memcpy_roundtrip
 *
 * Run with a custom element count (e.g., 50 million floats)
 * ---------------------------------------------------------
 *   ./memcpy_roundtrip 50000000
 *
 * Notes
 * -----
 *  - We use float elements for consistency with later chapters. A memcpy of floats is
 *    byte-exact, so verification uses a byte-wise memcmp for clarity.
 *  - We print device info and free/total memory before/after to make behavior visible.
 *  - Still no kernels: this is strictly a transfer/verification smoke test.
 */

#include <cstdio>                 // printf, fprintf
#include <cstdlib>                // atoi/atoll, EXIT_SUCCESS/FAILURE
#include <cstring>                // std::memcmp
#include <cuda_runtime.h>         // CUDA runtime APIs
#include <device_launch_parameters.h> // optional; helpful on some toolchains

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
    const double toMiB = 1.0 / (1024.0 * 1024.0);
    printf("%s\n", label);
    printf("  Device mem free : %.2f MiB\n", free_b * toMiB);
    printf("  Device mem total: %.2f MiB\n\n", total_b * toMiB);
}

// -----------------------------------------------------------------------------
// Initialize a recognizable pattern in the host source buffer.
//
// Rationale:
//  We want data that is easy to spot if you print a few values, but also stable
//  across runs so memcmp works. We avoid NaNs/Infs and keep values deterministic.
// -----------------------------------------------------------------------------
static void initPattern(float* h, size_t N)
{
    // Simple linear pattern with a mild non-integer increment to avoid trivial integers.
    // This guarantees deterministic bytes for memcmp and is easy to glance-check.
    for (size_t i = 0; i < N; ++i) {
        h[i] = 0.5f * static_cast<float>(i) + 3.25f; // e.g., 3.25, 3.75, 4.25, ...
    }
}

// -----------------------------------------------------------------------------
// Verify byte-exact equality between two host buffers of floats.
//
// We compare raw bytes using memcmp because a memcpy round-trip should be exact.
// (If you were *computing* floats, a tolerance-based compare would be appropriate.)
// -----------------------------------------------------------------------------
static bool verifyByteEqual(const float* a, const float* b, size_t N)
{
    const size_t bytes = N * sizeof(float);
    return std::memcmp(a, b, bytes) == 0;
}

// -----------------------------------------------------------------------------
// Pretty-printer for a few sample elements (useful if a mismatch occurs).
// -----------------------------------------------------------------------------
static void printSample(const float* h_src, const float* h_dst, size_t N, size_t count = 5)
{
    const size_t show = (N < count ? N : count);
    printf("Sample (first %zu elements):\n", show);
    for (size_t i = 0; i < show; ++i) {
        printf("  i=%zu  src=%g  dst=%g\n", i, h_src[i], h_dst[i]);
    }
    printf("\n");
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // ------------------------------
    // 1) Choose element count (N)
    // ------------------------------
    // Default: ~1 million floats; override with argv[1].
    size_t N = (argc > 1) ? static_cast<size_t>(std::atoll(argv[1])) : (1ULL << 20);
    if (N == 0) {
        fprintf(stderr, "Element count N must be positive.\n");
        return EXIT_FAILURE;
    }

    const size_t bytes = N * sizeof(float);

    printf("=== Host ↔ Device Round-Trip Copy Test ===\n");
    printf("Element count (N): %zu\n", N);
    printf("Bytes total      : %zu (%.2f MiB)\n\n", bytes, bytes / (1024.0 * 1024.0));

    // ------------------------------
    // 2) Device info (optional)
    // ------------------------------
    printDeviceInfo();

    // ------------------------------
    // 3) Allocate host buffers
    // ------------------------------
    float* h_src = static_cast<float*>(std::malloc(bytes));
    float* h_dst = static_cast<float*>(std::malloc(bytes));
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host malloc failed (requested %zu bytes per buffer).\n", bytes);
        std::free(h_src);
        std::free(h_dst);
        return EXIT_FAILURE;
    }

    // Initialize source; clear destination to sentinel values for confidence
    initPattern(h_src, N);
    std::memset(h_dst, 0xCD, bytes); // 0xCD is a common debug fill pattern

    // ------------------------------
    // 4) Report device memory before allocation
    // ------------------------------
    printMemInfo("[Before device allocation]");

    // ------------------------------
    // 5) Allocate device buffer
    // ------------------------------
    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buf), bytes));
    printf("Allocated device buffer (d_buf): %.2f MiB\n\n", bytes / (1024.0 * 1024.0));

    // ------------------------------
    // 6) Report device memory after allocation
    // ------------------------------
    printMemInfo("[After device allocation]");

    // ------------------------------
    // 7) Host → Device copy (H2D)
    // ------------------------------
    // This mirrors the math d_A[i] = A[i], ∀i ∈ [0, N-1].
    CUDA_CHECK(cudaMemcpy(d_buf, h_src, bytes, cudaMemcpyHostToDevice));
    printf("Copied Host → Device: %zu bytes\n", bytes);

    // ------------------------------
    // 8) Device → Host copy (D2H)
    // ------------------------------
    // This mirrors the math h_dst[i] = d_buf[i], ∀i ∈ [0, N-1].
    CUDA_CHECK(cudaMemcpy(h_dst, d_buf, bytes, cudaMemcpyDeviceToHost));
    printf("Copied Device → Host: %zu bytes\n\n", bytes);

    // ------------------------------
    // 9) Verify round-trip equality
    // ------------------------------
    const bool ok = verifyByteEqual(h_src, h_dst, N);
    if (ok) {
        printf("Round-trip verification: PASS (byte-exact match)\n\n");
    } else {
        printf("Round-trip verification: FAIL (mismatch detected)\n\n");
        printSample(h_src, h_dst, N);

        // Optional: find first mismatch to help learners debug
        const unsigned char* a = reinterpret_cast<const unsigned char*>(h_src);
        const unsigned char* b = reinterpret_cast<const unsigned char*>(h_dst);
        for (size_t i = 0; i < bytes; ++i) {
            if (a[i] != b[i]) {
                size_t elem = i / sizeof(float);
                printf("First mismatching byte at byte %zu (element %zu)\n", i, elem);
                break;
            }
        }
    }

    // ------------------------------
    // 10) Free device + host memory
    // ------------------------------
    CUDA_CHECK(cudaFree(d_buf));
    printMemInfo("[After device free]");

    std::free(h_src);
    std::free(h_dst);

    printf("Round-trip copy test completed.\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
