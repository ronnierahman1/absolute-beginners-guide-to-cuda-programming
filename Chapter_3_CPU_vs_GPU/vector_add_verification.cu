/**
 * @file    vector_add_verification.cu
 * @brief   Chapter 3 — Final program: GPU vector addition with CPU reference check.
 *
 * Overview
 * --------
 * This is the *teaching-quality* end-to-end version of the Chapter 3 example.
 * It performs vector addition on the GPU:
 *
 *     C[i] = A[i] + B[i]    for i = 0..N-1
 *
 * and then verifies the GPU result against a CPU "ground truth" implementation.
 * We allow a tiny floating-point tolerance (default 1e-5) to account for harmless
 * rounding differences between CPU and GPU arithmetic.
 *
 * What this program demonstrates
 * ------------------------------
 *  1) The Five-Step CUDA Workflow:
 *       (a) allocate device memory
 *       (b) copy inputs host→device
 *       (c) launch the kernel (compute on GPU)
 *       (d) copy results device→host
 *       (e) free device memory
 *  2) Robust error checking for every CUDA runtime call (CUDA_CHECK macro)
 *  3) Immediate surfacing of kernel errors (cudaGetLastError + cudaDeviceSynchronize)
 *  4) CPU reference computation and elementwise verification with a tolerance
 *
 * Build
 * -----
 *   nvcc vector_add_verification.cu -o vector_add_verification
 *
 * Run (defaults: N = 1,048,576 elements, tol = 1e-5)
 * ---------------------------------------------------
 *   ./vector_add_verification
 *
 * Override N (e.g., 10 million) and tolerance (e.g., 1e-6)
 * --------------------------------------------------------
 *   ./vector_add_verification 10000000 1e-6
 *
 * Expected Output (example)
 * -------------------------
 *   === Vector Add — GPU vs CPU Verification ===
 *   N elements  : 1048576
 *   Tolerance   : 1.0e-05
 *   Sample (i : A + B -> C_GPU vs C_CPU)
 *     0 : 0 + 0 -> 0  | 0
 *     1 : 1 + 2 -> 3  | 3
 *     2 : 2 + 4 -> 6  | 6
 *     3 : 3 + 6 -> 9  | 9
 *     4 : 4 + 8 -> 12 | 12
 *   Verification: PASS
 *   Done.
 */

#include <cstdio>                // printf, fprintf
#include <cstdlib>               // atoll, strtod, malloc/free, EXIT_*
#include <cmath>                 // std::fabs
#include <cuda_runtime.h>        // CUDA runtime API
#include <device_launch_parameters.h> // optional on some toolchains

// ============================================================================
// Error checking macro
//   - Wrap every CUDA API call
//   - Print filename, line number, and human-readable error string
//   - Exit immediately on failure ("fail fast")
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
// GPU kernel: one thread per element
// Computes C[i] = A[i] + B[i] with a bounds check
// ============================================================================
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global linear index
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// CPU reference implementation (ground truth)
// ============================================================================
static void vectorAddCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// Elementwise verification with tolerance
//   Returns true if |C_gpu[i] - C_cpu[i]| <= tol for all i
// ============================================================================
static bool verifyWithTolerance(const float* gpu, const float* cpu, int N, float tol)
{
    for (int i = 0; i < N; ++i) {
        // fabsf is fine; std::fabs also works (we included <cmath>)
        if (std::fabs(gpu[i] - cpu[i]) > tol) {
            fprintf(stderr,
                    "Mismatch at index %d: GPU=%0.7f, CPU=%0.7f (|diff|=%0.7f > tol=%0.7f)\n",
                    i, gpu[i], cpu[i], std::fabs(gpu[i] - cpu[i]), tol);
            return false;
        }
    }
    return true;
}

// ============================================================================
// Pretty-print a small sample: show A[i], B[i], C_gpu[i], and C_cpu[i]
// ============================================================================
static void printSample(const float* A, const float* B,
                        const float* Cgpu, const float* Ccpu,
                        int N, int count = 5)
{
    int show = (N < count) ? N : count;
    printf("Sample (i : A + B -> C_GPU | C_CPU)\n");
    for (int i = 0; i < show; ++i) {
        // Using %.0f because the chosen initialization yields integer-valued floats
        printf("  %d : %.0f + %.0f -> %.0f | %.0f\n", i, A[i], B[i], Cgpu[i], Ccpu[i]);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    // ------------------------------------------------------------------------
    // Parse inputs
    //   argv[1] : N (element count), default ~1M
    //   argv[2] : tolerance, default 1e-5
    // ------------------------------------------------------------------------
    int   N   = (argc > 1) ? static_cast<int>(std::atoll(argv[1])) : (1 << 20);
    float tol = (argc > 2) ? static_cast<float>(strtod(argv[2], nullptr)) : 1e-5f;

    if (N <= 0) {
        fprintf(stderr, "N must be positive.\n");
        return EXIT_FAILURE;
    }
    if (!(tol > 0.0f)) {
        fprintf(stderr, "Tolerance must be positive.\n");
        return EXIT_FAILURE;
    }

    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    printf("=== Vector Add — GPU vs CPU Verification ===\n");
    printf("N elements  : %d\n", N);
    printf("Tolerance   : %.1e\n\n", tol);

    // ------------------------------------------------------------------------
    // Host allocations
    // ------------------------------------------------------------------------
    float* h_A     = static_cast<float*>(std::malloc(bytes));
    float* h_B     = static_cast<float*>(std::malloc(bytes));
    float* h_Cgpu  = static_cast<float*>(std::malloc(bytes));
    float* h_Ccpu  = static_cast<float*>(std::malloc(bytes));

    if (!h_A || !h_B || !h_Cgpu || !h_Ccpu) {
        fprintf(stderr, "Host malloc failed (requested %zu bytes per vector).\n", bytes);
        std::free(h_A); std::free(h_B); std::free(h_Cgpu); std::free(h_Ccpu);
        return EXIT_FAILURE;
    }

    // Initialize inputs with a simple, deterministic pattern:
    // A[i] = i, B[i] = 2*i  →  expected C[i] = 3*i (exact in IEEE float for these magnitudes)
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
    //   - 256 threads per block (common baseline)
    //   - Enough blocks to cover N elements (round up)
    // ------------------------------------------------------------------------
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Catch launch configuration errors immediately (invalid args, etc.)
    CUDA_CHECK(cudaGetLastError());

    // Synchronize so that any runtime errors (e.g., illegal memory access) surface here
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------------
    // Device → Host copy of the GPU result
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_Cgpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // CPU reference + verification
    // ------------------------------------------------------------------------
    vectorAddCPU(h_A, h_B, h_Ccpu, N);

    printSample(h_A, h_B, h_Cgpu, h_Ccpu, N);

    bool pass = verifyWithTolerance(h_Cgpu, h_Ccpu, N, tol);
    printf("Verification: %s\n", pass ? "PASS" : "FAIL");

    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    std::free(h_Ccpu);
    std::free(h_Cgpu);
    std::free(h_B);
    std::free(h_A);

    printf("Done.\n");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
