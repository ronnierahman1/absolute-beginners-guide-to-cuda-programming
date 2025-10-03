
// chapter6_demo.cu
//
// Demonstrates Chapter 6 CUDA concepts:
//   - Shared memory as a team scratchpad
//   - Block-level reduction (min)
//   - Block-level argmin (value + index)
//   - Correct use of __syncthreads() for synchronization
//   - Writing partial results to global memory
//   - Combining results on CPU and verification

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <limits>
#include <cfloat>
#include <cuda_runtime.h>


/**
 * @brief Macro to check CUDA API calls for errors and exit on failure.
 */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",               \
                         cudaGetErrorString(err__), err__, __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)


/**
 * @brief CUDA kernel: Computes the minimum value in each block using shared memory.
 *
 * @param in         Input array of floats (device pointer)
 * @param block_mins Output array (device pointer), one min per block
 * @param n          Number of elements in input array
 *
 * Each block finds the minimum of its assigned segment and writes it to block_mins.
 * Uses parallel reduction in shared memory.
 */
__global__ void blockMinKernel(const float* __restrict__ in,
                               float* __restrict__ block_mins,
                               int n)
{
    extern __shared__ unsigned char smem[]; // Dynamic shared memory
    float* sVal = reinterpret_cast<float*>(smem); // Shared memory for values

    const int tid  = threadIdx.x; // Thread index within block
    const int gid  = blockIdx.x * blockDim.x + tid; // Global index

    // Load each thread's value into shared memory (scratchpad)
    float v = (gid < n) ? in[gid] : FLT_MAX; // Use FLT_MAX for out-of-bounds
    sVal[tid] = v;

    __syncthreads(); // Ensure all threads have written to shared memory

    // Parallel reduction (min): halve stride each iteration
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = sVal[tid + stride];
            if (other < sVal[tid]) sVal[tid] = other;
        }
        __syncthreads(); // Synchronize after each reduction step
    }

    // Reporter thread writes the block result to global memory
    if (tid == 0) {
        block_mins[blockIdx.x] = sVal[0];
    }
}


/**
 * @brief CUDA kernel: Computes the minimum value and its global index in each block.
 *
 * @param in              Input array of floats (device pointer)
 * @param block_min_vals  Output array (device pointer), min value per block
 * @param block_min_idxs  Output array (device pointer), min index per block
 * @param n               Number of elements in input array
 *
 * Each block finds the minimum value and its global index using shared memory.
 * Shared memory layout: [blockDim.x floats | blockDim.x ints]
 */
__global__ void blockArgMinKernel(const float* __restrict__ in,
                                  float* __restrict__ block_min_vals,
                                  int*   __restrict__ block_min_idxs,
                                  int n)
{
    extern __shared__ unsigned char smem[]; // Dynamic shared memory
    float* sVal = reinterpret_cast<float*>(smem); // Shared memory for values
    int*   sIdx = reinterpret_cast<int*>(sVal + blockDim.x); // Shared memory for indices

    const int tid  = threadIdx.x; // Thread index within block
    const int gid  = blockIdx.x * blockDim.x + tid; // Global index

    // Each thread contributes (value, global_index) or neutral if out-of-bounds
    if (gid < n) {
        sVal[tid] = in[gid];
        sIdx[tid] = gid;
    } else {
        sVal[tid] = FLT_MAX; // Neutral for min
        sIdx[tid] = -1;
    }

    __syncthreads(); // Ensure all threads have written to shared memory

    // Reduction that carries both value and index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float vOther = sVal[tid + stride];
            float vMine  = sVal[tid];
            if (vOther < vMine) {
                sVal[tid] = vOther;
                sIdx[tid] = sIdx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Reporter thread writes the block result to global memory
    if (tid == 0) {
        block_min_vals[blockIdx.x] = sVal[0];
        block_min_idxs[blockIdx.x] = sIdx[0];
    }
}


/**
 * @brief CPU helper: Finds the minimum value in a vector.
 * @param a Input vector
 * @return Minimum value in the vector
 */
float cpuMin(const std::vector<float>& a) {
    float m = FLT_MAX;
    for (float x : a) if (x < m) m = x;
    return m;
}

/**
 * @brief CPU helper: Finds the minimum value and its index in a vector.
 * @param a      Input vector
 * @param minVal Output: minimum value
 * @param minIdx Output: index of minimum value
 */
void cpuArgMin(const std::vector<float>& a, float& minVal, int& minIdx) {
    minVal = FLT_MAX; minIdx = -1;
    for (int i = 0; i < (int)a.size(); ++i) {
        if (a[i] < minVal) { minVal = a[i]; minIdx = i; }
    }
}

/**
 * @brief Program entry point. Demonstrates block-level reduction and argmin in CUDA.
 *
 * Usage: ./chapter6_demo [N]
 *   N: Number of elements (default: 1<<20)
 *
 * - Generates random input data
 * - Runs two CUDA kernels: block min and block argmin
 * - Combines partial results on CPU
 * - Verifies against CPU reference
 */
int main(int argc, char** argv) {
    // -------------------------
    // Config
    // -------------------------

    // Number of elements (default: ~1M if not specified)
    const int N         = (argc > 1) ? std::atoi(argv[1]) : 1 << 20;
    const int blockSize = 256; // Threads per block
    const int numBlocks = (N + blockSize - 1) / blockSize; // Number of blocks

    std::printf("N = %d, blocks = %d, blockSize = %d\n", N, numBlocks, blockSize);

    // -------------------------
    // Host data (random)
    // -------------------------

    // Host input: fill with random floats in [-1000, 1000]
    std::vector<float> h_in(N);
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

    // -------------------------
    // Device allocations
    // -------------------------

    // Device pointers for input and output arrays
    float *d_in = nullptr, *d_block_mins = nullptr, *d_block_argmin_vals = nullptr;
    int   *d_block_argmin_idxs = nullptr;


    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_in,               N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_mins,       numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_argmin_vals,numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_argmin_idxs,numBlocks * sizeof(int)));


    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // -------------------------
    // Launch: block minimum
    // Shared bytes: blockSize * sizeof(float)
    // -------------------------

    // Launch blockMinKernel: each block finds its minimum
    size_t shmemBytesMin = blockSize * sizeof(float);
    blockMinKernel<<<numBlocks, blockSize, shmemBytesMin>>>(d_in, d_block_mins, N);
    CUDA_CHECK(cudaGetLastError());

    // -------------------------
    // Launch: block argmin
    // Shared bytes: blockSize*(sizeof(float)+sizeof(int))
    // -------------------------

    // Launch blockArgMinKernel: each block finds its min value and index
    size_t shmemBytesArg = blockSize * (sizeof(float) + sizeof(int));
    blockArgMinKernel<<<numBlocks, blockSize, shmemBytesArg>>>(
        d_in, d_block_argmin_vals, d_block_argmin_idxs, N);
    CUDA_CHECK(cudaGetLastError());

    // -------------------------
    // Bring partial results back
    // -------------------------

    // Host arrays for partial results from each block
    std::vector<float> h_block_mins(numBlocks);
    std::vector<float> h_block_argmin_vals(numBlocks);
    std::vector<int>   h_block_argmin_idxs(numBlocks);


    // Copy partial results from device to host
    CUDA_CHECK(cudaMemcpy(h_block_mins.data(),        d_block_mins,
                          numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_argmin_vals.data(), d_block_argmin_vals,
                          numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_argmin_idxs.data(), d_block_argmin_idxs,
                          numBlocks * sizeof(int),   cudaMemcpyDeviceToHost));

    // -------------------------
    // CPU combination (final min)
    // -------------------------

    // Combine per-block minimums on CPU to get final minimum
    float finalMin = FLT_MAX;
    for (int b = 0; b < numBlocks; ++b) {
        if (h_block_mins[b] < finalMin) finalMin = h_block_mins[b];
    }

    // CPU combination (final argmin)

    // Combine per-block argmin results on CPU to get final min value and index
    float finalArgMinVal = FLT_MAX;
    int   finalArgMinIdx = -1;
    for (int b = 0; b < numBlocks; ++b) {
        float v = h_block_argmin_vals[b];
        int   i = h_block_argmin_idxs[b];
        if (v < finalArgMinVal) { finalArgMinVal = v; finalArgMinIdx = i; }
    }

    // -------------------------
    // CPU references
    // -------------------------

    // Compute reference results on CPU for verification
    float refMin = cpuMin(h_in);
    float refArgMinVal; int refArgMinIdx;
    cpuArgMin(h_in, refArgMinVal, refArgMinIdx);

    // -------------------------
    // Report + simple verification
    // -------------------------
    std::printf("\nGPU block-reduced MIN combined on CPU  : %.6f\n", finalMin);
    std::printf("CPU reference MIN                      : %.6f\n", refMin);

    std::printf("\nGPU block-reduced ARGMIN val/index     : %.6f @ %d\n",
                finalArgMinVal, finalArgMinIdx);
    std::printf("CPU reference ARGMIN val/index         : %.6f @ %d\n",
                refArgMinVal, refArgMinIdx);


    // Check if GPU and CPU results match (tolerance for float error)
    bool okMin    = (std::abs(finalMin - refMin) <= 1e-5f);
    bool okArgMin = (std::abs(finalArgMinVal - refArgMinVal) <= 1e-5f) &&
                    (finalArgMinIdx == refArgMinIdx);

    std::printf("\nVerification: MIN %s, ARGMIN %s\n",
                okMin ? "OK" : "MISMATCH",
                okArgMin ? "OK" : "MISMATCH");

    // -------------------------
    // Cleanup
    // -------------------------

    // Free device memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_block_mins));
    CUDA_CHECK(cudaFree(d_block_argmin_vals));
    CUDA_CHECK(cudaFree(d_block_argmin_idxs));

    // Return success if both verifications pass
    return (okMin && okArgMin) ? EXIT_SUCCESS : EXIT_FAILURE;
}
