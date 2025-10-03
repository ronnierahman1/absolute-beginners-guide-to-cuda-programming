#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

/**
 * @file scan_global_stitch.cu
 * @brief Demonstrates a two-pass global inclusive scan (prefix sum) using CUDA.
 *
 * - First pass: block-local inclusive scan, writing block sums
 * - Second pass: add per-block offsets to produce a global scan
 * - Includes CPU reference for verification
 */
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

/**
 * @brief Macro to check CUDA API calls for errors and exit on failure.
 */
#define CUDA_CHECK(call)                                                     \
    do {                                                                       \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
                            cudaGetErrorString(_e));                                       \
            std::exit(1);                                                          \
        }                                                                        \
    } while (0)

/**
 * @brief CUDA kernel: Performs block-local inclusive scan and writes block sums.
 *
 * @param d_in        Input array (device pointer)
 * @param d_out       Output array (device pointer)
 * @param n           Number of elements
 * @param d_blockSums Output array for per-block sums (device pointer)
 *
 * Each block computes an inclusive scan of its segment and writes the sum of its block to d_blockSums.
 * Uses shared memory for intra-block scan.
 */
__global__ void blockScanInclusiveWriteSums(const int* __restrict__ d_in,
                                            int* __restrict__ d_out,
                                            int  n,
                                            int* __restrict__ d_blockSums)
{
    extern __shared__ int temp[]; // Shared memory for scan
    const int tid  = threadIdx.x; // Thread index within block
    const int g0   = blockIdx.x * blockDim.x; // Start index of this block
    const int gid  = g0 + tid;                // Global index for this thread
    const int tail = n - g0;                  // Elements remaining from g0
    const int active = tail > blockDim.x ? blockDim.x : max(tail, 0); // Active threads in this block

    // Load input into shared memory (pad with 0 outside range)
    temp[tid] = (tid < active) ? d_in[gid] : 0;
    __syncthreads();

    // Iterative doubling (inclusive scan)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int add = 0;
        if (tid >= offset && tid < active) {
            add = temp[tid - offset];
        }
        __syncthreads();
        if (tid < active) {
            temp[tid] += add;
        }
        __syncthreads();
    }

    // Write local scanned chunk to global memory
    if (tid < active) {
        d_out[gid] = temp[tid];
    }

    // Thread 0 records the block's total sum into d_blockSums
    if (tid == 0) {
        int sum = (active > 0) ? temp[active - 1] : 0;
        d_blockSums[blockIdx.x] = sum;
    }
}

// ---------------------------------------------
/**
 * @brief CUDA kernel: Adds per-block offsets to produce a global inclusive scan.
 *
 * @param d_data         Data array to update (device pointer)
 * @param n              Number of elements
 * @param d_blockOffsets Array of per-block offsets (device pointer)
 *
 * Each block adds its exclusive prefix sum (offset) to its segment.
 */
__global__ void addBlockOffsets(int* __restrict__ d_data,
                                int n,
                                const int* __restrict__ d_blockOffsets)
{
    const int g0 = blockIdx.x * blockDim.x; // Start index of this block
    const int gid = g0 + threadIdx.x;       // Global index
    const int tail = n - g0;                // Elements remaining from g0
    const int active = tail > blockDim.x ? blockDim.x : max(tail, 0); // Active threads

    // Offset for this block: sum of all *previous* blocks (exclusive prefix)
    const int offset = d_blockOffsets[blockIdx.x];

    if (threadIdx.x < active) {
        d_data[gid] += offset;
    }
}

// ---------------------------------------------
/**
 * @brief CPU reference: inclusive scan (prefix sum) for verification.
 * @param in  Input vector
 * @param out Output vector (resized to match input)
 */
void cpuInclusive(const std::vector<int>& in, std::vector<int>& out) {
    out.resize(in.size());
    int run = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        run += in[i];
        out[i] = run;
    }
}

// ---------------------------------------------
// MAIN: demo
// ---------------------------------------------
/**
 * @brief Program entry point. Demonstrates global inclusive scan using CUDA and verifies against CPU.
 *
 * - Generates patterned input
 * - Runs two-pass scan on GPU
 * - Verifies result against CPU reference
 */
int main() {

    // Problem size and launch configuration
    const int N = 23;          // Try non-multiple to test tail block
    const int BLOCK = 8;       // Threads per block
    const int GRID  = (N + BLOCK - 1) / BLOCK; // Number of blocks


    // Host input: fill with a small repeating pattern
    std::vector<int> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = (i % 5) + 1;

    // Device buffers

    int *d_in = nullptr, *d_out = nullptr, *d_blockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockSums, GRID * sizeof(int)));


    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // 1) Block-local inclusive scan + collect block sums

    // 1) Block-local inclusive scan + collect block sums
    blockScanInclusiveWriteSums<<<GRID, BLOCK, BLOCK * sizeof(int)>>>(
        d_in, d_out, N, d_blockSums);
    CUDA_CHECK(cudaGetLastError());

    // 2) Copy block sums to host and build exclusive block offsets

    // 2) Copy block sums to host and build exclusive block offsets
    std::vector<int> h_blockSums(GRID, 0);
    CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums,
                          GRID * sizeof(int), cudaMemcpyDeviceToHost));


    // Build exclusive prefix sum of block sums (block offsets)
    std::vector<int> h_blockOffsets(GRID, 0);
    int accum = 0;
    for (int b = 0; b < GRID; ++b) {
        h_blockOffsets[b] = accum;     // exclusive prefix
        accum += h_blockSums[b];
    }


    // Copy offsets back to device
    int* d_blockOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, GRID * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_blockOffsets, h_blockOffsets.data(),
                          GRID * sizeof(int), cudaMemcpyHostToDevice));


    // 3) Add offsets per block → global inclusive scan
    addBlockOffsets<<<GRID, BLOCK>>>(d_out, N, d_blockOffsets);
    CUDA_CHECK(cudaGetLastError());


    // Fetch result from device
    std::vector<int> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost));


    // Compare with CPU inclusive scan (quick check; Section 7.6 will formalize this)
    std::vector<int> h_ref;
    cpuInclusive(h_in, h_ref);


    // Print results
    printf("Input:  ");
    for (int v : h_in)  printf("%d ", v);
    printf("\nGPU:    ");
    for (int v : h_out) printf("%d ", v);
    printf("\nCPU:    ");
    for (int v : h_ref) printf("%d ", v);
    printf("\nMatch?  %s\n", (h_out == h_ref ? "YES" : "NO"));

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockSums);
    cudaFree(d_blockOffsets);
    return 0;
}
