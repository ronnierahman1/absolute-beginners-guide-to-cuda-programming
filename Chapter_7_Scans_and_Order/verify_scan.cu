#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

/**
 * @file verify_scan.cu
 * @brief CUDA scan (prefix sum) verifier for Chapter 7.
 *
 * - Implements block-local scan, block sum collection, and global scan via offset add.
 * - Compares both inclusive and exclusive scan results against CPU reference for a variety of patterns, sizes, and block sizes.
 */

/**
 * @brief Macro to check CUDA API calls for errors and exit on failure.
 */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

// ---------------------------------------------
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
// ---------------------------------------------
__global__ void blockScanInclusiveWriteSums(const int* __restrict__ d_in,
                                            int* __restrict__ d_out,
                                            int  n,
                                            int* __restrict__ d_blockSums)
{
    extern __shared__ int temp[]; // size = blockDim.x
    const int tid  = threadIdx.x;
    const int g0   = blockIdx.x * blockDim.x;   // start of this block
    const int gid  = g0 + tid;
    const int tail = n - g0;
    const int active = tail > blockDim.x ? blockDim.x : max(tail, 0);

    temp[tid] = (tid < active) ? d_in[gid] : 0;
    __syncthreads();

    // iterative doubling (inclusive)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int add = 0;
        if (tid >= offset && tid < active) add = temp[tid - offset];
        __syncthreads();
        if (tid < active) temp[tid] += add;
        __syncthreads();
    }

    if (tid < active) d_out[gid] = temp[tid];

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
// ---------------------------------------------
__global__ void addBlockOffsets(int* __restrict__ d_data,
                                int n,
                                const int* __restrict__ d_blockOffsets)
{
    const int g0 = blockIdx.x * blockDim.x;
    const int gid = g0 + threadIdx.x;
    const int tail = n - g0;
    const int active = tail > blockDim.x ? blockDim.x : max(tail, 0);

    const int off = d_blockOffsets[blockIdx.x];
    if (threadIdx.x < active) d_data[gid] += off;
}

// ---------------------------------------------
/**
 * @brief CUDA kernel: Converts an inclusive scan to exclusive scan by shifting right.
 *
 * @param d_inclusive Input inclusive scan (device pointer)
 * @param d_exclusive Output exclusive scan (device pointer)
 * @param n           Number of elements
 */
// ---------------------------------------------
__global__ void postShiftExclusive(const int* __restrict__ d_inclusive,
                                   int* __restrict__ d_exclusive,
                                   int n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    d_exclusive[gid] = (gid == 0) ? 0 : d_inclusive[gid - 1];
}

// ---------------------------------------------
// =======================
// CPU reference scan routines
// =======================
// ---------------------------------------------
/**
 * @brief CPU reference: inclusive scan (prefix sum) for verification.
 * @param in  Input vector
 * @param out Output vector (resized to match input)
 */
void cpuInclusive(const std::vector<int>& in, std::vector<int>& out) {
    out.resize(in.size());
    long long run = 0; // avoid intermediate overflow for large patterns
    for (size_t i = 0; i < in.size(); ++i) {
        run += in[i];
        out[i] = static_cast<int>(run); // tighten back to int for comparison
    }
}

/**
 * @brief CPU reference: exclusive scan (prefix sum) for verification.
 * @param in  Input vector
 * @param out Output vector (resized to match input)
 */
void cpuExclusive(const std::vector<int>& in, std::vector<int>& out) {
    out.resize(in.size());
    long long run = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = static_cast<int>(run);
        run += in[i];
    }
}

// ---------------------------------------------
// =======================
// GPU scan pipeline helpers
// =======================
// ---------------------------------------------
/**
 * @brief Host wrapper: Performs stitched inclusive scan on device array.
 *
 * @param d_in   Input array (device pointer)
 * @param d_out  Output array (device pointer)
 * @param n      Number of elements
 * @param block  Threads per block
 */
void gpuInclusiveScan_stitched(const int* d_in, int* d_out, int n, int block) {
    if (n == 0) return;
    const int grid = (n + block - 1) / block;

    int *d_blockSums = nullptr, *d_blockOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums,   grid * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets,grid * sizeof(int)));

    // local scans + block sums
    blockScanInclusiveWriteSums<<<grid, block, block * sizeof(int)>>>(d_in, d_out, n, d_blockSums);
    CUDA_CHECK(cudaGetLastError());

    // pull to host, build exclusive offsets, push back
    std::vector<int> h_blockSums(grid, 0), h_blockOffsets(grid, 0);
    CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums, grid * sizeof(int), cudaMemcpyDeviceToHost));
    int accum = 0;
    for (int b = 0; b < grid; ++b) { h_blockOffsets[b] = accum; accum += h_blockSums[b]; }
    CUDA_CHECK(cudaMemcpy(d_blockOffsets, h_blockOffsets.data(), grid * sizeof(int), cudaMemcpyHostToDevice));

    // globalize
    addBlockOffsets<<<grid, block>>>(d_out, n, d_blockOffsets);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_blockSums);
    cudaFree(d_blockOffsets);
}

/**
 * @brief Host wrapper: Computes exclusive scan from input using inclusive scan and post-shift.
 *
 * @param d_in      Input array (device pointer)
 * @param d_excl    Output exclusive scan (device pointer)
 * @param n         Number of elements
 * @param block     Threads per block
 * @param scratchInclusive Temporary device buffer (size n)
 */
void gpuExclusiveScan_fromInclusive(const int* d_in, int* d_excl, int n, int block, int* scratchInclusive /*size n*/) {
    if (n == 0) return;
    gpuInclusiveScan_stitched(d_in, scratchInclusive, n, block);
    const int grid = (n + block - 1) / block;
    postShiftExclusive<<<grid, block>>>(scratchInclusive, d_excl, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------
// Input generators (deterministic by seed)
// ---------------------------------------------
/**
 * @brief Enum for input patterns used in scan verification.
 */
enum class Pattern { ONES, RAMP, RANDOM_SMALL, ALTERNATING, RANDOM_WIDE };

/**
 * @brief Generates input vector for verifier based on pattern.
 * @param n    Number of elements
 * @param p    Pattern type
 * @param seed Random seed
 * @return Generated vector
 */
std::vector<int> makeInput(size_t n, Pattern p, uint32_t seed=42) {
    std::vector<int> v(n, 0);
    std::mt19937 rng(seed);
    switch (p) {
        case Pattern::ONES:
            std::fill(v.begin(), v.end(), 1);
            break;
        case Pattern::RAMP:
            for (size_t i = 0; i < n; ++i) v[i] = static_cast<int>(i % 97) - 48; // small wrap, includes negatives
            break;
        case Pattern::RANDOM_SMALL: {
            std::uniform_int_distribution<int> dist(-3, 3);
            for (auto& x : v) x = dist(rng);
            break;
        }
        case Pattern::ALTERNATING:
            for (size_t i = 0; i < n; ++i) v[i] = (i & 1) ? -1 : 2;
            break;
        case Pattern::RANDOM_WIDE: {
            std::uniform_int_distribution<int> dist(-10000, 10000);
            for (auto& x : v) x = dist(rng);
            break;
        }
    }
    return v;
}

// ---------------------------------------------
// Mismatch reporter
// ---------------------------------------------
/**
 * @brief Reports mismatches between two vectors, up to maxShow.
 * @param got     Computed vector
 * @param ref     Reference vector
 * @param maxShow Maximum mismatches to show
 * @return true if all match, false otherwise
 */
bool reportMismatches(const std::vector<int>& got, const std::vector<int>& ref, int maxShow=5) {
    if (got.size() != ref.size()) {
        printf("Size mismatch: got %zu vs ref %zu\n", got.size(), ref.size());
        return false;
    }
    int shown = 0;
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != ref[i]) {
            if (shown < maxShow) {
                printf("  idx %zu: got %d, expected %d\n", i, got[i], ref[i]);
                ++shown;
            }
            ok = false;
        }
    }
    if (!ok && (int)got.size() > maxShow) {
        printf("  ... and possibly more mismatches not shown\n");
    }
    return ok;
}

// ---------------------------------------------
// Single test case
// ---------------------------------------------
/**
 * @brief Runs a single scan test case for the verifier.
 * @param N       Number of elements
 * @param block   Threads per block
 * @param pat     Pattern type
 * @param verbose Print details if true
 * @return true if test passes, false otherwise
 */
bool runCase(int N, int block, Pattern pat, bool verbose=false) {
    if (verbose) {
        printf("Case: N=%d, block=%d, pattern=%d\n", N, block, int(pat));
    }

    // host data
    std::vector<int> h_in = makeInput(N, pat);
    std::vector<int> h_ref_incl, h_ref_excl;
    cpuInclusive(h_in, h_ref_incl);
    cpuExclusive(h_in, h_ref_excl);

    // device buffers
    int *d_in = nullptr, *d_out_incl = nullptr, *d_out_excl = nullptr, *d_scratch = nullptr;
    if (N > 0) {
        CUDA_CHECK(cudaMalloc(&d_in,         N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_incl,   N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_excl,   N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scratch,    N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    }

    // GPU inclusive
    gpuInclusiveScan_stitched(d_in, d_out_incl, N, block);

    // GPU exclusive (via post-shift from inclusive)
    gpuExclusiveScan_fromInclusive(d_in, d_out_excl, N, block, d_scratch);

    // fetch & compare
    std::vector<int> h_out_incl(N), h_out_excl(N);
    if (N > 0) {
        CUDA_CHECK(cudaMemcpy(h_out_incl.data(), d_out_incl, N * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out_excl.data(), d_out_excl, N * sizeof(int), cudaMemcpyDeviceToHost));
    }

    bool okIncl = reportMismatches(h_out_incl, h_ref_incl);
    bool okExcl = reportMismatches(h_out_excl, h_ref_excl);

    if (verbose) {
        printf("  Inclusive: %s\n", okIncl ? "OK" : "FAIL");
        printf("  Exclusive: %s\n", okExcl ? "OK" : "FAIL");
    }

    cudaFree(d_in);
    cudaFree(d_out_incl);
    cudaFree(d_out_excl);
    cudaFree(d_scratch);

    return okIncl && okExcl;
}

// ---------------------------------------------
// MAIN: sweep a matrix of sizes, blocks, patterns
// ---------------------------------------------
/**
 * @brief Program entry point. Sweeps a matrix of sizes, blocks, and patterns to verify scan correctness.
 * @return 0 if all cases pass, 1 otherwise
 */
int main() {
    std::vector<int> sizes = {0, 1, 7, 8, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 1<<20};
    std::vector<int> blocks = {32, 64, 96, 128, 192, 256, 512}; // include non-powers to stress tail logic
    std::vector<Pattern> pats = {Pattern::ONES, Pattern::RAMP, Pattern::RANDOM_SMALL, Pattern::ALTERNATING, Pattern::RANDOM_WIDE};

    int total = 0, passed = 0;
    for (int N : sizes) {
        for (int B : blocks) {
            for (Pattern p : pats) {
                ++total;
                bool ok = runCase(N, B, p, /*verbose=*/false);
                if (!ok) {
                    printf("FAILED: N=%d, block=%d, pattern=%d\n", N, B, int(p));
                } else {
                    ++passed;
                }
            }
        }
    }

    printf("\nSummary: %d/%d cases passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
