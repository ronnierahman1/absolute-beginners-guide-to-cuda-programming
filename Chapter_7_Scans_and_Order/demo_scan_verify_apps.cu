
/**
 * @file demo_scan_verify_apps.cu
 * @brief CUDA demos + full §7.6 verifier.
 *
 * Demos reuse the stitched scan building blocks:
 *   1) Stream compaction (exclusive scan -> write positions)
 *   2) Histogram grouping (counts -> exclusive offsets -> grouped scatter)
 *   3) CDF construction (inclusive scan + normalize)
 *
 * Add --verify to run the §7.6 correctness harness before the demo.
 *
 * Build: nvcc -O2 demo_scan_apps.cu -o demo
 * Run:   ./demo --verify --mode compact|hist|cdf [--N <int>] [--BLOCK <int>] [--BINS <int>]
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

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

// =======================
// Scan (stitched) building blocks (int)
// =======================

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
    extern __shared__ int temp[]; // size = blockDim.x
    const int tid  = threadIdx.x;
    const int g0   = blockIdx.x * blockDim.x;
    const int gid  = g0 + tid;
    const int tail = n - g0;
    const int active = tail > blockDim.x ? blockDim.x : max(tail, 0);

    temp[tid] = (tid < active) ? d_in[gid] : 0;
    __syncthreads();

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
    const int g0 = blockIdx.x * blockDim.x;
    const int gid = g0 + threadIdx.x;
    const int tail = n - g0;
    const int active = tail > blockDim.x ? blockDim.x : max(tail, 0);

    const int off = d_blockOffsets[blockIdx.x];
    if (threadIdx.x < active) d_data[gid] += off;
}

/**
 * @brief CUDA kernel: Converts an inclusive scan to exclusive scan by shifting right.
 *
 * @param d_inclusive Input inclusive scan (device pointer)
 * @param d_exclusive Output exclusive scan (device pointer)
 * @param n           Number of elements
 */
__global__ void postShiftExclusive(const int* __restrict__ d_inclusive,
                                   int* __restrict__ d_exclusive,
                                   int n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    d_exclusive[gid] = (gid == 0) ? 0 : d_inclusive[gid - 1];
}

// =======================
// Host wrappers for scan (int)
// =======================
/**
 * @brief Host wrapper: Performs stitched inclusive scan on device array.
 *
 * @param d_in   Input array (device pointer)
 * @param d_out  Output array (device pointer)
 * @param n      Number of elements
 * @param block  Threads per block
 */
void gpuInclusiveScan_stitched(const int* d_in, int* d_out, int n, int block)
{
    if (n == 0) return;
    const int grid = (n + block - 1) / block;

    int *d_blockSums = nullptr, *d_blockOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums,   grid * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets,grid * sizeof(int)));

    blockScanInclusiveWriteSums<<<grid, block, block * sizeof(int)>>>(d_in, d_out, n, d_blockSums);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> h_blockSums(grid, 0), h_blockOffsets(grid, 0);
    CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums, grid * sizeof(int), cudaMemcpyDeviceToHost));

    int accum = 0;
    for (int b = 0; b < grid; ++b) { h_blockOffsets[b] = accum; accum += h_blockSums[b]; }
    CUDA_CHECK(cudaMemcpy(d_blockOffsets, h_blockOffsets.data(), grid * sizeof(int), cudaMemcpyHostToDevice));

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
void gpuExclusiveScan_fromInclusive(const int* d_in, int* d_excl, int n, int block, int* scratchInclusive /*size n*/)
{
    if (n == 0) return;
    gpuInclusiveScan_stitched(d_in, scratchInclusive, n, block);
    const int grid = (n + block - 1) / block;
    postShiftExclusive<<<grid, block>>>(scratchInclusive, d_excl, n);
    CUDA_CHECK(cudaGetLastError());
}

// =======================
// Float stitched scan (for CDF demo)
// =======================
/**
 * @brief CUDA kernel struct: Performs block-local inclusive scan and writes block sums (float version).
 */
struct FloatScanK1 {
    static __global__ void run(const float* __restrict__ d_in,
                               float* __restrict__ d_out,
                               int  n,
                               float* __restrict__ d_blockSums) {
        extern __shared__ float temp[]; // size = blockDim.x
        const int tid  = threadIdx.x;
        const int g0   = blockIdx.x * blockDim.x;
        const int gid  = g0 + tid;
        const int tail = n - g0;
        const int active = tail > blockDim.x ? blockDim.x : max(tail, 0);

        temp[tid] = (tid < active) ? d_in[gid] : 0.0f;
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            float add = 0.0f;
            if (tid >= offset && tid < active) add = temp[tid - offset];
            __syncthreads();
            if (tid < active) temp[tid] += add;
            __syncthreads();
        }

        if (tid < active) d_out[gid] = temp[tid];

        if (tid == 0) {
            float sum = (active > 0) ? temp[active - 1] : 0.0f;
            d_blockSums[blockIdx.x] = sum;
        }
    }
};

/**
 * @brief CUDA kernel struct: Adds per-block offsets to produce a global inclusive scan (float version).
 */
struct FloatScanK2 {
    static __global__ void run(float* __restrict__ d_data,
                               int n,
                               const float* __restrict__ d_blockOffsets) {
        const int g0 = blockIdx.x * blockDim.x;
        const int gid = g0 + threadIdx.x;
        const int tail = n - g0;
        const int active = tail > blockDim.x ? blockDim.x : max(tail, 0);
        const float off = d_blockOffsets[blockIdx.x];
        if (threadIdx.x < active) d_data[gid] += off;
    }
};

/**
 * @brief Host wrapper: Performs stitched inclusive scan on device float array.
 *
 * @param d_in   Input array (device pointer)
 * @param d_out  Output array (device pointer)
 * @param n      Number of elements
 * @param block  Threads per block
 */
void gpuInclusiveScan_stitched_float(const float* d_in, float* d_out, int n, int block)
{
    if (n == 0) return;
    const int grid = (n + block - 1) / block;

    float *d_blockSums=nullptr, *d_blockOffsets=nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums,   grid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets,grid*sizeof(float)));

    FloatScanK1::run<<<grid, block, block * sizeof(float)>>>(d_in, d_out, n, d_blockSums);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_blockSums(grid, 0.0f), h_blockOffsets(grid, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums, grid*sizeof(float), cudaMemcpyDeviceToHost));
    float accum = 0.0f;
    for (int b = 0; b < grid; ++b) { h_blockOffsets[b] = accum; accum += h_blockSums[b]; }
    CUDA_CHECK(cudaMemcpy(d_blockOffsets, h_blockOffsets.data(), grid*sizeof(float), cudaMemcpyHostToDevice));

    FloatScanK2::run<<<grid, block>>>(d_out, n, d_blockOffsets);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_blockSums); cudaFree(d_blockOffsets);
}

// =======================
// 1) Stream compaction (keep positives)
// =======================
/**
 * @brief CUDA kernel: Computes flags for stream compaction (keep positives).
 *
 * @param d_in   Input array (device pointer)
 * @param d_flag Output flags (1 if d_in > 0, else 0)
 * @param n      Number of elements
 */
__global__ void computeFlagsKeepPositive(const int* __restrict__ d_in,
                                         int* __restrict__ d_flag,
                                         int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    d_flag[gid] = (d_in[gid] > 0) ? 1 : 0;
}

/**
 * @brief CUDA kernel: Scatters kept elements to output using exclusive positions.
 *
 * @param d_in   Input array (device pointer)
 * @param d_flag Flags array (device pointer)
 * @param d_pos  Exclusive positions (device pointer)
 * @param d_out  Output array (device pointer)
 * @param n      Number of elements
 */
__global__ void scatterKept(const int* __restrict__ d_in,
                            const int* __restrict__ d_flag,
                            const int* __restrict__ d_pos,  // exclusive positions
                            int* __restrict__ d_out,
                            int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    if (d_flag[gid]) {
        int w = d_pos[gid];
        d_out[w] = d_in[gid];
    }
}

/**
 * @brief Runs stream compaction demo (keep > 0).
 *
 * @param N     Number of elements
 * @param BLOCK Threads per block
 */
void run_compaction(int N, int BLOCK)
{
    printf("=== Stream Compaction Demo (keep > 0) ===\n");
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);

    std::vector<int> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

    int *d_in=nullptr, *d_flag=nullptr, *d_pos_incl=nullptr, *d_pos=nullptr, *d_out=nullptr, *d_scratch=nullptr;
    if (N > 0) {
        CUDA_CHECK(cudaMalloc(&d_in,        N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_flag,      N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pos_incl,  N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pos,       N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out,       N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scratch,   N*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    }

    const int GRID = (N + BLOCK - 1) / BLOCK;

    computeFlagsKeepPositive<<<GRID, BLOCK>>>(d_in, d_flag, N);
    CUDA_CHECK(cudaGetLastError());

    // exclusive positions = exclusive_scan(flags)
    gpuInclusiveScan_stitched(d_flag, d_pos_incl, N, BLOCK);
    postShiftExclusive<<<GRID, BLOCK>>>(d_pos_incl, d_pos, N);
    CUDA_CHECK(cudaGetLastError());

    // total kept = lastInclusive + lastFlag
    int kept = 0;
    if (N > 0) {
        int lastIncl=0, lastFlag=0;
        CUDA_CHECK(cudaMemcpy(&lastIncl, d_pos_incl + (N-1), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&lastFlag, d_flag     + (N-1), sizeof(int), cudaMemcpyDeviceToHost));
        kept = lastIncl + lastFlag;
    }

    scatterKept<<<GRID, BLOCK>>>(d_in, d_flag, d_pos, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> h_out(kept);
    if (kept > 0) CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kept*sizeof(int), cudaMemcpyDeviceToHost));

    // CPU reference (stable)
    std::vector<int> ref; ref.reserve(N);
    for (int v : h_in) if (v > 0) ref.push_back(v);

    printf("Input (first 32): ");
    for (int i = 0; i < std::min(N,32); ++i) printf("%d ", h_in[i]);
    if (N > 32) printf("...");
    printf("\nKept=%d\nGPU out (first 32): ", kept);
    for (int i = 0; i < std::min(kept,32); ++i) printf("%d ", h_out[i]);
    if (kept > 32) printf("...");
    printf("\nCPU ref (first 32):  ");
    for (int i = 0; i < std::min((int)ref.size(),32); ++i) printf("%d ", ref[i]);
    if ((int)ref.size() > 32) printf("...");
    printf("\nMatch? %s\n", (h_out == ref ? "YES" : "NO"));

    cudaFree(d_in); cudaFree(d_flag); cudaFree(d_pos_incl); cudaFree(d_pos);
    cudaFree(d_out); cudaFree(d_scratch);
}

// =======================
// 2) Histogram grouping (values in 0..BINS-1)
// =======================
/**
 * @brief CUDA kernel: Computes histogram counts for values in 0..numBins-1.
 *
 * @param d_vals   Input values (device pointer)
 * @param n        Number of elements
 * @param numBins  Number of bins
 * @param d_counts Output counts per bin (device pointer)
 */
__global__ void histogramCounts(const int* __restrict__ d_vals,
                                int n, int numBins,
                                int* __restrict__ d_counts)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int b = d_vals[gid];
    if (b >= 0 && b < numBins) {
        atomicAdd(&d_counts[b], 1);
    }
}

/**
 * @brief CUDA kernel: Scatters values into grouped bins using per-bin offsets and atomic cursors.
 *
 * @param d_vals    Input values (device pointer)
 * @param n         Number of elements
 * @param numBins   Number of bins
 * @param d_offsets Per-bin offsets (device pointer)
 * @param d_cursors Per-bin atomic cursors (device pointer)
 * @param d_out     Output grouped array (device pointer)
 */
__global__ void histogramScatter(const int* __restrict__ d_vals,
                                 int n, int numBins,
                                 const int* __restrict__ d_offsets,   // start per bin (exclusive scan of counts)
                                 int* __restrict__ d_cursors,         // initialized to offsets
                                 int* __restrict__ d_out)             // length n
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int b = d_vals[gid];
    if (b >= 0 && b < numBins) {
        int pos = atomicAdd(&d_cursors[b], 1);
        d_out[pos] = b; // (or original index/record)
    }
}

/**
 * @brief Runs histogram grouping demo (values in 0..BINS-1).
 *
 * @param N     Number of elements
 * @param BINS  Number of bins
 * @param BLOCK Threads per block
 */
void run_histogram(int N, int BINS, int BLOCK)
{
    printf("=== Histogram Grouping Demo (values in [0,%d)) ===\n", BINS);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, std::max(1,BINS)-1);

    std::vector<int> h_vals(N);
    for (int i = 0; i < N; ++i) h_vals[i] = dist(rng);

    int *d_vals=nullptr, *d_counts=nullptr, *d_counts_incl=nullptr, *d_offsets=nullptr, *d_cursors=nullptr, *d_grouped=nullptr;
    if (N > 0) {
        CUDA_CHECK(cudaMalloc(&d_vals,     N*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_counts,      BINS*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts_incl, BINS*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets,     BINS*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cursors,     BINS*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grouped,     std::max(0,N)*sizeof(int)));

    const int GRID = (N + BLOCK - 1) / BLOCK;
    const int GRID_B = (BINS + BLOCK - 1) / BLOCK;

    CUDA_CHECK(cudaMemset(d_counts, 0, BINS*sizeof(int)));
    histogramCounts<<<GRID, BLOCK>>>(d_vals, N, BINS, d_counts);
    CUDA_CHECK(cudaGetLastError());

    // offsets = exclusive_scan(counts)
    gpuInclusiveScan_stitched(d_counts, d_counts_incl, BINS, BLOCK);
    postShiftExclusive<<<GRID_B, BLOCK>>>(d_counts_incl, d_offsets, BINS);
    CUDA_CHECK(cudaGetLastError());

    // init per-bin cursors = offsets
    CUDA_CHECK(cudaMemcpy(d_cursors, d_offsets, BINS*sizeof(int), cudaMemcpyDeviceToDevice));

    histogramScatter<<<GRID, BLOCK>>>(d_vals, N, BINS, d_offsets, d_cursors, d_grouped);
    CUDA_CHECK(cudaGetLastError());

    // Bring back counts/offsets for reporting
    std::vector<int> counts(BINS), offsets(BINS);
    CUDA_CHECK(cudaMemcpy(counts.data(),  d_counts,  BINS*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), d_offsets, BINS*sizeof(int), cudaMemcpyDeviceToHost));

    // Bring back a small prefix of grouped output to inspect
    std::vector<int> grouped(std::min(N, 128));
    if (N > 0) CUDA_CHECK(cudaMemcpy(grouped.data(), d_grouped, grouped.size()*sizeof(int), cudaMemcpyDeviceToHost));

    // Basic CPU validation: counts sum to N, offsets are exclusive prefix of counts
    int sumC = 0; for (int c : counts) sumC += c;
    bool offsetsOK = true;
    int run = 0;
    for (int b = 0; b < BINS; ++b) {
        if (offsets[b] != run) { offsetsOK = false; break; }
        run += counts[b];
    }

    printf("Counts:  ");
    for (int b = 0; b < std::min(BINS,16); ++b) printf("%d ", counts[b]);
    if (BINS > 16) printf("...");
    printf("\nOffsets: ");
    for (int b = 0; b < std::min(BINS,16); ++b) printf("%d ", offsets[b]);
    if (BINS > 16) printf("...");
    printf("\nSum(counts)=%d (expect %d). Offsets OK? %s\n", sumC, N, offsetsOK ? "YES" : "NO");

    printf("Grouped out (first %d): ", (int)grouped.size());
    for (int i = 0; i < (int)grouped.size(); ++i) printf("%d ", grouped[i]);
    if (N > (int)grouped.size()) printf("..."); printf("\n");

    // Optional strong check for small N
    if (N <= 4096) {
        std::vector<int> full(N);
        if (N > 0) CUDA_CHECK(cudaMemcpy(full.data(), d_grouped, N*sizeof(int), cudaMemcpyDeviceToHost));
        bool segmentsOK = true;
        for (int b = 0; b < BINS; ++b) {
            int start = offsets[b];
            int end   = start + counts[b];
            for (int i = start; i < end; ++i) {
                if (full[i] != b) { segmentsOK = false; break; }
            }
            if (!segmentsOK) break;
        }
        printf("Per-bin segments contain only bin id? %s\n", segmentsOK ? "YES" : "NO");
    } else {
        printf("Per-bin segment content check skipped (N too large).\n");
    }

    cudaFree(d_vals); cudaFree(d_counts); cudaFree(d_counts_incl);
    cudaFree(d_offsets); cudaFree(d_cursors); cudaFree(d_grouped);
}

// =======================
// 3) CDF construction (float weights >= 0)
// =======================
/**
 * @brief CUDA kernel: Normalizes inclusive scan to produce a CDF in [0,1].
 *
 * @param d_scanIncl Input inclusive scan (device pointer)
 * @param d_cdf      Output CDF (device pointer)
 * @param n          Number of elements
 * @param total      Total sum (for normalization)
 */
__global__ void normalizeCdf(const float* __restrict__ d_scanIncl,
                             float* __restrict__ d_cdf,
                             int n, float total)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    d_cdf[gid] = (total > 0.0f) ? (d_scanIncl[gid] / total) : 0.0f;
}

/**
 * @brief Runs CDF construction demo (float weights >= 0).
 *
 * @param N     Number of elements
 * @param BLOCK Threads per block
 */
void run_cdf(int N, int BLOCK)
{
    printf("=== CDF Demo (float weights >= 0) ===\n");
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_w(N);
    for (int i = 0; i < N; ++i) h_w[i] = dist(rng);

    float *d_w=nullptr, *d_scanIncl=nullptr, *d_cdf=nullptr;
    if (N > 0) {
        CUDA_CHECK(cudaMalloc(&d_w,        N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scanIncl, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cdf,      N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    }

    gpuInclusiveScan_stitched_float(d_w, d_scanIncl, N, BLOCK);

    float total = 0.0f;
    if (N > 0) CUDA_CHECK(cudaMemcpy(&total, d_scanIncl + (N-1), sizeof(float), cudaMemcpyDeviceToHost));

    const int GRID = (N + BLOCK - 1) / BLOCK;
    normalizeCdf<<<GRID, BLOCK>>>(d_scanIncl, d_cdf, N, total);
    CUDA_CHECK(cudaGetLastError());

    // Fetch a small prefix for display
    std::vector<float> h_cdf(std::min(N, 32));
    if (N > 0) CUDA_CHECK(cudaMemcpy(h_cdf.data(), d_cdf, h_cdf.size()*sizeof(float), cudaMemcpyDeviceToHost));

    printf("Weights (first 16): ");
    for (int i = 0; i < std::min(N,16); ++i) printf("%.3f ", h_w[i]);
    if (N > 16) printf("...");
    printf("\nCDF (first %d):      ", (int)h_cdf.size());
    for (int i = 0; i < (int)h_cdf.size(); ++i) printf("%.4f ", h_cdf[i]);
    if (N > (int)h_cdf.size()) printf("...");
    printf("\nTotal sum=%.6f (CDF[N-1] should be 1.0)\n", total);

    if (N > 0) {
        float last = 0.0f;
        CUDA_CHECK(cudaMemcpy(&last, d_cdf + (N-1), sizeof(float), cudaMemcpyDeviceToHost));
        printf("CDF[N-1]=%.6f\n", last);
    }

    cudaFree(d_w); cudaFree(d_scanIncl); cudaFree(d_cdf);
}

// =======================
// §7.6 Verifier (int) — runs when --verify is passed
// =======================
/**
 * @brief Namespace for §7.6 scan verifier routines.
 */
namespace verify76 {

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
            for (size_t i = 0; i < n; ++i) v[i] = static_cast<int>(i % 97) - 48; // includes negatives
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

/**
 * @brief CPU reference: inclusive scan (prefix sum) for verification.
 * @param in  Input vector
 * @param out Output vector (resized to match input)
 */
void cpuInclusive(const std::vector<int>& in, std::vector<int>& out) {
    out.resize(in.size());
    long long run = 0;
    for (size_t i = 0; i < in.size(); ++i) { run += in[i]; out[i] = (int)run; }
}

/**
 * @brief CPU reference: exclusive scan (prefix sum) for verification.
 * @param in  Input vector
 * @param out Output vector (resized to match input)
 */
void cpuExclusive(const std::vector<int>& in, std::vector<int>& out) {
    out.resize(in.size());
    long long run = 0;
    for (size_t i = 0; i < in.size(); ++i) { out[i] = (int)run; run += in[i]; }
}

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
        printf("  ... more mismatches not shown\n");
    }
    return ok;
}

/**
 * @brief Runs a single scan test case for the verifier.
 * @param N     Number of elements
 * @param block Threads per block
 * @param pat   Pattern type
 * @return true if test passes, false otherwise
 */
bool runCase(int N, int block, Pattern pat) {
    // host data
    std::vector<int> h_in = makeInput(N, pat);
    std::vector<int> h_ref_incl, h_ref_excl;
    cpuInclusive(h_in, h_ref_incl);
    cpuExclusive(h_in, h_ref_excl);

    // device buffers
    int *d_in=nullptr, *d_out_incl=nullptr, *d_out_excl=nullptr, *d_scratch=nullptr;
    if (N > 0) {
        CUDA_CHECK(cudaMalloc(&d_in,       N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_incl, N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_excl, N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scratch,  N*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    }

    // GPU inclusive & exclusive
    gpuInclusiveScan_stitched(d_in, d_out_incl, N, block);
    gpuExclusiveScan_fromInclusive(d_in, d_out_excl, N, block, d_scratch);

    // fetch & compare
    std::vector<int> h_out_incl(N), h_out_excl(N);
    if (N > 0) {
        CUDA_CHECK(cudaMemcpy(h_out_incl.data(), d_out_incl, N*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out_excl.data(), d_out_excl, N*sizeof(int), cudaMemcpyDeviceToHost));
    }

    bool okIncl = reportMismatches(h_out_incl, h_ref_incl);
    bool okExcl = reportMismatches(h_out_excl, h_ref_excl);

    cudaFree(d_in); cudaFree(d_out_incl); cudaFree(d_out_excl); cudaFree(d_scratch);
    return okIncl && okExcl;
}

/**
 * @brief Runs all scan test cases for the verifier.
 * @return 0 if all pass, 1 otherwise
 */
int runAll() {
    std::vector<int> sizes = {0, 1, 7, 8, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 1<<20};
    std::vector<int> blocks = {32, 64, 96, 128, 192, 256, 512}; // include non-powers
    std::vector<Pattern> pats = {Pattern::ONES, Pattern::RAMP, Pattern::RANDOM_SMALL, Pattern::ALTERNATING, Pattern::RANDOM_WIDE};

    int total = 0, passed = 0;
    for (int N : sizes) {
        for (int B : blocks) {
            for (Pattern p : pats) {
                ++total;
                bool ok = runCase(N, B, p);
                if (!ok) {
                    printf("FAILED: N=%d, block=%d, pattern=%d\n", N, B, int(p));
                } else {
                    ++passed;
                }
            }
        }
    }
    printf("\n§7.6 Verifier Summary: %d/%d cases passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}

} // namespace verify76

// =======================
// Tiny CLI
// =======================
/**
 * @brief Struct to hold command-line arguments.
 */
struct Args {
    std::string mode = "compact"; // compact | hist | cdf
    int N = 32;
    int BLOCK = 128;
    int BINS = 8;
    bool verify = false;
};

/**
 * @brief Prints usage information for the demo.
 */
void usage() {
    printf("Usage: ./demo --mode compact|hist|cdf [--N <int>] [--BLOCK <int>] [--BINS <int>] [--verify]\n");
}

/**
 * @brief Converts a string to int, with error checking.
 * @param s Input string
 * @return Integer value
 */
int toInt(const char* s) {
    char* end=nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') { fprintf(stderr, "Invalid int: %s\n", s); std::exit(1); }
    return (int)v;
}

/**
 * @brief Parses command-line arguments into Args struct.
 * @param argc Argument count
 * @param argv Argument values
 * @return Parsed Args struct
 */
Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--mode") && i+1 < argc) {
            a.mode = argv[++i];
        } else if (!std::strcmp(argv[i], "--N") && i+1 < argc) {
            a.N = toInt(argv[++i]);
        } else if (!std::strcmp(argv[i], "--BLOCK") && i+1 < argc) {
            a.BLOCK = toInt(argv[++i]);
        } else if (!std::strcmp(argv[i], "--BINS") && i+1 < argc) {
            a.BINS = toInt(argv[++i]);
        } else if (!std::strcmp(argv[i], "--verify")) {
            a.verify = true;
        } else if (!std::strcmp(argv[i], "--help")) {
            usage(); std::exit(0);
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(); std::exit(1);
        }
    }
    if (a.N < 0) a.N = 0;
    if (a.BLOCK <= 0) a.BLOCK = 128;
    if (a.BINS <= 0) a.BINS = 1;
    return a;
}

/**
 * @brief Program entry point. Runs selected scan demo and/or verifier based on command-line arguments.
 *
 * @param argc Argument count
 * @param argv Argument values
 * @return 0 on success
 */
int main(int argc, char** argv)
{
    Args a = parse(argc, argv);
    printf("Mode=%s, N=%d, BLOCK=%d, BINS=%d, verify=%s\n",
           a.mode.c_str(), a.N, a.BLOCK, a.BINS, a.verify ? "true" : "false");

    if (a.verify) {
        int vr = verify76::runAll();
        if (vr != 0) {
            printf("Verifier reported failures. Proceeding to demo anyway (use exit code if desired).\n");
        }
    }

    if (a.mode == "compact") {
        run_compaction(a.N, a.BLOCK);
    } else if (a.mode == "hist") {
        run_histogram(a.N, a.BINS, a.BLOCK);
    } else if (a.mode == "cdf") {
        run_cdf(a.N, a.BLOCK);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", a.mode.c_str());
        usage(); return 1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
