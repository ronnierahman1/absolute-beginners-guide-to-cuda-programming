/*
 * hello_grid.cu
 *
 * Goal:
 *   Show how to launch MULTIPLE BLOCKS of threads and print:
 *     - each thread's GLOBAL ID (unique across the whole launch)
 *     - the BLOCK ID (which team it belongs to)
 *     - the LOCAL THREAD ID inside the block (its number within the team)
 *
 * What you will see:
 *   Lines like:
 *     Hello from global 0 (block 0, local 0)
 *     Hello from global 1 (block 0, local 1)
 *     ...
 *     Hello from global 4 (block 1, local 0)
 *   The exact order of lines may vary because many threads can print at nearly
 *   the same time. That’s normal on a GPU.
 *
 * Requirements:
 *   - NVIDIA GPU that supports CUDA
 *   - CUDA Toolkit installed (so you have the 'nvcc' compiler)
 *
 * How to build (Linux/macOS):
 *   nvcc hello_grid.cu -o hello_grid
 *
 * How to run (Linux/macOS):
 *   ./hello_grid
 *
 * How to build & run (Windows, x64 Native Tools / Developer Command Prompt):
 *   nvcc hello_grid.cu -o hello_grid.exe
 *   hello_grid.exe
 */

#include <cstdio>          // printf / fprintf
#include <cuda_runtime.h>  // cudaDeviceSynchronize, cudaGetErrorString

// -----------------------------------------------------------------------------
// Kernel: code that runs on the GPU.
// "__global__" means: launch from the CPU (host), execute on the GPU (device).
//
// In this kernel we will:
//   1) Read the thread's local ID inside its block (threadIdx.x).
//   2) Read the block's ID inside the grid (blockIdx.x).
//   3) Read how many threads are in a block (blockDim.x).
//   4) Compute a GLOBAL thread ID that is unique across ALL blocks:
//
//        global_id = local_id + block_id * threads_per_block
//
//   5) Print a friendly line that shows all three values.
// -----------------------------------------------------------------------------
__global__ void helloGrid() {
    // Local thread number inside this block (0 .. blockDim.x-1)
    int local_id = threadIdx.x;

    // Block number inside the grid (0 .. gridDim.x-1)
    int block_id = blockIdx.x;

    // Number of threads per block (chosen at launch time)
    int threads_per_block = blockDim.x;

    // Global thread ID (unique across the entire launch)
    int global_id = local_id + block_id * threads_per_block;

    // Print a message that shows who we are.
    // NOTE: When many threads print, the order of lines is not guaranteed.
    printf("Hello from global %d (block %d, local %d)\n",
           global_id, block_id, local_id);
}

int main() {
    // -------------------------------
    // Choose how many blocks and threads to launch.
    // For beginners, start small so the output is short and readable.
    //
    // Example here:
    //   - 2 blocks
    //   - 4 threads per block
    //
    // Total threads launched = 2 * 4 = 8 global threads.
    // Global IDs will be 0..7.
    // Block 0 has locals 0..3 (global 0..3)
    // Block 1 has locals 0..3 (global 4..7)
    // -------------------------------
    int blocks = 2;
    int threads_per_block = 4;

    // Launch the kernel on the GPU.
    // The special syntax <<<blocks, threads_per_block>>> tells CUDA how many
    // blocks and threads per block to create.
    helloGrid<<<blocks, threads_per_block>>>();

    // Make the CPU wait until the GPU is done.
    // This also flushes all printf output from the GPU so you can see it.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0; // success
}

/*
 * Try these small experiments (one at a time):
 *
 * 1) More threads per block (still 2 blocks):
 *      threads_per_block = 8;    // expect global IDs 0..15
 *      helloGrid<<<2, threads_per_block>>>();
 *
 * 2) More blocks (still 4 threads per block):
 *      blocks = 3;               // expect 3 * 4 = 12 threads (global 0..11)
 *      helloGrid<<<blocks, 4>>>();
 *
 * 3) Mix both:
 *      blocks = 3;
 *      threads_per_block = 8;    // expect 24 global threads (0..23)
 *      helloGrid<<<blocks, threads_per_block>>>();
 *
 * 4) Predict the global ID:
 *      For block_id = 2 and local_id = 5 with threads_per_block = 8:
 *      global_id = 5 + 2 * 8 = 21
 *      Look for a line "global 21 (block 2, local 5)" in the output.
 */
