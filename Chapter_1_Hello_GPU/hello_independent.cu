/*
 * hello_independent.cu
 *
 * Goal:
 *   Prove that each GPU thread can work independently on its own piece of data.
 *   Each thread writes its unique GLOBAL ID into an array position that matches it.
 *   After the kernel runs, we copy the array back and check that out[i] == i.
 *
 * What this shows:
 *   - How to compute a global thread ID from block and thread indices.
 *   - How each thread writes to a unique index (no collisions).
 *   - How to copy results back to the CPU and do a simple check (PASS/FAIL).
 *
 * Requirements:
 *   - NVIDIA GPU with CUDA support
 *   - CUDA Toolkit installed (you should have the 'nvcc' compiler)
 *
 * Build (Linux/macOS):
 *   nvcc hello_independent.cu -o hello_independent
 *
 * Run (Linux/macOS):
 *   ./hello_independent
 *
 * Build & Run (Windows, x64 Developer Command Prompt):
 *   nvcc hello_independent.cu -o hello_independent.exe
 *   hello_independent.exe
 */

#include <cstdio>          // printf / fprintf
#include <vector>          // std::vector for easy host-side storage
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, cudaDeviceSynchronize, etc.

// ---------------------------
// KERNEL: runs on the GPU
// ---------------------------
// Each thread computes a GLOBAL ID and writes that ID into out[global_id].
// The small "if" check ensures we don't write past the end of the array.
__global__ void writeGlobalIds(int* out, int N) {
    // local thread number inside its block: 0..(blockDim.x-1)
    int local_id = threadIdx.x;

    // which block am I in? 0..(gridDim.x-1)
    int block_id = blockIdx.x;

    // how many threads per block were launched?
    int threads_per_block = blockDim.x;

    // GLOBAL id is unique across ALL threads in ALL blocks
    int global_id = local_id + block_id * threads_per_block;

    // Safety check: only write if the global_id is a valid array index
    if (global_id < N) {
        out[global_id] = global_id;
    }
}

int main() {
    // For beginners: keep numbers small so it’s easy to see what happens.
    // We’ll launch 2 blocks with 4 threads each = 8 total threads.
    int blocks = 2;
    int threads_per_block = 4;
    int total_threads = blocks * threads_per_block; // 8

    // Our array size N will match the number of launched threads.
    // (Later, if N doesn't match, the 'if (global_id < N)' keeps us safe.)
    int N = total_threads;

    // ---------------------------
    // 1) Allocate device memory
    // ---------------------------
    int* d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // OPTIONAL: Fill device memory with a known pattern first.
    // If any thread forgets to write, we’ll spot the leftover pattern.
    // 0xFF is often used; for int buffers it looks like -1 in memory.
    err = cudaMemset(d_out, 0xFF, N * sizeof(int));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // ---------------------------
    // 2) Launch the kernel
    // ---------------------------
    writeGlobalIds<<<blocks, threads_per_block>>>(d_out, N);

    // Wait for the GPU to finish and report any runtime errors.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel/Sync error: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // ---------------------------
    // 3) Copy results back to host (CPU) memory
    // ---------------------------
    std::vector<int> h_out(N); // host-side array to receive results
    err = cudaMemcpy(h_out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy DeviceToHost failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // ---------------------------
    // 4) Check the results (simple, clear test)
    //    Expect: out[i] == i for every i = 0..N-1
    // ---------------------------
    size_t mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != i) {
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::printf("PASS: all %d threads wrote their own indices correctly.\n", N);
    } else {
        std::printf("FAIL: %zu mismatches found.\n", mismatches);
        // For tiny N, it helps to show what we got:
        std::printf("Values: ");
        for (int i = 0; i < N; ++i) {
            std::printf("%d ", h_out[i]);
        }
        std::printf("\n");
    }

    // ---------------------------
    // 5) Clean up device memory
    // ---------------------------
    cudaFree(d_out);
    return (mismatches == 0) ? 0 : 1;
}

/*
 * Try these small experiments:
 *
 * 1) Change how many threads you launch:
 *      blocks = 3; threads_per_block = 4;  // 12 total threads
 *      blocks = 1; threads_per_block = 8;  // 8 total threads
 *    Rebuild and run. The program should still print PASS.
 *
 * 2) Make N bigger than the number of threads (e.g., N = 10 while launching 8 threads).
 *    Then only the first 8 slots will be written (0..7). The rest keep the pattern (-1).
 *    You’ll see mismatches > 0, which is expected in this experiment.
 *
 * 3) Make N smaller than the number of threads (e.g., N = 6 while launching 8 threads).
 *    The 'if (global_id < N)' check prevents out-of-bounds writes.
 *    You should still see PASS because we kept it safe.
 */
