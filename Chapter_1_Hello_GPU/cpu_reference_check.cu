/*
 * cpu_reference_check.cu
 *
 * Section 1.7 — Adding a CPU “Reference Check” (Count of Greetings)
 *
 * Goal (in plain words):
 *   - Launch several GPU threads.
 *   - Each GPU thread writes a simple mark (the number 1) into its own slot.
 *   - Copy the marks back to the CPU and COUNT them.
 *   - Compare the COUNT with what we EXPECT (blocks * threads_per_block).
 *   - If they match, we print PASS. If not, we print FAIL.
 *
 * Why this is useful:
 *   This pattern teaches you how to verify GPU work with a small, trusted
 *   CPU check. You will reuse this habit in bigger programs.
 *
 * Build (Linux/macOS):
 *   nvcc cpu_reference_check.cu -o cpu_reference_check
 *
 * Run (Linux/macOS):
 *   ./cpu_reference_check
 *
 * Build & Run (Windows, x64 Developer Command Prompt):
 *   nvcc cpu_reference_check.cu -o cpu_reference_check.exe
 *   cpu_reference_check.exe
 */

#include <cstdio>          // printf / fprintf
#include <vector>          // std::vector for simple CPU-side array
#include <cuda_runtime.h>  // basic CUDA runtime (cudaMalloc, cudaMemcpy, etc.)

// ---------------------------
// Small, friendly error check
// ---------------------------
static inline bool CUDA_OK(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n",
                     where, cudaGetErrorString(err));
        return false;
    }
    return true;
}

// ---------------------------
// GPU kernel: runs on the device
// Each thread sets out[tid] = 1 to say “I greeted!”
// ---------------------------
__global__ void recordGreeting(int* out, int N) {
    // Compute a unique global thread ID:
    // - threadIdx.x   : thread number inside its block (0..blockDim.x-1)
    // - blockIdx.x    : block number inside the grid  (0..gridDim.x-1)
    // - blockDim.x    : threads per block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Safety: only write if tid is a valid index (0..N-1)
    if (tid < N) {
        out[tid] = 1;  // mark that this thread ran
    }
}

int main() {
    // ---------------------------
    // 1) Decide how many threads to launch
    //    Start small so the output and logic stay simple.
    // ---------------------------
    int blocks = 2;              // number of teams
    int threads_per_block = 4;   // workers per team
    int launched_threads = blocks * threads_per_block; // total threads we ask for

    // We’ll use an array size N that matches the number of launched threads.
    // (If you change N, the kernel’s bounds check keeps things safe.)
    int N = launched_threads;

    // On the CPU, we already know what we EXPECT to count:
    // every thread writes one “1”, so expected = number of threads launched.
    int expected = launched_threads;

    // ---------------------------
    // 2) Allocate device (GPU) memory for N integers
    // ---------------------------
    int* d_out = nullptr;
    if (!CUDA_OK(cudaMalloc(&d_out, N * sizeof(int)), "cudaMalloc(d_out)")) {
        return 1;
    }

    // Optional: set device memory to 0 first.
    // After the kernel, only the threads that ran will flip their slot to 1.
    if (!CUDA_OK(cudaMemset(d_out, 0, N * sizeof(int)), "cudaMemset(d_out)")) {
        cudaFree(d_out);
        return 1;
    }

    // ---------------------------
    // 3) Launch the kernel
    // ---------------------------
    recordGreeting<<<blocks, threads_per_block>>>(d_out, N);

    // Wait for the GPU to finish; also catches runtime errors.
    if (!CUDA_OK(cudaDeviceSynchronize(), "cudaDeviceSynchronize()")) {
        cudaFree(d_out);
        return 1;
    }

    // ---------------------------
    // 4) Copy results back to the CPU
    // ---------------------------
    std::vector<int> h_out(N, 0);  // CPU-side array to receive the marks
    if (!CUDA_OK(cudaMemcpy(h_out.data(), d_out, N * sizeof(int),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(DeviceToHost)")) {
        cudaFree(d_out);
        return 1;
    }

    // ---------------------------
    // 5) CPU “reference check”: count how many 1’s we got
    // ---------------------------
    int count = 0;
    for (int i = 0; i < N; ++i) {
        count += h_out[i];  // each 1 means “one thread greeted”
    }

    // Compare with expected
    if (count == expected) {
        std::printf("PASS: count=%d  expected=%d  (blocks=%d, threads_per_block=%d)\n",
                    count, expected, blocks, threads_per_block);
    } else {
        std::printf("FAIL: count=%d  expected=%d  (blocks=%d, threads_per_block=%d)\n",
                    count, expected, blocks, threads_per_block);

        // For tiny N, it can help to show the raw values:
        std::printf("Values: ");
        for (int i = 0; i < N; ++i) std::printf("%d ", h_out[i]);
        std::printf("\n");
    }

    // ---------------------------
    // 6) Clean up GPU memory
    // ---------------------------
    cudaFree(d_out);

    // Return 0 on success, non-zero on failure (useful for scripts/tests)
    return (count == expected) ? 0 : 1;
}

/*
 * Experiments to try (one at a time):
 *
 * 1) Change the launch sizes:
 *      blocks = 3; threads_per_block = 8;   // expected = 24
 *    Rebuild and run. The PASS line should reflect the new expected count.
 *
 * 2) Make N larger than launched_threads (e.g., N = launched_threads + 4):
 *    - Only the first “launched_threads” positions can be set to 1.
 *    - The count will still match expected.
 *
 * 3) Make N smaller than launched_threads (e.g., N = launched_threads - 2):
 *    - The kernel’s (tid < N) check avoids out-of-bounds writes.
 *    - Fewer slots exist, so “count” can never exceed N.
 *    - Your expected should still be based on launched threads; compare and think:
 *        Are you testing kernel safety, or equality of counts?
 *
 * 4) Replace each “1” with the thread’s global ID and sum them,
 *    then compare against the known sum formula.
 *    This is another style of CPU reference check you’ll see later.
 */
