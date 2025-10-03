/*
 * hello.cu
 *
 * A minimal CUDA "Hello, World!" program.
 * - Launches a single GPU thread that prints a message.
 * - Keeps everything as small and readable as possible.
 *
 * Requirements
 *  - NVIDIA GPU with CUDA support
 *  - CUDA Toolkit installed (so you have `nvcc`)
 *
 * Build
 *   nvcc hello.cu -o hello
 *
 * Run
 *   ./hello
 *
 * Expected output
 *   Hello, GPU!
 *
 * Notes
 *  - The call to cudaDeviceSynchronize() makes sure the GPU finishes
 *    and flushes its printf buffer before the program exits.
 *  - If you want to see basic parallelism (still simple), change the
 *    launch to <<<1, 4>>> and print threadIdx.x inside the kernel.
 */

#include <cstdio>           // for printf / fprintf
#include <cuda_runtime.h>   // for cudaDeviceSynchronize, cudaGetErrorString

// __global__ marks a function as a GPU kernel.
// This kernel does exactly one thing: print a short message.
__global__ void hello() {
    printf("Hello, GPU!\n");
}

int main() {
    // Launch 1 block with 1 thread (the absolute simplest launch).
    hello<<<1, 1>>>();

    // Wait for the GPU to finish so the message appears before we exit.
    // Also report any runtime error in a friendly way.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0; // success
}

/*
 * Optional: Tiny variation to show a few parallel "hellos" (uncomment to try)
 *
 * __global__ void helloMany() {
 *     printf("Hello from thread %d\n", threadIdx.x);
 * }
 *
 * int main() {
 *     helloMany<<<1, 4>>>();
 *     cudaError_t err = cudaDeviceSynchronize();
 *     if (err != cudaSuccess) {
 *         std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
 *         return 1;
 *     }
 *     return 0;
 * }
 */
