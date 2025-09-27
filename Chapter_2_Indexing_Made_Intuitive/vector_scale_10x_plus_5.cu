// vector_scale_10x_plus_5.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale10x5(const float* __restrict__ x,
                          float* __restrict__ y,
                          int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {           // bounds guard
        y[i] = 10.0f * x[i] + 5.0f;
    }
}

int main() {
    const int N = 10;
    const int TPB = 8;                           // threads per block
    const int blocks = (N + TPB - 1) / TPB;      // enough blocks to cover N

    // Host data
    float h_x[N], h_y[N];
    for (int i = 0; i < N; ++i) h_x[i] = float(i); // 0..9

    // Device buffers
    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // H2D copy
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch: one thread per element (guarded)
    scale10x5<<<blocks, TPB>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    // D2H copy and check
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few values
    for (int i = 0; i < N; ++i) {
        printf("y[%d] = %5.1f  (expect %5.1f)\n", i, h_y[i], 10.0f * h_x[i] + 5.0f);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
