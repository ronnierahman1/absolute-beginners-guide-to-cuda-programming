// too_many_threads_demo.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale10x5_no_check(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // No bounds check (unsafe!)
    y[i] = 10.0f * x[i] + 5.0f;
}

__global__ void scale10x5_checked(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { // ✅ Safe access
        y[i] = 10.0f * x[i] + 5.0f;
    }
}

int main() {
    const int N = 10;                  // Only 10 elements
    const int threadsPerBlock = 12;    // Launching 12 threads (overshoot)
    const int blocks = 1;

    float h_x[N], h_y[N];
    for (int i = 0; i < N; ++i) h_x[i] = float(i);  // 0,1,2,...,9

    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    printf("=== Run WITHOUT bounds check (may expect random undefined behavior) ===\n");
    scale10x5_no_check<<<blocks, threadsPerBlock>>>(d_x, d_y, N);
    cudaError_t e1 = cudaDeviceSynchronize();
    if (e1 != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(e1));
    }
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) printf("y_nochk[%d] = %f\n", i, h_y[i]);

    // Re-init output to spot differences clearly
    cudaMemset(d_y, 0, N * sizeof(float));

    printf("\n=== Run WITH bounds check (safe) ===\n");
    scale10x5_checked<<<blocks, threadsPerBlock>>>(d_x, d_y, N);
    cudaError_t e2 = cudaDeviceSynchronize();
    if (e2 != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(e2));
    }
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) printf("y_chk[%d] = %f\n", i, h_y[i]);

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
