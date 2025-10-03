// moving_average.cu
#include <cuda_runtime.h>
#include <cstdio>

// Kernel: causal W-point moving average with edge fill.
// y[i] = mean(x[i-W+1 .. i]) for i >= W-1; otherwise y[i] = edge_fill.
__global__ void moving_average_causal(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int N, int W, float edge_fill)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Not enough past samples to form a full window: write the fill value.
    if (i < W - 1) {
        y[i] = edge_fill;
        return;
    }

    // Sum the W samples ending at i.
    float sum = 0.0f;
    int start = i - (W - 1);
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        sum += x[start + k];
    }
    y[i] = sum / static_cast<float>(W);
}

// --- minimal host-side sketch (for context) ---
void launch_moving_average(const float* d_x, float* d_y, int N, int W, float edge_fill)
{
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    moving_average_causal<<<blocks, threads>>>(d_x, d_y, N, W, edge_fill);
    cudaDeviceSynchronize();
}
