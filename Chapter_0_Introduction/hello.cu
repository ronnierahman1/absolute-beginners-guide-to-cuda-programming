#include <stdio.h>

__global__ void helloGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    helloGPU<<<1,1>>>();  
    cudaDeviceSynchronize();  
    return 0;
}
