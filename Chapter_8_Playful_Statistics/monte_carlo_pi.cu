// monte_carlo_pi.cu
//
// Monte Carlo π estimation on the GPU.
// Each thread generates random (x, y) points in [-1, 1] × [-1, 1],
// counts how many land inside the unit circle, and atomically
// contributes its local tally to a global counter.
//
// Build (typical):
//   nvcc -O3 -arch=sm_60 -o monte_carlo_pi monte_carlo_pi.cu
// (Adjust -arch to your GPU, e.g., sm_53 for Jetson Nano, sm_86 for RTX 30-series.)
//
// Run:
//   ./monte_carlo_pi [total_samples] [threads_per_block]
// Examples:
//   ./monte_carlo_pi            # uses defaults
//   ./monte_carlo_pi 100000000  # 1e8 samples
//   ./monte_carlo_pi 100000000 256
//
// Notes:
// - Uses cuRAND Philox per-thread states (statistically solid, counter-based).
// - Accumulates locally per thread, then one atomicAdd per thread (amortized).
// - 64-bit counters are used to safely handle very large sample counts.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ------------------------- Error checking helper -------------------------
#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA ERROR %s:%d: %s\n",                         \
                     __FILE__, __LINE__, cudaGetErrorString(_e));              \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

// ------------------------- Kernel configuration -------------------------
static inline int div_up(long long n, int d) {
    return static_cast<int>((n + d - 1) / d);
}

// ------------------------- RNG + simulation kernel -------------------------
__global__ void monte_carlo_pi_kernel(
    unsigned long long *global_hits,
    unsigned long long samples_per_thread,
    unsigned long long base_seed // host-chosen seed for reproducibility
) {
    const unsigned int tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize a per-thread Philox RNG state.
    // sequence = tid ensures unique stream per thread.
    curandStatePhilox4_32_10_t rng;
    curand_init(/*seed=*/base_seed,
                /*sequence=*/static_cast<unsigned long long>(tid),
                /*offset=*/0ULL,
                &rng);

    unsigned long long local_hits = 0ULL;

    for (unsigned long long k = 0; k < samples_per_thread; ++k) {
        // curand_uniform() returns (0, 1]; map to [-1, 1]
        float x = 2.0f * curand_uniform(&rng) - 1.0f;
        float y = 2.0f * curand_uniform(&rng) - 1.0f;

        // Inside circle test without sqrt
        float r2 = x * x + y * y;
        if (r2 <= 1.0f) {
            ++local_hits;
        }
    }

    // One atomic update per thread (amortized synchronization cost)
    atomicAdd(global_hits, local_hits);
}

// ------------------------- Host driver -------------------------
int main(int argc, char** argv) {
    // --- Tunables / defaults ---
    // Total samples (darts). Increase for better accuracy.
    unsigned long long total_samples = 50ULL * 1000ULL * 1000ULL; // 5e7 by default
    int threads_per_block = 256;
    // A fixed base seed makes runs reproducible; change for different sequences.
    unsigned long long base_seed = 123456789ULL;

    if (argc >= 2) {
        total_samples = std::strtoull(argv[1], nullptr, 10);
        if (total_samples == 0ULL) {
            std::fprintf(stderr, "Invalid total_samples.\n");
            return EXIT_FAILURE;
        }
    }
    if (argc >= 3) {
        threads_per_block = std::atoi(argv[2]);
        if (threads_per_block <= 0) {
            std::fprintf(stderr, "Invalid threads_per_block.\n");
            return EXIT_FAILURE;
        }
    }

    // Choose a grid size to get plenty of threads.
    // Heuristic: a few times the number of SMs is fine; we’ll just compute
    // samples_per_thread to match total_samples as closely as possible.
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;

    // Aim for ~ (sms * 2048) threads (rule of thumb for decent occupancy),
    // but clamp reasonably.
    long long target_threads = static_cast<long long>(sms) * 2048LL;
    if (target_threads < 8192)  target_threads = 8192;
    if (target_threads > 1LL << 22) target_threads = 1LL << 22; // safety clamp

    int blocks = div_up(static_cast<long long>(target_threads), threads_per_block);
    long long total_threads = static_cast<long long>(blocks) * threads_per_block;

    // Each thread will do this many samples. We round up to cover total_samples.
    unsigned long long samples_per_thread =
        static_cast<unsigned long long>((total_samples + total_threads - 1) / total_threads);

    // The actual total performed (may be >= requested because of ceiling above)
    unsigned long long actual_total_samples =
        static_cast<unsigned long long>(total_threads) * samples_per_thread;

    std::printf("GPU: %s  SMs=%d\n", prop.name, sms);
    std::printf("Config: blocks=%d, threads_per_block=%d, total_threads=%lld\n",
                blocks, threads_per_block, total_threads);
    std::printf("Requested samples=%llu, actual samples=%llu, samples/thread=%llu\n",
                (unsigned long long)total_samples,
                (unsigned long long)actual_total_samples,
                (unsigned long long)samples_per_thread);

    // Device allocations
    unsigned long long *d_hits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hits, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hits, 0, sizeof(unsigned long long)));

    // Optional: simple timing using CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Launch
    monte_carlo_pi_kernel<<<blocks, threads_per_block>>>(
        d_hits, samples_per_thread, base_seed
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy result back
    unsigned long long hits_host = 0ULL;
    CUDA_CHECK(cudaMemcpy(&hits_host, d_hits, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    // Compute π estimate
    double pi_est = 4.0 * static_cast<long double>(hits_host)
                          / static_cast<long double>(actual_total_samples);

    // Report
    std::printf("Hits inside circle = %llu\n", (unsigned long long)hits_host);
    std::printf("Estimated pi       = %.10f\n", pi_est);
    std::printf("Reference pi       = 3.1415926536\n");
    std::printf("Absolute error     = %.10f\n", std::fabs(pi_est - 3.14159265358979323846));
    std::printf("Kernel time        = %.3f ms\n", ms);

    // Cleanup
    CUDA_CHECK(cudaFree(d_hits));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
