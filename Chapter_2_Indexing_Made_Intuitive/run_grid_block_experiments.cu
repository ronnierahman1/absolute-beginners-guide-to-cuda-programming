// run_grid_block_experiments.cu
//
// Purpose:
//   Explore how different grid/block configurations map threads to data for the
//   operation y[i] = 10 * x[i] + 5. This program helps you *see* how the global
//   index is computed, when oversubscription happens, and why the bounds check
//   keeps everything safe.
//
// What this file demonstrates (skills from Chapter 2):
//   • Global index calculation: i = blockIdx.x * blockDim.x + threadIdx.x
//   • One-thread-per-element mapping, protected by if (i < N)
//   • Rounding up blocks: blocks = (N + TPB - 1) / TPB
//   • Running multiple (N, TPB, blocks) experiments in one executable
//   • Identifying undersubscription (too few threads) vs oversubscription
//   • Verifying correctness against a CPU reference
//
// Build:
//   nvcc -O2 run_grid_block_experiments.cu -o grid_expts
//
// Run (default experiments):
//   ./grid_expts
//
// Run with a custom N and TPB list (comma-separated), optional blocks override:
//   ./grid_expts 37 32,64,128
//   ./grid_expts 17 4,8,16  // auto-computes blocks for each TPB
//
// Advanced: pin blocks explicitly (uses the same blocks for every TPB in the list):
//   ./grid_expts 17 4,8,16 3
//
// Notes:
//   • For each configuration, we print a coverage summary and do a CPU vs GPU check.
//   • If totalThreads < N (undersubscription), some elements won’t be computed;
//     the CPU/GPU comparison will FAIL (useful to *see* why bounds check alone
//     doesn’t fix too few threads).
//   • If totalThreads >= N (oversubscription), extra threads are safely ignored
//     by the bounds check, and CPU/GPU should PASS.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>

// -------------------------------------
// CUDA error checking convenience macro
// -------------------------------------
#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t _err = (call);                                                     \
        if (_err != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_err));              \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

// -------------------------------------
// CPU reference: y[i] = 10*x[i] + 5
// -------------------------------------
void scale10x5_cpu(const float* x, float* y, int N) {
    for (int i = 0; i < N; ++i) {
        y[i] = 10.0f * x[i] + 5.0f;
    }
}

// -------------------------------------
// GPU kernel: one thread per element,
// protected by a bounds check.
// -------------------------------------
__global__ void scale10x5_gpu(const float* __restrict__ x,
                              float* __restrict__ y,
                              int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = 10.0f * x[i] + 5.0f;
    }
}

// -------------------------------------
// Compare arrays with tolerance.
// Returns true on PASS, false on FAIL.
// -------------------------------------
bool compare_results(const float* a,
                     const float* b,
                     int N,
                     float tol = 1e-5f,
                     int max_report = 10)
{
    bool pass = true;
    int reported = 0;        // how many we printed
    int total_mismatches = 0; // how many we actually saw

    for (int i = 0; i < N; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tol) {
            ++total_mismatches;
            if (reported < max_report) {
                printf("Mismatch at %d: CPU=%f, GPU=%f, |diff|=%f\n",
                       i, a[i], b[i], diff);
                ++reported;
            }
            pass = false;
        }
    }

    // Only print the "additional mismatches" line if we *know*
    // there were more mismatches than we reported.
    if (!pass && total_mismatches > reported) {
        printf("... additional mismatches not shown (reported %d of %d; limit %d)\n",
               reported, total_mismatches, max_report);
    }

    return pass;
}

// -------------------------------------
// Small helper to split "32,64,128" -> {32,64,128}
// -------------------------------------
std::vector<int> parse_csv_ints(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
    }
    return out;
}

// -------------------------------------
// Print a short banner explaining coverage
// -------------------------------------
void print_coverage_summary(int N, int TPB, int blocks) {
    long long totalThreads = static_cast<long long>(TPB) * blocks;
    printf("Config: N=%d, TPB=%d, blocks=%d  -> totalThreads=%lld\n",
           N, TPB, blocks, totalThreads);

    if (totalThreads < N) {
        printf("  Coverage: UNDERSUBSCRIPTION (missing %lld threads)\n",
               (long long)N - totalThreads);
        printf("  Expectation: Some elements remain uncomputed -> CPU vs GPU will FAIL.\n");
    } else if (totalThreads == N) {
        printf("  Coverage: EXACT FIT\n");
        printf("  Expectation: All elements covered exactly once -> CPU vs GPU should PASS.\n");
    } else {
        printf("  Coverage: OVERSUBSCRIPTION (overshoot %lld threads)\n",
               totalThreads - (long long)N);
        printf("  Expectation: Extra threads safely skip work via bounds check -> PASS.\n");
    }
}

// -------------------------------------
// Run one experiment and report:
//   • Coverage summary
//   • CPU vs GPU comparison
//   • First few elements for intuition
// -------------------------------------
void run_experiment(int N, int TPB, int blocks, bool auto_blocks) {
    // If auto mode requested, compute blocks to cover N
    if (auto_blocks) {
        blocks = (N + TPB - 1) / TPB; // round up
    }

    print_coverage_summary(N, TPB, blocks);

    // Host buffers
    std::vector<float> h_x(N), h_y_cpu(N), h_y_gpu(N, 0.0f);
    for (int i = 0; i < N; ++i) h_x[i] = static_cast<float>(i);

    // CPU reference
    scale10x5_cpu(h_x.data(), h_y_cpu.data(), N);

    // Device buffers
    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));

    // H2D
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch
    scale10x5_gpu<<<blocks, TPB>>>(d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // D2H
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    bool pass = compare_results(h_y_cpu.data(), h_y_gpu.data(), N);
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");

    // Show first few elements (helps visualizing mapping)
    int to_show = (N < 10) ? N : 10;
    printf("  First %d elements (CPU vs GPU vs expected):\n", to_show);
    for (int i = 0; i < to_show; ++i) {
        float expected = 10.0f * h_x[i] + 5.0f;
        printf("    i=%d: CPU=%8.3f  GPU=%8.3f  exp=%8.3f\n",
               i, h_y_cpu[i], h_y_gpu[i], expected);
    }
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// -------------------------------------
// Program entry
// -------------------------------------
int main(int argc, char** argv) {
    // Defaults chosen to illustrate a mix of exact fit, under-, and over-subscription.
    int N = 37;                                 // awkward size on purpose
    std::vector<int> TPB_list = { 4, 8, 16, 32, 64, 128 };
    int fixed_blocks = -1;                      // -1 means "auto" per TPB

    // Parse CLI:
    //   argv[1] -> N
    //   argv[2] -> TPB csv (e.g., "32,64,128")
    //   argv[3] -> fixed blocks (optional). If provided, overrides auto for all TPBs.
    if (argc >= 2) {
        N = std::atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "N must be > 0\n");
            return EXIT_FAILURE;
        }
    }
    if (argc >= 3) {
        TPB_list = parse_csv_ints(argv[2]);
        if (TPB_list.empty()) {
            fprintf(stderr, "Invalid TPB list. Example: 32,64,128\n");
            return EXIT_FAILURE;
        }
    }
    if (argc >= 4) {
        fixed_blocks = std::atoi(argv[3]);
        if (fixed_blocks <= 0) {
            fprintf(stderr, "blocks must be > 0 when specified\n");
            return EXIT_FAILURE;
        }
    }

    printf("=== Grid/Block Experiments: y = 10*x + 5 ===\n");
    printf("N=%d\n", N);
    printf("TPB list: ");
    for (size_t k = 0; k < TPB_list.size(); ++k) {
        printf("%d%s", TPB_list[k], (k + 1 == TPB_list.size()) ? "" : ",");
    }
    printf("\n");
    if (fixed_blocks > 0) {
        printf("Using FIXED blocks=%d for all TPBs\n\n", fixed_blocks);
    } else {
        printf("Using AUTO blocks=(N+TPB-1)/TPB for each TPB\n\n");
    }

    // Run each experiment
    for (int TPB : TPB_list) {
        bool auto_blocks = (fixed_blocks <= 0);
        int blocks = auto_blocks ? 0 : fixed_blocks;
        run_experiment(N, TPB, blocks, auto_blocks);
    }

    // Extra teaching: demonstrate an intentional UNDERSUBSCRIPTION scenario
    // for the first TPB (if auto blocks would normally cover N).
    if (!TPB_list.empty()) {
        int TPB = TPB_list.front();
        // Intentionally launch one fewer block than needed to cover all elements (undersubscription),
        // but ensure at least one block is launched (avoiding zero blocks).
        int too_few_blocks = std::max(1, (N + TPB - 1) / TPB - 1);
        printf("=== Intentional UNDERSUBSCRIPTION demo ===\n");
        run_experiment(N, TPB, too_few_blocks, /*auto_blocks=*/false);
    }

    printf("Done.\n");
    return 0;
}

