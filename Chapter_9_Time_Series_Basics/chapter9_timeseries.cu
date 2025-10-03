// chapter9_timeseries.cu
// Chapter 9 complete listing: SMA (loop + prefix-sum), EMA, CPU references, GPU kernels,
// auto dispatch, timing, and correctness checks.

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>

// Thrust (bundled with CUDA) for inclusive scan (prefix-sum)
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "csv_utils.hpp"

// ------------------------------
// CUDA error checking helper
// ------------------------------
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(_e));            \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// ------------------------------
// CPU references
// ------------------------------

// CPU SMA (causal window): y[i] = mean(x[i-W+1 .. i]) if i >= W-1, else edge_fill.
void sma_causal_cpu(const float* x, float* y, int N, int W, float edge_fill) {
    if (W <= 0) { std::fill(y, y + N, edge_fill); return; }
    for (int i = 0; i < N; ++i) {
        if (i < W - 1) {
            y[i] = edge_fill;
        } else {
            float sum = 0.0f;
            int start = i - (W - 1);
            for (int k = 0; k < W; ++k) sum += x[start + k];
            y[i] = sum / static_cast<float>(W);
        }
    }
}

// CPU EMA: y[0] = x[0]; y[i] = alpha*x[i] + (1-alpha)*y[i-1]
void ema_cpu(const float* x, float* y, int N, float alpha) {
    if (N <= 0) return;
    y[0] = x[0];
    for (int i = 1; i < N; ++i) {
        y[i] = alpha * x[i] + (1.0f - alpha) * y[i - 1];
    }
}

// ------------------------------
// GPU SMA: per-thread window loop (best for small W)
// ------------------------------
__global__ void sma_causal_window_kernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int N, int W, float edge_fill)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (i < W - 1) {
        y[i] = edge_fill;
        return;
    }

    float sum = 0.0f;
    int start = i - (W - 1);
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        sum += x[start + k];
    }
    y[i] = sum / static_cast<float>(W);
}

// ------------------------------
// GPU SMA: prefix-sum path (best for large W)
// Steps: 1) inclusive scan, 2) O(1) per output
// ------------------------------
__global__ void sma_from_prefix_kernel(const float* __restrict__ prefix,
                                       float* __restrict__ y,
                                       int N, int W, float edge_fill)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (i < W - 1) { y[i] = edge_fill; return; }

    float prev = (i - W >= 0) ? prefix[i - W] : 0.0f;
    float s    = prefix[i] - prev;
    y[i]       = s / static_cast<float>(W);
}

// ------------------------------
// GPU EMA (sequential, correctness baseline)
// ------------------------------
__global__ void ema_sequential_kernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int N, float alpha)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (N <= 0) return;
        y[0] = x[0];
        for (int i = 1; i < N; ++i) {
            y[i] = alpha * x[i] + (1.0f - alpha) * y[i - 1];
        }
    }
}

// ------------------------------
// Dispatcher for SMA (auto choose path by W)
// ------------------------------
void sma_causal_gpu_auto(const float* d_x, float* d_y,
                         int N, int W, float edge_fill,
                         int switch_threshold_W = 64,
                         float* d_tmp_prefix = nullptr)
{
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    if (W <= switch_threshold_W) {
        // Small windows → direct window loop
        sma_causal_window_kernel<<<blocks, threads>>>(d_x, d_y, N, W, edge_fill);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Large windows → prefix-sum
    float* d_prefix = d_tmp_prefix;
    bool   owner    = false;
    if (!d_prefix) { CUDA_CHECK(cudaMalloc(&d_prefix, N * sizeof(float))); owner = true; }

    thrust::device_ptr<const float> in(d_x);
    thrust::device_ptr<float>       out(d_prefix);
    thrust::inclusive_scan(in, in + N, out);

    sma_from_prefix_kernel<<<blocks, threads>>>(d_prefix, d_y, N, W, edge_fill);
    CUDA_CHECK(cudaGetLastError());

    if (owner) CUDA_CHECK(cudaFree(d_prefix));
}

// ------------------------------
// Comparison helpers
// ------------------------------
struct CompareResult {
    double max_abs_err;
    int    max_err_index;
    int    mismatches; // count of elements beyond tolerance
};

CompareResult compare_arrays_range(const float* a, const float* b,
                                   int start, int end, float atol)
{
    CompareResult r{0.0, -1, 0};
    for (int i = start; i < end; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > r.max_abs_err) {
            r.max_abs_err = diff;
            r.max_err_index = i;
        }
        if (diff > atol) r.mismatches++;
    }
    return r;
}

// ------------------------------
// Pretty print small sample
// ------------------------------
void print_sample(const char* label, const std::vector<float>& x,
                  const std::vector<float>& y1,
                  const std::vector<float>& y2, int count = 8)
{
    std::printf("\n%s (first %d elements):\n", label, count);
    for (int i = 0; i < std::min<int>(count, (int)x.size()); ++i) {
        std::printf("i=%d  x=% .5f  y1=% .5f  y2=% .5f\n",
                    i, x[i], y1[i], y2[i]);
    }
}

// ---- CSV mode shim (drop above main). Requires: csv_utils.hpp in include path. ----

struct CsvArgs {
    bool csv_mode = false;
    std::string csv_in, csv_out;
    int col = 1;           // 1-based column index
    std::string op = "sma"; // "sma" or "ema"
    int W = 5;             // for SMA
    float alpha = 0.2f;    // for EMA
    float edge_fill = 0.0f;
    int threshold = 64;    // SMA switch point
};

inline CsvArgs parse_csv_args(int argc, char** argv) {
    CsvArgs a;
    for (int i=1; i<argc; ++i) {
        std::string s = argv[i];
        auto need = [&](){ if(i+1>=argc) throw std::runtime_error("Missing value for "+s); return std::string(argv[++i]); };
        if (s=="--csv")      { a.csv_mode=true; a.csv_in=need(); }
        else if (s=="--col") { a.col=std::stoi(need()); }
        else if (s=="--op")  { a.op=need(); }
        else if (s=="--win") { a.W=std::stoi(need()); }
        else if (s=="--alpha"){ a.alpha=std::stof(need()); }
        else if (s=="--edge"){ a.edge_fill = (need()=="nan" ? std::numeric_limits<float>::quiet_NaN() : std::stof(argv[i])); }
        else if (s=="--out") { a.csv_out=need(); }
        else if (s=="--thresh"){ a.threshold=std::stoi(need()); }
    }
    return a;
}

// Runs either SMA or EMA on CSV column and writes CSV output (index,value).
inline int run_csv_pipeline(const CsvArgs& a) {
    using namespace tinycsv;
    if (a.csv_in.empty())  throw std::runtime_error("CSV mode requires --csv <file>");
    if (a.csv_out.empty()) throw std::runtime_error("CSV mode requires --out <file>");
    std::vector<float> x = read_csv_column(a.csv_in, a.col, /*header_row=*/true);
    if (x.empty()) throw std::runtime_error("No numeric data parsed from: " + a.csv_in);

    const int N = (int)x.size();
    std::vector<float> y(N);

    if (a.op=="sma") {
        // Allocate device, run your existing auto-dispatcher
        float *d_x=nullptr, *d_y=nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice));

        sma_causal_gpu_auto(d_x, d_y, N, a.W, a.edge_fill, a.threshold);

        CUDA_CHECK(cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));

        write_csv_indexed(a.csv_out, "index", "sma", y);
    } else if (a.op=="ema") {
        // Use CPU reference or the sequential GPU kernel; CPU is fine and fast for single series.
        ema_cpu(x.data(), y.data(), N, a.alpha);
        write_csv_indexed(a.csv_out, "index", "ema", y);
    } else {
        throw std::runtime_error("Unknown --op (use 'sma' or 'ema')");
    }

    std::printf("CSV OK: %s -> %s  (op=%s, W=%d, alpha=%.4f)\n",
                a.csv_in.c_str(), a.csv_out.c_str(), a.op.c_str(), a.W, a.alpha);
    return 0;
}


// ------------------------------
// Main: full test harness
// Args: N W edge_fill alpha switch_threshold
// ------------------------------
int main(int argc, char** argv)
{
    // CSV mode takes precedence if --csv is present.
    try {
        CsvArgs csv = parse_csv_args(argc, argv);
        if (csv.csv_mode) {
            return run_csv_pipeline(csv);
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "CSV mode error: %s\n", ex.what());
        return 2;
    }
    
    // Defaults
    int   N         = (1 << 20); // 1,048,576
    int   W         = 5;
    float edge_fill = 0.0f;      // can be NAN to mark edges
    float alpha     = 0.2f;      // EMA smoothing factor
    int   threshold = 64;        // switch SMA path at W <= threshold

    if (argc >= 2) N         = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) W         = std::max(1, std::atoi(argv[2]));
    if (argc >= 4) {
        if (std::strcmp(argv[3], "nan") == 0 || std::strcmp(argv[3], "NaN") == 0) {
            edge_fill = std::numeric_limits<float>::quiet_NaN();
        } else {
            edge_fill = std::atof(argv[3]);
        }
    }
    if (argc >= 5) alpha     = std::atof(argv[4]);
    if (argc >= 6) threshold = std::max(1, std::atoi(argv[5]));

    std::printf("N=%d, W=%d, edge_fill=%g, alpha=%.4f, switch@W<=%d → window else prefix\n",
                N, W, edge_fill, alpha, threshold);

    // --------------------------
    // Generate input on host
    // --------------------------
    std::vector<float> h_x(N);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < N; ++i) h_x[i] = dist(rng);

    // Host outputs
    std::vector<float> h_sma_cpu(N), h_sma_gpu(N);
    std::vector<float> h_ema_cpu(N), h_ema_gpu(N);

    // --------------------------
    // CPU references + timing
    // --------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    sma_causal_cpu(h_x.data(), h_sma_cpu.data(), N, W, edge_fill);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_sma_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    ema_cpu(h_x.data(), h_ema_cpu.data(), N, alpha);
    auto t3 = std::chrono::high_resolution_clock::now();
    double cpu_ema_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // --------------------------
    // Device alloc / copy
    // --------------------------
    float *d_x = nullptr, *d_sma = nullptr, *d_ema = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sma, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ema, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // --------------------------
    // GPU SMA (auto) timing
    // --------------------------
    cudaEvent_t ev_sma_start, ev_sma_stop;
    CUDA_CHECK(cudaEventCreate(&ev_sma_start));
    CUDA_CHECK(cudaEventCreate(&ev_sma_stop));

    CUDA_CHECK(cudaEventRecord(ev_sma_start));
    sma_causal_gpu_auto(d_x, d_sma, N, W, edge_fill, threshold);
    CUDA_CHECK(cudaEventRecord(ev_sma_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_sma_stop));
    CUDA_CHECK(cudaGetLastError());

    float gpu_sma_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_sma_ms, ev_sma_start, ev_sma_stop));
    CUDA_CHECK(cudaMemcpy(h_sma_gpu.data(), d_sma, N * sizeof(float), cudaMemcpyDeviceToHost));

    // --------------------------
    // GPU EMA (sequential) timing
    // --------------------------
    cudaEvent_t ev_ema_start, ev_ema_stop;
    CUDA_CHECK(cudaEventCreate(&ev_ema_start));
    CUDA_CHECK(cudaEventCreate(&ev_ema_stop));

    CUDA_CHECK(cudaEventRecord(ev_ema_start));
    ema_sequential_kernel<<<1, 1>>>(d_x, d_ema, N, alpha);
    CUDA_CHECK(cudaEventRecord(ev_ema_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_ema_stop));
    CUDA_CHECK(cudaGetLastError());

    float gpu_ema_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ema_ms, ev_ema_start, ev_ema_stop));
    CUDA_CHECK(cudaMemcpy(h_ema_gpu.data(), d_ema, N * sizeof(float), cudaMemcpyDeviceToHost));

    // --------------------------
    // Compare (skip early indices if NaN edge_fill)
    // --------------------------
    const float atol_sma = 1e-5f;
    const float atol_ema = 1e-4f; // EMA accumulates rounding more
    int start_sma = (std::isnan(edge_fill) ? std::min(W - 1, N) : 0);

    CompareResult cr_sma = compare_arrays_range(h_sma_cpu.data(), h_sma_gpu.data(),
                                                start_sma, N, atol_sma);
    CompareResult cr_ema = compare_arrays_range(h_ema_cpu.data(), h_ema_gpu.data(),
                                                0, N, atol_ema);

    // --------------------------
    // Print samples & timings
    // --------------------------
    print_sample("SMA CPU vs GPU", h_x, h_sma_cpu, h_sma_gpu, 8);
    print_sample("EMA CPU vs GPU", h_x, h_ema_cpu, h_ema_gpu, 8);

    std::printf("\nTimings (ms):\n");
    std::printf("  CPU SMA: %.3f | GPU SMA: %.3f  (includes scan if used)\n", cpu_sma_ms, gpu_sma_ms);
    std::printf("  CPU EMA: %.3f | GPU EMA (seq): %.3f\n", cpu_ema_ms, gpu_ema_ms);

    std::printf("\nSMA Max |CPU-GPU| error: %.6g at index %d | mismatches(>%.1e): %d (compared [%d,%d))\n",
                cr_sma.max_abs_err, cr_sma.max_err_index, cr_sma.mismatches, atol_sma, start_sma, N);
    std::printf("EMA Max |CPU-GPU| error: %.6g at index %d | mismatches(>%.1e): %d\n",
                cr_ema.max_abs_err, cr_ema.max_err_index, atol_ema, cr_ema.mismatches);

    bool ok_sma = (cr_sma.mismatches == 0);
    bool ok_ema = (cr_ema.mismatches == 0);
    std::printf("\nRESULTS: SMA %s | EMA %s\n",
                ok_sma ? "PASS ✅" : "FAIL ❌",
                ok_ema ? "PASS ✅" : "FAIL ❌");

    // --------------------------
    // Cleanup
    // --------------------------
    CUDA_CHECK(cudaEventDestroy(ev_sma_start));
    CUDA_CHECK(cudaEventDestroy(ev_sma_stop));
    CUDA_CHECK(cudaEventDestroy(ev_ema_start));
    CUDA_CHECK(cudaEventDestroy(ev_ema_stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_sma));
    CUDA_CHECK(cudaFree(d_ema));

    return (ok_sma && ok_ema) ? 0 : 1;
}
