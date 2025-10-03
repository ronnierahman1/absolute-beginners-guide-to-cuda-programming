// poly_46_testing.cu
//
// Section 4.6 — Testing with Known Polynomials
// -----------------------------------------------------------
// What this program does:
//   • Builds several known polynomials with trustworthy properties:
//       (1) Constant    P(x) = 7
//       (2) Linear      P(x) = 3x - 2
//       (3) Rooted      P(x) = (x-1)(x-3) = x^2 - 4x + 3 (zeros at x=1,3)
//       (4) Binomial    P(x) = (1 + x)^n   (coeffs via Pascal’s rule)
//       (5) Chebyshev   T_n(x) with T_n(cos θ) = cos(n θ)  (coeffs built recursively)
//   • Evaluates each on CPU (double, Horner) for reference.
//   • Evaluates each on GPU, in both float and double, using Horner.
//   • Reports max |error| and max relative error vs CPU double.
//   • Prints basic timings (CPU vs GPU kernel) for a feel of speed.
//
// Build:
//   nvcc -O3 poly_46_testing.cu -o poly46
//
// Run:
//   ./poly46
//
// Notes:
//   • Uses a pointer-based kernel (coeffs in global memory) for simplicity.
//   • Reuse patterns from 4.3–4.5 (Horner, FMA, coalesced xs/ys).
//   • Keep degrees moderate if running on constant-memory examples elsewhere.
//
// -----------------------------------------------------------

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t _err = (call);                                  \
    if (_err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s at %s:%d\n",             \
                cudaGetErrorString(_err), __FILE__, __LINE__);  \
        std::exit(1);                                           \
    }                                                           \
} while(0)

// ----------------------------- CPU reference (double, Horner) -----------------------------
static inline double horner_ref(const std::vector<double>& a, int deg, double x) {
    double r = a[deg];
    for (int k = deg - 1; k >= 0; --k) r = std::fma(r, x, a[k]); // FMA for consistency
    return r;
}

// Error helpers
static inline double abs_err(double y, double yref) { return std::fabs(y - yref); }
static inline double rel_err(double y, double yref) {
    double denom = std::max(1.0, std::fabs(yref));
    return std::fabs(y - yref) / denom;
}

// ----------------------------- CUDA kernels (templated) -----------------------------
template <typename T>
__global__ void eval_poly_ptr(const T* __restrict__ xs,
                              T*       __restrict__ ys,
                              const T* __restrict__ coeffs, // a[0..deg]
                              int n, int deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    T x   = xs[i];
    T acc = coeffs[deg];

    #pragma unroll
    for (int k = deg - 1; k >= 0; --k) {
        // std::fma has device overloads for float/double; maps to FMA
        acc = fma(acc, x, coeffs[k]);
    }
    ys[i] = acc;
}

// ----------------------------- Polynomial builders (coeffs in a[0..deg]) -----------------------------
// From roots r_j: P(x) = Π_j (x - r_j)
static std::vector<double> poly_from_roots(const std::vector<double>& roots) {
    std::vector<double> a(1, 1.0); // start with P(x)=1
    for (double r : roots) {
        std::vector<double> next(a.size() + 1, 0.0);
        // multiply by (x - r): next[k+1] += a[k]; next[k] += -r*a[k]
        for (size_t k = 0; k < a.size(); ++k) {
            next[k]     += -r * a[k];
            next[k + 1] += a[k];
        }
        a.swap(next);
    }
    return a; // size = deg+1
}

// Binomial: (1 + x)^n  using Pascal rule
static std::vector<double> binomial_1px_pow(int n) {
    std::vector<double> a(n + 1, 0.0);
    a[0] = 1.0;
    for (int i = 1; i <= n; ++i) {
        // update from high to low to avoid overwriting needed values
        for (int k = i; k >= 1; --k) a[k] += a[k - 1];
        // a[0] stays 1
    }
    return a;
}

// Chebyshev T_n coefficients via recurrence:
//   T_0 = 1
//   T_1 = x
//   T_{k+1} = 2 x T_k - T_{k-1}
static std::vector<double> chebyshev_T(int n) {
    if (n == 0) return std::vector<double>{1.0};
    if (n == 1) return std::vector<double>{0.0, 1.0}; // 0 + 1*x

    auto mul_x = [](const std::vector<double>& p) {
        std::vector<double> q(p.size() + 1, 0.0);
        for (size_t i = 0; i < p.size(); ++i) q[i + 1] = p[i];
        return q;
    };

    std::vector<double> Tkm1{1.0};              // T_0
    std::vector<double> Tk  {0.0, 1.0};         // T_1

    for (int k = 1; k < n; ++k) {
        auto two_x_Tk = mul_x(Tk);
        for (double &c : two_x_Tk) c *= 2.0;

        // T_{k+1} = two_x_Tk - T_{k-1}
        size_t m = std::max(two_x_Tk.size(), Tkm1.size());
        std::vector<double> Tkp1(m, 0.0);
        for (size_t i = 0; i < m; ++i) {
            double a = (i < two_x_Tk.size() ? two_x_Tk[i] : 0.0);
            double b = (i < Tkm1.size()     ? Tkm1[i]     : 0.0);
            Tkp1[i] = a - b;
        }

        Tkm1.swap(Tk);
        Tk.swap(Tkp1);
    }
    return Tk; // size = n+1
}

// ----------------------------- Test runner -----------------------------
struct TestResult {
    std::string name;
    int deg;
    int N;
    double max_abs_float, max_rel_float, ms_gpu_float;
    double max_abs_double, max_rel_double, ms_gpu_double;
    double ms_cpu_ref;
};

template <typename T>
static void to_device_buffer(const std::vector<double>& host, T** dptr) {
    std::vector<T> tmp(host.size());
    for (size_t i = 0; i < host.size(); ++i) tmp[i] = static_cast<T>(host[i]);
    CUDA_CHECK(cudaMalloc(dptr, tmp.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(*dptr, tmp.data(), tmp.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
static void to_device_array(const std::vector<T>& host, T** dptr) {
    CUDA_CHECK(cudaMalloc(dptr, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(*dptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
}

// Launch + time helper (returns ms)
template <typename T>
static float launch_eval_ptr_ms(const std::vector<double>& coeffs_dbl,
                                const std::vector<double>& xs_dbl,
                                std::vector<T>& ys_host,
                                int deg)
{
    int N = static_cast<int>(xs_dbl.size());
    // Build typed copies on device
    T *d_x = nullptr, *d_y = nullptr, *d_a = nullptr;
    {
        std::vector<T> a_t(coeffs_dbl.size());
        for (size_t i = 0; i < coeffs_dbl.size(); ++i) a_t[i] = static_cast<T>(coeffs_dbl[i]);
        std::vector<T> x_t(N);
        for (int i = 0; i < N; ++i) x_t[i] = static_cast<T>(xs_dbl[i]);
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_a, (deg + 1) * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_x, x_t.data(), N * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a, a_t.data(), (deg + 1) * sizeof(T), cudaMemcpyHostToDevice));
    }

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // Time with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    eval_poly_ptr<T><<<blocks, threads>>>(d_x, d_y, d_a, N, deg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    ys_host.resize(N);
    CUDA_CHECK(cudaMemcpy(ys_host.data(), d_y, N * sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_a));
    return ms;
}

// Run a single named test, fill result row
static TestResult run_named_test(const std::string& name,
                                 const std::vector<double>& coeffs,
                                 const std::vector<double>& xs)
{
    const int deg = static_cast<int>(coeffs.size()) - 1;
    const int N   = static_cast<int>(xs.size());

    // CPU reference timing
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> yref(N);
    for (int i = 0; i < N; ++i) yref[i] = horner_ref(coeffs, deg, xs[i]);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU float
    std::vector<float>  y_float;
    float ms_gpu_f = launch_eval_ptr_ms<float>(coeffs, xs, y_float, deg);

    // GPU double
    std::vector<double> y_double;
    float ms_gpu_d = launch_eval_ptr_ms<double>(coeffs, xs, y_double, deg);

    // Errors
    double max_abs_f = 0.0, max_rel_f = 0.0;
    double max_abs_d = 0.0, max_rel_d = 0.0;

    for (int i = 0; i < N; ++i) {
        double yf = static_cast<double>(y_float[i]);
        double yd = y_double[i];
        double yr = yref[i];

        max_abs_f = std::max(max_abs_f, abs_err(yf, yr));
        max_rel_f = std::max(max_rel_f, rel_err(yf, yr));
        max_abs_d = std::max(max_abs_d, abs_err(yd, yr));
        max_rel_d = std::max(max_rel_d, rel_err(yd, yr));
    }

    TestResult r;
    r.name = name; r.deg = deg; r.N = N;
    r.max_abs_float = max_abs_f; r.max_rel_float = max_rel_f; r.ms_gpu_float = ms_gpu_f;
    r.max_abs_double = max_abs_d; r.max_rel_double = max_rel_d; r.ms_gpu_double = ms_gpu_d;
    r.ms_cpu_ref = ms_cpu;
    return r;
}

// Pretty print one row
static void print_row(const TestResult& r) {
    printf("%-12s  deg=%2d  N=%7d | CPU %.2f ms | GPUf %.2f ms | GPUd %.2f ms | "
           "maxAbs(f)=%.3e  maxRel(f)=%.3e | maxAbs(d)=%.3e  maxRel(d)=%.3e\n",
           r.name.c_str(), r.deg, r.N,
           r.ms_cpu_ref, r.ms_gpu_float, r.ms_gpu_double,
           r.max_abs_float, r.max_rel_float,
           r.max_abs_double, r.max_rel_double);
}

int main() {
    // Problem sizes (adjust to taste)
    const int N = 1 << 16; // 65,536 sample points per test

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> Ux(-2.0, 2.0);

    // -------------- Build input x arrays --------------
    std::vector<double> xs(N);
    for (int i = 0; i < N; ++i) xs[i] = Ux(rng);

    // For Chebyshev, we want x = cos(theta) so T_n(x) = cos(n*theta)
    std::vector<double> xs_cheb(N);
    for (int i = 0; i < N; ++i) {
        double theta = (i + 0.5) * (3.14159265358979323846 / N); // (0, pi)
        xs_cheb[i] = std::cos(theta);
    }

    // -------------- Prepare test polynomials --------------
    // (1) Constant: P(x) = 7
    std::vector<double> p_const = {7.0};

    // (2) Linear: P(x) = 3x - 2  (a0 = -2, a1 = 3)
    std::vector<double> p_linear = {-2.0, 3.0};

    // (3) Rooted: P(x) = (x-1)(x-3) = x^2 - 4x + 3  (zeros at x=1 and x=3)
    std::vector<double> p_rooted = poly_from_roots({1.0, 3.0});

    // (4) Binomial (1 + x)^n  (choose n moderately to avoid overflow in float tests)
    int n_binom = 12;
    std::vector<double> p_binom = binomial_1px_pow(n_binom);

    // (5) Chebyshev T_n (degree n), verify with T_n(cos θ) = cos(n θ)
    int n_cheb = 12;
    std::vector<double> p_cheb = chebyshev_T(n_cheb);

    // -------------- Run tests --------------
    std::vector<TestResult> results;
    results.push_back(run_named_test("constant", p_const, xs));
    results.push_back(run_named_test("linear",   p_linear, xs));
    results.push_back(run_named_test("rooted",   p_rooted, xs));
    results.push_back(run_named_test("(1+x)^n",  p_binom,  xs));

    // Chebyshev uses xs_cheb for validation property; override CPU ref to cos(nθ)
    // We still run the standard harness, then *replace* the CPU reference comparison
    // by an analytical check here for clarity.
    {
        // GPU run via harness (float + double) but with xs_cheb
        auto tmp = run_named_test("Chebyshev", p_cheb, xs_cheb);

        // Recompute errors vs cos(n*theta) analytically
        // Re-run GPU paths once but keep timings from tmp
        const int deg = (int)p_cheb.size() - 1;
        const int N2  = (int)xs_cheb.size();

        // GPU float
        std::vector<float>  y_f;
        float ms_f = launch_eval_ptr_ms<float>(p_cheb, xs_cheb, y_f, deg);

        // GPU double
        std::vector<double> y_d;
        float ms_d = launch_eval_ptr_ms<double>(p_cheb, xs_cheb, y_d, deg);

        double max_abs_f = 0.0, max_rel_f = 0.0;
        double max_abs_d = 0.0, max_rel_d = 0.0;

        for (int i = 0; i < N2; ++i) {
            // theta chosen above so that xs_cheb[i] = cos(theta)
            double theta = (i + 0.5) * (3.14159265358979323846 / N2);
            double yref  = std::cos(n_cheb * theta);
            double yf    = (double)y_f[i];
            double yd    = y_d[i];

            max_abs_f = std::max(max_abs_f, abs_err(yf, yref));
            max_rel_f = std::max(max_rel_f, rel_err(yf, yref));
            max_abs_d = std::max(max_abs_d, abs_err(yd, yref));
            max_rel_d = std::max(max_rel_d, rel_err(yd, yref));
        }

        TestResult r;
        r.name = "chebyshev";
        r.deg  = (int)p_cheb.size() - 1;
        r.N    = N2;
        r.max_abs_float  = max_abs_f;
        r.max_rel_float  = max_rel_f;
        r.ms_gpu_float   = ms_f;
        r.max_abs_double = max_abs_d;
        r.max_rel_double = max_rel_d;
        r.ms_gpu_double  = ms_d;
        // Provide a CPU baseline time doing cos(nθ) to be comparable (very fast)
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile double sink = 0.0;
        for (int i = 0; i < N2; ++i) {
            double theta = (i + 0.5) * (3.14159265358979323846 / N2);
            sink += std::cos(n_cheb * theta);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        r.ms_cpu_ref = std::chrono::duration<double, std::milli>(t1 - t0).count();

        results.push_back(r);
    }

    // -------------- Print summary table --------------
    printf("\n=== Section 4.6 — GPU/CPU Validation Summary ===\n");
    for (const auto& r : results) print_row(r);

    // -------------- Quick spot-checks at special points --------------
    // For rooted polynomial, verify zeros at x=1 and x=3.
    auto spot = [&](double x){
        double yr = horner_ref(p_rooted, (int)p_rooted.size()-1, x);
        printf("Spot: P_rooted(%g) = %.3e (should be 0)\n", x, yr);
    };
    printf("\n--- Spot checks ---\n");
    spot(1.0);
    spot(3.0);

    printf("\nDone.\n");
    return 0;
}
