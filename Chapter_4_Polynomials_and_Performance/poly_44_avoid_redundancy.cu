// poly_44_avoid_redundancy.cu
// Demonstrates: (A) constant memory + reuse, (B) pointer-walk SoA layout,
// (C) fused pass for value+derivative, (D) shared-memory staging per block,
// plus CPU references and simple correctness checks.
//
// Build:
//   nvcc -O3 poly_44_avoid_redundancy.cu -o poly44
//
// Run:
//   ./poly44

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>

#ifndef MAX_DEGREE_PLUS_1
#define MAX_DEGREE_PLUS_1 64
#endif

// -------------------- CUDA error check --------------------
#define CUDA_CHECK(call) do {                                   \
    cudaError_t _err = (call);                                  \
    if (_err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s at %s:%d\n",             \
                cudaGetErrorString(_err), __FILE__, __LINE__);  \
        std::exit(1);                                           \
    }                                                           \
} while(0)

// -------------------- Constant memory (one polynomial) --------------------
__constant__ double d_coeffs[MAX_DEGREE_PLUS_1]; // a[0..deg]

// -------------------- CPU references --------------------
static inline double horner_ref(const std::vector<double>& a, int deg, double x) {
    double r = a[deg];
    for (int k = deg - 1; k >= 0; --k) r = r * x + a[k];
    return r;
}

// Returns P(x), P'(x)
static inline std::pair<double,double> horner_with_deriv_ref(const std::vector<double>& a, int deg, double x) {
    double r = a[deg]; // P
    double d = 0.0;    // P'
    for (int k = deg - 1; k >= 0; --k) {
        d = std::fma(d, x, r); // d = d*x + r
        r = std::fma(r, x, a[k]);
    }
    return {r, d};
}

// -------------------- Kernels --------------------

// (1) Constant memory, simple Horner (reads x once, reuses acc in registers)
__global__ void eval_horner_const_basic(const double* __restrict__ xs,
                                        double* __restrict__ ys,
                                        int n, int deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double x   = xs[i];           // keep in a register
    double acc = d_coeffs[deg];   // read once

    #pragma unroll
    for (int k = deg - 1; k >= 0; --k) {
        acc = fma(acc, x, d_coeffs[k]); // 1 FMA per step
    }
    ys[i] = acc; // one global store
}

// (2) Fused pass: compute P(x) and P'(x) together (no second walk)
__global__ void eval_horner_const_with_deriv(const double* __restrict__ xs,
                                             double* __restrict__ ys,
                                             double* __restrict__ dys,
                                             int n, int deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double x = xs[i];
    double r = d_coeffs[deg]; // P accumulator
    double d = 0.0;           // P' accumulator

    #pragma unroll
    for (int k = deg - 1; k >= 0; --k) {
        d = fma(d, x, r);        // d = d*x + r
        r = fma(r, x, d_coeffs[k]);
    }
    ys[i]  = r;
    dys[i] = d;
}

// (3) Many polynomials: Structure-of-Arrays (SoA) + pointer walk (avoid k*stride)
__global__ void eval_horner_soa_ptrwalk(const double* __restrict__ xs,
                                        const double* __restrict__ coeffs_soa, // (deg+1) * N
                                        double* __restrict__ ys,
                                        int N, int deg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // thread's polynomial + x index
    if (i >= N) return;

    int stride = N;
    const double* akp = coeffs_soa + deg * stride + i; // start at a_deg^(i)
    double x   = xs[i];
    double acc = *akp; // a_deg

    // Walk down by subtracting stride (no per-iter multiply)
    for (int k = deg - 1; k >= 0; --k) {
        akp -= stride;      // points to a_k^(i)
        acc  = fma(acc, x, *akp);
    }
    ys[i] = acc;
}

// (4) One polynomial per block: stage coefficients once in shared memory
// Each block owns one polynomial and processes n_per_block inputs (1 x per thread).
__global__ void eval_horner_shared_per_block(const double* __restrict__ xs,
                                             const double* __restrict__ coeffs_aos, // AoS: polynomials back-to-back
                                             double* __restrict__ ys,
                                             int n_per_block, int deg)
{
    extern __shared__ double a[]; // size >= deg+1 (dynamic shared mem)
    int poly_id = blockIdx.x;

    // Stage coefficients into shared memory (spread across threads)
    const double* coeffs = coeffs_aos + poly_id * (deg + 1);
    for (int t = threadIdx.x; t <= deg; t += blockDim.x) {
        a[t] = coeffs[t];
    }
    __syncthreads(); // ensure a[] is ready

    // Now each thread computes one x for this polynomial
    if (threadIdx.x >= n_per_block) return;
    int i   = poly_id * n_per_block + threadIdx.x;
    double x   = xs[i];
    double acc = a[deg];

    #pragma unroll
    for (int k = deg - 1; k >= 0; --k) {
        acc = fma(acc, x, a[k]);
    }
    ys[i] = acc;
}

// -------------------- Utility: error metrics --------------------
static inline double rel_err(double y, double yref) {
    double denom = std::max(1.0, std::fabs(yref));
    return std::fabs(y - yref) / denom;
}

int main() {
    // Problem sizes
    const int deg = 7;                    // degree (adjust ≤ MAX_DEGREE_PLUS_1-1)
    const int N   = 1 << 16;              // number of inputs for single-poly tests
    const int Npolys = 1 << 14;           // for SoA test (many polynomials)

    assert(deg + 1 <= MAX_DEGREE_PLUS_1);

    // Random setup
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> Ux(-2.0, 2.0);
    std::uniform_real_distribution<double> Ua(-1.0, 1.0);

    // Coefficients for the "single polynomial" tests
    std::vector<double> h_coeffs(deg + 1);
    for (int k = 0; k <= deg; ++k) h_coeffs[k] = Ua(rng);

    // X inputs
    std::vector<double> h_x(N), h_y(N), h_yref(N), h_dy(N), h_dyref(N);
    for (int i = 0; i < N; ++i) h_x[i] = Ux(rng);

    // CPU reference for single polynomial tasks
    for (int i = 0; i < N; ++i) {
        auto pr = horner_with_deriv_ref(h_coeffs, deg, h_x[i]);
        h_yref[i]  = pr.first;
        h_dyref[i] = pr.second;
    }

    // -------------------- Device buffers --------------------
    double *d_x = nullptr, *d_y = nullptr, *d_dy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dy, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Upload constant memory once (avoid redundant H2D if polynomial unchanged)
    CUDA_CHECK(cudaMemcpyToSymbol(d_coeffs, h_coeffs.data(), (deg + 1) * sizeof(double)));

    // Launch shape
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // (1) Constant memory basic Horner
    eval_horner_const_basic<<<blocks, threads>>>(d_x, d_y, N, deg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

    double max_rel_err = 0.0;
    for (int i = 0; i < N; ++i) max_rel_err = std::max(max_rel_err, rel_err(h_y[i], h_yref[i]));
    printf("[const/basic]   max relative error: %.3e\n", max_rel_err);

    // (2) Fused P and P' in one pass
    eval_horner_const_with_deriv<<<blocks, threads>>>(d_x, d_y, d_dy, N, deg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y.data(),  d_y,  N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dy.data(), d_dy, N * sizeof(double), cudaMemcpyDeviceToHost));

    double max_rel_err_y  = 0.0, max_rel_err_dy = 0.0;
    for (int i = 0; i < N; ++i) {
        max_rel_err_y  = std::max(max_rel_err_y,  rel_err(h_y[i],  h_yref[i]));
        max_rel_err_dy = std::max(max_rel_err_dy, rel_err(h_dy[i], h_dyref[i]));
    }
    printf("[const+deriv]   max rel err P:  %.3e,  P': %.3e\n",
           max_rel_err_y, max_rel_err_dy);

    // (3) SoA: many polynomials, one x per thread, pointer-walk addressing
    // Prepare data
    std::vector<double> h_x_many(Npolys), h_y_many(Npolys), h_y_many_ref(Npolys);
    for (int i = 0; i < Npolys; ++i) h_x_many[i] = Ux(rng);

    // Coeffs in SoA: coeffs_soa[k * Npolys + i] = a_k^(i)
    std::vector<double> h_coeffs_soa((deg + 1) * Npolys);
    for (int i = 0; i < Npolys; ++i) {
        for (int k = 0; k <= deg; ++k) {
            h_coeffs_soa[k * Npolys + i] = Ua(rng);
        }
    }

    // CPU reference for a subset (to keep demo snappy, do all anyway)
    for (int i = 0; i < Npolys; ++i) {
        // Gather a_i into a vector for ref (AoS style just for checking)
        std::vector<double> ai(deg + 1);
        for (int k = 0; k <= deg; ++k) ai[k] = h_coeffs_soa[k * Npolys + i];
        h_y_many_ref[i] = horner_ref(ai, deg, h_x_many[i]);
    }

    // Device buffers for SoA test
    double *d_x_many = nullptr, *d_y_many = nullptr, *d_coeffs_soa = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x_many,     Npolys * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_many,     Npolys * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_coeffs_soa, (deg + 1) * Npolys * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_many,     h_x_many.data(),     Npolys * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeffs_soa, h_coeffs_soa.data(), (deg + 1) * Npolys * sizeof(double), cudaMemcpyHostToDevice));

    int blocks_many = (Npolys + threads - 1) / threads;
    eval_horner_soa_ptrwalk<<<blocks_many, threads>>>(d_x_many, d_coeffs_soa, d_y_many, Npolys, deg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y_many.data(), d_y_many, Npolys * sizeof(double), cudaMemcpyDeviceToHost));

    double max_rel_err_many = 0.0;
    for (int i = 0; i < Npolys; ++i) max_rel_err_many = std::max(max_rel_err_many, rel_err(h_y_many[i], h_y_many_ref[i]));
    printf("[SoA/ptr-walk]  max relative error: %.3e\n", max_rel_err_many);

    // (4) Shared-memory staging: one polynomial per block, many x per block
    // For demo: let each block process exactly blockDim.x inputs
    const int n_per_block = 256;
    const int blocks_shared = std::max(1, N / n_per_block);
    const int total_inputs  = blocks_shared * n_per_block;

    // Prepare AoS coefficients: coeffs_aos[poly * (deg+1) + k]
    std::vector<double> h_coeffs_aos(blocks_shared * (deg + 1));
    for (int b = 0; b < blocks_shared; ++b) {
        for (int k = 0; k <= deg; ++k) h_coeffs_aos[b * (deg + 1) + k] = Ua(rng);
    }

    // Prepare xs for all blocks
    std::vector<double> h_x_shared(total_inputs), h_y_shared(total_inputs), h_y_shared_ref(total_inputs);
    for (int i = 0; i < total_inputs; ++i) h_x_shared[i] = Ux(rng);

    // Reference: each block has its own polynomial
    for (int b = 0; b < blocks_shared; ++b) {
        std::vector<double> a_blk(deg + 1);
        for (int k = 0; k <= deg; ++k) a_blk[k] = h_coeffs_aos[b * (deg + 1) + k];
        for (int t = 0; t < n_per_block; ++t) {
            int i = b * n_per_block + t;
            h_y_shared_ref[i] = horner_ref(a_blk, deg, h_x_shared[i]);
        }
    }

    double *d_x_shared=nullptr, *d_y_shared=nullptr, *d_coeffs_aos=nullptr;
    CUDA_CHECK(cudaMalloc(&d_x_shared,   total_inputs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_shared,   total_inputs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_coeffs_aos, blocks_shared * (deg + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x_shared,   h_x_shared.data(),   total_inputs * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeffs_aos, h_coeffs_aos.data(), blocks_shared * (deg + 1) * sizeof(double), cudaMemcpyHostToDevice));

    eval_horner_shared_per_block<<<blocks_shared, n_per_block, (deg + 1) * sizeof(double)>>>(
        d_x_shared, d_coeffs_aos, d_y_shared, n_per_block, deg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y_shared.data(), d_y_shared, total_inputs * sizeof(double), cudaMemcpyDeviceToHost));

    double max_rel_err_shared = 0.0;
    for (int i = 0; i < total_inputs; ++i) max_rel_err_shared = std::max(max_rel_err_shared, rel_err(h_y_shared[i], h_y_shared_ref[i]));
    printf("[shared/block]  max relative error: %.3e\n", max_rel_err_shared);

    // Spot-check: print a couple of values
    printf("\nSpot-check (first 3 of single-poly):\n");
    for (int i = 0; i < 3; ++i) {
        printf("x=% .4f  GPU=% .6f  CPU=% .6f\n", h_x[i], h_y[i], h_yref[i]);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_dy));
    CUDA_CHECK(cudaFree(d_x_many));
    CUDA_CHECK(cudaFree(d_y_many));
    CUDA_CHECK(cudaFree(d_coeffs_soa));
    CUDA_CHECK(cudaFree(d_x_shared));
    CUDA_CHECK(cudaFree(d_y_shared));
    CUDA_CHECK(cudaFree(d_coeffs_aos));

    printf("\nDone.\n");
    return 0;
}
