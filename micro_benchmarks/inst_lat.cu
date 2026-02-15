#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Operation structs — each defines a single arithmetic operation.
//
// "apply(x)" takes x as input and returns the result.  The benchmark kernel
// chains these: x = apply(x) repeatedly, creating a serial dependency.
// With ILP > 1, multiple INDEPENDENT chains run in parallel.
// ---------------------------------------------------------------------------

// --- ALU / FP32 pipeline ---
struct FaddOp  { static __device__ __forceinline__ float apply(float x) { return x + 1.0f; }};
struct FmulOp  { static __device__ __forceinline__ float apply(float x) { return x * 1.0000001f; }};
struct FmaOp   { static __device__ __forceinline__ float apply(float x) { return fmaf(x, 1.0000001f, 0.0001f); }};

// --- INT32 pipeline ---
struct IaddOp  { static __device__ __forceinline__ int apply(int x) { return x + 1; }};
struct ImulOp  { static __device__ __forceinline__ int apply(int x) { return x * 13; }};

// --- SFU / special function unit (directly chainable) ---
struct FrcpOp  { static __device__ __forceinline__ float apply(float x) { return __frcp_rn(x); }};  // oscillates: x → 1/x → x
struct FsqrtOp { static __device__ __forceinline__ float apply(float x) { return __fsqrt_rn(x); }}; // converges to 1.0
struct SinOp   { static __device__ __forceinline__ float apply(float x) { return __sinf(x); }};     // converges to 0.0

// --- SFU compound (need extra op to keep values valid) ---
struct LogOp   { static __device__ __forceinline__ float apply(float x) { return __logf(fabsf(x) + 2.0f); }};  // +FABS+FADD
struct ExpOp   { static __device__ __forceinline__ float apply(float x) { return __expf(x * 0.00001f); }};     // +FMUL

// ---------------------------------------------------------------------------
// Benchmark kernels
//
// Each kernel runs `steps` iterations.  In each iteration, ILP independent
// chains each execute one apply().  Total operations = steps × ILP.
//
// ILP=1: each op depends on the previous → measures LATENCY
// ILP=N: N independent chains → pipeline fills → measures THROUGHPUT
// ---------------------------------------------------------------------------
template <typename Op, int ILP>
__global__ void bench_float(int steps, long long* out) {
    float x[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++)
        x[i] = (float)(threadIdx.x + i * 32 + 1);

    long long t0 = clock64();
    #pragma unroll 1                    // do NOT unroll outer loop
    for (int s = 0; s < steps; s++) {
        #pragma unroll                  // DO unroll inner loop → creates ILP
        for (int i = 0; i < ILP; i++)
            x[i] = Op::apply(x[i]);
    }
    long long t1 = clock64();

    // Prevent dead-code elimination: compiler must keep all x[i] alive
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ILP; i++) sum += x[i];
    if (threadIdx.x == 0) *out = t1 - t0;
    if (sum == -INFINITY) *out = -1;
}

template <typename Op, int ILP>
__global__ void bench_int(int steps, long long* out) {
    int x[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++)
        x[i] = threadIdx.x + i * 32 + 1;

    long long t0 = clock64();
    #pragma unroll 1
    for (int s = 0; s < steps; s++) {
        #pragma unroll
        for (int i = 0; i < ILP; i++)
            x[i] = Op::apply(x[i]);
    }
    long long t1 = clock64();

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < ILP; i++) sum += x[i];
    if (threadIdx.x == 0) *out = t1 - t0;
    if (sum == INT_MIN) *out = -1;
}

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------
template <typename Op, int ILP>
double mf(int steps, long long* d_out) {     // measure float op
    bench_float<Op, ILP><<<1, 32>>>(steps, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    bench_float<Op, ILP><<<1, 32>>>(steps, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    long long cyc;
    CHECK_CUDA(cudaMemcpy(&cyc, d_out, sizeof(long long), cudaMemcpyDeviceToHost));
    return (double)cyc / ((double)steps * ILP);
}

template <typename Op, int ILP>
double mi(int steps, long long* d_out) {     // measure int op
    bench_int<Op, ILP><<<1, 32>>>(steps, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    bench_int<Op, ILP><<<1, 32>>>(steps, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    long long cyc;
    CHECK_CUDA(cudaMemcpy(&cyc, d_out, sizeof(long long), cudaMemcpyDeviceToHost));
    return (double)cyc / ((double)steps * ILP);
}

void print_row(const char* name, double c[8]) {
    printf("  %-18s %5.1f   %5.1f   %5.1f   %5.1f   %5.1f   %5.1f   %5.1f   %5.1f\n",
           name, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}

#define MEASURE_F(Op, name) do {                         \
    double c[8];                                          \
    c[0] = mf<Op, 1>(steps, d_out);                      \
    c[1] = mf<Op, 2>(steps, d_out);                      \
    c[2] = mf<Op, 4>(steps, d_out);                      \
    c[3] = mf<Op, 8>(steps, d_out);                      \
    c[4] = mf<Op, 16>(steps, d_out);                     \
    c[5] = mf<Op, 32>(steps, d_out);                     \
    c[6] = mf<Op, 64>(steps, d_out);                     \
    c[7] = mf<Op, 128>(steps, d_out);                    \
    print_row(name, c);                                   \
} while (0)

#define MEASURE_I(Op, name) do {                         \
    double c[8];                                          \
    c[0] = mi<Op, 1>(steps, d_out);                      \
    c[1] = mi<Op, 2>(steps, d_out);                      \
    c[2] = mi<Op, 4>(steps, d_out);                      \
    c[3] = mi<Op, 8>(steps, d_out);                      \
    c[4] = mi<Op, 16>(steps, d_out);                     \
    c[5] = mi<Op, 32>(steps, d_out);                     \
    c[6] = mi<Op, 64>(steps, d_out);                     \
    c[7] = mi<Op, 128>(steps, d_out);                    \
    print_row(name, c);                                   \
} while (0)

// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    double clock_ghz = prop.clockRate / 1.0e6;

    printf("Device: %s\n", prop.name);
    printf("SM clock: %.2f GHz\n\n", clock_ghz);
    printf("=== Instruction Latency & Throughput (single warp <<<1,32>>>) ===\n\n");
    printf("Each cell = cycles per warp-instruction (lower is better).\n");
    printf("ILP=1  → dependent chain → pure LATENCY.\n");
    printf("ILP=32 → 32 independent chains → converges to THROUGHPUT.\n");
    printf("ILP=128 → amortizes loop overhead → reveals true throughput.\n\n");

    long long *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(long long)));

    constexpr int steps = 100000;

    printf("                       ILP=1   ILP=2   ILP=4   ILP=8  ILP=16  ILP=32  ILP=64 ILP=128\n");
    printf("  ────────────────────────────────────────────────────────────────────────────────────────\n");
    MEASURE_F(FaddOp,  "FADD");
    MEASURE_F(FmulOp,  "FMUL");
    MEASURE_F(FmaOp,   "FMA");
    MEASURE_I(IaddOp,  "IADD");
    MEASURE_I(ImulOp,  "IMUL");
    printf("  ────────────────────────────────────────────────────────────────────────────────────────\n");
    MEASURE_F(FrcpOp,  "__frcp_rn");
    MEASURE_F(FsqrtOp, "__fsqrt_rn");
    MEASURE_F(SinOp,   "__sinf");
    MEASURE_F(LogOp,   "__logf (*)");
    MEASURE_F(ExpOp,   "__expf (*)");

    printf("\n");
    printf("(*) __logf chain: x = __logf(fabsf(x)+2); includes FABS+FADD overhead\n");
    printf("    __expf chain: x = __expf(x*1e-5);     includes FMUL overhead\n");

    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
