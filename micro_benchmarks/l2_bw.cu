#include <cstdio>
#include <cstdlib>
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
// L2 bandwidth copy kernel
//
// Uses __ldcg / __stcg intrinsics to BYPASS L1 cache entirely.
//   ld.global.cg  =  load from L2 (skip L1)
//   st.global.cg  =  store to L2 (skip L1)
//
// This guarantees every access hits L2, not L1, regardless of working set
// size per SM.  Combined with a working set that fits in L2 (but is loaded
// from HBM during warmup), this measures pure L2 bandwidth.
// ---------------------------------------------------------------------------
template <int UNROLL>
__global__ void copy_l2(const float* __restrict__ src,
                        float* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    size_t i = idx;
    for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
        float vals[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; u++)
            vals[u] = __ldcg(&src[i + u * stride]);
        #pragma unroll
        for (int u = 0; u < UNROLL; u++)
            __stcg(&dst[i + u * stride], vals[u]);
    }
    for (; i < n; i += stride)
        __stcg(&dst[i], __ldcg(&src[i]));
}

// Same kernel but with default (L1+L2) caching for comparison
template <int UNROLL>
__global__ void copy_default(const float* __restrict__ src,
                             float* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    size_t i = idx;
    for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
        float vals[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; u++)
            vals[u] = src[i + u * stride];
        #pragma unroll
        for (int u = 0; u < UNROLL; u++)
            dst[i + u * stride] = vals[u];
    }
    for (; i < n; i += stride)
        dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------
static constexpr int WARMUP = 20;
static constexpr int ITERS  = 100;

typedef void (*KernelFn)(const float*, float*, size_t);

double bench(KernelFn kernel, float* d_src, float* d_dst, size_t n,
             int blocks, int threads) {
    size_t bytes = n * sizeof(float);

    // Warmup — bring data into L2
    for (int i = 0; i < WARMUP; i++)
        kernel<<<blocks, threads>>>(d_src, d_dst, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        kernel<<<blocks, threads>>>(d_src, d_dst, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double sec = (ms / 1000.0) / ITERS;

    double total_bytes = 2.0 * bytes;  // read src + write dst
    double gb_s = (total_bytes / sec) / 1e9;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return gb_s;
}

// ---------------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int num_sms = prop.multiProcessorCount;

    printf("Device: %s  (%d SMs)\n", prop.name, num_sms);
    printf("L2 cache size: %d MB\n\n", prop.l2CacheSize / (1024 * 1024));

    // Working set: 16 MB src + 16 MB dst = 32 MB total  (fits in 50 MB L2)
    constexpr size_t DATA_BYTES = 16ULL * 1024 * 1024;
    constexpr size_t N = DATA_BYTES / sizeof(float);

    printf("Working set: %zu MB (src) + %zu MB (dst) = %zu MB total\n",
           DATA_BYTES >> 20, DATA_BYTES >> 20, (2 * DATA_BYTES) >> 20);
    printf("Kernel uses __ldcg / __stcg  =  every access goes to L2, bypasses L1\n");
    printf("Warmup: %d iters, Measured: %d iters\n\n", WARMUP, ITERS);

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, DATA_BYTES));
    CHECK_CUDA(cudaMalloc(&d_dst, DATA_BYTES));
    CHECK_CUDA(cudaMemset(d_src, 1, DATA_BYTES));
    CHECK_CUDA(cudaMemset(d_dst, 0, DATA_BYTES));

    // -------------------------------------------------------------------
    // Experiment 1: SM scaling with L1 bypass (__ldcg/__stcg)
    // -------------------------------------------------------------------
    printf("=== L2 Bandwidth vs SM Count (256 threads/block, float, UNROLL=8) ===\n");

    int sm_counts[] = {1, 2, 4, 8, 16, 32, 64, 132};
    int n_tests = (int)(sizeof(sm_counts) / sizeof(sm_counts[0]));
    sm_counts[n_tests - 1] = num_sms;

    double max_bw = 0;
    for (int t = 0; t < n_tests; t++) {
        int blocks = sm_counts[t];
        double gb_s = bench(copy_l2<8>, d_src, d_dst, N, blocks, 256);
        if (gb_s > max_bw) max_bw = gb_s;
        printf("  %3d SMs:  %8.1f GB/s\n", blocks, gb_s);
    }
    // Oversubscription
    {
        int blocks = num_sms * 2;
        double gb_s = bench(copy_l2<8>, d_src, d_dst, N, blocks, 256);
        if (gb_s > max_bw) max_bw = gb_s;
        printf("  %3d blocks (2x SMs):  %8.1f GB/s\n", blocks, gb_s);
    }
    {
        int blocks = num_sms * 4;
        double gb_s = bench(copy_l2<8>, d_src, d_dst, N, blocks, 256);
        if (gb_s > max_bw) max_bw = gb_s;
        printf("  %3d blocks (4x SMs):  %8.1f GB/s\n", blocks, gb_s);
    }
    printf("\n  Peak observed (L2, bypass L1): %.1f GB/s  (%.1f TB/s)\n\n",
           max_bw, max_bw / 1000.0);

    // -------------------------------------------------------------------
    // Experiment 2: Same but with default caching (L1+L2) for comparison
    // -------------------------------------------------------------------
    printf("=== L1+L2 Bandwidth vs SM Count (default caching, same working set) ===\n");

    double max_bw_l1 = 0;
    for (int t = 0; t < n_tests; t++) {
        int blocks = sm_counts[t];
        double gb_s = bench(copy_default<8>, d_src, d_dst, N, blocks, 256);
        if (gb_s > max_bw_l1) max_bw_l1 = gb_s;
        printf("  %3d SMs:  %8.1f GB/s\n", blocks, gb_s);
    }
    {
        int blocks = num_sms * 4;
        double gb_s = bench(copy_default<8>, d_src, d_dst, N, blocks, 256);
        if (gb_s > max_bw_l1) max_bw_l1 = gb_s;
        printf("  %3d blocks (4x SMs):  %8.1f GB/s\n", blocks, gb_s);
    }
    printf("\n  Peak observed (L1+L2): %.1f GB/s  (%.1f TB/s)\n\n",
           max_bw_l1, max_bw_l1 / 1000.0);

    // -------------------------------------------------------------------
    // Experiment 3: HBM comparison (256 MB — doesn't fit in L2)
    // -------------------------------------------------------------------
    printf("=== HBM Comparison (256 MB, doesn't fit in L2) ===\n");

    constexpr size_t HBM_BYTES = 256ULL * 1024 * 1024;
    constexpr size_t N_HBM = HBM_BYTES / sizeof(float);

    float *d_src_hbm, *d_dst_hbm;
    CHECK_CUDA(cudaMalloc(&d_src_hbm, HBM_BYTES));
    CHECK_CUDA(cudaMalloc(&d_dst_hbm, HBM_BYTES));
    CHECK_CUDA(cudaMemset(d_src_hbm, 1, HBM_BYTES));
    CHECK_CUDA(cudaMemset(d_dst_hbm, 0, HBM_BYTES));

    double hbm_bw;
    {
        int blocks = num_sms * 4;
        hbm_bw = bench(copy_l2<8>, d_src_hbm, d_dst_hbm, N_HBM, blocks, 256);
        printf("  HBM (all SMs):  %8.1f GB/s  (%.1f TB/s)\n", hbm_bw, hbm_bw / 1000.0);
    }
    printf("\n  L2 peak / HBM peak = %.1fx\n", max_bw / hbm_bw);

    CHECK_CUDA(cudaFree(d_src_hbm));
    CHECK_CUDA(cudaFree(d_dst_hbm));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    return 0;
}
