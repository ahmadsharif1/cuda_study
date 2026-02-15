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
// Core copy kernel — works for all configurations via launch params + template
// ---------------------------------------------------------------------------
template <typename T, int UNROLL = 1>
__global__ void copy_kernel(const T* __restrict__ src, T* __restrict__ dst,
                            size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    // Unrolled loop
    size_t i = idx;
    for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
        T vals[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; u++)
            vals[u] = src[i + u * stride];
        #pragma unroll
        for (int u = 0; u < UNROLL; u++)
            dst[i + u * stride] = vals[u];
    }
    // Remainder
    for (; i < n; i += stride)
        dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------
static constexpr double PEAK_BW_TB_S = 2.0;   // H100 SXM5 HBM peak
static constexpr double PEAK_BW_GB_S = PEAK_BW_TB_S * 1000.0;
static constexpr int WARMUP = 5;
static constexpr int ITERS  = 20;

struct BenchResult {
    double gb_s;
    double pct_peak;
};

template <typename T, int UNROLL>
BenchResult bench_copy(size_t num_elements, int blocks, int threads) {
    size_t bytes = num_elements * sizeof(T);

    T *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemset(d_src, 1, bytes));
    CHECK_CUDA(cudaMemset(d_dst, 0, bytes));

    // Warmup
    for (int i = 0; i < WARMUP; i++)
        copy_kernel<T, UNROLL><<<blocks, threads>>>(d_src, d_dst, num_elements);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed iterations
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        copy_kernel<T, UNROLL><<<blocks, threads>>>(d_src, d_dst, num_elements);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double sec = (ms / 1000.0) / ITERS;

    // 2× because we read src + write dst
    double total_bytes = 2.0 * bytes;
    double gb_s = (total_bytes / sec) / 1e9;
    double pct = (gb_s / PEAK_BW_GB_S) * 100.0;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    return {gb_s, pct};
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------
void print_row(const char* dtype, int unroll, BenchResult r) {
    printf("  %-12s UNROLL=%-4d %8.2f GB/s  (%5.2f%% peak)\n",
           dtype, unroll, r.gb_s, r.pct_peak);
}

void print_sm_row(int num_sms, BenchResult r) {
    printf("  %3d SMs:  %8.1f GB/s  (%5.1f%% peak)\n",
           num_sms, r.gb_s, r.pct_peak);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("Peak HBM BW (assumed): %.1f TB/s\n\n", PEAK_BW_TB_S);

    int num_sms = prop.multiProcessorCount;

    // 256 MB of data
    constexpr size_t DATA_BYTES = 256ULL * 1024 * 1024;
    constexpr size_t N_FLOAT  = DATA_BYTES / sizeof(float);
    constexpr size_t N_FLOAT4 = DATA_BYTES / sizeof(float4);

    // -----------------------------------------------------------------------
    // Experiment 1: Single Thread
    // -----------------------------------------------------------------------
    printf("=== Single Thread (1 thread, 1 SM) ===\n");
    {
        auto r = bench_copy<float, 1>(N_FLOAT, 1, 1);
        print_row("float", 1, r);
    }
    {
        auto r = bench_copy<float, 4>(N_FLOAT, 1, 1);
        print_row("float", 4, r);
    }
    {
        auto r = bench_copy<float4, 1>(N_FLOAT4, 1, 1);
        print_row("float4", 1, r);
    }
    {
        auto r = bench_copy<float4, 4>(N_FLOAT4, 1, 1);
        print_row("float4", 4, r);
    }
    {
        auto r = bench_copy<float, 32>(N_FLOAT, 1, 1);
        print_row("float", 32, r);
    }
    {
        auto r = bench_copy<float4, 32>(N_FLOAT4, 1, 1);
        print_row("float4", 32, r);
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Experiment 2: Single Warp
    // -----------------------------------------------------------------------
    printf("=== Single Warp (32 threads, 1 SM) ===\n");
    {
        auto r = bench_copy<float, 1>(N_FLOAT, 1, 32);
        print_row("float", 1, r);
    }
    {
        auto r = bench_copy<float, 4>(N_FLOAT, 1, 32);
        print_row("float", 4, r);
    }
    {
        auto r = bench_copy<float4, 1>(N_FLOAT4, 1, 32);
        print_row("float4", 1, r);
    }
    {
        auto r = bench_copy<float4, 4>(N_FLOAT4, 1, 32);
        print_row("float4", 4, r);
    }
    {
        auto r = bench_copy<float, 32>(N_FLOAT, 1, 32);
        print_row("float", 32, r);
    }
    {
        auto r = bench_copy<float4, 32>(N_FLOAT4, 1, 32);
        print_row("float4", 32, r);
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Experiment 3: Full SM (1024 threads, 1 block)
    // -----------------------------------------------------------------------
    printf("=== Full SM (1024 threads, 1 SM) ===\n");
    {
        auto r = bench_copy<float, 1>(N_FLOAT, 1, 1024);
        print_row("float", 1, r);
    }
    {
        auto r = bench_copy<float, 4>(N_FLOAT, 1, 1024);
        print_row("float", 4, r);
    }
    {
        auto r = bench_copy<float4, 1>(N_FLOAT4, 1, 1024);
        print_row("float4", 1, r);
    }
    {
        auto r = bench_copy<float4, 4>(N_FLOAT4, 1, 1024);
        print_row("float4", 4, r);
    }
    {
        auto r = bench_copy<float, 32>(N_FLOAT, 1, 1024);
        print_row("float", 32, r);
    }
    {
        auto r = bench_copy<float4, 32>(N_FLOAT4, 1, 1024);
        print_row("float4", 32, r);
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Experiment 4: SM Scaling (256 threads/block, float4, UNROLL=4)
    // -----------------------------------------------------------------------
    printf("=== SM Scaling (256 threads/block, float4, UNROLL=4) ===\n");

    // Test powers of 2, then specific counts up to and beyond all SMs
    int sm_counts[] = {1, 2, 4, 8, 16, 32, 64, 132};
    int n_tests = sizeof(sm_counts) / sizeof(sm_counts[0]);

    // Adjust last entry to actual SM count if different
    sm_counts[n_tests - 1] = num_sms;

    for (int t = 0; t < n_tests; t++) {
        int blocks = sm_counts[t];
        if (blocks > num_sms && blocks != num_sms)
            continue;  // skip entries beyond SM count (except the last)
        auto r = bench_copy<float4, 4>(N_FLOAT4, blocks, 256);
        print_sm_row(blocks, r);
    }

    // Also test 2× and 4× oversubscription
    {
        int blocks = num_sms * 2;
        auto r = bench_copy<float4, 4>(N_FLOAT4, blocks, 256);
        printf("  %3d blocks (2x SMs):  %8.1f GB/s  (%5.1f%% peak)\n",
               blocks, r.gb_s, r.pct_peak);
    }
    {
        int blocks = num_sms * 4;
        auto r = bench_copy<float4, 4>(N_FLOAT4, blocks, 256);
        printf("  %3d blocks (4x SMs):  %8.1f GB/s  (%5.1f%% peak)\n",
               blocks, r.gb_s, r.pct_peak);
    }
    printf("\n");

    return 0;
}
