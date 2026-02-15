#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
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
// Create a random single-cycle permutation (Hamiltonian cycle).
//
// arr[i] = next index to visit.  Starting from index 0 and following the
// chain visits every element exactly once before returning to 0.
//
// This is the standard "pointer chasing" setup: each load's ADDRESS depends
// on the previous load's VALUE, so the hardware cannot issue multiple loads
// in parallel.  We measure pure memory latency.
// ---------------------------------------------------------------------------
void create_random_cycle(int* arr, int n) {
    std::vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(perm[i], perm[j]);
    }
    for (int i = 0; i < n - 1; i++)
        arr[perm[i]] = perm[i + 1];
    arr[perm[n - 1]] = perm[0];
}

// ---------------------------------------------------------------------------
// Pointer-chase kernels, one per memory level
// ---------------------------------------------------------------------------

// Shared memory
__global__ void chase_smem(const int* __restrict__ arr_in, int n, int steps,
                           long long* out_cycles) {
    extern __shared__ int smem[];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        smem[i] = arr_in[i];
    __syncthreads();

    int idx = 0;
    long long t0 = clock64();
    for (int s = 0; s < steps; s++)
        idx = smem[idx];
    long long t1 = clock64();

    *out_cycles = t1 - t0;
    if (idx == -1) *out_cycles = 0;
}

// L1 cache — default caching (.ca), small working set fits in L1
__global__ void chase_l1(const int* __restrict__ arr, int steps,
                         long long* out_cycles) {
    int idx = 0;
    long long t0 = clock64();
    for (int s = 0; s < steps; s++)
        idx = arr[idx];
    long long t1 = clock64();

    *out_cycles = t1 - t0;
    if (idx == -1) *out_cycles = 0;
}

// L2 cache — __ldcg bypasses L1, working set fits in L2
__global__ void chase_l2(const int* __restrict__ arr, int steps,
                         long long* out_cycles) {
    int idx = 0;
    long long t0 = clock64();
    for (int s = 0; s < steps; s++)
        idx = __ldcg(&arr[idx]);
    long long t1 = clock64();

    *out_cycles = t1 - t0;
    if (idx == -1) *out_cycles = 0;
}

// Flush L2 by reading+writing a buffer larger than L2.
// Every cache line gets loaded, evicting all previous L2 contents.
__global__ void flush_l2_kernel(int* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride)
        data[i] += 1;   // read + write → forces cache line load
}

// ---------------------------------------------------------------------------
int main() {
    srand(42);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int num_sms = prop.multiProcessorCount;
    double clock_ghz = prop.clockRate / 1.0e6;
    int l2_bytes = prop.l2CacheSize;

    printf("Device: %s\n", prop.name);
    printf("SM clock: %.2f GHz\n", clock_ghz);
    printf("L2 cache: %d MB\n\n", l2_bytes / (1024 * 1024));
    printf("Method: pointer chasing (each load depends on the previous),\n");
    printf("        single thread <<<1,1>>>, measures pure latency.\n\n");

    long long *d_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(long long)));

    // Allocate an L2-flush buffer (larger than L2)
    size_t flush_bytes = (size_t)l2_bytes + 8 * 1024 * 1024;  // L2 + 8 MB
    size_t flush_n = flush_bytes / sizeof(int);
    int *d_flush;
    CHECK_CUDA(cudaMalloc(&d_flush, flush_bytes));
    CHECK_CUDA(cudaMemset(d_flush, 0, flush_bytes));

    constexpr int N_SMALL = 2048;                   //  8 KB
    constexpr int N_L2    = 1024 * 1024;            //  4 MB
    constexpr int N_HBM   = 64 * 1024 * 1024;      // 256 MB

    constexpr int STEPS       = 100000;
    constexpr int STEPS_L2    = 1000000;  // >= N_L2 so warmup visits all elements
    constexpr int STEPS_HBM   = 50000;

    printf("=== Memory Latency ===\n\n");

    // --- Shared Memory ---
    double smem_cycles, smem_ns;
    {
        std::vector<int> h(N_SMALL);
        create_random_cycle(h.data(), N_SMALL);

        int *d_arr;
        CHECK_CUDA(cudaMalloc(&d_arr, N_SMALL * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_arr, h.data(), N_SMALL * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Warmup
        chase_smem<<<1, 32, N_SMALL * sizeof(int)>>>(d_arr, N_SMALL, STEPS, d_cycles);
        CHECK_CUDA(cudaDeviceSynchronize());
        // Measure
        chase_smem<<<1, 32, N_SMALL * sizeof(int)>>>(d_arr, N_SMALL, STEPS, d_cycles);
        CHECK_CUDA(cudaDeviceSynchronize());

        long long cyc;
        CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(long long),
                              cudaMemcpyDeviceToHost));
        smem_cycles = (double)cyc / STEPS;
        smem_ns = smem_cycles / clock_ghz;

        printf("  Shared Memory   (%3d KB):  %6.1f cycles   %6.1f ns\n",
               (int)(N_SMALL * sizeof(int) / 1024), smem_cycles, smem_ns);
        CHECK_CUDA(cudaFree(d_arr));
    }

    // --- L1 Cache ---
    double l1_cycles, l1_ns;
    {
        std::vector<int> h(N_SMALL);
        create_random_cycle(h.data(), N_SMALL);

        int *d_arr;
        CHECK_CUDA(cudaMalloc(&d_arr, N_SMALL * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_arr, h.data(), N_SMALL * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Warmup (loads 8 KB into L1)
        chase_l1<<<1, 1>>>(d_arr, STEPS, d_cycles);
        CHECK_CUDA(cudaDeviceSynchronize());
        // Measure
        chase_l1<<<1, 1>>>(d_arr, STEPS, d_cycles);
        CHECK_CUDA(cudaDeviceSynchronize());

        long long cyc;
        CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(long long),
                              cudaMemcpyDeviceToHost));
        l1_cycles = (double)cyc / STEPS;
        l1_ns = l1_cycles / clock_ghz;

        printf("  L1 Cache        (%3d KB):  %6.1f cycles   %6.1f ns\n",
               (int)(N_SMALL * sizeof(int) / 1024), l1_cycles, l1_ns);
        CHECK_CUDA(cudaFree(d_arr));
    }

    // --- L2 Cache ---
    double l2_cycles, l2_ns;
    {
        std::vector<int> h(N_L2);
        create_random_cycle(h.data(), N_L2);

        int *d_arr;
        CHECK_CUDA(cudaMalloc(&d_arr, (size_t)N_L2 * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_arr, h.data(), (size_t)N_L2 * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Warmup: STEPS_L2 >= N_L2, so every element is visited and cached in L2
        chase_l2<<<1, 1>>>(d_arr, STEPS_L2, d_cycles);
        CHECK_CUDA(cudaDeviceSynchronize());
        // Measure
        chase_l2<<<1, 1>>>(d_arr, STEPS_L2, d_cycles);
        CHECK_CUDA(cudaDeviceSynchronize());

        long long cyc;
        CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(long long),
                              cudaMemcpyDeviceToHost));
        l2_cycles = (double)cyc / STEPS_L2;
        l2_ns = l2_cycles / clock_ghz;

        printf("  L2 Cache        (%3d MB):  %6.1f cycles   %6.1f ns\n",
               (int)((size_t)N_L2 * sizeof(int) / (1024 * 1024)),
               l2_cycles, l2_ns);
        CHECK_CUDA(cudaFree(d_arr));
    }

    // --- HBM / DRAM ---
    //
    // Strategy: use a 256 MB working set (>> 60 MB L2).
    // Before measurement, flush L2 by reading a separate large buffer.
    // Then run the pointer chase with __ldcg loads.  Since the data is NOT
    // in L2, every __ldcg load misses L2 and goes to HBM.
    //
    // We DON'T warmup the chase itself — that would load data into L2.
    // We do a separate GPU warmup (flush kernel) so the GPU isn't cold.
    double hbm_cycles, hbm_ns;
    {
        std::vector<int> h(N_HBM);
        create_random_cycle(h.data(), N_HBM);

        int *d_arr;
        CHECK_CUDA(cudaMalloc(&d_arr, (size_t)N_HBM * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_arr, h.data(), (size_t)N_HBM * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Flush L2: read+write a buffer larger than L2 to evict everything
        flush_l2_kernel<<<num_sms, 256>>>(d_flush, flush_n);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Measure — no warmup for the chase, so data starts cold (not in L2)
        chase_l2<<<1, 1>>>(d_arr, STEPS_HBM, d_cycles);
        CHECK_CUDA(cudaDeviceSynchronize());

        long long cyc;
        CHECK_CUDA(cudaMemcpy(&cyc, d_cycles, sizeof(long long),
                              cudaMemcpyDeviceToHost));
        hbm_cycles = (double)cyc / STEPS_HBM;
        hbm_ns = hbm_cycles / clock_ghz;

        printf("  HBM / DRAM    (%3d MB):  %6.1f cycles   %6.1f ns\n",
               (int)((size_t)N_HBM * sizeof(int) / (1024 * 1024)),
               hbm_cycles, hbm_ns);
        CHECK_CUDA(cudaFree(d_arr));
    }

    // --- Ratios ---
    printf("\n=== Latency Ratios ===\n\n");
    printf("  L1 / Shared Memory:   %5.1fx\n", l1_cycles / smem_cycles);
    printf("  L2 / L1:              %5.1fx\n", l2_cycles / l1_cycles);
    printf("  HBM / L2:             %5.1fx\n", hbm_cycles / l2_cycles);
    printf("  HBM / Shared Memory:  %5.1fx\n", hbm_cycles / smem_cycles);
    printf("\n");

    CHECK_CUDA(cudaFree(d_flush));
    CHECK_CUDA(cudaFree(d_cycles));
    return 0;
}
