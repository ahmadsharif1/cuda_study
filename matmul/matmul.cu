#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

// -------------------------------------------------------------------------
// Kernel Implementations
// -------------------------------------------------------------------------

// Naive: one thread per output element
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

// CuTe MMA kernel: global memory -> registers -> MMA (no shared memory)
// Follows the CuTe GEMM tutorial pattern (sgemm_1).
template <int BLK_M, int BLK_N, int BLK_K, class TiledMMA>
__global__ void matmul_cute_simple(const float* A_ptr, const float* B_ptr, float* C_ptr,
                                    int M, int K, int N, TiledMMA tiled_mma) {
    // Global memory tensors
    // A: (M, K) row-major
    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    // B: stored (K,N) row-major, viewed as (N,K) for the TN MMA
    auto mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), make_stride(Int<1>{}, N));
    // C: (M, N) row-major
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    // CTA tiling: blockIdx.x -> M, blockIdx.y -> N
    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                          make_coord(blockIdx.x, _));              // (BLK_M, BLK_K, k)
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}),
                          make_coord(blockIdx.y, _));              // (BLK_N, BLK_K, k)
    auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                          make_coord(blockIdx.x, blockIdx.y));     // (BLK_M, BLK_N)

    // Per-thread MMA partitions
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCgC = thr_mma.partition_C(gC);                           // (MMA, MMA_M, MMA_N)
    auto tCrC = thr_mma.partition_fragment_C(gC);                  // accumulator in registers
    clear(tCrC);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));         // A fragment in registers
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));         // B fragment in registers

    // Main loop over K
    int k_tile_max = size<2>(gA);
    for (int k = 0; k < k_tile_max; k++) {
        copy(thr_mma.partition_A(gA(_, _, k)), tCrA);
        copy(thr_mma.partition_B(gB(_, _, k)), tCrB);
        gemm(tiled_mma, tCrA, tCrB, tCrC);
    }

    // Store result
    copy(tCrC, tCgC);
}

// -------------------------------------------------------------------------
// Verification & Benchmarking Helpers
// -------------------------------------------------------------------------

// CPU dot product for a single output element C[r][c]
float matmul_cpu_single(const float* A, const float* B, int r, int c, int K, int N) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++)
        sum += A[r * K + i] * B[i * N + c];
    return sum;
}

// Probabilistic verification: spot-check num_samples random elements
bool verify(const float* gpu_C, const float* h_A, const float* h_B,
            int M, int K, int N, int num_samples = 1000) {
    float max_rel_err = 0.0f, max_abs_err = 0.0f;
    int mismatches = 0;
    float rtol = 1e-5f, atol = 1e-3f;

    // Fixed seed for deterministic sampling
    srand(42);
    for (int s = 0; s < num_samples; s++) {
        int r = rand() % M;
        int c = rand() % N;
        float got = gpu_C[r * N + c];
        float exp = matmul_cpu_single(h_A, h_B, r, c, K, N);
        float abs_err = fabsf(got - exp);
        float rel_err = abs_err / fmaxf(fabsf(exp), 1e-6f);

        if (abs_err > atol + rtol * fabsf(exp)) {
            if (mismatches < 5)
                printf("  MISMATCH C[%d][%d]: gpu=%.6f cpu=%.6f (abs=%.6f rel=%.6f)\n",
                       r, c, got, exp, abs_err, rel_err);
            mismatches++;
        }
        max_abs_err = fmaxf(max_abs_err, abs_err);
        max_rel_err = fmaxf(max_rel_err, rel_err);
    }
    printf("  max abs error: %e (over %d samples)\n", max_abs_err, num_samples);
    printf("  max rel error: %e\n", max_rel_err);
    if (mismatches == 0) {
        printf("  PASS: %d/%d samples match\n", num_samples, num_samples);
        return true;
    }
    printf("  FAIL: %d / %d samples mismatched\n", mismatches, num_samples);
    return false;
}

template <typename Func>
void run_benchmark(const char* name, Func kernel_launch,
                   float* d_C, float* h_C, const float* h_A, const float* h_B,
                   int M, int K, int N, int iterations, bool benchmark_mode) {

    size_t size_C = (size_t)M * N * sizeof(float);
    // 2*M*N*K flops: M*N*K multiplies + M*N*K adds
    double total_flops = 2.0 * M * N * (double)K;

    if (!benchmark_mode) {
        kernel_launch();
        cudaDeviceSynchronize();
        return;
    }

    // Warmup
    kernel_launch();
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, size_C);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
        kernel_launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;
    double tflops = total_flops / (avg_ms / 1000.0) / 1e12;

    // Verify
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("----------------------------------------------------------------\n");
    printf("%-20s\n", name);
    verify(h_C, h_A, h_B, M, K, N);
    printf("  Avg Time     : %.4f ms\n", avg_ms);
    printf("  Performance  : %.2f TFLOPS\n", tflops);
    printf("----------------------------------------------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

int main(int argc, char** argv) {
    int iterations = 100;
    bool benchmark_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = true;
        }
    }

    if (benchmark_mode)
        printf("Running in BENCHMARK mode with %d iteration(s).\n", iterations);
    else
        printf("Running in PROFILING mode (1 iteration, no warmup, no verification).\n");

    int M = 4096, K = 16384, N = 8192;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    // Host alloc and init
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    srand(123);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    // Device alloc
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Kernel configs
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    auto naive_op = [&]() {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    };

    // Run
    run_benchmark("Naive", naive_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);

    // CuTe simple MMA kernel (TF32, no shared memory)
    {
        using tiled_mma_t = TiledMMA<
            MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
            Layout<Shape<_2, _2, _1>>   // 4 warps = 128 threads
        >;

        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 8;
        tiled_mma_t tiled_mma;

        dim3 block_cute(size(tiled_mma));
        dim3 grid_cute(M / BLK_M, N / BLK_N);

        auto cute_op = [&]() {
            matmul_cute_simple<BLK_M, BLK_N, BLK_K>
                <<<grid_cute, block_cute>>>(d_A, d_B, d_C, M, K, N, tiled_mma);
        };

        run_benchmark("CuTe Simple MMA", cute_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // cuBLAS
    // cuBLAS is column-major, so we compute C = A*B in row-major as:
    //   C^T = B^T * A^T  in column-major
    // A row-major MxK matrix is a column-major KxM matrix (no data movement).
    // So: cublasSgemm(N, M, K, B, N, A, K, C, N)
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    auto cublas_op = [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    };

    run_benchmark("cuBLAS", cublas_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);

    // cuBLAS with TF32 tensor cores
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    auto cublas_tf32_op = [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    };

    run_benchmark("cuBLAS TF32", cublas_tf32_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);

    cublasDestroy(handle);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
