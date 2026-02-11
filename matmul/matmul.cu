#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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

// -------------------------------------------------------------------------
// Verification & Benchmarking Helpers
// -------------------------------------------------------------------------

// CPU reference matmul for verification (only called on small matrices)
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++)
                sum += A[r * K + i] * B[i * N + c];
            C[r * N + c] = sum;
        }
}

bool verify(const float* gpu, const float* ref, int M, int N) {
    float max_rel_err = 0.0f, max_abs_err = 0.0f;
    int mismatches = 0;
    float rtol = 1e-5f, atol = 1e-3f;

    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float got = gpu[r * N + c];
            float exp = ref[r * N + c];
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
    }
    printf("  max abs error: %e\n", max_abs_err);
    printf("  max rel error: %e\n", max_rel_err);
    if (mismatches == 0) {
        printf("  PASS: all %d elements match\n", M * N);
        return true;
    }
    printf("  FAIL: %d / %d mismatches\n", mismatches, M * N);
    return false;
}

template <typename Func>
void run_benchmark(const char* name, Func kernel_launch,
                   float* d_C, float* h_C, const float* h_ref,
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
    verify(h_C, h_ref, M, N);
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

    int M = 32, K = 8192, N = 64;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    // Host alloc and init
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_ref = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10);

    // CPU reference
    matmul_cpu(h_A, h_B, h_ref, M, K, N);

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
    run_benchmark("Naive", naive_op, d_C, h_C, h_ref, M, K, N, iterations, benchmark_mode);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
