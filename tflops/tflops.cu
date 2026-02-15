#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <cmath>

// H100 SXM5 TF32 Tensor Core peak: 989 TFLOPS (without sparsity)
static constexpr double H100_TF32_PEAK_TFLOPS = 989.0;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        exit(1); \
    } \
} while(0)

struct BenchConfig {
    int M, N, K;
    bool a_col_major;  // false = row-major
    bool b_col_major;
    bool c_col_major;
    const char* label;
};

double benchmark_cublas_tf32(cublasHandle_t handle, const BenchConfig& cfg, int warmup, int iters) {
    int M = cfg.M, N = cfg.N, K = cfg.K;

    // Allocate
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));

    // Initialize with random data
    {
        size_t total = (size_t)M*K + (size_t)K*N + (size_t)M*N;
        std::vector<float> h(total);
        for (size_t i = 0; i < total; i++) h[i] = (float)(rand() % 1000) / 1000.0f;
        CHECK_CUDA(cudaMemcpy(d_A, h.data(), (size_t)M*K*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h.data() + M*K, (size_t)K*N*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C, h.data() + M*K + K*N, (size_t)M*N*sizeof(float), cudaMemcpyHostToDevice));
    }

    // Enable TF32
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column-major natively.
    // We want: C = A * B  where A is MxK, B is KxN, C is MxN
    //
    // cuBLAS computes: C_col = op(A_col) * op(B_col)
    //
    // For row-major matrix X (MxN), the data layout is the same as a
    // column-major X^T (NxM). So we use the transpose trick:
    //
    // If C is row-major: C_row(MxN) = A_row(MxK) * B_row(KxN)
    //   In col-major view: C^T(NxM) = B^T(NxK) * A^T(KxM)
    //   => cublasSgemm(N, M, K, B, ldb, A, lda, C, ldc)
    //     with opB=N, opA=N, ldb=N, lda=K, ldc=N
    //
    // For col-major inputs, we can use them directly.
    // For mixed cases, we adjust op and leading dimensions accordingly.

    // Determine cuBLAS parameters based on storage orders
    cublasOperation_t opA_blas, opB_blas;
    int lda_blas, ldb_blas, ldc_blas;
    const float *ptrA_blas, *ptrB_blas;
    int m_blas, n_blas, k_blas;

    if (cfg.c_col_major) {
        // C is col-major: cuBLAS computes C directly
        // C(MxN,col) = op(A) * op(B)
        m_blas = M;
        n_blas = N;
        k_blas = K;
        ptrA_blas = d_A;
        ptrB_blas = d_B;

        if (cfg.a_col_major) {
            opA_blas = CUBLAS_OP_N;  // A is col-major MxK, ld=M
            lda_blas = M;
        } else {
            opA_blas = CUBLAS_OP_T;  // A is row-major MxK = col-major KxM, ld=K, need transpose
            lda_blas = K;
        }

        if (cfg.b_col_major) {
            opB_blas = CUBLAS_OP_N;  // B is col-major KxN, ld=K
            ldb_blas = K;
        } else {
            opB_blas = CUBLAS_OP_T;  // B is row-major KxN = col-major NxK, ld=N, need transpose
            ldb_blas = N;
        }

        ldc_blas = M;  // C col-major MxN, ld=M
    } else {
        // C is row-major: use transpose trick
        // C_row(MxN) means C^T_col(NxM) = B^T * A^T
        // So we swap A<->B and M<->N
        m_blas = N;  // cuBLAS "M"
        n_blas = M;  // cuBLAS "N"
        k_blas = K;

        // cuBLAS's "A" is our B (transposed view)
        // cuBLAS's "B" is our A (transposed view)
        ptrA_blas = d_B;  // cuBLAS "A" = our B
        ptrB_blas = d_A;  // cuBLAS "B" = our A

        if (cfg.b_col_major) {
            // Our B is col-major KxN. In transpose trick, cuBLAS sees it as "A".
            // cuBLAS "A" should be m_blas(=N) x k_blas(=K).
            // B_col(KxN) transposed = NxK col-major... we need op=T, ld=K
            opA_blas = CUBLAS_OP_T;
            lda_blas = K;
        } else {
            // Our B is row-major KxN = col-major NxK. cuBLAS "A" is NxK, op=N, ld=N
            opA_blas = CUBLAS_OP_N;
            lda_blas = N;
        }

        if (cfg.a_col_major) {
            // Our A is col-major MxK. cuBLAS "B" should be k_blas(=K) x n_blas(=M).
            // A_col(MxK) transposed = KxM col-major... we need op=T, ld=M
            opB_blas = CUBLAS_OP_T;
            ldb_blas = M;
        } else {
            // Our A is row-major MxK = col-major KxM. cuBLAS "B" is KxM, op=N, ld=K
            opB_blas = CUBLAS_OP_N;
            ldb_blas = K;
        }

        ldc_blas = N;  // C^T col-major is NxM, ld=N
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, opA_blas, opB_blas,
            m_blas, n_blas, k_blas,
            &alpha, ptrA_blas, lda_blas, ptrB_blas, ldb_blas,
            &beta, d_C, ldc_blas));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, opA_blas, opB_blas,
            m_blas, n_blas, k_blas,
            &alpha, ptrA_blas, lda_blas, ptrB_blas, ldb_blas,
            &beta, d_C, ldc_blas));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double avg_ms = ms / iters;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return tflops;
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int warmup = 10;
    int iters = 50;

    // Matrix sizes to test
    std::vector<std::pair<int,const char*>> sizes = {
        {4096, "4096"},
        {8192, "8192"},
        {16384, "16384"},
    };

    // Storage layout combinations
    struct LayoutCombo {
        bool a_col, b_col, c_col;
        const char* desc;
    };
    std::vector<LayoutCombo> layouts = {
        {false, false, false, "A:row  B:row  C:row"},
        {true,  true,  true,  "A:col  B:col  C:col"},
        {false, true,  false, "A:row  B:col  C:row"},
        {true,  false, true,  "A:col  B:row  C:col"},
        {true,  true,  false, "A:col  B:col  C:row"},
        {false, false, true,  "A:row  B:row  C:col"},
        {true,  false, false, "A:col  B:row  C:row"},
        {false, true,  true,  "A:row  B:col  C:col"},
    };

    printf("H100 SXM5 TF32 Tensor Core Peak: %.0f TFLOPS (without sparsity)\n", H100_TF32_PEAK_TFLOPS);
    printf("Warmup: %d iterations, Benchmark: %d iterations\n\n", warmup, iters);

    // Also test non-square matrices
    struct MatSize { int M, N, K; const char* label; };
    std::vector<MatSize> all_sizes = {
        {4096,  4096,  4096,  "4096x4096x4096"},
        {4096,  8192,  4096,  "4096x8192x4096"},
        {8192,  8192,  8192,  "8192x8192x8192"},
        {8192,  16384, 8192,  "8192x16384x8192"},
        {16384, 16384, 16384, "16384x16384x16384"},
        // The size from matmul.cu
        {4096,  8192,  8192,  "4096x8192x8192 (matmul.cu)"},
    };

    printf("%-28s  %-22s  %10s  %8s\n", "Matrix Size", "Layout", "TFLOPS", "% Peak");
    printf("%-28s  %-22s  %10s  %8s\n",
           "----------------------------", "----------------------", "----------", "--------");

    double best_tflops = 0;
    const char* best_label = "";
    const char* best_layout = "";

    for (auto& sz : all_sizes) {
        for (auto& lay : layouts) {
            BenchConfig cfg;
            cfg.M = sz.M;
            cfg.N = sz.N;
            cfg.K = sz.K;
            cfg.a_col_major = lay.a_col;
            cfg.b_col_major = lay.b_col;
            cfg.c_col_major = lay.c_col;
            cfg.label = sz.label;

            double tflops = benchmark_cublas_tf32(handle, cfg, warmup, iters);
            double pct = 100.0 * tflops / H100_TF32_PEAK_TFLOPS;

            printf("%-28s  %-22s  %10.1f  %7.1f%%\n",
                   sz.label, lay.desc, tflops, pct);

            if (tflops > best_tflops) {
                best_tflops = tflops;
                best_label = sz.label;
                best_layout = lay.desc;
            }
        }
        printf("\n");
    }

    printf("============================================================\n");
    printf("BEST: %-28s %-22s %.1f TFLOPS (%.1f%% of peak)\n",
           best_label, best_layout, best_tflops, 100.0 * best_tflops / H100_TF32_PEAK_TFLOPS);

    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
