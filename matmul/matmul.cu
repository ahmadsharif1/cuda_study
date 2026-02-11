#include <cstdio>
#include <cstdlib>
#include <cmath>

// C = A * B
// A is MxK, B is KxN, C is MxN, all row-major.
__global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

int main() {
    int M = 32, K = 8192, N = 64;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    // Host alloc and init
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10);

    // Device alloc
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Launch: one thread per output element
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify full output
    float max_rel_err = 0.0f, max_abs_err = 0.0f;
    int mismatches = 0;
    float rtol = 1e-5f, atol = 1e-3f;

    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float expected = 0.0f;
            for (int i = 0; i < K; i++)
                expected += h_A[r * K + i] * h_B[i * N + c];

            float got = h_C[r * N + c];
            float abs_err = fabsf(got - expected);
            float rel_err = abs_err / fmaxf(fabsf(expected), 1e-6f);

            if (abs_err > atol + rtol * fabsf(expected)) {
                if (mismatches < 5)
                    printf("MISMATCH C[%d][%d]: gpu=%.6f cpu=%.6f (abs=%.6f rel=%.6f)\n",
                           r, c, got, expected, abs_err, rel_err);
                mismatches++;
            }
            max_abs_err = fmaxf(max_abs_err, abs_err);
            max_rel_err = fmaxf(max_rel_err, rel_err);
        }
    }
    printf("max abs error: %e\n", max_abs_err);
    printf("max rel error: %e\n", max_rel_err);
    if (mismatches == 0)
        printf("PASS: all %d elements match\n", M * N);
    else
        printf("FAIL: %d / %d mismatches\n", mismatches, M * N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return mismatches > 0 ? 1 : 0;
}
