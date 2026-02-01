#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

#define TILE_DIM 32
#define BLOCK_ROWS 8

// -------------------------------------------------------------------------
// Kernel Implementations
// -------------------------------------------------------------------------

// CUDA kernel for matrix transpose (Basic)
__global__ void transpose(float *odata, const float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if (x < width && (y + j) < height)
        {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if (x < height && (y + j) < width)
        {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// CUDA kernel for matrix transpose (Vectorized)
__global__ void transpose_vectorized(float *odata, const float *idata, int width, int height)
{
    // Pad shared memory to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Calculate global indices
    // threadIdx.x operates on 4 columns at a time (vectorized)
    int x = blockIdx.x * TILE_DIM + threadIdx.x * 4;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x + 3 < width && y < height)
    {
        // Vectorized load from global memory
        float4 v = reinterpret_cast<const float4*>(&idata[y * width + x])[0];
        
        // Store to shared memory (cannot vector store due to padding/stride)
        tile[threadIdx.y][threadIdx.x * 4 + 0] = v.x;
        tile[threadIdx.y][threadIdx.x * 4 + 1] = v.y;
        tile[threadIdx.y][threadIdx.x * 4 + 2] = v.z;
        tile[threadIdx.y][threadIdx.x * 4 + 3] = v.w;
    }
    // Handle edges if width is not multiple of 4 (optional fallback)
    else if (x < width && y < height)
    {
         for (int k = 0; k < 4 && (x + k) < width; ++k)
            tile[threadIdx.y][threadIdx.x * 4 + k] = idata[y * width + x + k];
    }

    __syncthreads();

    // Transpose indices
    // Input x becomes output y, Input y becomes output x
    x = blockIdx.y * TILE_DIM + threadIdx.x * 4;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x + 3 < height && y < width)
    {
        // Read from shared memory (transposed)
        // We need 4 values that form a contiguous float4 in output
        // Output row y corresponds to Input col y
        // Output cols x..x+3 correspond to Input rows x..x+3
        // So we need tile[x][y], tile[x+1][y]...
        // Note: 'y' here is the original 'x' index inside the tile (0..31)
        // 'x' here includes block offset, so we need local part
        
        int local_x = threadIdx.x * 4; // 0, 4, 8...
        int local_y = threadIdx.y;     // 0..31

        float v0 = tile[local_x + 0][local_y];
        float v1 = tile[local_x + 1][local_y];
        float v2 = tile[local_x + 2][local_y];
        float v3 = tile[local_x + 3][local_y];

        // Vectorized store to global memory
        reinterpret_cast<float4*>(&odata[y * height + x])[0] = make_float4(v0, v1, v2, v3);
    }
    else if (x < height && y < width)
    {
        // Edge case fallback
         int local_x = threadIdx.x * 4;
         int local_y = threadIdx.y;
         
         for (int k = 0; k < 4 && (x + k) < height; ++k)
            odata[y * height + x + k] = tile[local_x + k][local_y];
    }
}

// CUDA kernel for matrix transpose (CuTe)
template <typename T, int TileM, int TileN>
__global__ void transpose_cute_kernel(T* odata, const T* idata, int m, int n)
{
    // Define the global layouts for input (Row Major) and output (Col Major)
    auto g_layout_in  = make_layout(make_shape(m, n), GenRowMajor{});
    auto g_layout_out = make_layout(make_shape(m, n), GenColMajor{});

    // Create global tensors
    Tensor g_in  = make_tensor(make_gmem_ptr(idata), g_layout_in);
    Tensor g_out = make_tensor(make_gmem_ptr(odata), g_layout_out);

    // Define tile shape and partition the global tensors
    auto tile_shape = make_shape(Int<TileM>{}, Int<TileN>{});
    auto g_tile_in  = local_tile(g_in,  tile_shape, make_coord(blockIdx.y, blockIdx.x));
    auto g_tile_out = local_tile(g_out, tile_shape, make_coord(blockIdx.y, blockIdx.x));

    // Shared memory tile and layout
    __shared__ T smem_data[TileM * TileN];
    auto s_layout = make_layout(tile_shape, GenRowMajor{});
    auto s_tile   = make_tensor(make_smem_ptr(smem_data), s_layout);

    // Thread partitioning
    auto thr_layout = make_layout(make_shape(Int<BLOCK_ROWS>{}, Int<TILE_DIM>{}));
    auto t_g_in  = local_partition(g_tile_in,  thr_layout, threadIdx.x + threadIdx.y * TILE_DIM);
    auto t_s_in  = local_partition(s_tile,     thr_layout, threadIdx.x + threadIdx.y * TILE_DIM);

    // Load from GMEM to SMEM
    copy(t_g_in, t_s_in);
    __syncthreads();

    // Now copy from SMEM to GMEM
    auto t_s_out = local_partition(s_tile,    thr_layout, threadIdx.x + threadIdx.y * TILE_DIM);
    auto t_g_out = local_partition(g_tile_out, thr_layout, threadIdx.x + threadIdx.y * TILE_DIM);

    copy(t_s_out, t_g_out);
}

// -------------------------------------------------------------------------
// Benchmarking & Verification Helpers
// -------------------------------------------------------------------------

void check_cuda(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", func, cudaGetErrorString(result));
        exit(1);
    }
}

template <typename Func>
void run_benchmark(const char* name, Func kernel_launch, 
                   float* d_idata, float* d_odata, 
                   float* h_idata, float* h_odata, 
                   int width, int height, int iterations) {
    
    int size = width * height * sizeof(float);
    
    // Warmup
    kernel_launch();
    check_cuda(cudaDeviceSynchronize(), "Warmup Kernel Launch");
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Reset output memory before timing run (optional, but clean)
    cudaMemset(d_odata, 0, size);
    
    cudaEventRecord(start, 0);
    for(int i = 0; i < iterations; ++i) {
        kernel_launch();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / iterations;
    
    // Verify results
    // Note: We verify the state after the last iteration
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);
    bool success = true;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (h_odata[j * height + i] != h_idata[i * width + j]) {
                success = false;
                // Print first failure for debugging
                // printf("Mismatch at (%d, %d): expected %f, got %f\n", i, j, h_idata[i * width + j], h_odata[j * height + i]); 
                break;
            }
        }
        if (!success) break;
    }
    
    // Report
    printf("----------------------------------------------------------------\n");
    printf("%-20s\n", name);
    printf("  Verification : %s\n", success ? "PASS" : "FAIL");
    
    float bandwidth = 2.0f * size / (avg_time / 1000.0f) / 1.0e9f;
    printf("  Bandwidth    : %.2f GB/s\n", bandwidth);
    printf("  Avg Time     : %.4f ms\n", avg_time);
    printf("----------------------------------------------------------------\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

int main(int argc, char **argv)
{
    // Parse command line arguments
    int iterations = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) {
                iterations = atoi(argv[i + 1]);
                i++;
            }
        }
    }
    
    printf("Running with %d iteration(s) (after 1 warmup).\n", iterations);

    int width = 2048, height = 2048;
    int size = width * height * sizeof(float);

    // Allocate host memory
    float *h_idata = (float *)malloc(size);
    float *h_odata = (float *)malloc(size);

    // Initialize host data
    for (int i = 0; i < width * height; i++)
    {
        h_idata[i] = (float)i;
    }

    // Allocate device memory
    float *d_idata, *d_odata;
    check_cuda(cudaMalloc((void **)&d_idata, size), "cudaMalloc d_idata");
    check_cuda(cudaMalloc((void **)&d_odata, size), "cudaMalloc d_odata");

    // Copy data from host to device
    check_cuda(cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice");

    // Define grid and block dimensions
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    // Define kernel lambdas
    auto cute_op = [&]() { 
        transpose_cute_kernel<float, TILE_DIM, TILE_DIM><<<dimGrid, dimBlock>>>(d_odata, d_idata, height, width); 
    };

    auto basic_op = [&]() { 
        transpose<<<dimGrid, dimBlock>>>(d_odata, d_idata, width, height); 
    };

    // Vectorized Kernel Launch Configuration
    // We use TILE_DIM/4 threads in X because each thread processes 4 elements
    // We use TILE_DIM threads in Y to cover the full tile height without looping
    dim3 dimBlockVectorized(TILE_DIM / 4, TILE_DIM, 1);
    auto vectorized_op = [&]() {
        transpose_vectorized<<<dimGrid, dimBlockVectorized>>>(d_odata, d_idata, width, height);
    };

    // Run Benchmarks
    // 1. CuTe Kernel
    run_benchmark("CuTe Kernel", cute_op, d_idata, d_odata, h_idata, h_odata, width, height, iterations);

    // 2. Basic Kernel
    run_benchmark("Basic Kernel", basic_op, d_idata, d_odata, h_idata, h_odata, width, height, iterations);

    // 3. Vectorized Kernel
    run_benchmark("Vectorized Kernel", vectorized_op, d_idata, d_odata, h_idata, h_odata, width, height, iterations);

    // Free resources
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
