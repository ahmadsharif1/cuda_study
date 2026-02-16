#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

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

// Shared memory tiled GEMM: load BLK×BLK tiles of A and B into smem,
// compute partial dot products, slide along K.  Pure FP32, no tensor cores.
template <int BLK>
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float As[BLK][BLK];
    __shared__ float Bs[BLK][BLK];

    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < K; t += BLK) {
        // Collaborative load: each thread loads one element of each tile
        As[threadIdx.y][threadIdx.x] = (row < M && t + threadIdx.x < K)
                                        ? A[row * K + t + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t + threadIdx.y < K && col < N)
                                        ? B[(t + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < BLK; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Register-tiled GEMM: each thread computes a TM×TN sub-block of outputs.
// The outer-product inner loop does TM+TN smem loads for TM×TN FMAs,
// dramatically improving the compute-to-load ratio vs the basic tiled kernel.
template <int BLK_M, int BLK_N, int BLK_K, int TM, int TN>
__global__ void matmul_tiled_reg(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float As[BLK_M][BLK_K];
    __shared__ float Bs[BLK_K][BLK_N];

    // Thread-block tile origin
    int block_row = blockIdx.y * BLK_M;
    int block_col = blockIdx.x * BLK_N;

    // Thread indices within the tile grid (each thread owns TM×TN outputs)
    constexpr int THREADS_M = BLK_M / TM;  // 16
    constexpr int THREADS_N = BLK_N / TN;  // 16
    int tid = threadIdx.x;
    int thread_row = (tid / THREADS_N) * TM;
    int thread_col = (tid % THREADS_N) * TN;

    // Accumulator registers
    float accum[TM][TN] = {};

    // Number of float elements each thread must load to cooperatively fill smem
    constexpr int BLOCK_SIZE = THREADS_M * THREADS_N;  // 256
    constexpr int A_TILE_ELEMS = BLK_M * BLK_K;        // 128*8 = 1024
    constexpr int B_TILE_ELEMS = BLK_K * BLK_N;        // 8*128 = 1024
    constexpr int A_LOADS_PER_THREAD = A_TILE_ELEMS / BLOCK_SIZE;  // 4
    constexpr int B_LOADS_PER_THREAD = B_TILE_ELEMS / BLOCK_SIZE;  // 4

    for (int t = 0; t < K; t += BLK_K) {
        // Cooperative load of A tile into shared memory
        #pragma unroll
        for (int i = 0; i < A_LOADS_PER_THREAD; i++) {
            int idx = tid + i * BLOCK_SIZE;
            int a_row = idx / BLK_K;
            int a_col = idx % BLK_K;
            int global_row = block_row + a_row;
            int global_col = t + a_col;
            As[a_row][a_col] = (global_row < M && global_col < K)
                                ? A[global_row * K + global_col] : 0.0f;
        }

        // Cooperative load of B tile into shared memory
        #pragma unroll
        for (int i = 0; i < B_LOADS_PER_THREAD; i++) {
            int idx = tid + i * BLOCK_SIZE;
            int b_row = idx / BLK_N;
            int b_col = idx % BLK_N;
            int global_row = t + b_row;
            int global_col = block_col + b_col;
            Bs[b_row][b_col] = (global_row < K && global_col < N)
                                ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        // Outer-product accumulation: TM+TN loads → TM×TN FMAs per k step
        #pragma unroll
        for (int k = 0; k < BLK_K; k++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = As[thread_row + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = Bs[k][thread_col + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    accum[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = block_row + thread_row + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int col = block_col + thread_col + j;
            if (col < N)
                C[row * N + col] = accum[i][j];
        }
    }
}

// Register-tiled GEMM with vectorized global memory loads (float4) and
// smem padding to avoid bank conflicts.
template <int BLK_M, int BLK_N, int BLK_K, int TM, int TN>
__global__ void matmul_tiled_vec(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int PAD = 4;
    __shared__ float As[BLK_M][BLK_K + PAD];
    __shared__ float Bs[BLK_K][BLK_N + PAD];

    int block_row = blockIdx.y * BLK_M;
    int block_col = blockIdx.x * BLK_N;

    constexpr int THREADS_M = BLK_M / TM;
    constexpr int THREADS_N = BLK_N / TN;
    int tid = threadIdx.x;
    int thread_row = (tid / THREADS_N) * TM;
    int thread_col = (tid % THREADS_N) * TN;

    float accum[TM][TN] = {};

    constexpr int BLOCK_SIZE = THREADS_M * THREADS_N;

    // A tile: BLK_M × BLK_K = 128 × 8 = 1024 floats = 256 float4s
    // Each thread loads 256/256 = 1 float4 for A
    constexpr int A_FLOAT4S = (BLK_M * BLK_K) / 4;
    constexpr int A_VEC_PER_THREAD = A_FLOAT4S / BLOCK_SIZE;

    // B tile: BLK_K × BLK_N = 8 × 128 = 1024 floats = 256 float4s
    constexpr int B_FLOAT4S = (BLK_K * BLK_N) / 4;
    constexpr int B_VEC_PER_THREAD = B_FLOAT4S / BLOCK_SIZE;

    for (int t = 0; t < K; t += BLK_K) {
        // Vectorized load of A tile: treat A row as float4s along K dimension
        // A is BLK_M×BLK_K, BLK_K=8 so 2 float4s per row, 128 rows = 256 float4s
        #pragma unroll
        for (int i = 0; i < A_VEC_PER_THREAD; i++) {
            int vec_idx = tid + i * BLOCK_SIZE;
            // Each row has BLK_K/4 = 2 float4s
            int a_row = vec_idx / (BLK_K / 4);
            int a_vec_col = vec_idx % (BLK_K / 4);
            int a_col = a_vec_col * 4;
            int global_row = block_row + a_row;
            int global_col = t + a_col;
            if (global_row < M && global_col + 3 < K) {
                float4 val = reinterpret_cast<const float4*>(&A[global_row * K + global_col])[0];
                As[a_row][a_col]     = val.x;
                As[a_row][a_col + 1] = val.y;
                As[a_row][a_col + 2] = val.z;
                As[a_row][a_col + 3] = val.w;
            } else {
                // Scalar fallback for boundary
                for (int s = 0; s < 4; s++) {
                    int gc = global_col + s;
                    As[a_row][a_col + s] = (global_row < M && gc < K) ? A[global_row * K + gc] : 0.0f;
                }
            }
        }

        // Vectorized load of B tile: B is BLK_K×BLK_N, vectorize along N (contiguous)
        // Each row has BLK_N/4 = 32 float4s, 8 rows = 256 float4s
        #pragma unroll
        for (int i = 0; i < B_VEC_PER_THREAD; i++) {
            int vec_idx = tid + i * BLOCK_SIZE;
            int b_row = vec_idx / (BLK_N / 4);
            int b_vec_col = vec_idx % (BLK_N / 4);
            int b_col = b_vec_col * 4;
            int global_row = t + b_row;
            int global_col = block_col + b_col;
            if (global_row < K && global_col + 3 < N) {
                float4 val = reinterpret_cast<const float4*>(&B[global_row * N + global_col])[0];
                Bs[b_row][b_col]     = val.x;
                Bs[b_row][b_col + 1] = val.y;
                Bs[b_row][b_col + 2] = val.z;
                Bs[b_row][b_col + 3] = val.w;
            } else {
                for (int s = 0; s < 4; s++) {
                    int gc = global_col + s;
                    Bs[b_row][b_col + s] = (global_row < K && gc < N) ? B[global_row * N + gc] : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLK_K; k++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = As[thread_row + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = Bs[k][thread_col + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    accum[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = block_row + thread_row + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int col = block_col + thread_col + j;
            if (col < N)
                C[row * N + col] = accum[i][j];
        }
    }
}

// Register-tiled GEMM with vectorized loads and double-buffered shared memory.
// Overlaps global memory loads of the next K-tile with compute on the current tile.
template <int BLK_M, int BLK_N, int BLK_K, int TM, int TN>
__global__ void matmul_tiled_pipe(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int PAD = 4;
    __shared__ float As[2][BLK_M][BLK_K + PAD];
    __shared__ float Bs[2][BLK_K][BLK_N + PAD];

    int block_row = blockIdx.y * BLK_M;
    int block_col = blockIdx.x * BLK_N;

    constexpr int THREADS_M = BLK_M / TM;
    constexpr int THREADS_N = BLK_N / TN;
    int tid = threadIdx.x;
    int thread_row = (tid / THREADS_N) * TM;
    int thread_col = (tid % THREADS_N) * TN;

    float accum[TM][TN] = {};

    constexpr int BLOCK_SIZE = THREADS_M * THREADS_N;
    constexpr int A_FLOAT4S = (BLK_M * BLK_K) / 4;
    constexpr int A_VEC_PER_THREAD = A_FLOAT4S / BLOCK_SIZE;
    constexpr int B_FLOAT4S = (BLK_K * BLK_N) / 4;
    constexpr int B_VEC_PER_THREAD = B_FLOAT4S / BLOCK_SIZE;

    int num_tiles = (K + BLK_K - 1) / BLK_K;

    // Helper lambdas for loading tiles into a specific buffer
    auto load_A_tile = [&](int t, int buf) {
        #pragma unroll
        for (int i = 0; i < A_VEC_PER_THREAD; i++) {
            int vec_idx = tid + i * BLOCK_SIZE;
            int a_row = vec_idx / (BLK_K / 4);
            int a_vec_col = vec_idx % (BLK_K / 4);
            int a_col = a_vec_col * 4;
            int global_row = block_row + a_row;
            int global_col = t + a_col;
            if (global_row < M && global_col + 3 < K) {
                float4 val = reinterpret_cast<const float4*>(&A[global_row * K + global_col])[0];
                As[buf][a_row][a_col]     = val.x;
                As[buf][a_row][a_col + 1] = val.y;
                As[buf][a_row][a_col + 2] = val.z;
                As[buf][a_row][a_col + 3] = val.w;
            } else {
                for (int s = 0; s < 4; s++) {
                    int gc = global_col + s;
                    As[buf][a_row][a_col + s] = (global_row < M && gc < K) ? A[global_row * K + gc] : 0.0f;
                }
            }
        }
    };

    auto load_B_tile = [&](int t, int buf) {
        #pragma unroll
        for (int i = 0; i < B_VEC_PER_THREAD; i++) {
            int vec_idx = tid + i * BLOCK_SIZE;
            int b_row = vec_idx / (BLK_N / 4);
            int b_vec_col = vec_idx % (BLK_N / 4);
            int b_col = b_vec_col * 4;
            int global_row = t + b_row;
            int global_col = block_col + b_col;
            if (global_row < K && global_col + 3 < N) {
                float4 val = reinterpret_cast<const float4*>(&B[global_row * N + global_col])[0];
                Bs[buf][b_row][b_col]     = val.x;
                Bs[buf][b_row][b_col + 1] = val.y;
                Bs[buf][b_row][b_col + 2] = val.z;
                Bs[buf][b_row][b_col + 3] = val.w;
            } else {
                for (int s = 0; s < 4; s++) {
                    int gc = global_col + s;
                    Bs[buf][b_row][b_col + s] = (global_row < K && gc < N) ? B[global_row * N + gc] : 0.0f;
                }
            }
        }
    };

    auto compute_tile = [&](int buf) {
        #pragma unroll
        for (int k = 0; k < BLK_K; k++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = As[buf][thread_row + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = Bs[buf][k][thread_col + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    accum[i][j] += a_reg[i] * b_reg[j];
        }
    };

    // Prologue: load first tile into buffer 0
    load_A_tile(0, 0);
    load_B_tile(0, 0);
    __syncthreads();

    for (int tile = 0; tile < num_tiles; tile++) {
        int cur_buf = tile & 1;
        int nxt_buf = 1 - cur_buf;

        // Start loading next tile into the other buffer
        if (tile + 1 < num_tiles) {
            load_A_tile((tile + 1) * BLK_K, nxt_buf);
            load_B_tile((tile + 1) * BLK_K, nxt_buf);
        }

        // Compute on current buffer (loads and compute overlap at the warp level
        // because loads go through the memory pipeline while FMAs use the math pipe)
        compute_tile(cur_buf);

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = block_row + thread_row + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int col = block_col + thread_col + j;
            if (col < N)
                C[row * N + col] = accum[i][j];
        }
    }
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

// CuTe SMEM kernel: global -> shared memory (cp.async) -> registers -> MMA
template <int BLK_M, int BLK_N, int BLK_K,
          class TiledMMA, class CopyA, class CopyB,
          class SmemLayoutA, class SmemLayoutB>
__global__ void matmul_cute_smem(const float* A_ptr, const float* B_ptr, float* C_ptr,
                                  int M, int K, int N,
                                  TiledMMA tiled_mma, CopyA copy_a, CopyB copy_b,
                                  SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
    extern __shared__ float smem[];

    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), make_stride(Int<1>{}, N));
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(blockIdx.x, _));
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}), make_coord(blockIdx.y, _));
    auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(blockIdx.x, blockIdx.y));

    auto sA = make_tensor(make_smem_ptr(smem), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smem + cosize(sA_layout)), sB_layout);

    // Copy partitions: global -> shared
    auto thr_copy_a = copy_a.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tAsA = thr_copy_a.partition_D(sA);

    auto thr_copy_b = copy_b.get_thread_slice(threadIdx.x);
    auto tBgB = thr_copy_b.partition_S(gB);
    auto tBsB = thr_copy_b.partition_D(sB);

    // MMA partitions: shared -> registers
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);
    auto tCgC = thr_mma.partition_C(gC);
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);

    int k_tile_max = size<2>(gA);
    for (int k = 0; k < k_tile_max; k++) {
        copy(copy_a, tAgA(_, _, _, k), tAsA);
        copy(copy_b, tBgB(_, _, _, k), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        copy(tCsA, tCrA);
        copy(tCsB, tCrB);
        gemm(tiled_mma, tCrA, tCrB, tCrC);
        __syncthreads();
    }

    copy(tCrC, tCgC);
}

// Persistent kernel: each block loops over multiple output tiles.
// Column-major tile ordering so blocks sharing B data are co-scheduled,
// maximizing L2 cache reuse.
template <int BLK_M, int BLK_N, int BLK_K,
          class TiledMMA, class CopyA, class CopyB,
          class SmemLayoutA, class SmemLayoutB>
__global__ void matmul_cute_persistent(const float* A_ptr, const float* B_ptr, float* C_ptr,
                                        int M, int K, int N,
                                        TiledMMA tiled_mma, CopyA copy_a, CopyB copy_b,
                                        SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
    extern __shared__ float smem[];

    int num_m_tiles = M / BLK_M;
    int num_n_tiles = N / BLK_N;
    int total_tiles = num_m_tiles * num_n_tiles;

    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), make_stride(Int<1>{}, N));
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    auto sA = make_tensor(make_smem_ptr(smem), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smem + cosize(sA_layout)), sB_layout);

    // Smem destination partitions (reused across tiles)
    auto thr_copy_a = copy_a.get_thread_slice(threadIdx.x);
    auto tAsA = thr_copy_a.partition_D(sA);

    auto thr_copy_b = copy_b.get_thread_slice(threadIdx.x);
    auto tBsB = thr_copy_b.partition_D(sB);

    // MMA partitions from smem (reused across tiles)
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);

    // Each block handles a contiguous range of tiles (L2-friendly)
    int tiles_per_block = (total_tiles + gridDim.x - 1) / gridDim.x;
    int tile_start = blockIdx.x * tiles_per_block;
    int tile_end   = min(tile_start + tiles_per_block, total_tiles);

    for (int tile_idx = tile_start; tile_idx < tile_end; tile_idx++) {
        // Column-major: M varies first. Consecutive tiles share B (same N-column).
        int tile_m = tile_idx % num_m_tiles;
        int tile_n = tile_idx / num_m_tiles;

        auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(tile_m, _));
        auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}), make_coord(tile_n, _));
        auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(tile_m, tile_n));

        auto tAgA = thr_copy_a.partition_S(gA);
        auto tBgB = thr_copy_b.partition_S(gB);
        auto tCgC = thr_mma.partition_C(gC);
        auto tCrC = thr_mma.partition_fragment_C(gC);
        clear(tCrC);

        int k_tile_max = size<2>(gA);
        for (int k = 0; k < k_tile_max; k++) {
            copy(copy_a, tAgA(_, _, _, k), tAsA);
            copy(copy_b, tBgB(_, _, _, k), tBsB);
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            copy(tCsA, tCrA);
            copy(tCsB, tCrB);
            gemm(tiled_mma, tCrA, tCrB, tCrC);
            __syncthreads();
        }

        copy(tCrC, tCgC);
    }
}

// CuTe SMEM kernel with CTA swizzle for L2 cache reuse.
// 1D grid; nearby blocks share A/B data for better L2 hit rate.
template <int BLK_M, int BLK_N, int BLK_K, int SWIZZLE,
          class TiledMMA, class CopyA, class CopyB,
          class SmemLayoutA, class SmemLayoutB>
__global__ void matmul_cute_swizzle(const float* A_ptr, const float* B_ptr, float* C_ptr,
                                     int M, int K, int N,
                                     TiledMMA tiled_mma, CopyA copy_a, CopyB copy_b,
                                     SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
    extern __shared__ float smem[];

    int num_m_tiles = M / BLK_M;
    int group_id     = blockIdx.x / (num_m_tiles * SWIZZLE);
    int first_n      = group_id * SWIZZLE;
    int idx_in_group = blockIdx.x - group_id * (num_m_tiles * SWIZZLE);
    int tile_m = idx_in_group % num_m_tiles;
    int tile_n = first_n + idx_in_group / num_m_tiles;

    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), make_stride(Int<1>{}, N));
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(tile_m, _));
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}), make_coord(tile_n, _));
    auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(tile_m, tile_n));

    auto sA = make_tensor(make_smem_ptr(smem), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smem + cosize(sA_layout)), sB_layout);

    auto thr_copy_a = copy_a.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tAsA = thr_copy_a.partition_D(sA);

    auto thr_copy_b = copy_b.get_thread_slice(threadIdx.x);
    auto tBgB = thr_copy_b.partition_S(gB);
    auto tBsB = thr_copy_b.partition_D(sB);

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);
    auto tCgC = thr_mma.partition_C(gC);
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);

    int k_tile_max = size<2>(gA);
    for (int k = 0; k < k_tile_max; k++) {
        copy(copy_a, tAgA(_, _, _, k), tAsA);
        copy(copy_b, tBgB(_, _, _, k), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        copy(tCsA, tCrA);
        copy(tCsB, tCrB);
        gemm(tiled_mma, tCrA, tCrB, tCrC);
        __syncthreads();
    }

    copy(tCrC, tCgC);
}

// CuTe SMEM kernel with double-buffered pipeline.
// Overlaps global->shared loads with shared->register + MMA compute.
template <int BLK_M, int BLK_N, int BLK_K,
          class TiledMMA, class CopyA, class CopyB,
          class SmemLayoutA, class SmemLayoutB>
__global__ void matmul_cute_pipe(const float* A_ptr, const float* B_ptr, float* C_ptr,
                                  int M, int K, int N,
                                  TiledMMA tiled_mma, CopyA copy_a, CopyB copy_b,
                                  SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
    extern __shared__ float smem[];

    int smem_a_sz = cosize(sA_layout);
    int smem_one  = smem_a_sz + cosize(sB_layout);

    // Double-buffered shared memory: buf0 = [sA0, sB0], buf1 = [sA1, sB1]
    auto sA0 = make_tensor(make_smem_ptr(smem), sA_layout);
    auto sB0 = make_tensor(make_smem_ptr(smem + smem_a_sz), sB_layout);
    auto sA1 = make_tensor(make_smem_ptr(smem + smem_one), sA_layout);
    auto sB1 = make_tensor(make_smem_ptr(smem + smem_one + smem_a_sz), sB_layout);

    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), make_stride(Int<1>{}, N));
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(blockIdx.x, _));
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}), make_coord(blockIdx.y, _));
    auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(blockIdx.x, blockIdx.y));

    auto thr_copy_a = copy_a.get_thread_slice(threadIdx.x);
    auto tAgA  = thr_copy_a.partition_S(gA);
    auto tAsA0 = thr_copy_a.partition_D(sA0);
    auto tAsA1 = thr_copy_a.partition_D(sA1);

    auto thr_copy_b = copy_b.get_thread_slice(threadIdx.x);
    auto tBgB  = thr_copy_b.partition_S(gB);
    auto tBsB0 = thr_copy_b.partition_D(sB0);
    auto tBsB1 = thr_copy_b.partition_D(sB1);

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA0 = thr_mma.partition_A(sA0);
    auto tCsB0 = thr_mma.partition_B(sB0);
    auto tCsA1 = thr_mma.partition_A(sA1);
    auto tCsB1 = thr_mma.partition_B(sB1);
    auto tCgC  = thr_mma.partition_C(gC);
    auto tCrC  = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    auto tCrA = thr_mma.partition_fragment_A(sA0);
    auto tCrB = thr_mma.partition_fragment_B(sB0);

    int k_max = size<2>(gA);

    // Prologue: load first K-tile into buffer 0
    copy(copy_a, tAgA(_, _, _, 0), tAsA0);
    copy(copy_b, tBgB(_, _, _, 0), tBsB0);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for (int k = 0; k < k_max; k++) {
        // Start loading next K-tile into the other buffer
        if (k + 1 < k_max) {
            if (k & 1) {
                copy(copy_a, tAgA(_, _, _, k + 1), tAsA0);
                copy(copy_b, tBgB(_, _, _, k + 1), tBsB0);
            } else {
                copy(copy_a, tAgA(_, _, _, k + 1), tAsA1);
                copy(copy_b, tBgB(_, _, _, k + 1), tBsB1);
            }
            cp_async_fence();
        }

        // Compute on current buffer while next loads in background
        if (k & 1) {
            copy(tCsA1, tCrA);
            copy(tCsB1, tCrB);
        } else {
            copy(tCsA0, tCrA);
            copy(tCsB0, tCrB);
        }
        gemm(tiled_mma, tCrA, tCrB, tCrC);

        // Wait for next buffer to be ready
        if (k + 1 < k_max) {
            cp_async_wait<0>();
        }
        __syncthreads();
    }

    copy(tCrC, tCgC);
}

// CuTe SM90 wgmma kernel: global -> swizzled shared memory (cp.async) -> wgmma.
// wgmma reads A and B directly from shared memory via descriptors,
// eliminating the smem->register copy. Requires K-major swizzled smem layouts
// and a pre-transposed B matrix (K contiguous).
template <int BLK_M, int BLK_N, int BLK_K,
          class TiledMMA, class CopyA, class CopyB,
          class SmemLayoutA, class SmemLayoutB>
__global__ void matmul_cute_wgmma(const float* A_ptr, const float* B_T_ptr, float* C_ptr,
                                   int M, int K, int N,
                                   TiledMMA tiled_mma, CopyA copy_a, CopyB copy_b,
                                   SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
    extern __shared__ float smem[];

    // A: (M, K) row-major, K contiguous
    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    // B_T: (N, K) with K contiguous (pre-transposed from K×N)
    auto mB = make_tensor(make_gmem_ptr(B_T_ptr), make_shape(N, K), make_stride(K, Int<1>{}));
    // C: (M, N) row-major
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(blockIdx.x, _));
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}), make_coord(blockIdx.y, _));
    auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(blockIdx.x, blockIdx.y));

    // Two views of shared memory:
    // - float view for cp.async copies (matches global memory type)
    // - tfloat32_t view for wgmma (matches MMA atom's expected operand type)
    auto sA_copy = make_tensor(make_smem_ptr(smem), sA_layout);
    auto sB_copy = make_tensor(make_smem_ptr(smem + cosize(sA_layout)), sB_layout);

    using tfloat32_t = cutlass::tfloat32_t;
    auto sA = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem)), sA_layout);
    auto sB = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem + cosize(sA_layout))), sB_layout);

    // Copy partitions: global (float) -> swizzled smem (float view)
    auto thr_copy_a = copy_a.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tAsA = thr_copy_a.partition_D(sA_copy);

    auto thr_copy_b = copy_b.get_thread_slice(threadIdx.x);
    auto tBgB = thr_copy_b.partition_S(gB);
    auto tBsB = thr_copy_b.partition_D(sB_copy);

    // MMA partitions: smem descriptors from tfloat32_t view
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);
    auto tCgC = thr_mma.partition_C(gC);
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    int k_tile_max = size<2>(gA);
    for (int k = 0; k < k_tile_max; k++) {
        // Load A and B tiles into swizzled smem via cp.async
        copy(copy_a, tAgA(_, _, _, k), tAsA);
        copy(copy_b, tBgB(_, _, _, k), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // wgmma: reads A, B from smem descriptors; accumulates in registers
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(tiled_mma, tCsA, tCsB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);
        __syncthreads();
    }

    copy(tCrC, tCgC);
}

// CuTe SM90 wgmma kernel with double-buffered pipeline.
// Overlaps cp.async loads of the next k-tile with wgmma compute on the current tile.
// Uses 2x shared memory buffers and alternates between them.
// SWIZZLE > 0: use 1D grid with CTA swizzle for L2 cache reuse.
template <int BLK_M, int BLK_N, int BLK_K, int SWIZZLE = 0,
          class TiledMMA, class CopyA, class CopyB,
          class SmemLayoutA, class SmemLayoutB>
__global__ void matmul_cute_wgmma_pipe(const float* A_ptr, const float* B_T_ptr, float* C_ptr,
                                        int M, int K, int N,
                                        TiledMMA tiled_mma, CopyA copy_a, CopyB copy_b,
                                        SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
    extern __shared__ float smem[];

    int smem_a_sz = cosize(sA_layout);
    int smem_one  = smem_a_sz + cosize(sB_layout);  // one buffer = sA + sB

    // Compute tile coordinates: 2D grid or 1D swizzled grid
    int tile_m, tile_n;
    if constexpr (SWIZZLE > 0) {
        int num_m_tiles = M / BLK_M;
        int group_id     = blockIdx.x / (num_m_tiles * SWIZZLE);
        int first_n      = group_id * SWIZZLE;
        int idx_in_group = blockIdx.x - group_id * (num_m_tiles * SWIZZLE);
        tile_m = idx_in_group % num_m_tiles;
        tile_n = first_n + idx_in_group / num_m_tiles;
    } else {
        tile_m = blockIdx.x;
        tile_n = blockIdx.y;
    }

    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B_T_ptr), make_shape(N, K), make_stride(K, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(tile_m, _));
    auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}), make_coord(tile_n, _));
    auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(tile_m, tile_n));

    using tfloat32_t = cutlass::tfloat32_t;

    // Double buffers: float views for cp.async, tfloat32_t views for wgmma
    auto sA_copy0 = make_tensor(make_smem_ptr(smem), sA_layout);
    auto sB_copy0 = make_tensor(make_smem_ptr(smem + smem_a_sz), sB_layout);
    auto sA0 = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem)), sA_layout);
    auto sB0 = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem + smem_a_sz)), sB_layout);

    auto sA_copy1 = make_tensor(make_smem_ptr(smem + smem_one), sA_layout);
    auto sB_copy1 = make_tensor(make_smem_ptr(smem + smem_one + smem_a_sz), sB_layout);
    auto sA1 = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem + smem_one)), sA_layout);
    auto sB1 = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem + smem_one + smem_a_sz)), sB_layout);

    // Copy partitions for both buffers
    auto thr_copy_a = copy_a.get_thread_slice(threadIdx.x);
    auto tAgA    = thr_copy_a.partition_S(gA);
    auto tAsA0   = thr_copy_a.partition_D(sA_copy0);
    auto tAsA1   = thr_copy_a.partition_D(sA_copy1);

    auto thr_copy_b = copy_b.get_thread_slice(threadIdx.x);
    auto tBgB    = thr_copy_b.partition_S(gB);
    auto tBsB0   = thr_copy_b.partition_D(sB_copy0);
    auto tBsB1   = thr_copy_b.partition_D(sB_copy1);

    // MMA partitions for both buffers (smem descriptors from tfloat32_t view)
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCsA0 = thr_mma.partition_A(sA0);
    auto tCsB0 = thr_mma.partition_B(sB0);
    auto tCsA1 = thr_mma.partition_A(sA1);
    auto tCsB1 = thr_mma.partition_B(sB1);
    auto tCgC  = thr_mma.partition_C(gC);
    auto tCrC  = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    int k_max = size<2>(gA);

    // Prologue: load k=0 into buffer 0
    copy(copy_a, tAgA(_, _, _, 0), tAsA0);
    copy(copy_b, tBgB(_, _, _, 0), tBsB0);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for (int k = 0; k < k_max; k++) {
        // Start loading next k-tile into the other buffer
        if (k + 1 < k_max) {
            if (k & 1) {
                copy(copy_a, tAgA(_, _, _, k + 1), tAsA0);
                copy(copy_b, tBgB(_, _, _, k + 1), tBsB0);
            } else {
                copy(copy_a, tAgA(_, _, _, k + 1), tAsA1);
                copy(copy_b, tBgB(_, _, _, k + 1), tBsB1);
            }
            cp_async_fence();
        }

        // wgmma on current buffer
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        if (k & 1) {
            gemm(tiled_mma, tCsA1, tCsB1, tCrC);
        } else {
            gemm(tiled_mma, tCsA0, tCsB0, tCrC);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);

        // Wait for next buffer's loads to complete
        if (k + 1 < k_max) {
            cp_async_wait<0>();
        }
        __syncthreads();
    }

    copy(tCrC, tCgC);
}

// Shared storage for TMA kernel: smem arrays + barriers for producer-consumer sync.
// PIPE is the number of pipeline stages.
template <class SmemLayoutA, class SmemLayoutB>
struct TmaSmemStorage {
    alignas(128) cute::ArrayEngine<float, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<float, cosize_v<SmemLayoutB>> B;
    uint64_t tma_barrier[size<2>(SmemLayoutA{})];  // producer (TMA) barriers
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];  // consumer (MMA) barriers
};

// CuTe SM90 wgmma + TMA kernel.
// TMA loads data from global -> swizzled smem using a single thread.
// wgmma reads A,B from smem via descriptors. Multi-stage pipeline with
// barrier-based producer-consumer synchronization.
template <class CtaTiler,
          class TmaA, class TmaB,
          class SmemLayoutA, class SmemLayoutB,
          class TiledMMA>
__global__ static
__launch_bounds__(decltype(size(TiledMMA{}))::value)
void matmul_cute_wgmma_tma(int M, int K, int N,
                            CUTLASS_GRID_CONSTANT TmaA const tma_a,
                            CUTLASS_GRID_CONSTANT TmaB const tma_b,
                            float* C_ptr, TiledMMA tiled_mma) {
    using namespace cute;

    // Shared memory
    extern __shared__ char shared_memory[];
    using SharedStorage = TmaSmemStorage<SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA_float = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB_float = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    // tfloat32_t views for wgmma (same memory, different type)
    using tfloat32_t = cutlass::tfloat32_t;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem.A.begin())), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem.B.begin())), SmemLayoutB{});

    // TMA global tensors (reconstructed from TMA descriptors)
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));  // (M,K)
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));  // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));

    // Tile the global tensors
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, CtaTiler{}, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, CtaTiler{}, cta_coord, Step< X,_1, _1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, CtaTiler{}, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

    // TMA partitions (float view for loading)
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sA_float), group_modes<0,2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sB_float), group_modes<0,2>(gB));

    // Transaction bytes for barrier
    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                        + sizeof(make_tensor_like(tensor<0>(tBsB)));

    auto K_PIPE_MAX = size<1>(tAsA);
    int k_tile_count = size<1>(tAgA);
    int k_tile = 0;

    // Initialize barriers (only warp 0, lane 0)
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx == 0) && lane_predicate) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], size(tiled_mma));
        }
    }
    cluster_sync();

    // Prologue: fill all pipeline stages
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx == 0) && lane_predicate) {
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }

    // MMA partitions (tfloat32_t view for wgmma descriptors)
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);   // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);   // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC);   // (MMA,MMA_M,MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    // Fragments are smem descriptors for SS atoms
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)

    // Main loop
    auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
    auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX) {
        // Wait for TMA producer to complete this stage
        int read_pipe = read_state.index();
        ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

        // wgmma on this stage
        warpgroup_arrive();
        gemm(tiled_mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        // Signal consumption done
        ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        // Issue next TMA load (only thread 0)
        if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0)) {
            int pipe = write_state.index();
            ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++write_state;
        }
        --k_tile_count;
        ++k_tile;
    }

    // Store result
    copy(tCrC, tCgC);
}
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
// CUTLASS 3.0 PingPong Warp-Specialized GEMM (from example 49)
// -------------------------------------------------------------------------
namespace cutlass_ws {
    using         ElementA    = float;
    using         LayoutA     = cutlass::layout::RowMajor;
    constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;

    using         ElementB    = float;
    using         LayoutB     = cutlass::layout::ColumnMajor;
    constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;

    using         ElementC    = float;
    using         LayoutC     = cutlass::layout::RowMajor;
    constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementAccumulator  = float;
    using TileShape           = Shape<_128, _128, _32>;
    using ClusterShape        = Shape<_1, _1, _1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC, AlignmentC,
        ElementC, LayoutC, AlignmentC,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
}

// Case-insensitive substring match: run kernel if any filter matches, or if no filters given.
bool should_run(const char* name, const std::vector<std::string>& filters) {
    if (filters.empty()) return true;
    std::string lower_name(name);
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    for (const auto& f : filters) {
        std::string lower_f(f);
        std::transform(lower_f.begin(), lower_f.end(), lower_f.begin(), ::tolower);
        if (lower_name.find(lower_f) != std::string::npos) return true;
    }
    return false;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

int main(int argc, char** argv) {
    int iterations = 100;
    bool benchmark_mode = false;
    std::vector<std::string> kernel_filters;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = true;
        } else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--kernel") == 0) {
            if (i + 1 < argc) kernel_filters.push_back(argv[++i]);
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

    // Transpose B from K×N row-major to N×K row-major (K contiguous) for wgmma
    float* h_B_T = (float*)malloc(size_B);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            h_B_T[n * K + k] = h_B[k * N + n];
    float* d_B_T;
    cudaMalloc(&d_B_T, size_B);
    cudaMemcpy(d_B_T, h_B_T, size_B, cudaMemcpyHostToDevice);
    free(h_B_T);

    // Kernel configs
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    if (should_run("Naive", kernel_filters)) {
        auto naive_op = [&]() {
            matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        };

        run_benchmark("Naive", naive_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // Tiled shared memory kernel (pure FP32)
    if (should_run("Tiled SMEM", kernel_filters)) {
        constexpr int BLK = 32;
        dim3 block_tiled(BLK, BLK);
        dim3 grid_tiled((N + BLK - 1) / BLK, (M + BLK - 1) / BLK);

        auto tiled_op = [&]() {
            matmul_tiled<BLK><<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, M, K, N);
        };

        run_benchmark("Tiled SMEM", tiled_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // Register-tiled kernel: each thread computes 8×8 outputs (pure FP32)
    if (should_run("Tiled RegTile", kernel_filters)) {
        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 8;
        constexpr int TM = 8, TN = 8;
        constexpr int THREADS = (BLK_M / TM) * (BLK_N / TN);  // 256
        dim3 grid_reg((N + BLK_N - 1) / BLK_N, (M + BLK_M - 1) / BLK_M);

        auto reg_op = [&]() {
            matmul_tiled_reg<BLK_M, BLK_N, BLK_K, TM, TN>
                <<<grid_reg, THREADS>>>(d_A, d_B, d_C, M, K, N);
        };

        run_benchmark("Tiled RegTile", reg_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // Register-tiled + vectorized loads + smem padding (pure FP32)
    if (should_run("Tiled VecLoad", kernel_filters)) {
        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 8;
        constexpr int TM = 8, TN = 8;
        constexpr int THREADS = (BLK_M / TM) * (BLK_N / TN);
        dim3 grid_vec((N + BLK_N - 1) / BLK_N, (M + BLK_M - 1) / BLK_M);

        auto vec_op = [&]() {
            matmul_tiled_vec<BLK_M, BLK_N, BLK_K, TM, TN>
                <<<grid_vec, THREADS>>>(d_A, d_B, d_C, M, K, N);
        };

        run_benchmark("Tiled VecLoad", vec_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // Register-tiled + vectorized loads + double-buffered pipeline (pure FP32)
    if (should_run("Tiled DblBuf", kernel_filters)) {
        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 8;
        constexpr int TM = 8, TN = 8;
        constexpr int THREADS = (BLK_M / TM) * (BLK_N / TN);
        dim3 grid_pipe((N + BLK_N - 1) / BLK_N, (M + BLK_M - 1) / BLK_M);

        auto pipe_op = [&]() {
            matmul_tiled_pipe<BLK_M, BLK_N, BLK_K, TM, TN>
                <<<grid_pipe, THREADS>>>(d_A, d_B, d_C, M, K, N);
        };

        run_benchmark("Tiled DblBuf", pipe_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // CuTe simple MMA kernel (TF32, no shared memory)
    if (should_run("CuTe Simple MMA", kernel_filters)) {
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

    // CuTe shared memory kernel with BLK_K=8
    if (should_run("CuTe SMEM k=8", kernel_filters)) {
        using mma_t = TiledMMA<
            MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
            Layout<Shape<_2, _2, _1>>
        >;
        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 8;
        mma_t tiled_mma;

        // Async 128-bit copies: A vectorizes along K, B vectorizes along N
        using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>;
        auto copy_a = make_tiled_copy(CopyAtom{},
            Layout<Shape<_64, _2>, Stride<_2, _1>>{},
            Layout<Shape< _1, _4>>{});
        auto copy_b = make_tiled_copy(CopyAtom{},
            Layout<Shape<_32, _4>>{},
            Layout<Shape< _4, _1>>{});

        // Padded shared memory layouts to avoid bank conflicts
        // Without padding: bank = k%32, so all M at same K conflict
        // With +4 padding: bank = (m*(BLK_K+4) + k)%32, depends on both m and k
        constexpr int PAD = 4;
        auto sA_layout = make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                                     make_stride(Int<BLK_K + PAD>{}, Int<1>{}));
        auto sB_layout = make_layout(make_shape(Int<BLK_N>{}, Int<BLK_K>{}),
                                     make_stride(Int<1>{}, Int<BLK_N + PAD>{}));

        int smem_size = (cosize(sA_layout) + cosize(sB_layout)) * sizeof(float);
        dim3 block_s(size(tiled_mma));
        dim3 grid_s(M / BLK_M, N / BLK_N);

        auto smem_k8_op = [&]() {
            matmul_cute_smem<BLK_M, BLK_N, BLK_K>
                <<<grid_s, block_s, smem_size>>>(d_A, d_B, d_C, M, K, N,
                    tiled_mma, copy_a, copy_b, sA_layout, sB_layout);
        };

        run_benchmark("CuTe SMEM k=8", smem_k8_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // CuTe shared memory kernel with BLK_K=32
    if (should_run("CuTe SMEM k=32", kernel_filters)) {
        using mma_t = TiledMMA<
            MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
            Layout<Shape<_2, _2, _1>>
        >;
        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 32;
        mma_t tiled_mma;

        using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>;
        auto copy_a = make_tiled_copy(CopyAtom{},
            Layout<Shape<_16, _8>, Stride<_8, _1>>{},
            Layout<Shape< _1, _4>>{});
        auto copy_b = make_tiled_copy(CopyAtom{},
            Layout<Shape<_32, _4>>{},
            Layout<Shape< _4, _1>>{});

        constexpr int PAD = 4;
        auto sA_layout = make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                                     make_stride(Int<BLK_K + PAD>{}, Int<1>{}));
        auto sB_layout = make_layout(make_shape(Int<BLK_N>{}, Int<BLK_K>{}),
                                     make_stride(Int<1>{}, Int<BLK_N + PAD>{}));

        int smem_size = (cosize(sA_layout) + cosize(sB_layout)) * sizeof(float);
        dim3 block_s(size(tiled_mma));
        dim3 grid_s(M / BLK_M, N / BLK_N);

        auto smem_k32_op = [&]() {
            matmul_cute_smem<BLK_M, BLK_N, BLK_K>
                <<<grid_s, block_s, smem_size>>>(d_A, d_B, d_C, M, K, N,
                    tiled_mma, copy_a, copy_b, sA_layout, sB_layout);
        };

        run_benchmark("CuTe SMEM k=32", smem_k32_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // CuTe SM90 wgmma: reads A and B from swizzled smem (no smem->reg copy)
    if (should_run("CuTe WGMMA", kernel_filters)) {
        using MmaAtom = SM90_64x128x8_F32TF32TF32_SS_TN<>;
        auto tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _1, _1>>{});
        // 2 warp groups in M = 128×128 output, 256 threads

        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 32;

        // K-major swizzled smem layouts required by wgmma descriptors
        using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<float>;
        auto sA_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
        auto sB_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BLK_N>{}, Int<BLK_K>{}));

        // Async 128-bit copies with 256 threads, vectorized along K (contiguous)
        using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>;
        auto copy_a = make_tiled_copy(CopyAtom{},
            Layout<Shape<_32, _8>, Stride<_8, _1>>{},
            Layout<Shape< _1, _4>>{});
        auto copy_b = make_tiled_copy(CopyAtom{},
            Layout<Shape<_32, _8>, Stride<_8, _1>>{},
            Layout<Shape< _1, _4>>{});

        int smem_size = (cosize(sA_layout) + cosize(sB_layout)) * sizeof(float);
        dim3 block_wg(size(tiled_mma));  // 256 threads
        dim3 grid_wg(M / BLK_M, N / BLK_N);

        auto wgmma_op = [&]() {
            matmul_cute_wgmma<BLK_M, BLK_N, BLK_K>
                <<<grid_wg, block_wg, smem_size>>>(d_A, d_B_T, d_C, M, K, N,
                    tiled_mma, copy_a, copy_b, sA_layout, sB_layout);
        };

        run_benchmark("CuTe WGMMA", wgmma_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // CuTe SM90 wgmma with double-buffered pipeline: overlap cp.async with wgmma
    if (should_run("CuTe WGMMA Pipe", kernel_filters)) {
        using MmaAtom = SM90_64x128x8_F32TF32TF32_SS_TN<>;
        auto tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _1, _1>>{});

        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 32;

        using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<float>;
        auto sA_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
        auto sB_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BLK_N>{}, Int<BLK_K>{}));

        using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>;
        auto copy_a = make_tiled_copy(CopyAtom{},
            Layout<Shape<_32, _8>, Stride<_8, _1>>{},
            Layout<Shape< _1, _4>>{});
        auto copy_b = make_tiled_copy(CopyAtom{},
            Layout<Shape<_32, _8>, Stride<_8, _1>>{},
            Layout<Shape< _1, _4>>{});

        // Double buffer: 2x the shared memory
        int smem_size = 2 * (cosize(sA_layout) + cosize(sB_layout)) * sizeof(float);
        dim3 block_wgp(size(tiled_mma));
        dim3 grid_wgp(M / BLK_M, N / BLK_N);

        // May need >48KB smem — set dynamic smem limit
        cudaFuncSetAttribute(
            matmul_cute_wgmma_pipe<BLK_M, BLK_N, BLK_K, 0,
                decltype(tiled_mma), decltype(copy_a), decltype(copy_b),
                decltype(sA_layout), decltype(sB_layout)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        auto wgmma_pipe_op = [&]() {
            matmul_cute_wgmma_pipe<BLK_M, BLK_N, BLK_K, 0>
                <<<grid_wgp, block_wgp, smem_size>>>(d_A, d_B_T, d_C, M, K, N,
                    tiled_mma, copy_a, copy_b, sA_layout, sB_layout);
        };

        run_benchmark("CuTe WGMMA Pipe", wgmma_pipe_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // CuTe SM90 wgmma pipe + CTA swizzle: same as above but with L2-friendly tile scheduling
    if (should_run("CuTe WGMMA Pipe+Swiz", kernel_filters)) {
        using MmaAtom = SM90_64x128x8_F32TF32TF32_SS_TN<>;
        auto tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _1, _1>>{});

        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 32;
        constexpr int SWIZZLE = 4;  // group 4 N-tiles together

        using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<float>;
        auto sA_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
        auto sB_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BLK_N>{}, Int<BLK_K>{}));

        using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>;
        auto copy_a = make_tiled_copy(CopyAtom{},
            Layout<Shape<_32, _8>, Stride<_8, _1>>{},
            Layout<Shape< _1, _4>>{});
        auto copy_b = make_tiled_copy(CopyAtom{},
            Layout<Shape<_32, _8>, Stride<_8, _1>>{},
            Layout<Shape< _1, _4>>{});

        int smem_size = 2 * (cosize(sA_layout) + cosize(sB_layout)) * sizeof(float);
        dim3 block_wgs(size(tiled_mma));
        int total_tiles = (M / BLK_M) * (N / BLK_N);
        dim3 grid_wgs(total_tiles);  // 1D grid for swizzle

        cudaFuncSetAttribute(
            matmul_cute_wgmma_pipe<BLK_M, BLK_N, BLK_K, SWIZZLE,
                decltype(tiled_mma), decltype(copy_a), decltype(copy_b),
                decltype(sA_layout), decltype(sB_layout)>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        auto wgmma_swiz_op = [&]() {
            matmul_cute_wgmma_pipe<BLK_M, BLK_N, BLK_K, SWIZZLE>
                <<<grid_wgs, block_wgs, smem_size>>>(d_A, d_B_T, d_C, M, K, N,
                    tiled_mma, copy_a, copy_b, sA_layout, sB_layout);
        };

        run_benchmark("CuTe WGMMA Pipe+Swiz", wgmma_swiz_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // CuTe SM90 wgmma + TMA: hardware-accelerated loads, multi-stage pipeline
    if (should_run("CuTe WGMMA TMA", kernel_filters)) {
        using MmaAtom = SM90_64x128x8_F32TF32TF32_SS_TN<>;
        auto tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _1, _1>>{});

        constexpr int BLK_M = 128, BLK_N = 128, BLK_K = 32;
        constexpr int PIPE = 3;  // pipeline stages
        auto bM = Int<BLK_M>{};
        auto bN = Int<BLK_N>{};
        auto bK = Int<BLK_K>{};
        auto bP = Int<PIPE>{};
        auto cta_tiler = make_shape(bM, bN, bK);

        // Swizzled smem layouts with pipeline dimension (M/N, K, PIPE)
        using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<float>;
        auto sA_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(bM, bK, bP));
        auto sB_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(bN, bK, bP));

        // Create TMA atoms on host: inspect global tensor layout + smem layout
        // A: (M,K) row-major with K contiguous
        Tensor host_mA = make_tensor(make_gmem_ptr(d_A), make_shape(M, K), make_stride(K, Int<1>{}));
        // B_T: (N,K) with K contiguous
        Tensor host_mB = make_tensor(make_gmem_ptr(d_B_T), make_shape(N, K), make_stride(K, Int<1>{}));

        auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, host_mA, sA_layout(_, _, 0), make_shape(bM, bK));
        auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, host_mB, sB_layout(_, _, 0), make_shape(bN, bK));

        using SharedStorage = TmaSmemStorage<decltype(sA_layout), decltype(sB_layout)>;
        int smem_size = sizeof(SharedStorage);

        using KernelType = decltype(&matmul_cute_wgmma_tma<decltype(cta_tiler),
            decltype(tma_a), decltype(tma_b),
            decltype(sA_layout), decltype(sB_layout),
            decltype(tiled_mma)>);
        KernelType kernel_ptr = &matmul_cute_wgmma_tma<decltype(cta_tiler),
            decltype(tma_a), decltype(tma_b),
            decltype(sA_layout), decltype(sB_layout),
            decltype(tiled_mma)>;

        cudaFuncSetAttribute(kernel_ptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        dim3 dimBlock(size(tiled_mma));  // 256 threads
        dim3 dimCluster(1, 1, 1);       // no multicast
        dim3 dimGrid(M / BLK_M, N / BLK_N);
        cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

        auto wgmma_tma_op = [&]() {
            cutlass::Status status = cutlass::launch_kernel_on_cluster(params,
                reinterpret_cast<void const*>(kernel_ptr),
                M, K, N, tma_a, tma_b, d_C, tiled_mma);
        };

        run_benchmark("CuTe WGMMA TMA", wgmma_tma_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // CuTe SM90 wgmma + TMA: 128x256 tile (matching cuBLAS tile footprint)
    if (should_run("CuTe WGMMA TMA 128x256", kernel_filters)) {
        using MmaAtom = SM90_64x128x8_F32TF32TF32_SS_TN<>;
        auto tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _2, _1>>{});

        constexpr int BLK_M = 128, BLK_N = 256, BLK_K = 32;
        constexpr int PIPE = 3;
        auto bM = Int<BLK_M>{};
        auto bN = Int<BLK_N>{};
        auto bK = Int<BLK_K>{};
        auto bP = Int<PIPE>{};
        auto cta_tiler = make_shape(bM, bN, bK);

        using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<float>;
        auto sA_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(bM, bK, bP));
        auto sB_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(bN, bK, bP));

        Tensor host_mA = make_tensor(make_gmem_ptr(d_A), make_shape(M, K), make_stride(K, Int<1>{}));
        Tensor host_mB = make_tensor(make_gmem_ptr(d_B_T), make_shape(N, K), make_stride(K, Int<1>{}));

        auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, host_mA, sA_layout(_, _, 0), make_shape(bM, bK));
        auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, host_mB, sB_layout(_, _, 0), make_shape(bN, bK));

        using SharedStorage = TmaSmemStorage<decltype(sA_layout), decltype(sB_layout)>;
        int smem_size = sizeof(SharedStorage);

        using KernelType = decltype(&matmul_cute_wgmma_tma<decltype(cta_tiler),
            decltype(tma_a), decltype(tma_b),
            decltype(sA_layout), decltype(sB_layout),
            decltype(tiled_mma)>);
        KernelType kernel_ptr = &matmul_cute_wgmma_tma<decltype(cta_tiler),
            decltype(tma_a), decltype(tma_b),
            decltype(sA_layout), decltype(sB_layout),
            decltype(tiled_mma)>;

        cudaFuncSetAttribute(kernel_ptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        dim3 dimBlock(size(tiled_mma));  // 512 threads (4 warp groups)
        dim3 dimCluster(1, 1, 1);
        dim3 dimGrid(M / BLK_M, N / BLK_N);
        cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

        auto wgmma_tma_256_op = [&]() {
            cutlass::Status status = cutlass::launch_kernel_on_cluster(params,
                reinterpret_cast<void const*>(kernel_ptr),
                M, K, N, tma_a, tma_b, d_C, tiled_mma);
        };

        run_benchmark("CuTe WGMMA TMA 128x256", wgmma_tma_256_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
    }

    // CUTLASS 3.0 Warp-Specialized GEMM (example 48: TMA + warp specialization + persistent tile scheduler)
    if (should_run("CUTLASS WS GEMM", kernel_filters)) {
        using namespace cutlass_ws;

        auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

        cutlass::KernelHardwareInfo hw_info =
            cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(0);

        float gemm_alpha = 1.0f, gemm_beta = 0.0f;

        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {d_A, stride_a, d_B, stride_b},
            {{gemm_alpha, gemm_beta}, d_C, stride_c, d_C, stride_d},
            hw_info
        };

        arguments.scheduler.max_swizzle_size = 4;

        Gemm gemm;
        size_t workspace_size = Gemm::get_workspace_size(arguments);
        uint8_t* workspace_ptr = nullptr;
        if (workspace_size > 0) cudaMalloc(&workspace_ptr, workspace_size);

        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
            printf("CUTLASS WS GEMM: cannot implement this problem\n");
        } else {
            auto cutlass_ws_op = [&]() {
                gemm.initialize(arguments, workspace_ptr);
                gemm.run();
            };

            run_benchmark("CUTLASS WS GEMM", cutlass_ws_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
        }

        if (workspace_ptr) cudaFree(workspace_ptr);
    }

    // cuBLAS
    // cuBLAS is column-major, so we compute C = A*B in row-major as:
    //   C^T = B^T * A^T  in column-major
    // A row-major MxK matrix is a column-major KxM matrix (no data movement).
    // So: cublasSgemm(N, M, K, B, N, A, K, C, N)
    if (should_run("cuBLAS", kernel_filters) || should_run("cuBLAS TF32", kernel_filters)) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;

        if (should_run("cuBLAS", kernel_filters)) {
            auto cublas_op = [&]() {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
            };

            run_benchmark("cuBLAS", cublas_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
        }

        // cuBLAS with TF32 tensor cores
        if (should_run("cuBLAS TF32", kernel_filters)) {
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

            auto cublas_tf32_op = [&]() {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
            };

            run_benchmark("cuBLAS TF32", cublas_tf32_op, d_C, h_C, h_A, h_B, M, K, N, iterations, benchmark_mode);
        }

        cublasDestroy(handle);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_B_T);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
