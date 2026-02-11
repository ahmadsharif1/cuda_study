#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

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

    // CuTe shared memory kernel with BLK_K=8
    {
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
    {
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
    {
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
    {
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
    {
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

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_B_T);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
