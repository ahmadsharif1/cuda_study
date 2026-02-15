# CUDA Performance Study — NVIDIA H100 SXM5

A collection of micro-benchmarks and kernel implementations exploring GPU performance
from first principles on an NVIDIA H100 SXM5 (80 GB HBM3, SM 9.0 Hopper).

All measurements taken at the observed boost clock of **1.98 GHz** with CUDA 12.8.

---

## 1. Theoretical Peak Performance (Simple Model)

The H100 SM has 4 processing blocks (sub-partitions), each with its own warp scheduler.
Each scheduler issues 1 warp-instruction per cycle to 32 threads.

| Resource | Per SM per cycle | Full GPU (132 SMs × 1.98 GHz) |
|---|---|---|
| FP32 FMA (CUDA cores) | 4 schedulers × 32 threads × 2 FLOPS = 256 FLOPS | **66.9 TFLOPS** |
| FP32 FADD or FMUL only | 4 × 32 × 1 = 128 ops | 33.4 TFLOPS |
| FP64 FMA (CUDA cores) | 128 FLOPS (1:2 vs FP32) | 33.4 TFLOPS |
| TF32 Tensor Core (dense) | — | **494.7 TFLOPS** |
| TF32 Tensor Core (sparse) | — | 989.4 TFLOPS |
| FP16/BF16 Tensor Core (dense) | — | 989.4 TFLOPS |
| FP16/BF16 Tensor Core (sparse) | — | 1978.9 TFLOPS |
| HBM3 bandwidth (theoretical) | — | **3.35 TB/s** |
| HBM3 bandwidth (measured, copy kernel) | — | **2.0 TB/s** |
| L2 cache | 50 MB (spec) / 60 MB (reported by driver) | — |
| L2 bandwidth (measured) | — | **3.6 TB/s** |

**FP32 derivation:** 132 SMs × 128 FP32 cores/SM × 2 (FMA) × 1.98 GHz = 66.9 TFLOPS.
Each FP32 core executes one fused multiply-add per cycle; each FMA counts as 2 FLOPS
(one multiply + one add). Tensor core peaks are from NVIDIA's official datasheet.

---

## 2. Instruction Latency & Throughput

**Source:** [`micro_benchmarks/inst_lat.cu`](micro_benchmarks/inst_lat.cu)

### Methodology

A single warp (`<<<1, 32>>>`) executes a tight loop of dependent operations.
`ILP=1` creates a serial dependency chain measuring pure **latency** (cycles until
the result is available). Higher ILP values run multiple independent chains in
parallel, filling the pipeline and converging toward **throughput** (cycles per
warp-instruction at steady state).

The outer loop uses `#pragma unroll 1` to prevent the compiler from unrolling it
(which would change the dependency structure). The inner ILP loop uses `#pragma unroll`
to fully unroll into independent chains. `clock64()` measures SM cycles.

### Results (cycles per warp-instruction)

```
                       ILP=1   ILP=2   ILP=4   ILP=8  ILP=16  ILP=32  ILP=64 ILP=128
  FADD                 29.0    14.5     7.8     3.9     1.9     1.4     1.2     1.1
  FMUL                 29.0    14.5     7.8     3.9     1.9     1.4     1.2     1.1
  FMA                  29.0    14.5     7.3     3.6     2.4     1.7     1.4     1.2
  IADD                 29.0    14.5     7.3     3.9     1.8     0.9     0.5     1.1
  IMUL                 29.0    14.5     7.3     3.9     2.6     2.4     2.2     2.1
  ─────────────────────────────────────────────────────────────────────────────────────
  __frcp_rn            87.0    82.0    79.5    77.8    77.6    77.5    77.1    76.8
  __fsqrt_rn           66.0    61.5    58.0    56.5    55.9    56.9    56.1    55.8
  __sinf               29.0    15.0    11.0    10.3     9.6     9.3     9.2     9.1
  __logf (*)           45.0    23.0    15.3    12.0    10.5     9.8     9.4     9.3
  __expf (*)           54.0    31.5    20.3    14.3    11.1     9.6     9.0     8.5
```

`(*)` `__logf` chain includes `fabsf(x)+2.0f` wrapper; `__expf` includes `x*1e-5` wrapper.

### Key observations

- **Pipeline latency = 29 cycles** for all simple ALU ops (FADD, FMUL, FMA, IADD, IMUL).
  This is the Hopper pipeline depth — much deeper than older architectures (~4 cycles on
  Volta/Ampere), enabling higher clock speeds.
- **True throughput = 1.0 cycle/warp-instruction** for FADD/FMUL. The residual ~0.1 at
  ILP=128 is amortized loop overhead (~13 cycles per iteration from 3 loop-control
  instructions + branch penalty).
- **IADD reaches 0.5 cycles/op** at ILP=64, suggesting dual integer ALU pipes per
  sub-partition. Degrades at ILP=128, likely from register bank conflicts.
- **IMUL converges to ~2.1 cycles/op** — integer multiply is half-rate.
- **`__frcp_rn` / `__fsqrt_rn` barely scale with ILP** (87→77, 66→56). These are
  multi-instruction IEEE-compliant sequences with long internal dependency chains.
- **SFU ops** (`__sinf`, `__logf`, `__expf`) converge to ~9 cycles/op. The SFU is a
  separate functional unit with limited throughput.

---

## 3. Memory Bandwidth & Latency

### 3a. Memory Latency

**Source:** [`micro_benchmarks/latency.cu`](micro_benchmarks/latency.cu)

**Methodology:** Pointer chasing through a random Hamiltonian cycle (each load's
address depends on the previous load's value). Single thread `<<<1,1>>>`, measured
with `clock64()`. This eliminates memory-level parallelism and measures pure latency.

- **Shared memory / L1:** 8 KB working set, fits entirely in the target memory.
- **L2:** 4 MB working set with `__ldcg` loads (bypass L1, cache in L2). Warmup pass
  ensures all data is resident in L2.
- **HBM:** 256 MB working set (>> 60 MB L2) with cache-line-strided elements (128 bytes
  apart). L2 is flushed before measurement by reading a buffer larger than L2.

| Level | Working Set | Latency (cycles) | Latency (ns) |
|---|---|---|---|
| Shared Memory | 8 KB | 28.6 | 14.4 |
| L1 Cache | 8 KB | 40.2 | 20.3 |
| L2 Cache | 4 MB | 259.9 | 131.3 |
| HBM / DRAM | 256 MB | 570.5 | 288.1 |

| Ratio | Value |
|---|---|
| L1 / Shared Memory | 1.4x |
| L2 / L1 | 6.5x |
| HBM / L2 | 2.2x |
| HBM / Shared Memory | 20.0x |

### 3b. HBM Bandwidth Scaling

**Source:** [`micro_benchmarks/copy_bw.cu`](micro_benchmarks/copy_bw.cu)

**Methodology:** Copy 256 MB (`src → dst`), reporting `2 × data_size / time` as
bandwidth (read + write). Template kernel parameterized on data type (`float` vs
`float4`) and unroll factor. Timed with CUDA events over 20 iterations.

| Configuration | Bandwidth | % of 2.0 TB/s |
|---|---|---|
| 1 thread, float, UNROLL=1 | 0.09 GB/s | 0.00% |
| 1 thread, float4, UNROLL=32 | 0.29 GB/s | 0.01% |
| 1 warp (32 threads), float4, UNROLL=32 | 7.64 GB/s | 0.38% |
| 1 SM (1024 threads), float4, UNROLL=4 | 66.67 GB/s | 3.33% |
| 32 SMs | 1,183 GB/s | 59.2% |
| 64 SMs | 1,722 GB/s | 86.1% |
| **132 SMs (all)** | **1,992 GB/s** | **99.6%** |
| 528 blocks (4× SMs) | 2,034 GB/s | 101.7% |

Reaching ~99% of the achievable HBM bandwidth requires **all 132 SMs**. This follows
from Little's Law: `BW = outstanding_bytes / latency`. With 288 ns HBM latency, each
SM must keep many loads in flight, and the total outstanding bytes across all SMs must
saturate the HBM controllers.

### 3c. L2 Bandwidth Scaling

**Source:** [`micro_benchmarks/l2_bw.cu`](micro_benchmarks/l2_bw.cu)

**Methodology:** Copy a 16 MB working set (32 MB total for src + dst, fits in 60 MB
L2) using `__ldcg` / `__stcg` intrinsics that bypass L1 and operate directly on L2.
Warmup ensures data is L2-resident.

| Configuration | L2 Bandwidth |
|---|---|
| 1 SM | 16.8 GB/s |
| 16 SMs | 323 GB/s |
| 64 SMs | 2,252 GB/s |
| 132 SMs (all) | 3,223 GB/s |
| **528 blocks (4× SMs)** | **3,574 GB/s (3.6 TB/s)** |
| HBM (256 MB, doesn't fit L2) | 887 GB/s |
| **L2 / HBM ratio** | **4.0×** |

---

## 4. Matrix Transpose

**Source:** [`transpose/transpose.cu`](transpose/transpose.cu)

Transpose a 32768×32768 float matrix (4 GB). All kernels use 32×32 shared memory
tiles. Bandwidth = `2 × matrix_size / time` (read input + write output).

### Kernels

| Kernel | Description |
|---|---|
| [**Basic**](transpose/transpose.cu#L17) | Shared memory tiled transpose. 32×8 thread block, each thread handles 4 rows. No bank-conflict padding — causes 32-way bank conflicts on shared memory reads. |
| [**Vectorized**](transpose/transpose.cu#L47) | `float4` vectorized global loads/stores. Padding (`TILE_DIM+1`) eliminates bank conflicts. 8×32 thread block where each thread processes 4 contiguous floats. |
| [**CuTe Scalar**](transpose/transpose.cu#L115) | CuTe layout abstractions with scalar copy atoms. Row-major input → shared memory → column-major output. Padding = 1. |
| [**CuTe Vectorized**](transpose/transpose.cu#L115) | CuTe with `UniversalCopy<float4>` for input and `AutoVectorizingCopy` for output. Padding = 4 for alignment. |

### Results (32768 × 32768, 100 iterations)

| Kernel | Bandwidth (GB/s) | Time (ms) | % of HBM peak |
|---|---|---|---|
| Basic | 1,224 | 7.02 | 61% |
| Vectorized | 1,904 | 4.51 | 95% |
| CuTe Scalar | 1,902 | 4.52 | 95% |
| CuTe Vectorized | 1,890 | 4.55 | 94% |

The basic kernel is bandwidth-limited by shared memory bank conflicts (no padding).
All three optimized variants reach ~95% of HBM bandwidth, showing that the transpose
is fully memory-bound once bank conflicts are eliminated.

---

## 5. Matrix Multiplication (GEMM)

**Source:** [`matmul/matmul.cu`](matmul/matmul.cu)

Multiply `A(4096×16384) × B(16384×8192) → C(4096×8192)` in FP32/TF32.
Total FLOPs per GEMM = 2 × 4096 × 16384 × 8192 ≈ 1.1 × 10¹².

### Kernels

| Kernel | Description |
|---|---|
| [**Naive**](matmul/matmul.cu#L33) | One thread per output element. No shared memory, no tiling. Pure FP32. |
| [**CuTe Simple MMA**](matmul/matmul.cu#L47) | TF32 tensor core MMA (`SM80_16x8x8`), loads directly from global memory to registers. 128×128×8 tiles, 128 threads (4 warps). |
| [**CuTe SMEM k=8**](matmul/matmul.cu#L88) | Adds shared memory staging with `cp.async` (128-bit async copies). Padded smem layouts. BLK_K=8. |
| [**CuTe SMEM k=32**](matmul/matmul.cu#L88) | Same as above but with BLK_K=32 — more work per k-tile, fewer global memory round-trips. |
| [**CuTe WGMMA**](matmul/matmul.cu#L380) | SM90 warp-group MMA (`SM90_64x128x8`). Reads A and B directly from swizzled shared memory via GMMA descriptors (no smem→register copy). 256 threads. |
| [**CuTe WGMMA Pipe**](matmul/matmul.cu#L453) | Double-buffered pipeline: overlaps `cp.async` loads of the next k-tile with wgmma compute on the current tile. |
| [**CuTe WGMMA Pipe+Swiz**](matmul/matmul.cu#L453) | Adds CTA swizzle (1D grid with tile reordering) for L2 cache reuse across neighboring CTAs. |
| [**CuTe WGMMA TMA 128×128**](matmul/matmul.cu#L575) | Replaces `cp.async` with TMA (Tensor Memory Accelerator) — a single thread loads an entire tile via hardware. 3-stage pipeline with producer/consumer barriers. |
| [**CuTe WGMMA TMA 128×256**](matmul/matmul.cu#L575) | Larger output tile (128×256) with 512 threads (4 warp groups). More compute per memory access. |
| [**CUTLASS WS GEMM**](matmul/matmul.cu#L800) | CUTLASS 3.0 warp-specialized persistent kernel. Dedicated warp groups for data movement vs compute. Persistent tile scheduler with swizzle. |
| [**cuBLAS**](matmul/matmul.cu#L1283) | NVIDIA's hand-tuned SGEMM. Pure FP32 (no tensor cores). |
| [**cuBLAS TF32**](matmul/matmul.cu#L1299) | cuBLAS with `CUBLAS_TF32_TENSOR_OP_MATH`. Uses TF32 tensor cores for ~9× speedup over FP32. |

### Results (M=4096, K=16384, N=8192, 100 iterations)

| Kernel | Time (ms) | TFLOPS | % of TF32 peak (494.7) |
|---|---|---|---|
| Naive (FP32) | 296.23 | 3.7 | — |
| CuTe Simple MMA (TF32) | 26.04 | 42.2 | 8.5% |
| CuTe SMEM k=8 (TF32) | 17.85 | 61.6 | 12.5% |
| CuTe SMEM k=32 (TF32) | 11.55 | 95.2 | 19.2% |
| CuTe WGMMA | 6.26 | 175.6 | 35.5% |
| CuTe WGMMA Pipe | 4.50 | 244.3 | 49.4% |
| CuTe WGMMA Pipe+Swiz | 4.47 | 245.9 | 49.7% |
| CuTe WGMMA TMA 128×128 | 4.62 | 238.2 | 48.1% |
| **CuTe WGMMA TMA 128×256** | **3.65** | **300.9** | **60.8%** |
| CUTLASS WS GEMM | 4.80 | 228.9 | 46.3% |
| cuBLAS (FP32) | 30.97 | 35.5 | — |
| **cuBLAS TF32** | **3.44** | **319.7** | **64.6%** |

**Note on verification:** Kernels using TF32 tensor cores show "FAIL" in verification
because TF32 truncates the FP32 mantissa from 23 to 10 bits before multiplication.
This introduces ~0.01–0.08% relative error vs the FP32 CPU reference — expected and
correct behavior for TF32 precision. The naive kernel and cuBLAS (FP32 mode) pass
exact/near-exact verification.

### Progression

The kernels demonstrate a clear optimization progression:

1. **Naive → CuTe Simple MMA** (11×): Switch from FP32 scalar to TF32 tensor cores.
2. **Simple → SMEM k=32** (2.3×): Stage data through shared memory with `cp.async`,
   increase K-tile to reduce global memory traffic.
3. **SMEM → WGMMA** (1.8×): Use SM90 warp-group MMA that reads directly from
   shared memory (eliminates smem→register copy).
4. **WGMMA → WGMMA Pipe** (1.4×): Double-buffer shared memory to overlap loads with
   compute.
5. **Pipe → TMA 128×256** (1.2×): Hardware-accelerated TMA loads + larger tile for
   better compute-to-memory ratio.
6. **Best hand-written vs cuBLAS TF32**: 300.9 vs 319.7 TFLOPS (94% of cuBLAS).

---

## Building

```bash
# Micro-benchmarks
cd micro_benchmarks
make NVCC=/usr/local/cuda-12.8/bin/nvcc
./bin/copy_bw          # HBM bandwidth scaling
./bin/l2_bw            # L2 bandwidth scaling
./bin/latency          # Memory latency (smem, L1, L2, HBM)
./bin/inst_lat         # Instruction latency & throughput

# Transpose
cd transpose
make NVCC=/usr/local/cuda-12.8/bin/nvcc
./bin/transpose -b     # Run with benchmarking (warmup + timing + verification)

# Matrix multiplication
cd matmul
make NVCC=/usr/local/cuda-12.8/bin/nvcc
./bin/matmul -b        # Run with benchmarking
```

All kernels require an SM 9.0 (Hopper) GPU. The matmul kernels require
[CUTLASS](https://github.com/NVIDIA/cutlass) headers (expected at `./cutlass/`).
