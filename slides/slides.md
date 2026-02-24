---
theme: default
title: "CUDA Performance Study — H100 SXM5"
info: |
  Micro-benchmarks and kernel implementations exploring GPU performance
  from first principles on NVIDIA H100 SXM5.
class: text-center
drawings:
  persist: false
transition: fade
mdc: true
---

# CUDA Performance Study

## NVIDIA H100 SXM5 (Hopper, SM 9.0)

<br>

Building a matrix multiply from scratch — from 3.7 TFLOPS to 301 TFLOPS (94% of cuBLAS).

Measured at **1.98 GHz** boost clock, CUDA 12.8.

---

# H100 SM Architecture

<br>

Each SM has **4 processing blocks** (sub-partitions), each with:

- 1 warp scheduler → issues **1 warp-instruction per cycle** (32 threads)
- 32 FP32 CUDA cores (128 total per SM)
- 4th-gen Tensor Cores (TF32, FP16, INT8)
- 256 KB register file per SM

<br>

**Full GPU:** 132 SMs × 1.98 GHz boost

<br>

```
┌─────────────────────────────────────────────────────────────┐
│                    H100 SXM5 (132 SMs)                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐         │
│  │ SM 0 │ │ SM 1 │ │ SM 2 │ │ SM 3 │ ... │SM 131│         │
│  │ 4×TC │ │ 4×TC │ │ 4×TC │ │ 4×TC │     │ 4×TC │         │
│  │128 FP│ │128 FP│ │128 FP│ │128 FP│     │128 FP│         │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘     └──┬───┘         │
│     └────┬───┴────┬───┴────┬───┴─────...────┘              │
│          │   L2 Cache (60 MB, 3.6 TB/s)      │              │
│          └──────────────┬────────────────────┘              │
│                    HBM3 (80 GB, 2.0 TB/s)                   │
└─────────────────────────────────────────────────────────────┘
```

---

# Theoretical Peak Performance

<br>

| Resource | Per SM / cycle | Full GPU (132 SMs × 1.98 GHz) |
|---|---|---|
| FP32 FMA | 4 × 32 × 2 = 256 FLOPS | **66.9 TFLOPS** |
| TF32 Tensor Core (dense) | — | **494.7 TFLOPS** |
| HBM3 bandwidth | — | 3.35 TB/s spec / **2.0 TB/s** measured |
| L2 cache bandwidth | — | **3.6 TB/s** measured |

<br>

### Memory hierarchy latency (measured)

| Level | Latency | Bandwidth |
|---|---|---|
| Shared Memory | 14 ns (28 cycles) | ~19 TB/s aggregate |
| L1 Cache | 20 ns (40 cycles) | |
| L2 Cache | 131 ns (260 cycles) | 3.6 TB/s |
| HBM / DRAM | 288 ns (571 cycles) | 2.0 TB/s |

---

# Why Matrix Multiply?

<br>

### Every neural network is built on GEMM

- **Linear layers:** `y = Wx + b` → matrix multiply
- **Attention:** `QKᵀ` and `(QKᵀ)V` → two matrix multiplies per head
- **Convolutions:** im2col + GEMM is the standard implementation

<br>

### The arithmetic intensity makes it interesting

```
GEMM: A(M×K) × B(K×N) → C(M×N)
  FLOPs:        2·M·K·N
  Bytes loaded:  4·(M·K + K·N)           (FP32)
  Arithmetic intensity:  M·N / (2·(M+N))  ≈ M/4  for square
```

For M=4096: AI ≈ 1024 FLOP/byte — firmly **compute-bound**.

But reaching peak requires mastery of the full memory hierarchy:
registers → shared memory → L2 → HBM, plus tensor cores, async copies, and hardware scheduling.

---

# GEMM Problem Setup

<br>

### Dimensions

`A(4096 × 16384) × B(16384 × 8192) → C(4096 × 8192)`

**1.1 × 10¹² FLOPs** per GEMM (1.1 TFLOP)

<br>

### Progression

```
Naive FP32          3.7 TFLOPS  ──┐
                                  │  11× (FP32 → TF32 tensor cores)
CuTe Simple MMA    42.2 TFLOPS ──┘──┐
                                    │  2.3× (+ shared memory, cp.async)
CuTe SMEM k=32     95.2 TFLOPS ────┘──┐
                                       │  1.8× (SM90 wgmma, no smem→reg copy)
CuTe WGMMA        175.6 TFLOPS ───────┘──┐
                                          │  1.4× (+ double-buffered pipeline)
CuTe WGMMA Pipe   244.3 TFLOPS ──────────┘──┐
                                             │  1.2× (+ TMA, larger tile)
CuTe WGMMA TMA    300.9 TFLOPS  ────────────┘
       128×256
                                    ┌─────────────────────
cuBLAS TF32        319.7 TFLOPS ────┘  hand-written = 94%
```

---

# Kernel 1: Naive FP32

<br>

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

### Approach
Each thread computes **one element** of C:
```cpp
for (int k = 0; k < K; k++)
    C[row][col] += A[row][k] * B[k][col];
```

### Performance
| Metric | Value |
|---|---|
| Time | 296.2 ms |
| TFLOPS | 3.7 |
| % cuBLAS | 1.2% |

### Bottleneck
- Every thread re-reads **entire K-row** of A and K-column of B from global memory
- No data reuse between threads
- 97.6% memory throughput saturated — pure memory bound

</div>
<div>
<img src="./svgs/01_naive.svg" class="h-90" />
</div>
</div>

---

# Kernel 2: CuTe Simple MMA — Tensor Cores

<br>

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

### What changed
- **TF32 tensor cores** via CuTe `SM80_16x8x8` MMA atom
- 128 threads (4 warps), tiled 2×2 MMA layout
- Still loads A,B directly from **global memory** (no shared memory)

### Performance
| Metric | Value |
|---|---|
| Time | 26.0 ms |
| TFLOPS | 42.2 |
| % cuBLAS | 13.2% |
| Speedup | **11× over naive** |

### Why only 13%?
- No shared memory → redundant global loads (3× vs smem)
- L1 cache provides 67.6% hit rate (implicit reuse)
- MMA partition layout ≠ memory-optimal layout → uncoalesced

</div>
<div>
<img src="./svgs/06_cute_simple.svg" class="h-90" />
</div>
</div>

---

# Kernel 3: CuTe SMEM — Shared Memory + cp.async

<br>

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

### What changed
- **`cp.async`**: global → shared memory (bypasses registers)
- Cooperative tile loading: all 128 threads load A and B tiles
- Each tile loaded **once**, reused by all MMA iterations

### Performance (k=8 → k=32)
| Variant | Time | TFLOPS | % cuBLAS |
|---|---|---|---|
| k=8 | 17.9 ms | 61.6 | 19.3% |
| **k=32** | **11.6 ms** | **95.2** | **29.8%** |

### Key insight
Larger K-tile (32 vs 8) means **4× fewer global loads** per output tile.
The smem→register copy is now the bottleneck — reads A,B from shared memory
into register fragments before each MMA.

### Profile note
1.5-way shared load bank conflicts from padded layout. 128 regs/thread.

</div>
<div>
<img src="./svgs/07_cute_smem.svg" class="h-90" />
</div>
</div>

---

# Kernel 4: CuTe WGMMA — SM90 Warpgroup MMA

<br>

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

### What changed
- **SM90 `wgmma.mma_async`**: reads A,B **directly from shared memory**
- No smem→register copy needed (hardware descriptor-based)
- **Swizzled layouts** (SW128) eliminate bank conflicts entirely
- 256 threads (8 warps = 1 warpgroup)

### Performance
| Metric | Value |
|---|---|
| Time | 6.3 ms |
| TFLOPS | 175.6 |
| % cuBLAS | 54.9% |
| Speedup | **1.8× over SMEM k=32** |

### Why the big jump?
Eliminating the smem→register copy removes an entire pipeline stage.
The hardware MMA unit reads directly from shared memory via descriptors,
freeing register file bandwidth for accumulation.

Zero bank conflicts (ncu confirmed).

</div>
<div>
<img src="./svgs/11_cute_wgmma.svg" class="h-90" />
</div>
</div>

---

# Kernel 5: WGMMA Pipe — Double-Buffered Pipeline

<br>

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

### What changed
- **Double buffering**: 2× shared memory (buf0, buf1)
- While MMA computes on buf0, `cp.async` fills buf1
- Load latency **hidden** behind compute

### Performance
| Metric | Value |
|---|---|
| Time | 4.5 ms |
| TFLOPS | 244.3 |
| % cuBLAS | 76.4% |
| Speedup | **1.4× over WGMMA** |

### With CTA swizzle (+Swiz variant)
Column-major tile scheduling for L2 cache reuse of B matrix.
**245.9 TFLOPS (76.9%)** — marginal gain since L2 hit rate was already decent.

### Profile
- 124 regs/thread, 25% occupancy
- 65.5 KB dynamic shared memory (2× buffers)

</div>
<div>
<img src="./svgs/12_cute_wgmma_pipe.svg" class="h-90" />
</div>
</div>

---

# Kernel 6: WGMMA TMA — Hardware Tensor Memory Accelerator

<br>

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

### What changed
- **TMA** (Tensor Memory Accelerator): dedicated hardware for N-D copies
- Replaces `cp.async` — no thread participation needed for loads
- **3-stage pipeline** (vs 2-stage): deeper overlap
- Fence-based synchronization (`arrive`/`wait` barriers)

### Performance
| Variant | Time | TFLOPS | % cuBLAS |
|---|---|---|---|
| 128×128 tile | 4.6 ms | 238.2 | 74.5% |
| **128×256 tile** | **3.65 ms** | **300.9** | **94.1%** |

### Why larger tile matters
128×256 tile doubles work per CTA → better amortization of
scheduling overhead and barrier synchronization costs.
At 95.7% SM busy, nearly all compute is utilized.

### Best hand-written: **94.1% of cuBLAS TF32**

</div>
<div>
<img src="./svgs/13_cute_wgmma_tma.svg" class="h-90" />
</div>
</div>

---

# GEMM — Results Table

<style>
table { font-size: 0.62em; }
th, td { padding: 2px 8px; }
</style>

| Kernel | Key Technique | Time (ms) | TFLOPS | % cuBLAS |
|---|---|---|---|---|
| Naive | FP32 scalar, no tiling | 296.2 | 3.7 | 1.2% |
| CuTe Simple MMA | TF32 tensor cores, gmem→reg | 26.0 | 42.2 | 13.2% |
| CuTe SMEM k=8 | + `cp.async` to shared mem | 17.9 | 61.6 | 19.3% |
| CuTe SMEM k=32 | + larger K-tile (4× less traffic) | 11.6 | 95.2 | 29.8% |
| CuTe WGMMA | SM90 wgmma (smem descriptors) | 6.3 | 175.6 | 54.9% |
| CuTe WGMMA Pipe | + double-buffered pipeline | 4.5 | 244.3 | 76.4% |
| WGMMA Pipe+Swiz | + CTA swizzle for L2 reuse | 4.5 | 245.9 | 76.9% |
| WGMMA TMA 128×128 | + TMA hardware loads, 3-stage | 4.6 | 238.2 | 74.5% |
| **WGMMA TMA 128×256** | **+ larger tile (128×256)** | **3.65** | **300.9** | **94.1%** |
| **cuBLAS TF32** | **NVIDIA hand-tuned (TC)** | **3.44** | **319.7** | **100%** |

<br>

<div class="text-center text-lg">
Best hand-written kernel: <strong>94% of cuBLAS TF32</strong> — an <strong>81× speedup</strong> over naive.
</div>

---

# What's Missing? The Last 6%

<br>

### cuBLAS achieves 319.7 vs our 300.9 TFLOPS. What are we leaving on the table?

<br>

### 1. Warp Specialization
Dedicate separate warps to **producing** (loading data) vs **consuming** (computing MMA).
CUTLASS's warp-specialized persistent kernel does this — avoids resource contention.

### 2. Persistent Kernel + Tile Scheduling
Launch exactly `num_SMs` blocks that loop over tiles. Combined with stream-K
decomposition for better load balancing across SMs with uneven tile counts.

### 3. More Pipeline Stages
4-5 stages instead of 3 — deeper overlap between TMA loads, smem→register copies,
and MMA compute. More buffering hides longer latency chains.

### 4. Epilogue Fusion + Register Optimization
Fuse the C store with the last MMA iteration. Careful register allocation
to avoid spills (our 128×256 kernel uses 90 regs — room for more).

### 5. Auto-Tuning
Tile sizes, swizzle modes, pipeline depth, cluster size — cuBLAS searches a large
configuration space per GPU and problem size.

---
layout: section
---

# Backup Slides

Micro-benchmarks: instruction latency, memory hierarchy, matrix transpose

---

# Instruction Latency — Methodology

<br>

Single warp `<<<1, 32>>>`, tight loop with **serial dependency chains**.

```cpp
template <typename Op, int ILP>
__global__ void bench_float(int steps, long long* out) {
    float x[ILP];                          // ILP independent chains
    // ...
    long long t0 = clock64();
    #pragma unroll 1                       // do NOT unroll outer loop
    for (int s = 0; s < steps; s++) {
        #pragma unroll                     // DO unroll inner → creates ILP
        for (int i = 0; i < ILP; i++)
            x[i] = Op::apply(x[i]);        // dependent within each chain
    }
    long long t1 = clock64();
    // ...
}
```

- **ILP=1** → every op depends on the previous → measures **latency**
- **ILP=N** → N independent chains fill the pipeline → measures **throughput**
- Report: `(t1 - t0) / (steps × ILP)` = cycles per warp-instruction

---

# Instruction Latency — Results

<style>
table { font-size: 0.7em; }
th, td { padding: 2px 6px; }
</style>

<br>

| Instruction | ILP=1 | ILP=2 | ILP=4 | ILP=8 | ILP=16 | ILP=32 | ILP=64 | ILP=128 |
|---|---|---|---|---|---|---|---|---|
| **FADD** | 29.0 | 14.5 | 7.8 | 3.9 | 1.9 | 1.4 | 1.2 | **1.1** |
| **FMUL** | 29.0 | 14.5 | 7.8 | 3.9 | 1.9 | 1.4 | 1.2 | **1.1** |
| **FMA** | 29.0 | 14.5 | 7.3 | 3.6 | 2.4 | 1.7 | 1.4 | **1.2** |
| **IADD** | 29.0 | 14.5 | 7.3 | 3.9 | 1.8 | 0.9 | **0.5** | 1.1 |
| **IMUL** | 29.0 | 14.5 | 7.3 | 3.9 | 2.6 | 2.4 | 2.2 | **2.1** |
| | | | | | | | | |
| **__frcp_rn** | 87.0 | 82.0 | 79.5 | 77.8 | 77.6 | 77.5 | 77.1 | **76.8** |
| **__fsqrt_rn** | 66.0 | 61.5 | 58.0 | 56.5 | 55.9 | 56.9 | 56.1 | **55.8** |
| **__sinf** | 29.0 | 15.0 | 11.0 | 10.3 | 9.6 | 9.3 | 9.2 | **9.1** |
| **__logf** | 45.0 | 23.0 | 15.3 | 12.0 | 10.5 | 9.8 | 9.4 | **9.3** |
| **__expf** | 54.0 | 31.5 | 20.3 | 14.3 | 11.1 | 9.6 | 9.0 | **8.5** |

Cycles per warp-instruction. ILP=1 = pure latency; higher ILP → throughput.

---

# Instruction Latency — Key Observations

<br>

### Pipeline latency = 29 cycles

All simple ALU ops (FADD, FMUL, FMA, IADD, IMUL) share the same 29-cycle latency.
This is the Hopper pipeline depth — much deeper than Volta/Ampere (~4 cycles).

### True throughput = 1.0 cycle / warp-instruction

FADD/FMUL show 1.1 at ILP=128, not 1.0. The residual is **loop overhead**:
~13 cycles per iteration (3 control instructions + branch penalty), amortized over ILP ops.

| ILP | Predicted `(ILP + 13) / ILP` | Measured |
|---|---|---|
| 32 | 1.41 | 1.4 |
| 64 | 1.20 | 1.2 |
| 128 | 1.10 | 1.1 |

### IADD: 0.5 cycles/op at ILP=64

Suggests **dual integer ALU pipes** per sub-partition. Degrades at ILP=128 (register bank conflicts).

### SFU ops converge to ~9 cycles/op

`__frcp_rn` / `__fsqrt_rn` barely scale — multi-instruction IEEE sequences with internal dependencies.

---

# Memory Latency

<br>

### Methodology: Pointer Chasing

Single thread `<<<1,1>>>`. Random Hamiltonian cycle — each load's **address depends
on the previous load's value**. Eliminates memory-level parallelism → pure latency.

- **L2 test:** `__ldcg` loads (bypass L1), warmup ensures L2-resident
- **HBM test:** 256 MB working set, L2 flushed before measurement, cache-line strided (128B apart)

<br>

| Level | Working Set | Cycles | Nanoseconds | Ratio |
|---|---|---|---|---|
| **Shared Memory** | 8 KB | 28.6 | 14.4 ns | 1.0× |
| **L1 Cache** | 8 KB | 40.2 | 20.3 ns | 1.4× |
| **L2 Cache** | 4 MB | 259.9 | 131.3 ns | 9.1× |
| **HBM / DRAM** | 256 MB | 570.5 | 288.1 ns | 20.0× |

<br>

L2 is **6.5× slower** than L1. HBM is only **2.2× slower** than L2 —
the L2 is large (60 MB) and distant from the SMs.

---

# HBM Bandwidth Scaling

<br>

Copy 256 MB (`src → dst`), `float4` with UNROLL=4. Bandwidth = 2 × data / time.

<br>

| Configuration | Bandwidth | % of 2.0 TB/s |
|---|---|---|
| 1 thread | 0.09 GB/s | 0.00% |
| 1 warp (32 threads) | 7.6 GB/s | 0.4% |
| 1 SM (1024 threads) | 66.7 GB/s | 3.3% |
| 8 SMs | 362.8 GB/s | 18.1% |
| 32 SMs | 1,183 GB/s | 59.2% |
| 64 SMs | 1,722 GB/s | 86.1% |
| **132 SMs (all)** | **1,992 GB/s** | **99.6%** |

<br>

### Why so many SMs?

**Little's Law:** `BW = outstanding_bytes / latency`

With 288 ns HBM latency, each SM can only sustain ~67 GB/s.
Need all 132 SMs to collectively keep enough loads in flight
to saturate the HBM controllers.

---

# L2 Bandwidth Scaling

<br>

16 MB working set, `__ldcg` / `__stcg` (bypass L1, operate on L2 directly).

<br>

| SMs | L2 Bandwidth | | SMs | L2 Bandwidth |
|---|---|---|---|---|
| 1 | 16.8 GB/s | | 64 | 2,252 GB/s |
| 4 | 64.8 GB/s | | 132 (all) | 3,223 GB/s |
| 16 | 323 GB/s | | **528 blocks (4×)** | **3,574 GB/s** |

<br>

| Metric | Value |
|---|---|
| **L2 peak bandwidth** | 3.6 TB/s |
| **HBM peak bandwidth** | 2.0 TB/s |
| **L2 / HBM ratio** | **4.0×** |

<br>

L2 also needs all SMs to saturate — same Little's Law argument applies
(L2 latency is 131 ns, still requires many outstanding requests).

---

# Matrix Transpose

<br>

32768 × 32768 float matrix (4 GB). All kernels use 32×32 shared memory tiles.

<br>

| Kernel | Approach | BW (GB/s) | Time (ms) | % HBM |
|---|---|---|---|---|
| **Basic** | Shared mem tile, no padding | 1,224 | 7.02 | 61% |
| **Vectorized** | `float4` loads/stores, +1 padding | 1,904 | 4.51 | **95%** |
| **CuTe Scalar** | CuTe abstractions, scalar copies | 1,902 | 4.52 | **95%** |
| **CuTe Vectorized** | CuTe with `float4` copy atoms | 1,890 | 4.55 | **94%** |

<br>

### Why is Basic only 61%?

No shared memory padding → **32-way bank conflicts** on the transposed read
(`tile[threadIdx.x][threadIdx.y+j]`). All 32 threads in a warp hit the same bank.

Adding **+1 padding** (`tile[32][33]`) shifts each row by one bank, eliminating
conflicts entirely → jumps to 95% of HBM bandwidth.
