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

Each SM has **4 processing blocks** (sub-partitions), each with:

- 1 warp scheduler → issues **1 warp-instruction per cycle** (32 threads)
- 32 FP32 CUDA cores (128 total per SM)
- 4th-gen Tensor Cores (TF32, FP16, INT8)
- 256 KB register file per SM

**Full GPU:** 132 SMs × 1.98 GHz boost

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

<style scoped>
pre { font-size: 0.65em; line-height: 1.2; margin-top: 0.5em; }
li { line-height: 1.5; }
</style>

---

# Theoretical Peak Performance

| Resource | Per SM / cycle | Full GPU (132 SMs × 1.98 GHz) |
|---|---|---|
| FP32 FMA | 4 × 32 × 2 = 256 FLOPS | **66.9 TFLOPS** |
| TF32 Tensor Core (dense) | — | **494.7 TFLOPS** |
| HBM3 bandwidth | — | 3.35 TB/s spec / **2.0 TB/s** measured |
| L2 cache bandwidth | — | **3.6 TB/s** measured |

### Memory hierarchy latency (measured)

| Level | Latency | Bandwidth |
|---|---|---|
| Shared Memory | 14 ns (28 cycles) | ~19 TB/s aggregate |
| L1 Cache | 20 ns (40 cycles) | |
| L2 Cache | 131 ns (260 cycles) | 3.6 TB/s |
| HBM / DRAM | 288 ns (571 cycles) | 2.0 TB/s |

<style scoped>
table { font-size: 0.78em; }
th, td { padding: 4px 8px; }
h3 { margin-top: 1em; margin-bottom: 0.3em; }
</style>

---

# Why Matrix Multiply?

### Every neural network is built on GEMM

- **Linear layers:** `y = Wx + b` → matrix multiply
- **Attention:** `QKᵀ` and `(QKᵀ)V` → two matrix multiplies per head
- **Convolutions:** im2col + GEMM is the standard implementation

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

<style scoped>
pre { font-size: 0.75em; line-height: 1.3; }
</style>

---

# GEMM Problem Setup

### Dimensions

`A(4096 × 16384) × B(16384 × 8192) → C(4096 × 8192)`

**1.1 × 10¹² FLOPs** per GEMM (1.1 TFLOP)

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

<style scoped>
pre { font-size: 0.6em; line-height: 1.25; }
h3 { margin-top: 0.5em; margin-bottom: 0.2em; }
</style>

---

# Kernel 1: Naive FP32 — Code

Each thread computes **one element** of C with no data reuse:

```cpp
__global__ void matmul_naive(const float* A, const float* B,
                             float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];  // 2 loads, 1 FMA
    }
    C[row * N + col] = sum;
}
// Launch: block(16,16), grid((N+15)/16, (M+15)/16)
```

- Every thread re-reads the **entire K-row** of A and K-column of B from global memory
- No data reuse between threads — ~8 bytes per FLOP (need ~0.1)
- B reads are coalesced within half-warp; A reads broadcast via L1

---

# Kernel 1: Naive FP32 — Diagram

<img src="./svgs/01_naive.svg" class="mx-auto h-90" />

---

# Kernel 1: Naive FP32 — Performance

| Metric | Value |
|---|---|
| Time | 296.2 ms |
| TFLOPS | 3.7 |
| % cuBLAS | 1.2% |

### Bottleneck: Pure Memory Bound

- 97.6% memory throughput saturated
- Each thread: K loads from A + K loads from B = **2K global loads** for K FMAs
- 16× redundant loads for both A and B elements within a thread block
- Ratio: 2 loads per 1 FMA = **0.5 compute:load** — need 100× more compute per byte

---

# Kernel 2: CuTe Simple MMA — Code

### What changed
- **TF32 tensor cores** via CuTe `SM80_16x8x8` MMA atom
- 128 threads (4 warps), tiled 2×2 MMA layout
- Still loads A,B directly from **global memory** (no shared memory)

```cpp
// Main loop: global memory → registers → tensor core MMA
auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));

for (int k = 0; k < k_tile_max; k++) {
    copy(thr_mma.partition_A(gA(_, _, k)), tCrA);  // global → regs
    copy(thr_mma.partition_B(gB(_, _, k)), tCrB);  // global → regs
    gemm(tiled_mma, tCrA, tCrB, tCrC);             // mma.sync
}
```

Pipeline: `Global → Registers → MMA → Registers → Global`

<style scoped>
pre { font-size: 0.7em; line-height: 1.3; }
</style>

---

# Kernel 2: CuTe Simple MMA — Diagram

<img src="./svgs/06_cute_simple.svg" class="mx-auto h-90" />

---

# Kernel 2: CuTe Simple MMA — Performance

| Metric | Value |
|---|---|
| Time | 26.0 ms |
| TFLOPS | 42.2 |
| % cuBLAS | 13.2% |
| Speedup | **11× over naive** |

### Why only 13% of cuBLAS?

- No shared memory → redundant global loads (~3× vs smem version)
- MMA partition layout ≠ memory-optimal layout → **uncoalesced access**
- L1 cache provides 67.6% hit rate → implicit data reuse saves it from being worse
- No `__syncthreads()` overhead — each thread operates independently

The 11× speedup comes entirely from **tensor cores** (TF32 FMA throughput).

---

# Kernel 3: CuTe SMEM — Code

### What changed
- **`cp.async`**: global → shared memory (bypasses registers)
- Cooperative tile loading: all 128 threads load A and B tiles
- Each tile loaded **once**, reused by all MMA iterations

```cpp
for (int k = 0; k < k_tile_max; k++) {
    // Phase 1: cooperative load global → shared (cp.async)
    copy(copy_a, tAgA(_, _, _, k), tAsA);
    copy(copy_b, tBgB(_, _, _, k), tBsB);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Phase 2: shared → registers → tensor core MMA
    copy(tCsA, tCrA);
    copy(tCsB, tCrB);
    gemm(tiled_mma, tCrA, tCrB, tCrC);
    __syncthreads();
}
```

Pipeline: `Global →(cp.async)→ SMEM →(copy)→ Registers →(mma.sync)→ Registers`

<style scoped>
pre { font-size: 0.65em; line-height: 1.3; }
</style>

---

# Kernel 3: CuTe SMEM — Diagram

<img src="./svgs/07_cute_smem.svg" class="mx-auto h-90" />

---

# Kernel 3: CuTe SMEM — Performance

| Variant | Time | TFLOPS | % cuBLAS |
|---|---|---|---|
| k=8 | 17.9 ms | 61.6 | 19.3% |
| **k=32** | **11.6 ms** | **95.2** | **29.8%** |

<br>

### Key insight: larger K-tile = fewer global loads

Larger K-tile (32 vs 8) means **4× fewer global loads** per output tile, because each tile is loaded once and reused across all MMA iterations within that tile.

### Remaining bottleneck

The **smem→register copy** is now the bottleneck — reading A,B from shared memory into register fragments before each MMA instruction. This copy step will be eliminated in the next kernel.

Profile: 1.5-way shared load bank conflicts from padded layout. 128 regs/thread.

---

# Kernel 4: CuTe WGMMA — Code

### What changed
- **SM90 `wgmma.mma_async`**: reads A,B **directly from shared memory**
- No smem→register copy needed (hardware descriptor-based)
- **Swizzled layouts** (SW128) eliminate bank conflicts entirely

```cpp
for (int k = 0; k < k_tile_max; k++) {
    copy(copy_a, tAgA(_, _, _, k), tAsA);  // global → swizzled smem
    copy(copy_b, tBgB(_, _, _, k), tBsB);
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();

    // wgmma reads A,B from smem descriptors — NO register copy!
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    gemm(tiled_mma, tCsA, tCsB, tCrC);  // smem descriptors
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    __syncthreads();
}
```

Pipeline: `Global →(cp.async)→ Swizzled SMEM →(wgmma descriptors)→ Registers`

<style scoped>
pre { font-size: 0.62em; line-height: 1.3; }
</style>

---

# Kernel 4: CuTe WGMMA — Diagram

<img src="./svgs/11_cute_wgmma.svg" class="mx-auto h-90" />

---

# Kernel 4: CuTe WGMMA — Performance

| Metric | Value |
|---|---|
| Time | 6.3 ms |
| TFLOPS | 175.6 |
| % cuBLAS | 54.9% |
| Speedup | **1.8× over SMEM k=32** |

### Why the big jump?

Eliminating the smem→register copy removes an entire pipeline stage. The hardware MMA unit reads directly from shared memory via descriptors, freeing register file bandwidth for accumulation.

- **MMA atom**: 64×128×8 = 65,536 FMAs (vs 16×8×8 = 1,024 for SM80)
- **256 threads** (2 warp groups of 128), only C accumulator in registers
- Zero bank conflicts (ncu confirmed) via SW128 swizzle

<style scoped>
table { font-size: 0.8em; }
th, td { padding: 3px 8px; }
h3 { margin-top: 0.5em; margin-bottom: 0.2em; }
p { font-size: 0.85em; line-height: 1.4; margin: 0.3em 0; }
li { font-size: 0.85em; }
</style>

---

# Kernel 5: WGMMA Pipe — Code

### What changed
- **Double buffering**: 2× shared memory (buf0, buf1)
- While MMA computes on buf0, `cp.async` fills buf1
- Load latency **hidden** behind compute

```cpp
// Prologue: load k=0 into buffer 0
copy(copy_a, tAgA(_, _, _, 0), tAsA0);
copy(copy_b, tBgB(_, _, _, 0), tBsB0);
cp_async_fence(); cp_async_wait<0>(); __syncthreads();

for (int k = 0; k < k_max; k++) {
    if (k + 1 < k_max) {  // load next tile into other buffer
        copy(copy_a, tAgA(_, _, _, k+1), (k&1) ? tAsA0 : tAsA1);
        copy(copy_b, tBgB(_, _, _, k+1), (k&1) ? tBsB0 : tBsB1);
        cp_async_fence();
    }
    warpgroup_fence_operand(tCrC); warpgroup_arrive();
    gemm(tiled_mma, (k&1)?tCsA1:tCsA0, (k&1)?tCsB1:tCsB0, tCrC);
    warpgroup_commit_batch(); warpgroup_wait<0>();
    // ... wait for next buffer, sync
}
```

<style scoped>
pre { font-size: 0.6em; line-height: 1.25; }
</style>

---

# Kernel 5: WGMMA Pipe — Diagram

<img src="./svgs/12_cute_wgmma_pipe.svg" class="mx-auto h-90" />

---

# Kernel 5: WGMMA Pipe — Performance

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
- Pipeline hides ~90% of load latency behind compute

<style scoped>
table { font-size: 0.8em; }
th, td { padding: 3px 8px; }
h3 { margin-top: 0.5em; margin-bottom: 0.2em; }
p { font-size: 0.85em; line-height: 1.4; margin: 0.3em 0; }
li { font-size: 0.85em; }
</style>

---

# Kernel 6: WGMMA TMA — Code

### What changed
- **TMA** (Tensor Memory Accelerator): dedicated hardware for N-D copies
- Replaces `cp.async` — **no thread participation** needed for loads
- **3-stage pipeline** (vs 2-stage): deeper overlap

```cpp
// Prologue: thread 0 fills all 3 pipeline stages via TMA
for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k), tAsA(_, pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k), tBsB(_, pipe));
    }
    ++k_tile;
}
// Main loop: all 256 threads compute, only thread 0 issues TMA loads
while (k_tile_count > -K_PIPE_MAX) {
    ProducerBarType::wait(&producer_mbar[read_pipe], phase);
    warpgroup_arrive();
    gemm(tiled_mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
    // ... barrier-based producer-consumer sync
}
```

<style scoped>
pre { font-size: 0.58em; line-height: 1.25; }
</style>

---

# Kernel 6: WGMMA TMA — Diagram

<img src="./svgs/13_cute_wgmma_tma.svg" class="mx-auto h-90" />

---

# Kernel 6: WGMMA TMA — Performance

| Variant | Time | TFLOPS | % cuBLAS |
|---|---|---|---|
| 128×128 tile | 4.6 ms | 238.2 | 74.5% |
| **128×256 tile** | **3.65 ms** | **300.9** | **94.1%** |

### Why larger tile matters

128×256 tile doubles work per CTA → better amortization of scheduling overhead and barrier synchronization costs. At 95.7% SM busy, nearly all compute is utilized.

### TMA advantages

- **1 thread** issues loads → 255 threads 100% dedicated to WGMMA compute
- Hardware handles address calculation, bounds checking, swizzle
- Fence-based barriers (`arrive`/`wait`) instead of `__syncthreads()`

### Best hand-written: **94.1% of cuBLAS TF32**

---

# GEMM — Results Table

<style scoped>
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

cuBLAS achieves 319.7 vs our 300.9 TFLOPS. What are we leaving on the table?

**1. Warp Specialization** — Dedicate separate warps to **producing** (loading data) vs **consuming** (computing MMA). Avoids resource contention.

**2. Persistent Kernel + Tile Scheduling** — Launch exactly `num_SMs` blocks that loop over tiles. Stream-K decomposition for better load balancing.

**3. More Pipeline Stages** — 4-5 stages instead of 3 for deeper overlap between TMA loads, smem→register copies, and MMA compute.

**4. Epilogue Fusion + Register Optimization** — Fuse the C store with the last MMA iteration. Careful register allocation to avoid spills (our 128×256 kernel uses 90 regs).

**5. Auto-Tuning** — Tile sizes, swizzle modes, pipeline depth, cluster size — cuBLAS searches a large configuration space per GPU and problem size.

<style scoped>
p { font-size: 0.85em; line-height: 1.5; margin: 0.3em 0; }
</style>

---
layout: section
---

# Backup Slides

Micro-benchmarks: instruction latency, memory hierarchy, matrix transpose

---

# Instruction Latency — Methodology

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

<style scoped>
pre { font-size: 0.6em; line-height: 1.3; }
li { font-size: 0.9em; }
</style>

---

# Instruction Latency — Results

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

<style scoped>
table { font-size: 0.6em; }
th, td { padding: 2px 5px; }
p { font-size: 0.85em; }
</style>

---

# Instruction Latency — Key Observations

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

### IADD: 0.5 cycles/op at ILP=64 — **dual integer ALU pipes** per sub-partition

### SFU ops converge to ~9 cycles/op — multi-instruction IEEE sequences

<style scoped>
h3 { font-size: 1.05em; margin-top: 0.5em; margin-bottom: 0.1em; }
p { font-size: 0.82em; line-height: 1.4; margin: 0.2em 0; }
table { font-size: 0.72em; }
th, td { padding: 2px 8px; }
</style>

---

# Memory Latency

### Methodology: Pointer Chasing

Single thread `<<<1,1>>>`. Random Hamiltonian cycle — each load's **address depends
on the previous load's value**. Eliminates memory-level parallelism → pure latency.

- **L2 test:** `__ldcg` loads (bypass L1), warmup ensures L2-resident
- **HBM test:** 256 MB working set, L2 flushed before measurement, cache-line strided (128B apart)

| Level | Working Set | Cycles | Nanoseconds | Ratio |
|---|---|---|---|---|
| **Shared Memory** | 8 KB | 28.6 | 14.4 ns | 1.0× |
| **L1 Cache** | 8 KB | 40.2 | 20.3 ns | 1.4× |
| **L2 Cache** | 4 MB | 259.9 | 131.3 ns | 9.1× |
| **HBM / DRAM** | 256 MB | 570.5 | 288.1 ns | 20.0× |

L2 is **6.5× slower** than L1. HBM is only **2.2× slower** than L2 —
the L2 is large (60 MB) and distant from the SMs.

<style scoped>
table { font-size: 0.72em; }
th, td { padding: 3px 8px; }
h3 { margin-top: 0.4em; margin-bottom: 0.2em; }
p { font-size: 0.85em; margin: 0.3em 0; }
li { font-size: 0.82em; }
</style>

---

# HBM Bandwidth Scaling

Copy 256 MB (`src → dst`), `float4` with UNROLL=4. Bandwidth = 2 × data / time.

| Configuration | Bandwidth | % of 2.0 TB/s |
|---|---|---|
| 1 thread | 0.09 GB/s | 0.00% |
| 1 warp (32 threads) | 7.6 GB/s | 0.4% |
| 1 SM (1024 threads) | 66.7 GB/s | 3.3% |
| 8 SMs | 362.8 GB/s | 18.1% |
| 32 SMs | 1,183 GB/s | 59.2% |
| 64 SMs | 1,722 GB/s | 86.1% |
| **132 SMs (all)** | **1,992 GB/s** | **99.6%** |

### Why so many SMs?

**Little's Law:** `BW = outstanding_bytes / latency`

With 288 ns HBM latency, each SM can only sustain ~67 GB/s.
Need all 132 SMs to collectively keep enough loads in flight to saturate the HBM controllers.

<style scoped>
table { font-size: 0.7em; }
th, td { padding: 2px 8px; }
h3 { margin-top: 0.5em; margin-bottom: 0.2em; }
p { font-size: 0.85em; margin: 0.3em 0; }
</style>

---

# L2 Bandwidth Scaling

16 MB working set, `__ldcg` / `__stcg` (bypass L1, operate on L2 directly).

| SMs | L2 Bandwidth | | SMs | L2 Bandwidth |
|---|---|---|---|---|
| 1 | 16.8 GB/s | | 64 | 2,252 GB/s |
| 4 | 64.8 GB/s | | 132 (all) | 3,223 GB/s |
| 16 | 323 GB/s | | **528 blocks (4×)** | **3,574 GB/s** |

| Metric | Value |
|---|---|
| **L2 peak bandwidth** | 3.6 TB/s |
| **HBM peak bandwidth** | 2.0 TB/s |
| **L2 / HBM ratio** | **4.0×** |

L2 also needs all SMs to saturate — same Little's Law argument applies
(L2 latency is 131 ns, still requires many outstanding requests).

<style scoped>
table { font-size: 0.72em; }
th, td { padding: 3px 8px; }
p { font-size: 0.85em; margin: 0.3em 0; }
</style>

---

# Matrix Transpose

32768 × 32768 float matrix (4 GB). All kernels use 32×32 shared memory tiles.

| Kernel | Approach | BW (GB/s) | Time (ms) | % HBM |
|---|---|---|---|---|
| **Basic** | Shared mem tile, no padding | 1,224 | 7.02 | 61% |
| **Vectorized** | `float4` loads/stores, +1 padding | 1,904 | 4.51 | **95%** |
| **CuTe Scalar** | CuTe abstractions, scalar copies | 1,902 | 4.52 | **95%** |
| **CuTe Vectorized** | CuTe with `float4` copy atoms | 1,890 | 4.55 | **94%** |

### Why is Basic only 61%?

No shared memory padding → **32-way bank conflicts** on the transposed read
(`tile[threadIdx.x][threadIdx.y+j]`). All 32 threads in a warp hit the same bank.

Adding **+1 padding** (`tile[32][33]`) shifts each row by one bank, eliminating
conflicts entirely → jumps to 95% of HBM bandwidth.

<style scoped>
table { font-size: 0.68em; }
th, td { padding: 3px 8px; }
h3 { margin-top: 0.5em; margin-bottom: 0.2em; }
p { font-size: 0.82em; margin: 0.3em 0; }
</style>
