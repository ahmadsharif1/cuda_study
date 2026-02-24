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

Micro-benchmarks for instruction throughput, memory latency & bandwidth,
matrix transpose, and GEMM — measured at **1.98 GHz** boost clock, CUDA 12.8.

---

# H100 SM Architecture — Simple Model

<br>

Each SM has **4 processing blocks** (sub-partitions), each with:

- 1 warp scheduler → issues **1 warp-instruction per cycle** (32 threads)
- 32 FP32 CUDA cores (128 total per SM)
- FP64 cores (1:2 ratio vs FP32)
- Tensor cores (4th gen)

<br>

**Full GPU:** 132 SMs × 1.98 GHz boost

<br>

| Resource | Per SM / cycle | Full GPU |
|---|---|---|
| FP32 FMA | 4 × 32 × 2 = 256 FLOPS | **66.9 TFLOPS** |
| FP32 FADD/FMUL only | 128 ops | 33.4 TFLOPS |
| TF32 Tensor Core (dense) | — | **494.7 TFLOPS** |
| HBM3 (spec / measured) | — | 3.35 / **2.0 TB/s** |
| L2 cache (measured BW) | — | 60 MB, **3.6 TB/s** |

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

<br>

<style>
table { font-size: 0.7em; }
th, td { padding: 2px 6px; }
</style>

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

---

# GEMM — Kernel Progression

<br>

`A(4096×16384) × B(16384×8192)` — 1.1 × 10¹² FLOPs per GEMM.

<br>

```
Naive FP32          3.7 TFLOPS  ──┐
                                  │  11× (FP32 → TF32 tensor cores)
CuTe Simple MMA    42.2 TFLOPS ──┘──┐
                                    │  2.3× (+ shared memory, larger K-tile)
CuTe SMEM k=32     95.2 TFLOPS ────┘──┐
                                       │  1.8× (SM90 wgmma, no smem→reg copy)
CuTe WGMMA        175.6 TFLOPS ───────┘──┐
                                          │  1.4× (+ double-buffered pipeline)
CuTe WGMMA Pipe   244.3 TFLOPS ──────────┘──┐
                                             │  1.2× (+ TMA, larger tile)
CuTe WGMMA TMA    300.9 TFLOPS  ────────────┘
       128×256
                                    ┌─────────────────────
cuBLAS TF32        319.7 TFLOPS ────┘  hand-written = 94% of cuBLAS
```

---

# GEMM — Results Table

<style>
table { font-size: 0.75em; }
</style>

<br>

| Kernel | Key Technique | Time (ms) | TFLOPS | % cuBLAS |
|---|---|---|---|---|
| Naive | FP32 scalar, no tiling | 296.23 | 3.7 | 1.2% |
| CuTe Simple MMA | TF32 `SM80_16x8x8`, gmem→reg | 26.04 | 42.2 | 13.2% |
| CuTe SMEM k=8 | + `cp.async` to shared mem | 17.85 | 61.6 | 19.3% |
| CuTe SMEM k=32 | + larger K-tile (4× less traffic) | 11.55 | 95.2 | 29.8% |
| CuTe WGMMA | SM90 wgmma (smem descriptors) | 6.26 | 175.6 | 54.9% |
| CuTe WGMMA Pipe | + double-buffered pipeline | 4.50 | 244.3 | 76.4% |
| WGMMA Pipe+Swiz | + CTA swizzle for L2 reuse | 4.47 | 245.9 | 76.9% |
| WGMMA TMA 128×128 | + TMA hardware loads, 3-stage | 4.62 | 238.2 | 74.5% |
| **WGMMA TMA 128×256** | **+ larger tile (128×256)** | **3.65** | **300.9** | **94.1%** |
| CUTLASS WS | Warp-specialized persistent | 4.80 | 228.9 | 71.6% |
| cuBLAS FP32 | NVIDIA hand-tuned (no TC) | 30.97 | 35.5 | 11.1% |
| **cuBLAS TF32** | **NVIDIA hand-tuned (TC)** | **3.44** | **319.7** | **100%** |

<br>

Best hand-written kernel achieves **94% of cuBLAS TF32** performance.

---

# Key Takeaways

<br>

### Hopper has a deep pipeline
29-cycle instruction latency (vs ~4 on Volta/Ampere). Need ≥32 independent
operations in flight per scheduler to approach throughput.

### Bandwidth requires massive parallelism
A single SM can only drive ~67 GB/s of HBM bandwidth.
All 132 SMs needed for 99%+ saturation (Little's Law).

### L2 is fast but not free
3.6 TB/s (4× HBM), 131 ns latency (6.5× L1).
Also needs all SMs to saturate.

### Transpose is pure memory
Once bank conflicts are eliminated (+1 padding), all kernel variants
hit 95% of HBM bandwidth — identical performance.

### GEMM optimization is layered
Each technique contributes a measurable speedup:
tensor cores (11×) → shared memory (2.3×) → wgmma (1.8×) →
pipelining (1.4×) → TMA + larger tiles (1.2×) = **81× over naive**.
