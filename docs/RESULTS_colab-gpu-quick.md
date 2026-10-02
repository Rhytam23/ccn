# GPU benchmark results (Tesla T4)

**Measured on Tesla T4 (Google Colab, 2 vCPU), `quick` profile, 20,000 surface-code shots and 1,000 qLDPC shots per point; data: `results/colab-gpu-quick/`.**

*Throughput ratio = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. A value above 1 means the GPU pipeline is faster, below 1 means it is slower. It is not a latency.*

* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio 0.18x to 0.22x** vs the C++ `ldpc` BP+OSD (best: BB[[72,12,6]], p=0.02, 0.22x; the GPU is slightly slower at the points below 1.00x). Logical-error counts are identical on the same shots at every point (not a proof of equivalence; intervals in `docs/RESULTS.md`).
  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*
* **Surface code (circuit-level noise), fast r=1 rule: throughput ratio 0.90x to 2.53x** (3 of 4 points above 1.02x; 12 more logical errors than PyMatching, summed over the points where ours was worse).
* **Surface code (circuit-level noise), safe r=2 rule: throughput ratio 1.08x to 1.32x** (4 of 4 points above 1.02x; 1 more logical errors than PyMatching, summed over the points where ours was worse).
* Best single surface-code point: fast r=1 rule, d=7, p=0.002, ratio 2.53x.

## qLDPC: GPU batched BP vs C++ ldpc

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) | LER GPU [95% CI] | LER ldpc [95% CI] |
|---|---|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 83,117 | 18,542 | **0.22x** | 10 / 10 | 1.00e-02 [5.4e-03, 1.8e-02] | 1.00e-02 [5.4e-03, 1.8e-02] |
| BB[[72,12,6]] | 0.04 | 45,441 | 8,039 | **0.18x** | 88 / 88 | 8.80e-02 [7.2e-02, 1.1e-01] | 8.80e-02 [7.2e-02, 1.1e-01] |

## Surface code: GPU local pre-decoder (safe radius-2 and fast radius-1 rules, fp16 stage 1) + PyMatching vs PyMatching alone

| d | p | rule | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | throughput ratio | PyMatching-stage time ratio | logical errors (ours / PyMatching) | LER ours [95% CI] | LER PyMatching [95% CI] | syndrome weight left |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.002 | safe r=2 | 887,497 | 1,045,061 | **1.18x** | 1.39x | 27 / 27 | 1.35e-03 [9.3e-04, 2.0e-03] | 1.35e-03 [9.3e-04, 2.0e-03] | 55 % |
| 5 | 0.002 | fast r=1 | 887,497 | 1,292,647 | **1.46x** | 1.77x | 27 / 27 | 1.35e-03 [9.3e-04, 2.0e-03] | 1.35e-03 [9.3e-04, 2.0e-03] | 37 % |
| 5 | 0.004 | safe r=2 | 401,090 | 434,188 | **1.08x** | 1.16x | 155 / 155 | 7.75e-03 [6.6e-03, 9.1e-03] | 7.75e-03 [6.6e-03, 9.1e-03] | 73 % |
| 5 | 0.004 | fast r=1 | 401,090 | 359,338 | **0.90x** | 0.95x | 158 / 155 | 7.90e-03 [6.8e-03, 9.2e-03] | 7.75e-03 [6.6e-03, 9.1e-03] | 51 % |
| 7 | 0.002 | safe r=2 | 161,357 | 212,478 | **1.32x** | 1.45x | 5 / 5 | 2.50e-04 [1.1e-04, 5.9e-04] | 2.50e-04 [1.1e-04, 5.9e-04] | 63 % |
| 7 | 0.002 | fast r=1 | 161,357 | 407,567 | **2.53x** | 2.95x | 5 / 5 | 2.50e-04 [1.1e-04, 5.9e-04] | 2.50e-04 [1.1e-04, 5.9e-04] | 37 % |
| 7 | 0.004 | safe r=2 | 106,940 | 123,290 | **1.15x** | 1.21x | 72 / 71 | 3.60e-03 [2.9e-03, 4.5e-03] | 3.55e-03 [2.8e-03, 4.5e-03] | 83 % |
| 7 | 0.004 | fast r=1 | 106,940 | 158,455 | **1.48x** | 1.57x | 80 / 71 | 4.00e-03 [3.2e-03, 5.0e-03] | 3.55e-03 [2.8e-03, 4.5e-03] | 53 % |

Caveats: batch throughput only (single-shot p50 latency was 630-881 µs on the GPU vs 18-38 µs for PyMatching, and at 256-shot micro-batches the GPU pipeline was slower at 6 of 8 rows (faster at 2)); only d=5 and d=7 with 20,000 shots per point, so error counts are small (a difference of a few errors is within noise); the qLDPC rows use only 1,000 shots, too few to load a GPU, so the GPU BP looks slow here: the 20,000-shot `full` run (`results/colab-gpu/`) is the fair qLDPC comparison. The `safe` rule showed no extra logical errors in these runs; the `fast` rule trades some logical errors for throughput. Ratios are relative to CPU baselines on the same Colab machine.

## Environment (from `results/colab-gpu-quick/meta.json`)

| item | value |
|---|---|
| GPU | Tesla T4 |
| GPU memory (GB) | not recorded in this run |
| CUDA (torch build) | not recorded in this run |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| Stim | 1.16.0 |
| PyMatching | 2.4.0 |
| ldpc | not recorded in this run |
| CUDA-Q QEC | unavailable (ModuleNotFoundError: No module named 'cudaq_qec') |
| CPU cores (Colab) | 2 |
| Platform | Linux-6.6.122+-x86_64-with-glibc2.39 |
| Git commit | not recorded in this run |
| Benchmark date | 2026-10-02 10:30:58 |
