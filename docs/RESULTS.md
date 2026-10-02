# GPU benchmark results (Tesla T4)

**Measured on Tesla T4 (Google Colab, 2 vCPU), `full` profile, 200,000 surface-code shots and 20,000 qLDPC shots per point; data: `results/colab-gpu/`.**

*Throughput ratio = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. A value above 1 means the GPU pipeline is faster, below 1 means it is slower. It is not a latency.*

* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio 0.98x to 1.58x** vs the C++ `ldpc` BP+OSD (best: BB[[144,12,12]], p=0.02, 1.58x; the GPU is slightly slower at the points below 1.00x). Logical-error counts are identical on the same shots at every point (not a proof of equivalence; intervals in `docs/RESULTS.md`).
  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*
* **Surface code (circuit-level noise), first version (r=1) rule: throughput ratio 0.62x to 1.19x** (4 of 12 points above 1.02x; 344 more logical errors than PyMatching, summed over the points where ours was worse).
* Best single surface-code point: first version (r=1) rule, d=5, p=0.002, ratio 1.19x.

## qLDPC: GPU batched BP vs C++ ldpc

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) | LER GPU [95% CI] | LER ldpc [95% CI] |
|---|---|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 77,638 | 86,407 | **1.11x** | 226 / 226 | 1.13e-02 [9.9e-03, 1.3e-02] | 1.13e-02 [9.9e-03, 1.3e-02] |
| BB[[72,12,6]] | 0.04 | 39,769 | 38,941 | **0.98x** | 1770 / 1770 | 8.85e-02 [8.5e-02, 9.3e-02] | 8.85e-02 [8.5e-02, 9.3e-02] |
| BB[[72,12,6]] | 0.06 | 18,921 | 21,240 | **1.12x** | 4991 / 4991 | 2.50e-01 [2.4e-01, 2.6e-01] | 2.50e-01 [2.4e-01, 2.6e-01] |
| BB[[144,12,12]] | 0.02 | 39,426 | 62,340 | **1.58x** | 13 / 13 | 6.50e-04 [3.8e-04, 1.1e-03] | 6.50e-04 [3.8e-04, 1.1e-03] |
| BB[[144,12,12]] | 0.04 | 19,885 | 24,982 | **1.26x** | 233 / 233 | 1.17e-02 [1.0e-02, 1.3e-02] | 1.17e-02 [1.0e-02, 1.3e-02] |
| BB[[144,12,12]] | 0.06 | 7,627 | 7,853 | **1.03x** | 1772 / 1772 | 8.86e-02 [8.5e-02, 9.3e-02] | 8.86e-02 [8.5e-02, 9.3e-02] |

## Surface code: GPU local pre-decoder (first, aggressive radius-1 rule) + PyMatching vs PyMatching alone

| d | p | rule | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | throughput ratio | PyMatching-stage time ratio | logical errors (ours / PyMatching) | LER ours [95% CI] | LER PyMatching [95% CI] | syndrome weight left |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.001 | first version (r=1) | 1,407,668 | 1,411,246 | **1.00x** | 3.20x | 23 / 23 | 1.15e-04 [7.7e-05, 1.7e-04] | 1.15e-04 [7.7e-05, 1.7e-04] | 29 % |
| 5 | 0.002 | first version (r=1) | 748,339 | 892,596 | **1.19x** | 2.16x | 196 / 182 | 9.80e-04 [8.5e-04, 1.1e-03] | 9.10e-04 [7.9e-04, 1.1e-03] | 37 % |
| 5 | 0.004 | first version (r=1) | 375,909 | 445,623 | **1.19x** | 1.54x | 1550 / 1532 | 7.75e-03 [7.4e-03, 8.1e-03] | 7.66e-03 [7.3e-03, 8.1e-03] | 50 % |
| 7 | 0.001 | first version (r=1) | 449,641 | 366,068 | **0.81x** | 3.11x | 10 / 5 | 5.00e-05 [2.7e-05, 9.2e-05] | 2.50e-05 [1.1e-05, 5.9e-05] | 26 % |
| 7 | 0.002 | first version (r=1) | 241,949 | 244,920 | **1.01x** | 2.03x | 75 / 50 | 3.75e-04 [3.0e-04, 4.7e-04] | 2.50e-04 [1.9e-04, 3.3e-04] | 37 % |
| 7 | 0.004 | first version (r=1) | 112,760 | 119,404 | **1.06x** | 1.41x | 970 / 859 | 4.85e-03 [4.6e-03, 5.2e-03] | 4.30e-03 [4.0e-03, 4.6e-03] | 54 % |
| 9 | 0.001 | first version (r=1) | 197,373 | 149,050 | **0.76x** | 2.67x | 0 / 0 | 0.00e+00 [0.0e+00, 1.9e-05] | 0.00e+00 [0.0e+00, 1.9e-05] | 25 % |
| 9 | 0.002 | first version (r=1) | 105,566 | 101,249 | **0.96x** | 1.86x | 30 / 13 | 1.50e-04 [1.1e-04, 2.1e-04] | 6.50e-05 [3.8e-05, 1.1e-04] | 37 % |
| 9 | 0.004 | first version (r=1) | 47,091 | 48,560 | **1.03x** | 1.38x | 586 / 486 | 2.93e-03 [2.7e-03, 3.2e-03] | 2.43e-03 [2.2e-03, 2.7e-03] | 56 % |
| 13 | 0.001 | first version (r=1) | 59,860 | 36,976 | **0.62x** | 2.52x | 0 / 0 | 0.00e+00 [0.0e+00, 1.9e-05] | 0.00e+00 [0.0e+00, 1.9e-05] | 24 % |
| 13 | 0.002 | first version (r=1) | 32,605 | 26,710 | **0.82x** | 1.85x | 2 / 0 | 1.00e-05 [2.7e-06, 3.6e-05] | 0.00e+00 [0.0e+00, 1.9e-05] | 38 % |
| 13 | 0.004 | first version (r=1) | 13,293 | 12,726 | **0.96x** | 1.32x | 161 / 109 | 8.05e-04 [6.9e-04, 9.4e-04] | 5.45e-04 [4.5e-04, 6.6e-04] | 58 % |

Caveats: batch throughput only (single-shot latency is worse on the GPU than on the CPU); the surface-code rows are for the first, aggressive local rule, which also adds logical errors at some points (compare the error columns); the fp16 stage 1 and the `radius=2` rule were not part of this run. Ratios are relative to CPU baselines on the same Colab machine.

## Environment (from `results/colab-gpu/meta.json`)

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
| Benchmark date | 2026-10-02 08:17:26 |
