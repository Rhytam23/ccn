# GPU benchmark results (CPU)

Timings use repeated medians; raw repetitions and paired ratios are in the CSV.

**Measured on CPU, `quick` profile, 20,000 surface-code shots and 1,000 qLDPC shots per point; data: `results/local-cpu-corrected/`. Hardware details are recorded below.**

*Throughput ratio = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. A value above 1 means the GPU pipeline is faster, below 1 means it is slower. It is not a latency.*

* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio 0.29x to 0.29x** vs the C++ `ldpc` BP+OSD (best: BB[[72,12,6]], p=0.04, 0.29x; the GPU is slower at points below 1.00x). Logical-error counts are identical at every point on the same shots (not a proof of equivalence; intervals in the tables below).
  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*
* **Surface code (circuit-level noise), conservative r=2 rule: throughput ratio 0.15x to 0.28x** (0 of 4 points above 1.02x; 0 more logical errors than PyMatching, summed over the points where ours was worse).
* **Surface code (circuit-level noise), fast r=1 rule: throughput ratio 0.23x to 0.36x** (0 of 4 points above 1.02x; 23 more logical errors than PyMatching, summed over the points where ours was worse).
* Best single surface-code point: fast r=1 rule, d=7, p=0.004, ratio 0.36x.

## qLDPC: GPU batched BP vs C++ ldpc

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) | LER GPU [95% CI] | LER ldpc [95% CI] |
|---|---|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 124,060 | 36,297 | **0.29x** | 10 / 10 | 1.00e-02 [5.4e-03, 1.8e-02] | 1.00e-02 [5.4e-03, 1.8e-02] |
| BB[[72,12,6]] | 0.04 | 56,938 | 16,748 | **0.29x** | 88 / 88 | 8.80e-02 [7.2e-02, 1.1e-01] | 8.80e-02 [7.2e-02, 1.1e-01] |

## Surface code: GPU local pre-decoder (conservative radius-2 and fast radius-1 rules, fp16 stage 1) + PyMatching vs PyMatching alone

| d | p | rule | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | throughput ratio | PyMatching-stage time ratio | logical errors (ours / PyMatching) | LER ours [95% CI] | LER PyMatching [95% CI] | syndrome weight left |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.002 | conservative r=2 | 1,013,510 | 183,455 | **0.18x** | 1.65x | 23 / 23 | 1.15e-03 [7.7e-04, 1.7e-03] | 1.15e-03 [7.7e-04, 1.7e-03] | 55 % |
| 5 | 0.002 | fast r=1 | 1,013,510 | 234,584 | **0.23x** | 2.03x | 24 / 23 | 1.20e-03 [8.1e-04, 1.8e-03] | 1.15e-03 [7.7e-04, 1.7e-03] | 37 % |
| 5 | 0.004 | conservative r=2 | 501,410 | 141,900 | **0.28x** | 1.26x | 148 / 148 | 7.40e-03 [6.3e-03, 8.7e-03] | 7.40e-03 [6.3e-03, 8.7e-03] | 73 % |
| 5 | 0.004 | fast r=1 | 501,410 | 177,336 | **0.35x** | 1.50x | 154 / 148 | 7.70e-03 [6.6e-03, 9.0e-03] | 7.40e-03 [6.3e-03, 8.7e-03] | 50 % |
| 7 | 0.002 | conservative r=2 | 344,828 | 51,918 | **0.15x** | 1.32x | 2 / 2 | 1.00e-04 [2.7e-05, 3.6e-04] | 1.00e-04 [2.7e-05, 3.6e-04] | 63 % |
| 7 | 0.002 | fast r=1 | 344,828 | 79,070 | **0.23x** | 1.87x | 5 / 2 | 2.50e-04 [1.1e-04, 5.9e-04] | 1.00e-04 [2.7e-05, 3.6e-04] | 37 % |
| 7 | 0.004 | conservative r=2 | 160,111 | 44,014 | **0.27x** | 1.07x | 85 / 85 | 4.25e-03 [3.4e-03, 5.3e-03] | 4.25e-03 [3.4e-03, 5.3e-03] | 83 % |
| 7 | 0.004 | fast r=1 | 160,111 | 58,183 | **0.36x** | 1.24x | 98 / 85 | 4.90e-03 [4.0e-03, 6.0e-03] | 4.25e-03 [3.4e-03, 5.3e-03] | 54 % |

Caveats: batch throughput only (single-shot p50 latency was 142-199 µs on the GPU vs 7-13 µs for PyMatching, and at 256-shot micro-batches the GPU pipeline was slower at 8 of 8 rows (faster at 0)); d=5 to 7 with 20,000 shots per point; low error counts do not establish accuracy equivalence. The qLDPC rows use only 1,000 shots; consult a larger repeated run before interpreting GPU scaling. Both rules are heuristics: inspect the per-point error counts, including any radius-2 degradation. Ratios are relative to CPU baselines on the same machine.

## Repeated timing dispersion

Paired ratios use the corresponding CPU repetition; these ranges are timing spread, not confidence intervals.

| case | decoder | throughput min–max (shots/s) | paired ratio median [min, max] |
|---|---|---|---|
| d=5, p=0.002 | zero-syndrome shortcut + pymatching | 942,369–1,042,047 | 0.978x [0.934, 1.006] |
| d=5, p=0.002 | GPU local pre-decoder (conservative, r=2) + pymatching | 149,643–190,029 | 0.173x [0.149, 0.181] |
| d=5, p=0.002 | GPU local pre-decoder (fast, r=1) + pymatching | 186,744–262,667 | 0.231x [0.185, 0.253] |
| d=5, p=0.004 | zero-syndrome shortcut + pymatching | 462,163–500,636 | 0.963x [0.927, 0.998] |
| d=5, p=0.004 | GPU local pre-decoder (conservative, r=2) + pymatching | 129,143–145,492 | 0.284x [0.253, 0.304] |
| d=5, p=0.004 | GPU local pre-decoder (fast, r=1) + pymatching | 162,148–201,553 | 0.348x [0.339, 0.402] |
| d=7, p=0.002 | zero-syndrome shortcut + pymatching | 324,301–335,683 | 0.960x [0.948, 0.968] |
| d=7, p=0.002 | GPU local pre-decoder (conservative, r=2) + pymatching | 42,291–58,294 | 0.151x [0.124, 0.165] |
| d=7, p=0.002 | GPU local pre-decoder (fast, r=1) + pymatching | 74,473–83,337 | 0.232x [0.216, 0.236] |
| d=7, p=0.004 | zero-syndrome shortcut + pymatching | 153,896–156,397 | 0.969x [0.963, 0.980] |
| d=7, p=0.004 | GPU local pre-decoder (conservative, r=2) + pymatching | 40,676–47,568 | 0.273x [0.255, 0.295] |
| d=7, p=0.004 | GPU local pre-decoder (fast, r=1) + pymatching | 43,416–64,182 | 0.361x [0.271, 0.398] |
| BB[[72,12,6]], p=0.02 | GPU min-sum BP + OSD fallback | 33,716–38,869 | 0.302x [0.272, 0.308] |
| BB[[72,12,6]], p=0.04 | GPU min-sum BP + OSD fallback | 14,421–17,965 | 0.295x [0.275, 0.316] |

## Environment (from `results/local-cpu-corrected/meta.json`)

| item | value |
|---|---|
| GPU | not recorded in this run |
| GPU memory (GB) | not recorded in this run |
| NVIDIA driver | not recorded in this run |
| CUDA (torch build) | not recorded in this run |
| Python | 3.13.7 |
| PyTorch | 2.14.1+cpu |
| Stim | 1.16.0 |
| PyMatching | 2.4.0 |
| ldpc | 2.4.1 |
| CUDA-Q QEC | unavailable (ModuleNotFoundError: No module named 'cudaq_qec') |
| CPU logical cores | 16 |
| Torch threads | 2 |
| Platform | Windows-11-10.0.26200-SP0 |
| Git commit | 1e0c610 |
| Git dirty | True |
| Source SHA256 | d55de1ba5484be92d8fddefe8506cd59350784bc4f5c91cd72eb02d8f3dfed82 |
| Protocol | median-paired-v2 |
| Throughput repetitions | 5 |
| Benchmark date | 2026-10-03 01:48:56 |
