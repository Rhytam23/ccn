# GPU benchmark results (NVIDIA T600 Laptop GPU)

Timings use repeated medians; raw repetitions and paired ratios are in the CSV.

**Measured on NVIDIA T600 Laptop GPU, `quick` profile, 20,000 surface-code shots and 1,000 qLDPC shots per point; data: `results/local-gpu-corrected-quick/`. Hardware details are recorded below.**

*Throughput ratio = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. A value above 1 means the GPU pipeline is faster, below 1 means it is slower. It is not a latency.*

* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio 0.14x to 0.20x** vs the C++ `ldpc` BP+OSD (best: BB[[72,12,6]], p=0.04, 0.20x; the GPU is slower at points below 1.00x). Logical-error counts are identical at every point on the same shots (not a proof of equivalence; intervals in the tables below).
  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*
* **Surface code (circuit-level noise), conservative r=2 rule: throughput ratio 0.50x to 0.74x** (0 of 4 points above 1.02x; 0 more logical errors than PyMatching, summed over the points where ours was worse).
* **Surface code (circuit-level noise), fast r=1 rule: throughput ratio 0.66x to 0.94x** (0 of 4 points above 1.02x; 23 more logical errors than PyMatching, summed over the points where ours was worse).
* Best single surface-code point: fast r=1 rule, d=5, p=0.004, ratio 0.94x.

## qLDPC: GPU batched BP vs C++ ldpc

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) | LER GPU [95% CI] | LER ldpc [95% CI] |
|---|---|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 120,220 | 16,835 | **0.14x** | 10 / 10 | 1.00e-02 [5.4e-03, 1.8e-02] | 1.00e-02 [5.4e-03, 1.8e-02] |
| BB[[72,12,6]] | 0.04 | 61,117 | 12,452 | **0.20x** | 88 / 88 | 8.80e-02 [7.2e-02, 1.1e-01] | 8.80e-02 [7.2e-02, 1.1e-01] |

## Surface code: GPU local pre-decoder (conservative radius-2 and fast radius-1 rules, fp16 stage 1) + PyMatching vs PyMatching alone

| d | p | rule | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | throughput ratio | PyMatching-stage time ratio | logical errors (ours / PyMatching) | LER ours [95% CI] | LER PyMatching [95% CI] | syndrome weight left |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.002 | conservative r=2 | 1,055,320 | 618,443 | **0.59x** | 1.60x | 23 / 23 | 1.15e-03 [7.7e-04, 1.7e-03] | 1.15e-03 [7.7e-04, 1.7e-03] | 55 % |
| 5 | 0.002 | fast r=1 | 1,055,320 | 786,624 | **0.75x** | 2.00x | 24 / 23 | 1.20e-03 [8.1e-04, 1.8e-03] | 1.15e-03 [7.7e-04, 1.7e-03] | 37 % |
| 5 | 0.004 | conservative r=2 | 512,344 | 379,561 | **0.74x** | 1.22x | 148 / 148 | 7.40e-03 [6.3e-03, 8.7e-03] | 7.40e-03 [6.3e-03, 8.7e-03] | 73 % |
| 5 | 0.004 | fast r=1 | 512,344 | 480,303 | **0.94x** | 1.51x | 154 / 148 | 7.70e-03 [6.6e-03, 9.0e-03] | 7.40e-03 [6.3e-03, 8.7e-03] | 50 % |
| 7 | 0.002 | conservative r=2 | 348,749 | 173,490 | **0.50x** | 1.35x | 2 / 2 | 1.00e-04 [2.7e-05, 3.6e-04] | 1.00e-04 [2.7e-05, 3.6e-04] | 63 % |
| 7 | 0.002 | fast r=1 | 348,749 | 229,520 | **0.66x** | 1.82x | 5 / 2 | 2.50e-04 [1.1e-04, 5.9e-04] | 1.00e-04 [2.7e-05, 3.6e-04] | 37 % |
| 7 | 0.004 | conservative r=2 | 157,970 | 101,869 | **0.64x** | 1.09x | 85 / 85 | 4.25e-03 [3.4e-03, 5.3e-03] | 4.25e-03 [3.4e-03, 5.3e-03] | 83 % |
| 7 | 0.004 | fast r=1 | 157,970 | 133,257 | **0.84x** | 1.39x | 98 / 85 | 4.90e-03 [4.0e-03, 6.0e-03] | 4.25e-03 [3.4e-03, 5.3e-03] | 54 % |

Caveats: batch throughput only (single-shot p50 latency was 575-702 µs on the GPU vs 7-13 µs for PyMatching, and at 256-shot micro-batches the GPU pipeline was slower at 8 of 8 rows (faster at 0)); d=5 to 7 with 20,000 shots per point; low error counts do not establish accuracy equivalence. The qLDPC rows use only 1,000 shots; consult a larger repeated run before interpreting GPU scaling. Both rules are heuristics: inspect the per-point error counts, including any radius-2 degradation. Ratios are relative to CPU baselines on the same machine.

## Repeated timing dispersion

Paired ratios use the corresponding CPU repetition; these ranges are timing spread, not confidence intervals.

| case | decoder | throughput min–max (shots/s) | paired ratio median [min, max] |
|---|---|---|---|
| d=5, p=0.002 | zero-syndrome shortcut + pymatching | 950,760–1,016,203 | 0.949x [0.912, 0.975] |
| d=5, p=0.002 | GPU local pre-decoder (conservative, r=2) + pymatching | 582,842–620,923 | 0.583x [0.548, 0.594] |
| d=5, p=0.002 | GPU local pre-decoder (fast, r=1) + pymatching | 747,448–794,291 | 0.746x [0.700, 0.755] |
| d=5, p=0.004 | zero-syndrome shortcut + pymatching | 489,899–497,759 | 0.969x [0.961, 0.989] |
| d=5, p=0.004 | GPU local pre-decoder (conservative, r=2) + pymatching | 370,832–384,295 | 0.749x [0.720, 0.754] |
| d=5, p=0.004 | GPU local pre-decoder (fast, r=1) + pymatching | 453,695–481,328 | 0.937x [0.881, 0.972] |
| d=7, p=0.002 | zero-syndrome shortcut + pymatching | 297,925–333,373 | 0.948x [0.868, 0.970] |
| d=7, p=0.002 | GPU local pre-decoder (conservative, r=2) + pymatching | 112,520–175,089 | 0.497x [0.328, 0.505] |
| d=7, p=0.002 | GPU local pre-decoder (fast, r=1) + pymatching | 186,375–230,933 | 0.657x [0.543, 0.668] |
| d=7, p=0.004 | zero-syndrome shortcut + pymatching | 144,260–154,586 | 0.974x [0.908, 0.976] |
| d=7, p=0.004 | GPU local pre-decoder (conservative, r=2) + pymatching | 93,569–107,949 | 0.655x [0.590, 0.689] |
| d=7, p=0.004 | GPU local pre-decoder (fast, r=1) + pymatching | 119,185–136,586 | 0.852x [0.752, 0.867] |
| BB[[72,12,6]], p=0.02 | GPU min-sum BP + OSD fallback | 16,177–17,185 | 0.143x [0.134, 0.146] |
| BB[[72,12,6]], p=0.04 | GPU min-sum BP + OSD fallback | 12,073–12,733 | 0.203x [0.193, 0.208] |

## Environment (from `results/local-gpu-corrected-quick/meta.json`)

| item | value |
|---|---|
| GPU | NVIDIA T600 Laptop GPU |
| GPU memory (GB) | 4.0 |
| NVIDIA driver | 596.71 |
| CUDA (torch build) | 13.0 |
| Python | 3.13.7 |
| PyTorch | 2.14.1+cu130 |
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
| Benchmark date | 2026-10-03 01:54:41 |
