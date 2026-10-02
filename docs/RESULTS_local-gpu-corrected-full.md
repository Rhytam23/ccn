# GPU benchmark results (NVIDIA T600 Laptop GPU)

Timings use repeated medians; raw repetitions and paired ratios are in the CSV.

**Measured on NVIDIA T600 Laptop GPU, `full` profile, 200,000 surface-code shots and 20,000 qLDPC shots per point; data: `results/local-gpu-corrected-full/`. Hardware details are recorded below.**

*Throughput ratio = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. A value above 1 means the GPU pipeline is faster, below 1 means it is slower. It is not a latency.*

* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio 0.59x to 0.87x** vs the C++ `ldpc` BP+OSD (best: BB[[144,12,12]], p=0.02, 0.87x; the GPU is slower at points below 1.00x). Logical-error counts are identical at every point on the same shots (not a proof of equivalence; intervals in the tables below).
  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*
* **Surface code (circuit-level noise), conservative r=2 rule: throughput ratio 0.09x to 0.81x** (0 of 12 points above 1.02x; 5 more logical errors than PyMatching, summed over the points where ours was worse).
* **Surface code (circuit-level noise), fast r=1 rule: throughput ratio 0.12x to 1.12x** (2 of 12 points above 1.02x; 418 more logical errors than PyMatching, summed over the points where ours was worse).
* Best single surface-code point: fast r=1 rule, d=7, p=0.004, ratio 1.12x.

## qLDPC: GPU batched BP vs C++ ldpc

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) | LER GPU [95% CI] | LER ldpc [95% CI] |
|---|---|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 92,736 | 72,570 | **0.78x** | 227 / 227 | 1.14e-02 [1.0e-02, 1.3e-02] | 1.14e-02 [1.0e-02, 1.3e-02] |
| BB[[72,12,6]] | 0.04 | 49,546 | 34,775 | **0.70x** | 1771 / 1771 | 8.86e-02 [8.5e-02, 9.3e-02] | 8.86e-02 [8.5e-02, 9.3e-02] |
| BB[[72,12,6]] | 0.06 | 24,611 | 14,448 | **0.59x** | 4989 / 4989 | 2.49e-01 [2.4e-01, 2.6e-01] | 2.49e-01 [2.4e-01, 2.6e-01] |
| BB[[144,12,12]] | 0.02 | 60,457 | 52,297 | **0.87x** | 13 / 13 | 6.50e-04 [3.8e-04, 1.1e-03] | 6.50e-04 [3.8e-04, 1.1e-03] |
| BB[[144,12,12]] | 0.04 | 25,648 | 19,628 | **0.77x** | 233 / 233 | 1.17e-02 [1.0e-02, 1.3e-02] | 1.17e-02 [1.0e-02, 1.3e-02] |
| BB[[144,12,12]] | 0.06 | 9,970 | 6,387 | **0.64x** | 1773 / 1773 | 8.87e-02 [8.5e-02, 9.3e-02] | 8.87e-02 [8.5e-02, 9.3e-02] |

## Surface code: GPU local pre-decoder (conservative radius-2 and fast radius-1 rules, fp16 stage 1) + PyMatching vs PyMatching alone

| d | p | rule | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | throughput ratio | PyMatching-stage time ratio | logical errors (ours / PyMatching) | LER ours [95% CI] | LER PyMatching [95% CI] | syndrome weight left |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.001 | conservative r=2 | 2,027,682 | 1,272,715 | **0.63x** | 2.27x | 17 / 17 | 8.50e-05 [5.3e-05, 1.4e-04] | 8.50e-05 [5.3e-05, 1.4e-04] | 39 % |
| 5 | 0.001 | fast r=1 | 2,027,682 | 1,582,342 | **0.78x** | 2.76x | 19 / 17 | 9.50e-05 [6.1e-05, 1.5e-04] | 8.50e-05 [5.3e-05, 1.4e-04] | 29 % |
| 5 | 0.002 | conservative r=2 | 1,059,367 | 859,974 | **0.81x** | 1.61x | 204 / 204 | 1.02e-03 [8.9e-04, 1.2e-03] | 1.02e-03 [8.9e-04, 1.2e-03] | 54 % |
| 5 | 0.002 | fast r=1 | 1,059,367 | 1,093,702 | **1.03x** | 2.04x | 214 / 204 | 1.07e-03 [9.4e-04, 1.2e-03] | 1.02e-03 [8.9e-04, 1.2e-03] | 37 % |
| 5 | 0.004 | conservative r=2 | 510,418 | 320,568 | **0.63x** | 1.23x | 1529 / 1529 | 7.64e-03 [7.3e-03, 8.0e-03] | 7.64e-03 [7.3e-03, 8.0e-03] | 74 % |
| 5 | 0.004 | fast r=1 | 510,418 | 503,439 | **0.99x** | 1.51x | 1562 / 1529 | 7.81e-03 [7.4e-03, 8.2e-03] | 7.64e-03 [7.3e-03, 8.0e-03] | 50 % |
| 7 | 0.001 | conservative r=2 | 667,406 | 227,180 | **0.34x** | 1.90x | 3 / 3 | 1.50e-05 [5.1e-06, 4.4e-05] | 1.50e-05 [5.1e-06, 4.4e-05] | 44 % |
| 7 | 0.001 | fast r=1 | 667,406 | 299,129 | **0.45x** | 2.57x | 10 / 3 | 5.00e-05 [2.7e-05, 9.2e-05] | 1.50e-05 [5.1e-06, 4.4e-05] | 26 % |
| 7 | 0.002 | conservative r=2 | 329,693 | 143,445 | **0.44x** | 1.42x | 52 / 52 | 2.60e-04 [2.0e-04, 3.4e-04] | 2.60e-04 [2.0e-04, 3.4e-04] | 63 % |
| 7 | 0.002 | fast r=1 | 329,693 | 227,911 | **0.69x** | 1.88x | 73 / 52 | 3.65e-04 [2.9e-04, 4.6e-04] | 2.60e-04 [2.0e-04, 3.4e-04] | 37 % |
| 7 | 0.004 | conservative r=2 | 117,143 | 94,102 | **0.80x** | 1.41x | 883 / 882 | 4.41e-03 [4.1e-03, 4.7e-03] | 4.41e-03 [4.1e-03, 4.7e-03] | 83 % |
| 7 | 0.004 | fast r=1 | 117,143 | 130,740 | **1.12x** | 1.75x | 1020 / 882 | 5.10e-03 [4.8e-03, 5.4e-03] | 4.41e-03 [4.1e-03, 4.7e-03] | 54 % |
| 9 | 0.001 | conservative r=2 | 241,152 | 56,777 | **0.24x** | 1.76x | 1 / 0 | 5.00e-06 [8.8e-07, 2.8e-05] | 0.00e+00 [0.0e+00, 1.9e-05] | 47 % |
| 9 | 0.001 | fast r=1 | 241,152 | 82,034 | **0.34x** | 2.60x | 1 / 0 | 5.00e-06 [8.8e-07, 2.8e-05] | 0.00e+00 [0.0e+00, 1.9e-05] | 25 % |
| 9 | 0.002 | conservative r=2 | 121,116 | 47,850 | **0.40x** | 1.47x | 11 / 11 | 5.50e-05 [3.1e-05, 9.8e-05] | 5.50e-05 [3.1e-05, 9.8e-05] | 68 % |
| 9 | 0.002 | fast r=1 | 121,116 | 68,700 | **0.57x** | 2.09x | 29 / 11 | 1.45e-04 [1.0e-04, 2.1e-04] | 5.50e-05 [3.1e-05, 9.8e-05] | 37 % |
| 9 | 0.004 | conservative r=2 | 60,265 | 32,541 | **0.54x** | 1.02x | 449 / 448 | 2.25e-03 [2.0e-03, 2.5e-03] | 2.24e-03 [2.0e-03, 2.5e-03] | 87 % |
| 9 | 0.004 | fast r=1 | 60,265 | 43,374 | **0.72x** | 1.33x | 558 / 448 | 2.79e-03 [2.6e-03, 3.0e-03] | 2.24e-03 [2.0e-03, 2.5e-03] | 56 % |
| 13 | 0.001 | conservative r=2 | 85,691 | 7,855 | **0.09x** | 1.52x | 0 / 0 | 0.00e+00 [0.0e+00, 1.9e-05] | 0.00e+00 [0.0e+00, 1.9e-05] | 51 % |
| 13 | 0.001 | fast r=1 | 85,691 | 10,579 | **0.12x** | 2.25x | 0 / 0 | 0.00e+00 [0.0e+00, 1.9e-05] | 0.00e+00 [0.0e+00, 1.9e-05] | 24 % |
| 13 | 0.002 | conservative r=2 | 40,629 | 7,182 | **0.18x** | 1.26x | 1 / 1 | 5.00e-06 [8.8e-07, 2.8e-05] | 5.00e-06 [8.8e-07, 2.8e-05] | 73 % |
| 13 | 0.002 | fast r=1 | 40,629 | 9,709 | **0.24x** | 1.80x | 2 / 1 | 1.00e-05 [2.7e-06, 3.6e-05] | 5.00e-06 [8.8e-07, 2.8e-05] | 38 % |
| 13 | 0.004 | conservative r=2 | 19,202 | 5,687 | **0.30x** | 1.01x | 118 / 116 | 5.90e-04 [4.9e-04, 7.1e-04] | 5.80e-04 [4.8e-04, 7.0e-04] | 91 % |
| 13 | 0.004 | fast r=1 | 19,202 | 7,269 | **0.38x** | 1.18x | 193 / 116 | 9.65e-04 [8.4e-04, 1.1e-03] | 5.80e-04 [4.8e-04, 7.0e-04] | 58 % |

Caveats: batch throughput only (single-shot p50 latency was 585-1580 µs on the GPU vs 6-62 µs for PyMatching, and at 256-shot micro-batches the GPU pipeline was slower at 24 of 24 rows (faster at 0)); d=5 to 13 with 200,000 shots per point; low error counts do not establish accuracy equivalence. Both rules are heuristics: inspect the per-point error counts, including any radius-2 degradation. Ratios are relative to CPU baselines on the same machine.

## Repeated timing dispersion

Paired ratios use the corresponding CPU repetition; these ranges are timing spread, not confidence intervals.

| case | decoder | throughput min–max (shots/s) | paired ratio median [min, max] |
|---|---|---|---|
| d=5, p=0.001 | zero-syndrome shortcut + pymatching | 1,884,311–1,933,570 | 0.953x [0.929, 0.963] |
| d=5, p=0.001 | GPU local pre-decoder (conservative, r=2) + pymatching | 887,630–1,275,302 | 0.624x [0.443, 0.636] |
| d=5, p=0.001 | GPU local pre-decoder (fast, r=1) + pymatching | 1,432,080–1,595,970 | 0.782x [0.715, 0.789] |
| d=5, p=0.002 | zero-syndrome shortcut + pymatching | 1,001,102–1,032,947 | 0.966x [0.950, 1.010] |
| d=5, p=0.002 | GPU local pre-decoder (conservative, r=2) + pymatching | 495,146–875,272 | 0.808x [0.484, 0.835] |
| d=5, p=0.002 | GPU local pre-decoder (fast, r=1) + pymatching | 1,013,966–1,100,070 | 1.016x [0.940, 1.065] |
| d=5, p=0.004 | zero-syndrome shortcut + pymatching | 480,664–499,867 | 0.963x [0.949, 0.979] |
| d=5, p=0.004 | GPU local pre-decoder (conservative, r=2) + pymatching | 292,337–411,734 | 0.628x [0.576, 0.806] |
| d=5, p=0.004 | GPU local pre-decoder (fast, r=1) + pymatching | 490,140–512,021 | 0.977x [0.964, 1.011] |
| d=7, p=0.001 | zero-syndrome shortcut + pymatching | 599,737–627,158 | 0.922x [0.900, 0.940] |
| d=7, p=0.001 | GPU local pre-decoder (conservative, r=2) + pymatching | 172,541–227,451 | 0.335x [0.259, 0.347] |
| d=7, p=0.001 | GPU local pre-decoder (fast, r=1) + pymatching | 296,769–299,864 | 0.449x [0.436, 0.456] |
| d=7, p=0.002 | zero-syndrome shortcut + pymatching | 227,664–326,179 | 0.963x [0.885, 1.020] |
| d=7, p=0.002 | GPU local pre-decoder (conservative, r=2) + pymatching | 126,613–173,927 | 0.482x [0.418, 0.583] |
| d=7, p=0.002 | GPU local pre-decoder (fast, r=1) + pymatching | 178,200–230,285 | 0.683x [0.676, 0.886] |
| d=7, p=0.004 | zero-syndrome shortcut + pymatching | 132,730–146,219 | 1.141x [0.945, 1.237] |
| d=7, p=0.004 | GPU local pre-decoder (conservative, r=2) + pymatching | 85,156–103,390 | 0.727x [0.622, 0.883] |
| d=7, p=0.004 | GPU local pre-decoder (fast, r=1) + pymatching | 112,910–133,021 | 0.964x [0.864, 1.179] |
| d=9, p=0.001 | zero-syndrome shortcut + pymatching | 171,144–269,768 | 0.959x [0.668, 1.073] |
| d=9, p=0.001 | GPU local pre-decoder (conservative, r=2) + pymatching | 52,088–61,661 | 0.233x [0.203, 0.271] |
| d=9, p=0.001 | GPU local pre-decoder (fast, r=1) + pymatching | 71,121–82,147 | 0.339x [0.278, 0.364] |
| d=9, p=0.002 | zero-syndrome shortcut + pymatching | 115,442–133,608 | 1.060x [0.892, 1.244] |
| d=9, p=0.002 | GPU local pre-decoder (conservative, r=2) + pymatching | 46,556–49,554 | 0.384x [0.365, 0.471] |
| d=9, p=0.002 | GPU local pre-decoder (fast, r=1) + pymatching | 67,582–69,448 | 0.565x [0.505, 0.681] |
| d=9, p=0.004 | zero-syndrome shortcut + pymatching | 51,103–60,720 | 0.970x [0.959, 1.046] |
| d=9, p=0.004 | GPU local pre-decoder (conservative, r=2) + pymatching | 30,729–33,082 | 0.543x [0.506, 0.632] |
| d=9, p=0.004 | GPU local pre-decoder (fast, r=1) + pymatching | 40,407–44,375 | 0.726x [0.719, 0.827] |
| d=13, p=0.001 | zero-syndrome shortcut + pymatching | 49,160–80,426 | 0.904x [0.891, 1.199] |
| d=13, p=0.001 | GPU local pre-decoder (conservative, r=2) + pymatching | 7,756–7,904 | 0.092x [0.087, 0.190] |
| d=13, p=0.001 | GPU local pre-decoder (fast, r=1) + pymatching | 10,491–10,604 | 0.123x [0.118, 0.257] |
| d=13, p=0.002 | zero-syndrome shortcut + pymatching | 39,112–40,126 | 0.969x [0.958, 1.059] |
| d=13, p=0.002 | GPU local pre-decoder (conservative, r=2) + pymatching | 7,137–7,191 | 0.177x [0.172, 0.192] |
| d=13, p=0.002 | GPU local pre-decoder (fast, r=1) + pymatching | 9,646–9,717 | 0.239x [0.230, 0.262] |
| d=13, p=0.004 | zero-syndrome shortcut + pymatching | 16,829–19,437 | 0.979x [0.897, 0.998] |
| d=13, p=0.004 | GPU local pre-decoder (conservative, r=2) + pymatching | 5,592–5,826 | 0.296x [0.292, 0.337] |
| d=13, p=0.004 | GPU local pre-decoder (fast, r=1) + pymatching | 7,165–7,305 | 0.380x [0.362, 0.430] |
| BB[[72,12,6]], p=0.02 | GPU min-sum BP + OSD fallback | 64,771–74,190 | 0.747x [0.712, 0.787] |
| BB[[72,12,6]], p=0.04 | GPU min-sum BP + OSD fallback | 25,271–35,299 | 0.691x [0.510, 0.717] |
| BB[[72,12,6]], p=0.06 | GPU min-sum BP + OSD fallback | 13,872–17,311 | 0.591x [0.542, 0.722] |
| BB[[144,12,12]], p=0.02 | GPU min-sum BP + OSD fallback | 45,378–53,864 | 0.875x [0.751, 0.978] |
| BB[[144,12,12]], p=0.04 | GPU min-sum BP + OSD fallback | 16,550–19,987 | 0.710x [0.652, 0.766] |
| BB[[144,12,12]], p=0.06 | GPU min-sum BP + OSD fallback | 6,334–6,904 | 0.649x [0.628, 0.692] |

## Environment (from `results/local-gpu-corrected-full/meta.json`)

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
| Benchmark date | 2026-10-03 01:55:30 |
