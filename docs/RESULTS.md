# GPU speed-up results (Tesla T4)

**Measured on Tesla T4 (Google Colab, 2 vCPU), `full` profile, 200,000 surface-code shots and 20,000 qLDPC shots per point; data: `results/colab-gpu/`.**

* **qLDPC (BB codes, code-capacity noise): 0.98x to 1.58x batch throughput** vs the C++ `ldpc` BP+OSD (best: BB[[144,12,12]], p=0.02, 1.58x), with **identical logical error counts at every point**.
* **Surface code (circuit-level noise): 0.62x to 1.19x end-to-end** (4 of 12 points above 1.02x; best: d=5, p=0.002, 1.19x). The GPU pre-decoder makes the PyMatching stage itself 1.3x to 3.2x faster, but at d>=7 the first-stage cost cancels most of that gain.

## qLDPC: GPU batched BP vs C++ ldpc

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | speed-up | logical errors (GPU / ldpc) |
|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 77,638 | 86,407 | **1.11x** | 226 / 226 |
| BB[[72,12,6]] | 0.04 | 39,769 | 38,941 | **0.98x** | 1770 / 1770 |
| BB[[72,12,6]] | 0.06 | 18,921 | 21,240 | **1.12x** | 4991 / 4991 |
| BB[[144,12,12]] | 0.02 | 39,426 | 62,340 | **1.58x** | 13 / 13 |
| BB[[144,12,12]] | 0.04 | 19,885 | 24,982 | **1.26x** | 233 / 233 |
| BB[[144,12,12]] | 0.06 | 7,627 | 7,853 | **1.03x** | 1772 / 1772 |

## Surface code: GPU local pre-decoder + PyMatching vs PyMatching alone

| d | p | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | end-to-end | PyMatching stage only | logical errors (ours / PyMatching) | syndrome weight left |
|---|---|---|---|---|---|---|---|
| 5 | 0.001 | 1,407,668 | 1,411,246 | **1.00x** | 3.20x | 23 / 23 | 29 % |
| 5 | 0.002 | 748,339 | 892,596 | **1.19x** | 2.16x | 196 / 182 | 37 % |
| 5 | 0.004 | 375,909 | 445,623 | **1.19x** | 1.54x | 1550 / 1532 | 50 % |
| 7 | 0.001 | 449,641 | 366,068 | **0.81x** | 3.11x | 10 / 5 | 26 % |
| 7 | 0.002 | 241,949 | 244,920 | **1.01x** | 2.03x | 75 / 50 | 37 % |
| 7 | 0.004 | 112,760 | 119,404 | **1.06x** | 1.41x | 970 / 859 | 54 % |
| 9 | 0.001 | 197,373 | 149,050 | **0.76x** | 2.67x | 0 / 0 | 25 % |
| 9 | 0.002 | 105,566 | 101,249 | **0.96x** | 1.86x | 30 / 13 | 37 % |
| 9 | 0.004 | 47,091 | 48,560 | **1.03x** | 1.38x | 586 / 486 | 56 % |
| 13 | 0.001 | 59,860 | 36,976 | **0.62x** | 2.52x | 0 / 0 | 24 % |
| 13 | 0.002 | 32,605 | 26,710 | **0.82x** | 1.85x | 2 / 0 | 38 % |
| 13 | 0.004 | 13,293 | 12,726 | **0.96x** | 1.32x | 161 / 109 | 58 % |

Caveats: batch throughput only (single-shot latency is worse on the GPU than on the CPU); the surface-code rows are for the first, aggressive local rule, which also adds logical errors at some points (compare the error columns); the faster fp16 stage 1 and the lossless `radius=2` rule are CPU-verified and not yet re-measured on the GPU. Speed-ups are relative to CPU baselines on the same Colab machine.
