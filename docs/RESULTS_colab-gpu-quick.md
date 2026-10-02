# GPU speed-up results (Tesla T4)

**Measured on Tesla T4 (Google Colab, 2 vCPU), `quick` profile, 20,000 surface-code shots and 1,000 qLDPC shots per point; data: `results/colab-gpu-quick/`.**

* **qLDPC (BB codes, code-capacity noise): 0.18x to 0.22x batch throughput** vs the C++ `ldpc` BP+OSD (best: BB[[72,12,6]], p=0.02, 0.22x), with **identical logical error counts at every point**.
* **Surface code (circuit-level noise), fast r=1 rule: 0.90x to 2.53x end-to-end** (3 of 4 points above 1.02x; 12 extra logical errors in total vs PyMatching).
* **Surface code (circuit-level noise), safe r=2 rule: 1.08x to 1.32x end-to-end** (4 of 4 points above 1.02x; 1 extra logical errors in total vs PyMatching).
* Best single surface-code point: fast r=1 rule, d=7, p=0.002, 2.53x.

## qLDPC: GPU batched BP vs C++ ldpc

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | speed-up | logical errors (GPU / ldpc) |
|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 83,117 | 18,542 | **0.22x** | 10 / 10 |
| BB[[72,12,6]] | 0.04 | 45,441 | 8,039 | **0.18x** | 88 / 88 |

## Surface code: GPU local pre-decoder (safe radius-2 and fast radius-1 rules, fp16 stage 1) + PyMatching vs PyMatching alone

| d | p | rule | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | end-to-end | PyMatching stage only | logical errors (ours / PyMatching) | syndrome weight left |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.002 | safe r=2 | 887,497 | 1,045,061 | **1.18x** | 1.39x | 27 / 27 | 55 % |
| 5 | 0.002 | fast r=1 | 887,497 | 1,292,647 | **1.46x** | 1.77x | 27 / 27 | 37 % |
| 5 | 0.004 | safe r=2 | 401,090 | 434,188 | **1.08x** | 1.16x | 155 / 155 | 73 % |
| 5 | 0.004 | fast r=1 | 401,090 | 359,338 | **0.90x** | 0.95x | 158 / 155 | 51 % |
| 7 | 0.002 | safe r=2 | 161,357 | 212,478 | **1.32x** | 1.45x | 5 / 5 | 63 % |
| 7 | 0.002 | fast r=1 | 161,357 | 407,567 | **2.53x** | 2.95x | 5 / 5 | 37 % |
| 7 | 0.004 | safe r=2 | 106,940 | 123,290 | **1.15x** | 1.21x | 72 / 71 | 83 % |
| 7 | 0.004 | fast r=1 | 106,940 | 158,455 | **1.48x** | 1.57x | 80 / 71 | 53 % |

Caveats: batch throughput only (the GPU is slower than the CPU at batch size 1 and at 256-shot micro-batches in these runs); only d=5 and d=7 with 20,000 shots per point, so error counts are small (a +/-few difference is noise); the qLDPC rows use only 1,000 shots, too few to load a GPU, so the GPU BP looks slow here: the 20,000-shot `full` run (`results/colab-gpu/`) is the fair qLDPC comparison. The `safe` rule is the lossless one; the `fast` rule trades some logical errors for speed. Speed-ups are relative to CPU baselines on the same Colab machine.
