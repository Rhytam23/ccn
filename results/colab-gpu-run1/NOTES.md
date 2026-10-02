# Colab T4 run 1 (printed log supplied by the team)

Hardware: Tesla T4 (Colab). Code version: first pushed commit (dense BP, sparse-CSR local pre-decoder, `--profile full`).
Only the console output was kept, so this run has no CSV/meta and is not shown on the website.

Observed (shots/s, T4 + Colab CPU):

| case | PyMatching (CPU) | GPU local pre-decoder + PyMatching | note |
|---|---|---|---|
| d=5 p=0.002 | 430,752 | 538,511 | 1.25x, LER 9.1e-4 -> 9.8e-4 |
| d=5 p=0.004 | 366,129 | 424,905 | 1.16x |
| d=7 p=0.002 | 240,586 | 223,059 | slower, LER 2.5e-4 -> 3.75e-4 |
| d=9 p=0.004 | 45,814 | 46,248 | ~equal, accept rate 0 |
| d=13 p=0.004 | 13,344 | 13,297 | equal, accept rate 0 |

| qLDPC case | ldpc BP+OSD (CPU) | GPU min-sum BP + OSD fallback |
|---|---|---|
| [[72,12,6]] p=0.02 | 79,257 | 9,387 |
| [[144,12,12]] p=0.02 | 39,427 | 2,745 |
| [[144,12,12]] p=0.06 | 7,290 | 2,191 |

Conclusions: no end-to-end surface-code speed-up from the local pre-decoder (it resolves ~0 shots at d>=9 and costs
extra logical errors); the dense GPU BP was 5-15x slower than C++ ldpc. Follow-up: BP rewritten with edge lists and
active-set shrinking (20-160x faster than the dense version on CPU, identical output; see tests); needs a new Colab run.
