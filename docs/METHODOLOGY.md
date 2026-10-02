# Benchmark methodology

Read this before quoting any number from the repo.

## Definitions (used everywhere)
* **Throughput ratio** = GPU-pipeline shots/s &divide; CPU-baseline shots/s on the same shots in the same session. &gt;1 means the GPU pipeline is faster, &lt;1 means it is slower. It is *not* a latency.
* **Latency ratio** (if quoted) = CPU latency &divide; GPU latency for the same batch size.
* **End-to-end** = stage 1 (pre-decoder) + stage 2 (global decoder), transfers included, GPU-synchronised.
* **Identical logical-error counts** on the same shots mean the two decoders made the same number of mistakes there; this is evidence, not a proof of statistical equivalence. Wilson 95 % intervals are in the CSVs and the detailed tables.

## Noise models
* **Surface code:** Stim `surface_code:rotated_memory_z`, `rounds = d`, uniform circuit-level depolarising/flip noise `p` on gates, resets, measurements and idle data qubits.
* **qLDPC:** bivariate-bicycle codes [[72,12,6]] and [[144,12,12]] under **code-capacity** i.i.d. bit-flip noise `p` (perfect syndrome measurement). This is a simplification and is labelled as such in every table.

## Data splits
Train / validation / test use different seeds (1 / 2 / 3). The learned gate is trained on train, its threshold calibrated on validation, and all reported numbers come from test.

## Metrics
| Metric | Definition |
|---|---|
| LER | logical failures / shots, with a 95 % Wilson interval (`ler_lo`, `ler_hi`) |
| Throughput | test shots / wall-clock seconds of one batched decode, best of 2 runs after a warm-up |
| Latency | per-shot wall-clock for batch size 1, p50/p95/p99 over `lat` shots after warm-up |
| `accept_rate` | surface: fraction of shots fully resolved before the global decoder; qLDPC: fraction where GPU BP converged |
| `syndrome_weight_kept` | residual syndrome weight / original weight after pre-decoding |

## Baselines
* Surface: PyMatching (CPU) on the raw syndrome; plus a trivial non-AI shortcut (empty syndrome => no flip) so AI/GPU gains are not overstated.
* qLDPC: `ldpc.BpOsdDecoder` (C++), min-sum, 50 iterations, OSD-CS order 7, scaling 0.8.

## Fairness rules
1. CPU and GPU numbers are compared **on the same machine/session** (Colab: 2 vCPU + T4). Do not mix numbers across machines.
2. The GPU, driver, library versions and CPU count are saved in `results/<tag>/meta.json` and shown on the website.
3. Batch-1 latency is reported separately from batched throughput. A GPU is expected to lose at batch size 1 for small codes; report where the crossover is.
4. LER parity is a precondition: a faster decoder with a worse LER is shown as a trade-off, not a win.
5. Statistics: low-LER points (d>=9) need many shots; if `errors` < ~30 treat the point as indicative only.

## Known limitations
* CPU PyMatching is extremely fast at small distance; expect GPU benefit to appear at larger d and larger batches.
* Code-capacity qLDPC results do not transfer to circuit-level noise.
* The CUDA-Q QEC adapter has not been verified against a real install. The Ising adapter is verified for plumbing with random weights only; trained-weight results need a user-supplied Hugging Face token.
* The Ising comparison uses NVIDIA's circuit (25-parameter noise, boundary detectors, basis X), which differs from the plain Stim circuit used elsewhere, so those rows are kept in a separate table and never mixed.
* Dense BP memory scales as B x m x n.
