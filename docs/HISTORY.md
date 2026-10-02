# Benchmark history

Chronological record of the Colab T4 runs. `1.58x` etc. are throughput ratios (GPU shots/s ÷ CPU-baseline shots/s on the same shots; below 1 means the GPU was slower), see `docs/METHODOLOGY.md`. The README keeps only the current picture.


## Colab T4, run 1 (old code; notes in `docs/runs/colab-gpu-run1.md`)
* **Surface code:** the first GPU pre-decoder gave throughput ratios of at most 1.25x (d=5, p=0.002); from d=9 it resolved ~0 shots, with a ratio near 1; at some points it added logical errors (d=7, p=0.002: 3.75e-4 vs 2.5e-4).
* **qLDPC:** the first GPU BP implementation (dense tensors) had a throughput ratio of 0.07x-0.34x (about 3-14x slower than the C++ `ldpc` library).

## Colab T4, run 2: full profile, 200,000 shots per surface point, 20,000 per qLDPC point (`results/colab-gpu/`, code at commit `a381f7f`)
* **qLDPC:** after the BP rewrite (edge lists, converged shots dropped) the GPU/CPU throughput ratio was 0.98x-1.58x with equal logical-error counts (headline table above). Single-shot latency was about 1.6-6 ms on the GPU vs 11-40 µs for `ldpc`.
  The "BP converged" column in that run was wrong (read after the latency loop); fixed afterwards.
* **Surface code, first aggressive radius-1 rule:** throughput ratio 0.62x-1.19x. At d>=7 stage 1 cost much of what it saved (d=13, p=0.001: 3.60 s of stage-1 time while the PyMatching stage fell from 3.34 s to 1.33 s) and logical errors were higher
  (d=13, p=0.004: 161 vs 109; d=9, p=0.004: 586 vs 486). The MLP gate had a ratio below 1 at most points.

## NVIDIA Ising head-to-head, one T4 case with trained weights (`docs/runs/colab-ising-run1.md`)
At d=9, p=0.003 (20k shots, NVIDIA's circuit), the trained Ising model left 2.9 % of the syndrome weight and fully resolved 47 % of shots, and the PyMatching stage took 0.10 s vs 0.39 s for PyMatching alone,
with 26 vs 22 logical errors in the respective runs (no statistical comparison was made). Our first local rule left about 48 % of the weight (36 logical errors in its run).
The run's stage-1 time (15.6 s) included torch.compile warm-up and is not a valid speed measurement; the adapter has since been changed (warm-up outside the timed region, fixed 2,048-shot chunks).

## Colab T4, run 3: current code, quick profile (20k shots; `results/colab-gpu-quick/`, table in [docs/RESULTS_colab-gpu-quick.md](docs/RESULTS_colab-gpu-quick.md))

| rule | throughput ratio vs PyMatching | logical errors (ours / PyMatching) | stage 1 per 20k shots |
|---|---|---|---|
| safe, radius 2 (no extra errors observed in our tests) | 1.08x - 1.32x (4 of 4 points) | 27/27, 155/155, 5/5, 72/71 | 3-8 ms |
| fast, radius 1 | 0.90x - 2.53x (3 of 4 points above 1.02x) | 27/27, 158/155, 5/5, 80/71 | 3-7 ms |

* Stage 1 took about 2-4x less time per 20k shots than in run 2 (the rule also changed, so the ratios are not attributable to one cause).
* Caveats: only d=5 and 7; at 256-shot micro-batches the GPU surface-code pipeline was slower than PyMatching at 6 of 8 rows (1.2-3.2 ms vs 0.6-2.5 ms) and faster at d=7, p=0.002 (1.6-1.8 ms vs 2.0 ms); for qLDPC it was 49-74 ms vs 3-6 ms;
  the quick profile's 1,000-shot qLDPC runs are too small to load the GPU (ratio 0.18x-0.22x), so the 20,000-shot full run is the fair qLDPC comparison.
* The Ising and full-pipeline steps wrote only `meta.json` in this run: every case failed. The adapter now runs NVIDIA's pipeline in fixed 2,048-shot chunks and frees GPU memory between cases;
  the traceback of the failed run was not captured, so this is a hardening, not a confirmed diagnosis.

## What changed along the way
* **Accuracy fix found on CPU:** requiring that nothing else fired within two hops of an isolated pair (`radius=2`, now the default) removed the extra logical errors in our CPU tests
  (d=9, p=0.004: 449 vs 448 baseline errors; d=7, p=0.002: 52 vs 52, 200k shots each); the PyMatching stage then ran 1.1-1.3x faster instead of 1.4-1.7x. The aggressive rule is kept as `radius=1`.
* **Stage 1 on the GPU:** 1-byte uploads widened on the device, fp16 matrix products (the counts are small integers, representable in fp16), host-side bookkeeping outside the timed region; the GPU code path is unit-tested for equality with the CPU path.

