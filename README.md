# qechybrid: GPU-accelerated hybrid quantum error-correction decoding

Real-time error correction creates demanding classical workloads for fault-tolerant quantum computers
(see the motivation in NVIDIA's [Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding) cookbook and [paper](https://arxiv.org/abs/2604.12841)).
This repository is a **measured, reproducible benchmark** of GPU-batched decoders on a free Google Colab T4, with an **experimental AI-pre-decoder track**
built around NVIDIA's trained Ising model. It reports where the GPU wins, where it loses, and which parts are not yet measured.

## Pipelines

Surface code (circuit-level noise, Stim). A pre-decoder removes the easy, local part of the syndrome; the exact decoder handles what is left:

```
syndrome ──► pre-decoder (GPU) ──► residual syndrome ──► PyMatching (CPU, MWPM) ──► logical flip
              our local rule  or   (empty residual = done)
              NVIDIA Ising CNN
```

qLDPC codes (bivariate-bicycle codes, code-capacity noise). A batched GPU decoder handles the shots it can; the rest go to OSD:

```
syndrome ──► batched min-sum BP (GPU) ──► converged? ──yes──► estimate
                                           │ no
                                           └──► BP+OSD on those shots only (CPU `ldpc`; CUDA-Q QEC backend unverified)
```

**Experimental full pipeline** (results pending): the same two ideas combined on the surface code,

```
syndrome ──► NVIDIA Ising (GPU) ──► residual syndrome ──► batched BP (GPU) ──► [OSD fallback, CPU] ──► logical flip XOR Ising's partial flip
```

*Why BP is valid on the surface code here:* Stim turns the noisy circuit into a **detector error model (DEM)**, i.e. a binary parity-check matrix **H** (rows = detectors, columns = fault mechanisms)
and an observable matrix **L** (rows = logical observables). Any decoder that accepts (H, syndrome) can decode it: BP+OSD estimates which faults fired (ê) and the logical outcome is **L·ê mod 2**.
The Ising model outputs a partial logical correction and a *residual* syndrome (the syndrome with the corrections it predicted removed); the residual is still a valid syndrome of the remaining faults under the same DEM,
so BP (using the undecomposed DEM) or PyMatching (using the decomposed, graph-like DEM) can finish the job. Final answer = Ising's partial flip XOR the global decoder's flip.
PyMatching stays the main surface-code baseline because it is the fastest practical CPU decoder; BP+OSD is run on the residual to test the GPU-decoder idea and to compare like with like against CPU BP+OSD.

## How to read the numbers

* **Throughput ratio** = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. **Above 1: the GPU pipeline is faster. Below 1: it is slower.** It is not a latency.
* **Latency ratio** (where quoted) = CPU latency ÷ GPU latency for the same batch size.
* **End-to-end** = pre-decoder + global decoder, transfers included, GPU-synchronised.
* **Logical errors (ours / baseline)** are counts on the same shots; identical counts are evidence of similar decoding quality, not a proof. Wilson 95 % intervals are in the CSVs and in [docs/RESULTS.md](docs/RESULTS.md).

## Glossary

* **PyMatching / MWPM:** minimum-weight perfect matching on a graph of detectors; the standard fast surface-code decoder.
* **BP (belief propagation):** iterative message passing on the parity-check graph; fast and parallel, but it can fail to converge on short cycles.
* **OSD (ordered-statistics decoding):** a linear-algebra post-processing step that rescues shots where BP does not converge.
* **Ising pre-decoder:** NVIDIA's learned 3D-CNN that removes easy local errors from the syndrome before a global decoder runs ([repo](https://github.com/NVIDIA/Ising-Decoding)).
* **Stim / DEM:** circuit simulator and the detector error model it derives (see above).
* **qLDPC / bivariate-bicycle (BB) code:** a high-rate quantum code family; we use [[72,12,6]] and [[144,12,12]].
* **Code-capacity vs circuit-level noise:** code-capacity assumes perfect syndrome measurement (simpler, used for the qLDPC codes); circuit-level models noisy gates and measurements (used for the surface code).

<!-- RESULTS:START -->
### GPU benchmark results (Tesla T4)

**Measured on Tesla T4 (Google Colab, 2 vCPU), `full` profile, 200,000 surface-code shots and 20,000 qLDPC shots per point; data: `results/colab-gpu/`.**

*Throughput ratio = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. A value above 1 means the GPU pipeline is faster, below 1 means it is slower. It is not a latency.*

* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio 0.98x to 1.58x** vs the C++ `ldpc` BP+OSD (best: BB[[144,12,12]], p=0.02, 1.58x; the GPU is slightly slower at the points below 1.00x). Logical-error counts are identical on the same shots at every point (evidence of the same decoding quality, not a proof; intervals in `docs/RESULTS.md`).
  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*
* **Surface code (circuit-level noise), first version (r=1) rule: throughput ratio 0.62x to 1.19x** (4 of 12 points above 1.02x; 344 more logical errors than PyMatching, summed over the points where ours was worse).
* Best single surface-code point: first version (r=1) rule, d=5, p=0.002, ratio 1.19x.

| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) |
|---|---|---|---|---|---|
| BB[[72,12,6]] | 0.02 | 77,638 | 86,407 | **1.11x** | 226 / 226 |
| BB[[72,12,6]] | 0.04 | 39,769 | 38,941 | **0.98x** | 1770 / 1770 |
| BB[[72,12,6]] | 0.06 | 18,921 | 21,240 | **1.12x** | 4991 / 4991 |
| BB[[144,12,12]] | 0.02 | 39,426 | 62,340 | **1.58x** | 13 / 13 |
| BB[[144,12,12]] | 0.04 | 19,885 | 24,982 | **1.26x** | 233 / 233 |
| BB[[144,12,12]] | 0.06 | 7,627 | 7,853 | **1.03x** | 1772 / 1772 |

Caveats: batch throughput only (single-shot latency is worse on the GPU than on the CPU); the surface-code rows are for the first, aggressive local rule, which also adds logical errors at some points (compare the error columns); the fp16 stage 1 and the lossless `radius=2` rule were not part of this run. Ratios are relative to CPU baselines on the same Colab machine.

Confidence intervals, surface-code tables and the full environment record: [docs/RESULTS.md](docs/RESULTS.md).
<!-- RESULTS:END -->

## Status at a glance

| Component | Status |
|---|---|
| Batched GPU BP + OSD fallback (qLDPC) | **Measured on a T4** (full profile, 20k shots/point): see the table above |
| Local surface-code pre-decoder, first aggressive radius-1 rule | **Measured on a T4** (full profile, 200k shots): throughput ratio 0.62x-1.19x and extra logical errors at some points |
| Local pre-decoder, lossless radius-2 rule and fast radius-1 rule with the fp16 stage 1 | **Measured on a T4, quick profile only** (d=5 and 7, 20k shots) |
| Learned MLP gate | **Measured on a T4**: negative result (resolves at most ~2 % of shots at d>=7); kept as an ablation |
| NVIDIA Ising trained model | **One trained-weight T4 case** (d=9, p=0.003); its timing was invalid (compile warm-up inside the timed region); later runs produced no results |
| CUDA-Q QEC `nv-qldpc-decoder` | **Not verified**: `cudaq-qec` was not installed in any run (the decoder is a closed-source library, see the [CUDA-Q QEC docs](https://nvidia.github.io/cudaq-qec/)) |
| Full pipeline: Ising (GPU) -> GPU BP+OSD vs CPU baselines | **Not yet measured**: plumbing checked on CPU with random weights only |

## What is new here, what is not

**Our contribution is the combination and the measurement**: a GPU-batched BP implementation, a GPU surface-code pre-decoder with an accuracy/throughput dial, and a reproducible, identical-shot comparison framework
around NVIDIA's Ising pre-decoder, with explicit throughput, latency and logical-error measurements, including the negative results.

**Not ours (used as-is, credited):** NVIDIA's [Ising pre-decoder and weights](https://github.com/NVIDIA/Ising-Decoding) (NVIDIA Open Model License; code Apache-2.0),
[CUDA-Q QEC](https://github.com/NVIDIA/cudaqx) ([docs](https://nvidia.github.io/cudaq-qec/)), [Stim](https://github.com/quantumlib/Stim), [PyMatching](https://github.com/oscarhiggott/PyMatching),
[`ldpc`](https://github.com/quantumgizmos/ldpc), PyTorch.

**Ours:**
* a batched, edge-list **GPU belief-propagation decoder** that drops converged shots from the working batch (unit-tested to match a dense reference bit for bit; equal logical-error counts to `ldpc` on a T4),
* a **local GPU pre-decoder for the surface code** with a measured accuracy/throughput dial (`radius=2` lossless in our tests, `radius=1` faster but lossy); in the one measured case (d=9, p=0.003, NVIDIA's circuit) it leaves substantially more syndrome weight than the Ising model (48 % vs 2.9 %),
* the **benchmark harness** (same shots, same session, Wilson intervals, stage timings, single-shot and 256-shot micro-batch latency) and the **head-to-head wiring around NVIDIA's pipeline**,
* the reproducible Colab notebook and results site.

**What we claim (and only this):**
1. GPU BP gives equal logical-error counts to `ldpc` on the same shots and a GPU/CPU batch-throughput ratio of 0.98x-1.58x on a T4 (qLDPC, code-capacity noise; best case [[144,12,12]], p=0.02).
2. On a T4 (run 3, quick profile, d=5 and 7, 20k shots) the lossless `radius=2` rule has a throughput ratio of 1.08x-1.32x with the same logical-error counts as PyMatching in that run; the fast `radius=1` rule reaches 2.53x (d=7, p=0.002) with 80 vs 71 errors at d=7, p=0.004.
   This is a small quick-profile result; the earlier full run used the first aggressive rule and a slower stage 1 (0.62x-1.19x).
3. In one case (d=9, p=0.003) NVIDIA's trained Ising model left 2.9 % of the syndrome weight and resolved 47 % of shots completely. Its end-to-end timing has not been validly measured.

**What we do not claim:** a real-time (single-shot, microsecond) GPU advantage (in run 3 the GPU surface-code path took 630-880 µs per single shot vs 18-38 µs for PyMatching, and was slower at 6 of 8 points at 256-shot micro-batches); any Ising or full-pipeline throughput result
(`results/colab-ising/` and `results/colab-pipeline/` do not exist yet); equivalence of the local rule to MWPM (it is a heuristic).

### The full-pipeline experiment (`scripts/run_full_pipeline.py`, run by notebook 04): results pending
On identical shots (NVIDIA's circuit, trained Ising weights) it measures total wall-clock time and logical errors for CPU PyMatching, CPU BP+OSD, Ising + PyMatching, Ising + CPU BP+OSD,
and the full pipeline Ising (GPU) -> GPU BP with CPU OSD fallback. Two CPU baselines are reported because "the CPU baseline" is ambiguous: CPU BP+OSD (same algorithm family, the like-for-like test of the GPU decoder)
and PyMatching (the fastest practical CPU decoder, the test of practical usefulness). The experiment will determine whether the full pipeline improves on CPU BP+OSD and whether it can compete with PyMatching.
In CPU plumbing runs (random weights, so the logical-error numbers are meaningless) CPU BP+OSD took about 13-33 ms per shot and PyMatching about 26-63 µs per shot at d=9.

## What is in the repo

| Piece | File | Tests |
|---|---|---|
| Circuit-level surface-code data + detector error model (Stim) | `src/qechybrid/data.py` | unit-tested |
| Local GPU pre-decoder (clique-style heuristic, not provably MWPM-equivalent) | `local_predecoder.py` | unit-tested; CPU/GPU code paths checked for equality |
| Learned MLP gate | `gate.py` | unit-tested |
| Hybrid surface-code pipeline + trivial non-AI baseline | `pipeline.py` | unit-tested |
| Batched min-sum BP (CUDA/CPU) with BP+OSD fallback; circuit-level `DemBpOsd` | `bp_gpu.py`, `decoders.py` | unit-tested against a dense reference |
| Bivariate-bicycle qLDPC codes, exact logical-failure check over GF(2) | `codes.py`, `gf2.py` | unit-tested |
| Optional CUDA-Q QEC `nv-qldpc-decoder` backend | `cudaq_qec_adapter.py` | not tested (library not installed) |
| NVIDIA Ising head-to-head and full pipeline | `ising_adapter.py`, `scripts/run_ising_bench.py`, `scripts/run_full_pipeline.py` | plumbing test with random weights |
| Benchmark harness (LER with Wilson CIs, throughput, p50/p95/p99 single-shot latency, 256-shot micro-batch latency, environment record) | `bench.py`, `scripts/run_benchmarks.py` | unit-tested |
| Results website (GitHub Pages ready) | `docs/index.html` (built by `scripts/make_report.py`) | element-id test |

### Results layout
`results/<tag>/` holds data only: `colab-gpu` (T4, full profile), `colab-gpu-quick` (T4, current code, quick profile), `local-cpu` (CPU), and, when they exist, `colab-ising` and `colab-pipeline`.
Run notes live in `docs/runs/` (`colab-gpu-run1.md`, `colab-ising-run1.md`). From now on every `meta.json` also records the git commit, `ldpc` version, CUDA version and GPU memory;
older runs lack some of these and `docs/RESULTS.md` says "not recorded".

## Documentation

| Doc | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md), [docs/RESULTS_colab-gpu-quick.md](docs/RESULTS_colab-gpu-quick.md) | full generated tables with confidence intervals and the environment record |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | pipeline, modules, design decisions, extension points |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | definitions, noise models, metrics, baselines, fairness rules, limitations |
| [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md) | running on Colab, troubleshooting, profiling |
| [docs/PUBLISHING.md](docs/PUBLISHING.md) | public repo + GitHub Pages checklist |
| [docs/PITCH.md](docs/PITCH.md) | 3-minute pitch outline |
| [CONTRIBUTING.md](CONTRIBUTING.md) | dev workflow; `make test / quick / full / report` also work |

## Quick start (local, CPU, virtual environment)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1     # Windows: creates .venv, installs, runs tests
# or: bash scripts/setup.sh                                    # Linux / macOS / WSL
.\.venv\Scripts\python scripts/run_benchmarks.py --profile quick --device cpu --tag local-cpu
.\.venv\Scripts\python scripts/make_report.py                  # -> docs/index.html
```

## GPU run (free: Google Colab T4)

**Use one notebook: `notebooks/04_run_everything_colab.ipynb`.** It runs everything below in order with *Runtime > Run all*.

| notebook | purpose |
|---|---|
| **`04_run_everything_colab`** | **setup check, tests, surface + qLDPC benchmarks, Ising head-to-head, full pipeline, report + zip (use this one)** |
| `00_setup_check_colab` | subset: GPU/install/tests only |
| `01_run_benchmarks_colab` | subset: surface + qLDPC benchmarks only |
| `02_cudaq_qec_bposd` | subset: optional CUDA-Q QEC backend (unverified) |
| `03_ising_head_to_head` | subset: Ising comparison only (does not include the full pipeline) |

1. Push this repo to a **public** GitHub repository and open notebook 04 from GitHub in Colab (runtime: an NVIDIA GPU; set `GITHUB_REPO` and `PROFILE` in the first cell).
2. For the Ising and full-pipeline steps you need a free Hugging Face account, acceptance of the model terms, and a *read* token stored as the Colab secret `HF_TOKEN` (never paste it into a cell). Without it those steps are skipped.
3. Download the zip, copy the `results/colab-*` folders into the repo, run `python scripts/make_report.py` and `python scripts/make_summary.py` (and `... colab-gpu-quick`), commit, and enable **GitHub Pages** (Settings > Pages > `main` / `docs`).

## Results history

### Colab T4, run 1 (old code; notes in `docs/runs/colab-gpu-run1.md`)
* **Surface code:** the first GPU pre-decoder gave throughput ratios of at most 1.25x (d=5, p=0.002); from d=9 it resolved ~0 shots, with a ratio near 1; at some points it added logical errors (d=7, p=0.002: 3.75e-4 vs 2.5e-4).
* **qLDPC:** the first GPU BP implementation (dense tensors) had a throughput ratio of 0.07x-0.34x (about 3-14x slower than the C++ `ldpc` library).

### Colab T4, run 2: full profile, 200,000 shots per surface point, 20,000 per qLDPC point (`results/colab-gpu/`, code at commit `a381f7f`)
* **qLDPC:** after the BP rewrite (edge lists, converged shots dropped) the GPU/CPU throughput ratio was 0.98x-1.58x with equal logical-error counts (headline table above). Single-shot latency was about 1.6-6 ms on the GPU vs 11-40 µs for `ldpc`.
  The "BP converged" column in that run was wrong (read after the latency loop); fixed afterwards.
* **Surface code, first aggressive radius-1 rule:** throughput ratio 0.62x-1.19x. At d>=7 stage 1 cost much of what it saved (d=13, p=0.001: 3.60 s of stage-1 time while the PyMatching stage fell from 3.34 s to 1.33 s) and logical errors were higher
  (d=13, p=0.004: 161 vs 109; d=9, p=0.004: 586 vs 486). The MLP gate had a ratio below 1 at most points.

### NVIDIA Ising head-to-head, one T4 case with trained weights (`docs/runs/colab-ising-run1.md`)
At d=9, p=0.003 (20k shots, NVIDIA's circuit), the trained Ising model left 2.9 % of the syndrome weight and fully resolved 47 % of shots, and the PyMatching stage took 0.10 s vs 0.39 s for PyMatching alone,
with 26 vs 22 logical errors in the respective runs (no statistical comparison was made). Our first local rule left about 48 % of the weight (36 logical errors in its run).
The run's stage-1 time (15.6 s) included torch.compile warm-up and is not a valid speed measurement; the adapter has since been changed (warm-up outside the timed region, fixed 2,048-shot chunks).

### Colab T4, run 3: current code, quick profile (20k shots; `results/colab-gpu-quick/`, table in [docs/RESULTS_colab-gpu-quick.md](docs/RESULTS_colab-gpu-quick.md))

| rule | throughput ratio vs PyMatching | logical errors (ours / PyMatching) | stage 1 per 20k shots |
|---|---|---|---|
| safe, radius 2 (lossless in our tests) | 1.08x - 1.32x (4 of 4 points) | 27/27, 155/155, 5/5, 72/71 | 3-8 ms |
| fast, radius 1 | 0.90x - 2.53x (3 of 4 points above 1.02x) | 27/27, 158/155, 5/5, 80/71 | 3-7 ms |

* Stage 1 took about 2-4x less time per 20k shots than in run 2 (the rule also changed, so the ratios are not attributable to one cause).
* Caveats: only d=5 and 7; at 256-shot micro-batches the GPU surface-code pipeline was slower than PyMatching at 6 of 8 rows (1.2-3.2 ms vs 0.6-2.5 ms) and faster at d=7, p=0.002 (1.6-1.8 ms vs 2.0 ms); for qLDPC it was 49-74 ms vs 3-6 ms;
  the quick profile's 1,000-shot qLDPC runs are too small to load the GPU (ratio 0.18x-0.22x), so the 20,000-shot full run is the fair qLDPC comparison.
* The Ising and full-pipeline steps wrote only `meta.json` in this run: every case failed. The adapter now runs NVIDIA's pipeline in fixed 2,048-shot chunks and frees GPU memory between cases;
  the traceback of the failed run was not captured, so this is a hardening, not a confirmed diagnosis.

### What changed along the way
* **Accuracy fix found on CPU:** requiring that nothing else fired within two hops of an isolated pair (`radius=2`, now the default) removed the extra logical errors in our CPU tests
  (d=9, p=0.004: 449 vs 448 baseline errors; d=7, p=0.002: 52 vs 52, 200k shots each); the PyMatching stage then ran 1.1-1.3x faster instead of 1.4-1.7x. The aggressive rule is kept as `radius=1`.
* **Stage 1 on the GPU:** 1-byte uploads widened on the device, fp16 matrix products (exact for these small integer counts), host-side bookkeeping outside the timed region; the GPU code path is unit-tested for equality with the CPU path.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes

* Noise: circuit-level (Stim `rotated_memory_z`) for the surface code; code-capacity bit-flip noise for the qLDPC codes (stated wherever results are shown). The Ising comparison uses NVIDIA's own circuit and noise model and is kept in separate tables.
* NVIDIA Ising weights are under the NVIDIA Open Model License and are downloaded at run time, never committed.
* License: MIT for this repository's code.
