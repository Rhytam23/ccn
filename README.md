# qechybrid: GPU-accelerated, AI-enhanced hybrid QEC decoding

Real-time error correction is one of the most demanding classical workloads of a fault-tolerant quantum computer.
This project builds a **hierarchical decoder**: a cheap, massively parallel *pre-decoder* resolves the easy, local part of every syndrome
on the GPU, and only the hard residual reaches the exact global decoder.

```
Stim syndromes ──► [GPU] pre-decoder ──► empty residual? ──yes──► done (fast path)
 (circuit noise)    (local / learned)          │no
                                               ▼
                       surface code:  PyMatching (MWPM) on the *sparser* residual
                       qLDPC codes :  GPU batched BP ──► converged? ──yes──► done
                                                            │no
                                                            ▼
                                            BP+OSD (ldpc on CPU, or CUDA-Q QEC on GPU)
```

## What is in the repo

| Piece | File | Status |
|---|---|---|
| Circuit-level surface-code data + detector error model (Stim) | `src/qechybrid/data.py` | tested |
| **Local GPU pre-decoder** (clique-style heuristic: clears isolated fault pairs with matrix ops; NOT provably MWPM-equivalent) | `local_predecoder.py` | tested on CPU; CUDA path written, **run it on Colab to confirm** |
| **Learned gate** (small MLP with a *calibrated* confidence threshold so it adds ~no logical errors) | `gate.py` | tested |
| Hybrid pipeline + trivial non-AI baseline | `pipeline.py` | tested |
| **Batched min-sum BP in PyTorch** (CUDA/CPU) with BP+OSD fallback for non-converged shots | `bp_gpu.py`, `decoders.py` | tested on CPU (matches ldpc LER); CUDA path to confirm on Colab |
| Bivariate-bicycle qLDPC codes [[72,12,6]], [[144,12,12]], exact logical-failure check over GF(2) | `codes.py`, `gf2.py` | tested |
| Optional NVIDIA **CUDA-Q QEC** `nv-qldpc-decoder` backend | `cudaq_qec_adapter.py` | **unverified** (written from public docs; needs Colab GPU) |
| **Head-to-head with NVIDIA's Ising 3D-CNN pre-decoder** on identical shots (NVIDIA's circuit + noise model): PyMatching vs Ising+PyMatching vs ours | `ising_adapter.py`, `scripts/run_ising_bench.py`, `notebooks/03_ising_head_to_head.ipynb` | wiring follows NVIDIA's cookbook and is **verified locally with random weights** (plumbing/timing only); **trained-weights numbers pending**: weights are gated on Hugging Face and need your own token |
| Benchmark harness (LER with Wilson CIs, throughput, p50/p95/p99 single-shot latency) | `bench.py`, `scripts/run_benchmarks.py` | tested |
| Interactive results website (GitHub Pages ready) | `docs/index.html` (built by `scripts/make_report.py`) | working |

## Documentation

| Doc | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | pipeline, modules, design decisions, extension points (Ising CNN, CUDA-Q QEC) |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | noise models, metrics, baselines, fairness rules, limitations |
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

1. Push this repo to a **public** GitHub repository.
2. Open `notebooks/01_run_benchmarks_colab.ipynb` in Colab (set the runtime to a T4 GPU, set `GITHUB_REPO` in the first cell) and run all.
3. Download the zip, copy `results/colab-gpu/` into the repo, run `python scripts/make_report.py`, commit, and enable **GitHub Pages** (Settings > Pages > `main` / `docs`).
   The same page then shows CPU and GPU runs side by side via the run selector.
4. Optional: `02_cudaq_qec_bposd.ipynb` (NVIDIA GPU BP+OSD) and `03_ising_head_to_head.ipynb` (real NVIDIA Ising model vs ours; needs a free Hugging Face account, acceptance of the model terms, and a read token stored as the Colab secret `HF_TOKEN`).

## Honest results so far

### Colab T4, run 1 (old code; details in `results/colab-gpu-run1/NOTES.md`)
* **Surface code:** no meaningful end-to-end speed-up. Best case 1.25x (d=5, p=0.002); from d=9 the local pre-decoder resolves ~0 shots and speed equals PyMatching.
  At some points it adds logical errors (d=7, p=0.002: 3.75e-4 vs 2.5e-4).
* **qLDPC:** the first GPU BP implementation (dense tensors) was 5-15x *slower* than the C++ `ldpc` library.

### Colab T4, run 2 (current code, `--profile quick`; data in `results/colab-gpu/`)
Surface code, 20,000 shots per point, GPU local pre-decoder + PyMatching vs PyMatching alone (same Colab session, 2 vCPU):

| d | p | end-to-end | stage 1 (GPU) | PyMatching on residual vs raw | logical errors (ours / PyMatching) |
|---|---|---|---|---|---|
| 5 | 0.002 | 1.22x | 0.01 s | 0.02 s vs 0.04 s | 27 / 27 |
| 5 | 0.004 | 1.24x | 0.01 s | 0.03 s vs 0.05 s | 158 / 155 |
| 7 | 0.002 | 1.14x | 0.02 s | 0.04 s vs 0.08 s | 5 / 5 |
| 7 | 0.004 | 1.62x | 0.03 s | 0.14 s vs 0.28 s | 80 / 71 |

* **First real GPU result:** a modest 1.1-1.6x end-to-end speed-up, with logical error counts equal within statistical noise except d=7, p=0.004 (80 vs 71). This is the quick profile and small distances;
  the `full` profile (d up to 13, 200k shots) has not been run on the new code.
* **qLDPC (1,000 shots, quick profile):** the rewritten GPU BP is now about 20k shots/s vs 84k shots/s for the C++ `ldpc` on [[72,12,6]], p=0.02 (it was 9k in run 1), identical LER. 1,000 shots is too small to
  load a GPU, so this is not yet a fair comparison: rerun with `PROFILE = "full"` (20,000 shots) before drawing conclusions. No qLDPC speed-up has been shown.
* CUDA-Q QEC was not installed and the Ising comparison has not run (no Hugging Face token yet).

### What changed since run 1, and what it means
* **BP rewritten** (edge lists + dropping converged shots from the working batch): identical output to the dense version (unit-tested) and 20-160x faster on the same CPU
  (4000 shots: 9.8 s -> 0.24 s on [[72,12,6]], 40.9 s -> 0.25 s on [[144,12,12]], p=0.02). **Not yet re-measured on a GPU against `ldpc`: rerun notebook 01.**
* **The local surface-code pre-decoder is a weak, approximate pre-decoder:** it clears 40-60 % of syndrome weight, which makes PyMatching on the residual 1.4-2x faster, but it costs
  4-25 % extra logical errors (e.g. 37 vs 21 at d=7, p=0.002) and extra passes or exact shortest-path validation do not change that. This is the classical baseline that motivates a *learned* pre-decoder.
* **The comparison that matters** is NVIDIA's trained Ising CNN vs this baseline vs PyMatching on identical shots (`notebooks/03_ising_head_to_head.ipynb`, needs your Hugging Face token). It has not been run with trained weights.
* The MLP gate is a negative result beyond d=5 and is kept only as an ablation.

Quote only what a `results/colab-gpu*` run with CSV + `meta.json` shows, with its hardware, profile and logical error counts.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes

* Noise: circuit-level (Stim `rotated_memory_z`) for the surface code; **code-capacity** bit-flip noise for the qLDPC codes (stated everywhere it is shown).
* NVIDIA Ising weights are under the NVIDIA Open Model License and are downloaded at run time, never committed.
* License: MIT for this repository's code.
