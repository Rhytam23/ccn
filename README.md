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
| **Local GPU pre-decoder** (clique-style: clears isolated fault pairs with sparse-matrix ops, exact-by-construction w.r.t. MWPM tie-break) | `local_predecoder.py` | tested on CPU; CUDA path written, **run it on Colab to confirm** |
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

## Honest results so far (local CPU, quick profile; GPU runs pending)

Measured (`results/local-cpu/`, 20k shots per point, circuit-level noise, Stim):

| d | p | PyMatching time on residual vs raw syndrome | syndrome weight left | shots fully resolved by stage 1 | logical errors (ours / PyMatching) |
|---|---|---|---|---|---|
| 5 | 0.002 | **2.2x faster** | 37 % | 49 % | 24 / 23 |
| 5 | 0.004 | 1.35x faster | 50 % | 21 % | 154 / 148 |
| 7 | 0.002 | 1.65x faster | 37 % | 18 % | 5 / 2 (low counts) |
| 7 | 0.004 | 1.39x faster | 54 % | 2 % | 98 / 85 |

* The stage-2 (global decoder) speed-up is **hardware independent** and real: pre-decoding hands MWPM a much sparser problem. LER stays within a few percent at d=5; the
  d=7, p=0.004 point shows about 15 % more logical errors (+13 events), so treat the pairing rule as a trade-off, not a free lunch, and check the confidence intervals in the CSV.
* On a **CPU**, the end-to-end pipeline is *slower* than PyMatching alone because stage 1 (run here in NumPy/SciPy on CPU) costs more than it saves; likewise batched BP is slower than the C++ `ldpc`.
  The end-to-end claim is that stage 1 becomes nearly free on a GPU; **that is untested until `results/colab-gpu/` exists**, and the website labels every run with its hardware.
* NVIDIA's Ising adapter is wired up and checked for plumbing (random weights); the trained-model comparison needs your Hugging Face token (see GPU runbook).
* The MLP gate is an honest negative result beyond d=5 (it resolves almost no shots at d=7), which is why a local pre-decoder is the primary design. It is kept as an ablation.
* qLDPC: batched BP converges on about 98 % of shots at p=0.03 on [[72,12,6]] (code-capacity noise), leaving only the rest for OSD.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes

* Noise: circuit-level (Stim `rotated_memory_z`) for the surface code; **code-capacity** bit-flip noise for the qLDPC codes (stated everywhere it is shown).
* NVIDIA Ising weights are under the NVIDIA Open Model License and are downloaded at run time, never committed.
* License: MIT for this repository's code.
