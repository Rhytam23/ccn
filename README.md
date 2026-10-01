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
| Reproduction of NVIDIA's **Ising** 3D-CNN pre-decoder reference | `notebooks/03_ising_reproduce.ipynb` | follows NVIDIA's README; **unverified** |
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
4. Optional: `02_cudaq_qec_bposd.ipynb` (NVIDIA GPU BP+OSD) and `03_ising_reproduce.ipynb` (NVIDIA Ising reference numbers).

## Honest results so far (local CPU, quick profile)

All logical error rates match the plain decoders within statistical error (see CIs in the CSVs). On a **CPU** the GPU-style pre-decoder and
batched BP are *slower* than PyMatching and the C++ `ldpc` library respectively: that is expected, since they are written for wide
data-parallel hardware. The claim to test on the GPU is:

* **Surface code:** local pre-decoding removes ~60-80 % of syndrome weight (measured: `syndrome_weight_kept` = 0.2-0.4) and fully resolves
  up to ~50 % of shots at d=5, p=0.002. PyMatching on the residual ran ~2.4x faster than on the raw syndromes at d=13
  (3.48 s -> 1.43 s per 100k shots in a CPU scratch experiment). With the pre-decode step on a GPU this should translate into end-to-end
  speed-up that grows with distance; NVIDIA reports ~2.5x for their CNN at d=13, p=0.003.
* **qLDPC:** batched GPU BP resolves the converged shots (~98 % at p=0.03 on [[72,12,6]]) in parallel; only the rest need OSD.

The MLP gate is an honest negative result for larger distances: a global classifier accepts few shots at d=7 and above, which is why the
local pre-decoder (and NVIDIA's local 3D-CNN) is the right design. It is kept as an ablation.

Do not quote GPU speed-ups until `results/colab-gpu/` exists; the site labels every run with its hardware.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes

* Noise: circuit-level (Stim `rotated_memory_z`) for the surface code; **code-capacity** bit-flip noise for the qLDPC codes (stated everywhere it is shown).
* NVIDIA Ising weights are under the NVIDIA Open Model License and are downloaded at run time, never committed.
* License: MIT for this repository's code.
