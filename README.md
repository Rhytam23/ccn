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

### Colab T4, run 2: full profile, 200,000 shots/point (data in `results/colab-gpu/`, code at commit `a381f7f`)

**qLDPC (code-capacity noise): the GPU BP is now a modest real win and matches `ldpc` exactly.**

| code | p | GPU hybrid vs C++ ldpc (shots/s) | logical errors (GPU / ldpc) |
|---|---|---|---|
| [[72,12,6]] | 0.02 | 86k vs 78k = 1.11x | 226 / 226 |
| [[72,12,6]] | 0.06 | 21k vs 19k = 1.12x | 4991 / 4991 |
| [[144,12,12]] | 0.02 | 62k vs 39k = **1.58x** | 13 / 13 |
| [[144,12,12]] | 0.04 | 25k vs 20k = 1.26x | 233 / 233 |
| [[144,12,12]] | 0.06 | 7.9k vs 7.6k = 1.03x | 1772 / 1772 |

Batch throughput only: single-shot latency is ~1.6-6 ms on the GPU vs ~11-40 us for `ldpc`, so this is not a real-time latency win.
(The "BP converged" column in that run is wrong: it was read after the latency loop. Fixed in the next commit.)

**Surface code, run-2 code (aggressive radius-1 local rule): not a win.** At d>=7 stage 1 cost more than it saved (d=13: stage 1 3.6 s vs 3.3 s saved) and it added logical
errors (d=13, p=0.004: 161 vs 109; d=9, p=0.004: 586 vs 486). The MLP gate was slower than PyMatching almost everywhere.

### NVIDIA Ising head-to-head (Colab T4, trained weights; `results/colab-ising-run1/NOTES.md`): one case so far
At d=9, p=0.003 (20k shots, NVIDIA's circuit) the trained Ising model leaves only **2.9 %** of the syndrome weight and fully resolves **47 %** of shots, making the PyMatching stage ~3.8x faster
(0.10 s vs 0.39 s) at a similar error count (26 vs 22). Our classical local rule keeps ~48 % of the weight. The run's stage-1 time (15.6 s) is a torch.compile warm-up artifact, now fixed in the adapter,
so **no Ising end-to-end speed-up is claimed yet**: rerun notebook 04 to get a valid timing and the missing cases.

### What changed after run 2 (CPU-verified, GPU not yet re-measured)
* **Accuracy fix found on CPU:** requiring that *nothing else fired within two hops* of an isolated pair (`radius=2`, now the default) removes the extra logical errors entirely
  (d=9, p=0.004: 449 vs 448 baseline errors; d=7, p=0.002: 52 vs 52, 200k shots each) at the price of a smaller speed-up for the global decoder (1.1-1.3x instead of 1.4-1.7x).
  The aggressive rule is kept as `radius=1` ("fast"), so the benchmark now reports both: a lossless dial and a faster-but-lossy one.
* **Stage 1 made cheap on the GPU:** 1-byte uploads widened on the device, fp16 matrix products on tensor cores (exact for these small integer counts), and host-side
  bookkeeping moved out of the timed region. The GPU code path is unit-tested for equality with the CPU path (in fp32). **Run notebook 04 again to measure it.**
* **Honest expectation:** with radius-2 the safe rule makes the PyMatching stage only ~1.1-1.3x faster, so even a free stage 1 gives about that much end-to-end. A larger surface-code gain
  needs a stronger pre-decoder, which is what the NVIDIA Ising comparison (notebook 04, needs the Hugging Face token) is for. It has not produced results yet.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Notes

* Noise: circuit-level (Stim `rotated_memory_z`) for the surface code; **code-capacity** bit-flip noise for the qLDPC codes (stated everywhere it is shown).
* NVIDIA Ising weights are under the NVIDIA Open Model License and are downloaded at run time, never committed.
* License: MIT for this repository's code.
