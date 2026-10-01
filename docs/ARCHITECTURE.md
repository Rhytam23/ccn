# Architecture

## Pipeline

```
 syndromes (B x N, uint8)
        │
        ▼
 ┌───────────────────────────┐   residual all zero      ┌───────────────┐
 │ Stage 1: pre-decoder      │ ───────────────────────► │ prediction    │
 │  (GPU, batched)           │                          └───────────────┘
 └───────────┬───────────────┘
             │ residual syndrome (sparser) + partial logical flip
             ▼
 ┌───────────────────────────┐
 │ Stage 2: exact global     │ ──► prediction = partial flip XOR global flip
 │  decoder (CPU or GPU)     │
 └───────────────────────────┘
```

Two instantiations share the same idea (cheap parallel stage first, exact stage only for what is left):

| | Surface code (circuit-level noise) | qLDPC (bivariate bicycle, code-capacity noise) |
|---|---|---|
| Stage 1 | `LocalPreDecoder` (or `Gate` MLP) | `BatchedMinSumBP` (PyTorch) |
| "Easy" means | residual syndrome empty | BP converged to a syndrome-consistent estimate |
| Stage 2 | PyMatching on the residual | BP+OSD (`ldpc` CPU, or CUDA-Q QEC GPU) on non-converged shots |
| Module | `pipeline.HybridDecoder` | `decoders.HybridBpOsd` |

## Modules (`src/qechybrid/`)

| Module | Responsibility |
|---|---|
| `data.py` | Stim rotated surface-code circuits, detector sampling, detector-error-model to (H, L, priors) |
| `local_predecoder.py` | Sparse-matrix local rule: mutually isolated fired pairs are matched along their edge if that is no costlier than two boundary matches |
| `gate.py` | MLP predicting the logical flip; threshold on \|logit\| calibrated on held-out data to bound extra logical errors |
| `pipeline.py` | `HybridDecoder` with modes `none`, `zero`, `nn`, `local`; batch and single-shot paths |
| `bp_gpu.py` | Dense batched normalised min-sum BP, chunked over the batch |
| `decoders.py` | `MatchingDecoder`, `CpuBpOsd`, `HybridBpOsd` |
| `codes.py`, `gf2.py` | BB codes, GF(2) RREF, exact logical-failure test (residual in row space of Hx?) |
| `cudaq_qec_adapter.py` | Optional NVIDIA CUDA-Q QEC backend |
| `bench.py`, `timing.py` | Benchmark profiles, latency percentiles, Wilson intervals, environment metadata |

## Design decisions

* **Local over global pre-decoding.** A global classifier (the MLP gate) must be right about the whole volume, so its confident region shrinks as distance grows
  (measured: 10 % accepted at d=5, about 0 % at d=7). A local rule scales with the number of independent fault clusters instead.
* **Exactness preserved.** Stage 2 is always an exact decoder on whatever Stage 1 leaves, so errors can only come from Stage 1's local decisions. The pairing rule is restricted
  to cases where MWPM would make the same choice; a boundary rule exists (`use_boundary=True`) but is off by default because it measurably hurt LER.
* **Calibration, not hope.** The learned gate's threshold is chosen on a separate validation set so that it adds at most 5 % extra errors relative to MWPM.
* **Dense BP.** Dense (B x checks x variables) tensors keep the kernel simple and fast for small and medium codes. Memory grows with B x m x n, hence the `chunk` argument.
  Larger codes would need an edge-list (sparse) implementation.
* **Honest timing.** GPU stages are timed with `torch.cuda.synchronize()`; throughput includes host/device transfers; latency is single-shot.

## Extension points

* **NVIDIA Ising CNN as Stage 1:** implement an object with `predecode(dets) -> (residual, flip)` (same contract as `LocalPreDecoder`) and pass it as `local=`.
  The Ising model uses its own input layout (4 channels x T x D x D), so the adapter must map Stim detectors to it; see NVIDIA/Ising-Decoding `code/qec`.
* **CUDA-Q QEC OSD:** pass `osd_backend=CudaqQecBpOsd(H)` to `HybridBpOsd` to use NVIDIA's GPU BP+OSD for the fallback shots.
* **Other codes:** any Stim circuit works for the surface-code path; any binary parity-check matrix works for the BP path.
