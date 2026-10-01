"""Head-to-head with NVIDIA's Ising 3D-CNN pre-decoder (https://github.com/NVIDIA/Ising-Decoding, Apache-2.0).

We do NOT vendor NVIDIA's code. Clone it next to this repo (the Colab notebook does) and pass its path.
The wiring below follows NVIDIA's own cookbook (`cookbook/predecoder.ipynb`):

    Stim syndromes (their MemoryCircuit + 25-parameter noise model, boundary detectors)
      -> PreDecoderMemoryEvalModule  (3D CNN, returns [partial logical flip | residual detectors])
      -> PyMatching on the residual   -> XOR of both flips

and compares it on the SAME shots against (a) plain PyMatching and (b) our own local pre-decoder.

Model weights (NVIDIA Open Model License) are gated on Hugging Face: you need to accept the terms there
and provide your own token (env var HF_TOKEN or a cached `huggingface-cli login`). Tokens are never
stored or printed here. With `weights=None` the network is randomly initialised: the plumbing and timing
can be checked, but the logical error rate is meaningless and rows are labelled accordingly.
"""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import numpy as np

from .timing import now, percentiles_us, wilson

HF_REPOS = {
    1: ("nvidia/ising_decoder_surface_code_1_fast", "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors"),
    2: ("nvidia/ising_decoder_surface_code_1_accurate", "ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors"),
}


def download_weights(model_id: int = 1) -> str:
    """Download gated weights using the caller's own HF credentials."""
    from huggingface_hub import hf_hub_download

    repo, fname = HF_REPOS[model_id]
    return hf_hub_download(repo_id=repo, filename=fname, token=os.environ.get("HF_TOKEN") or True)


def _import_nvidia(repo: str):
    code = os.path.join(os.path.abspath(repo), "code")
    if not os.path.isdir(code):
        raise FileNotFoundError(f"{code} not found: clone https://github.com/NVIDIA/Ising-Decoding first")
    if code not in sys.path:
        sys.path.insert(0, code)
    from evaluation.logical_error_rate import PreDecoderMemoryEvalModule, _build_stab_maps
    from model.factory import ModelFactory
    from model.registry import get_model_spec
    from qec.noise_model import NoiseModel
    from qec.surface_code.memory_circuit import MemoryCircuit

    return PreDecoderMemoryEvalModule, _build_stab_maps, ModelFactory, get_model_spec, NoiseModel, MemoryCircuit


def build_context(repo, distance=9, n_rounds=None, p=0.005, shots=5000, basis="X", code_rotation="XV",
                  model_id=1, weights=None, device="cpu", seed=0):
    import pymatching
    import torch

    Pipe, build_maps, Factory, get_spec, Noise, Circuit = _import_nvidia(repo)
    n_rounds = n_rounds or distance
    noise = Noise.from_single_p(p)
    pmax = noise.get_max_probability()
    circ = Circuit(distance=distance, n_rounds=n_rounds, basis=basis, code_rotation=code_rotation,
                   idle_error=pmax, sqgate_error=pmax, tqgate_error=pmax, spam_error=(2 / 3) * pmax,
                   noise_model=noise, add_boundary_detectors=True)
    circ.set_error_rates()
    sc = circ.stim_circuit
    meas = sc.compile_sampler(seed=seed).sample(shots)
    do = sc.compile_m2d_converter().convert(measurements=meas, append_observables=True)
    nobs = sc.num_observables
    det, obs = do[:, :-nobs].astype(np.uint8), do[:, -nobs:].astype(np.uint8)
    dem = sc.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    spec = get_spec(model_id)
    mcfg = SimpleNamespace(code="surface", distance=distance, n_rounds=n_rounds, model=SimpleNamespace(
        version="predecoder_memory_v1", num_filters=list(spec.num_filters), kernel_size=list(spec.kernel_size),
        dropout_p=0.0, activation="gelu", input_channels=4, out_channels=4))
    model = Factory.create_model(mcfg)
    if weights:
        from safetensors.torch import load_file

        sd = load_file(weights, device="cpu")
        model.load_state_dict({(k[7:] if k.startswith("module.") else k): v.float() for k, v in sd.items()})
    model = model.to(device).eval()
    ecfg = SimpleNamespace(distance=distance, enable_fp16=False, data=SimpleNamespace(code_rotation=code_rotation),
                           test=SimpleNamespace(meas_basis_test=basis, th_data=0.0, th_syn=0.0, sampling_mode="threshold",
                                                temperature=1.0, temperature_data=None, temperature_syn=None, n_rounds=n_rounds))
    pipe = Pipe(model, ecfg, build_maps(distance, code_rotation), torch.device(device)).to(device).eval()
    return SimpleNamespace(pipe=pipe, det=det, obs=obs[:, 0], matcher=matcher, device=device, d=distance, rounds=n_rounds,
                           p=p, trained=bool(weights), model_id=model_id, shots=shots)


def run_comparison(ctx, lat_shots: int = 200, log=print) -> list[dict]:
    """Rows (same schema as the surface benchmark) for PyMatching, NVIDIA Ising + PyMatching, our local pre-decoder."""
    import torch

    from .local_predecoder import LocalPreDecoder
    from .pipeline import HybridDecoder

    dev, det, obs, m = ctx.device, ctx.det, ctx.obs, ctx.matcher

    class _Matcher:  # HybridDecoder expects decode_batch / decode_one
        decode_batch = staticmethod(lambda d: np.asarray(m.decode_batch(d), dtype=np.uint8).reshape(len(d), -1))
        decode_one = staticmethod(lambda x: np.asarray(m.decode(x), dtype=np.uint8))

    n = len(det)
    tag = "" if ctx.trained else " [RANDOM WEIGHTS: LER not meaningful]"
    rows = []

    def row(name, errs, thr, lat, accept, kept, t1, t2, extra=""):
        ler, lo, hi = wilson(errs, n)
        return dict(code="surface-nvidia-circuit", d=ctx.d, rounds=ctx.rounds, p=ctx.p, decoder=name + (tag if "Ising" in name else ""), device=dev, shots=n,
                    errors=errs, ler=ler, ler_lo=lo, ler_hi=hi, throughput_sps=thr, lat_p50_us=lat.get("p50", np.nan),
                    lat_p95_us=lat.get("p95", np.nan), lat_p99_us=lat.get("p99", np.nan), accept_rate=accept,
                    syndrome_weight_kept=kept, t_stage1_s=t1, t_global_s=t2, extra=extra)

    def lat_of(fn):
        try:
            items = [det[i] for i in range(min(lat_shots, n))]
            for x in items[:10]:
                fn(x)
            ts = []
            for x in items:
                t0 = now(dev)
                fn(x)
                ts.append(now(dev) - t0)
            return percentiles_us(ts)
        except Exception as exc:  # e.g. recompilation limits at batch size 1
            log(f"  latency skipped: {type(exc).__name__}")
            return {}

    # 1) PyMatching only
    m.decode_batch(det[:500])
    t0 = time.perf_counter()
    pm = np.asarray(m.decode_batch(det), dtype=np.uint8).reshape(n, -1)[:, 0]
    t_pm = time.perf_counter() - t0
    rows.append(row("pymatching (CPU)", int((pm != obs).sum()), n / t_pm, lat_of(lambda x: m.decode(x)), 0.0, 1.0, 0.0, t_pm))

    # 2) NVIDIA Ising + PyMatching
    dt = torch.from_numpy(det).to(torch.uint8).to(dev)
    with torch.no_grad():
        pipe_out = ctx.pipe(dt[: min(n, 256)])  # warm-up / compile
    t0 = now(dev)
    with torch.no_grad():
        out = ctx.pipe(dt)
    t_pd = now(dev) - t0
    flip = out[:, 0].cpu().numpy().astype(np.uint8)
    res = out[:, 1:].cpu().numpy().astype(np.uint8)
    t0 = time.perf_counter()
    pmr = np.asarray(m.decode_batch(res), dtype=np.uint8).reshape(n, -1)[:, 0]
    t_pr = time.perf_counter() - t0
    pred = flip ^ pmr

    def ising_one(x):
        with torch.no_grad():
            o = ctx.pipe(torch.from_numpy(x[None]).to(torch.uint8).to(dev))
        r = o[0, 1:].cpu().numpy().astype(np.uint8)
        return int(o[0, 0].item()) ^ int(m.decode(r)[0])

    rows.append(row("NVIDIA Ising + pymatching", int((pred != obs).sum()), n / (t_pd + t_pr), lat_of(ising_one),
                    float((res.sum(1) == 0).mean()), float(res.sum()) / max(1.0, float(det.sum())), t_pd, t_pr,
                    extra=f"model_id={ctx.model_id}"))

    # 3) our local pre-decoder + PyMatching, same shots
    hd = HybridDecoder(_Matcher, mode="local", device=dev, local=LocalPreDecoder(m, device=dev))
    hd.decode_batch(det[:500])
    p3 = hd.decode_batch(det)[:, 0]
    s = hd.stats
    rows.append(row("ours: local pre-decoder + pymatching", int((p3 != obs).sum()), n / s["t_total"], lat_of(lambda x: hd.decode_one(x)),
                    s["accept_rate"], s["syndrome_weight_kept"], s["t_gate"], s["t_match"]))
    for r in rows:
        log(f"  d={ctx.d} p={ctx.p} {r['decoder']:55s} LER={r['ler']:.2e} {r['throughput_sps']:,.0f} shots/s")
    return rows
