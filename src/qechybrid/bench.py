"""Benchmark harness: logical error rate, throughput and latency percentiles."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cudaq_qec_adapter
from .codes import bb_72_12_6, bb_144_12_12
from .data import sample, surface_circuit
from .decoders import CpuBpOsd, HybridBpOsd, MatchingDecoder
from .gate import Gate
from .local_predecoder import LocalPreDecoder
from .pipeline import HybridDecoder
from .timing import now, percentiles_us, wilson

PROFILES = {
    "quick": dict(
        surface=dict(ds=[5, 7], ps=[0.002, 0.004], shots=20000, train=200000, val=50000, epochs=6, lat=500),
        qldpc=dict(codes=["bb72"], ps=[0.02, 0.04], shots=1000, lat=200),
    ),
    "full": dict(
        surface=dict(ds=[5, 7, 9, 13], ps=[0.001, 0.002, 0.004], shots=200000, train=400000, val=100000, epochs=8, lat=2000),
        qldpc=dict(codes=["bb72", "bb144"], ps=[0.02, 0.04, 0.06], shots=20000, lat=1000),
    ),
}


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def environment_meta(device: str) -> dict:
    import pymatching
    import stim
    import torch

    try:
        import ldpc

        ldpc_version = getattr(ldpc, "__version__", "unknown")
    except Exception:
        ldpc_version = "unavailable"

    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    commit = _git("rev-parse", "--short", "HEAD")
    cuda_device = device if str(device).startswith("cuda") and torch.cuda.is_available() else None
    driver = None
    if cuda_device is not None:
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10, check=True).stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        git_commit=commit or "unknown",
        git_branch=_git("branch", "--show-current"),
        git_dirty=bool(_git("status", "--porcelain")) if commit else None,
        ldpc=ldpc_version,
        cuda=torch.version.cuda,
        nvidia_driver=driver,
        gpu_memory_gb=round(torch.cuda.get_device_properties(cuda_device).total_memory / 2**30, 1) if cuda_device else None,
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        torch=torch.__version__,
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        stim=stim.__version__,
        pymatching=pymatching.__version__,
        device=device,
        gpu=torch.cuda.get_device_name(cuda_device) if cuda_device else None,
    )
    source = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    meta["source_sha256"] = source.hexdigest()
    ok, info = cudaq_qec_adapter.available()
    meta["cudaq_qec"] = info if ok else f"unavailable ({info})"
    return meta


def _latency(fn, items, warmup=20, device="cpu"):
    for x in items[:warmup]:
        fn(x)
    ts = []
    for x in items:
        t0 = now(device)
        fn(x)
        ts.append(now(device) - t0)
    return percentiles_us(ts)


def _batch_latency_ms(decode_batch, data, bs=256, reps=20, device="cpu"):
    """Wall-clock per micro-batch of `bs` shots (median over `reps`). Relevant when many logical qubits
    stream syndromes in parallel; batch size 1 (lat_p50_us) is the worst case for accelerators."""
    n = len(data)
    if n < bs:
        return float("nan")
    decode_batch(data[:bs])  # warm-up
    ts = []
    for r in range(reps):
        i = (r * bs) % max(1, n - bs)
        t0 = now(device)
        decode_batch(data[i : i + bs])
        ts.append(now(device) - t0)
    return float(np.median(ts) * 1e3)


def _row(**kw):
    return kw


def _measure_throughput(backends, data, reps=5, internal=False):
    """Alternate decoder order per repetition; retain all timings and check predictions.

    Surface pipelines supply their own timer (excluding untimed bookkeeping).
    Other backends are measured around the complete decode_batch call.
    Warm-up is the caller's responsibility.
    """
    if reps < 3:
        raise ValueError("throughput requires at least 3 repetitions")
    measured = [dict(times=[], stats=[], pred=None) for _ in backends]
    for repetition in range(reps):
        order = range(len(backends)) if repetition % 2 == 0 else reversed(range(len(backends)))
        for i in order:
            dec, dev = backends[i]
            if internal:
                pred = dec.decode_batch(data)
                stats = dec.stats.copy()
                dt = stats["t_total"]
            else:
                t0 = now(dev)
                pred = dec.decode_batch(data)
                dt = now(dev) - t0
                stats = dict(accept_rate=getattr(dec, "last_converged_frac", np.nan))
            result = measured[i]
            if result["pred"] is None:
                result["pred"] = pred.copy()
            elif not np.array_equal(result["pred"], pred):
                raise RuntimeError("decoder predictions changed between throughput repetitions")
            result["times"].append(dt)
            result["stats"].append(stats)
    return measured


def _throughput_metrics(measured, shots, baseline):
    times = np.asarray(measured["times"])
    rates = shots / times
    ratios = np.asarray(baseline["times"]) / times
    return dict(
        throughput_sps=float(np.median(rates)), throughput_reps=len(times),
        throughput_min_sps=float(rates.min()), throughput_max_sps=float(rates.max()),
        throughput_q25_sps=float(np.percentile(rates, 25)), throughput_q75_sps=float(np.percentile(rates, 75)),
        throughput_times_s=json.dumps(times.tolist()),
        paired_baseline_times_s=json.dumps(baseline["times"]),
        paired_ratio_median=float(np.median(ratios)), paired_ratio_min=float(ratios.min()),
        paired_ratio_max=float(ratios.max()), paired_ratio_q25=float(np.percentile(ratios, 25)),
        paired_ratio_q75=float(np.percentile(ratios, 75)),
    )


def run_surface(d, p, cfg, device, log=print, on_row=None):
    circ = surface_circuit(d, p)
    matcher = MatchingDecoder(circ)
    te_d, te_o = sample(circ, cfg["shots"], seed=3)
    n = len(te_d)
    lat_items = [te_d[i] for i in range(min(cfg["lat"], n))]
    rows = []
    modes = [
        ("pymatching (CPU)", "none", None),
        ("zero-syndrome shortcut + pymatching", "zero", None),
        ("GPU local pre-decoder (conservative, r=2) + pymatching", "local", 2),
        ("GPU local pre-decoder (fast, r=1) + pymatching", "local", 1),
    ]
    selected = cfg.get("modes", ("none", "zero", "local", "nn"))
    modes = [entry for entry in modes if entry[1] in selected]
    # Retain the practical CPU baseline even when only an accelerator mode is selected.
    if not any(mode == "none" for _, mode, _ in modes):
        modes.insert(0, ("pymatching (CPU)", "none", None))
    backends = []
    for name, mode, radius in modes:
        timing_device = device if mode == "local" else "cpu"
        local = LocalPreDecoder(matcher.m, device=device, radius=radius) if mode == "local" else None
        dec = HybridDecoder(matcher, mode=mode, device=timing_device, local=local)
        dec.decode_batch(te_d[:2000])
        backends.append((dec, timing_device))
    measured = _measure_throughput(backends, te_d, reps=cfg.get("reps", 5), internal=True)

    def append_row(name, dec, timing_device, result, baseline, extra=""):
        stats = {key: float(np.median([s[key] for s in result["stats"]])) for key in result["stats"][0]}
        pred = result["pred"]
        errs = int((pred[:, 0] != te_o[:, 0]).sum())
        ler, lo, hi = wilson(errs, n)
        lat = _latency(dec.decode_one, lat_items, device=timing_device)
        lat_b = _batch_latency_ms(dec.decode_batch, te_d, device=timing_device)
        metrics = _throughput_metrics(result, n, baseline)
        rows.append(
            _row(
                code="surface", d=d, rounds=d, p=p, decoder=name, device=timing_device,
                shots=n, errors=errs, ler=ler, ler_lo=lo, ler_hi=hi,
                **metrics, lat_p50_us=lat["p50"], lat_p95_us=lat["p95"], lat_p99_us=lat["p99"], lat_b256_ms=lat_b,
                accept_rate=stats["accept_rate"], syndrome_weight_kept=stats["syndrome_weight_kept"],
                t_stage1_s=stats["t_gate"], t_global_s=stats["t_match"], extra=extra,
            )
        )
        if on_row is not None:
            on_row(rows[-1])
        log(f"  surface d={d} p={p} {name:38s} LER={ler:.2e} {metrics['throughput_sps']:,.0f} median shots/s accept={stats['accept_rate']:.3f}")

    for (name, _, _), (dec, dev), result in zip(modes, backends, measured):
        append_row(name, dec, dev, result, measured[0])

    if "nn" in selected:
        # CPU/local measurements finish before training allocates MLP memory.
        tr_d, tr_o = sample(circ, cfg["train"], seed=1)
        va_d, va_o = sample(circ, cfg["val"], seed=2)
        gate = Gate(te_d.shape[1], 1, device=device)
        t0 = now(device)
        gate.fit(tr_d, tr_o, epochs=cfg["epochs"])
        train_s = now(device) - t0
        gate.calibrate(va_d, va_o, matcher.decode_batch(va_d), max_extra_error_frac=0.05)
        dec = HybridDecoder(matcher, gate, mode="nn", device=device)
        dec.decode_batch(te_d[:2000])
        # Fresh CPU repetitions pair with the MLP in this separate measurement group.
        paired = _measure_throughput([backends[0], (dec, device)], te_d, reps=cfg.get("reps", 5), internal=True)
        append_row("AI gate (MLP, strict) + pymatching", dec, device, paired[1], paired[0],
                   extra=f"gate_train_s={train_s:.1f};thr_logit={gate.threshold:.3f}")
    return rows


def run_qldpc(code, p, cfg, device, log=print):
    pri = np.full(code.n, p)
    e, S = code.sample(cfg["shots"], p, seed=7)
    n = len(S)
    lat_items = [S[i : i + 1] for i in range(min(cfg["lat"], n))]
    backends = [("ldpc BP+OSD (CPU)", CpuBpOsd(code.hz, pri), "cpu"), (
        f"GPU min-sum BP + OSD fallback", HybridBpOsd(code.hz, pri, device=device), device)]
    ok, _ = cudaq_qec_adapter.available()
    if ok and device.startswith("cuda"):
        try:
            backends.append(("CUDA-Q QEC nv-qldpc-decoder (GPU)", cudaq_qec_adapter.CudaqQecBpOsd(code.hz), device))
        except Exception as exc:  # pragma: no cover
            log(f"  CUDA-Q QEC adapter failed: {exc}")
    rows = []
    for _, dec, _ in backends:
        dec.decode_batch(S[:64])
    measured = _measure_throughput([(dec, dev) for _, dec, dev in backends], S, reps=cfg.get("reps", 5))
    for (name, dec, dev), result in zip(backends, measured):
        est = result["pred"]
        conv = result["stats"][0]["accept_rate"]
        metrics = _throughput_metrics(result, n, measured[0])
        errs = int(code.logical_failure(e, est).sum())
        ler, lo, hi = wilson(errs, n)
        fn = (lambda x: dec.decode_batch(x)) if not isinstance(dec, CpuBpOsd) else (lambda x: dec.decode_one(x[0]))
        lat = _latency(fn, lat_items, warmup=10, device=dev)
        lat_b = _batch_latency_ms(dec.decode_batch, S, device=dev)
        rows.append(
            _row(
                code=code.name, d=0, rounds=0, p=p, decoder=name, device=dev, shots=n, errors=errs, ler=ler, ler_lo=lo,
                ler_hi=hi, **metrics, lat_p50_us=lat["p50"], lat_p95_us=lat["p95"], lat_p99_us=lat["p99"], lat_b256_ms=lat_b,
                accept_rate=conv, extra="code-capacity noise; accept_rate = BP converged fraction",
            )
        )
        log(f"  {code.name} p={p} {name:38s} LER={ler:.2e} {metrics['throughput_sps']:,.0f} median shots/s")
    return rows


def run_profile(profile: str, device: str, tag: str, out_root: str = "results", which=("surface", "qldpc"), log=print,
                reps: int = 5, surface_modes=None):
    if reps < 3:
        raise ValueError("throughput requires at least 3 repetitions")
    device = resolve_device(device)
    cfg = copy.deepcopy(PROFILES[profile])
    for section in cfg.values():
        section["reps"] = reps
    if surface_modes is not None:
        if not surface_modes or set(surface_modes) - {"none", "zero", "local", "nn"}:
            raise ValueError("surface_modes must contain none, zero, local, and/or nn")
        cfg["surface"]["modes"] = list(surface_modes)
    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    meta = environment_meta(device)
    meta["profile"] = profile
    meta["which"] = list(which)
    meta["benchmark_protocol"] = "median-paired-v2"
    meta["throughput_reps"] = reps
    meta["decoder_order"] = "forward on even repetitions, reverse on odd; MLP separately paired with fresh CPU baseline"
    meta["config"] = cfg
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"device={device} gpu={meta['gpu']} tag={tag}")
    if "surface" in which:
        rows = []

        def save_surface_row(row):
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "surface.csv", index=False)

        for d in cfg["surface"]["ds"]:
            for p in cfg["surface"]["ps"]:
                run_surface(d, p, cfg["surface"], device, log, on_row=save_surface_row)
    if "qldpc" in which:
        rows = []
        for name in cfg["qldpc"]["codes"]:
            code = bb_72_12_6() if name == "bb72" else bb_144_12_12()
            for p in cfg["qldpc"]["ps"]:
                rows += run_qldpc(code, p, cfg["qldpc"], device, log)
                pd.DataFrame(rows).to_csv(out / "qldpc.csv", index=False)
    log(f"wrote {out}")
    return out
