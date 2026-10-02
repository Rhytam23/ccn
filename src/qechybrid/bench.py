"""Benchmark harness: logical error rate, throughput and latency percentiles."""
from __future__ import annotations

import json
import os
import platform
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

    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        torch=torch.__version__,
        stim=stim.__version__,
        pymatching=pymatching.__version__,
        device=device,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
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


def _row(**kw):
    return kw


def run_surface(d, p, cfg, device, log=print):
    circ = surface_circuit(d, p)
    matcher = MatchingDecoder(circ)
    tr_d, tr_o = sample(circ, cfg["train"], seed=1)
    va_d, va_o = sample(circ, cfg["val"], seed=2)
    te_d, te_o = sample(circ, cfg["shots"], seed=3)
    gate = Gate(te_d.shape[1], 1, device=device)
    t0 = time.perf_counter()
    gate.fit(tr_d, tr_o, epochs=cfg["epochs"])
    train_s = time.perf_counter() - t0
    va_match = matcher.decode_batch(va_d)
    n = len(te_d)
    lat_items = [te_d[i] for i in range(min(cfg["lat"], n))]
    rows = []
    modes = [
        ("pymatching (CPU)", "none", None),
        ("zero-syndrome shortcut + pymatching", "zero", None),
        ("GPU local pre-decoder (safe, r=2) + pymatching", "local", 2),
        ("GPU local pre-decoder (fast, r=1) + pymatching", "local", 1),
        ("AI gate (MLP, strict) + pymatching", "nn", 0.05),
    ]
    locals_ = {rad: LocalPreDecoder(matcher.m, device=device, radius=rad) for rad in (1, 2)}
    for name, mode, budget in modes:
        if mode == "nn":
            gate.calibrate(va_d, va_o, va_match, max_extra_error_frac=budget)
        dec = HybridDecoder(matcher, gate, mode=mode, device=device, local=locals_.get(budget))
        dec.decode_batch(te_d[:2000])  # warm-up
        best = None
        for _ in range(2):
            pred = dec.decode_batch(te_d)
            if best is None or dec.stats["t_total"] < best[0]["t_total"]:
                best = (dec.stats, pred)
        stats, pred = best
        errs = int((pred[:, 0] != te_o[:, 0]).sum())
        ler, lo, hi = wilson(errs, n)
        lat = _latency(dec.decode_one, lat_items, device=device)
        rows.append(
            _row(
                code="surface", d=d, rounds=d, p=p, decoder=name, device=device if mode in ("nn", "local") else "cpu",
                shots=n, errors=errs, ler=ler, ler_lo=lo, ler_hi=hi,
                throughput_sps=n / stats["t_total"], lat_p50_us=lat["p50"], lat_p95_us=lat["p95"], lat_p99_us=lat["p99"],
                accept_rate=stats["accept_rate"], syndrome_weight_kept=stats["syndrome_weight_kept"], t_stage1_s=stats["t_gate"], t_global_s=stats["t_match"], extra=f"gate_train_s={train_s:.1f};thr_logit={gate.threshold:.3f}",
            )
        )
        log(f"  surface d={d} p={p} {name:38s} LER={ler:.2e} {n / stats['t_total']:,.0f} shots/s accept={stats['accept_rate']:.3f}")
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
    for name, dec, dev in backends:
        dec.decode_batch(S[:64])
        t0 = now(dev)
        est = dec.decode_batch(S)
        dt = now(dev) - t0
        conv = getattr(dec, "last_converged_frac", None)  # read before the latency loop overwrites it
        errs = int(code.logical_failure(e, est).sum())
        ler, lo, hi = wilson(errs, n)
        fn = (lambda x: dec.decode_batch(x)) if not isinstance(dec, CpuBpOsd) else (lambda x: dec.decode_one(x[0]))
        lat = _latency(fn, lat_items, warmup=10, device=dev)
        rows.append(
            _row(
                code=code.name, d=0, rounds=0, p=p, decoder=name, device=dev, shots=n, errors=errs, ler=ler, ler_lo=lo,
                ler_hi=hi, throughput_sps=n / dt, lat_p50_us=lat["p50"], lat_p95_us=lat["p95"], lat_p99_us=lat["p99"],
                accept_rate=conv if conv is not None else np.nan, extra="code-capacity noise; accept_rate = BP converged fraction",
            )
        )
        log(f"  {code.name} p={p} {name:38s} LER={ler:.2e} {n / dt:,.0f} shots/s")
    return rows


def run_profile(profile: str, device: str, tag: str, out_root: str = "results", which=("surface", "qldpc"), log=print):
    device = resolve_device(device)
    cfg = PROFILES[profile]
    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    meta = environment_meta(device)
    meta["profile"] = profile
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"device={device} gpu={meta['gpu']} tag={tag}")
    if "surface" in which:
        rows = []
        for d in cfg["surface"]["ds"]:
            for p in cfg["surface"]["ps"]:
                rows += run_surface(d, p, cfg["surface"], device, log)
                pd.DataFrame(rows).to_csv(out / "surface.csv", index=False)
    if "qldpc" in which:
        rows = []
        for name in cfg["qldpc"]["codes"]:
            code = bb_72_12_6() if name == "bb72" else bb_144_12_12()
            for p in cfg["qldpc"]["ps"]:
                rows += run_qldpc(code, p, cfg["qldpc"], device, log)
                pd.DataFrame(rows).to_csv(out / "qldpc.csv", index=False)
    log(f"wrote {out}")
    return out
