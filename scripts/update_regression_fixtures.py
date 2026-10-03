"""Explicitly regenerate small scientific fixtures after reviewing decoder changes.

Run from the repository root. Never run this as part of tests: doing so would
silently bless a regression. Commit and review changed fixtures and counts.
"""
import json
from pathlib import Path

import numpy as np

from qechybrid.bench import environment_meta
from qechybrid.codes import bb_72_12_6
from qechybrid.data import sample, surface_circuit
from qechybrid.decoders import CpuBpOsd, HybridBpOsd, MatchingDecoder
from qechybrid.local_predecoder import LocalPreDecoder
from qechybrid.pipeline import HybridDecoder


def main():
    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out.mkdir(exist_ok=True)
    c = surface_circuit(5, 0.004)
    dets, obs = sample(c, 4096, seed=19)
    matcher = MatchingDecoder(c)
    surface = dict(dets=dets, obs=obs, matching=matcher.decode_batch(dets))
    for radius in (1, 2):
        surface[f"radius{radius}"] = HybridDecoder(
            matcher, mode="local", local=LocalPreDecoder(matcher.m, radius=radius)).decode_batch(dets)
    np.savez_compressed(out / "surface_d5_p004.npz", **surface)
    code = bb_72_12_6()
    errors, syndromes = code.sample(1024, 0.04, seed=19)
    priors = np.full(code.n, 0.04)
    cpu = CpuBpOsd(code.hz, priors).decode_batch(syndromes)
    hybrid = HybridBpOsd(code.hz, priors).decode_batch(syndromes)
    np.savez_compressed(out / "bb72_p04.npz", errors=errors, syndromes=syndromes, cpu=cpu, hybrid=hybrid)
    meta = dict(environment=environment_meta("cpu"), seed=19,
                surface=dict(d=5, p=0.004, shots=len(dets),
                             errors={k: int((surface[k] != obs).any(axis=1).sum()) for k in ("matching", "radius1", "radius2")}),
                qldpc=dict(code="bb72", p=0.04, shots=len(errors),
                           errors=dict(cpu=int(code.logical_failure(errors, cpu).sum()),
                                       hybrid=int(code.logical_failure(errors, hybrid).sum()))))
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
