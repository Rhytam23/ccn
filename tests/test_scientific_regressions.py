"""Frozen inputs and outputs: intentional updates require fixture review.

Broader finite-sample quality smoke checks remain in test_core.py. These tests
detect deterministic behavior changes, not statistical significance or MWPM
equivalence of the heuristic on arbitrary data.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from qechybrid.codes import bb_72_12_6
from qechybrid.data import surface_circuit
from qechybrid.decoders import CpuBpOsd, HybridBpOsd, MatchingDecoder
from qechybrid.local_predecoder import LocalPreDecoder
from qechybrid.pipeline import HybridDecoder

FIXTURES = Path(__file__).resolve().parent / "fixtures"
META = json.loads((FIXTURES / "meta.json").read_text())


@pytest.mark.slow
@pytest.mark.parametrize("radius", [1, 2])
def test_frozen_surface_predictions_and_error_counts(radius):
    with np.load(FIXTURES / "surface_d5_p004.npz") as f:
        matcher = MatchingDecoder(surface_circuit(5, 0.004))
        baseline = matcher.decode_batch(f["dets"])
        actual = HybridDecoder(matcher, mode="local", local=LocalPreDecoder(
            matcher.m, radius=radius)).decode_batch(f["dets"])
        np.testing.assert_array_equal(baseline, f["matching"])
        np.testing.assert_array_equal(actual, f[f"radius{radius}"])
        counts = META["surface"]["errors"]
        assert int((baseline != f["obs"]).any(axis=1).sum()) == counts["matching"]
        assert int((actual != f["obs"]).any(axis=1).sum()) == counts[f"radius{radius}"]


@pytest.mark.slow
def test_frozen_qldpc_predictions_syndromes_and_error_counts():
    code = bb_72_12_6()
    priors = np.full(code.n, 0.04)
    with np.load(FIXTURES / "bb72_p04.npz") as f:
        for name, decoder in (("cpu", CpuBpOsd(code.hz, priors)),
                              ("hybrid", HybridBpOsd(code.hz, priors))):
            actual = decoder.decode_batch(f["syndromes"])
            np.testing.assert_array_equal(actual, f[name])
            np.testing.assert_array_equal((actual.astype(int) @ code.hz.T.astype(int)) % 2, f["syndromes"])
            assert int(code.logical_failure(f["errors"], actual).sum()) == META["qldpc"]["errors"][name]
