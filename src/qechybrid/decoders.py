"""Decoder wrappers: PyMatching, CPU BP+OSD (ldpc), and the hybrid accelerated-BP -> OSD fallback."""
from __future__ import annotations

import numpy as np
import pymatching
import scipy.sparse as sp
from ldpc import BpOsdDecoder

from .bp_gpu import BatchedMinSumBP


class MatchingDecoder:
    def __init__(self, circuit):
        dem = circuit.detector_error_model(decompose_errors=True)
        self.m = pymatching.Matching.from_detector_error_model(dem)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        if len(dets) == 0:
            return np.zeros((0, 1), np.uint8)
        return np.asarray(self.m.decode_batch(dets.astype(np.uint8)), dtype=np.uint8)

    def decode_one(self, det: np.ndarray) -> np.ndarray:
        return np.asarray(self.m.decode(det.astype(np.uint8)), dtype=np.uint8)


class CpuBpOsd:
    def __init__(self, H, priors, max_iter: int = 50, osd_order: int = 7, ms_scaling: float = 0.8):
        self.n = H.shape[1]
        self.dec = BpOsdDecoder(
            sp.csr_matrix(H, dtype=np.uint8),
            channel_probs=np.asarray(priors, dtype=float),
            max_iter=max_iter,
            bp_method="minimum_sum",
            ms_scaling_factor=ms_scaling,
            osd_method="osd_cs",
            osd_order=osd_order,
        )

    def decode_one(self, syn: np.ndarray) -> np.ndarray:
        return np.asarray(self.dec.decode(syn.astype(np.uint8)), dtype=np.uint8)

    def decode_batch(self, S: np.ndarray) -> np.ndarray:
        if len(S) == 0:
            return np.zeros((0, self.n), np.uint8)
        return np.stack([self.decode_one(s) for s in S])


class HybridBpOsd:
    """Batched min-sum BP on the accelerator; only non-converged shots go to BP+OSD."""

    def __init__(self, H, priors, device="cpu", max_iter=50, osd_order=7, alpha=0.8, osd_backend=None):
        self.bp = BatchedMinSumBP(H, priors, max_iter=max_iter, alpha=alpha, device=device)
        self.osd = osd_backend or CpuBpOsd(H, priors, max_iter=max_iter, osd_order=osd_order, ms_scaling=alpha)
        self.last_converged_frac = 1.0

    def decode_batch(self, S: np.ndarray) -> np.ndarray:
        est, conv = self.bp.decode(S)
        bad = np.nonzero(~conv)[0]
        if len(bad):
            est[bad] = self.osd.decode_batch(S[bad])
        self.last_converged_frac = float(conv.mean()) if len(conv) else 1.0
        return est
