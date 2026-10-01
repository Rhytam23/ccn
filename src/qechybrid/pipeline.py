"""Hybrid surface-code pipeline: fast-path pre-decoder -> exact global decoder (MWPM) for the rest."""
from __future__ import annotations

import numpy as np
import torch

from .timing import now


class HybridDecoder:
    """Modes
    none  : plain global decoding (baseline)
    zero  : trivial shortcut (empty syndrome => no logical flip)
    nn    : learned MLP gate answers confident shots, rest -> global decoder
    local : GPU local pre-decoder clears easy clusters; empty residual => done,
            otherwise the *residual* syndrome (sparser) goes to the global decoder
    """

    def __init__(self, matcher, gate=None, mode: str = "nn", device: str = "cpu", chunk: int = 65536, local=None):
        self.matcher, self.gate, self.mode, self.device, self.chunk = matcher, gate, mode, device, chunk
        self.local = local
        self.stats: dict = {}

    def _front(self, dets):
        """Returns (accepted mask, prediction for all rows, residual syndromes for the rest)."""
        B = len(dets)
        if self.mode == "none":
            return np.zeros(B, bool), np.zeros((B, 1), np.uint8), dets
        if self.mode == "zero":
            return ~dets.any(axis=1), np.zeros((B, 1), np.uint8), dets
        if self.mode == "local":
            resid, flip = self.local.predecode(dets)
            acc = (resid.sum(dim=1) == 0).cpu().numpy()
            self._res_w = float(resid.sum()) / max(1.0, float(np.sum(dets)))
            return acc, flip.cpu().numpy()[:, None], resid.cpu().numpy()
        z = self.gate.logits(dets)
        acc = (z.abs().amax(-1) >= self.gate.threshold).cpu().numpy()
        return acc, (z > 0).to(torch.uint8).cpu().numpy(), dets

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        t0 = now(self.device)
        preds, accepted = [], 0
        t_gate = t_match = 0.0
        w_in = w_out = 0.0
        for i in range(0, len(dets), self.chunk):
            d = dets[i : i + self.chunk]
            a = now(self.device)
            acc, p, res = self._front(d)
            b = now(self.device)
            out = p.copy()
            rest = ~acc
            if rest.any():
                m = self.matcher.decode_batch(res[rest])
                out[rest] = (p[rest] ^ m) if self.mode == "local" else m
            c = now(self.device)
            t_gate += b - a
            t_match += c - b
            accepted += int(acc.sum())
            w_in += float(d.sum())
            w_out += float(res.sum())
            preds.append(out)
        self.stats = {
            "accept_rate": accepted / max(1, len(dets)),
            "t_total": now(self.device) - t0,
            "t_gate": t_gate,
            "t_match": t_match,
            "syndrome_weight_kept": w_out / max(1.0, w_in),
        }
        return np.concatenate(preds) if preds else np.zeros((0, 1), np.uint8)

    def decode_one(self, det: np.ndarray) -> np.ndarray:
        acc, p, res = self._front(det[None])
        if acc[0]:
            return p[0]
        m = self.matcher.decode_one(res[0])
        return (p[0] ^ m) if self.mode == "local" else m
