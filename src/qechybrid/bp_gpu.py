"""Batched normalized min-sum belief propagation in PyTorch (runs on CUDA or CPU).

Dense (shots x checks x vars) message tensors: simple, fast for the small/medium
parity-check matrices used here, and each decode is fully vectorised over the batch.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

_BIG = 1e9


class BatchedMinSumBP:
    def __init__(self, H, priors, max_iter: int = 50, alpha: float = 0.8, device: str = "cpu", chunk: int = 2048):
        Hd = H.toarray() if sp.issparse(H) else np.asarray(H)
        self.device = device
        self.mask = torch.as_tensor(Hd != 0, device=device)
        self.Hf = self.mask.float()
        self.m, self.n = self.mask.shape
        p = np.clip(np.asarray(priors, dtype=np.float64), 1e-9, 1 - 1e-9)
        self.llr0 = torch.as_tensor(np.log((1 - p) / p), dtype=torch.float32, device=device)
        self.max_iter, self.alpha, self.chunk = max_iter, alpha, chunk
        self.col = torch.arange(self.n, device=device)

    @torch.no_grad()
    def _decode_chunk(self, syn: torch.Tensor):
        B = syn.shape[0]
        mask = self.mask[None]
        v2c = self.llr0.expand(B, self.m, self.n) * mask
        sgn_s = 1.0 - 2.0 * syn
        conv = torch.zeros(B, dtype=torch.bool, device=self.device)
        out = torch.zeros(B, self.n, dtype=torch.bool, device=self.device)
        hard = out
        for _ in range(self.max_iter):
            neg = (v2c < 0) & mask
            sign_tot = 1.0 - 2.0 * (neg.sum(-1) % 2).float()
            own = torch.where(v2c < 0, -1.0, 1.0)
            sign_excl = sign_tot[..., None] * own
            absv = v2c.abs().masked_fill(~mask, _BIG)
            min1, idx1 = absv.min(-1)
            min2 = absv.scatter(-1, idx1[..., None], _BIG).min(-1).values
            is_min = self.col[None, None, :] == idx1[..., None]
            min_excl = torch.where(is_min, min2[..., None], min1[..., None])
            c2v = self.alpha * sgn_s[..., None] * sign_excl * min_excl * mask
            total = self.llr0 + c2v.sum(1)
            hard = total < 0
            ok = (((hard.float() @ self.Hf.T) % 2) == syn).all(-1)
            newly = ok & ~conv
            out[newly] = hard[newly]
            conv |= ok
            if bool(conv.all()):
                break
            v2c = ((total[:, None, :] - c2v) * mask).clamp(-50, 50)
        out[~conv] = hard[~conv]
        return out, conv

    def decode(self, syndromes):
        """syndromes: [B, m] array in {0,1}. Returns (estimates uint8, converged bool) as numpy."""
        s = torch.as_tensor(np.asarray(syndromes), dtype=torch.float32, device=self.device)
        outs, convs = [], []
        for i in range(0, s.shape[0], self.chunk):
            o, c = self._decode_chunk(s[i : i + self.chunk])
            outs.append(o)
            convs.append(c)
        o = torch.cat(outs).to(torch.uint8).cpu().numpy()
        c = torch.cat(convs).cpu().numpy()
        return o, c
