"""Batched normalized min-sum belief propagation in PyTorch (runs on CUDA or CPU).

`BatchedMinSumBP` (default): edge-list messages, converged shots are dropped from the working
batch as soon as they converge. `DenseMinSumBP`: the original dense (shots x checks x vars)
reference implementation, kept for equivalence tests.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

_BIG = 1e9


class DenseMinSumBP:
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
        if s.shape[0] == 0:
            return np.empty((0, self.n), dtype=np.uint8), np.empty(0, dtype=bool)
        outs, convs = [], []
        for i in range(0, s.shape[0], self.chunk):
            o, c = self._decode_chunk(s[i : i + self.chunk])
            outs.append(o)
            convs.append(c)
        o = torch.cat(outs).to(torch.uint8).cpu().numpy()
        c = torch.cat(convs).cpu().numpy()
        return o, c


class BatchedMinSumBP:
    """Edge-list normalised min-sum BP. Memory/traffic scale with the number of edges, not m*n,
    and the working batch shrinks as shots converge (so one hard shot no longer drags the batch)."""

    def __init__(self, H, priors, max_iter: int = 50, alpha: float = 0.8, device: str = "cpu", chunk: int = 16384):
        Hd = (H.toarray() if sp.issparse(H) else np.asarray(H)) != 0
        self.device, self.max_iter, self.alpha, self.chunk = device, max_iter, alpha, chunk
        m, n = Hd.shape
        self.m, self.n = m, n
        rows = [np.nonzero(Hd[i])[0] for i in range(m)]
        cols = [np.nonzero(Hd[:, j])[0] for j in range(n)]
        dc = max(len(r) for r in rows)
        dv = max(max(len(c) for c in cols), 1)
        self.dc = dc
        cv = np.zeros((m, dc), np.int64)
        cmask = np.zeros((m, dc), bool)
        slot_of = {}
        for i, r in enumerate(rows):
            cv[i, : len(r)] = r
            cmask[i, : len(r)] = True
            for s_, j in enumerate(r):
                slot_of[(i, j)] = i * dc + s_
        dummy = m * dc  # extra always-zero message slot used for padding
        vc = np.full((n, dv), dummy, np.int64)
        for j, c in enumerate(cols):
            for k, i in enumerate(c):
                vc[j, k] = slot_of[(i, j)]
        t = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)
        self.cv_idx, self.cmask, self.vc_edge = t(cv), t(cmask), t(vc)
        self.cv_flat = self.cv_idx.reshape(-1)
        self.cmask_flat = self.cmask.reshape(-1)
        pr = np.clip(np.asarray(priors, dtype=np.float64), 1e-9, 1 - 1e-9)
        self.llr0 = t(np.log((1 - pr) / pr), torch.float32)
        self.slot = t(np.arange(dc))
        # keep the (shots x checks x max-degree) message tensors to a sane size for large, irregular matrices
        self.chunk = int(min(self.chunk, max(64, 4e7 // max(1, m * dc))))

    @torch.no_grad()
    def _decode_chunk(self, syn: torch.Tensor):
        B, m, n, dc = syn.shape[0], self.m, self.n, self.dc
        out = torch.zeros(B, n, dtype=torch.bool, device=self.device)
        conv = torch.zeros(B, dtype=torch.bool, device=self.device)
        act = torch.arange(B, device=self.device)  # original index of each active shot
        sgn_s = 1.0 - 2.0 * syn
        v2c = (self.llr0[self.cv_idx] * self.cmask).expand(B, m, dc).contiguous()
        BIG = 1e9
        for it in range(self.max_iter):
            A = v2c.shape[0]
            neg = (v2c < 0) & self.cmask
            sign_tot = 1.0 - 2.0 * (neg.sum(-1) % 2).float()
            own = torch.where(v2c < 0, -1.0, 1.0)
            absv = v2c.abs().masked_fill(~self.cmask, BIG)
            vals, idx = absv.topk(min(2, dc), dim=-1, largest=False)
            min1, min2 = vals[..., 0], (vals[..., 1] if dc > 1 else vals[..., 0])
            min_excl = torch.where(self.slot[None, None, :] == idx[..., :1], min2[..., None], min1[..., None])
            c2v = self.alpha * sgn_s[..., None] * (sign_tot[..., None] * own) * min_excl * self.cmask
            c2v_flat = torch.cat([c2v.reshape(A, m * dc), torch.zeros(A, 1, device=self.device)], dim=1)
            total = self.llr0 + c2v_flat[:, self.vc_edge].sum(-1)  # (A, n)
            hard = total < 0
            par = (hard[:, self.cv_idx] & self.cmask).sum(-1) % 2  # (A, m)
            ok = (par.float() == (1.0 - sgn_s) / 2.0).all(-1)
            if bool(ok.any()):
                out[act[ok]] = hard[ok]
                conv[act[ok]] = True
                keep = ~ok
                if not bool(keep.any()):
                    return out, conv
                act, sgn_s, hard, total, c2v = act[keep], sgn_s[keep], hard[keep], total[keep], c2v[keep]
            v2c = ((total[:, self.cv_idx] - c2v) * self.cmask).clamp(-50, 50)
        out[act] = hard
        return out, conv

    def decode(self, syndromes, to_numpy: bool = True):
        """syndromes: [B, m] in {0,1} (numpy or torch, any device). Returns (estimates, converged);
        numpy uint8/bool by default, or bool tensors left on the device with to_numpy=False."""
        if torch.is_tensor(syndromes):
            s = syndromes.to(device=self.device, dtype=torch.float32)
        else:
            s = torch.as_tensor(np.asarray(syndromes), dtype=torch.float32, device=self.device)
        if s.shape[0] == 0:
            if to_numpy:
                return np.empty((0, self.n), dtype=np.uint8), np.empty(0, dtype=bool)
            return (torch.empty((0, self.n), dtype=torch.bool, device=self.device),
                    torch.empty(0, dtype=torch.bool, device=self.device))
        outs, convs = [], []
        for i in range(0, s.shape[0], self.chunk):
            o, c = self._decode_chunk(s[i : i + self.chunk])
            outs.append(o)
            convs.append(c)
        o, c = torch.cat(outs), torch.cat(convs)
        return (o.to(torch.uint8).cpu().numpy(), c.cpu().numpy()) if to_numpy else (o, c)
