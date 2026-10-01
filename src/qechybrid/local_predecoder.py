"""Local, GPU-vectorised pre-decoder (clique-style) for matching graphs.

Idea (same spirit as NVIDIA's Ising pre-decoder: remove the easy, local part of the
syndrome so the global decoder sees a sparser problem):

  * a fired detector with exactly ONE fired neighbour, where that neighbour also has
    exactly one fired neighbour (a mutually isolated pair), is almost certainly the
    two ends of a single fault -> pair them, record the fault's logical effect, clear both;
  * optionally, a fired detector with NO fired neighbours that touches the boundary is
    matched to the boundary.

A pair is only taken if the direct edge is no more expensive than sending both ends
to the boundary, which is the one case where MWPM would choose differently.
Everything is sparse-matrix products over the whole batch, so it runs on CUDA unchanged.
The residual syndrome goes to the exact global decoder, so correctness is preserved by
construction wherever the local rule agrees with MWPM; calibrate/inspect LER to confirm.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


class LocalPreDecoder:
    def __init__(self, matching, device: str = "cpu", use_boundary: bool = False):
        self.device, self.use_boundary = device, use_boundary
        N = matching.num_detectors
        self.N = N
        pair_w: dict[tuple[int, int], tuple[float, int]] = {}
        bnd_w = np.full(N, np.inf)
        bnd_obs = np.zeros(N)
        for u, v, a in matching.edges():
            o = 1 if 0 in a["fault_ids"] else 0
            w = a["weight"]
            if v is None or u is None:
                x = u if v is None else v
                if w < bnd_w[x]:
                    bnd_w[x], bnd_obs[x] = w, o
            else:
                k = (min(u, v), max(u, v))
                if k not in pair_w or w < pair_w[k][0]:
                    pair_w[k] = (w, o)
        ai, aj, oi, oj, ci, cj = [], [], [], [], [], []
        for (u, v), (w, o) in pair_w.items():
            ci += [u, v]
            cj += [v, u]
            if w <= bnd_w[u] + bnd_w[v]:  # direct pairing no worse than two boundary matches
                ai += [u, v]
                aj += [v, u]
                if o:
                    oi += [u, v]
                    oj += [v, u]
        self.A_all = self._sp(ci, cj, N)
        self.A_ok = self._sp(ai, aj, N)
        self.A_obs = self._sp(oi, oj, N)
        has_b = np.isfinite(bnd_w)
        self.b_ok = torch.as_tensor(has_b, dtype=torch.float32, device=device)
        self.b_obs = torch.as_tensor(bnd_obs * has_b, dtype=torch.float32, device=device)

    def _sp(self, i, j, N):
        """CPU: scipy CSR (fast). CUDA: torch sparse CSR."""
        A = sp.csr_matrix((np.ones(len(i), np.float32), (i, j)), shape=(N, N))
        if str(self.device).startswith("cuda"):
            return torch.sparse_csr_tensor(
                torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                torch.as_tensor(A.data), size=(N, N)).to(self.device)
        return A

    def _mm(self, A, X):  # X: (B, N) -> X @ A^T (A symmetric)
        if sp.issparse(A):
            return torch.from_numpy(np.ascontiguousarray((A @ X.cpu().numpy().T).T))
        return (A @ X.T.contiguous()).T

    @torch.no_grad()
    def predecode(self, dets):
        """dets: (B,N) uint8. Returns (residual (B,N) uint8 torch, obs_flip (B,) uint8 torch)."""
        f = torch.as_tensor(dets, dtype=torch.float32, device=self.device)
        cnt = self._mm(self.A_all, f)
        iso = f * (cnt == 1).float()
        mutual = iso * (self._mm(self.A_ok, iso) == 1).float()
        flip = (mutual * self._mm(self.A_obs, mutual)).sum(1) / 2.0
        cleared = mutual
        if self.use_boundary:
            lone = f * (cnt == 0).float() * self.b_ok
            flip = flip + (lone * self.b_obs).sum(1)
            cleared = cleared + lone
        residual = (f - cleared).clamp(min=0).to(torch.uint8)
        return residual, (flip.round().long() % 2).to(torch.uint8)
