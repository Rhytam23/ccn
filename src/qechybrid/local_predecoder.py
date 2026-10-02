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
The residual syndrome goes to the exact global decoder. The local rule is a HEURISTIC, not
provably MWPM-equivalent: measured, it costs ~4-25 % extra logical errors (see README), so
always report logical-error counts next to any throughput ratio.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra


class LocalPreDecoder:
    def __init__(self, matching, device: str = "cpu", use_boundary: bool = False, passes: int = 1, strict: bool = True,
                 radius: int = 2, escape: bool = False, dense: bool | None = None):
        self.device, self.use_boundary, self.passes = device, use_boundary, passes
        self.radius, self.escape = radius, escape
        # CUDA: dense fp16 GEMMs (tensor cores; counts are small integers so fp16 is exact).
        # CPU: scipy sparse. dense=True on CPU (fp32) exists to test the GPU code path.
        self.dense = str(device).startswith("cuda") if dense is None else dense
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
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
        if strict:  # all-pairs shortest paths on the matching graph; node N is the boundary
            ii, jj, ww = [], [], []
            for (u, v), (w, _) in pair_w.items():
                ii += [u, v]; jj += [v, u]; ww += [w, w]
            for x in range(N):
                if np.isfinite(bnd_w[x]):
                    ii += [x, N]; jj += [N, x]; ww += [bnd_w[x], bnd_w[x]]
            G = sp.csr_matrix((ww, (ii, jj)), shape=(N + 1, N + 1))
            D = dijkstra(G, directed=False)
        inc: dict[int, list[tuple[float, int]]] = {x: [] for x in range(N)}
        for (u, v), (w, _) in pair_w.items():
            inc[u].append((w, v)); inc[v].append((w, u))

        def esc_min(u, partner):  # cheapest way out of u that is not the u-partner edge
            c = [w for w, y in inc[u] if y != partner] + ([bnd_w[u]] if np.isfinite(bnd_w[u]) else [])
            return min(c) if c else np.inf
        ai, aj, oi, oj, ci, cj = [], [], [], [], [], []
        for (u, v), (w, o) in pair_w.items():
            ci += [u, v]
            cj += [v, u]
            if strict:
                ok_pair = w <= D[u, v] + 1e-9  # direct edge is a shortest path (boundary routes included)
            else:
                ok_pair = w <= bnd_w[u] + bnd_w[v]
            if ok_pair and escape:
                ok_pair = w <= esc_min(u, v) + esc_min(v, u) + 1e-9
            if ok_pair:
                ai += [u, v]
                aj += [v, u]
                if o:
                    oi += [u, v]
                    oj += [v, u]
        self.A_all = self._sp(ci, cj, N)
        A1 = sp.csr_matrix((np.ones(len(ci), np.float32), (ci, cj)), shape=(N, N))
        R2 = ((A1 + sp.identity(N, format="csr")) @ (A1 + sp.identity(N, format="csr"))).tocoo()
        keep = R2.row != R2.col
        self.A_r2 = self._sp(R2.row[keep], R2.col[keep], N)
        self.A_ok = self._sp(ai, aj, N)
        self.A_obs = self._sp(oi, oj, N)
        has_b = np.isfinite(bnd_w)
        self.b_ok = torch.as_tensor(has_b, dtype=self.dtype, device=device)
        self.b_obs = torch.as_tensor(bnd_obs * has_b, dtype=self.dtype, device=device)

    def _sp(self, i, j, N):
        """CPU: scipy CSR (fast). CUDA: dense float matrix (N <= a few thousand, so a plain
        GEMM is robust and fast; avoids relying on torch sparse-CSR kernels)."""
        A = sp.csr_matrix((np.ones(len(i), np.float32), (i, j)), shape=(N, N))
        if self.dense:
            return torch.as_tensor(A.toarray(), dtype=self.dtype, device=self.device)
        return A

    def _mm(self, A, X):  # X: (B, N) -> X @ A (all adjacency matrices are symmetric)
        if sp.issparse(A):
            return torch.from_numpy(np.ascontiguousarray((A @ X.cpu().numpy().T).T))
        return X @ A

    @torch.no_grad()
    def predecode(self, dets):
        """dets: (B,N) uint8. Returns (residual (B,N) uint8 torch, obs_flip (B,) uint8 torch)."""
        if self.dense:  # upload 1 byte/detector, widen on the device
            f = torch.as_tensor(dets, device=self.device).to(self.dtype)
        else:
            f = torch.as_tensor(dets, dtype=torch.float32, device=self.device)
        flip = torch.zeros(f.shape[0], device=self.device, dtype=torch.float32)
        one = torch.ones((), dtype=f.dtype, device=self.device)
        for _ in range(self.passes):  # clearing a pair can expose new isolated pairs
            cnt = self._mm(self.A_all, f)
            iso = f * (cnt == one).to(f.dtype)
            if self.radius >= 2:  # nothing else fired within two hops either (lossless in our tests)
                iso = iso * (self._mm(self.A_r2, f) == one).to(f.dtype)
            mutual = iso * (self._mm(self.A_ok, iso) == one).to(f.dtype)
            flip = flip + (mutual * self._mm(self.A_obs, mutual)).sum(1, dtype=torch.float32) / 2.0
            cleared = mutual
            if self.use_boundary:
                lone = f * (cnt == 0).to(f.dtype) * self.b_ok
                flip = flip + (lone * self.b_obs).sum(1, dtype=torch.float32)
                cleared = cleared + lone
            f = (f - cleared).clamp(min=0)
        return f.to(torch.uint8), (flip.round().long() % 2).to(torch.uint8)
