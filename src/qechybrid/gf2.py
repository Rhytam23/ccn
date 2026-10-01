"""Small GF(2) linear-algebra helpers (dense, numpy)."""
from __future__ import annotations

import numpy as np


def rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon form over GF(2). Returns (nonzero rows, pivot columns)."""
    M = (np.asarray(M) % 2).astype(np.uint8).copy()
    rows, cols = M.shape
    piv: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        idx = np.nonzero(M[r:, c])[0]
        if idx.size == 0:
            continue
        p = idx[0] + r
        if p != r:
            M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]
        others = others[others != r]
        M[others] ^= M[r]
        piv.append(c)
        r += 1
    return M[:r], piv


def rank(M: np.ndarray) -> int:
    return len(rref(M)[1])


def reduce_by_basis(basis: np.ndarray, pivots: list[int], V: np.ndarray) -> np.ndarray:
    """Reduce each row of V by an RREF basis. Zero result <=> row lies in the row space."""
    V = (np.asarray(V) % 2).astype(np.uint8).copy()
    for row, c in zip(basis, pivots):
        mask = V[:, c].astype(bool)
        if mask.any():
            V[mask] ^= row
    return V
