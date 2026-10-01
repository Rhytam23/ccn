"""qLDPC code construction + code-capacity noise (bivariate bicycle codes)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gf2 import rank, reduce_by_basis, rref


def _shift(n: int, k: int) -> np.ndarray:
    return np.roll(np.eye(n, dtype=np.int64), k, axis=1)


def _poly(terms, x, y) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros((n, n), dtype=np.int64)
    for var, power in terms:
        base = x if var == "x" else y
        out += np.linalg.matrix_power(base, power)
    return (out % 2).astype(np.uint8)


@dataclass
class BBCode:
    """Bivariate bicycle code (Bravyi et al. 2024). X errors are detected by `hz`."""

    name: str
    hx: np.ndarray
    hz: np.ndarray
    n: int
    k: int
    _basis: tuple = field(repr=False, default=None)

    def sample(self, shots: int, p: float, seed: int = 0):
        """Code-capacity bit-flip noise. Returns (errors, syndromes)."""
        rng = np.random.default_rng(seed)
        e = (rng.random((shots, self.n)) < p).astype(np.uint8)
        syn = ((e.astype(np.int32) @ self.hz.T.astype(np.int32)) % 2).astype(np.uint8)
        return e, syn

    def logical_failure(self, errors: np.ndarray, estimates: np.ndarray) -> np.ndarray:
        """True where the residual error is not a stabilizer (or leaves a syndrome)."""
        r = (errors ^ estimates).astype(np.uint8)
        bad_syn = ((r.astype(np.int32) @ self.hz.T.astype(np.int32)) % 2).any(axis=1)
        basis, piv = self._basis
        nonstab = reduce_by_basis(basis, piv, r).any(axis=1)
        return bad_syn | nonstab

    @property
    def priors(self):
        return None


def bb_code(l: int, m: int, a_terms, b_terms, name: str) -> BBCode:
    x = np.kron(_shift(l, 1), np.eye(m, dtype=np.int64))
    y = np.kron(np.eye(l, dtype=np.int64), _shift(m, 1))
    A, B = _poly(a_terms, x, y), _poly(b_terms, x, y)
    hx = np.hstack([A, B]).astype(np.uint8)
    hz = np.hstack([B.T, A.T]).astype(np.uint8)
    assert not ((hx.astype(int) @ hz.T.astype(int)) % 2).any(), "checks must commute"
    n = hx.shape[1]
    k = n - rank(hx) - rank(hz)
    return BBCode(name, hx, hz, n, k, rref(hx))


def bb_72_12_6() -> BBCode:
    return bb_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)], "BB[[72,12,6]]")


def bb_144_12_12() -> BBCode:
    return bb_code(12, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)], "BB[[144,12,12]]")
