from __future__ import annotations

import time

import numpy as np


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def now(device: str = "cpu") -> float:
    sync(device)
    return time.perf_counter()


def percentiles_us(samples_s) -> dict:
    a = np.asarray(samples_s) * 1e6
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return ph, max(0.0, c - h), min(1.0, c + h)
