"""Optional adapter for NVIDIA CUDA-Q QEC's GPU BP+OSD decoder ("nv-qldpc-decoder").

UNVERIFIED against every cudaq-qec release: written from the public CUDA-Q QEC docs.
It is only used when `cudaq_qec` imports successfully (e.g. on a Colab GPU runtime);
if the constructor kwargs differ in your installed version, adjust `KWARGS` below.
The benchmark records a clear "unavailable" note instead of failing.
"""
from __future__ import annotations

import numpy as np

KWARGS = dict(use_sparsity=True, max_iterations=50, use_osd=True, osd_method=1, osd_order=7)


def available() -> tuple[bool, str]:
    try:
        import cudaq_qec  # noqa: F401

        return True, getattr(cudaq_qec, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - depends on environment
        return False, f"{type(exc).__name__}: {exc}"


class CudaqQecBpOsd:
    def __init__(self, H: np.ndarray, **overrides):
        import cudaq_qec as qec

        self.decoder = qec.get_decoder("nv-qldpc-decoder", np.asarray(H, dtype=np.uint8), **{**KWARGS, **overrides})
        self.n = H.shape[1]

    def decode_one(self, syn: np.ndarray) -> np.ndarray:
        r = self.decoder.decode([float(x) for x in syn])
        return (np.asarray(r.result) > 0.5).astype(np.uint8)

    def decode_batch(self, S: np.ndarray) -> np.ndarray:
        if hasattr(self.decoder, "decode_batch"):
            res = self.decoder.decode_batch([[float(x) for x in s] for s in S])
            return np.stack([(np.asarray(r.result) > 0.5).astype(np.uint8) for r in res])
        return np.stack([self.decode_one(s) for s in S])
