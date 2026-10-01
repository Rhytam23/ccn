"""Stim circuit-level surface-code data and detector-error-model matrices."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import stim


def surface_circuit(d: int, p: float, rounds: int | None = None) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds or d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def sample(circuit: stim.Circuit, shots: int, seed: int = 0):
    """Returns (detectors uint8 [shots, n_det], observables uint8 [shots, n_obs])."""
    s = circuit.compile_detector_sampler(seed=seed)
    dets, obs = s.sample(shots, separate_observables=True)
    return dets.astype(np.uint8), obs.astype(np.uint8)


@dataclass
class DemMatrices:
    H: sp.csr_matrix  # detectors x faults
    L: sp.csr_matrix  # observables x faults
    priors: np.ndarray


def dem_matrices(circuit: stim.Circuit) -> DemMatrices:
    """Undecomposed DEM -> check matrix, observable matrix, priors (identical faults merged)."""
    dem = circuit.detector_error_model(decompose_errors=False)
    merged: dict[tuple, float] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets = tuple(sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id()))
        obs = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
        key = (dets, obs)
        q = merged.get(key, 0.0)
        merged[key] = q * (1 - p) + p * (1 - q)
    keys = list(merged)
    rows_h, cols_h, rows_l, cols_l = [], [], [], []
    for j, (dets, obs) in enumerate(keys):
        rows_h += list(dets)
        cols_h += [j] * len(dets)
        rows_l += list(obs)
        cols_l += [j] * len(obs)
    nF = len(keys)
    H = sp.csr_matrix((np.ones(len(rows_h), np.uint8), (rows_h, cols_h)), shape=(dem.num_detectors, nF))
    L = sp.csr_matrix((np.ones(len(rows_l), np.uint8), (rows_l, cols_l)), shape=(dem.num_observables, nF))
    return DemMatrices(H, L, np.array([merged[k] for k in keys]))
