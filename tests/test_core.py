import numpy as np

from qechybrid.codes import bb_72_12_6, bb_144_12_12
from qechybrid.data import dem_matrices, sample, surface_circuit
from qechybrid.decoders import CpuBpOsd, HybridBpOsd, MatchingDecoder
from qechybrid.gate import Gate
from qechybrid.gf2 import rank, reduce_by_basis, rref
from qechybrid.local_predecoder import LocalPreDecoder
from qechybrid.pipeline import HybridDecoder


def test_gf2_rowspace():
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], np.uint8)
    assert rank(M) == 2
    basis, piv = rref(M)
    assert not reduce_by_basis(basis, piv, M).any()


def test_bb_codes_have_12_logicals():
    for c in (bb_72_12_6(), bb_144_12_12()):
        assert c.k == 12
        assert not ((c.hx.astype(int) @ c.hz.T.astype(int)) % 2).any()


def test_logical_failure_logic():
    c = bb_72_12_6()
    e, s = c.sample(50, 0.02, seed=1)
    assert not c.logical_failure(e, e).any()  # perfect estimate never fails
    stab = c.hx[0][None].repeat(50, 0)  # adding a stabilizer is not a logical error
    assert not c.logical_failure(e, e ^ stab).any()


def test_gpu_bp_matches_cpu_bposd_quality():
    c = bb_72_12_6()
    p = 0.02
    e, s = c.sample(300, p, seed=3)
    pri = np.full(c.n, p)
    ler_cpu = c.logical_failure(e, CpuBpOsd(c.hz, pri).decode_batch(s)).mean()
    h = HybridBpOsd(c.hz, pri, device="cpu")
    est = h.decode_batch(s)
    assert (((est.astype(int) @ c.hz.T.astype(int)) % 2) == s).all()  # always satisfies syndrome
    assert abs(c.logical_failure(e, est).mean() - ler_cpu) < 0.02
    assert h.last_converged_frac > 0.8


def test_dem_matrices_consistent():
    circ = surface_circuit(3, 0.002)
    dm = dem_matrices(circ)
    assert dm.H.shape[0] == circ.num_detectors and dm.L.shape[0] == 1
    assert (dm.priors > 0).all() and (dm.priors < 0.5).all()


def test_hybrid_surface_not_worse_than_matching():
    circ = surface_circuit(5, 0.004)
    m = MatchingDecoder(circ)
    trd, tro = sample(circ, 150000, 1)
    vd, vo = sample(circ, 20000, 2)
    td, to = sample(circ, 20000, 3)
    g = Gate(trd.shape[1]).fit(trd, tro, epochs=6)
    g.calibrate(vd, vo, m.decode_batch(vd))
    base = (m.decode_batch(td)[:, 0] != to[:, 0]).sum()
    hyb = HybridDecoder(m, g, mode="nn")
    errs = (hyb.decode_batch(td)[:, 0] != to[:, 0]).sum()
    assert hyb.stats["accept_rate"] > 0.05
    assert errs <= base * 1.25 + 8
    zero = HybridDecoder(m, mode="zero")
    zero.decode_batch(td)
    assert 0 < zero.stats["accept_rate"] < 1


def test_local_predecoder_is_exact_enough_and_sparsifies():
    circ = surface_circuit(5, 0.002)
    m = MatchingDecoder(circ)
    td, to = sample(circ, 60000, 5)
    base = (m.decode_batch(td)[:, 0] != to[:, 0]).sum()
    h = HybridDecoder(m, mode="local", local=LocalPreDecoder(m.m))
    errs = (h.decode_batch(td)[:, 0] != to[:, 0]).sum()
    assert h.stats["syndrome_weight_kept"] < 0.7  # removes a large share of syndrome weight
    assert 0.2 < h.stats["accept_rate"] <= 1.0
    assert errs <= base * 1.3 + 8
    one = np.array([h.decode_one(td[i])[0] for i in range(200)])
    assert (one == h.decode_batch(td[:200])[:, 0]).all()  # single-shot path == batch path
