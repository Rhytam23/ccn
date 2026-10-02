import numpy as np
import pytest

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


@pytest.mark.slow
@pytest.mark.quality_smoke
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


@pytest.mark.slow
@pytest.mark.quality_smoke
def test_hybrid_surface_quality_smoke():
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


@pytest.mark.slow
@pytest.mark.quality_smoke
def test_local_predecoder_quality_smoke_and_sparsifies():
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


def test_ising_adapter_plumbing_if_repo_present():
    """Runs only when NVIDIA/Ising-Decoding is cloned in third_party/ (random weights: plumbing only)."""
    import os

    import pytest

    repo = os.path.join(os.path.dirname(__file__), "..", "third_party", "Ising-Decoding")
    if not os.path.isdir(repo):
        pytest.skip("NVIDIA/Ising-Decoding not cloned")
    from qechybrid.ising_adapter import build_context, run_comparison

    ctx = build_context(repo, distance=9, p=0.005, shots=300, weights=None, device="cpu")
    rows = run_comparison(ctx, lat_shots=5, log=lambda *_: None)
    assert [r["decoder"].split(" [")[0] for r in rows] == ["pymatching (CPU)", "NVIDIA Ising + pymatching", "ours: local pre-decoder + pymatching"]
    assert rows[2]["ler"] < 0.2  # our pre-decoder + MWPM stays sane on NVIDIA's circuit


def test_edge_list_bp_matches_dense_reference():
    from qechybrid.bp_gpu import BatchedMinSumBP, DenseMinSumBP

    c = bb_72_12_6()
    for p in (0.02, 0.05):
        _, s = c.sample(300, p, seed=11)
        pri = np.full(c.n, p)
        en, cn = BatchedMinSumBP(c.hz, pri).decode(s)
        eo, co = DenseMinSumBP(c.hz, pri, chunk=300).decode(s)
        np.testing.assert_array_equal(cn, co)
        np.testing.assert_array_equal(en, eo)
        np.testing.assert_array_equal((en[cn].astype(int) @ c.hz.T.astype(int)) % 2, s[cn])


def test_dense_gpu_code_path_matches_sparse_cpu_path():
    """The CUDA path (dense GEMMs) is exercised on CPU in fp32 and must equal the scipy path."""
    import torch

    circ = surface_circuit(5, 0.004)
    m = MatchingDecoder(circ)
    td, _ = sample(circ, 3000, 9)
    for radius in (1, 2):
        ra, fa = LocalPreDecoder(m.m, radius=radius, dense=False).predecode(td)
        rb, fb = LocalPreDecoder(m.m, radius=radius, dense=True).predecode(td)
        assert torch.equal(ra, rb) and torch.equal(fa, fb)


@pytest.mark.slow
@pytest.mark.quality_smoke
def test_conservative_radius_quality_smoke_and_sparsifies():
    circ = surface_circuit(7, 0.003)
    m = MatchingDecoder(circ)
    td, to = sample(circ, 60000, 4)
    base = int((m.decode_batch(td)[:, 0] != to[:, 0]).sum())
    h = HybridDecoder(m, mode="local", local=LocalPreDecoder(m.m, radius=2))
    errs = int((h.decode_batch(td)[:, 0] != to[:, 0]).sum())
    assert h.stats["syndrome_weight_kept"] < 0.9
    assert errs <= base * 1.1 + 4


@pytest.mark.slow
@pytest.mark.quality_smoke
def test_dem_bp_osd_decoder_is_in_the_same_ballpark_as_mwpm():
    """Circuit-level BP+OSD (GPU-BP path, run on CPU here) must give sane logical predictions."""
    from qechybrid.decoders import DemBpOsd

    circ = surface_circuit(5, 0.004)
    dm = dem_matrices(circ)
    m = MatchingDecoder(circ)
    td, to = sample(circ, 1500, 12)
    dec = DemBpOsd(dm, device="cpu")
    mine = int((dec.decode_batch(td)[:, 0] != to[:, 0]).sum())
    base = int((m.decode_batch(td)[:, 0] != to[:, 0]).sum())
    assert mine <= 4 * base + 15  # BP+OSD-0 is weaker than MWPM but must not be broken
    assert 0.0 <= dec.last["fallback_frac"] <= 1.0


def test_bp_accepts_tensor_input_without_numpy_roundtrip():
    import torch

    from qechybrid.bp_gpu import BatchedMinSumBP

    c = bb_72_12_6()
    _, s = c.sample(200, 0.02, seed=2)
    bp = BatchedMinSumBP(c.hz, np.full(c.n, 0.02))
    a, ca = bp.decode(s)
    b, cb = bp.decode(torch.as_tensor(s), to_numpy=False)
    assert (a == b.to(torch.uint8).numpy()).all() and (ca == cb.numpy()).all()
