"""Harness regressions use synthetic clocks; no accelerator is required."""
import numpy as np
import pytest
import torch

from qechybrid import bench, pipeline
from qechybrid.bp_gpu import BatchedMinSumBP, DenseMinSumBP


@pytest.mark.parametrize("decoder_type", [BatchedMinSumBP, DenseMinSumBP])
def test_bp_empty_numpy_batch(decoder_type):
    bp = decoder_type(np.array([[1, 1]], np.uint8), [0.1, 0.1])
    estimates, converged = bp.decode(np.empty((0, 1), np.uint8))
    assert estimates.shape == (0, 2) and estimates.dtype == np.uint8
    assert converged.shape == (0,) and converged.dtype == np.bool_


def test_bp_empty_tensor_batch():
    bp = BatchedMinSumBP(np.array([[1, 1]], np.uint8), [0.1, 0.1])
    estimates, converged = bp.decode(torch.empty((0, 1)), to_numpy=False)
    assert estimates.shape == (0, 2) and estimates.dtype == torch.bool
    assert converged.shape == (0,) and converged.dtype == torch.bool
    assert estimates.device == bp.llr0.device == converged.device


@pytest.mark.parametrize("mode", ["none", "zero"])
def test_cpu_pipeline_never_synchronizes_cuda(monkeypatch, mode):
    from qechybrid.data import surface_circuit, sample
    from qechybrid.decoders import MatchingDecoder

    def forbidden_sync(*args, **kwargs):
        pytest.fail("CPU-only decoder synchronized CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_sync)
    circuit = surface_circuit(3, 0.002)
    dets, _ = sample(circuit, 4, seed=3)
    decoder = pipeline.HybridDecoder(MatchingDecoder(circuit), mode=mode, device="cuda")
    decoder.decode_batch(dets)


def test_surface_cpu_baselines_use_cpu_latency_timers(monkeypatch):
    # Exercise the real run_surface call site while replacing accelerator work.
    devices = []

    class FakeLocal:
        def __init__(self, *args, **kwargs):
            pass

    class FakeGate:
        threshold = float("inf")

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def calibrate(self, *args, **kwargs):
            pass

    class FakeDecoder:
        def __init__(self, *args, device, **kwargs):
            self.device = device

        def decode_batch(self, dets):
            self.stats = dict(t_total=1.0, t_gate=0.2, t_match=0.8,
                              accept_rate=0.0, syndrome_weight_kept=1.0)
            return np.zeros((len(dets), 1), np.uint8)

        def decode_one(self, dets):
            return np.zeros(1, np.uint8)

    def latency(*args, device, **kwargs):
        devices.append(device)
        return dict(p50=1.0, p95=1.0, p99=1.0)

    monkeypatch.setattr(bench, "LocalPreDecoder", FakeLocal)
    monkeypatch.setattr(bench, "Gate", FakeGate)
    monkeypatch.setattr(bench, "HybridDecoder", FakeDecoder)
    monkeypatch.setattr(bench, "_latency", latency)
    monkeypatch.setattr(bench, "_batch_latency_ms", lambda *a, **k: 1.0)
    monkeypatch.setattr(bench, "now", lambda device: 1.0)
    cfg = dict(train=4, val=4, shots=4, epochs=1, lat=2, reps=5)
    rows = bench.run_surface(3, 0.002, cfg, "cuda", log=lambda *_: None)
    assert devices == ["cpu", "cpu", "cuda", "cuda", "cuda"]
    assert [r["device"] for r in rows] == devices


def test_repeated_throughput_is_median_and_paired():
    order = []

    class Decoder:
        def __init__(self, name, durations):
            self.name, self.durations = name, iter(durations)

        def decode_batch(self, data):
            order.append(self.name)
            self.stats = dict(t_total=next(self.durations), t_gate=0.1, t_match=0.2,
                              accept_rate=0.0, syndrome_weight_kept=1.0)
            return np.zeros((len(data), 1), np.uint8)

    decoders = [Decoder("cpu", [4.0, 1.0, 2.0]), Decoder("gpu", [2.0, 0.5, 4.0])]
    measured = bench._measure_throughput(
        [(d, "cpu") for d in decoders], np.zeros((12, 1), np.uint8), reps=3, internal=True)
    assert order == ["cpu", "gpu", "gpu", "cpu", "cpu", "gpu"]
    metrics = bench._throughput_metrics(measured[0], 12, measured[0])
    assert metrics["throughput_sps"] == 6.0  # best would be 12
    assert metrics["throughput_min_sps"] == 3.0
    assert metrics["throughput_max_sps"] == 12.0
    assert metrics["throughput_reps"] == 3
    gpu_metrics = bench._throughput_metrics(measured[1], 12, measured[0])
    assert gpu_metrics["paired_ratio_median"] == 2.0
    assert gpu_metrics["paired_ratio_min"] == 0.5


def test_surface_without_nn_never_constructs_gate_or_samples_training(monkeypatch):
    from qechybrid.data import sample

    def forbidden_gate(*args, **kwargs):
        pytest.fail("gate was constructed although nn was disabled")

    sampled = []

    def tracked_sample(circuit, shots, seed):
        sampled.append(seed)
        return sample(circuit, shots, seed)

    monkeypatch.setattr(bench, "Gate", forbidden_gate)
    monkeypatch.setattr(bench, "sample", tracked_sample)
    cfg = dict(shots=8, lat=2, reps=3, modes=["none", "zero", "local"])
    rows = bench.run_surface(3, 0.002, cfg, "cpu", log=lambda *_: None)
    assert sampled == [3]
    assert len(rows) == 4


def test_surface_checkpoints_baseline_before_gate_failure(monkeypatch):
    def failed_gate(*args, **kwargs):
        raise RuntimeError("training allocation failed")

    saved = []
    monkeypatch.setattr(bench, "Gate", failed_gate)
    cfg = dict(train=8, val=8, shots=8, epochs=1, lat=2, reps=3, modes=["none", "nn"])
    with pytest.raises(RuntimeError, match="training allocation failed"):
        bench.run_surface(3, 0.002, cfg, "cpu", log=lambda *_: None, on_row=saved.append)
    assert len(saved) == 1 and saved[0]["decoder"] == "pymatching (CPU)"


def test_throughput_rejects_nondeterministic_predictions():
    class Decoder:
        calls = 0

        def decode_batch(self, data):
            self.calls += 1
            self.stats = dict(t_total=1.0)
            return np.full((len(data), 1), self.calls % 2, np.uint8)

    with pytest.raises(RuntimeError, match="predictions changed"):
        bench._measure_throughput([(Decoder(), "cpu")], np.zeros((2, 1)), reps=3, internal=True)


def test_cuda_timer_synchronizes_the_selected_device(monkeypatch):
    from qechybrid.timing import now

    devices = []
    monkeypatch.setattr(torch.cuda, "synchronize", devices.append)
    now("cpu")
    now("cuda:1")
    assert devices == ["cuda:1"]


def test_empty_hybrid_bp_never_calls_fallback():
    from qechybrid.decoders import HybridBpOsd

    class ForbiddenFallback:
        def decode_batch(self, syndromes):
            pytest.fail("empty batch reached fallback")

    decoder = HybridBpOsd(np.array([[1, 1]], np.uint8), [0.1, 0.1], osd_backend=ForbiddenFallback())
    out = decoder.decode_batch(np.empty((0, 1), np.uint8))
    assert out.shape == (0, 2) and out.dtype == np.uint8
    assert decoder.last_converged_frac == 1.0


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("radius", [1, 2])
def test_local_correction_residual_and_logical_flip_invariant(dense, radius):
    import pymatching
    from qechybrid.local_predecoder import LocalPreDecoder

    graph = pymatching.Matching()
    graph.add_edge(0, 1, weight=1, fault_ids={0})
    graph.add_edge(2, 3, weight=1, fault_ids=set())
    dets = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0]], np.uint8)
    residual, flip = LocalPreDecoder(graph, dense=dense, radius=radius).predecode(dets)
    np.testing.assert_array_equal(residual.numpy(), [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
    np.testing.assert_array_equal(flip.numpy(), [1, 1, 0])


def test_gate_initialization_and_training_are_seeded_and_stream_host_batches(monkeypatch):
    from qechybrid.gate import Gate

    dets = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [1, 0]], np.uint8)
    obs = np.array([[0], [1], [1], [0], [1]], np.uint8)
    original = torch.as_tensor
    host_datasets = []

    def tracked_as_tensor(data, **kwargs):
        if data is dets or data is obs:
            host_datasets.append(kwargs)
        return original(data, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", tracked_as_tensor)
    rng = torch.random.get_rng_state().clone()
    a = Gate(2, hidden=8, seed=19)
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.rand(7)  # unrelated caller activity must not affect model initialization
    b = Gate(2, hidden=8, seed=19)
    a.fit(dets, obs, epochs=2, batch=2, seed=19)
    b.fit(dets, obs, epochs=2, batch=2, seed=19)
    assert len(host_datasets) == 4
    assert all(k["device"] == "cpu" and "dtype" not in k for k in host_datasets)
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA PyTorch unavailable")
def test_cuda_empty_and_reference_paths():
    from qechybrid.codes import bb_72_12_6

    code = bb_72_12_6()
    priors = np.full(code.n, 0.04)
    gpu = BatchedMinSumBP(code.hz, priors, device="cuda")
    out, converged = gpu.decode(np.empty((0, code.hz.shape[0]), np.uint8), to_numpy=False)
    assert out.shape == (0, code.n) and out.device.type == "cuda"
    assert converged.shape == (0,) and converged.device.type == "cuda"
    _, syndromes = code.sample(100, 0.04, seed=19)
    a, ca = gpu.decode(syndromes)
    b, cb = DenseMinSumBP(code.hz, priors).decode(syndromes)
    np.testing.assert_array_equal(ca, cb)
    np.testing.assert_array_equal(a[ca], b[cb])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA PyTorch unavailable")
@pytest.mark.parametrize("radius", [1, 2])
def test_actual_cuda_local_path_matches_cpu_reference(radius):
    from qechybrid.data import sample, surface_circuit
    from qechybrid.decoders import MatchingDecoder
    from qechybrid.local_predecoder import LocalPreDecoder

    matcher = MatchingDecoder(surface_circuit(5, 0.004))
    dets, _ = sample(surface_circuit(5, 0.004), 512, seed=19)
    a, fa = LocalPreDecoder(matcher.m, radius=radius).predecode(dets)
    b, fb = LocalPreDecoder(matcher.m, radius=radius, device="cuda").predecode(dets)
    assert torch.equal(a, b.cpu()) and torch.equal(fa, fb.cpu())
