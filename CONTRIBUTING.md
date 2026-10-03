# Contributing

Use an editable install; this src-layout package is not importable from a fresh clone
until installed. Install the appropriate PyTorch build first, then the package:

```bash
python -m venv .venv
# Activate .venv (Windows: .venv\Scripts\Activate.ps1; Unix: source .venv/bin/activate).
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
```

For CUDA, select the PyTorch CUDA wheel index for that machine instead of the CPU
index. `pip install .` declares and installs Torch, but does not select a GPU wheel
index for you. Preinstalling the chosen build makes the installation explicit.

`requirements-lock.txt` is a verified Windows x86-64 / CPython 3.13 CPU environment
snapshot, not a cross-platform lock. It excludes the editable project and Torch;
the header records the tested Torch build and the commands to install it. Other
Python/platform combinations should resolve `.[dev]` and run the tests.

```bash
bash scripts/setup.sh              # or scripts/setup.ps1 on Windows; creates .venv, installs, runs tests
pytest -q                          # must pass
python scripts/run_benchmarks.py --profile quick --device cpu --tag local-cpu
python scripts/make_report.py
```

Exact invariants and harness regressions run with `python -m pytest -m "not slow"`.
The slower scientific/quality checks run with `python -m pytest -m slow`; both run
by default and in CI. Frozen small regression fixtures pin error counts as well as
syndrome validity. Broad quality checks remain smoke tests and are labelled as such;
passing them alone does not establish decoder equivalence or scientific significance.

* Keep decoders behind the small interfaces used today: `decode_batch(np.ndarray) -> np.ndarray` and, for latency, `decode_one`.
* Any new decoder needs (1) a correctness test against an existing decoder and (2) a row in `bench.py`.
* Never commit model weights, tokens, or `.env` files. Results CSVs and `meta.json` should be committed so the website is reproducible.
* Quote a throughput ratio only with its hardware, batch size and logical-error counts, and define it (see `docs/METHODOLOGY.md`).
