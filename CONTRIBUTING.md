# Contributing

```bash
bash scripts/setup.sh              # or scripts/setup.ps1 on Windows; creates .venv, installs, runs tests
pytest -q                          # must pass
python scripts/run_benchmarks.py --profile quick --device cpu --tag local-cpu
python scripts/make_report.py
```

* Keep decoders behind the small interfaces used today: `decode_batch(np.ndarray) -> np.ndarray` and, for latency, `decode_one`.
* Any new decoder needs (1) a correctness test against an existing decoder and (2) a row in `bench.py`.
* Never commit model weights, tokens, or `.env` files. Results CSVs and `meta.json` should be committed so the website is reproducible.
* Quote a speed-up only with its hardware, batch size and LER (see `docs/METHODOLOGY.md`).
