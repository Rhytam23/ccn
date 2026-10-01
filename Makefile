PY ?= python

.PHONY: test quick full report notebooks
test:      ; $(PY) -m pytest -q
quick:     ; $(PY) scripts/run_benchmarks.py --profile quick --device auto --tag local
full:      ; $(PY) scripts/run_benchmarks.py --profile full --device auto --tag full
report:    ; $(PY) scripts/make_report.py
notebooks: ; $(PY) scripts/make_notebooks.py
