#!/usr/bin/env bash
# Linux/macOS/WSL: python3 -m venv + install + tests.
set -euo pipefail
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
