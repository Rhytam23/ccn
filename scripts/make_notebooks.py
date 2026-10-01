"""Generate the Colab notebooks (run once: python scripts/make_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(exist_ok=True)

SETUP = '''# ---- edit this: your public GitHub repo ----
GITHUB_REPO = "GITHUB_USER/qechybrid"
import os, subprocess, sys
if not os.path.exists("/content/repo"):
    subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{GITHUB_REPO}.git", "/content/repo"], check=True)
%cd /content/repo
!pip -q install -r requirements-colab.txt
!pip -q install -e . --no-deps
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'''


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["accelerator"] = "GPU"
    n.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nbf.write(n, OUT / name)


nb("00_setup_check_colab.ipynb", [
    ("md", "# 00 - Colab setup check\n**Runtime > Change runtime type > T4 GPU** first. Verifies the GPU, installs the package, runs the unit tests and checks whether NVIDIA CUDA-Q QEC is importable."),
    ("code", SETUP),
    ("code", "!pip -q install pytest\n!pytest -q"),
    ("code", '''from qechybrid import cudaq_qec_adapter
print("cudaq_qec:", cudaq_qec_adapter.available())
# Optional (Linux + NVIDIA GPU): !pip -q install cudaq-qec   # then restart runtime and re-run this cell'''),
])

nb("01_run_benchmarks_colab.ipynb", [
    ("md", "# 01 - Full GPU benchmark\nRuns the **full** profile on the Colab GPU (about 20-40 min on a T4; use `--profile quick` for a 5 min smoke test) and rebuilds the website. Download the zip or commit `results/` + `docs/index.html` to GitHub."),
    ("code", SETUP),
    ("code", "!python scripts/run_benchmarks.py --profile full --device cuda --tag colab-gpu"),
    ("code", "!python scripts/make_report.py\n!zip -qr results_colab.zip results docs/index.html docs/figs\nfrom google.colab import files; files.download('results_colab.zip')"),
])

nb("02_cudaq_qec_bposd.ipynb", [
    ("md", "# 02 - qLDPC: CUDA-Q QEC GPU BP+OSD vs our hybrid\nIf `cudaq-qec` imports, the harness adds NVIDIA's `nv-qldpc-decoder` as a third backend automatically. The adapter was written from NVIDIA's public docs and may need small kwarg changes for your installed version (see `src/qechybrid/cudaq_qec_adapter.py`)."),
    ("code", SETUP),
    ("code", "!pip -q install cudaq-qec"),
    ("code", "# restart runtime if the import below fails, then re-run\nfrom qechybrid import cudaq_qec_adapter\nprint(cudaq_qec_adapter.available())"),
    ("code", "!python scripts/run_benchmarks.py --profile full --device cuda --tag colab-qldpc --only qldpc"),
])

nb("03_ising_reproduce.ipynb", [
    ("md", "# 03 - Reproduce NVIDIA's Ising pre-decoder baseline\nDownloads NVIDIA's open weights (NVIDIA Open Model License; do not redistribute them - we only download at run time) and runs **their** inference pipeline (3D-CNN pre-decoder + PyMatching). This gives the reference numbers our own pre-decoder is compared against. Follow the README of https://github.com/NVIDIA/Ising-Decoding if a command below has changed."),
    ("code", '''!git clone --depth 1 https://github.com/NVIDIA/Ising-Decoding.git /content/ising
%cd /content/ising
!pip -q install -r code/requirements_public_inference.txt huggingface_hub
!hf download nvidia/Ising-Decoder-SurfaceCode-1-Fast --local-dir models/
!ls models'''),
    ("code", '''import glob, os
ckpt = glob.glob("/content/ising/models/*fast*fp16.safetensors")[0]
os.environ["PREDECODER_SAFETENSORS_CHECKPOINT"] = ckpt
os.environ["WORKFLOW"] = "inference"
!bash code/scripts/local_run.sh'''),
    ("md", "Copy the reported LER / latency numbers into `results/ising-reference/notes.md` and cite them next to our results."),
])
print("notebooks written to", OUT)
