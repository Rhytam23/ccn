"""Head-to-head vs NVIDIA Ising (needs a clone of NVIDIA/Ising-Decoding).

  git clone --depth 1 https://github.com/NVIDIA/Ising-Decoding.git third_party/Ising-Decoding
  python scripts/run_ising_bench.py --repo third_party/Ising-Decoding --download --device cuda --tag colab-ising

--download fetches the gated weights using YOUR Hugging Face credentials (env HF_TOKEN or cached login).
Without --weights/--download the network is random (plumbing/timing check only; rows are labelled).
"""
import argparse
from pathlib import Path

import pandas as pd

from qechybrid.bench import environment_meta, resolve_device
from qechybrid.ising_adapter import build_context, download_weights, run_comparison

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="third_party/Ising-Decoding")
ap.add_argument("--weights", default=None)
ap.add_argument("--download", action="store_true")
ap.add_argument("--model-id", type=int, default=1)
ap.add_argument("--distances", type=int, nargs="+", default=[9, 13])
ap.add_argument("--ps", type=float, nargs="+", default=[0.003, 0.005])
ap.add_argument("--shots", type=int, default=20000)
ap.add_argument("--device", default="auto")
ap.add_argument("--tag", default="ising")
a = ap.parse_args()

device = resolve_device(a.device)
weights = a.weights or (download_weights(a.model_id) if a.download else None)
out = Path("results") / a.tag
out.mkdir(parents=True, exist_ok=True)
import json

meta = environment_meta(device)
meta["ising_weights"] = "trained" if weights else "RANDOM (plumbing check only)"
(out / "meta.json").write_text(json.dumps(meta, indent=2))
rows = []
for d in a.distances:
    for p in a.ps:
        try:
            ctx = build_context(a.repo, distance=d, p=p, shots=a.shots, model_id=a.model_id, weights=weights, device=device)
            rows += run_comparison(ctx)
            pd.DataFrame(rows).to_csv(out / "ising.csv", index=False)
        except Exception:
            import traceback

            print(f"!! case d={d} p={p} failed:")
            traceback.print_exc()
        finally:
            import gc

            ctx = None
            gc.collect()
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
print("wrote", out / "ising.csv")
