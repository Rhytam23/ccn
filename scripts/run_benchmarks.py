"""CLI: python scripts/run_benchmarks.py --profile quick --device auto --tag local-cpu"""
import argparse

from qechybrid.bench import run_profile

ap = argparse.ArgumentParser()
ap.add_argument("--profile", choices=["quick", "full"], default="quick")
ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
ap.add_argument("--tag", default="local-cpu", help="results/<tag>/ output folder")
ap.add_argument("--only", choices=["surface", "qldpc"], default=None)
ap.add_argument("--reps", type=int, default=5, help="throughput repetitions (minimum 3; default 5)")
ap.add_argument("--surface-modes", nargs="+", choices=["none", "zero", "local", "nn"], default=None,
                help="surface modes to evaluate; CPU baseline is always included")
a = ap.parse_args()
if a.reps < 3:
    ap.error("--reps must be at least 3")
run_profile(a.profile, a.device, a.tag, which=(a.only,) if a.only else ("surface", "qldpc"),
            reps=a.reps, surface_modes=a.surface_modes)
