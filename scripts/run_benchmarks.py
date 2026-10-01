"""CLI: python scripts/run_benchmarks.py --profile quick --device auto --tag local-cpu"""
import argparse

from qechybrid.bench import run_profile

ap = argparse.ArgumentParser()
ap.add_argument("--profile", choices=["quick", "full"], default="quick")
ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
ap.add_argument("--tag", default="local-cpu", help="results/<tag>/ output folder")
ap.add_argument("--only", choices=["surface", "qldpc"], default=None)
a = ap.parse_args()
run_profile(a.profile, a.device, a.tag, which=(a.only,) if a.only else ("surface", "qldpc"))
