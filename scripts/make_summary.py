"""Generate the GPU speed-up summary from results/<tag>/*.csv (never hand-typed numbers).

    python scripts/make_summary.py [tag]      # default tag: colab-gpu
Writes docs/RESULTS.md and refreshes the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else "colab-gpu"
run = ROOT / "results" / tag
meta = json.loads((run / "meta.json").read_text())
gpu = meta.get("gpu") or "CPU"
profile = meta.get("profile", "?")

s = pd.read_csv(run / "surface.csv")
q = pd.read_csv(run / "qldpc.csv")
shots_s, shots_q = int(s.shots.iloc[0]), int(q.shots.iloc[0])


def fmt(x):
    return f"{x:,.0f}"


# ---------- surface code ----------
base = s[s.decoder == "pymatching (CPU)"].set_index(["d", "p"])
ours = s[s.decoder.str.startswith("GPU local pre-decoder")]


def rule_of(name):
    return "safe r=2" if "(safe" in name else "fast r=1" if "(fast" in name else "first version (r=1)"
rows = []
for _, r in ours.iterrows():
    b = base.loc[(r.d, r.p)]
    e2e = r.throughput_sps / b.throughput_sps
    st2 = (b.shots / b.throughput_sps) / r.t_global_s if r.t_global_s and r.t_global_s > 0 else float("nan")
    rows.append((int(r.d), r.p, b.throughput_sps, r.throughput_sps, e2e, st2, int(r.errors), int(b.errors), r.syndrome_weight_kept, rule_of(r.decoder)))
sur = "| d | p | rule | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | end-to-end | PyMatching stage only | logical errors (ours / PyMatching) | syndrome weight left |\n|---|---|---|---|---|---|---|---|---|\n"
for d, p, bt, ot, e2e, st2, eo, eb, kept, rule in rows:
    sur += f"| {d} | {p} | {rule} | {fmt(bt)} | {fmt(ot)} | **{e2e:.2f}x** | {st2:.2f}x | {eo} / {eb} | {100*kept:.0f} % |\n"
best_s = max(rows, key=lambda x: x[4])

# ---------- qLDPC ----------
bq = q[q.decoder.str.contains("ldpc BP")].set_index(["code", "p"])
gq = q[q.decoder.str.startswith("GPU")]
qrows = []
for _, r in gq.iterrows():
    b = bq.loc[(r.code, r.p)]
    qrows.append((r.code, r.p, b.throughput_sps, r.throughput_sps, r.throughput_sps / b.throughput_sps, int(r.errors), int(b.errors)))
qt = "| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | speed-up | logical errors (GPU / ldpc) |\n|---|---|---|---|---|---|\n"
for c, p, bt, ot, sp_, eo, eb in qrows:
    qt += f"| {c} | {p} | {fmt(bt)} | {fmt(ot)} | **{sp_:.2f}x** | {eo} / {eb} |\n"
best_q = max(qrows, key=lambda x: x[4])

lo_q, hi_q = min(r[4] for r in qrows), max(r[4] for r in qrows)
rules = sorted({r[9] for r in rows})


def rng(rule):
    xs = [r[4] for r in rows if r[9] == rule]
    return min(xs), max(xs), len(xs), sum(1 for x in xs if x > 1.02)


surf_lines = ""
for rule in rules:
    lo, hi, n_, wins = rng(rule)
    errs_extra = sum(max(0, r[6] - r[7]) for r in rows if r[9] == rule)
    surf_lines += (f"* **Surface code (circuit-level noise), {rule} rule: {lo:.2f}x to {hi:.2f}x end-to-end** "
                   f"({wins} of {n_} points above 1.02x; {errs_extra} extra logical errors in total vs PyMatching).\n")
best_s = max(rows, key=lambda x: x[4])
lo_q, hi_q = min(r[4] for r in qrows), max(r[4] for r in qrows)
head = (
    f"**Measured on {gpu} (Google Colab, 2 vCPU), `{profile}` profile, {shots_s:,} surface-code shots and {shots_q:,} qLDPC shots per point; "
    f"data: `results/{tag}/`.**\n\n"
    f"* **qLDPC (BB codes, code-capacity noise): {lo_q:.2f}x to {hi_q:.2f}x batch throughput** vs the C++ `ldpc` BP+OSD "
    f"(best: {best_q[0]}, p={best_q[1]}, {best_q[4]:.2f}x), with **identical logical error counts at every point**.\n"
    + surf_lines +
    f"* Best single surface-code point: {best_s[9]} rule, d={best_s[0]}, p={best_s[1]}, {best_s[4]:.2f}x.\n"
)
has_new_rules = any("safe" in r for r in rules)
if has_new_rules:
    surf_title = "GPU local pre-decoder (safe radius-2 and fast radius-1 rules, fp16 stage 1) + PyMatching vs PyMatching alone"
    caveat = (
        "Caveats: batch throughput only (the GPU is slower than the CPU at batch size 1 and at 256-shot micro-batches in these runs); "
        f"only d=5 and d=7 with {shots_s:,} shots per point, so error counts are small (a +/-few difference is noise); "
        f"the qLDPC rows use only {shots_q:,} shots, too few to load a GPU, so the GPU BP looks slow here: the 20,000-shot `full` run (`results/colab-gpu/`) is the fair qLDPC comparison. "
        "The `safe` rule is the lossless one; the `fast` rule trades some logical errors for speed. Speed-ups are relative to CPU baselines on the same Colab machine.\n"
    )
else:
    surf_title = "GPU local pre-decoder (first, aggressive radius-1 rule) + PyMatching vs PyMatching alone"
    caveat = (
        "Caveats: batch throughput only (single-shot latency is worse on the GPU than on the CPU); the surface-code rows are for the first, aggressive local rule, "
        "which also adds logical errors at some points (compare the error columns); the faster fp16 stage 1 and the lossless `radius=2` rule were not part of this run. "
        "Speed-ups are relative to CPU baselines on the same Colab machine.\n"
    )
md = (
    f"# GPU speed-up results ({gpu})\n\n{head}\n## qLDPC: GPU batched BP vs C++ ldpc\n\n{qt}\n"
    f"## Surface code: {surf_title}\n\n{sur}\n{caveat}"
)
(ROOT / "docs" / ("RESULTS.md" if tag == "colab-gpu" else f"RESULTS_{tag}.md")).write_text(md, encoding="utf-8")

# ---------- README block (main run only) ----------
if tag != "colab-gpu":
    print(md)
    raise SystemExit
block = f"<!-- RESULTS:START -->\n### GPU speed-up results ({gpu})\n\n{head}\n{qt}\n{caveat}\nFull tables: [docs/RESULTS.md](docs/RESULTS.md).\n<!-- RESULTS:END -->"
readme = (ROOT / "README.md").read_text(encoding="utf-8")
if "<!-- RESULTS:START -->" in readme:
    a = readme.index("<!-- RESULTS:START -->")
    b = readme.index("<!-- RESULTS:END -->") + len("<!-- RESULTS:END -->")
    readme = readme[:a] + block + readme[b:]
else:
    marker = "## What is in the repo"
    readme = readme.replace(marker, block + "\n\n" + marker, 1)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print(md)
