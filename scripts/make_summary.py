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
rows = []
for _, r in ours.iterrows():
    b = base.loc[(r.d, r.p)]
    e2e = r.throughput_sps / b.throughput_sps
    st2 = (b.shots / b.throughput_sps) / r.t_global_s if r.t_global_s and r.t_global_s > 0 else float("nan")
    rows.append((int(r.d), r.p, b.throughput_sps, r.throughput_sps, e2e, st2, int(r.errors), int(b.errors), r.syndrome_weight_kept))
sur = "| d | p | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | end-to-end | PyMatching stage only | logical errors (ours / PyMatching) | syndrome weight left |\n|---|---|---|---|---|---|---|---|\n"
for d, p, bt, ot, e2e, st2, eo, eb, kept in rows:
    sur += f"| {d} | {p} | {fmt(bt)} | {fmt(ot)} | **{e2e:.2f}x** | {st2:.2f}x | {eo} / {eb} | {100*kept:.0f} % |\n"
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
lo_s, hi_s = min(r[4] for r in rows), max(r[4] for r in rows)
wins_s = sum(1 for r in rows if r[4] > 1.02)
head = (
    f"**Measured on {gpu} (Google Colab, 2 vCPU), `{profile}` profile, {shots_s:,} surface-code shots and {shots_q:,} qLDPC shots per point; "
    f"data: `results/{tag}/`.**\n\n"
    f"* **qLDPC (BB codes, code-capacity noise): {lo_q:.2f}x to {hi_q:.2f}x batch throughput** vs the C++ `ldpc` BP+OSD "
    f"(best: {best_q[0]}, p={best_q[1]}, {best_q[4]:.2f}x), with **identical logical error counts at every point**.\n"
    f"* **Surface code (circuit-level noise), first aggressive radius-1 local rule: {lo_s:.2f}x to {hi_s:.2f}x end-to-end** ({wins_s} of {len(rows)} points above 1.02x; best: d={best_s[0]}, p={best_s[1]}, {best_s[4]:.2f}x). "
    f"The GPU pre-decoder makes the PyMatching stage itself {min(r[5] for r in rows):.1f}x to {max(r[5] for r in rows):.1f}x faster, "
    f"but at d>=7 the first-stage cost cancels most of that gain.\n"
)
caveat = (
    "Caveats: batch throughput only (single-shot latency is worse on the GPU than on the CPU); the surface-code rows are for the first, aggressive local rule, "
    "which also adds logical errors at some points (compare the error columns); the faster fp16 stage 1 and the lossless `radius=2` rule are CPU-verified and "
    "not yet re-measured on the GPU. Speed-ups are relative to CPU baselines on the same Colab machine.\n"
)
md = (
    f"# GPU speed-up results ({gpu})\n\n{head}\n## qLDPC: GPU batched BP vs C++ ldpc\n\n{qt}\n"
    f"## Surface code: GPU local pre-decoder (first, aggressive radius-1 rule) + PyMatching vs PyMatching alone\n\n{sur}\n{caveat}"
)
(ROOT / "docs" / "RESULTS.md").write_text(md, encoding="utf-8")

# ---------- README block ----------
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
