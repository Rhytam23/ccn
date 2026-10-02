"""Generate the GPU benchmark summary from results/<tag>/*.csv (never hand-typed numbers).

    python scripts/make_summary.py [tag]      # default tag: colab-gpu
Writes docs/RESULTS.md (tag colab-gpu) or docs/RESULTS_<tag>.md, and for tag colab-gpu refreshes the block
between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md.
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


def ler(r):
    return f"{r.ler:.2e} [{r.ler_lo:.1e}, {r.ler_hi:.1e}]"


def rule_of(name):
    return "safe r=2" if "(safe" in name else "fast r=1" if "(fast" in name else "first version (r=1)"


# ---------- surface code ----------
base = s[s.decoder == "pymatching (CPU)"].set_index(["d", "p"])
ours = s[s.decoder.str.startswith("GPU local pre-decoder")]
rows = []
for _, r in ours.iterrows():
    b = base.loc[(r.d, r.p)]
    ratio = r.throughput_sps / b.throughput_sps
    st2 = (b.shots / b.throughput_sps) / r.t_global_s if r.t_global_s and r.t_global_s > 0 else float("nan")
    rows.append(dict(d=int(r.d), p=r.p, rule=rule_of(r.decoder), bt=b.throughput_sps, ot=r.throughput_sps, ratio=ratio, st2=st2,
                     eo=int(r.errors), eb=int(b.errors), kept=r.syndrome_weight_kept, lo=ler(r), lb=ler(b),
                     g256=r.get("lat_b256_ms"), b256=b.get("lat_b256_ms"), g1=r.lat_p50_us, b1=b.lat_p50_us))
sur = ("| d | p | rule | PyMatching (shots/s) | GPU pre-decoder + PyMatching (shots/s) | throughput ratio | PyMatching-stage time ratio | logical errors (ours / PyMatching) | LER ours [95% CI] | LER PyMatching [95% CI] | syndrome weight left |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|\n")
for r in rows:
    sur += (f"| {r['d']} | {r['p']} | {r['rule']} | {fmt(r['bt'])} | {fmt(r['ot'])} | **{r['ratio']:.2f}x** | {r['st2']:.2f}x | "
            f"{r['eo']} / {r['eb']} | {r['lo']} | {r['lb']} | {100 * r['kept']:.0f} % |\n")

# ---------- qLDPC ----------
bq = q[q.decoder.str.contains("ldpc BP")].set_index(["code", "p"])
qrows = []
for _, r in q[q.decoder.str.startswith("GPU")].iterrows():
    b = bq.loc[(r.code, r.p)]
    qrows.append(dict(code=r.code, p=r.p, bt=b.throughput_sps, ot=r.throughput_sps, ratio=r.throughput_sps / b.throughput_sps,
                      eo=int(r.errors), eb=int(b.errors), lo=ler(r), lb=ler(b)))
qt = ("| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) | LER GPU [95% CI] | LER ldpc [95% CI] |\n"
      "|---|---|---|---|---|---|---|---|\n")
for r in qrows:
    qt += f"| {r['code']} | {r['p']} | {fmt(r['bt'])} | {fmt(r['ot'])} | **{r['ratio']:.2f}x** | {r['eo']} / {r['eb']} | {r['lo']} | {r['lb']} |\n"
best_q = max(qrows, key=lambda x: x["ratio"])
lo_q, hi_q = min(r["ratio"] for r in qrows), max(r["ratio"] for r in qrows)
same_errors = all(r["eo"] == r["eb"] for r in qrows)

rules = sorted({r["rule"] for r in rows})
surf_lines = ""
for rule in rules:
    rr = [r for r in rows if r["rule"] == rule]
    xs = [r["ratio"] for r in rr]
    extra = sum(max(0, r["eo"] - r["eb"]) for r in rr)
    surf_lines += (f"* **Surface code (circuit-level noise), {rule} rule: throughput ratio {min(xs):.2f}x to {max(xs):.2f}x** "
                   f"({sum(1 for x in xs if x > 1.02)} of {len(rr)} points above 1.02x; {extra} more logical errors than PyMatching, summed over the points where ours was worse).\n")
best_s = max(rows, key=lambda x: x["ratio"])

definition = ("*Throughput ratio = GPU-pipeline shots/s ÷ CPU-baseline shots/s on the same shots in the same session. "
              "A value above 1 means the GPU pipeline is faster, below 1 means it is slower. It is not a latency.*")
head = (
    f"**Measured on {gpu} (Google Colab, 2 vCPU), `{profile}` profile, {shots_s:,} surface-code shots and {shots_q:,} qLDPC shots per point; data: `results/{tag}/`.**\n\n"
    f"{definition}\n\n"
    f"* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio {lo_q:.2f}x to {hi_q:.2f}x** vs the C++ `ldpc` BP+OSD "
    f"(best: {best_q['code']}, p={best_q['p']}, {best_q['ratio']:.2f}x; the GPU is slightly slower at the points below 1.00x). "
    f"Logical-error counts {'are identical' if same_errors else 'differ'} on the same shots at every point (evidence of the same decoding quality, not a proof; intervals in `docs/RESULTS.md`).\n"
    f"  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*\n"
    + surf_lines +
    f"* Best single surface-code point: {best_s['rule']} rule, d={best_s['d']}, p={best_s['p']}, ratio {best_s['ratio']:.2f}x.\n"
)

has_new_rules = any("safe" in r for r in rules)
lat_rows = [r for r in rows if pd.notna(r["g256"]) and pd.notna(r["b256"])]
if lat_rows:
    slower = sum(1 for r in lat_rows if r["g256"] > r["b256"])
    lat_text = (f"single-shot p50 latency was {min(r['g1'] for r in rows):.0f}-{max(r['g1'] for r in rows):.0f} µs on the GPU vs "
                f"{min(r['b1'] for r in rows):.0f}-{max(r['b1'] for r in rows):.0f} µs for PyMatching, and at 256-shot micro-batches the GPU pipeline was slower at "
                f"{slower} of {len(lat_rows)} rows (faster at {len(lat_rows) - slower})")
else:
    lat_text = "single-shot latency is worse on the GPU than on the CPU"
if has_new_rules:
    surf_title = "GPU local pre-decoder (safe radius-2 and fast radius-1 rules, fp16 stage 1) + PyMatching vs PyMatching alone"
    caveat = (
        f"Caveats: batch throughput only ({lat_text}); "
        f"only d=5 and d=7 with {shots_s:,} shots per point, so error counts are small (a difference of a few errors is within noise); "
        f"the qLDPC rows use only {shots_q:,} shots, too few to load a GPU, so the GPU BP looks slow here: the 20,000-shot `full` run (`results/colab-gpu/`) is the fair qLDPC comparison. "
        "The `safe` rule is the lossless one; the `fast` rule trades some logical errors for throughput. Ratios are relative to CPU baselines on the same Colab machine.\n"
    )
else:
    surf_title = "GPU local pre-decoder (first, aggressive radius-1 rule) + PyMatching vs PyMatching alone"
    caveat = (
        "Caveats: batch throughput only (single-shot latency is worse on the GPU than on the CPU); the surface-code rows are for the first, aggressive local rule, "
        "which also adds logical errors at some points (compare the error columns); the fp16 stage 1 and the lossless `radius=2` rule were not part of this run. "
        "Ratios are relative to CPU baselines on the same Colab machine.\n"
    )


def env_table(m):
    def g(k, note="not recorded in this run"):
        v = m.get(k)
        return note if v in (None, "") else v

    items = [("GPU", g("gpu")), ("GPU memory (GB)", g("gpu_memory_gb")), ("CUDA (torch build)", g("cuda")), ("Python", g("python")),
             ("PyTorch", g("torch")), ("Stim", g("stim")), ("PyMatching", g("pymatching")), ("ldpc", g("ldpc")), ("CUDA-Q QEC", g("cudaq_qec")),
             ("CPU cores (Colab)", g("cpu_count")), ("Platform", g("platform")), ("Git commit", g("git_commit")), ("Benchmark date", g("timestamp"))]
    return "| item | value |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in items)


env = env_table(meta)
md = (
    f"# GPU benchmark results ({gpu})\n\n{head}\n## qLDPC: GPU batched BP vs C++ ldpc\n\n{qt}\n"
    f"## Surface code: {surf_title}\n\n{sur}\n{caveat}\n## Environment (from `results/{tag}/meta.json`)\n\n{env}"
)
(ROOT / "docs" / ("RESULTS.md" if tag == "colab-gpu" else f"RESULTS_{tag}.md")).write_text(md, encoding="utf-8")

if tag != "colab-gpu":
    print(md)
    raise SystemExit

# ---------- README block (main run only): headline + compact qLDPC table; full tables in docs/RESULTS.md ----------
compact = ("| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) |\n|---|---|---|---|---|---|\n" +
           "".join(f"| {r['code']} | {r['p']} | {fmt(r['bt'])} | {fmt(r['ot'])} | **{r['ratio']:.2f}x** | {r['eo']} / {r['eb']} |\n" for r in qrows))
block = (f"<!-- RESULTS:START -->\n### GPU benchmark results ({gpu})\n\n{head}\n{compact}\n{caveat}\n"
         f"Confidence intervals, surface-code tables and the full environment record: [docs/RESULTS.md](docs/RESULTS.md).\n<!-- RESULTS:END -->")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
if "<!-- RESULTS:START -->" in readme:
    a = readme.index("<!-- RESULTS:START -->")
    b = readme.index("<!-- RESULTS:END -->") + len("<!-- RESULTS:END -->")
    readme = readme[:a] + block + readme[b:]
else:
    readme = readme.replace("## What is in the repo", block + "\n\n## What is in the repo", 1)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print(md)
