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
    return "conservative r=2" if "r=2" in name else "fast r=1" if "(fast" in name else "first version (r=1)"


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
qldpc_bullet = (
    f"* **qLDPC (BB codes, code-capacity noise): GPU/CPU batch-throughput ratio {lo_q:.2f}x to {hi_q:.2f}x** vs the C++ `ldpc` BP+OSD "
    f"(best: {best_q['code']}, p={best_q['p']}, {best_q['ratio']:.2f}x; the GPU is slower at points below 1.00x). "
    f"Logical-error counts {'are identical at every point' if same_errors else 'differ at some points'} on the same shots (not a proof of equivalence; intervals in the tables below).\n"
    f"  *This is batch throughput, not latency: single-shot latency is worse on the GPU than on the CPU, so this is not a real-time result.*\n"
)
head = (
    f"**Measured on {gpu}, `{profile}` profile, {shots_s:,} surface-code shots and {shots_q:,} qLDPC shots per point; data: `results/{tag}/`. Hardware details are recorded below.**\n\n"
    f"{definition}\n\n"
    + qldpc_bullet
    + surf_lines +
    f"* Best single surface-code point: {best_s['rule']} rule, d={best_s['d']}, p={best_s['p']}, ratio {best_s['ratio']:.2f}x.\n"
)

has_new_rules = any("r=2" in r for r in rules)
lat_rows = [r for r in rows if pd.notna(r["g256"]) and pd.notna(r["b256"])]
if lat_rows:
    slower = sum(1 for r in lat_rows if r["g256"] > r["b256"])
    lat_text = (f"single-shot p50 latency was {min(r['g1'] for r in rows):.0f}-{max(r['g1'] for r in rows):.0f} µs on the GPU vs "
                f"{min(r['b1'] for r in rows):.0f}-{max(r['b1'] for r in rows):.0f} µs for PyMatching, and at 256-shot micro-batches the GPU pipeline was slower at "
                f"{slower} of {len(lat_rows)} rows (faster at {len(lat_rows) - slower})")
else:
    lat_text = "single-shot latency is worse on the GPU than on the CPU"
if has_new_rules:
    surf_title = "GPU local pre-decoder (conservative radius-2 and fast radius-1 rules, fp16 stage 1) + PyMatching vs PyMatching alone"
    caveat = (
        f"Caveats: batch throughput only ({lat_text}); "
        f"d={int(s.d.min())} to {int(s.d.max())} with {shots_s:,} shots per point; low error counts do not establish accuracy equivalence. "
        + (f"The qLDPC rows use only {shots_q:,} shots; consult a larger repeated run before interpreting GPU scaling. " if shots_q < 20000 else "") +
        "Both rules are heuristics: inspect the per-point error counts, including any radius-2 degradation. Ratios are relative to CPU baselines on the same machine.\n"
    )
else:
    surf_title = "GPU local pre-decoder (first, aggressive radius-1 rule) + PyMatching vs PyMatching alone"
    caveat = (
        "Caveats: batch throughput only (single-shot latency is worse on the GPU than on the CPU); the surface-code rows are for the first, aggressive local rule, "
        "which also adds logical errors at some points (compare the error columns); the fp16 stage 1 and the `radius=2` rule were not part of this run. "
        "Ratios are relative to CPU baselines on the same Colab machine.\n"
    )


def env_table(m):
    def g(k, note="not recorded in this run"):
        v = m.get(k)
        return note if v in (None, "") else v

    items = [("GPU", g("gpu")), ("GPU memory (GB)", g("gpu_memory_gb")), ("NVIDIA driver", g("nvidia_driver")), ("CUDA (torch build)", g("cuda")), ("Python", g("python")),
             ("PyTorch", g("torch")), ("Stim", g("stim")), ("PyMatching", g("pymatching")), ("ldpc", g("ldpc")), ("CUDA-Q QEC", g("cudaq_qec")),
             ("CPU logical cores", g("cpu_count")), ("Torch threads", g("torch_num_threads")),
             ("Platform", g("platform")), ("Git commit", g("git_commit")), ("Git dirty", g("git_dirty")),
             ("Source SHA256", g("source_sha256")), ("Protocol", g("benchmark_protocol")),
             ("Throughput repetitions", g("throughput_reps")), ("Benchmark date", g("timestamp"))]
    return "| item | value |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in items)


env = env_table(meta)
protocol_note = ("Timings use repeated medians; raw repetitions and paired ratios are in the CSV."
                 if meta.get("benchmark_protocol") == "median-paired-v2" else
                 "Historical timing protocol: surface used best-of-two, qLDPC used one repetition, and CPU-only surface modes synchronized CUDA in GPU runs. Marginal surface ratios need a corrected GPU rerun.")


def repetition_table(df):
    if "paired_ratio_median" not in df:
        return ""
    table = ("\n## Repeated timing dispersion\n\n"
             "Paired ratios use the corresponding CPU repetition; these ranges are timing spread, not confidence intervals.\n\n"
             "| case | decoder | throughput min–max (shots/s) | paired ratio median [min, max] |\n"
             "|---|---|---|---|\n")
    for _, row in df.iterrows():
        if row.decoder in ("pymatching (CPU)", "ldpc BP+OSD (CPU)"):
            continue
        case = f"d={int(row.d)}" if row.code == "surface" else row.code
        table += (f"| {case}, p={row.p} | {row.decoder} | {fmt(row.throughput_min_sps)}–{fmt(row.throughput_max_sps)} | "
                  f"{row.paired_ratio_median:.3f}x [{row.paired_ratio_min:.3f}, {row.paired_ratio_max:.3f}] |\n")
    return table


md = (
    f"# GPU benchmark results ({gpu})\n\n{protocol_note}\n\n{head}\n## qLDPC: GPU batched BP vs C++ ldpc\n\n{qt}\n"
    f"## Surface code: {surf_title}\n\n{sur}\n{caveat}"
    + repetition_table(pd.concat([s, q], ignore_index=True)) +
    f"\n## Environment (from `results/{tag}/meta.json`)\n\n{env}"
)
(ROOT / "docs" / ("RESULTS.md" if tag == "colab-gpu" else f"RESULTS_{tag}.md")).write_text(md, encoding="utf-8")

if tag != "colab-gpu":
    print(md)
    raise SystemExit

# ---------- README block (main run only) ----------
def surface_rows(df):
    base_ = df[df.decoder == "pymatching (CPU)"].set_index(["d", "p"])
    out = []
    for _, r in df[df.decoder.str.startswith("GPU local pre-decoder")].iterrows():
        b = base_.loc[(r.d, r.p)]
        out.append(dict(d=int(r.d), p=r.p, rule=rule_of(r.decoder), ratio=r.throughput_sps / b.throughput_sps, eo=int(r.errors), eb=int(b.errors)))
    return out


def rule_bullets(rs, title):
    out = ""
    order = {"conservative r=2": 0, "fast r=1": 1}
    for rule in sorted({r["rule"] for r in rs}, key=lambda x: order.get(x, 2)):
        rr = [r for r in rs if r["rule"] == rule]
        xs = [r["ratio"] for r in rr]
        extra = sum(max(0, r["eo"] - r["eb"]) for r in rr)
        label = "" if rule.startswith("first version") else f", {rule} rule"
        errs = "no extra logical errors at any point" if extra == 0 else f"{extra} extra logical error{'s' if extra != 1 else ''} in total at the points where ours was worse"
        out += (f"* **Surface code, {title}{label}: throughput ratio {min(xs):.2f}x to {max(xs):.2f}x** "
                f"({sum(1 for x in xs if x > 1.02)} of {len(rr)} points above 1.02x; {errs}).\n")
    return out


quick_dir = ROOT / "results" / "colab-gpu-quick"
cur = ""
cur_note = ""
if quick_dir.exists():
    qm = json.loads((quick_dir / "meta.json").read_text())
    qs = pd.read_csv(quick_dir / "surface.csv")
    cur = rule_bullets(surface_rows(qs), f"historical radius-1/2 code, quick profile (d=5, 7; {int(qs.shots.iloc[0]):,} shots; `results/colab-gpu-quick/`)")
    cur_note = ("Historical surface-code results for both radii come from the quick profile only (d=5 and 7, small error counts); "
                "the older full-profile surface-code result (last bullet above) used the first, aggressive radius-1 rule and a slower stage 1.")
old = rule_bullets(rows, f"older code, full profile (d=5 to 13; {shots_s:,} shots; `results/{tag}/`)")

compact = ("| code | p | C++ ldpc BP+OSD (shots/s) | GPU BP + OSD fallback (shots/s) | throughput ratio | logical errors (GPU / ldpc) |\n|---|---|---|---|---|---|\n" +
           "".join(f"| {r['code']} | {r['p']} | {fmt(r['bt'])} | {fmt(r['ot'])} | **{r['ratio']:.2f}x** | {r['eo']} / {r['eb']} |\n" for r in qrows))
readme_caveat = (
    protocol_note + " Caveats: batch throughput only (the GPU is slower than the CPU for single shots; see `docs/RESULTS.md`). " + cur_note + " "
    "Ratios are relative to CPU baselines on the same Colab machine."
)
block = (
    f"<!-- RESULTS:START -->\n## GPU benchmark results ({gpu})\n\n"
    f"**Measured on {gpu} (Google Colab, 2 vCPU); qLDPC: `{profile}` profile, {shots_q:,} shots per point (`results/{tag}/`).**\n\n{definition}\n\n"
    f"{qldpc_bullet}{cur}{old}\n{compact}\n{readme_caveat}\n\n"
    f"Confidence intervals, surface-code tables and the environment record: [docs/RESULTS.md](docs/RESULTS.md), [docs/RESULTS_colab-gpu-quick.md](docs/RESULTS_colab-gpu-quick.md).\n<!-- RESULTS:END -->"
)
readme = (ROOT / "README.md").read_text(encoding="utf-8")
if "<!-- RESULTS:START -->" in readme:
    a_ = readme.index("<!-- RESULTS:START -->")
    b_ = readme.index("<!-- RESULTS:END -->") + len("<!-- RESULTS:END -->")
    readme = readme[:a_] + block + readme[b_:]
else:
    readme = readme.replace("## Status at a glance", block + "\n\n## Status at a glance", 1)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print(md)
