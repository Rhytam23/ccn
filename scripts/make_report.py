"""Build docs/index.html (self-contained interactive site) + PNG figures from results/*/ CSVs.

    python scripts/make_report.py            # uses every results/<tag>/ folder
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, DOCS = ROOT / "results", ROOT / "docs"


def _clean(df):
    df = df.astype(object).where(df.notna(), None)
    return df.to_dict("records")


def load():
    runs = {}
    for d in sorted(p for p in RES.iterdir() if p.is_dir() and "plumbing" not in p.name and any(p.glob("*.csv"))):
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        item = {"meta": meta}
        for name in ("surface", "qldpc", "ising", "pipeline"):
            f = d / f"{name}.csv"
            item[name] = _clean(pd.read_csv(f)) if f.exists() else []
        runs[d.name] = item
    return runs


def figures(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = DOCS / "figs"
    out.mkdir(parents=True, exist_ok=True)
    for tag, run in runs.items():
        s = pd.DataFrame(run["surface"])
        if s.empty:
            continue
        piv = s.pivot_table(index=["d", "p"], columns="decoder", values="throughput_sps")
        base = piv["pymatching (CPU)"]
        sp = piv.div(base, axis=0).drop(columns="pymatching (CPU)")
        ax = sp.plot(kind="bar", figsize=(10, 4))
        ax.axhline(1, color="k", lw=0.8)
        ax.set_ylabel("throughput ratio vs PyMatching (>1: faster)")
        ax.set_title(f"Surface code throughput ratio [{tag}]")
        plt.tight_layout()
        plt.savefig(out / f"{tag}_surface_speedup.png", dpi=140)
        plt.close()
        # A trade-off view: timing and accuracy must be read together.
        baseline = s[s.decoder == "pymatching (CPU)"].set_index(["d", "p"])
        local = s[s.decoder.str.startswith("GPU local pre-decoder")]
        if local.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.2))
        cases = sorted({(int(r.d), r.p) for _, r in local.iterrows()})
        case_id = {key: i + 1 for i, key in enumerate(cases)}
        for name, group in local.groupby("decoder"):
            radius2 = "r=2" in name
            xs, ys = [], []
            for _, row in group.iterrows():
                ref = baseline.loc[(row.d, row.p)]
                xs.append(row.throughput_sps / ref.throughput_sps)
                ys.append(1e5 * (row.ler - ref.ler))
                ax.annotate(f"[{case_id[(int(row.d), row.p)]}]", (xs[-1], ys[-1]),
                            xytext=(5, 5 if radius2 else -13), textcoords="offset points", fontsize=8)
            ax.scatter(xs, ys, marker="o" if radius2 else "^", s=50,
                       label="conservative r=2" if radius2 else "aggressive r=1")
        ax.axvline(1, color="0.4", lw=0.8, linestyle="--")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xlabel("Batch throughput / PyMatching throughput (>1: faster)")
        ax.set_ylabel("LER difference × 100,000 (ours − PyMatching)")
        protocol = run["meta"].get("benchmark_protocol")
        ax.set_title(f"Surface accuracy / throughput trade-off [{tag}]")
        ax.legend(fontsize=8)
        ax.margins(x=0.1, y=0.15)
        note = ("Median repeated timings; differences are finite-sample observations."
                if protocol == "median-paired-v2" else
                "Historical timings: CPU timer used CUDA sync; corrected GPU rerun required.")
        labels = [f"[{case_id[key]}] d={key[0]}, p={key[1]}" for key in cases]
        case_legend = "\n".join("    ".join(labels[i:i + 4]) for i in range(0, len(labels), 4))
        fig.text(0.5, 0.01, case_legend + "\n" + note + " No equivalence or optimal-frontier claim.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.06 + 0.03 * ((len(labels) + 3) // 4), 1, 1))
        fig.savefig(out / f"{tag}_surface_frontier.png", dpi=140)
        plt.close(fig)


def main():
    runs = load()
    if not runs:
        raise SystemExit("no results found; run scripts/run_benchmarks.py first")
    DOCS.mkdir(exist_ok=True)
    try:
        figures(runs)
    except Exception as exc:  # figures are optional
        print("figure generation skipped:", exc)
    tpl = (DOCS / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(runs))
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print("wrote", DOCS / "index.html", "for runs:", list(runs))


if __name__ == "__main__":
    main()
