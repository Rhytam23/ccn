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
    for d in sorted(p for p in RES.iterdir() if p.is_dir()):
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        item = {"meta": meta}
        for name in ("surface", "qldpc"):
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
        ax.set_ylabel("throughput vs PyMatching (x)")
        ax.set_title(f"Surface code throughput speed-up [{tag}]")
        plt.tight_layout()
        plt.savefig(out / f"{tag}_surface_speedup.png", dpi=140)
        plt.close()


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
