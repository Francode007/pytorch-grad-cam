"""
Phase 4 statistics over per-image CAM CSVs (Tier 2.2).

Computes mean ± SD, bootstrap 95% CIs, paired Wilcoxon (Holm), Cliff's δ,
and a win/tie/loss table. CPU-only — no GPU required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Project root on path when run as module
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from XAI_Enhancer_module.common.resource_monitor import ResourceMonitor

METRICS = ("ins", "del", "road")
# Higher is better for these conventions in our logs
HIGHER_IS_BETTER = {"ins": True, "del": False, "road": True}


def _discover_csvs(input_dir: Path) -> List[Path]:
    paths = sorted(input_dir.rglob("per_image/*.csv"))
    if not paths:
        paths = sorted(input_dir.rglob("*.csv"))
        paths = [p for p in paths if p.name != "comparison_report.csv"]
    return paths


def _load_frames(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skip {p}: {e}")
            continue
        if "method" not in df.columns:
            continue
        # Infer run context from path: .../{arch}/seedN|foldN/cam_eval.../per_image/
        parts = p.parts
        arch = ""
        run_id = ""
        for i, part in enumerate(parts):
            if part in ("kvasir", "ibs") and i + 1 < len(parts):
                arch = parts[i + 1]
            if part.startswith("seed") or part.startswith("fold"):
                run_id = part
        df = df.copy()
        df["source_csv"] = str(p)
        df["arch"] = arch
        df["run_id"] = run_id
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No per-image CSVs under {input_dir}")
    return pd.concat(frames, ignore_index=True)


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return mean, lo, hi


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size of x vs y (positive ⇒ x tends larger than y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Prefer paired if same length
    if len(x) == len(y) and len(x) > 0:
        d = x - y
        pos = np.sum(d > 0)
        neg = np.sum(d < 0)
        n = len(d)
        return float((pos - neg) / n) if n else float("nan")
    # Unpaired fallback
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (nx * ny))


def _holm(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [1.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * float(pvals[idx]))
        adj = max(adj, prev)
        adjusted[int(idx)] = adj
        prev = adj
    return adjusted


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 5 or np.allclose(diff, 0):
        return 1.0
    try:
        res = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return 1.0


def summarize_methods(
    df: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    baseline_methods: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (summary, pairwise, win_tie_loss).

    Pairwise compares each method to each baseline on shared image_ids.
    """
    baselines = list(baseline_methods or ["GradCAM", "Uniform (T→∞)", "LayerCAM", "HR-CAM"])
    present = sorted(df["method"].unique())
    baselines = [b for b in baselines if b in present]

    summary_rows = []
    for (arch, method), g in df.groupby(["arch", "method"], dropna=False):
        row = {"arch": arch, "method": method, "n": int(len(g))}
        for metric in METRICS:
            if metric not in g.columns:
                continue
            vals = g[metric].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["arch", "method"])

    pair_rows = []
    wtl_rows = []
    for arch, adf in df.groupby("arch"):
        methods = sorted(adf["method"].unique())
        for metric in METRICS:
            if metric not in adf.columns:
                continue
            for baseline in baselines:
                base = adf[adf["method"] == baseline][["image_id", metric]].dropna()
                if base.empty:
                    continue
                comps = []
                for method in methods:
                    if method == baseline:
                        continue
                    other = adf[adf["method"] == method][["image_id", metric]].dropna()
                    merged = base.merge(other, on="image_id", suffixes=("_b", "_m"))
                    if len(merged) < 5:
                        continue
                    xb = merged[f"{metric}_b"].to_numpy()
                    xm = merged[f"{metric}_m"].to_numpy()
                    p = _wilcoxon_p(xm, xb)
                    delta = _cliffs_delta(xm, xb)
                    mean_diff = float(np.mean(xm - xb))
                    comps.append(
                        {
                            "arch": arch,
                            "metric": metric,
                            "method": method,
                            "baseline": baseline,
                            "n_paired": int(len(merged)),
                            "mean_diff": mean_diff,
                            "p_raw": p,
                            "cliffs_delta": delta,
                        }
                    )
                if not comps:
                    continue
                p_adj = _holm([c["p_raw"] for c in comps])
                for c, padj in zip(comps, p_adj):
                    c["p_holm"] = padj
                    # Win if significantly better under metric direction
                    better = (c["mean_diff"] > 0) if HIGHER_IS_BETTER[metric] else (c["mean_diff"] < 0)
                    if padj >= 0.05:
                        outcome = "tie"
                    elif better:
                        outcome = "win"
                    else:
                        outcome = "loss"
                    c["outcome"] = outcome
                    pair_rows.append(c)
                    wtl_rows.append(
                        {
                            "arch": arch,
                            "metric": metric,
                            "method": c["method"],
                            "baseline": baseline,
                            "outcome": outcome,
                            "p_holm": padj,
                            "cliffs_delta": c["cliffs_delta"],
                        }
                    )

    pairwise = pd.DataFrame(pair_rows)
    wtl = pd.DataFrame(wtl_rows)
    if not wtl.empty:
        # Aggregate win/tie/loss counts per method vs each baseline
        agg = (
            wtl.groupby(["arch", "method", "baseline", "outcome"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ("win", "tie", "loss"):
            if col not in agg.columns:
                agg[col] = 0
        wtl_summary = agg
    else:
        wtl_summary = pd.DataFrame(columns=["arch", "method", "baseline", "win", "tie", "loss"])

    return summary, pairwise, wtl_summary


def to_latex_table1(summary: pd.DataFrame, pairwise: pd.DataFrame) -> str:
    """Simple LaTeX mean±SD table; bold where enhanced beats GradCAM (p_holm<0.05)."""
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Arch & Method & Ins ↑ & Del ↓ & ROAD ↑ \\",
        r"\midrule",
    ]
    sig = set()
    if pairwise is not None and not pairwise.empty:
        hit = pairwise[
            (pairwise["baseline"] == "GradCAM")
            & (pairwise["p_holm"] < 0.05)
            & (pairwise["outcome"] == "win")
        ]
        for _, r in hit.iterrows():
            sig.add((r["arch"], r["method"], r["metric"]))

    def fmt(arch, method, metric, mean, std):
        s = f"{mean:.3f} ± {std:.3f}"
        if (arch, method, metric) in sig:
            return r"\textbf{" + s + "}"
        return s

    for _, r in summary.iterrows():
        lines.append(
            f"{r['arch']} & {r['method']} & "
            f"{fmt(r['arch'], r['method'], 'ins', r.get('ins_mean', float('nan')), r.get('ins_std', float('nan')))} & "
            f"{fmt(r['arch'], r['method'], 'del', r.get('del_mean', float('nan')), r.get('del_std', float('nan')))} & "
            f"{fmt(r['arch'], r['method'], 'road', r.get('road_mean', float('nan')), r.get('road_std', float('nan')))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def run_stats(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ResourceMonitor(label="stats") as mon:
        paths = _discover_csvs(input_dir)
        print(f"Found {len(paths)} per-image CSV(s) under {input_dir}")
        df = _load_frames(paths)
        summary, pairwise, wtl = summarize_methods(df, n_boot=n_boot, seed=seed)
        mon.sample()

        summary_path = output_dir / "summary.csv"
        pairwise_path = output_dir / "pairwise.csv"
        wtl_path = output_dir / "win_tie_loss.csv"
        latex_path = output_dir / "table1.tex"
        summary.to_csv(summary_path, index=False)
        pairwise.to_csv(pairwise_path, index=False)
        wtl.to_csv(wtl_path, index=False)
        latex_path.write_text(to_latex_table1(summary, pairwise))
        mon.write(output_dir / "resources.json")

    meta = {
        "n_csvs": len(paths),
        "n_rows": int(len(df)),
        "n_boot": n_boot,
        "seed": seed,
        "resources": mon.report.to_dict(),
        "outputs": {
            "summary": str(summary_path),
            "pairwise": str(pairwise_path),
            "win_tie_loss": str(wtl_path),
            "table1_tex": str(latex_path),
        },
    }
    with open(output_dir / "stats_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Stats written to {output_dir}")
    return meta


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Bootstrap / Wilcoxon stats over per-image CAM CSVs")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    run_stats(args.input_dir, args.output_dir, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
