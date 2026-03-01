#!/usr/bin/env python3
"""
Regularisation Proof: softmax reduces layer-wise variance.

Reads the CSV produced by ``raw_softmax_extraction.py`` and, for every
image, computes the standard deviation across the 5 layer values.
Results are grouped by (Model, Metric_Type) to show that the Softmax
operation consistently compresses the layer-wise spread compared to
the Raw cosine similarities.

Prints a Markdown table and optionally saves to a text file.

Usage:
    python -m XAI_Enhancer_module.ablation.regularization_proof \
        --input-csv runs/ablation/raw_softmax_scores.csv \
        --output-txt runs/ablation/regularization_table.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_LAYER_COLS = [
    "Layer_5_Val",
    "Layer_4_Val",
    "Layer_3_Val",
    "Layer_2_Val",
    "Layer_1_Val",
]

_MODEL_DISPLAY = {
    "vgg16": "VGG-16",
    "vgg19": "VGG-19",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
}


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-image layer std, then average by (Model, Metric_Type)."""
    df = df.copy()
    df["Layer_Std"] = df[_LAYER_COLS].std(axis=1)

    summary = (
        df.groupby(["Model", "Metric_Type"])["Layer_Std"]
        .agg(Avg_Std="mean", Median_Std="median", Max_Std="max")
        .reset_index()
    )

    # Pivot so Raw and Softmax sit side by side per model
    raw = summary[summary["Metric_Type"] == "Raw"].set_index("Model")
    soft = summary[summary["Metric_Type"] == "Softmax"].set_index("Model")

    result_rows = []
    for model in raw.index.intersection(soft.index):
        r_avg = raw.loc[model, "Avg_Std"]
        s_avg = soft.loc[model, "Avg_Std"]
        reduction_pct = ((r_avg - s_avg) / r_avg * 100) if r_avg > 1e-9 else 0.0

        result_rows.append({
            "Model": _MODEL_DISPLAY.get(model, model),
            "Raw Avg Std": round(r_avg, 6),
            "Softmax Avg Std": round(s_avg, 6),
            "Reduction (%)": round(reduction_pct, 2),
            "Raw Median Std": round(raw.loc[model, "Median_Std"], 6),
            "Softmax Median Std": round(soft.loc[model, "Median_Std"], 6),
        })

    return pd.DataFrame(result_rows)


def to_markdown(table: pd.DataFrame) -> str:
    """Render the summary DataFrame as a Markdown table."""
    lines = []
    cols = list(table.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Prove that softmax regularises layer-wise weight variance.",
    )
    p.add_argument("--input-csv", default="runs/ablation/raw_softmax_scores.csv")
    p.add_argument("--output-txt", default=None,
                    help="Optional path to save the Markdown table.")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    table = build_table(df)
    md = to_markdown(table)

    print("\n## Regularisation Proof: Raw vs Softmax Layer-wise Variance\n")
    print(md)

    if args.output_txt:
        out = Path(args.output_txt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "## Regularisation Proof: Raw vs Softmax Layer-wise Variance\n\n" + md + "\n"
        )
        print(f"\nTable saved to {out}")


if __name__ == "__main__":
    main()
