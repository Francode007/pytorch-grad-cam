#!/usr/bin/env python3
"""
Regularisation Proof: softmax reduces layer-wise variance.

Reads the long-format CSV produced by ``raw_softmax_extraction.py``
(columns: Dataset, Model, Image_ID, Layer_Index, Raw_Similarity,
Softmax_Weight) and, for every (Image_ID, Model) group, computes the
standard deviation of Raw_Similarity and Softmax_Weight across ALL
layers.

Averages are then grouped by Model and printed as a Markdown table
that quantifies how much the softmax operation reduces the layer-wise
spread.

Usage:
    python -m XAI_Enhancer_module.ablation.regularization_proof \
        --input-csv runs/ablation/raw_softmax_scores.csv \
        --output-txt runs/ablation/regularization_table.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_MODEL_DISPLAY = {
    "vgg16": "VGG-16",
    "vgg19": "VGG-19",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
}


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-image layer std for both metrics, then average by Model."""
    per_image = (
        df.groupby(["Dataset", "Model", "Image_ID"])
        .agg(
            Raw_Std=("Raw_Similarity", "std"),
            Softmax_Std=("Softmax_Weight", "std"),
            Num_Layers=("Layer_Index", "count"),
        )
        .reset_index()
    )

    summary = (
        per_image.groupby("Model")
        .agg(
            Avg_Raw_Std=("Raw_Std", "mean"),
            Median_Raw_Std=("Raw_Std", "median"),
            Avg_Softmax_Std=("Softmax_Std", "mean"),
            Median_Softmax_Std=("Softmax_Std", "median"),
            Avg_Num_Layers=("Num_Layers", "mean"),
        )
        .reset_index()
    )

    result_rows = []
    for _, row in summary.iterrows():
        model = row["Model"]
        r_avg = row["Avg_Raw_Std"]
        s_avg = row["Avg_Softmax_Std"]
        reduction = ((r_avg - s_avg) / r_avg * 100) if r_avg > 1e-9 else 0.0

        result_rows.append({
            "Model": _MODEL_DISPLAY.get(model, model),
            "Layers": int(round(row["Avg_Num_Layers"])),
            "Raw Avg Std": round(r_avg, 6),
            "Softmax Avg Std": round(s_avg, 6),
            "Reduction (%)": round(reduction, 2),
            "Raw Median Std": round(row["Median_Raw_Std"], 6),
            "Softmax Median Std": round(row["Median_Softmax_Std"], 6),
        })

    return pd.DataFrame(result_rows)


def to_markdown(table: pd.DataFrame) -> str:
    """Render the summary DataFrame as a Markdown table."""
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
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
            "## Regularisation Proof: Raw vs Softmax Layer-wise Variance\n\n"
            + md + "\n"
        )
        print(f"\nTable saved to {out}")


if __name__ == "__main__":
    main()
