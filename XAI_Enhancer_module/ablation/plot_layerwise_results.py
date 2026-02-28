#!/usr/bin/env python3
"""
Publication-quality visualisations for the layer-wise ROAD ablation study.

Reads the CSV produced by ``layerwise_road_extraction.py`` and generates:

* **Plot A (Architectural Contrast):** ROAD scores across layer indices
  for all five models on the Kvasir-v2 dataset.
* **Plot B (Cross-Dataset Generalisability):** ROAD scores for ResNet-50
  on both the IBS and Kvasir-v2 datasets.

Outputs are saved as 300 DPI PNG and PDF in the specified directory.

Usage:
    python -m XAI_Enhancer_module.ablation.plot_layerwise_results \
        --input-csv runs/ablation/layerwise_road.csv \
        --output-dir runs/ablation/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Human-readable model names for axis legends
_MODEL_DISPLAY = {
    "vgg16": "VGG-16",
    "vgg19": "VGG-19",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
}

_DATASET_DISPLAY = {
    "kvasir": "Kvasir-v2",
    "ibs": "IBS",
}

# Consistent marker cycle for up to 5 models
_MARKERS = ["o", "s", "D", "^", "v"]


def _apply_theme():
    """Set a clean, publication-ready seaborn/matplotlib theme."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.4,
        rc={
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "font.family": "serif",
        },
    )


def plot_architectural_contrast(df: pd.DataFrame, output_dir: Path):
    """Plot A -- all models, Kvasir-v2 only."""
    subset = df[df["Dataset"].str.lower() == "kvasir"].copy()
    if subset.empty:
        print("WARNING: No Kvasir data found in CSV; skipping Plot A.")
        return

    subset["Model_Display"] = subset["Model"].map(_MODEL_DISPLAY)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    models_present = subset["Model"].unique()
    palette = sns.color_palette("deep", n_colors=len(models_present))

    for i, model in enumerate(sorted(models_present)):
        model_data = subset[subset["Model"] == model].sort_values("Layer_Index")
        ax.plot(
            model_data["Layer_Index"],
            model_data["ROAD_Score"],
            label=_MODEL_DISPLAY.get(model, model),
            marker=_MARKERS[i % len(_MARKERS)],
            color=palette[i],
        )

    ax.set_xlabel("Layer Index (from end)")
    ax.set_ylabel("ROAD Score")
    ax.set_title("Layer-wise ROAD Score \u2014 Kvasir-v2")
    ax.set_xticks(sorted(subset["Layer_Index"].unique()))
    ax.legend(title="Architecture", frameon=True, fancybox=False, edgecolor="0.7")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"plot_a_architectural_contrast.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def plot_cross_dataset(df: pd.DataFrame, output_dir: Path):
    """Plot B -- ResNet-50 only, both datasets."""
    subset = df[df["Model"].str.lower() == "resnet50"].copy()
    if subset.empty:
        print("WARNING: No ResNet-50 data found in CSV; skipping Plot B.")
        return

    subset["Dataset_Display"] = subset["Dataset"].map(_DATASET_DISPLAY)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = sns.color_palette("Set2", n_colors=2)

    for i, ds in enumerate(sorted(subset["Dataset"].unique())):
        ds_data = subset[subset["Dataset"] == ds].sort_values("Layer_Index")
        ax.plot(
            ds_data["Layer_Index"],
            ds_data["ROAD_Score"],
            label=_DATASET_DISPLAY.get(ds, ds),
            marker=_MARKERS[i % len(_MARKERS)],
            color=palette[i],
        )

    ax.set_xlabel("Layer Index (from end)")
    ax.set_ylabel("ROAD Score")
    ax.set_title("ResNet-50 Layer-wise ROAD \u2014 Cross-Dataset")
    ax.set_xticks(sorted(subset["Layer_Index"].unique()))
    ax.legend(title="Dataset", frameon=True, fancybox=False, edgecolor="0.7")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"plot_b_cross_dataset.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate publication-quality plots from layer-wise ROAD CSV."
    )
    p.add_argument(
        "--input-csv", type=str, default="runs/ablation/layerwise_road.csv",
        help="Path to the CSV produced by layerwise_road_extraction.py.",
    )
    p.add_argument(
        "--output-dir", type=str, default="runs/ablation/figures",
        help="Directory to save figures.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_theme()

    print("\nPlot A: Architectural Contrast (Kvasir-v2)")
    plot_architectural_contrast(df, output_dir)

    print("\nPlot B: Cross-Dataset Generalisability (ResNet-50)")
    plot_cross_dataset(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
