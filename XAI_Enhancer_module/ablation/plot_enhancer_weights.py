#!/usr/bin/env python3
"""
Adaptive-Shift Visualisation for XAI-Enhancer layer weights.

Reads the per-image CSV from ``enhancer_weight_extraction.py`` and
produces two publication-quality plots:

* **Plot A (Cross-Dataset):** Average layer weights of ResNet-50
  on IBS vs Kvasir-v2  (grouped bar chart).
* **Plot B (Cross-Architecture):** Average layer weights of VGG-16
  vs ResNet-50 on Kvasir-v2  (grouped bar chart).

Usage:
    python -m XAI_Enhancer_module.ablation.plot_enhancer_weights \
        --input-csv runs/ablation/enhancer_weights.csv \
        --output-dir runs/ablation/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_LAYER_COLS = [
    "Layer_5_Weight",
    "Layer_4_Weight",
    "Layer_3_Weight",
    "Layer_2_Weight",
    "Layer_1_Weight",
]
_LAYER_LABELS = ["-5", "-4", "-3", "-2", "-1"]

_MODEL_DISPLAY = {"vgg16": "VGG-16", "resnet50": "ResNet-50"}
_DATASET_DISPLAY = {"kvasir": "Kvasir-v2", "ibs": "IBS"}


def _apply_theme():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.4,
        rc={
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "font.family": "serif",
        },
    )


def _melt_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Melt wide weight columns into long format for seaborn."""
    id_vars = [c for c in df.columns if c not in _LAYER_COLS]
    long = df.melt(id_vars=id_vars, value_vars=_LAYER_COLS,
                   var_name="Layer", value_name="Weight")
    layer_map = dict(zip(_LAYER_COLS, _LAYER_LABELS))
    long["Layer"] = long["Layer"].map(layer_map)
    return long


# ---------------------------------------------------------------------------
# Plot A – Cross-Dataset (ResNet-50: IBS vs Kvasir)
# ---------------------------------------------------------------------------

def plot_cross_dataset(df: pd.DataFrame, output_dir: Path):
    subset = df[df["Model"].str.lower() == "resnet50"].copy()
    if subset.empty:
        print("WARNING: No ResNet-50 data; skipping Plot A.")
        return

    long = _melt_weights(subset)
    long["Dataset_Display"] = long["Dataset"].map(_DATASET_DISPLAY)

    # Compute mean + std per (dataset, layer)
    agg = (long.groupby(["Dataset_Display", "Layer"])["Weight"]
           .agg(["mean", "std"]).reset_index())

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("Set2", n_colors=2)
    datasets = sorted(agg["Dataset_Display"].unique())
    x = np.arange(len(_LAYER_LABELS))
    bar_width = 0.35

    for i, ds in enumerate(datasets):
        sub = agg[agg["Dataset_Display"] == ds].set_index("Layer").loc[_LAYER_LABELS]
        ax.bar(
            x + i * bar_width, sub["mean"], bar_width,
            yerr=sub["std"], label=ds, color=palette[i],
            capsize=3, edgecolor="0.3", linewidth=0.6,
        )

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(_LAYER_LABELS)
    ax.set_xlabel("Layer Index (from end)")
    ax.set_ylabel("Enhancer Weight")
    ax.set_title("ResNet-50 Enhancer Weights \u2014 Cross-Dataset")
    ax.legend(title="Dataset", frameon=True, fancybox=False, edgecolor="0.7")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"weight_plot_a_cross_dataset.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot B – Cross-Architecture (Kvasir: VGG-16 vs ResNet-50)
# ---------------------------------------------------------------------------

def plot_cross_architecture(df: pd.DataFrame, output_dir: Path):
    subset = df[df["Dataset"].str.lower() == "kvasir"].copy()
    if subset.empty:
        print("WARNING: No Kvasir data; skipping Plot B.")
        return

    long = _melt_weights(subset)
    long["Model_Display"] = long["Model"].map(_MODEL_DISPLAY)

    agg = (long.groupby(["Model_Display", "Layer"])["Weight"]
           .agg(["mean", "std"]).reset_index())

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("deep", n_colors=2)
    models = sorted(agg["Model_Display"].unique())
    x = np.arange(len(_LAYER_LABELS))
    bar_width = 0.35

    for i, mdl in enumerate(models):
        sub = agg[agg["Model_Display"] == mdl].set_index("Layer").loc[_LAYER_LABELS]
        ax.bar(
            x + i * bar_width, sub["mean"], bar_width,
            yerr=sub["std"], label=mdl, color=palette[i],
            capsize=3, edgecolor="0.3", linewidth=0.6,
        )

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(_LAYER_LABELS)
    ax.set_xlabel("Layer Index (from end)")
    ax.set_ylabel("Enhancer Weight")
    ax.set_title("Enhancer Weights \u2014 VGG-16 vs ResNet-50 (Kvasir-v2)")
    ax.legend(title="Architecture", frameon=True, fancybox=False, edgecolor="0.7")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"weight_plot_b_cross_architecture.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise XAI-Enhancer layer weights."
    )
    p.add_argument("--input-csv", default="runs/ablation/enhancer_weights.csv")
    p.add_argument("--output-dir", default="runs/ablation/figures")
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

    print("\nPlot A: Cross-Dataset (ResNet-50)")
    plot_cross_dataset(df, output_dir)

    print("\nPlot B: Cross-Architecture (Kvasir-v2)")
    plot_cross_architecture(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
