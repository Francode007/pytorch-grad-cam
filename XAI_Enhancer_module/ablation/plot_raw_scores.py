#!/usr/bin/env python3
"""
Architectural Bottleneck Visualisation for raw logit-similarity scores.

Reads the CSV produced by ``raw_softmax_extraction.py`` (Metric_Type == 'Raw')
and generates two publication-quality line plots:

* **Plot A (Bottleneck Proof):** Average raw scores of ResNet-18,
  ResNet-34, and ResNet-50 on the IBS dataset.
* **Plot B (Information Diffusion):** Average raw scores of VGG-16
  vs ResNet-50 on the Kvasir-v2 dataset.

Usage:
    python -m XAI_Enhancer_module.ablation.plot_raw_scores \
        --input-csv runs/ablation/raw_softmax_scores.csv \
        --output-dir runs/ablation/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_LAYER_COLS = [
    "Layer_5_Val",
    "Layer_4_Val",
    "Layer_3_Val",
    "Layer_2_Val",
    "Layer_1_Val",
]
_LAYER_LABELS = ["-5", "-4", "-3", "-2", "-1"]

_MODEL_DISPLAY = {
    "vgg16": "VGG-16",
    "vgg19": "VGG-19",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
}
_DATASET_DISPLAY = {"kvasir": "Kvasir-v2", "ibs": "IBS"}

_MARKERS = ["o", "s", "D", "^", "v"]


def _apply_theme():
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


def _raw_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Metric_Type"] == "Raw"].copy()


def _melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide layer columns to long format."""
    id_vars = [c for c in df.columns if c not in _LAYER_COLS]
    long = df.melt(id_vars=id_vars, value_vars=_LAYER_COLS,
                   var_name="Layer", value_name="Score")
    long["Layer"] = long["Layer"].map(dict(zip(_LAYER_COLS, _LAYER_LABELS)))
    return long


# ---------------------------------------------------------------------------
# Plot A -- Bottleneck Proof (ResNets on IBS)
# ---------------------------------------------------------------------------

def plot_bottleneck_proof(df: pd.DataFrame, output_dir: Path):
    raw = _raw_subset(df)
    subset = raw[
        (raw["Dataset"].str.lower() == "ibs")
        & (raw["Model"].isin(["resnet18", "resnet34", "resnet50"]))
    ]
    if subset.empty:
        print("WARNING: No IBS ResNet data found; skipping Plot A.")
        return

    long = _melt_to_long(subset)
    agg = long.groupby(["Model", "Layer"])["Score"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = sns.color_palette("deep", n_colors=3)
    model_order = ["resnet18", "resnet34", "resnet50"]

    for i, model in enumerate(model_order):
        sub = agg[agg["Model"] == model].set_index("Layer").loc[_LAYER_LABELS]
        ax.plot(
            _LAYER_LABELS, sub["mean"],
            label=_MODEL_DISPLAY[model],
            marker=_MARKERS[i],
            color=palette[i],
        )
        ax.fill_between(
            _LAYER_LABELS,
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            alpha=0.15, color=palette[i],
        )

    ax.set_xlabel("Layer Index (from end)")
    ax.set_ylabel(r"Raw Cosine Similarity ($\alpha_l$)")
    ax.set_title("Architectural Bottleneck \u2014 ResNets on IBS")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"raw_plot_a_bottleneck_proof.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot B -- Information Diffusion (VGG-16 vs ResNet-50 on Kvasir)
# ---------------------------------------------------------------------------

def plot_information_diffusion(df: pd.DataFrame, output_dir: Path):
    raw = _raw_subset(df)
    subset = raw[
        (raw["Dataset"].str.lower() == "kvasir")
        & (raw["Model"].isin(["vgg16", "resnet50"]))
    ]
    if subset.empty:
        print("WARNING: No Kvasir VGG/ResNet data found; skipping Plot B.")
        return

    long = _melt_to_long(subset)
    agg = long.groupby(["Model", "Layer"])["Score"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = sns.color_palette("Set2", n_colors=2)
    model_order = ["vgg16", "resnet50"]

    for i, model in enumerate(model_order):
        sub = agg[agg["Model"] == model].set_index("Layer").loc[_LAYER_LABELS]
        ax.plot(
            _LAYER_LABELS, sub["mean"],
            label=_MODEL_DISPLAY[model],
            marker=_MARKERS[i],
            color=palette[i],
        )
        ax.fill_between(
            _LAYER_LABELS,
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            alpha=0.15, color=palette[i],
        )

    ax.set_xlabel("Layer Index (from end)")
    ax.set_ylabel(r"Raw Cosine Similarity ($\alpha_l$)")
    ax.set_title("Information Diffusion \u2014 VGG-16 vs ResNet-50 (Kvasir-v2)")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"raw_plot_b_information_diffusion.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise raw logit-similarity scores across layers.",
    )
    p.add_argument("--input-csv", default="runs/ablation/raw_softmax_scores.csv")
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

    print("\nPlot A: Bottleneck Proof (ResNets on IBS)")
    plot_bottleneck_proof(df, output_dir)

    print("\nPlot B: Information Diffusion (VGG-16 vs ResNet-50 on Kvasir)")
    plot_information_diffusion(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
