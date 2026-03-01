#!/usr/bin/env python3
"""
Architectural Bottleneck Visualisation for raw logit-similarity scores.

Reads the long-format CSV produced by ``raw_softmax_extraction.py`` and
generates two publication-quality line plots using normalised network
depth on the X-axis (0.0 = first Conv2d, 1.0 = last Conv2d) so that
models with different total layer counts can be compared side-by-side.

* **Plot A (Bottleneck Proof):** ResNet-18, ResNet-34, ResNet-50 on IBS.
* **Plot B (Information Diffusion):** VGG-16 vs ResNet-50 on Kvasir-v2.

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

_MODEL_DISPLAY = {
    "vgg16": "VGG-16",
    "vgg19": "VGG-19",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
}

_MARKERS = ["o", "s", "D", "^", "v"]

# Number of bins to discretise normalised depth for aggregation
_N_BINS = 20


def _apply_theme():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.4,
        rc={
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.family": "serif",
        },
    )


def _add_normalised_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``Depth`` column in [0, 1] normalised per (Dataset, Model, Image_ID)."""
    df = df.copy()
    max_idx = df.groupby(["Dataset", "Model"])["Layer_Index"].transform("max")
    df["Depth"] = df["Layer_Index"] / max_idx.clip(lower=1)
    return df


def _bin_depth(df: pd.DataFrame, n_bins: int = _N_BINS) -> pd.DataFrame:
    """Discretise continuous Depth into fixed bins so models align on the same grid."""
    df = df.copy()
    df["Depth_Bin"] = (df["Depth"] * n_bins).round() / n_bins
    return df


# ---------------------------------------------------------------------------
# Plot A -- Bottleneck Proof (ResNets on IBS)
# ---------------------------------------------------------------------------

def plot_bottleneck_proof(df: pd.DataFrame, output_dir: Path):
    subset = df[
        (df["Dataset"].str.lower() == "ibs")
        & (df["Model"].isin(["resnet18", "resnet34", "resnet50"]))
    ]
    if subset.empty:
        print("  WARNING: No IBS ResNet data found; skipping Plot A.")
        return

    subset = _add_normalised_depth(subset)
    subset = _bin_depth(subset)

    agg = (
        subset.groupby(["Model", "Depth_Bin"])["Raw_Similarity"]
        .agg(["mean", "std"]).reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = sns.color_palette("deep", n_colors=3)
    model_order = ["resnet18", "resnet34", "resnet50"]

    for i, model in enumerate(model_order):
        sub = agg[agg["Model"] == model].sort_values("Depth_Bin")
        x = sub["Depth_Bin"].values
        y = sub["mean"].values
        err = sub["std"].values
        ax.plot(x, y, label=_MODEL_DISPLAY[model],
                marker=_MARKERS[i], color=palette[i], markevery=2)
        ax.fill_between(x, y - err, y + err, alpha=0.12, color=palette[i])

    ax.set_xlabel("Normalised Network Depth (0 = first Conv, 1 = last Conv)")
    ax.set_ylabel(r"Raw Cosine Similarity ($\alpha_l$)")
    ax.set_title("Architectural Bottleneck \u2014 ResNets on IBS")
    ax.set_xlim(-0.02, 1.02)
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
    subset = df[
        (df["Dataset"].str.lower() == "kvasir")
        & (df["Model"].isin(["vgg16", "resnet50"]))
    ]
    if subset.empty:
        print("  WARNING: No Kvasir VGG/ResNet data found; skipping Plot B.")
        return

    subset = _add_normalised_depth(subset)
    subset = _bin_depth(subset)

    agg = (
        subset.groupby(["Model", "Depth_Bin"])["Raw_Similarity"]
        .agg(["mean", "std"]).reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = sns.color_palette("Set2", n_colors=2)
    model_order = ["vgg16", "resnet50"]

    for i, model in enumerate(model_order):
        sub = agg[agg["Model"] == model].sort_values("Depth_Bin")
        x = sub["Depth_Bin"].values
        y = sub["mean"].values
        err = sub["std"].values
        ax.plot(x, y, label=_MODEL_DISPLAY[model],
                marker=_MARKERS[i], color=palette[i], markevery=2)
        ax.fill_between(x, y - err, y + err, alpha=0.12, color=palette[i])

    ax.set_xlabel("Normalised Network Depth (0 = first Conv, 1 = last Conv)")
    ax.set_ylabel(r"Raw Cosine Similarity ($\alpha_l$)")
    ax.set_title("Information Diffusion \u2014 VGG-16 vs ResNet-50 (Kvasir-v2)")
    ax.set_xlim(-0.02, 1.02)
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
        description="Visualise raw logit-similarity scores across all layers.",
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
