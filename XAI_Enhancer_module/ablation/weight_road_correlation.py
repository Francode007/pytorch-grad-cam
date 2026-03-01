#!/usr/bin/env python3
"""
Correlation proof: Spearman rank correlation between per-layer ROAD
scores and the XAI-Enhancer logit-similarity weights.

For each of 100 random Kvasir-v2 validation images (ResNet-50):
  1. Compute ROAD scores for each of the last 5 Conv2d layers.
  2. Extract the Enhancer weights for those same 5 layers.
  3. Compute Spearman ρ between the two 5-element arrays.

Prints and saves the per-image correlations plus the average ρ.

Usage:
    python -m XAI_Enhancer_module.ablation.weight_road_correlation \
        --kvasir-data-root data/kvasir-v2 \
        --checkpoint ./kvasir_runs/resnet50/best.pth \
        --num-images 100 \
        --output-csv runs/ablation/weight_road_corr.csv \
        --device cuda
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from XAI_Enhancer_module.ablation.layerwise_road_extraction import compute_road
from XAI_Enhancer_module.ablation.enhancer_weight_extraction import (
    get_enhancer_weights,
    _SplitDataset,
)
from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.kvasir.data import (
    load_split_file as kvasir_load_split,
    get_val_transforms as kvasir_val_transforms,
    KVASIR_NUM_CLASSES,
)
from XAI_Enhancer_module.utils.model_utils import get_device

NUM_LAYERS = 5
ARCH = "resnet50"


def _get_last_n_conv2d(model: nn.Module, n: int = NUM_LAYERS) -> list[nn.Module]:
    all_conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return all_conv[-n:] if len(all_conv) >= n else all_conv


def compute_per_layer_road(
    image_tensor: torch.Tensor,
    model: nn.Module,
    target_layers: list[nn.Module],
    predicted_label: int,
    device: torch.device,
) -> np.ndarray:
    """Return an array of ROAD scores, one per target layer."""
    img_batch = image_tensor.unsqueeze(0).to(device)
    scores = []
    for layer in target_layers:
        cam = GradCAM(model=model, target_layers=[layer])
        targets = [ClassifierOutputTarget(predicted_label)]
        grayscale = cam(input_tensor=img_batch, targets=targets)  # (1, H, W)
        road = compute_road(image_tensor, grayscale[0], predicted_label, model, device)
        scores.append(road)
        cam.__del__()
    return np.array(scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Spearman correlation between ROAD scores and Enhancer weights.",
    )
    p.add_argument("--kvasir-data-root", default="data/kvasir-v2")
    p.add_argument("--checkpoint", required=True, help="ResNet-50 Kvasir checkpoint.")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-csv", default="runs/ablation/weight_road_corr.csv")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Device: {device}")

    # Load model
    model = build_kvasir_model(ARCH, num_classes=KVASIR_NUM_CLASSES, pretrained=False)
    load_kvasir_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    target_layers = _get_last_n_conv2d(model, NUM_LAYERS)
    print(f"Target layers ({len(target_layers)}): {[str(l)[:60] for l in target_layers]}")

    # Load data and sample
    root = Path(args.kvasir_data_root)
    samples = kvasir_load_split(root / "splits" / "val.txt", root)
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    samples = samples[: args.num_images]
    print(f"Using {len(samples)} images (seed={args.seed})")

    transform = kvasir_val_transforms()

    rows = []
    correlations = []

    for path, label in tqdm(samples, desc="Images"):
        img = transform(open_rgb(path)).to(device)

        road_scores = compute_per_layer_road(img, model, target_layers, label, device)
        weights = get_enhancer_weights(img, model, target_layers, device, label)

        # Spearman ρ (requires at least some variance)
        if np.std(road_scores) > 1e-9 and np.std(weights) > 1e-9:
            rho, pval = stats.spearmanr(road_scores, weights)
        else:
            rho, pval = float("nan"), float("nan")

        correlations.append(rho)
        row = {"image": str(path), "label": label, "spearman_rho": rho, "p_value": pval}
        for i in range(NUM_LAYERS):
            row[f"road_{NUM_LAYERS - i}"] = round(float(road_scores[i]), 6)
            row[f"weight_{NUM_LAYERS - i}"] = round(float(weights[i]), 6)
        rows.append(row)

    # Summary
    valid = [c for c in correlations if not np.isnan(c)]
    avg_rho = float(np.mean(valid)) if valid else float("nan")
    print(f"\n{'='*50}")
    print(f"Average Spearman ρ: {avg_rho:.4f}  (over {len(valid)} valid samples)")
    print(f"{'='*50}")

    # Save
    df = pd.DataFrame(rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Per-image results saved to {out}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def open_rgb(path) -> "PIL.Image.Image":
    from PIL import Image
    return Image.open(path).convert("RGB")


if __name__ == "__main__":
    main()
