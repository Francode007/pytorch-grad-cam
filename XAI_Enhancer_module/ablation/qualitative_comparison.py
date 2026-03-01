#!/usr/bin/env python3
"""
Qualitative Visual Comparison of XAI Methods.

Generates a 2x4 publication-quality figure comparing three saliency-map
methods on one IBS image (Row 1) and one Kvasir-v2 image (Row 2):

    Col 1: Original   Col 2: Base HiResCAM   Col 3: HR-CAM   Col 4: Enhanced HiResCAM (Ours)

Usage:
    python -m XAI_Enhancer_module.ablation.qualitative_comparison \
        --img-ibs  data/IBS-preprocessed-dataset/IBS/0.jpg \
        --img-kvasir data/kvasir-v2/polyps/polyps/00072d5f-7cd8-434c-8a5a-1a0bb2c9711d.jpg \
        --ckpt-ibs  ./ibs_runs/resnet50/best.pth \
        --ckpt-kvasir ./kvasir_runs/resnet50/best.pth \
        --output qualitative_comparison.pdf \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2
from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.kvasir.data import (
    get_val_transforms as kvasir_val_transforms,
    KVASIR_NUM_CLASSES,
)
from XAI_Enhancer_module.ibs.models import build_ibs_model, load_ibs_checkpoint
from XAI_Enhancer_module.ibs.data import (
    get_val_transforms as ibs_val_transforms,
    IBS_NUM_CLASSES,
)
from XAI_Enhancer_module.utils.model_utils import get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IBS_MEAN = [0.6380, 0.3422, 0.2275]
IBS_STD = [0.2448, 0.2060, 0.1710]

ARCH = "resnet50"
NUM_HRCAM_LAYERS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _denormalize(tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """Reverse normalisation to get a [0, 1] float RGB image for overlay."""
    img = tensor.cpu().clone()
    for c, m, s in zip(img, mean, std):
        c.mul_(s).add_(m)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _get_all_conv2d(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def _load_and_preprocess(image_path: str, transform) -> tuple:
    """Return (tensor [C,H,W], PIL Image)."""
    pil_img = Image.open(image_path).convert("RGB")
    tensor = transform(pil_img)
    return tensor, pil_img


def _predict(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
    return logits.argmax(dim=1).item()


# ---------------------------------------------------------------------------
# Heatmap generators
# ---------------------------------------------------------------------------

def generate_base_hirescam(
    model: nn.Module, tensor: torch.Tensor, label: int, device: torch.device,
) -> np.ndarray:
    """Single-layer HiResCAM on the final conv layer → (H, W) grayscale map."""
    target_layer = [model.layer4[-1]]
    cam = HiResCAM(model=model, target_layers=target_layer)
    targets = [ClassifierOutputTarget(label)]
    grayscale = cam(input_tensor=tensor.unsqueeze(0).to(device), targets=targets)
    return grayscale[0]  # (H, W)


def generate_hrcam(
    model: nn.Module, tensor: torch.Tensor, label: int, device: torch.device,
) -> np.ndarray:
    """Multi-layer HiResCAM (uniform mean) on last N conv layers → (H, W)."""
    all_conv = _get_all_conv2d(model)
    target_layers = all_conv[-NUM_HRCAM_LAYERS:]
    cam = HiResCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(label)]
    grayscale = cam(input_tensor=tensor.unsqueeze(0).to(device), targets=targets)
    return grayscale[0]


def generate_enhanced_hirescam(
    model: nn.Module, model_name: str, tensor: torch.Tensor,
    label: int, device: torch.device,
) -> np.ndarray:
    """XAI-Enhancer weighted aggregation → (H, W) grayscale map."""
    all_conv = _get_all_conv2d(model)
    target_layers = all_conv[-NUM_HRCAM_LAYERS:]
    extractor = EnhancedExtractorV2(
        model=model,
        model_name=model_name,
        conv_layers=target_layers,
        cam_method="HiResCAMEnhanced",
        aggregation_config={"type": "standard"},
    )
    _, saliency = extractor.extract_saliency_map(
        tensor.unsqueeze(0).to(device), label,
    )
    cam_np = saliency[0].cpu().numpy()
    cam_np = cam_np - cam_np.min()
    denom = cam_np.max()
    if denom > 1e-8:
        cam_np = cam_np / denom
    return cam_np


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def _overlay(rgb_01: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a grayscale CAM on an RGB [0,1] image using jet colormap."""
    return show_cam_on_image(rgb_01, cam, use_rgb=True, image_weight=alpha)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate qualitative comparison figure.",
    )
    p.add_argument("--img-ibs", required=True, help="Path to IBS image")
    p.add_argument("--img-kvasir", required=True, help="Path to Kvasir-v2 image")
    p.add_argument("--ckpt-ibs", required=True, help="ResNet-50 IBS checkpoint")
    p.add_argument("--ckpt-kvasir", required=True, help="ResNet-50 Kvasir checkpoint")
    p.add_argument("--output", default="qualitative_comparison.pdf")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Device: {device}")

    # --- Load models ---
    print("Loading IBS ResNet-50...")
    model_ibs = build_ibs_model(ARCH, num_classes=IBS_NUM_CLASSES, pretrained=False)
    load_ibs_checkpoint(model_ibs, args.ckpt_ibs, device)
    model_ibs = model_ibs.to(device).eval()

    print("Loading Kvasir ResNet-50...")
    model_kvasir = build_kvasir_model(ARCH, num_classes=KVASIR_NUM_CLASSES, pretrained=False)
    load_kvasir_checkpoint(model_kvasir, args.ckpt_kvasir, device)
    model_kvasir = model_kvasir.to(device).eval()

    # --- Preprocess images ---
    ibs_tensor, _ = _load_and_preprocess(args.img_ibs, ibs_val_transforms())
    kvasir_tensor, _ = _load_and_preprocess(args.img_kvasir, kvasir_val_transforms())

    ibs_rgb = _denormalize(ibs_tensor, IBS_MEAN, IBS_STD)
    kvasir_rgb = _denormalize(kvasir_tensor, IMAGENET_MEAN, IMAGENET_STD)

    # --- Predict labels ---
    ibs_label = _predict(model_ibs, ibs_tensor, device)
    kvasir_label = _predict(model_kvasir, kvasir_tensor, device)
    print(f"IBS predicted class: {ibs_label}")
    print(f"Kvasir predicted class: {kvasir_label}")

    # --- Generate heatmaps ---
    print("Generating heatmaps...")
    heatmaps = {}
    for tag, model, tensor, label, name in [
        ("ibs", model_ibs, ibs_tensor, ibs_label, ARCH),
        ("kvasir", model_kvasir, kvasir_tensor, kvasir_label, ARCH),
    ]:
        heatmaps[(tag, "base")] = generate_base_hirescam(model, tensor, label, device)
        heatmaps[(tag, "hrcam")] = generate_hrcam(model, tensor, label, device)
        heatmaps[(tag, "enhanced")] = generate_enhanced_hirescam(
            model, name, tensor, label, device,
        )

    # --- Build figure ---
    col_titles = ["Original Image", "Base HiResCAM", "HR-CAM",
                   "Enhanced HiResCAM (Ours)"]
    row_data = [
        ("ibs", ibs_rgb),
        ("kvasir", kvasir_rgb),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for row_idx, (tag, rgb) in enumerate(row_data):
        # Col 0: Original
        axes[row_idx, 0].imshow(rgb)
        # Col 1-3: Overlays
        for col_idx, method_key in enumerate(["base", "hrcam", "enhanced"], start=1):
            cam = heatmaps[(tag, method_key)]
            overlay = _overlay(rgb, cam, alpha=0.5)
            axes[row_idx, col_idx].imshow(overlay)

    for ax in axes.ravel():
        ax.axis("off")

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight="bold", pad=8)

    # Row labels
    axes[0, 0].set_ylabel("IBS", fontsize=13, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("Kvasir-v2", fontsize=13, fontweight="bold", labelpad=10)
    for row_idx in range(2):
        axes[row_idx, 0].yaxis.set_visible(True)
        axes[row_idx, 0].tick_params(left=False, labelleft=False)

    plt.tight_layout(pad=1.0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {output_path}")

    # Also save PNG variant
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
