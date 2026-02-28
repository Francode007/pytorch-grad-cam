#!/usr/bin/env python3
"""
Bottleneck Visualisation: side-by-side GradCAM heatmaps from the -4
Conv2d layer of ResNet-34 and ResNet-50.

Provides both:
* A reusable ``visualize_bottleneck()`` function.
* A CLI entry-point for quick one-off visualisations.

Usage:
    python -m XAI_Enhancer_module.ablation.bottleneck_visualization \
        --image-path data/kvasir-v2/polyps/some_image.jpg \
        --dataset kvasir \
        --resnet34-ckpt /path/to/resnet34.pth \
        --resnet50-ckpt /path/to/resnet50.pth \
        --output-path runs/ablation/bottleneck_vis.png \
        --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.kvasir.data import (
    KVASIR_NUM_CLASSES,
    IMAGENET_MEAN as KVASIR_MEAN,
    IMAGENET_STD as KVASIR_STD,
    get_val_transforms as kvasir_val_transforms,
)
from XAI_Enhancer_module.ibs.models import build_ibs_model, load_ibs_checkpoint
from XAI_Enhancer_module.ibs.data import (
    IBS_NUM_CLASSES,
    IBS_MEAN,
    IBS_STD,
    get_val_transforms as ibs_val_transforms,
)
from XAI_Enhancer_module.utils.model_utils import get_device

# Normalisation constants per dataset
_NORM = {
    "kvasir": {"mean": KVASIR_MEAN, "std": KVASIR_STD},
    "ibs": {"mean": IBS_MEAN, "std": IBS_STD},
}

BOTTLENECK_LAYER_IDX = -4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_conv2d_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def _load_model(dataset: str, arch: str, ckpt: str, device: torch.device) -> nn.Module:
    if dataset == "kvasir":
        model = build_kvasir_model(arch, num_classes=KVASIR_NUM_CLASSES, pretrained=False)
        load_kvasir_checkpoint(model, ckpt, device)
    elif dataset == "ibs":
        model = build_ibs_model(arch, num_classes=IBS_NUM_CLASSES, pretrained=False)
        load_ibs_checkpoint(model, ckpt, device)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return model.to(device).eval()


def _denormalize(tensor: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """Convert a normalised (C, H, W) tensor back to a uint8 RGB numpy array."""
    img = tensor.cpu().clone()
    for c, m, s in zip(img, mean, std):
        c.mul_(s).add_(m)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _extract_cam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    layer: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Run GradCAM on a single layer and return the raw (H, W) activation map."""
    cam = GradCAM(model=model, target_layers=[layer])
    img_batch = image_tensor.unsqueeze(0).to(device) if image_tensor.dim() == 3 else image_tensor.to(device)

    # Predict the label so the heatmap reflects the model's own decision
    with torch.no_grad():
        pred = model(img_batch).argmax(dim=1).item()

    grayscale_cam = cam(input_tensor=img_batch, targets=[ClassifierOutputTarget(pred)])
    cam.__del__()
    return grayscale_cam[0]  # (H, W)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_bottleneck(
    image_tensor: torch.Tensor,
    resnet34_ckpt: str,
    resnet50_ckpt: str,
    dataset: str = "kvasir",
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Extract and plot GradCAM heatmaps from the -4 Conv2d layer of
    ResNet-34 and ResNet-50 alongside the original image.

    Args:
        image_tensor: Pre-processed (C, H, W) tensor (normalised).
        resnet34_ckpt: Path to the ResNet-34 checkpoint.
        resnet50_ckpt: Path to the ResNet-50 checkpoint.
        dataset: ``"kvasir"`` or ``"ibs"`` (selects model builder and norm).
        device: Torch device string.
        output_path: If given, saves the figure to this path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    dev = torch.device(device)
    norm = _NORM[dataset]

    # Load both models
    r34 = _load_model(dataset, "resnet34", resnet34_ckpt, dev)
    r50 = _load_model(dataset, "resnet50", resnet50_ckpt, dev)

    # Target layers
    r34_layer = _get_conv2d_layers(r34)[BOTTLENECK_LAYER_IDX]
    r50_layer = _get_conv2d_layers(r50)[BOTTLENECK_LAYER_IDX]

    cam_r34 = _extract_cam(r34, image_tensor, r34_layer, dev)
    cam_r50 = _extract_cam(r50, image_tensor, r50_layer, dev)

    # De-normalise original image for display
    original_rgb = _denormalize(image_tensor, norm["mean"], norm["std"])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im1 = axes[1].imshow(cam_r34, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"ResNet-34  (layer {BOTTLENECK_LAYER_IDX})")
    axes[1].axis("off")

    im2 = axes[2].imshow(cam_r50, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title(f"ResNet-50  (layer {BOTTLENECK_LAYER_IDX})")
    axes[2].axis("off")

    fig.colorbar(im2, ax=axes.tolist(), fraction=0.02, pad=0.04)
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved bottleneck visualisation to {out}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise GradCAM heatmaps from the -4 Conv2d layer of ResNet-34 / ResNet-50."
    )
    p.add_argument("--image-path", type=str, required=True,
                    help="Path to a single input image (JPEG/PNG).")
    p.add_argument("--dataset", type=str, required=True, choices=["kvasir", "ibs"],
                    help="Dataset whose model weights and normalisation to use.")
    p.add_argument("--resnet34-ckpt", type=str, required=True,
                    help="Path to ResNet-34 checkpoint.")
    p.add_argument("--resnet50-ckpt", type=str, required=True,
                    help="Path to ResNet-50 checkpoint.")
    p.add_argument("--output-path", type=str, default="runs/ablation/bottleneck_vis.png",
                    help="Where to save the figure.")
    p.add_argument("--device", type=str, default="cuda",
                    help="Device preference (cuda, mps, cpu, auto).")
    return p.parse_args()


def main():
    args = parse_args()
    device_str = get_device(args.device)

    # Load and transform the image using the dataset's val transforms
    if args.dataset == "kvasir":
        transform = kvasir_val_transforms()
    else:
        transform = ibs_val_transforms()

    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image)

    fig = visualize_bottleneck(
        image_tensor=image_tensor,
        resnet34_ckpt=args.resnet34_ckpt,
        resnet50_ckpt=args.resnet50_ckpt,
        dataset=args.dataset,
        device=device_str,
        output_path=args.output_path,
    )
    plt.show()


if __name__ == "__main__":
    main()
