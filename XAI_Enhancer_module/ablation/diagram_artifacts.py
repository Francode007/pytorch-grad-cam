#!/usr/bin/env python3
"""
Architecture Diagram Artifact Extractor.

Extracts 5 high-resolution intermediate visual artifacts from the
XAI-Enhancer pipeline at a specific convolutional layer, suitable for
building a paper figure / architecture diagram.

Outputs (all 300 DPI PNG, no axes/margins):
    diagram_1_input_X.png          -- Original RGB input
    diagram_2_base_map_Ml.png      -- Raw saliency map at layer resolution (jet)
    diagram_3_upsampled_map.png    -- Upsampled + normalised map (jet)
    diagram_4_masked_input_X_hat.png -- X ⊙ upsampled map (salient regions only)
    diagram_5_final_enhanced.png   -- Final XAI-Enhancer output overlaid on X

Usage:
    python -m XAI_Enhancer_module.ablation.diagram_artifacts \
        --image data/kvasir-v2/polyps/polyps/00072d5f-7cd8-434c-8a5a-1a0bb2c9711d.jpg \
        --dataset kvasir \
        --checkpoint ./kvasir_runs/resnet50/best.pth \
        --layer-name layer4.0.conv2 \
        --output-dir runs/ablation/diagram \
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

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import scale_cam_image

from XAI_Enhancer_module.enhanced_cams import HiResCAMEnhanced
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

ARCH = "resnet50"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IBS_MEAN = [0.6380, 0.3422, 0.2275]
IBS_STD = [0.2448, 0.2060, 0.1710]
NUM_ENHANCED_LAYERS = 5

_SAVE_KW = dict(dpi=300, bbox_inches="tight", pad_inches=0, transparent=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _denormalize(tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """Reverse normalisation → [0, 1] float32 RGB numpy array (H, W, 3)."""
    img = tensor.cpu().clone()
    for c, m, s in zip(img, mean, std):
        c.mul_(s).add_(m)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _get_all_conv2d(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def _resolve_layer_by_name(model: nn.Module, name: str) -> nn.Module:
    """Resolve a dot-separated layer name like 'layer4.0.conv2'."""
    parts = name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _save_image(arr: np.ndarray, path: Path):
    """Save a numpy RGB [0,1] or [0,255] image with zero margins."""
    fig, ax = plt.subplots(figsize=(4, 4))
    if arr.dtype == np.float64 or arr.dtype == np.float32:
        ax.imshow(arr.clip(0, 1))
    else:
        ax.imshow(arr)
    ax.axis("off")
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    print(f"  Saved {path}")


def _save_heatmap(cam: np.ndarray, path: Path):
    """Save a grayscale [0,1] CAM as a pure jet heatmap (no image background)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cam, cmap="jet", vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(path, **_SAVE_KW)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_artifacts(
    model: nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    rgb_01: np.ndarray,
    target_layer: nn.Module,
    label: int,
    device: torch.device,
) -> dict:
    """Run the enhancer pipeline and extract 5 diagram artifacts.

    Returns a dict with keys matching the 5 output filenames (without
    prefix/extension).
    """
    img_batch = image_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(label)]
    input_h, input_w = img_batch.shape[2], img_batch.shape[3]

    # ----- Artifact 1: Original image -----
    artifacts = {"input_X": rgb_01}

    # ----- Artifacts 2 & 3: Raw + upsampled saliency via HiResCAMEnhanced -----
    cam_method = HiResCAMEnhanced(model, target_layers=[target_layer])
    cam_per_layer, _mod_act = cam_method(img_batch, targets)

    # cam_per_layer[0] is already upsampled & normalised: [B, 1, H_in, W_in]
    upsampled_map = cam_per_layer[0][0, 0]  # (H_in, W_in) in [0, 1]
    artifacts["upsampled_map"] = upsampled_map

    # Recompute the raw CAM at layer resolution from stored activations/grads
    act = cam_method.activations_and_grads.activations[0]  # tensor [B, C, H_l, W_l]
    grad = cam_method.activations_and_grads.gradients[0]   # tensor [B, C, H_l, W_l]
    act_np = act.cpu().detach().numpy()
    grad_np = grad.cpu().detach().numpy()
    raw_cam = (grad_np * act_np).sum(axis=1)  # [B, H_l, W_l]
    raw_cam = np.maximum(raw_cam[0], 0)       # (H_l, W_l)

    # Min-max normalise to [0, 1] for visualisation
    raw_min, raw_max = raw_cam.min(), raw_cam.max()
    if (raw_max - raw_min) > 1e-8:
        raw_cam_norm = (raw_cam - raw_min) / (raw_max - raw_min)
    else:
        raw_cam_norm = np.zeros_like(raw_cam)
    artifacts["base_map_Ml"] = raw_cam_norm

    try:
        cam_method.activations_and_grads.release()
    except Exception:
        pass

    # ----- Artifact 4: Masked input  X_hat = X ⊙ M_up -----
    mask_3ch = np.stack([upsampled_map] * 3, axis=-1)  # (H, W, 3)
    masked_input = rgb_01 * mask_3ch
    artifacts["masked_input_X_hat"] = masked_input

    # ----- Artifact 5: Final enhanced output overlaid on image -----
    all_conv = _get_all_conv2d(model)
    enhanced_layers = all_conv[-NUM_ENHANCED_LAYERS:]
    extractor = EnhancedExtractorV2(
        model=model,
        model_name=model_name,
        conv_layers=enhanced_layers,
        cam_method="HiResCAMEnhanced",
        aggregation_config={"type": "standard"},
    )
    _, final_cam = extractor.extract_saliency_map(img_batch, label)
    final_np = final_cam[0].cpu().numpy()  # (H, W) in [0, 1]
    final_np = final_np - final_np.min()
    denom = final_np.max()
    if denom > 1e-8:
        final_np = final_np / denom

    # Overlay with jet colormap at alpha=0.5
    heatmap = cv2.applyColorMap(
        np.uint8(255 * final_np), cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.5 * rgb_01 + 0.5 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    artifacts["final_enhanced"] = overlay

    return artifacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract architecture-diagram visual artifacts.",
    )
    p.add_argument("--image", required=True, help="Path to the input image")
    p.add_argument("--dataset", required=True, choices=["kvasir", "ibs"])
    p.add_argument("--checkpoint", required=True, help="ResNet-50 checkpoint path")
    p.add_argument("--layer-name", default="layer4.0.conv2",
                   help="Dot-separated layer name (e.g. layer4.0.conv2)")
    p.add_argument("--output-dir", default="runs/ablation/diagram")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Device: {device}")

    # --- Load model ---
    if args.dataset == "kvasir":
        model = build_kvasir_model(ARCH, num_classes=KVASIR_NUM_CLASSES, pretrained=False)
        load_kvasir_checkpoint(model, args.checkpoint, device)
        transform = kvasir_val_transforms()
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        model = build_ibs_model(ARCH, num_classes=IBS_NUM_CLASSES, pretrained=False)
        load_ibs_checkpoint(model, args.checkpoint, device)
        transform = ibs_val_transforms()
        mean, std = IBS_MEAN, IBS_STD
    model = model.to(device).eval()

    # --- Resolve target layer ---
    target_layer = _resolve_layer_by_name(model, args.layer_name)
    print(f"Target layer: {args.layer_name} → {target_layer}")

    # --- Load + preprocess image ---
    pil_img = Image.open(args.image).convert("RGB")
    image_tensor = transform(pil_img)
    rgb_01 = _denormalize(image_tensor, mean, std)

    # --- Predict label ---
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
    label = logits.argmax(dim=1).item()
    print(f"Predicted class: {label}")

    # --- Extract artifacts ---
    print("\nExtracting artifacts...")
    artifacts = extract_artifacts(
        model, ARCH, image_tensor, rgb_01, target_layer, label, device,
    )

    # --- Save ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "input_X":           "diagram_1_input_X.png",
        "base_map_Ml":       "diagram_2_base_map_Ml.png",
        "upsampled_map":     "diagram_3_upsampled_map.png",
        "masked_input_X_hat":"diagram_4_masked_input_X_hat.png",
        "final_enhanced":    "diagram_5_final_enhanced.png",
    }

    for key, filename in file_map.items():
        arr = artifacts[key]
        path = out_dir / filename
        if key in ("base_map_Ml", "upsampled_map"):
            _save_heatmap(arr, path)
        else:
            _save_image(arr, path)

    print(f"\nAll 5 artifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
