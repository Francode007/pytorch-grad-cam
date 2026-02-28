#!/usr/bin/env python3
"""
Layer-wise ROAD Metric Extraction for Ablation Study.

Iterates through the last 5 Conv2d layers (indices -1 to -5) of five
pre-trained architectures (vgg16, vgg19, resnet18, resnet34, resnet50)
on two datasets (IBS, Kvasir-v2). For each layer it computes the ROAD
metric using standard GradCAM and exports a structured CSV.

Usage:
    python -m XAI_Enhancer_module.ablation.layerwise_road_extraction \
        --kvasir-data-root data/kvasir-v2 \
        --ibs-data-root data/IBS-preprocessed-dataset \
        --checkpoints kvasir:resnet50:/path/to/ckpt.pth \
                      ibs:vgg16:/path/to/ckpt.pth \
        --output-csv runs/ablation/layerwise_road.csv \
        --max-images 100 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.kvasir.data import (
    load_split_file as kvasir_load_split,
    get_val_transforms as kvasir_val_transforms,
    KVASIR_NUM_CLASSES,
)
from XAI_Enhancer_module.ibs.models import build_ibs_model, load_ibs_checkpoint
from XAI_Enhancer_module.ibs.data import (
    load_split_file as ibs_load_split,
    get_val_transforms as ibs_val_transforms,
    IBS_NUM_CLASSES,
)
from XAI_Enhancer_module.utils.model_utils import get_device

MODELS = ["vgg16", "vgg19", "resnet18", "resnet34", "resnet50"]
DATASETS = ["kvasir", "ibs"]
LAYER_INDICES = list(range(-5, 0))  # -5, -4, -3, -2, -1
ROAD_THRESHOLDS = [20, 40, 60, 80]


# ---------------------------------------------------------------------------
# Lightweight dataset wrapper (mirrors _ImageNetDataset in the evaluator)
# ---------------------------------------------------------------------------

class _SplitDataset(Dataset):
    """Wraps (path, label) pairs with a transform for DataLoader usage."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


# ---------------------------------------------------------------------------
# Standalone ROAD computation (extracted from ImageNetProperAUCEvaluator)
# ---------------------------------------------------------------------------

def compute_road(
    image_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    predicted_label: int,
    model: nn.Module,
    device: torch.device,
    thresholds: list[int] = ROAD_THRESHOLDS,
) -> float:
    """Compute the multi-threshold averaged ROAD score for a single image.

    The ROAD score measures the confidence drop when the most important
    pixels (as indicated by *saliency_map*) are replaced with a blurred
    version of the image.  Scores at each threshold are averaged to produce
    a single scalar.

    Args:
        image_tensor: (C, H, W) or (1, C, H, W) input tensor.
        saliency_map: (H, W) or (1, H, W) numpy array.
        predicted_label: Target class index for the model.
        model: The classification model (in eval mode).
        device: Torch device.
        thresholds: Percentile thresholds for pixel removal.

    Returns:
        Mean ROAD score across all thresholds (non-negative float).
    """
    image_tensor = image_tensor.to(device).clone()
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    if saliency_map.ndim == 3:
        saliency_map = saliency_map[0]

    flat_saliency = saliency_map.flatten()

    # Blur-based imputation
    blurred = TF.gaussian_blur(image_tensor, kernel_size=11, sigma=5.0)
    imputation_flat = blurred.view(blurred.shape[0], -1)

    batch_images = [image_tensor]
    for p in thresholds:
        percentile_val = np.percentile(flat_saliency, 100 - p)
        mask_flat = (saliency_map > percentile_val).flatten()
        mask_indices = torch.where(torch.from_numpy(mask_flat).to(device))[0]

        modified = image_tensor.clone()
        modified_flat = modified.view(modified.shape[0], -1)
        modified_flat[:, mask_indices] = imputation_flat[:, mask_indices]
        batch_images.append(modified)

    batch_tensor = torch.stack(batch_images)

    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(batch_tensor)
        else:
            outputs = model(batch_tensor)
        probs = torch.softmax(outputs, dim=1)[:, predicted_label]

    original_conf = probs[0].item()
    scores = []
    for i in range(len(thresholds)):
        scores.append(max(0.0, original_conf - probs[i + 1].item()))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Layer helpers
# ---------------------------------------------------------------------------

def get_all_conv2d_layers(model: nn.Module) -> list[nn.Module]:
    """Return every nn.Conv2d in forward order."""
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


# ---------------------------------------------------------------------------
# Model / data factory helpers
# ---------------------------------------------------------------------------

def load_model(dataset: str, arch: str, checkpoint: str, device: torch.device) -> nn.Module:
    """Build the appropriate model and load its checkpoint."""
    if dataset == "kvasir":
        model = build_kvasir_model(arch, num_classes=KVASIR_NUM_CLASSES, pretrained=False)
        load_kvasir_checkpoint(model, checkpoint, device)
    elif dataset == "ibs":
        model = build_ibs_model(arch, num_classes=IBS_NUM_CLASSES, pretrained=False)
        load_ibs_checkpoint(model, checkpoint, device)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    model.to(device).eval()
    return model


def build_dataloader(
    dataset: str,
    data_root: str,
    max_images: int,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:
    """Create a validation DataLoader for the requested dataset."""
    data_root = Path(data_root)
    splits_dir = data_root / "splits"

    if dataset == "kvasir":
        samples = kvasir_load_split(splits_dir / "val.txt", data_root)
        transform = kvasir_val_transforms()
    elif dataset == "ibs":
        samples = ibs_load_split(splits_dir / "val.txt", data_root)
        transform = ibs_val_transforms()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if 0 < max_images < len(samples):
        samples = samples[:max_images]

    ds = _SplitDataset(samples, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


# ---------------------------------------------------------------------------
# Core evaluation routine
# ---------------------------------------------------------------------------

def evaluate_layer(
    model: nn.Module,
    target_layer: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Run GradCAM on *target_layer* for every image and return mean ROAD."""
    cam = GradCAM(model=model, target_layers=[target_layer])

    road_scores: list[float] = []
    for img_tensor, label in tqdm(dataloader, desc="    images", leave=False):
        img_tensor = img_tensor.to(device)
        label_int = label.item() if isinstance(label, torch.Tensor) else int(label)

        targets = [ClassifierOutputTarget(label_int)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)  # (B, H, W)

        # Process each sample in the (mini-)batch
        for b in range(img_tensor.shape[0]):
            score = compute_road(
                img_tensor[b],
                grayscale_cam[b],
                label_int,
                model,
                device,
            )
            road_scores.append(score)

    cam.__del__()  # release hooks
    return float(np.mean(road_scores)) if road_scores else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Layer-wise ROAD ablation study across models and datasets.",
    )
    p.add_argument("--kvasir-data-root", type=str, default="data/kvasir-v2",
                    help="Root directory for Kvasir-v2 dataset.")
    p.add_argument("--ibs-data-root", type=str, default="data/IBS-preprocessed-dataset",
                    help="Root directory for IBS dataset.")
    p.add_argument(
        "--checkpoints", nargs="+", required=True, metavar="DATASET:ARCH:PATH",
        help=(
            "Checkpoint specifications as dataset:arch:path triples. "
            "Example: kvasir:resnet50:/weights/kvasir_r50.pth ibs:vgg16:/weights/ibs_vgg16.pth"
        ),
    )
    p.add_argument("--output-csv", type=str, default="runs/ablation/layerwise_road.csv",
                    help="Path for the output CSV.")
    p.add_argument("--max-images", type=int, default=-1,
                    help="Max validation images per dataset (-1 = all).")
    p.add_argument("--device", type=str, default="cuda",
                    help="Device preference (cuda, mps, cpu, auto).")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(get_device(args.device))
    print(f"Using device: {device}")

    # Parse checkpoint mapping: {(dataset, arch): path}
    ckpt_map: dict[tuple[str, str], str] = {}
    for spec in args.checkpoints:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid checkpoint spec '{spec}'. Expected dataset:arch:path"
            )
        ds, arch, path = parts
        ckpt_map[(ds.lower(), arch.lower())] = path

    data_roots = {"kvasir": args.kvasir_data_root, "ibs": args.ibs_data_root}

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from a partial CSV if one already exists (crash recovery)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        rows: list[dict] = existing_df.to_dict("records")
        done = {(r["Dataset"], r["Model"]) for r in rows}
        print(f"Resuming: loaded {len(rows)} existing rows from {output_path}")
    else:
        rows = []
        done = set()

    def _save():
        pd.DataFrame(rows).to_csv(output_path, index=False)

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*60}")

        loader = build_dataloader(ds, data_roots[ds], args.max_images, device)

        for arch in MODELS:
            key = (ds, arch)
            if key not in ckpt_map:
                print(f"  [SKIP] No checkpoint provided for {ds}/{arch}")
                continue

            if (ds, arch) in done:
                print(f"  [CACHED] {ds}/{arch} already in CSV, skipping")
                continue

            print(f"\n  Model: {arch}")
            model = load_model(ds, arch, ckpt_map[key], device)
            conv_layers = get_all_conv2d_layers(model)
            n_conv = len(conv_layers)
            print(f"    Total Conv2d layers: {n_conv}")

            if n_conv < 5:
                print(f"    WARNING: model has fewer than 5 Conv2d layers; using all {n_conv}")

            for idx in LAYER_INDICES:
                if abs(idx) > n_conv:
                    continue
                layer = conv_layers[idx]
                print(f"    Layer index {idx:+d} ({layer})")
                road_mean = evaluate_layer(model, layer, loader, device)
                print(f"      ROAD (mean): {road_mean:.4f}")
                rows.append({
                    "Dataset": ds,
                    "Model": arch,
                    "Layer_Index": idx,
                    "ROAD_Score": round(road_mean, 6),
                })

            # Flush to disk after every model so progress survives crashes
            _save()
            print(f"    [SAVED] incremental results to {output_path}")

            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

    # Final save
    _save()
    df = pd.DataFrame(rows)
    print(f"\nResults saved to {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
