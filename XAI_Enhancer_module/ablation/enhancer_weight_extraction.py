#!/usr/bin/env python3
"""
XAI-Enhancer Logit-Similarity Weight Extraction.

For every validation image, runs HiResCAMEnhanced on the last 5 Conv2d
layers, computes modified outputs via activation injection, derives
cosine-similarity scores, and applies softmax to obtain the per-layer
weights that the Enhancer would assign.

Exports a per-image CSV:
    Dataset, Model, Layer_5_Weight, Layer_4_Weight, Layer_3_Weight,
    Layer_2_Weight, Layer_1_Weight

Usage:
    python -m XAI_Enhancer_module.ablation.enhancer_weight_extraction \
        --kvasir-data-root data/kvasir-v2 \
        --ibs-data-root data/IBS-preprocessed-dataset \
        --checkpoints kvasir:resnet50:/path/to/ckpt.pth \
                      ibs:vgg16:/path/to/ckpt.pth \
        --output-csv runs/ablation/enhancer_weights.csv \
        --max-images 100 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from XAI_Enhancer_module.enhanced_cams import HiResCAMEnhanced
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

MODELS = ["vgg16", "resnet50"]
DATASETS = ["kvasir", "ibs"]
NUM_LAYERS = 5


# ---------------------------------------------------------------------------
# Dataset wrapper (shared with layerwise_road_extraction)
# ---------------------------------------------------------------------------

class _SplitDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


# ---------------------------------------------------------------------------
# Core: extract the 5 Enhancer weights for one image
# ---------------------------------------------------------------------------

def get_enhancer_weights(
    image_tensor: torch.Tensor,
    model: nn.Module,
    target_layers: list[nn.Module],
    device: torch.device,
    predicted_label: int,
) -> np.ndarray:
    """Return the softmax logit-similarity weights for *target_layers*.

    Pipeline (mirrors OptimizedCamExtractor):
      1. Run HiResCAMEnhanced → per-layer CAMs + masked activations.
      2. Forward the original image → actual logits.
      3. For each layer, inject its masked activations → modified logits.
      4. Cosine similarity(actual, modified) per layer.
      5. Softmax over layers → weights.

    Args:
        image_tensor: Single image tensor (C, H, W) – already normalised.
        model: Classification model (eval mode, on *device*).
        target_layers: List of nn.Conv2d modules (length N).
        device: Torch device.
        predicted_label: Target class for GradCAM.

    Returns:
        1-D numpy array of length N with the per-layer weights (sum ≈ 1).
    """
    img_batch = image_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(predicted_label)]

    # --- Step 1: enhanced CAM forward ---
    cam_method = HiResCAMEnhanced(model, target_layers)
    cam_per_layer, mod_act_per_layer = cam_method(img_batch, targets)
    # cam_per_layer:     list of ndarray [1, 1, H, W]
    # mod_act_per_layer: list of ndarray [1, C_l, H_l, W_l]

    # --- Step 2: actual output ---
    model.eval()
    with torch.no_grad():
        actual_output = model(img_batch)  # [1, num_classes]

    # --- Step 3 + 4: modified outputs & cosine similarities ---
    similarities = []
    for layer_idx, mod_act in enumerate(mod_act_per_layer):
        mod_tensor = torch.from_numpy(mod_act).to(device) if isinstance(mod_act, np.ndarray) else mod_act.to(device)
        layer = target_layers[layer_idx]

        hook_replacement = mod_tensor
        captured = {}

        def _hook(module, inp, out, replacement=hook_replacement):
            return replacement

        handle = layer.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                modified_output = model(img_batch)  # [1, num_classes]
        finally:
            handle.remove()

        cos = F.cosine_similarity(actual_output, modified_output, dim=1)  # [1]
        similarities.append(cos.item())

    # --- Step 5: softmax → weights ---
    sim_tensor = torch.tensor(similarities, dtype=torch.float32)
    weights = F.softmax(sim_tensor, dim=0).numpy()

    # Clean up hooks registered by HiResCAMEnhanced
    try:
        cam_method.activations_and_grads.release()
    except Exception:
        pass

    return weights


def get_enhancer_raw_and_softmax(
    image_tensor: torch.Tensor,
    model: nn.Module,
    target_layers: list[nn.Module],
    device: torch.device,
    predicted_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return *both* the pre-softmax raw cosine similarities (alpha_l)
    and the post-softmax weights for *target_layers*.

    The pipeline is identical to :func:`get_enhancer_weights` but exposes
    the intermediate raw scores as well.

    Returns:
        ``(raw_scores, softmax_weights)`` -- two 1-D numpy arrays of
        length N.  ``raw_scores`` are the per-layer cosine similarities
        in [-1, 1]; ``softmax_weights`` sum to ~1.
    """
    img_batch = image_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(predicted_label)]

    cam_method = HiResCAMEnhanced(model, target_layers)
    cam_per_layer, mod_act_per_layer = cam_method(img_batch, targets)

    model.eval()
    with torch.no_grad():
        actual_output = model(img_batch)

    similarities: list[float] = []
    for layer_idx, mod_act in enumerate(mod_act_per_layer):
        mod_tensor = (torch.from_numpy(mod_act).to(device)
                      if isinstance(mod_act, np.ndarray)
                      else mod_act.to(device))
        layer = target_layers[layer_idx]

        def _hook(module, inp, out, replacement=mod_tensor):
            return replacement

        handle = layer.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                modified_output = model(img_batch)
        finally:
            handle.remove()

        cos = F.cosine_similarity(actual_output, modified_output, dim=1)
        similarities.append(cos.item())

    try:
        cam_method.activations_and_grads.release()
    except Exception:
        pass

    raw_scores = np.array(similarities, dtype=np.float64)
    softmax_weights = F.softmax(
        torch.tensor(similarities, dtype=torch.float32), dim=0,
    ).numpy()
    return raw_scores, softmax_weights


def get_enhancer_raw_and_softmax_all_layers(
    image_tensor: torch.Tensor,
    model: nn.Module,
    target_layers: list[nn.Module],
    device: torch.device,
    predicted_label: int,
) -> list[tuple[int, float, float]]:
    """Extract raw cosine similarity and softmax weight for *every* layer.

    Identical pipeline to :func:`get_enhancer_raw_and_softmax` but designed
    for an arbitrary number of target layers (not just the last 5).

    Args:
        image_tensor: (C, H, W) normalised image tensor.
        model:        Classification model (eval, on *device*).
        target_layers: **All** Conv2d modules to evaluate (length N).
        device:       Torch device.
        predicted_label: Target class for GradCAM.

    Returns:
        List of ``(layer_index, raw_similarity, softmax_weight)`` tuples,
        one per layer.  ``layer_index`` runs from 0 to N-1 (network order).
        Softmax is computed over all N layers jointly.
    """
    img_batch = image_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(predicted_label)]

    cam_method = HiResCAMEnhanced(model, target_layers)
    _cam_per_layer, mod_act_per_layer = cam_method(img_batch, targets)

    model.eval()
    with torch.no_grad():
        actual_output = model(img_batch)

    similarities: list[float] = []
    for layer_idx, mod_act in enumerate(mod_act_per_layer):
        mod_tensor = (torch.from_numpy(mod_act).to(device)
                      if isinstance(mod_act, np.ndarray)
                      else mod_act.to(device))
        layer = target_layers[layer_idx]

        def _hook(module, inp, out, replacement=mod_tensor):
            return replacement

        handle = layer.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                modified_output = model(img_batch)
        finally:
            handle.remove()

        cos = F.cosine_similarity(actual_output, modified_output, dim=1)
        similarities.append(cos.item())

    try:
        cam_method.activations_and_grads.release()
    except Exception:
        pass

    softmax_weights = F.softmax(
        torch.tensor(similarities, dtype=torch.float32), dim=0,
    ).numpy()

    return [
        (i, similarities[i], float(softmax_weights[i]))
        for i in range(len(similarities))
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_all_conv2d(model: nn.Module) -> list[nn.Module]:
    """Return *every* Conv2d layer in network order."""
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def _get_last_n_conv2d(model: nn.Module, n: int = NUM_LAYERS) -> list[nn.Module]:
    all_conv = _get_all_conv2d(model)
    return all_conv[-n:] if len(all_conv) >= n else all_conv


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


def _build_dataloader(
    dataset: str, data_root: str, max_images: int, device: torch.device,
) -> DataLoader:
    root = Path(data_root)
    splits_dir = root / "splits"
    if dataset == "kvasir":
        samples = kvasir_load_split(splits_dir / "val.txt", root)
        transform = kvasir_val_transforms()
    else:
        samples = ibs_load_split(splits_dir / "val.txt", root)
        transform = ibs_val_transforms()
    if 0 < max_images < len(samples):
        samples = samples[:max_images]
    return DataLoader(
        _SplitDataset(samples, transform),
        batch_size=1, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract XAI-Enhancer logit-similarity weights per layer.",
    )
    p.add_argument("--kvasir-data-root", default="data/kvasir-v2")
    p.add_argument("--ibs-data-root", default="data/IBS-preprocessed-dataset")
    p.add_argument("--checkpoints", nargs="+", required=True, metavar="DS:ARCH:PATH")
    p.add_argument("--output-csv", default="runs/ablation/enhancer_weights.csv")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Device: {device}")

    ckpt_map: dict[tuple[str, str], str] = {}
    for spec in args.checkpoints:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad checkpoint spec '{spec}'")
        ckpt_map[(parts[0].lower(), parts[1].lower())] = parts[2]

    data_roots = {"kvasir": args.kvasir_data_root, "ibs": args.ibs_data_root}

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for ds in DATASETS:
        print(f"\n{'='*60}\nDataset: {ds.upper()}\n{'='*60}")
        loader = _build_dataloader(ds, data_roots[ds], args.max_images, device)

        for arch in MODELS:
            key = (ds, arch)
            if key not in ckpt_map:
                print(f"  [SKIP] {ds}/{arch}")
                continue

            print(f"\n  Model: {arch}")
            model = _load_model(ds, arch, ckpt_map[key], device)
            target_layers = _get_last_n_conv2d(model, NUM_LAYERS)
            print(f"    Using last {len(target_layers)} Conv2d layers")

            for img_tensor, label in tqdm(loader, desc=f"    {ds}/{arch}", leave=False):
                img_tensor = img_tensor.squeeze(0).to(device)
                label_int = label.item()

                weights = get_enhancer_weights(
                    img_tensor, model, target_layers, device, label_int,
                )

                row = {"Dataset": ds, "Model": arch}
                for i, w in enumerate(weights):
                    col = f"Layer_{NUM_LAYERS - i}_Weight"
                    row[col] = round(float(w), 6)
                rows.append(row)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
