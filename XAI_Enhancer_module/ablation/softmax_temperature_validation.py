#!/usr/bin/env python3
"""
Softmax Temperature (T) Ablation Study.

Empirically evaluates how the temperature parameter T in the XAI-Enhancer
layer-weighting formula affects explanation faithfulness:

    w_l = exp(S_l / T) / sum_i exp(S_i / T)

For each T in {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} (configurable), the script:
  1. Runs XAI-Enhancer on a subset of validation images (e.g. 100).
  2. Computes Insertion AUC, Deletion AUC, and ROAD score per image.
  3. Records weight std-dev across layers to quantify sharpness.
  4. Saves a CSV summary and a matplotlib figure (two subplots).

Usage:
    python -m XAI_Enhancer_module.ablation.softmax_temperature_validation \
        --data-root data/kvasir-v2 \
        --checkpoint ./kvasir_runs/resnet50/best.pth \
        --arch resnet50 \
        --max-images 100 \
        --output-csv runs/ablation/temp_ablation.csv \
        --output-fig runs/ablation/temp_ablation.png \
        --device cuda
"""

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_TOP = Path(__file__).resolve().parent.parent.parent
if _TOP not in [Path(p).resolve() for p in sys.path]:
    sys.path.append(str(_TOP))

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from XAI_Enhancer_module.enhanced_cams import HiResCAMEnhanced
from XAI_Enhancer_module.kvasir.eval_cams import KvasirProperAUCEvaluator
from XAI_Enhancer_module.utils.model_utils import get_device

DEFAULT_T_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


class _SplitDataset(Dataset):
    """Minimal dataset backed by a split file (path, label) pairs."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, str(path)


def _extract_cam_and_scores(
    image_tensor: torch.Tensor,
    model: nn.Module,
    target_layers: List[nn.Module],
    device: torch.device,
    predicted_label: int,
) -> Tuple[List[torch.Tensor], np.ndarray]:
    """
    Run the XAI-Enhancer forward pipeline for one image and return:
      * per-layer CAMs
      * pre-softmax cosine similarity scores S_l for every target layer.
    """
    img_batch = image_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(predicted_label)]

    cam_method = HiResCAMEnhanced(model, target_layers)
    cam_per_layer, mod_act_per_layer = cam_method(img_batch, targets)

    model.eval()
    with torch.no_grad():
        actual_output = model(img_batch)

    similarities: List[float] = []
    for layer_idx, mod_act in enumerate(mod_act_per_layer):
        mod_tensor = (
            torch.from_numpy(mod_act).to(device)
            if isinstance(mod_act, np.ndarray)
            else mod_act.to(device)
        )
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

    # Release hooks from HiResCAMEnhanced
    try:
        cam_method.activations_and_grads.release()
    except Exception:
        pass

    return cam_per_layer, np.array(similarities, dtype=np.float64)


def _aggregate_cam_with_temperature(
    cam_per_layer: List[torch.Tensor],
    raw_scores: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Compute the final saliency map by re-weighting per-layer CAMs with
    softmax(S / T) for a given temperature T.

    Returns a 2D saliency map [H, W] normalised to [0, 1].
    """
    scores_t = torch.tensor(raw_scores, dtype=torch.float32)
    weights = F.softmax(scores_t / temperature, dim=0)

    first = cam_per_layer[0]
    if isinstance(first, np.ndarray):
        first = torch.from_numpy(first)
    if first.dim() == 4:
        first = first.squeeze(1)
    if first.dim() == 3:
        first = first.squeeze(0)

    final_cam = torch.zeros_like(first).float()
    for i, w in enumerate(weights):
        cam = cam_per_layer[i]
        if isinstance(cam, np.ndarray):
            cam = torch.from_numpy(cam)
        if cam.dim() == 4:
            cam = cam.squeeze(1)
        if cam.dim() == 3:
            cam = cam.squeeze(0)
        final_cam += w.item() * cam.float()

    final_cam = final_cam - final_cam.min()
    denom = final_cam.max()
    if denom > 1e-8:
        final_cam = final_cam / denom

    return final_cam.cpu().numpy()


def _evaluate_saliency_metrics(
    evaluator: KvasirProperAUCEvaluator,
    image_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    predicted_label: int,
) -> Dict[str, float]:
    """
    Compute Insertion AUC, Deletion AUC, and ROAD score using the project's
    existing evaluator.
    """
    metrics = evaluator._evaluate_saliency_map(
        image_tensor,
        saliency_map,
        predicted_label,
        step_size=224,   # 224x224 resolution → one row worth of pixels per step
        verbose=False,
        batch_size=2048,  # large chunk of masks per forward to better utilize GPU
    )
    road_vals = [v for k, v in metrics.items() if k.startswith("road_")]
    metrics["road_mean"] = float(np.mean(road_vals)) if road_vals else 0.0
    return metrics


def run_temperature_ablation(
    model: nn.Module,
    evaluator: KvasirProperAUCEvaluator,
    loader: DataLoader,
    target_layers: List[nn.Module],
    device: torch.device,
    t_values: List[float],
) -> List[Dict]:
    """
    Core loop: for each image, extract raw scores once, then for each T
    re-weight and evaluate.
    """
    # Per-temperature accumulators
    accum = {
        t: {"ins": [], "del": [], "road": [], "wt_std": []}
        for t in t_values
    }

    for img_tensor, label, path in tqdm(loader, desc="Images", unit="img"):
        img_tensor = img_tensor.squeeze(0)  # [C, H, W]
        label = label.item()

        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).item()

        cam_per_layer, raw_scores = _extract_cam_and_scores(
            img_tensor, model, target_layers, device, pred,
        )

        for t in t_values:
            weights_t = F.softmax(
                torch.tensor(raw_scores, dtype=torch.float32) / t, dim=0,
            ).numpy()
            wt_std = float(np.std(weights_t))

            saliency = _aggregate_cam_with_temperature(cam_per_layer, raw_scores, t)

            metrics = _evaluate_saliency_metrics(evaluator, img_tensor, saliency, pred)
            accum[t]["ins"].append(metrics.get("insertion_auc", 0.0))
            accum[t]["del"].append(metrics.get("deletion_auc", 0.0))
            accum[t]["road"].append(metrics.get("road_mean", 0.0))
            accum[t]["wt_std"].append(wt_std)

    results = []
    for t in t_values:
        results.append({
            "Temperature": t,
            "Insertion_AUC": float(np.mean(accum[t]["ins"])),
            "Deletion_AUC": float(np.mean(accum[t]["del"])),
            "ROAD_Score": float(np.mean(accum[t]["road"])),
            "Weight_StdDev": float(np.mean(accum[t]["wt_std"])),
            "Num_Images": len(accum[t]["ins"]),
        })
    return results


def save_csv(results: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Temperature", "Insertion_AUC", "Deletion_AUC",
                   "ROAD_Score", "Weight_StdDev", "Num_Images"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to {path}", flush=True)


def save_figure(results: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temps = [r["Temperature"] for r in results]
    road = [r["ROAD_Score"] for r in results]
    wt_std = [r["Weight_StdDev"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(temps, road, marker="o", linewidth=2, color="#2b6cb0")
    ax1.set_xscale("log")
    ax1.set_xlabel("Temperature T (log scale)", fontsize=12)
    ax1.set_ylabel("ROAD Score", fontsize=12)
    ax1.set_title("ROAD Score vs Temperature", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(temps, wt_std, marker="s", linewidth=2, color="#c53030")
    ax2.set_xscale("log")
    ax2.set_xlabel("Temperature T (log scale)", fontsize=12)
    ax2.set_ylabel("Weight Std Dev", fontsize=12)
    ax2.set_title("Weight Sharpness vs Temperature", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {path}", flush=True)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Softmax temperature T ablation for XAI-Enhancer layer weights.",
    )
    p.add_argument("--data-root", type=str, default="data/kvasir-v2",
                    help="Kvasir-v2 data root (must contain splits/val.txt).")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained Kvasir model checkpoint.")
    p.add_argument("--arch", type=str, default="resnet50",
                    choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"])
    p.add_argument("--max-images", type=int, default=100,
                    help="Number of validation images to evaluate.")
    p.add_argument("--temperatures", type=float, nargs="+",
                    default=DEFAULT_T_VALUES,
                    help="Temperature values to sweep (default: 0.1 0.5 1.0 2.0 5.0 10.0).")
    p.add_argument("--output-csv", type=str, default="runs/ablation/temp_ablation.csv")
    p.add_argument("--output-fig", type=str, default="runs/ablation/temp_ablation.png")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Device: {device}", flush=True)
    print(f"Architecture: {args.arch}", flush=True)
    print(f"Temperatures: {args.temperatures}", flush=True)
    print(f"Max images: {args.max_images}", flush=True)
    print("", flush=True)

    # --- Evaluator: same as kvasir/eval_cams.py (KvasirProperAUCEvaluator) ---
    print(f"Initializing KvasirProperAUCEvaluator (same as eval_cams.py)...", flush=True)
    evaluator = KvasirProperAUCEvaluator(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        arch=args.arch,
        split="val",
        device_preference=args.device,
        layer_mode="all",  # all conv layers for XAI-Enhancer
    )
    model = evaluator.model
    device = evaluator.device
    target_layers = evaluator.conv_layers
    print(f"Using all {len(target_layers)} convolutional layers for XAI-Enhancer.", flush=True)

    # --- Dataset: same image list and predicted labels as eval_cams ---
    image_paths, predicted_labels, _ = evaluator.get_imagenet_images(max_images=args.max_images)
    samples = list(zip(image_paths, predicted_labels))
    print(f"Validation images: {len(samples)}", flush=True)

    loader = DataLoader(
        _SplitDataset(samples, evaluator.transform),
        batch_size=1,
        shuffle=False,
        num_workers=24,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=6 if 24 > 0 else None,
    )

    # --- Run ---
    print("\nStarting temperature ablation...", flush=True)
    results = run_temperature_ablation(
        model, evaluator, loader, target_layers, device, args.temperatures,
    )

    # --- Print summary ---
    print("\n=== Temperature Ablation Results ===", flush=True)
    header = f"{'T':>6s}  {'Ins AUC':>8s}  {'Del AUC':>8s}  {'ROAD':>8s}  {'Wt Std':>8s}  {'N':>4s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        print(
            f"{r['Temperature']:6.1f}  {r['Insertion_AUC']:8.4f}  "
            f"{r['Deletion_AUC']:8.4f}  {r['ROAD_Score']:8.4f}  "
            f"{r['Weight_StdDev']:8.4f}  {r['Num_Images']:4d}",
            flush=True,
        )

    # --- Save ---
    save_csv(results, args.output_csv)
    save_figure(results, args.output_fig)

    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
