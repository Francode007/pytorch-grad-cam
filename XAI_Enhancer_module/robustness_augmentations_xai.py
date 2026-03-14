import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Project root so "XAI_Enhancer_module" can be imported when run as
# python XAI_Enhancer_module/robustness_augmentations_xai.py ...
_TOP = Path(__file__).resolve().parent.parent
if _TOP not in [Path(p).resolve() for p in sys.path]:
    sys.path.append(str(_TOP))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from XAI_Enhancer_module.utils.model_utils import (
    get_device,
    MEAN,
    STD,
    transformations,
)
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor


@dataclass
class RobustnessConfig:
    image_dir: str
    batch_size: int = 8
    num_workers: int = 0
    gaussian_sigma: float = 0.05
    contrast_alpha: float = 0.25  # contrast factor range around 1.0
    brightness_beta: float = 0.10  # brightness shift range around 0.0
    max_visualizations: int = 3
    output_dir: str = "robustness_outputs"


class MedicalImageDataset(Dataset):
    """
    Lightweight dataset that loads all images from a directory (recursively).
    Intended for qualitative robustness analysis on medical images.
    """

    def __init__(self, image_dir: str):
        self.image_paths: List[str] = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                    self.image_paths.append(os.path.join(root, f))
        if not self.image_paths:
            raise RuntimeError(f"No image files found in directory: {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = plt.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (224, 224))
        # Keep both the raw image (for visualization/perturbation) and the normalized tensor
        tensor = transformations(img).float()  # [C, H, W]
        return tensor, img, path


def create_perturbations(
    images_np: np.ndarray,
    cfg: RobustnessConfig,
) -> np.ndarray:
    """
    Apply clinical-style perturbations to a batch of images in [0,1] numpy format.

    Perturbations:
      - Gaussian sensor noise
      - Contrast shift
      - Brightness adjustment
    """
    # images_np: [B, H, W, C] in [0,1]
    noisy = images_np.copy()

    # Gaussian sensor noise
    noise = np.random.normal(0.0, cfg.gaussian_sigma, size=noisy.shape).astype(np.float32)
    noisy = noisy + noise

    # Contrast shift: (x - 0.5) * alpha + 0.5, with alpha in [1-alpha, 1+alpha]
    alpha = 1.0 + np.random.uniform(-cfg.contrast_alpha, cfg.contrast_alpha, size=(noisy.shape[0], 1, 1, 1)).astype(
        np.float32
    )
    noisy = (noisy - 0.5) * alpha + 0.5

    # Brightness adjustment: add beta in [-beta, beta]
    beta = np.random.uniform(-cfg.brightness_beta, cfg.brightness_beta, size=(noisy.shape[0], 1, 1, 1)).astype(
        np.float32
    )
    noisy = noisy + beta

    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


def normalize_batch_from_np(images_np: np.ndarray) -> torch.Tensor:
    """
    Convert a batch of [B, H, W, C] images in [0,1] to normalized tensors [B, C, H, W].
    Uses the global MEAN and STD from model_utils to stay consistent with training.
    """
    # Move channels first
    imgs = torch.from_numpy(images_np).permute(0, 3, 1, 2)  # [B, C, H, W]
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (imgs - mean) / std


def ssim_global(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Global SSIM over all pixels of 2D heatmaps (single-channel).
    Simplified version (no window) sufficient for comparative robustness analysis.
    """
    x = x.view(-1)
    y = y.view(-1)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var(unbiased=False)
    sigma_y = y.var(unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    return numerator / (denominator + eps)


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation coefficient between two 2D heatmaps.
    """
    x = x.view(-1)
    y = y.view(-1)
    vx = x - x.mean()
    vy = y - y.mean()
    cov = (vx * vy).mean()
    sx = torch.sqrt((vx ** 2).mean() + eps)
    sy = torch.sqrt((vy ** 2).mean() + eps)
    return cov / (sx * sy + eps)


def get_resnet50_for_explainability(device: torch.device) -> nn.Module:
    """
    Load a ResNet-50 suitable for Grad-CAM and XAI-Enhancer.
    Uses ImageNet-pretrained weights if available.
    """
    from torchvision import models
    try:
        from torchvision.models import ResNet50_Weights

        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet50(pretrained=True)

    model.eval()
    model.to(device)
    return model


def get_resnet50_target_layers(model: nn.Module) -> Dict[str, List[nn.Module]]:
    """
    Select target layers for explainability:

    - For standard Grad-CAM: use ONLY the last convolutional layer,
      matching your standard-method setup across experiments.
    - For the XAI-Enhancer: use ALL convolutional layers, matching
      the "all" layer_mode used in your enhanced evaluators.
    """
    all_conv_layers: List[nn.Module] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            all_conv_layers.append(module)

    if not all_conv_layers:
        raise ValueError("No convolutional layers found in ResNet-50.")

    standard_target = [all_conv_layers[-1]]
    enhancer_targets = all_conv_layers

    return {"standard": standard_target, "enhancer": enhancer_targets}


def compute_heatmaps(
    model: nn.Module,
    target_layers: Dict[str, List[nn.Module]],
    batch_tensor: torch.Tensor,
    device: torch.device,
    predicted_labels: List[int],
):
    """
    Compute Grad-CAM and XAI-Enhancer heatmaps for a batch of inputs.
    Returns heatmaps in [0,1] as torch tensors of shape [B, H, W].
    """
    # Standard Grad-CAM
    gradcam = GradCAM(model=model, target_layers=target_layers["standard"])
    targets = [ClassifierOutputTarget(int(lbl)) for lbl in predicted_labels]
    cams_standard = gradcam(input_tensor=batch_tensor.to(device), targets=targets)  # [B, 1, H, W] or [B, H, W]
    cams_standard_t = torch.from_numpy(cams_standard)
    if cams_standard_t.dim() == 4:
        cams_standard_t = cams_standard_t.squeeze(1)

    # XAI-Enhancer via optimized extractor
    enhancer_extractor = OptimizedCamExtractor(
        model=model,
        model_name="resnet50",
        conv_layers=target_layers["enhancer"],
        cam_method="HiResCAMEnhanced",
        device_preference=device.type,
        layer_batch_size=4,
    )
    _, cams_enhancer = enhancer_extractor.extract_saliency_map(
        input_data=batch_tensor, predicted_label=predicted_labels, use_cache=False
    )  # [B, H, W]

    # Ensure both methods are in [0,1]
    def _normalize_batch(hm: torch.Tensor) -> torch.Tensor:
        b = hm.shape[0]
        hm_flat = hm.view(b, -1)
        hm_min = hm_flat.min(dim=1).values.view(b, 1, 1)
        hm_max = hm_flat.max(dim=1).values.view(b, 1, 1)
        return (hm - hm_min) / (hm_max - hm_min + 1e-8)

    cams_standard_t = _normalize_batch(cams_standard_t)
    cams_enhancer = _normalize_batch(cams_enhancer)

    return cams_standard_t.cpu(), cams_enhancer.cpu()


def visualize_sample(
    idx: int,
    orig_img: np.ndarray,
    pert_img: np.ndarray,
    gradcam_orig: torch.Tensor,
    gradcam_pert: torch.Tensor,
    enh_orig: torch.Tensor,
    enh_pert: torch.Tensor,
    out_dir: str,
) -> None:
    """
    Create a side-by-side visualization for one sample:

        Row 1: Original image, Grad-CAM (orig), XAI-Enhancer (orig)
        Row 2: Perturbed image, Grad-CAM (pert), XAI-Enhancer (pert)
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Row 1: original
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(orig_img, alpha=0.5)
    axes[0, 1].imshow(gradcam_orig.numpy(), cmap="jet", alpha=0.5)
    axes[0, 1].set_title("Grad-CAM (orig)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(orig_img, alpha=0.5)
    axes[0, 2].imshow(enh_orig.numpy(), cmap="jet", alpha=0.5)
    axes[0, 2].set_title("XAI-Enhancer (orig)")
    axes[0, 2].axis("off")

    # Row 2: perturbed
    axes[1, 0].imshow(pert_img)
    axes[1, 0].set_title("Perturbed image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pert_img, alpha=0.5)
    axes[1, 1].imshow(gradcam_pert.numpy(), cmap="jet", alpha=0.5)
    axes[1, 1].set_title("Grad-CAM (pert)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(pert_img, alpha=0.5)
    axes[1, 2].imshow(enh_pert.numpy(), cmap="jet", alpha=0.5)
    axes[1, 2].set_title("XAI-Enhancer (pert)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"robustness_sample_{idx:03d}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run_robustness_experiment(cfg: RobustnessConfig) -> None:
    device = torch.device(get_device("cuda"))
    dataset = MedicalImageDataset(cfg.image_dir)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = get_resnet50_for_explainability(device)
    target_layers = get_resnet50_target_layers(model)

    gradcam_ssim_vals: List[float] = []
    gradcam_pearson_vals: List[float] = []
    enh_ssim_vals: List[float] = []
    enh_pearson_vals: List[float] = []

    vis_count = 0

    for batch_idx, (batch_tensors, batch_imgs_np, paths) in enumerate(loader):
        # batch_tensors: [B, C, H, W] (already normalized)
        b = batch_tensors.shape[0]

        # Create perturbed images in numpy space
        batch_imgs_np = np.stack(batch_imgs_np, axis=0)  # [B, H, W, C] in [0,1]
        pert_imgs_np = create_perturbations(batch_imgs_np, cfg)

        # Normalize both original and perturbed for the model
        orig_batch_norm = batch_tensors  # already normalized via transformations
        pert_batch_norm = normalize_batch_from_np(pert_imgs_np)

        orig_batch_norm = orig_batch_norm.to(device)
        pert_batch_norm = pert_batch_norm.to(device)

        # Use model predictions on original images as targets
        with torch.no_grad():
            logits = model(orig_batch_norm)
            preds = torch.argmax(logits, dim=1).tolist()

        # Compute heatmaps for original and perturbed images
        gradcam_orig, enh_orig = compute_heatmaps(
            model, target_layers, orig_batch_norm, device, predicted_labels=preds
        )
        gradcam_pert, enh_pert = compute_heatmaps(
            model, target_layers, pert_batch_norm, device, predicted_labels=preds
        )

        for i in range(b):
            g_o = gradcam_orig[i]
            g_p = gradcam_pert[i]
            e_o = enh_orig[i]
            e_p = enh_pert[i]

            # SSIM
            gradcam_ssim_vals.append(float(ssim_global(g_o, g_p)))
            enh_ssim_vals.append(float(ssim_global(e_o, e_p)))

            # Pearson
            gradcam_pearson_vals.append(float(pearson_corr(g_o, g_p)))
            enh_pearson_vals.append(float(pearson_corr(e_o, e_p)))

            # Visualize a limited number of samples
            if vis_count < cfg.max_visualizations:
                visualize_sample(
                    idx=vis_count,
                    orig_img=batch_imgs_np[i],
                    pert_img=pert_imgs_np[i],
                    gradcam_orig=g_o,
                    gradcam_pert=g_p,
                    enh_orig=e_o,
                    enh_pert=e_p,
                    out_dir=cfg.output_dir,
                )
                vis_count += 1

    # Aggregate metrics
    gradcam_ssim_mean = np.mean(gradcam_ssim_vals)
    enh_ssim_mean = np.mean(enh_ssim_vals)
    gradcam_pearson_mean = np.mean(gradcam_pearson_vals)
    enh_pearson_mean = np.mean(enh_pearson_vals)

    print("=== Robustness to Clinical Perturbations ===")
    print(f"Number of evaluated images: {len(gradcam_ssim_vals)}")
    print("")
    print("Method,SSIM_mean,Pearson_mean")
    print(f"Grad-CAM,{gradcam_ssim_mean:.4f},{gradcam_pearson_mean:.4f}")
    print(f"XAI-Enhancer,{enh_ssim_mean:.4f},{enh_pearson_mean:.4f}")
    print("")
    print(f"Visualization samples saved to: {cfg.output_dir}")


if __name__ == "__main__":
    # Example usage:
    #   python -m XAI_Enhancer_module.robustness_augmentations_xai /path/to/medical_images
    import argparse

    parser = argparse.ArgumentParser(description="Robustness of XAI-Enhancer vs Grad-CAM under clinical perturbations.")
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing medical images (will be scanned recursively).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for robustness evaluation.")
    parser.add_argument("--gaussian-sigma", type=float, default=0.05, help="Std of Gaussian sensor noise.")
    parser.add_argument(
        "--contrast-alpha",
        type=float,
        default=0.25,
        help="Contrast factor range around 1.0 (alpha in [1-alpha, 1+alpha]).",
    )
    parser.add_argument(
        "--brightness-beta",
        type=float,
        default=0.10,
        help="Brightness shift range around 0 (beta in [-beta, beta]).",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=3,
        help="Maximum number of visualization figures to save.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="robustness_outputs",
        help="Directory to save visualization figures.",
    )

    args = parser.parse_args()
    config = RobustnessConfig(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        gaussian_sigma=args.gaussian_sigma,
        contrast_alpha=args.contrast_alpha,
        brightness_beta=args.brightness_beta,
        max_visualizations=args.max_visualizations,
        output_dir=args.output_dir,
    )
    run_robustness_experiment(config)

