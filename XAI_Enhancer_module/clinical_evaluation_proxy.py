import os
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor


@dataclass
class ClinicalEvalConfig:
    kvasir_seg_dir: str
    kvasir_model_arch: str = "resnet50"
    kvasir_checkpoint_path: str = ""
    batch_size: int = 4
    num_workers: int = 0
    image_size: int = 224
    csv_output_path: str = "clinical_eval_results.csv"
    max_visualizations: int = 5
    viz_output_dir: str = "clinical_eval_visualizations"


class KvasirSegDataset(Dataset):
    """
    Dataset for Kvasir-SEG style folders.

    Expected directory structure (typical Kvasir-SEG):
        root/
          images/
            xxx.png
          masks/
            xxx.png

    The same stem name is used to pair images and masks.
    """

    def __init__(self, root_dir: str, image_size: int = 224):
        self.root_dir = root_dir
        self.image_size = image_size

        images_dir = os.path.join(root_dir, "images")
        masks_dir = os.path.join(root_dir, "masks")

        if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
            raise RuntimeError(
                f"Kvasir-SEG directory should contain 'images' and 'masks' subfolders. "
                f"Got: images_dir={images_dir}, masks_dir={masks_dir}"
            )

        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []

        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                continue
            image_path = os.path.join(images_dir, fname)
            mask_path = os.path.join(masks_dir, fname)
            if not os.path.exists(mask_path):
                # Try with common alternative suffixes if needed
                base, _ = os.path.splitext(fname)
                candidates = [
                    os.path.join(masks_dir, base + "_mask.png"),
                    os.path.join(masks_dir, base + "_mask.jpg"),
                    os.path.join(masks_dir, base + ".png"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        mask_path = c
                        break
            if not os.path.exists(mask_path):
                continue

            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)

        if not self.image_paths:
            raise RuntimeError(f"No valid image/mask pairs found under {root_dir}")

        # Basic normalization for ResNet-50 (ImageNet stats)
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size))
        img_rgb_float = img_rgb.astype(np.float32) / 255.0

        # For visualization later, keep a copy of resized RGB [H,W,C] in [0,1]
        vis_img = img_rgb_float.copy()

        # Apply normalization transform
        img_tensor = self.image_transform(img_rgb_float)  # [C,H,W]

        # Load mask (binary)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        # Ensure binary: threshold at >0
        _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        mask_bin = (mask_bin > 0).astype(np.uint8)  # [H,W] in {0,1}

        return img_tensor, mask_bin, vis_img, os.path.basename(img_path)


class GradCAMWrapper:
    """
    Wrapper around pytorch-grad-cam's GradCAM to match the
    .generate_map(input_tensor) -> np.ndarray[H,W] interface.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def generate_map(self, input_tensor: torch.Tensor) -> np.ndarray:
        # input_tensor: [1,C,H,W] already on device
        with torch.no_grad():
            preds = self.model(input_tensor)
            target_class = int(preds.argmax(dim=1).item())
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.astype(np.float32)


class XAIEnhancerWrapper:
    """
    Wrapper around the OptimizedCamExtractor-based XAI-Enhancer to expose
    .generate_map(input_tensor) -> np.ndarray[H,W] on a single image.
    Uses all convolutional layers for the enhancer (as in your experiments).
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        conv_layers: List[nn.Module] = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                conv_layers.append(module)
        if not conv_layers:
            raise ValueError("No convolutional layers found in model for XAI-Enhancer.")
        self.extractor = OptimizedCamExtractor(
            model=self.model,
            model_name="resnet50",
            conv_layers=conv_layers,
            cam_method="HiResCAMEnhanced",
            device_preference="cuda" if device.type == "cuda" else "cpu",
            layer_batch_size=16,
        )

    def generate_map(self, input_tensor: torch.Tensor) -> np.ndarray:
        # input_tensor: [1,C,H,W] on device
        with torch.no_grad():
            preds = self.model(input_tensor)
            target_class = int(preds.argmax(dim=1).item())
        _, saliency_map = self.extractor.extract_saliency_map(
            input_data=input_tensor, predicted_label=target_class, use_cache=False
        )
        heatmap_t = saliency_map[0]
        heatmap = heatmap_t.cpu().numpy().astype(np.float32)
        heatmap = np.clip(heatmap, 0.0, 1.0)
        return heatmap


def otsu_binarize(heatmap: np.ndarray) -> np.ndarray:
    """
    Binarize a continuous [0,1] 2D heatmap using Otsu's thresholding.
    Returns binary mask in {0,1}.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got shape {heatmap.shape}")

    # Scale to [0,255] uint8 for OpenCV
    heat_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(heat_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (thresh > 0).astype(np.uint8)


def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray, eps: float = 1e-8) -> float:
    """
    Intersection over Union between two binary masks.
    Both masks must be {0,1} and same shape.
    """
    if mask_pred.shape != mask_gt.shape:
        raise ValueError(f"Mask shapes do not match: {mask_pred.shape} vs {mask_gt.shape}")

    mask_pred = mask_pred.astype(bool)
    mask_gt = mask_gt.astype(bool)

    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()

    return float(intersection / (union + eps))


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray, eps: float = 1e-8) -> float:
    """
    Dice Coefficient between two binary masks.
    """
    if mask_pred.shape != mask_gt.shape:
        raise ValueError(f"Mask shapes do not match: {mask_pred.shape} vs {mask_gt.shape}")

    mask_pred = mask_pred.astype(bool)
    mask_gt = mask_gt.astype(bool)

    intersection = np.logical_and(mask_pred, mask_gt).sum()
    size_pred = mask_pred.sum()
    size_gt = mask_gt.sum()

    return float((2.0 * intersection) / (size_pred + size_gt + eps))


def get_kvasir_model_and_target_layer(
    arch: str, checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, nn.Module]:
    """
    Build the Kvasir classification model, load trained weights,
    and return (model, last_conv_layer_for_gradcam).
    """
    model = build_kvasir_model(arch=arch, pretrained=True)
    if checkpoint_path:
        model = load_kvasir_checkpoint(model, checkpoint_path, device=device)
    model.to(device)
    model.eval()

    last_conv: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Could not find a Conv2d layer for Grad-CAM target.")
    return model, last_conv


def visualize_sample(
    idx: int,
    vis_img: np.ndarray,
    mask_gt: np.ndarray,
    grad_bin: np.ndarray,
    enh_bin: np.ndarray,
    out_dir: str,
) -> None:
    """
    Save a 1x4 grid: Original, Ground Truth, Grad-CAM binarized, XAI-Enhancer binarized.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    axes[0].imshow(vis_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask_gt, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(grad_bin, cmap="gray")
    axes[2].set_title("Grad-CAM Mask")
    axes[2].axis("off")

    axes[3].imshow(enh_bin, cmap="gray")
    axes[3].set_title("XAI-Enhancer Mask")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"clinical_eval_sample_{idx:03d}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run_clinical_evaluation(cfg: ClinicalEvalConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KvasirSegDataset(cfg.kvasir_seg_dir, image_size=cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model, target_layer = get_kvasir_model_and_target_layer(
        arch=cfg.kvasir_model_arch,
        checkpoint_path=cfg.kvasir_checkpoint_path,
        device=device,
    )

    gradcam = GradCAMWrapper(model, target_layer, device)
    enhancer = XAIEnhancerWrapper(model, device)

    grad_iou_vals: List[float] = []
    grad_dice_vals: List[float] = []
    enh_iou_vals: List[float] = []
    enh_dice_vals: List[float] = []

    vis_count = 0

    for batch_idx, (imgs_tensor, masks_gt, vis_imgs, names) in enumerate(loader):
        imgs_tensor = imgs_tensor.to(device)  # [B,C,H,W]

        with torch.no_grad():
            _ = model(imgs_tensor)  # forward pass, if needed for real CAMs

        b = imgs_tensor.shape[0]

        for i in range(b):
            img_single = imgs_tensor[i].unsqueeze(0)  # [1,C,H,W]
            mask_gt = masks_gt[i].numpy().astype(np.uint8)
            vis_img = vis_imgs[i].numpy()

            # Generate continuous heatmaps in [0,1]
            heat_grad = gradcam.generate_map(img_single)
            heat_enh = enhancer.generate_map(img_single)

            # Binarize via Otsu
            bin_grad = otsu_binarize(heat_grad)
            bin_enh = otsu_binarize(heat_enh)

            # Compute IoU and Dice vs expert mask
            grad_iou_vals.append(compute_iou(bin_grad, mask_gt))
            grad_dice_vals.append(compute_dice(bin_grad, mask_gt))
            enh_iou_vals.append(compute_iou(bin_enh, mask_gt))
            enh_dice_vals.append(compute_dice(bin_enh, mask_gt))

            # Visualization for first N samples
            if vis_count < cfg.max_visualizations:
                visualize_sample(
                    idx=vis_count,
                    vis_img=vis_img,
                    mask_gt=mask_gt,
                    grad_bin=bin_grad,
                    enh_bin=bin_enh,
                    out_dir=cfg.viz_output_dir,
                )
                vis_count += 1

    grad_miou = float(np.mean(grad_iou_vals)) if grad_iou_vals else 0.0
    grad_mdice = float(np.mean(grad_dice_vals)) if grad_dice_vals else 0.0
    enh_miou = float(np.mean(enh_iou_vals)) if enh_iou_vals else 0.0
    enh_mdice = float(np.mean(enh_dice_vals)) if enh_dice_vals else 0.0

    print("=== Clinical Evaluation Proxy Results ===")
    print(f"Samples evaluated: {len(grad_iou_vals)}")
    print("")
    print("Method,Mean_IoU,Mean_Dice")
    print(f"Grad-CAM,{grad_miou:.4f},{grad_mdice:.4f}")
    print(f"XAI-Enhancer,{enh_miou:.4f},{enh_mdice:.4f}")

    # Save to CSV
    with open(cfg.csv_output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Mean_IoU", "Mean_Dice"])
        writer.writerow(["Grad-CAM", grad_miou, grad_mdice])
        writer.writerow(["XAI-Enhancer", enh_miou, enh_mdice])

    print(f"\nResults saved to: {cfg.csv_output_path}")
    print(f"Visualization samples saved to: {cfg.viz_output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clinical Evaluation Proxy: Compare Grad-CAM vs XAI-Enhancer against expert masks."
    )
    parser.add_argument(
        "kvasir_seg_dir",
        type=str,
        help="Path to Kvasir-SEG root directory with 'images' and 'masks' subfolders.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        help="Kvasir classification backbone architecture (e.g., resnet50, resnet18).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to trained Kvasir model checkpoint (.pth).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution (square).")
    parser.add_argument(
        "--csv-output",
        type=str,
        default="clinical_eval_results.csv",
        help="Path to CSV file where summary metrics are stored.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=5,
        help="Number of qualitative examples to save.",
    )
    parser.add_argument(
        "--viz-output-dir",
        type=str,
        default="clinical_eval_visualizations",
        help="Directory to save visualization grids.",
    )

    args = parser.parse_args()

    config = ClinicalEvalConfig(
        kvasir_seg_dir=args.kvasir_seg_dir,
        kvasir_model_arch=args.arch,
        kvasir_checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        image_size=args.image_size,
        csv_output_path=args.csv_output,
        max_visualizations=args.max_visualizations,
        viz_output_dir=args.viz_output_dir,
    )

    run_clinical_evaluation(config)

