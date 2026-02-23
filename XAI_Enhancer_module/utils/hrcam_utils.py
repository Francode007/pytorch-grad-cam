"""
HR-CAM utilities -- PyTorch port of High Resolution Class Activation Maps as
described in:

    Shinde et al., "HR-CAM: Precise Localization of pathology using
    multi-level learning in CNNs", MICCAI 2019.

HR-CAM aggregates feature maps from multiple CNN layers.  A lightweight
trainable classification head (GAP per layer -> concatenate -> Dense) learns
per-channel importance weights while the backbone stays frozen.  The final
CAM is:

    A = sum_i( W_i^c * upsample(f_i) )

where f_i are feature maps from selected layers, W_i^c are learned Dense-layer
weights for class c, and all maps are bilinearly upsampled to input resolution
before summation.

This module provides:

* ``HRCAMHead``        -- nn.Module wrapping frozen backbone + trainable head.
* ``train_hrcam_head`` -- train the head on a given DataLoader.
* ``extract_hrcam``    -- generate an HR-CAM saliency map from a trained head.

Layer selection is delegated to ``get_layercam_stage_layers`` from
``layercam_utils`` (same pre-downsampling layer endpoints).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# HRCAMHead -- frozen backbone + hooks + GAP + Linear
# ---------------------------------------------------------------------------

class HRCAMHead(nn.Module):
    """Wraps an existing (frozen) backbone with forward hooks on selected
    stage layers and a single trainable ``nn.Linear`` classifier on their
    globally-average-pooled feature maps.

    Parameters
    ----------
    backbone : nn.Module
        Pre-trained classification model (weights will be frozen).
    stage_layers : list[nn.Module]
        Ordered target layers (shallow -> deep) whose outputs will be hooked.
        Typically obtained via ``get_layercam_stage_layers``.
    num_classes : int
        Number of output classes for the HR-CAM head.
    input_size : tuple[int, int]
        Spatial size ``(H, W)`` of a single input tensor used for the dummy
        forward pass that determines per-layer channel counts.  Default 224x224.
    """

    def __init__(
        self,
        backbone: nn.Module,
        stage_layers: List[nn.Module],
        num_classes: int,
        input_size: tuple = (224, 224),
    ):
        super().__init__()
        self.backbone = backbone
        self.stage_layers = list(stage_layers)
        self.num_classes = num_classes

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Storage populated by hooks during forward pass
        self._feature_maps: Dict[int, torch.Tensor] = {}
        self._hooks: list = []

        for idx, layer in enumerate(self.stage_layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

        # Dummy forward to discover per-layer channel counts
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_size, device=device)
            self.backbone(dummy)

        self._layer_channels: List[int] = []
        total_channels = 0
        for idx in range(len(self.stage_layers)):
            c = self._feature_maps[idx].shape[1]
            self._layer_channels.append(c)
            total_channels += c
        self.total_channels = total_channels

        self.classifier = nn.Linear(total_channels, num_classes)

    # -- hook factory -----------------------------------------------------

    def _make_hook(self, idx: int):
        def _hook(module, inp, out):
            self._feature_maps[idx] = out
        return _hook

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._feature_maps = {}
        self.backbone(x)

        gaps = []
        for idx in range(len(self.stage_layers)):
            fm = self._feature_maps[idx]                      # [B, C_i, H_i, W_i]
            gap = F.adaptive_avg_pool2d(fm, 1).flatten(1)     # [B, C_i]
            gaps.append(gap)

        concatenated = torch.cat(gaps, dim=1)                 # [B, N]
        return self.classifier(concatenated)                  # [B, num_classes]

    # -- cleanup ----------------------------------------------------------

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_hrcam_head(
    backbone: nn.Module,
    stage_layers: List[nn.Module],
    train_loader: DataLoader,
    num_classes: int,
    *,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 20,
    lr: float = 3e-4,
    device: torch.device | str = "cpu",
    save_path: Optional[str] = None,
) -> HRCAMHead:
    """Build and train an :class:`HRCAMHead`.

    Only the ``classifier`` (single Linear layer) is trained; the backbone
    remains frozen throughout.

    Parameters
    ----------
    backbone, stage_layers, num_classes
        Forwarded to :class:`HRCAMHead`.
    train_loader : DataLoader
        Yields ``(images, labels, ...)`` batches.
    val_loader : DataLoader, optional
        If provided, validation accuracy is reported each epoch and the best
        checkpoint is kept.
    epochs, lr : int, float
        Training hyper-parameters.
    device : str or torch.device
    save_path : str, optional
        If given the best (or final) head state-dict is saved here.

    Returns
    -------
    HRCAMHead
        The trained head (in eval mode, on *device*).
    """
    device = torch.device(device) if isinstance(device, str) else device
    backbone.to(device).eval()

    head = HRCAMHead(backbone, stage_layers, num_classes).to(device)
    head.backbone.eval()

    optimizer = torch.optim.Adam(head.classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # -- train --------------------------------------------------------
        head.train()
        head.backbone.eval()  # keep BN in eval mode
        running_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"HR-CAM train {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images, labels = batch[0].to(device), batch[1].to(device)
            logits = head(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n += images.size(0)
            pbar.set_postfix(loss=f"{running_loss / n:.4f}")
        epoch_loss = running_loss / n if n else 0.0
        scheduler.step(epoch_loss)

        # -- validate -----------------------------------------------------
        if val_loader is not None:
            head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    images, labels = batch[0].to(device), batch[1].to(device)
                    preds = head(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total if total else 0.0
            print(f"  Epoch {epoch}: loss={epoch_loss:.4f}  val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in head.classifier.state_dict().items()}
        else:
            print(f"  Epoch {epoch}: loss={epoch_loss:.4f}")
            best_state = {k: v.cpu().clone() for k, v in head.classifier.state_dict().items()}

    # Restore best classifier weights
    if best_state is not None:
        head.classifier.load_state_dict(best_state)
    head.to(device).eval()

    if save_path is not None:
        torch.save({"classifier_state_dict": head.classifier.state_dict(),
                     "layer_channels": head._layer_channels,
                     "total_channels": head.total_channels,
                     "num_classes": num_classes}, save_path)
        print(f"HR-CAM head saved to {save_path}")

    return head


# ---------------------------------------------------------------------------
# Load a previously saved HR-CAM head
# ---------------------------------------------------------------------------

def load_hrcam_head(
    backbone: nn.Module,
    stage_layers: List[nn.Module],
    num_classes: int,
    checkpoint_path: str,
    device: torch.device | str = "cpu",
) -> HRCAMHead:
    """Reconstruct an :class:`HRCAMHead` and load a saved classifier."""
    device = torch.device(device) if isinstance(device, str) else device
    backbone.to(device).eval()
    head = HRCAMHead(backbone, stage_layers, num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("classifier_state_dict", ckpt)
    head.classifier.load_state_dict(state)
    head.eval()
    print(f"HR-CAM head loaded from {checkpoint_path}")
    return head


# ---------------------------------------------------------------------------
# HR-CAM saliency map extraction (paper Eq. 3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_hrcam(
    hrcam_head: HRCAMHead,
    input_tensor: torch.Tensor,
    predicted_label: int,
) -> np.ndarray:
    """Generate an HR-CAM saliency map for a single image.

    Parameters
    ----------
    hrcam_head : HRCAMHead
        Trained HR-CAM head (in eval mode).
    input_tensor : torch.Tensor
        Pre-processed input, shape ``[1, C, H, W]``.
    predicted_label : int
        Target class index.

    Returns
    -------
    np.ndarray
        Saliency map of shape ``(H, W)`` with values in [0, 1].
    """
    hrcam_head.eval()
    device = next(hrcam_head.parameters()).device
    input_tensor = input_tensor.to(device)
    _, _, h, w = input_tensor.shape

    # Forward pass to populate hooked feature maps
    hrcam_head._feature_maps = {}
    hrcam_head.backbone(input_tensor)

    # Weights from the trained classifier for the target class
    # classifier.weight shape: [num_classes, total_channels]
    class_weights = hrcam_head.classifier.weight[predicted_label]  # [total_channels]

    cam = torch.zeros(h, w, device=device)
    offset = 0

    for idx in range(len(hrcam_head.stage_layers)):
        fm = hrcam_head._feature_maps[idx]  # [1, C_i, H_i, W_i]
        c_i = hrcam_head._layer_channels[idx]
        layer_w = class_weights[offset:offset + c_i]  # [C_i]
        offset += c_i

        # Weighted sum across channels: [C_i] x [1, C_i, H_i, W_i] -> [1, 1, H_i, W_i]
        weighted = (fm[0] * layer_w[:, None, None]).sum(dim=0, keepdim=True)  # [H_i, W_i] keepdim->[1, H_i, W_i]

        # Bilinear upsample to input resolution
        upsampled = F.interpolate(
            weighted.unsqueeze(0),  # [1, 1, H_i, W_i]
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # [H, W]

        cam = cam + upsampled

    # ReLU + normalise to [0, 1]
    cam = torch.relu(cam)
    cam_np = cam.cpu().numpy()
    cam_min = cam_np.min()
    cam_max = cam_np.max()
    if cam_max - cam_min > 1e-8:
        cam_np = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_np = np.zeros_like(cam_np)

    return cam_np
