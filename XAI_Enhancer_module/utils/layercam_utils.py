"""
LayerCAM utilities -- architecture-aware stage-layer selection and multi-layer
fusion as described in:

    Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation Maps
    for Localization", IEEE TIP 2021.

Single-layer LayerCAM (Eqs. 6-8) is already implemented in
``pytorch_grad_cam.layer_cam.LayerCAM``.  This module adds:

* ``get_layercam_stage_layers`` -- per-architecture ordered list of one target
  layer per network stage (shallow -> deep).
* ``extract_layercam_fused`` -- multi-layer fusion (Eq. 9): run LayerCAM on
  each stage independently, apply tanh scaling to shallow stages, and combine
  via element-wise maximum.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam.layer_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ---------------------------------------------------------------------------
# Stage-layer resolution for every supported architecture
# ---------------------------------------------------------------------------

def get_layercam_stage_layers(model: nn.Module, arch: str) -> List[nn.Module]:
    """Return an **ordered** list of target layers (shallow -> deep), one per
    network "stage", suitable for hierarchical LayerCAM fusion.

    The mapping follows the conventions in the LayerCAM paper (VGG16
    stages = last conv before each max-pool; ResNet stages = layer1-4;
    DenseNet stages = last conv in each dense block).
    """
    arch = arch.lower()

    if arch in ("resnet18", "resnet34", "resnet50"):
        return [
            model.layer1[-1],   # Stage 1 (shallowest)
            model.layer2[-1],   # Stage 2
            model.layer3[-1],   # Stage 3
            model.layer4[-1],   # Stage 4 (deepest)
        ]

    if arch == "vgg16":
        f = model.features
        return [
            f[2],    # conv1_2  (Stage 1)
            f[7],    # conv2_2  (Stage 2)
            f[14],   # conv3_3  (Stage 3)
            f[21],   # conv4_3  (Stage 4)
            f[28],   # conv5_3  (Stage 5 -- deepest)
        ]

    if arch == "vgg19":
        f = model.features
        return [
            f[2],    # conv1_2  (Stage 1)
            f[7],    # conv2_2  (Stage 2)
            f[16],   # conv3_4  (Stage 3)
            f[25],   # conv4_4  (Stage 4)
            f[34],   # conv5_4  (Stage 5 -- deepest)
        ]

    if arch == "densenet121":
        feat = model.features
        return [
            feat.denseblock1.denselayer6.conv2,    # Stage 1
            feat.denseblock2.denselayer12.conv2,   # Stage 2
            feat.denseblock3.denselayer24.conv2,   # Stage 3
            feat.denseblock4.denselayer16.conv2,   # Stage 4 (deepest)
        ]

    raise ValueError(
        f"Unsupported architecture for LayerCAM stage layers: {arch}"
    )


# ---------------------------------------------------------------------------
# Multi-layer fusion (paper Eq. 9)
# ---------------------------------------------------------------------------

def _tanh_scale(cam: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """Apply the tanh scaling from Eq. 9:  tanh(gamma * M / max(M))."""
    max_val = cam.max()
    if max_val < 1e-8:
        return cam
    return np.tanh(gamma * cam / max_val)


def extract_layercam_fused(
    model: nn.Module,
    stage_layers: List[nn.Module],
    input_tensor: torch.Tensor,
    predicted_label: int,
    gamma: float = 2.0,
    fuse_stages: Optional[List[int]] = None,
) -> np.ndarray:
    """Generate a fused LayerCAM saliency map following the paper's Eq. 9.

    Parameters
    ----------
    model : nn.Module
        The classification model (already on the correct device, in eval mode).
    stage_layers : list[nn.Module]
        Ordered target layers returned by ``get_layercam_stage_layers``
        (shallow -> deep).
    input_tensor : torch.Tensor
        Pre-processed input image, shape ``[1, C, H, W]``.
    predicted_label : int
        Target class index.
    gamma : float
        Scaling factor for tanh normalisation of shallow layers (default 2,
        per the paper's Table III).
    fuse_stages : list[int] or None
        **1-based** stage indices to include.  ``None`` selects all stages
        except stage 1 for VGG (5-stage architectures) or all stages for
        4-stage architectures -- matching the paper's best configurations.

    Returns
    -------
    np.ndarray
        Fused saliency map, shape ``[H, W]``, values in [0, 1].
    """
    num_stages = len(stage_layers)

    if fuse_stages is None:
        if num_stages == 5:
            # VGG: paper Table X -- omit stage 1 (lacks class discrimination)
            fuse_stages = [2, 3, 4, 5]
        else:
            # ResNet / DenseNet: use all stages
            fuse_stages = list(range(1, num_stages + 1))

    # Validate
    for s in fuse_stages:
        if s < 1 or s > num_stages:
            raise ValueError(
                f"fuse_stages contains {s}, but model only has stages 1..{num_stages}"
            )

    deepest_stage = max(fuse_stages)
    targets = [ClassifierOutputTarget(predicted_label)]
    _, _, h, w = input_tensor.shape

    fused: Optional[np.ndarray] = None

    for stage_idx in fuse_stages:
        target_layer = stage_layers[stage_idx - 1]  # 1-based -> 0-based

        with LayerCAM(model=model, target_layers=[target_layer]) as cam_obj:
            grayscale_cam = cam_obj(input_tensor=input_tensor, targets=targets)

        # grayscale_cam shape: [batch=1, H_out, W_out]
        cam_2d = grayscale_cam[0]  # [H_out, W_out]

        # Upscale to input resolution if needed (shallow layers are larger,
        # deep layers are smaller -- scale_cam_image in base_cam already
        # handles this, but we double-check).
        if cam_2d.shape != (h, w):
            import cv2
            cam_2d = cv2.resize(cam_2d, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply tanh scaling to all stages except the deepest (Eq. 9)
        if stage_idx != deepest_stage:
            cam_2d = _tanh_scale(cam_2d, gamma)

        # Combine via element-wise maximum
        if fused is None:
            fused = cam_2d.copy()
        else:
            fused = np.maximum(fused, cam_2d)

    # Final normalisation to [0, 1]
    if fused is not None:
        fused = fused - fused.min()
        max_val = fused.max()
        if max_val > 1e-8:
            fused = fused / max_val

    return fused
