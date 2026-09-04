"""
Layer-set selectors for CAM evaluation (revision protocol R1 / D1 / D10).

Modes
-----
all            — every Conv2d/Conv1d/Conv3d module
conv3x3        — Conv2d with kernel_size == (3, 3)
block_outputs  — ResNet: each bottleneck/basic block; VGG: each conv;
                 DenseNet: each denselayer conv2
stage_outputs  — one layer per network stage (same as LayerCAM stages)
last_5         — last 5 conv modules
last           — final conv module
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


LAYER_SET_CHOICES = (
    "all",
    "conv3x3",
    "block_outputs",
    "stage_outputs",
    "last_5",
    "last",
)


def _all_convs(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    out: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            out.append((name, module))
    return out


def _resnet_block_outputs(model: nn.Module) -> List[nn.Module]:
    layers = []
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        if not hasattr(model, stage_name):
            continue
        stage = getattr(model, stage_name)
        for block in stage:
            layers.append(block)
    return layers


def _vgg_block_outputs(model: nn.Module) -> List[nn.Module]:
    """Every conv in features (each is a 'block' for VGG)."""
    if not hasattr(model, "features"):
        return []
    return [m for m in model.features if isinstance(m, nn.Conv2d)]


def _densenet_block_outputs(model: nn.Module) -> List[nn.Module]:
    """Last 3x3 conv (conv2) of each DenseLayer."""
    layers = []
    if not hasattr(model, "features"):
        return layers
    for name, module in model.features.named_modules():
        if name.endswith("conv2") and isinstance(module, nn.Conv2d):
            layers.append(module)
    return layers


def _stage_outputs(model: nn.Module, arch: str) -> List[nn.Module]:
    from XAI_Enhancer_module.utils.layercam_utils import get_layercam_stage_layers

    return get_layercam_stage_layers(model, arch)


def select_cam_layers(
    model: nn.Module,
    layer_set: str,
    arch: str | None = None,
) -> List[nn.Module]:
    """
    Return target modules for the given layer-set name.

    ``arch`` is required for ``stage_outputs`` (and used as a hint for
    block_outputs when the backbone family is ambiguous).
    """
    layer_set = (layer_set or "last").lower().strip()
    if layer_set not in LAYER_SET_CHOICES:
        raise ValueError(
            f"Invalid layer_set '{layer_set}'. "
            f"Must be one of {list(LAYER_SET_CHOICES)}"
        )

    arch_l = (arch or getattr(model, "model_name", "") or "").lower()
    named = _all_convs(model)
    if not named:
        raise ValueError("No convolutional layers found in model")

    all_modules = [m for _, m in named]

    if layer_set == "all":
        return all_modules

    if layer_set == "last":
        return [all_modules[-1]]

    if layer_set == "last_5":
        return all_modules[-5:] if len(all_modules) >= 5 else all_modules

    if layer_set == "conv3x3":
        selected = [
            m
            for m in all_modules
            if isinstance(m, nn.Conv2d) and tuple(m.kernel_size) == (3, 3)
        ]
        if not selected:
            raise ValueError("No 3x3 Conv2d layers found for layer_set=conv3x3")
        return selected

    if layer_set == "stage_outputs":
        if not arch_l:
            raise ValueError("arch is required for layer_set=stage_outputs")
        return _stage_outputs(model, arch_l)

    # block_outputs
    if arch_l.startswith("resnet") or hasattr(model, "layer4"):
        selected = _resnet_block_outputs(model)
    elif arch_l.startswith("densenet") or (
        hasattr(model, "features") and hasattr(getattr(model, "features", None), "denseblock1")
    ):
        selected = _densenet_block_outputs(model)
    elif arch_l.startswith("vgg") or hasattr(model, "features"):
        selected = _vgg_block_outputs(model)
    else:
        selected = all_modules

    if not selected:
        raise ValueError(f"No layers found for layer_set=block_outputs (arch={arch_l!r})")
    return selected
