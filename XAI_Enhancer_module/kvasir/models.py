"""
Build and load Kvasir classification models (ImageNet-pretrained backbone + Kvasir head).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Optional

from XAI_Enhancer_module.kvasir.data import KVASIR_NUM_CLASSES

# Support archs that match existing CAM/extractor usage
ARCH_HEAD = {
    "resnet50": ("fc", 2048),
    "resnet18": ("fc", 512),
    "resnet34": ("fc", 512),
    "densenet121": ("classifier", 1024),
}


def build_kvasir_model(
    arch: str,
    num_classes: int = KVASIR_NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build model for Kvasir: ImageNet backbone + linear head for num_classes.
    """
    arch = arch.lower()
    if arch not in ARCH_HEAD:
        raise ValueError(f"Unsupported arch: {arch}. Supported: {list(ARCH_HEAD.keys())}")
    head_name, in_features = ARCH_HEAD[arch]

    if arch == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(arch)
    return model


def load_kvasir_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """
    Load checkpoint into model. Handles both raw state_dict and dict with 'model_state_dict' / 'state_dict'.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    # Checkpoints saved from torch.compile() have keys prefixed with "_orig_mod."
    if state and any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
    return model
