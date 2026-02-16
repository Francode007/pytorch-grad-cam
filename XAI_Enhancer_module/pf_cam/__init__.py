"""
PF-CAM (Pyramid Fusion CAM) — Standalone Module

A fully independent multi-layer CAM aggregation method that combines
saliency maps from multiple convolutional layers using hierarchical
top-down fusion with soft gating.

This module has ZERO dependency on enhanced_cams/ or OptimizedCamExtractor.
It uses standard pytorch-grad-cam GradCAM directly.
"""

from XAI_Enhancer_module.pf_cam.normalization import NormStrategy, normalize_activations
from XAI_Enhancer_module.pf_cam.aggregator import PFCamAggregator
from XAI_Enhancer_module.pf_cam.extractor import PFCamExtractor
from XAI_Enhancer_module.pf_cam.weight_logger import WeightLogger

__all__ = [
    "PFCamExtractor",
    "PFCamAggregator",
    "NormStrategy",
    "normalize_activations",
    "WeightLogger",
]
