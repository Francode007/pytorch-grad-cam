"""
PF-CAM Aggregator — Pyramid Fusion CAM aggregation strategies.

Standalone aggregation module for combining CAM maps from multiple layers
using stage-based grouping, Top-K selection, and top-down soft gating.

This file is self-contained with no dependencies on enhanced_cams/ or
OptimizedCamExtractor.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict


class PFCamAggregator:
    """
    Aggregator for combining CAMs from multiple layers using PF-CAM methodology.

    Supports:
    - Standard Softmax (baseline)
    - Temperature Scaling
    - Top-K Sparsity
    - Stagewise (hierarchical)
    - Pyramid Fusion (recommended: stage-based Top-K + top-down soft gating)
    """

    @staticmethod
    def aggregate_standard(cams: List[torch.Tensor], scores: np.ndarray) -> torch.Tensor:
        """Baseline: Softmax(scores) weighted sum of CAMs."""
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        weights = F.softmax(scores_tensor, dim=0)
        final_cam = torch.zeros_like(cams[0])
        for i, cam in enumerate(cams):
            final_cam += weights[i] * cam
        return final_cam

    @staticmethod
    def aggregate_temperature(
        cams: List[torch.Tensor], scores: np.ndarray, temp: float = 0.1
    ) -> torch.Tensor:
        """Temperature scaling: Softmax(scores / temp) weighted sum."""
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        weights = F.softmax(scores_tensor / temp, dim=0)
        final_cam = torch.zeros_like(cams[0])
        for i, cam in enumerate(cams):
            final_cam += weights[i] * cam
        return final_cam

    @staticmethod
    def aggregate_top_k(
        cams: List[torch.Tensor],
        scores: np.ndarray,
        k: int = 5,
        soft: bool = True,
        temp: float = 1.0,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Top-K sparsity: keep only the top K layers.

        Returns:
            Tuple of (aggregated CAM, list of selected indices)
        """
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        k = min(k, len(scores))
        topk_values, topk_indices = torch.topk(scores_tensor, k=k)

        full_weights = torch.zeros_like(scores_tensor)
        topk_weights = F.softmax(topk_values / temp, dim=0)
        full_weights[topk_indices] = topk_weights

        final_cam = torch.zeros_like(cams[0])
        for i, cam in enumerate(cams):
            if full_weights[i] > 0:
                final_cam += full_weights[i] * cam

        return final_cam, topk_indices.tolist()

    @staticmethod
    def aggregate_pyramid_fusion(
        cams: List[torch.Tensor],
        scores: np.ndarray,
        layer_shapes: List[Tuple[int, int]],
        config: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Pyramid Fusion CAM (PF-CAM).

        1. Group layers by Stage (spatial resolution).
        2. Within each stage, select Top-K layers.
        3. Normalize per-stage CAMs.
        4. Fuse top-down (deep → shallow) using soft gating.

        Args:
            cams: List of per-layer CAM tensors (all scaled to same size)
            scores: Array of cosine similarity scores per layer
            layer_shapes: Original spatial dims (H, W) per layer
            config: Dict with keys: beta, k_percent, k_min, temp

        Returns:
            Tuple of (final fused CAM, diagnostics dict with stage info)
        """
        # 1. Group layers by spatial resolution (stage detection)
        stages: Dict[Tuple[int, int], List[int]] = {}
        for idx, shape in enumerate(layer_shapes):
            if shape not in stages:
                stages[shape] = []
            stages[shape].append(idx)

        # Sort stages by resolution: smallest (deepest) first
        sorted_shapes = sorted(stages.keys(), key=lambda s: s[0] * s[1])

        stage_cams = []
        stage_scores_list = []
        stage_selected_indices: List[List[int]] = []

        # Extract config
        k_percent = config.get("k_percent", 0.15)
        k_min = config.get("k_min", 2)
        temp = config.get("temp", 0.1)
        beta = config.get("beta", 0.4)

        # 2. Process each stage
        for shape in sorted_shapes:
            indices = stages[shape]
            stage_layer_scores = scores[indices]
            stage_layer_cams = [cams[i] for i in indices]

            # Top-K selection per stage
            num_layers = len(indices)
            k = max(k_min, int(num_layers * k_percent))
            k = min(k, num_layers)

            # Aggregate within stage using Top-K
            stage_cam, selected_local = PFCamAggregator.aggregate_top_k(
                stage_layer_cams, stage_layer_scores, k=k, soft=True, temp=temp
            )
            # Map local selected indices back to global
            selected_global = [indices[li] for li in selected_local]
            stage_selected_indices.append(selected_global)

            # Energy normalization
            if stage_cam.max() > 1e-7:
                stage_cam = stage_cam / stage_cam.max()

            stage_cams.append(stage_cam)
            stage_scores_list.append(float(np.mean(stage_layer_scores)))

        # 3. Top-down fusion with soft gating
        # stage_cams[0] = deepest (smallest spatial), stage_cams[-1] = shallowest
        fused_cam = stage_cams[0]

        for i in range(1, len(stage_cams)):
            next_stage_cam = stage_cams[i]  # shallower

            # Gate from current fused (deep)
            gate = fused_cam.clone()
            if gate.max() > 1e-7:
                gate = gate / gate.max()

            # Soft gating: signal * (β + (1-β)*gate)
            soft_gate = beta + (1.0 - beta) * gate
            masked_details = next_stage_cam * soft_gate

            fused_cam = fused_cam + masked_details

            # Renormalize for next iteration
            if fused_cam.max() > 1e-7:
                fused_cam = fused_cam / fused_cam.max()

        # Compute stage weights for diagnostics
        stage_scores_tensor = torch.tensor(stage_scores_list, dtype=torch.float32)
        stage_weights = F.softmax(stage_scores_tensor / temp, dim=0).tolist()

        diagnostics = {
            "stage_shapes": sorted_shapes,
            "stage_scores": stage_scores_list,
            "stage_weights": stage_weights,
            "stage_selected_indices": stage_selected_indices,
            "num_stages": len(sorted_shapes),
        }

        return fused_cam, diagnostics

    @staticmethod
    def aggregate(
        cams: List[torch.Tensor],
        scores: np.ndarray,
        layer_shapes: List[Tuple[int, int]],
        config: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Dispatch to the appropriate aggregation method.

        Args:
            config: Must include "type" key. One of:
                "standard", "temp", "topk", "pyramid"

        Returns:
            Tuple of (final CAM, diagnostics dict)
        """
        method = config.get("type", "standard")

        if method == "pyramid":
            return PFCamAggregator.aggregate_pyramid_fusion(
                cams, scores, layer_shapes, config
            )
        elif method == "topk":
            cam, indices = PFCamAggregator.aggregate_top_k(
                cams, scores,
                k=config.get("k", 5),
                soft=config.get("soft", True),
                temp=config.get("temp", 1.0),
            )
            return cam, {"selected_indices": indices}
        elif method == "temp":
            cam = PFCamAggregator.aggregate_temperature(
                cams, scores, temp=config.get("temp", 1.0)
            )
            return cam, {}
        else:
            cam = PFCamAggregator.aggregate_standard(cams, scores)
            return cam, {}
