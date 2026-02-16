"""
Weight Logger for PF-CAM diagnostics.

Records per-image layer weights, stage assignments, and stage weights
to enable detailed analysis of which layers/stages contribute most
to each saliency heatmap.
"""

import csv
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class ImageWeightRecord:
    """Weight record for a single image."""
    image_path: str
    predicted_label: int
    layer_names: List[str]
    raw_scores: List[float]         # Cosine similarities before softmax
    softmax_weights: List[float]    # After temperature-scaled softmax
    layer_shapes: List[Tuple[int, int]]  # Original spatial dims per layer
    stage_ids: List[int]            # Which stage each layer belongs to
    stage_scores: List[float]       # Average score per stage
    stage_weights: List[float]      # Softmax weight per stage
    top_k_selected: List[bool] = field(default_factory=list)  # Which layers survived Top-K


class WeightLogger:
    """
    Logs per-image layer and stage weights for PF-CAM diagnostics.

    Usage:
        logger = WeightLogger(output_dir="results/")
        # ... during extraction ...
        logger.log(record)
        # ... after all images ...
        logger.save()
    """

    def __init__(self, output_dir: str = ".", prefix: str = "pf_cam"):
        self.output_dir = output_dir
        self.prefix = prefix
        self.records: List[ImageWeightRecord] = []
        os.makedirs(output_dir, exist_ok=True)

    def log(self, record: ImageWeightRecord):
        """Add a weight record for one image."""
        self.records.append(record)

    def save_csv(self, filename: Optional[str] = None):
        """
        Save weight log as CSV (one row per layer per image).

        Columns: image_idx, image_path, label, layer_name, layer_shape,
                 stage_id, raw_score, softmax_weight, top_k_selected,
                 stage_score, stage_weight
        """
        if not self.records:
            return

        filepath = os.path.join(
            self.output_dir,
            filename or f"{self.prefix}_weight_log.csv"
        )

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_idx", "image_path", "label",
                "layer_name", "layer_shape_h", "layer_shape_w",
                "stage_id", "raw_score", "softmax_weight",
                "top_k_selected", "stage_score", "stage_weight"
            ])

            for img_idx, rec in enumerate(self.records):
                for i, name in enumerate(rec.layer_names):
                    stage_id = rec.stage_ids[i] if i < len(rec.stage_ids) else -1
                    stage_score = rec.stage_scores[stage_id] if stage_id < len(rec.stage_scores) else 0.0
                    stage_weight = rec.stage_weights[stage_id] if stage_id < len(rec.stage_weights) else 0.0
                    top_k = rec.top_k_selected[i] if i < len(rec.top_k_selected) else True

                    writer.writerow([
                        img_idx, rec.image_path, rec.predicted_label,
                        name,
                        rec.layer_shapes[i][0], rec.layer_shapes[i][1],
                        stage_id, f"{rec.raw_scores[i]:.6f}",
                        f"{rec.softmax_weights[i]:.6f}",
                        top_k, f"{stage_score:.6f}", f"{stage_weight:.6f}"
                    ])

        print(f"Weight log saved to {filepath} ({len(self.records)} images, "
              f"{sum(len(r.layer_names) for r in self.records)} layer records)")
        return filepath

    def save_json(self, filename: Optional[str] = None):
        """Save weight log as JSON (structured, one entry per image)."""
        if not self.records:
            return

        filepath = os.path.join(
            self.output_dir,
            filename or f"{self.prefix}_weight_log.json"
        )

        data = []
        for rec in self.records:
            entry = {
                "image_path": rec.image_path,
                "predicted_label": rec.predicted_label,
                "layers": [],
                "stages": []
            }
            # Per-layer data
            for i, name in enumerate(rec.layer_names):
                entry["layers"].append({
                    "name": name,
                    "shape": list(rec.layer_shapes[i]),
                    "stage_id": rec.stage_ids[i] if i < len(rec.stage_ids) else -1,
                    "raw_score": float(rec.raw_scores[i]),
                    "softmax_weight": float(rec.softmax_weights[i]),
                    "top_k_selected": rec.top_k_selected[i] if i < len(rec.top_k_selected) else True,
                })
            # Per-stage data
            for s_idx in range(len(rec.stage_scores)):
                entry["stages"].append({
                    "stage_id": s_idx,
                    "score": float(rec.stage_scores[s_idx]),
                    "weight": float(rec.stage_weights[s_idx]),
                })
            data.append(entry)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Weight log (JSON) saved to {filepath}")
        return filepath

    def save(self):
        """Save both CSV and JSON formats."""
        self.save_csv()
        self.save_json()

    def get_summary(self) -> Dict:
        """
        Get aggregate statistics across all logged images.

        Returns dict with:
          - mean_score_per_layer: average cosine similarity per layer position
          - mean_weight_per_layer: average softmax weight per layer position
          - mean_stage_weight: average weight per stage
        """
        if not self.records:
            return {}

        # All records should have the same number of layers
        n_layers = len(self.records[0].layer_names)
        n_stages = len(self.records[0].stage_scores)

        all_scores = np.array([r.raw_scores for r in self.records])      # (N_images, N_layers)
        all_weights = np.array([r.softmax_weights for r in self.records])
        all_stage_weights = np.array([r.stage_weights for r in self.records])

        return {
            "num_images": len(self.records),
            "num_layers": n_layers,
            "num_stages": n_stages,
            "layer_names": self.records[0].layer_names,
            "mean_score_per_layer": all_scores.mean(axis=0).tolist(),
            "std_score_per_layer": all_scores.std(axis=0).tolist(),
            "mean_weight_per_layer": all_weights.mean(axis=0).tolist(),
            "mean_stage_weight": all_stage_weights.mean(axis=0).tolist(),
        }
