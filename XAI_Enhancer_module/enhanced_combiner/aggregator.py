
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class EnhancedCAMAggregator:
    """
    Aggregator for combining CAMs from different layers using enhanced methodologies.
    Supports:
    - Standard Softmax (Baseline)
    - Temperature Scaling
    - Top-K Sparsity (Hard & Soft)
    - Stagewise Aggregation (Hierarchical)
    """

    @staticmethod
    def aggregate_standard(cams: List[torch.Tensor], scores: np.ndarray) -> torch.Tensor:
        """
        Baseline aggregation: Softmax(scores) * CAMs.
        """
        # Convert scores to tensor
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        # Apply Softmax
        weights = F.softmax(scores_tensor, dim=0)
        
        # Weighted Sum
        final_cam = torch.zeros_like(cams[0])
        for i, cam in enumerate(cams):
            final_cam += weights[i] * cam
            
        return final_cam

    @staticmethod
    def aggregate_uniform(cams: List[torch.Tensor], scores: np.ndarray = None) -> torch.Tensor:
        """
        Uniform multi-layer average (T→∞ baseline): w_l = 1/L.
        ``scores`` is ignored; kept for signature parity with other aggregators.
        """
        n = len(cams)
        if n == 0:
            raise ValueError("aggregate_uniform requires at least one CAM")
        weight = 1.0 / float(n)
        final_cam = torch.zeros_like(cams[0])
        for cam in cams:
            final_cam = final_cam + weight * cam
        return final_cam

    @staticmethod
    def aggregate_temperature(cams: List[torch.Tensor], scores: np.ndarray, temp: float = 0.1) -> torch.Tensor:
        """
        Temperature Scaling: Softmax(scores / temp) * CAMs.
        Lower temp (< 1.0) makes distribution sharper (closer to max).
        Higher temp (> 1.0) makes distribution flatter (uniform).
        """
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        # Apply Temperature
        weights = F.softmax(scores_tensor / temp, dim=0)
        
        final_cam = torch.zeros_like(cams[0])
        for i, cam in enumerate(cams):
            final_cam += weights[i] * cam
            
        return final_cam

    @staticmethod
    def aggregate_top_k(cams: List[torch.Tensor], scores: np.ndarray, k: int = 5, soft: bool = False, temp: float = 1.0) -> torch.Tensor:
        """
        Top-K Sparsity: Only keep the top K layers.
        
        Args:
            k: Number of layers to keep.
            soft: If True, apply softmax to the top K scores. 
                  If False, just normalize the raw scores (or 1s?).
                  Usually we re-normalize the scores of the chosen K.
            temp: Temperature to apply if soft=True.
        """
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # Get Top K indices
        # We want the indices of the largest scores
        topk_values, topk_indices = torch.topk(scores_tensor, k=min(k, len(scores)))
        
        # Create a zero weight vector
        full_weights = torch.zeros_like(scores_tensor)
        
        if soft:
            # Re-compute softmax over ONLY the top K values
            # This ensures they sum to 1
            topk_weights = F.softmax(topk_values / temp, dim=0)
        else:
             # "Hard" Top-K might just mean "Average the top K" or "Proportional to raw score"
             # Let's assume we want to preserve relative importance but zero out others.
             # Option A: Simple Average
             # topk_weights = torch.ones_like(topk_values) / k
             
             # Option B: Proportional to Score (Min-Max scaled or similar?)
             # A safe bet is Softmax on them with T=1, same as "soft=True, temp=1.0"
             # If "Hard" means "Winner Takes All" (Top-1), that's k=1.
             # Let's stick to Softmax re-normalization as the default behavior for "keeping top K".
             topk_weights = F.softmax(topk_values, dim=0)

        # Scatter back to full size
        full_weights[topk_indices] = topk_weights
        
        final_cam = torch.zeros_like(cams[0])
        for i, cam in enumerate(cams):
            if full_weights[i] > 0:
                final_cam += full_weights[i] * cam
                
        return final_cam

    @staticmethod
    def aggregate_stagewise(cams: List[torch.Tensor], 
                            scores: np.ndarray, 
                            layer_shapes: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Hierarchical Aggregation:
        1. Group layers by spatial resolution (Stage).
        2. Within each stage, compute a "Stage Score" (e.g., Mean of layer scores).
        3. Compute "Stage Weights" = Softmax(Stage Scores).
        4. Within each stage, compute "Layer Weights" = Softmax(Layer Scores).
        5. Final Weight for Layer L = Stage_Weight * Layer_Weight_in_Stage. (Actually this might double count).
        
        Alternative Strategy (from Plan):
        1. "Stage Score" = Mean(Scores of layers in stage).
        2. "Stage Weight" = Softmax(Stage Scores).
        3. "Stage CAM" = Weighted Sum of layers in stage (using local softmax).
        4. Final CAM = Weighted Sum of Stage CAMs (using Stage Weights).
        """
        
        # 1. Group layers by shape
        stages = {} # shape -> list of indices
        for idx, shape in enumerate(layer_shapes):
            if shape not in stages:
                stages[shape] = []
            stages[shape].append(idx)
            
        stage_keys = list(stages.keys()) # sort? typically order is preserved or irrelevant if we just iterate
        # Sort keys by resolution size (descending) or order of appearance? 
        # Order of appearance is safer for "Stage 1, Stage 2..."
        # Let's stick to the order they appear.
        sorted_stage_shapes = []
        seen = set()
        for s in layer_shapes:
            if s not in seen:
                sorted_stage_shapes.append(s)
                seen.add(s)

        # 2. Compute Stage Scores
        stage_scores = []
        for shape in sorted_stage_shapes:
            indices = stages[shape]
            # Average score of layers in this stage
            avg_score = np.mean(scores[indices]) 
            stage_scores.append(avg_score)
            
        stage_scores_tensor = torch.tensor(stage_scores, dtype=torch.float32)
        stage_weights = F.softmax(stage_scores_tensor, dim=0) # [Num_Stages]

        final_cam = torch.zeros_like(cams[0])
        
        # 3. Aggregate
        for stage_idx, shape in enumerate(sorted_stage_shapes):
            indices = stages[shape]
            stage_weight = stage_weights[stage_idx]
            
            # Aggregate WITHIN stage
            # Option: Standard Softmax of layers in this stage
            local_scores = torch.tensor(scores[indices], dtype=torch.float32)
            local_weights = F.softmax(local_scores, dim=0)
            
            stage_cam = torch.zeros_like(cams[0])
            for i, local_weight in zip(indices, local_weights):
                stage_cam += local_weight * cams[i]
                
            # Add to final
            final_cam += stage_weight * stage_cam
            
        return final_cam

    @staticmethod
    def aggregate_hybrid(cams: List[torch.Tensor], 
                         scores: np.ndarray, 
                         layer_shapes: List[Tuple[int, int]],
                         method_config: Dict) -> torch.Tensor:
        """
        Flexible aggregator.
        method_config = {
          "type": "stagewise" | "topk" | "temp" | "standard" | "uniform" | "pyramid",
          "k": 5,
          "temp": 0.1,
          "soft": True
        }
        """
        m_type = method_config.get("type", "standard")
        
        if m_type == "stagewise":
            return EnhancedCAMAggregator.aggregate_stagewise(cams, scores, layer_shapes)
        elif m_type == "topk":
            return EnhancedCAMAggregator.aggregate_top_k(cams, scores, 
                                                         k=method_config.get("k", 5),
                                                         soft=method_config.get("soft", True),
                                                         temp=method_config.get("temp", 1.0))
        elif m_type == "temp":
            return EnhancedCAMAggregator.aggregate_temperature(cams, scores, 
                                                               temp=method_config.get("temp", 1.0))
        elif m_type == "pyramid":
             return EnhancedCAMAggregator.aggregate_pyramid_fusion(cams, scores, layer_shapes, method_config)
        elif m_type == "uniform":
            return EnhancedCAMAggregator.aggregate_uniform(cams, scores)
        else:
            return EnhancedCAMAggregator.aggregate_standard(cams, scores)

    @staticmethod
    def aggregate_pyramid_fusion(cams: List[torch.Tensor], 
                                 scores: np.ndarray, 
                                 layer_shapes: List[Tuple[int, int]],
                                 config: Dict) -> torch.Tensor:
        """
        Pyramid Fusion CAM (PF-CAM) Implementation.
        
        Logic:
        1. Group layers by Stage (Resolution).
        2. Within each stage, select Top-K layers (based on config).
        3. Compute normalized "Stage CAM" for each stage.
        4. Fuse Stage CAMs Top-Down (Deep -> Shallow) using Gating.
        """
        
        # 1. Group layers by shape (Stage Detection)
        # We assume smaller shape = Deeper stage.
        # stages: dict { shape: [indices] }
        stages = {}
        for idx, shape in enumerate(layer_shapes):
            if shape not in stages:
                stages[shape] = []
            stages[shape].append(idx)
            
        # Sort stages by resolution (Smallest/Deepest first)
        # shape is (H, W). We use H*W as proxy for resolution.
        sorted_shapes = sorted(stages.keys(), key=lambda s: s[0]*s[1])
        
        stage_cams = []
        
        # 2. Process each stage
        for shape in sorted_shapes:
            indices = stages[shape]
            stage_layer_scores = scores[indices]
            stage_layer_cams = [cams[i] for i in indices]
            
            # --- Top-K Selection per Stage ---
            k_percent = config.get("k_percent", 0.15) # Default Top 15%
            k_min = config.get("k_min", 2)
            
            num_layers = len(indices)
            k = max(k_min, int(num_layers * k_percent))
            k = min(k, num_layers) # Safety
            
            # Use aggregate_top_k helper for intra-stage aggregation
            # We treat the stage as a mini-aggregation problem
            # Normalized within stage
            stage_cam = EnhancedCAMAggregator.aggregate_top_k(
                stage_layer_cams, 
                stage_layer_scores, 
                k=k, 
                soft=True, 
                temp=config.get("temp", 0.1) # Sharp softmax for selection
            )
            
            # --- Energy Normalization ---
            # Max-Normalize first to get to 0-1 range
            if stage_cam.max() > 1e-7:
                stage_cam = stage_cam / stage_cam.max()
            
            # Scale by Energy Factor? 
            # If we want to penalize high-res, we could divide by resolution.
            # But "masking" in Top-Down fusion handles the noise.
            # So simple 0-1 normalization per stage is good enough for now.
            
            stage_cams.append(stage_cam)
            
        # 3. Top-Down Fusion (Soft Gating)
        # stage_cams[0] is Deepest (C4), stage_cams[-1] is Shallowest (C1)
        
        fused_cam = stage_cams[0]
        
        # Soft Gating Parameter (Beta)
        # beta = 1.0 -> No Gating (Sum)
        # beta = 0.0 -> Hard Gating (Product)
        # Recommended: 0.3 - 0.5 (Residual-like connection)
        beta = config.get("beta", 0.4) 
        
        for i in range(1, len(stage_cams)):
            next_stage_cam = stage_cams[i] # Shallower
            
            # Create Gate from current fused (Deep)
            # Sigmoid-like gate to determine "Objectness"
            gate = fused_cam.clone()
            if gate.max() > 1e-7:
               gate = gate / gate.max()
            
            # Apply Soft Gating
            # "Allow `beta` of the signal effectively, and then gate the rest"
            # Formula: Signal * (beta + (1-beta)*Gate)
            # When Gate=1 (Object), Mask = 1.0.
            # When Gate=0 (Background), Mask = beta.
            
            soft_gate = beta + (1.0 - beta) * gate
            masked_details = next_stage_cam * soft_gate
            
            # Add to formulation
            fused_cam = fused_cam + masked_details
            
            # Renormalize after addition to keep valid range for next gate
            if fused_cam.max() > 1e-7:
                fused_cam = fused_cam / fused_cam.max()
                
        return fused_cam
