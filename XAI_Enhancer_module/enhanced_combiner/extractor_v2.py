
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
from XAI_Enhancer_module.enhanced_combiner.aggregator import EnhancedCAMAggregator
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

class EnhancedExtractorV2(OptimizedCamExtractor):
    """
    Subclass of OptimizedCamExtractor that uses the EnhancedCAMAggregator.
    Supports Batch Processing.
    """
    
    def __init__(self, model, model_name: str, conv_layers: List[nn.Module], 
                 cam_method: str = "GradCAMEnhanced", device_preference: str = "auto",
                 layer_batch_size: int = 32,
                 aggregation_config: Dict = None):
        """
        Args:
            aggregation_config: Dict defining the aggregation method.
                e.g., {"type": "stagewise"} or {"type": "temp", "temp": 0.5}
        """
        super().__init__(model, model_name, conv_layers, cam_method, device_preference, layer_batch_size)
        self.aggregation_config = aggregation_config or {"type": "standard"}
        # Last extraction's per-layer similarity scores [B, L] (for per-image logs / Fig. 3)
        self.last_layer_scores: Optional[np.ndarray] = None

    def extract_saliency_map(self, 
                           input_data: Union[str, torch.Tensor],
                           predicted_label: Union[int, List[int]],
                           use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overridden to use EnhancedCAMAggregator with Batch Support.
        """
        # 1. Handle Input
        if isinstance(input_data, str):
            # Single Image Path
            image = plt.imread(input_data)
            # Handle grayscale/alpha
            if image.ndim == 2: image = np.stack((image,)*3, axis=-1)
            elif image.ndim == 3 and image.shape[2] == 4: image = image[:,:,:3]
                
            input_tensor = self.preprocess_image(image) # [1, C, H, W]
            batch_size = 1
            predicted_labels = [predicted_label]
            cache_keys = [self._get_image_key(input_data)] if use_cache else None
        elif isinstance(input_data, torch.Tensor):
            # Batch Tensor
            input_tensor = input_data
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            batch_size = input_tensor.shape[0]
            
            if isinstance(predicted_label, int):
                predicted_labels = [predicted_label] * batch_size
            else:
                predicted_labels = predicted_label
            cache_keys = None 
        else:
            raise ValueError("input_data must be image path (str) or Tensor")

        # 2. Get Actual Outputs
        actual_outputs = self.get_actual_output_batch(input_tensor, cache_keys) # [B, Classes]
        
        # 3. Generate CAM and Modified Activations
        targets = [ClassifierOutputTarget(lbl) for lbl in predicted_labels]
        # cam_per_layer: List of np.ndarray [B, H, W]
        # modified_activations_per_layer: List of tensors
        cam_per_layer, modified_activations_per_layer = self.cam_method(
            input_tensor.to(self.device), 
            targets
        )
        
        # 4. Compute Modified Outputs
        modified_outputs = self.compute_modified_outputs_batch(
            input_tensor, 
            modified_activations_per_layer
        )
        
        # 5. Compute Similarities
        cosine_similarities = self.compute_cosine_similarities(actual_outputs, modified_outputs) # [Num_Layers, B]
        
        # 6. Hybrid Aggregation
        # We need to pass [Num_Layers, B] scores and cams to Aggregator.
        # But Aggregator typically expects single sample inputs or handles batching?
        # EnhancedCAMAggregator currently seems designed for List[Tensor(H,W)] and Scores(N).
        # We need to vectorize it or loop.
        
        # Let's inspect shapes:
        # cosine_similarities: [Num_Layers, B]
        # cam_per_layer: List[ [B, 1, H, W] ] or [B, H, W]
        
        # Infer Layer Shapes
        layer_shapes = []
        for mod_act in modified_activations_per_layer:
             # mod_act shape is (B, C, H, W) or (B, C, D, H, W)
             if isinstance(mod_act, torch.Tensor):
                 shape = mod_act.shape[-2:] # H, W
             else:
                 shape = mod_act.shape[-2:]
             layer_shapes.append(tuple(shape))
        
        # Prepare CAM tensor list
        # List of [B, H, W]
        cams_tensor_list = []
        for c in cam_per_layer:
            t = torch.from_numpy(c).to(self.device)
            if t.dim() == 4: t = t.squeeze(1)
            cams_tensor_list.append(t)
            
        final_cam_batch = torch.zeros(batch_size, input_tensor.shape[2], input_tensor.shape[3], device=self.device)
        
        # Iterate over batch to apply aggregation (safest for complex aggregators like Pyramid)
        # TODO: Fully vectorize Aggregator later.
        scores_t = torch.from_numpy(cosine_similarities).to(self.device) # [Num_Layers, B]
        # Persist [B, L] for per-image logging (Tier 1.4)
        self.last_layer_scores = cosine_similarities.T.copy()

        for b in range(batch_size):
            # Extract single sample data
            sample_cams = [c[b] for c in cams_tensor_list] # List of [H, W]
            sample_scores = scores_t[:, b].cpu().numpy() # [Num_Layers]
            
            weighted_cam = EnhancedCAMAggregator.aggregate_hybrid(
                cams=sample_cams,
                scores=sample_scores,
                layer_shapes=layer_shapes,
                method_config=self.aggregation_config
            )
            
            # Normalize
            weighted_cam = weighted_cam - weighted_cam.min()
            weighted_cam = weighted_cam / (1e-7 + weighted_cam.max())
            
            final_cam_batch[b] = weighted_cam
            
        return input_tensor, final_cam_batch
