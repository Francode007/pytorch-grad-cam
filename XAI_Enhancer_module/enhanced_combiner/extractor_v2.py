
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
from XAI_Enhancer_module.enhanced_combiner.aggregator import EnhancedCAMAggregator
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class EnhancedExtractorV2(OptimizedCamExtractor):
    """
    Subclass of OptimizedCamExtractor that uses the EnhancedCAMAggregator.
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
        
    def extract_saliency_map(self, 
                           image_path: str,
                           predicted_label: int,
                           use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overridden to use EnhancedCAMAggregator.
        """
        import matplotlib.pyplot as plt # lazy import
        
        # Load and preprocess image
        try:
            image = plt.imread(image_path)
            if image.ndim == 2: # Grayscale
                 image = np.stack((image,)*3, axis=-1)
            elif image.shape[2] == 4: # RGBA
                 image = image[:,:,:3]
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            return None, None

        input_tensor = self.preprocess_image(image)
        
        # Get actual output (with caching if enabled)
        cache_key = self._get_image_key(image_path) if use_cache else None
        actual_output = self.get_actual_output(input_tensor, cache_key)
        
        # Generate CAM and modified activations
        targets = [ClassifierOutputTarget(predicted_label)]
        # cam_per_layer: List of np.ndarray [1, H, W] (already scaled)
        # modified_activations_per_layer: List of tensors/arrays
        cam_per_layer, modified_activations_per_layer = self.cam_method(
            input_tensor.to(self.device), 
            targets
        )
        
        # Compute modified outputs efficiently
        # modified_outputs: List of np.ndarray [num_classes]
        modified_outputs = self.compute_modified_outputs_batch(
            input_tensor, 
            modified_activations_per_layer
        )
        
        # Compute cosine similarities
        cosine_similarities = self.compute_cosine_similarities(actual_output, modified_outputs)
        
        # --- NEW AGGREGATION LOGIC ---
        
        # Prepare CAMs as tensors
        # cam_per_layer is a list of arrays of shape (1, H, W)
        # We need individual tensors of shape (H, W) or (1, H, W)
        cams_tensor_list = [torch.from_numpy(c[0, :]).float() for c in cam_per_layer]

        # Prepare Layer Shapes for Stagewise Aggregation
        # We can infer the shape from the raw CAMs BEFORE scaling, but here we only have scaled CAMs.
        # Wait, BaseCAM.compute_cam_per_layer returns *scaled* cams.
        # However, for stagewise aggregation we need to know which "stage" a layer belongs to.
        # Usually stages are defined by resolution.
        # BUT, `cam_per_layer` in `OptimizedCamExtractor` calls `scale_cam_image`.
        # So all cams are already 224x224 (or target size).
        # We need the ORIGINAL resolution to group by stage.
        
        # Workaround: Inspect the `modified_activations_per_layer`. 
        # They should preserve the spatial dimensions of that layer.
        layer_shapes = []
        for mod_act in modified_activations_per_layer:
             # mod_act shape is (1, C, H, W) or (1, C, D, H, W)
             if isinstance(mod_act, torch.Tensor):
                 shape = mod_act.shape[-2:] # H, W
             else:
                 shape = mod_act.shape[-2:]
             layer_shapes.append(tuple(shape))
        
        # Call Aggregator
        weighted_cam = EnhancedCAMAggregator.aggregate_hybrid(
            cams=cams_tensor_list,
            scores=cosine_similarities,
            layer_shapes=layer_shapes,
            method_config=self.aggregation_config
        )
        
        # Normalize final CAM to [0, 1]
        weighted_cam = weighted_cam - weighted_cam.min()
        weighted_cam = weighted_cam / (1e-7 + weighted_cam.max())
        
        return input_tensor.squeeze(0), weighted_cam
