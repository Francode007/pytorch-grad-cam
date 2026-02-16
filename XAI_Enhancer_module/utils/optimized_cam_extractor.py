"""
Optimized CAM Extractor for enhanced explainability evaluation.
This module provides efficient extraction of saliency maps using the novel XAI method.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from XAI_Enhancer_module.utils.model_utils import get_device, transformations, CLASS_TO_IDX, IDX_TO_CLASS
from XAI_Enhancer_module.enhanced_cams import (
    GradCAMEnhanced, 
    GradCAMPlusPlusEnhanced, 
    HiResCAMEnhanced, 
    ScoreCAMEnhanced, 
    AblationCAMEnhanced
)


class OptimizedCamExtractor:
    """
    Optimized extractor for saliency maps using enhanced CAM methods.
    Reduces redundant computations and improves efficiency.
    Supports Batch Processing.
    """
    
    def __init__(self, model, model_name: str, conv_layers: List[nn.Module], 
                 cam_method: str = "GradCAMEnhanced", device_preference: str = "auto",
                 layer_batch_size: int = 16): # Optimized for A100 40GB; mega-batch = 16 × DataLoader_batch
        """
        Initialize the CAM extractor.
        
        Args:
            model: The trained neural network model
            model_name: Name of the model (affects image size)
            conv_layers: List of convolutional layers for CAM computation
            cam_method: Enhanced CAM method to use
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
            layer_batch_size: Batch size for simultaneous layer processing (number of layers processed in parallel per image)
        """
        self.model = model
        self.model_name = model_name
        self.conv_layers = conv_layers
        self.device = get_device(device_preference)
        self.img_size = 384 if model_name.startswith("b4") else 224
        self.cam_method_name = cam_method
        self.layer_batch_size = layer_batch_size
        
        # Initialize the specified enhanced CAM method
        enhanced_cam_methods = {
            'GradCAMEnhanced': GradCAMEnhanced,
            'GradCAMPlusPlusEnhanced': GradCAMPlusPlusEnhanced,
            'HiResCAMEnhanced': HiResCAMEnhanced,
            'ScoreCAMEnhanced': ScoreCAMEnhanced,
            'AblationCAMEnhanced': AblationCAMEnhanced
        }
        
        if cam_method not in enhanced_cam_methods:
            raise ValueError(f"Unknown enhanced CAM method: {cam_method}. "
                           f"Available methods: {list(enhanced_cam_methods.keys())}")
        
        self.cam_method = enhanced_cam_methods[cam_method](model, conv_layers)
        
        # Pre-compute transformations
        self.transforms = transformations
        
        # Cache for actual outputs to avoid recomputation
        self._actual_output_cache = {}
        
    def _get_image_key(self, image_path: str) -> str:
        """Generate a unique key for caching based on image path."""
        return image_path
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Efficiently preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor ready for model input [1, C, H, W]
        """
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        
        # Apply transformations
        image_tensor = self.transforms(image).float()
        return torch.unsqueeze(image_tensor, dim=0)
    
    def get_actual_output_batch(self, input_tensor: torch.Tensor, cache_keys: List[str] = None) -> np.ndarray:
        """
        Get actual model output for a batch with optional caching.
        
        Args:
            input_tensor: Preprocessed input tensor [B, C, H, W]
            cache_keys: Optional list of keys for caching results
            
        Returns:
            Model output as numpy array [B, num_classes]
        """
        batch_size = input_tensor.shape[0]
        outputs = []
        
        # Check cache if keys provided
        indices_to_compute = []
        if cache_keys:
            computed_outputs = [None] * batch_size
            for i, key in enumerate(cache_keys):
                if key in self._actual_output_cache:
                    computed_outputs[i] = self._actual_output_cache[key]
                else:
                    indices_to_compute.append(i)
        else:
            indices_to_compute = list(range(batch_size))
            computed_outputs = [None] * batch_size

        if indices_to_compute:
            # We need to compute for some or all images
            # If all need computation, just run full batch
            if len(indices_to_compute) == batch_size:
                 with torch.no_grad():
                    output_tensor = self.model(input_tensor.to(self.device))
                    batch_start_idx = 0
                    for i in range(batch_size):
                        out = output_tensor[i].cpu().numpy()
                        computed_outputs[i] = out
                        if cache_keys:
                             self._actual_output_cache[cache_keys[i]] = out
            else:
                 # Only compute needed
                 tensors_to_compute = input_tensor[indices_to_compute].to(self.device)
                 with torch.no_grad():
                    output_subset = self.model(tensors_to_compute)
                    for idx_in_subset, global_idx in enumerate(indices_to_compute):
                        out = output_subset[idx_in_subset].cpu().numpy()
                        computed_outputs[global_idx] = out
                        if cache_keys:
                            self._actual_output_cache[cache_keys[global_idx]] = out
                            
        return np.array(computed_outputs)
    
    def get_actual_output(self, input_tensor: torch.Tensor, cache_key: Optional[str] = None) -> np.ndarray:
        """Legacy single-item wrapper"""
        out = self.get_actual_output_batch(input_tensor, [cache_key] if cache_key else None)
        return out[0]
    
    def compute_modified_outputs_batch(self, 
                                     input_tensor: torch.Tensor,
                                     modified_activations_per_layer: List[Union[np.ndarray, torch.Tensor]]) -> List[np.ndarray]:
        """
        Efficiently compute modified outputs for all layers using Smart Batching.
        Handles Batch Input [B, C, H, W].
        
        Returns:
            List of arrays, where each element is [B, Num_Classes] corresponding to a layer modification.
            Length of list = Num Layers.
        """
        num_layers = len(self.conv_layers)
        batch_size = input_tensor.shape[0]
        
        # Result container: List of [B, Num_Classes]
        # We initialize with None
        modified_outputs_per_layer = [None] * num_layers
        
        # Ensure input is on device
        input_tensor = input_tensor.to(self.device) # [B, C, H, W]
        
        # Process layers in chunks (Layer Batching)
        # We process 'layer_batch_size' layers at a time.
        # Total batch size sent to GPU = batch_size * layer_batch_size
        # We need to be careful with memory.
        
        # Check max safe batch size
        # If batch_size=32 and layer_batch_size=32 -> 1024 images. Might be too big for A100 if ResNet50.
        # Let's dynamically adjust step size if needed, but for now stick to logic.
        
        for i in range(0, num_layers, self.layer_batch_size):
            batch_slice = slice(i, min(i + self.layer_batch_size, num_layers))
            current_layers = self.conv_layers[batch_slice]
            current_mod_activations = modified_activations_per_layer[batch_slice] # List of [B, C, H, W] or [1, C, H, W]
            
            num_current_layers = len(current_layers)
            
            # Prepare Mega Batch: [B * Num_Current_Layers, C, H, W]
            # We explicitly repeat the input tensor so that:
            # First B images are for Layer 1
            # Next B images are for Layer 2 
            # ...
            # reshape: (Layers, Batch, ...) -> flatten
            
            # shape: [Layers, B, C, H, W]
            mega_batch_input = input_tensor.unsqueeze(0).expand(num_current_layers, -1, -1, -1, -1)
            mega_batch_input = mega_batch_input.reshape(-1, *input_tensor.shape[1:]) # [Layers*B, C, H, W]
            
            hooks = []
            
            # Register hooks
            # Each hook needs to know which "slice" of the mega-batch it belongs to.
            # Slice k corresponds to Layer k in the current chunk.
            # Indices for Layer k are [k*B : (k+1)*B]
            
            for local_layer_idx, (layer, mod_act) in enumerate(zip(current_layers, current_mod_activations)):
                
                # Prepare replacement tensor
                # mod_act should be [B, C, H, W]. If [1, C, H, W], expand it.
                if isinstance(mod_act, np.ndarray):
                    mod_act_tensor = torch.from_numpy(mod_act).to(self.device)
                else:
                    mod_act_tensor = mod_act.to(self.device)
                
                # Handle dimensions (sometimes 5D/6D artifacts from CAM lib)
                while mod_act_tensor.dim() > 4:
                    mod_act_tensor = mod_act_tensor.squeeze(1)
                
                if mod_act_tensor.shape[0] == 1 and batch_size > 1:
                     mod_act_tensor = mod_act_tensor.expand(batch_size, -1, -1, -1)
                
                # Verify shape
                if mod_act_tensor.shape[0] != batch_size:
                    raise ValueError(f"Modified activation batch size {mod_act_tensor.shape[0]} does not match input batch size {batch_size}")

                # Define Smart Hook
                def create_hook(layer_idx_in_chunk, replacement_batch):
                    def hook_fn(module, input, output):
                        # output shape is [Layers*B, C, H, W] (for this layer's forward pass? NO)
                        # Wait, when we run `model(mega_batch_input)`, every layer receives [Layers*B, ...] input.
                        # We only want to modify the subset of the batch corresponding to THIS layer.
                        # Indices: start = layer_idx_in_chunk * batch_size
                        #          end   = (layer_idx_in_chunk + 1) * batch_size
                        
                        start = layer_idx_in_chunk * batch_size
                        end = (layer_idx_in_chunk + 1) * batch_size
                        
                        # Replace the output for the specific slice
                        # replacement_batch is [B, C, H, W]
                        # output slice is [B, C, H, W]
                        
                        # Check compatibility
                        current_slice = output[start:end]
                        if current_slice.shape != replacement_batch.shape:
                            # Resize/View fallback
                             try:
                                 reshaped = replacement_batch.view(current_slice.shape)
                                 output[start:end] = reshaped
                             except:
                                 # dimension mismatch that can't be viewed (e.g. spatial size change)
                                 # Only apply if spatial dims match? 
                                 # Usually CAM replacement matches exactly.
                                 pass
                        else:
                            output[start:end] = replacement_batch
                        
                        return output
                    return hook_fn

                hooks.append(layer.register_forward_hook(create_hook(local_layer_idx, mod_act_tensor)))
            
            # Run batched inference
            try:
                with torch.no_grad():
                   mega_batch_output = self.model(mega_batch_input)
            finally:
                for h in hooks:
                    h.remove()
            
            # Collect outputs
            # mega_batch_output is [Layers*B, Num_Classes]
            # Unwrap back to lists
            for local_layer_idx in range(num_current_layers):
                global_layer_idx = i + local_layer_idx
                start = local_layer_idx * batch_size
                end = (local_layer_idx + 1) * batch_size
                
                layer_output_batch = mega_batch_output[start:end].cpu().numpy() # [B, Num_Classes]
                modified_outputs_per_layer[global_layer_idx] = layer_output_batch
                
        return modified_outputs_per_layer
    
    def compute_cosine_similarities(self, 
                                  actual_output: np.ndarray,
                                  modified_outputs: List[np.ndarray]) -> np.ndarray:
        """
        Compute cosine similarities for batch.
        
        Args:
            actual_output: [B, Num_Classes]
            modified_outputs: List of [B, Num_Classes] (length = Num Layers)
            
        Returns:
            Array of shape [Num_Layers, B] containing cosine similarities.
        """
        actual_tensor = torch.from_numpy(actual_output) # [B, Class]
        similarities = []
        
        for modified_output in modified_outputs:
            modified_tensor = torch.from_numpy(modified_output) # [B, Class]
            # Compute cosine similarity along dim 1 (classes)
            # Result: [B]
            similarity = nn.functional.cosine_similarity(actual_tensor, modified_tensor, dim=1)
            similarities.append(similarity.cpu().numpy())
            
        return np.array(similarities) # [Num_Layers, B]
    
    def extract_saliency_map(self, 
                           input_data: Union[str, torch.Tensor],
                           predicted_label: Union[int, List[int]],
                           use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract optimized saliency map for single image OR batch.
        
        Args:
            input_data: Image path (str) OR Batch Tensor ([B, C, H, W])
            predicted_label: Predicted class label (int) or List of ints
            use_cache: Whether to use caching for actual outputs (if using path)
            
        Returns:
            Tuple of (processed_image_tensor, weighted_saliency_map)
            Dimensions: 
               Image: [B, C, H, W]
               Map:   [B, H, W]
        """
        # 1. Handle Input
        if isinstance(input_data, str):
            # Single Image Path
            image = plt.imread(input_data)
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
            
            # No caching for tensor input usually (unless caller manages keys)
            cache_keys = None 
        else:
            raise ValueError("input_data must be image path (str) or Tensor")
            
        image_size_h, image_size_w = input_tensor.shape[2], input_tensor.shape[3]

        # 2. Get Actual Outputs
        actual_outputs = self.get_actual_output_batch(input_tensor, cache_keys) # [B, Classes]
        
        # 3. Generate CAM and Modified Activations (Forward Pass 1)
        targets = [ClassifierOutputTarget(lbl) for lbl in predicted_labels]
        
        # cam_per_layer: List of np.ndarray [B, H, W] (already scaled)
        # modified_activations_per_layer: List of tensors [B, C, H, W]
        cam_per_layer, modified_activations_per_layer = self.cam_method(
            input_tensor.to(self.device), 
            targets
        )
        
        # 4. Compute Modified Outputs Efficiently (Batched Forward Passes)
        # modified_outputs: List of [B, Classes]
        modified_outputs = self.compute_modified_outputs_batch(
            input_tensor, 
            modified_activations_per_layer
        )
        
        # 5. Compute Weights
        # similarities: [Num_Layers, B]
        cosine_similarities = self.compute_cosine_similarities(actual_outputs, modified_outputs)
        
        # Softmax over layers (dim 0) -> [Num_Layers, B]
        softmax_weights = torch.softmax(torch.from_numpy(cosine_similarities), dim=0).to(self.device)
        
        # 6. Weighted Sum
        # cam_per_layer is list of [B, 1, H, W] or [B, H, W] ?
        # BaseCAM usually returns list of [Batch, 1, H, W] or [Batch, H, W] depending on version
        # Let's inspect first element
        first_cam = cam_per_layer[0] 
        if isinstance(first_cam, np.ndarray):
            first_cam = torch.from_numpy(first_cam).to(self.device)
            
        if first_cam.dim() == 4: # [B, 1, H, W]
            first_cam = first_cam.squeeze(1) # [B, H, W]
            
        final_cam = torch.zeros_like(first_cam)
        
        for i, weight_per_batch in enumerate(softmax_weights):
            # weight_per_batch: [B]
            cam = cam_per_layer[i]
            if isinstance(cam, np.ndarray):
                cam = torch.from_numpy(cam).to(self.device)
            if cam.dim() == 4:
                cam = cam.squeeze(1)
                
            # Broadcast weight: [B] -> [B, 1, 1]
            w = weight_per_batch.view(batch_size, 1, 1)
            final_cam += w * cam
        
        # 7. Normalize
        # Normalize per image in batch
        # Min across H,W -> [B, 1, 1]
        min_vals = final_cam.flatten(1).min(1)[0].view(batch_size, 1, 1)
        final_cam = final_cam - min_vals
        max_vals = final_cam.flatten(1).max(1)[0].view(batch_size, 1, 1)
        final_cam = final_cam / (max_vals + 1e-7)
        
        return input_tensor, final_cam
    
    def clear_cache(self):
        """Clear the actual output cache to free memory."""
        self._actual_output_cache.clear()


class OptimizedCamDataset(Dataset):
    """
    Optimized dataset wrapper.
    Legacy support.
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 predicted_labels: List[int],
                 cam_extractor: OptimizedCamExtractor):
        self.image_paths = image_paths
        self.predicted_labels = predicted_labels
        self.cam_extractor = cam_extractor
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        # This forces single-item extraction which defeats Batch Processing if used in DataLoader
        # We should avoid using this Dataset if we want batching.
        # But keeping for compatibility.
        image_path = self.image_paths[idx]
        predicted_label = self.predicted_labels[idx]
        
        image_tensor, saliency_map = self.cam_extractor.extract_saliency_map(
            image_path, predicted_label
        )
        
        # If batch dim exists, squeeze it
        if image_tensor.dim() == 4:
             image_tensor = image_tensor.squeeze(0)
        if saliency_map.dim() == 3:
             saliency_map = saliency_map.squeeze(0)
             
        return image_tensor, saliency_map, image_path
