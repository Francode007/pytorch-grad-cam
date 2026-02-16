"""
Optimized CAM Extractor for enhanced explainability evaluation.
This module provides efficient extraction of saliency maps using the novel XAI method.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from XAI_Enhancer_module.utils.model_utils import get_device, transformations, CLASS_TO_IDX, IDX_TO_CLASS
from pytorch_grad_cam import (
    GradCAM, 
    GradCAMPlusPlus, 
    HiResCAM, 
    ScoreCAM, 
    AblationCAM
)


class OptimizedCamExtractor:
    """
    Optimized extractor for saliency maps using enhanced CAM methods.
    Reduces redundant computations and improves efficiency.
    """
    
    def __init__(self, model, model_name: str, conv_layers: List[nn.Module], 
                 cam_method: str = "GradCAMEnhanced", device_preference: str = "auto",
                 layer_batch_size: int = 32):
        """
        Initialize the CAM extractor.
        
        Args:
            model: The trained neural network model
            model_name: Name of the model (affects image size)
            conv_layers: List of convolutional layers for CAM computation
            cam_method: Enhanced CAM method to use
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
            layer_batch_size: Batch size for simultaneous layer processing (Smart Batching)
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
            'GradCAMEnhanced': GradCAM,
            'GradCAMPlusPlusEnhanced': GradCAMPlusPlus,
            'HiResCAMEnhanced': HiResCAM,
            'ScoreCAMEnhanced': ScoreCAM,
            'AblationCAMEnhanced': AblationCAM,
             # Also allow standard names
            'GradCAM': GradCAM,
            'GradCAMPlusPlus': GradCAMPlusPlus,
            'HiResCAM': HiResCAM,
            'ScoreCAM': ScoreCAM,
            'AblationCAM': AblationCAM
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
            Preprocessed tensor ready for model input
        """
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        
        # Apply transformations
        image_tensor = self.transforms(image).float()
        return torch.unsqueeze(image_tensor, dim=0)
    
    def get_actual_output(self, input_tensor: torch.Tensor, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Get actual model output with optional caching.
        
        Args:
            input_tensor: Preprocessed input tensor
            cache_key: Optional key for caching results
            
        Returns:
            Model output as numpy array
        """
        if cache_key and cache_key in self._actual_output_cache:
            return self._actual_output_cache[cache_key]
            
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            output_np = output[0].cpu().numpy()
            
        if cache_key:
            self._actual_output_cache[cache_key] = output_np
            
        return output_np
    
    def compute_modified_outputs_batch(self, 
                                     input_tensor: torch.Tensor,
                                     modified_activations_per_layer: List[np.ndarray]) -> List[np.ndarray]:
        """
        Efficiently compute modified outputs for all layers using Smart Batching.
        Instead of N separate passes, we run ceil(N/layer_batch_size) passes.
        """
        modified_outputs = [None] * len(self.conv_layers)
        num_layers = len(self.conv_layers)
        
        # Ensure input is on device
        input_tensor = input_tensor.to(self.device)
        
        # Process layers in batches
        for i in range(0, num_layers, self.layer_batch_size):
            batch_slice = slice(i, min(i + self.layer_batch_size, num_layers))
            current_layers = self.conv_layers[batch_slice]
            current_mod_activations = modified_activations_per_layer[batch_slice]
            actual_batch_size = len(current_layers)
            
            # Prepare batch input: duplicate the single image N times
            batch_input = input_tensor.repeat(actual_batch_size, 1, 1, 1)
            
            hooks = []
            
            # Register hooks for this batch
            for local_idx, (layer, mod_act) in enumerate(zip(current_layers, current_mod_activations)):
                
                # Prepare replacement tensor
                if isinstance(mod_act, np.ndarray):
                    mod_act_tensor = torch.from_numpy(mod_act).to(self.device)
                else:
                    mod_act_tensor = mod_act.to(self.device)
                
                if mod_act_tensor.dim() == 5:
                    mod_act_tensor = mod_act_tensor.squeeze(1)
                elif mod_act_tensor.dim() == 6:
                    mod_act_tensor = mod_act_tensor.squeeze(1)

                # Define Smart Hook
                def create_hook(idx_in_batch, replacement):
                    def hook_fn(module, input, output):
                        # ONLY modify the batch element corresponding to this layer
                        reshaped_replacement = replacement
                        if replacement.shape != output[idx_in_batch].shape:
                             try:
                                 reshaped_replacement = replacement.view(output[idx_in_batch].shape)
                             except:
                                 # Fallback (should not happen if shapes match logic)
                                 return output
                        
                        output[idx_in_batch] = reshaped_replacement
                        return output
                    return hook_fn

                hooks.append(layer.register_forward_hook(create_hook(local_idx, mod_act_tensor)))
            
            # Run batched inference
            try:
                with torch.no_grad():
                   batch_output = self.model(batch_input)
            finally:
                for h in hooks:
                    h.remove()
            
            # Collect outputs
            # batch_output is [N, num_classes]
            for local_idx in range(actual_batch_size):
                global_idx = i + local_idx
                modified_outputs[global_idx] = batch_output[local_idx].cpu().numpy()
                
        return modified_outputs
    
    def compute_cosine_similarities(self, 
                                  actual_output: np.ndarray,
                                  modified_outputs: List[np.ndarray]) -> np.ndarray:
        """
        Compute cosine similarities between actual and modified outputs.
        
        Args:
            actual_output: Original model output
            modified_outputs: List of modified outputs
            
        Returns:
            Array of cosine similarities
        """
        actual_tensor = torch.from_numpy(actual_output).unsqueeze(0)
        similarities = []
        
        for modified_output in modified_outputs:
            modified_tensor = torch.from_numpy(modified_output).unsqueeze(0)
            similarity = nn.functional.cosine_similarity(actual_tensor, modified_tensor, dim=1)
            similarities.append(similarity.item())
            
        return np.array(similarities)
    
    def extract_saliency_map(self, 
                           image_path: str,
                           predicted_label: int,
                           use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract optimized saliency map for a single image.
        
        Args:
            image_path: Path to the input image
            predicted_label: Predicted class label
            use_cache: Whether to use caching for actual outputs
            
        Returns:
            Tuple of (processed_image, weighted_saliency_map)
        """
        # Load and preprocess image
        image = plt.imread(image_path)
        input_tensor = self.preprocess_image(image)
        
        # Get actual output (with caching if enabled)
        cache_key = self._get_image_key(image_path) if use_cache else None
        actual_output = self.get_actual_output(input_tensor, cache_key)
        
        # Generate CAM and modified activations
        targets = [ClassifierOutputTarget(predicted_label)]
        cam_per_layer, modified_activations_per_layer = self.cam_method(
            input_tensor.to(self.device), 
            targets
        )
        
        # Compute modified outputs efficiently
        modified_outputs = self.compute_modified_outputs_batch(
            input_tensor, 
            modified_activations_per_layer
        )
        
        # Compute cosine similarities
        cosine_similarities = self.compute_cosine_similarities(actual_output, modified_outputs)
        
        # Apply softmax weighting
        softmax_weights = torch.softmax(torch.from_numpy(cosine_similarities), dim=0)
        
        # Compute weighted CAM
        weighted_cam = torch.zeros_like(torch.from_numpy(cam_per_layer[0][0, :]))
        for i, weight in enumerate(softmax_weights):
            cam_tensor = torch.from_numpy(cam_per_layer[i][0, :])
            weighted_cam += weight * cam_tensor
        
        # Normalize the weighted CAM
        weighted_cam = weighted_cam - weighted_cam.min()
        weighted_cam = weighted_cam / (1e-7 + weighted_cam.max())
        
        return input_tensor.squeeze(0), weighted_cam
    
    def clear_cache(self):
        """Clear the actual output cache to free memory."""
        self._actual_output_cache.clear()


class OptimizedCamDataset(Dataset):
    """
    Optimized dataset for batch processing of CAM extraction.
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 predicted_labels: List[int],
                 cam_extractor: OptimizedCamExtractor):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image file paths
            predicted_labels: List of predicted labels for each image
            cam_extractor: Optimized CAM extractor instance
        """
        self.image_paths = image_paths
        self.predicted_labels = predicted_labels
        self.cam_extractor = cam_extractor
        
        assert len(image_paths) == len(predicted_labels), \
            "Number of images and labels must match"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image_tensor, saliency_map, image_path)
        """
        image_path = self.image_paths[idx]
        predicted_label = self.predicted_labels[idx]
        
        image_tensor, saliency_map = self.cam_extractor.extract_saliency_map(
            image_path, predicted_label
        )
        
        return image_tensor, saliency_map, image_path


def create_optimized_dataloader(image_paths: List[str],
                              predicted_labels: List[int],
                              model,
                              model_name: str,
                              conv_layers: List[nn.Module],
                              batch_size: int = 8,
                              num_workers: int = 0,
                              device_preference: str = "auto") -> DataLoader:
    """
    Create an optimized DataLoader for batch CAM extraction.
    
    Args:
        image_paths: List of image file paths
        predicted_labels: List of predicted labels
        model: The trained model
        model_name: Name of the model
        conv_layers: List of convolutional layers
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        Optimized DataLoader
    """
    cam_extractor = OptimizedCamExtractor(model, model_name, conv_layers, device_preference=device_preference)
    dataset = OptimizedCamDataset(image_paths, predicted_labels, cam_extractor)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
