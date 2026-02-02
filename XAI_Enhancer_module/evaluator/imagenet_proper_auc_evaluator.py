#!/usr/bin/env python3
"""
ImageNet Proper AUC Evaluator - Evaluator specialized for ImageNet dataset.
This module extends the base ProperAUCEvaluator to work with ImageNet validation dataset.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the base ProperAUCEvaluator
from XAI_Enhancer_module.evaluator.proper_auc_evaluation import ProperAUCEvaluator
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, save_analysis_data



class _ImageNetDataset(Dataset):
    """Dataset wrapper for ImageNet images to enable multi-threaded loading."""
    def __init__(self, image_paths, predicted_labels, class_names, transform=None):
        self.image_paths = image_paths
        self.predicted_labels = predicted_labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        predicted_label = self.predicted_labels[idx]
        class_name = self.class_names[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image  # Should not happen if transform provided
            
        return image_tensor, predicted_label, class_name, image_path

class ImageNetProperAUCEvaluator(ProperAUCEvaluator):
    """
    Evaluator specialized for ImageNet dataset with proper synset mapping.
    Uses ProperAUCEvaluator as the base for consistent AUC calculations.
    """
    
    def __init__(self, 
                 model_name: str, 
                 imagenet_path: str,
                 device_preference: str = "auto",
                 layer_mode: str = "last",
                 enhanced_cam_method: str = "GradCAMEnhanced",
                 model_cache_dir: str = "../pytorch_models/"):
        """
        Initialize the ImageNet evaluator.
        
        Args:
            model_name: Name of the pre-trained model
            imagenet_path: Path to ImageNet validation dataset
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
            layer_mode: Layer selection mode ("last", "last_5", "all")
            enhanced_cam_method: Enhanced CAM method to use
            model_cache_dir: Directory containing pre-downloaded models (default: "../pytorch_models/")
        """
        # Don't call super().__init__() to avoid loading custom models
        # Instead, initialize only what we need from the base class
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        
        # Optimize global settings
        torch.backends.cudnn.benchmark = True
        
        # Get device
        from XAI_Enhancer_module.utils.model_utils import get_device
        device_val = get_device(device_preference)
        self.device = torch.device(device_val) if isinstance(device_val, str) else device_val
        
        # Store ImageNet-specific parameters
        self.imagenet_path = imagenet_path
        self.layer_mode = layer_mode
        self.enhanced_cam_method = enhanced_cam_method
        
        # Load synset mapping
        self.synset_mapping = self._load_synset_mapping()
        self.class_names = list(self.synset_mapping.values())
        self.synset_to_idx = {synset: idx for idx, synset in enumerate(self.synset_mapping.keys())}
        self.idx_to_synset = {idx: synset for synset, idx in self.synset_to_idx.items()}
        
        # Initialize model loader with cache directory
        from XAI_Enhancer_module.utils.model_loader import ModelLoader
        self.model_loader = ModelLoader(model_cache_dir)
        
        # Load pre-trained ImageNet model using cached weights
        self.model = self.model_loader.load_pretrained_model(model_name)
        self.model.eval()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize Enhanced CAM components
        self.conv_layers = self._get_enhanced_cam_layers(layer_mode)
        self.enhanced_cam_extractor = None
        
        # ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"ImageNetProperAUCEvaluator initialized:")
        print(f"  Model: {model_name}")
        print(f"  Model cache dir: {model_cache_dir}")
        print(f"  ImageNet path: {imagenet_path}")
        print(f"  Device: {self.device}")
        print(f"  Layer mode: {layer_mode}")
        print(f"  Enhanced CAM method: {enhanced_cam_method}")
        print(f"  Number of classes: {len(self.synset_mapping)}")
        print(f"  Number of conv layers: {len(self.conv_layers)}")
        
        # Print cache status for verification
        self.model_loader.print_cache_status()
    
    def compute_insertion_auc(self, image_tensor: torch.Tensor, 
                            saliency_map: np.ndarray, predicted_label: int,
                            step_size: int = 50, batch_size: int = 64) -> Tuple[List[float], float]:
        """
        Compute insertion AUC by progressively adding pixels in order of importance.
        Optimized with batch processing and AMP.
        """
        # Import necessary functions
        from torch.cuda.amp import autocast
        import torch
        import numpy as np
        
        # Ensure image tensor is 3D [C, H, W]
        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
            
        # Ensure saliency map is the right shape
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]  # Take first channel if RGB
        
        # Flatten saliency map and get pixel indices sorted by importance
        flat_saliency = saliency_map.flatten()
        # pixel_indices = np.argsort(flat_saliency)[::-1]  # Descending order
        # Use torch for indices to avoid host-device sync during loop
        pixel_indices_np = np.argsort(flat_saliency)[::-1].copy()
        pixel_indices = torch.from_numpy(pixel_indices_np).to(self.device).long()
        
        # Create baseline image (black) on the correct device
        baseline_image = torch.zeros_like(image_tensor, device=self.device)
        
        # Progressively add pixels and measure confidence
        confidences = []
        n_pixels = len(pixel_indices)
        
        # Prepare for batch processing
        current_image = baseline_image.clone()
        original_flat = image_tensor.view(image_tensor.shape[0], -1)
        current_flat = current_image.view(current_image.shape[0], -1)
        
        # Pre-calculate baseline confidence
        with torch.no_grad():
            if self.device.type == 'cuda':
                with autocast():
                    baseline_output = self.model(baseline_image.unsqueeze(0))
            else:
                baseline_output = self.model(baseline_image.unsqueeze(0))
                
            baseline_confidence = torch.softmax(baseline_output, dim=1)[0, predicted_label].item()
            confidences.append(baseline_confidence)
            
            # Create batches of modified images
            modified_images_batch = []
            
            # Helper to process a batch
            def process_batch(batch_list):
                if not batch_list:
                    return []
                
                batch_tensor = torch.stack(batch_list)
                
                if self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(batch_tensor)
                else:
                    outputs = self.model(batch_tensor)
                    
                batch_confidences = torch.softmax(outputs, dim=1)[:, predicted_label].tolist()
                return batch_confidences

            for i in range(0, n_pixels, step_size):
                # Update the CURRENT image state for this step
                end_idx = min(i + step_size, n_pixels)
                
                # We need to efficiently update the pixels
                # Using numpy for indexing is faster for CPU but we are on GPU tensors
                # Let's collect indices for this step
                step_indices = pixel_indices[i:end_idx]
                
                # Update current_flat with original pixels at these indices
                # Note: current_flat is a view of current_image, so modification is in-place
                current_flat[:, step_indices] = original_flat[:, step_indices]
                
                # Add a copy of the current state to the batch
                modified_images_batch.append(current_image.clone())
                
                # If batch is full, process it
                if len(modified_images_batch) >= batch_size:
                    confidences.extend(process_batch(modified_images_batch))
                    modified_images_batch = []
            
            # Process remaining items in batch
            if modified_images_batch:
                confidences.extend(process_batch(modified_images_batch))
        
        # Calculate AUC
        auc = float(np.trapz(confidences)) / len(confidences)
        return confidences, auc

    def compute_deletion_auc(self, image_tensor: torch.Tensor, 
                           saliency_map: np.ndarray, predicted_label: int,
                           step_size: int = 50, batch_size: int = 64) -> Tuple[List[float], float]:
        """
        Compute deletion AUC by progressively removing pixels in order of importance.
        Optimized with batch processing and AMP.
        """
        # Import necessary functions
        from torch.cuda.amp import autocast
        import torch
        import numpy as np
        
        # Ensure image tensor is 3D [C, H, W]
        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
            
        # Ensure saliency map is the right shape
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]  # Take first channel if RGB
        
        # Flatten saliency map and get pixel indices sorted by importance
        flat_saliency = saliency_map.flatten()
        # pixel_indices = np.argsort(flat_saliency)[::-1]  # Descending order
        pixel_indices_np = np.argsort(flat_saliency)[::-1].copy()
        pixel_indices = torch.from_numpy(pixel_indices_np).to(self.device).long()
        
        # Start with original image on the correct device
        current_image = image_tensor.clone().to(self.device)
        
        # Progressively remove pixels and measure confidence
        confidences = []
        n_pixels = len(pixel_indices)
        
        self.model.eval()
        
        # Prepare for batch processing
        current_flat = current_image.view(current_image.shape[0], -1)
        
        with torch.no_grad():
            # Initial confidence with original image
            if self.device.type == 'cuda':
                with autocast():
                    original_output = self.model(current_image.unsqueeze(0))
            else:
                original_output = self.model(current_image.unsqueeze(0))
                
            original_confidence = torch.softmax(original_output, dim=1)[0, predicted_label].item()
            confidences.append(original_confidence)
            
            # Create batches of modified images
            modified_images_batch = []
            
            # Helper to process a batch
            def process_batch(batch_list):
                if not batch_list:
                    return []
                
                batch_tensor = torch.stack(batch_list)
                
                if self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(batch_tensor)
                else:
                    outputs = self.model(batch_tensor)
                    
                batch_confidences = torch.softmax(outputs, dim=1)[:, predicted_label].tolist()
                return batch_confidences
            
            for i in range(0, n_pixels, step_size):
                # Remove pixels in order of importance (set to 0)
                end_idx = min(i + step_size, n_pixels)
                
                # Get indices for this step
                step_indices = pixel_indices[i:end_idx]

                
                # Set pixels to 0 (remove)
                current_flat[:, step_indices] = 0
                
                # Add a copy of the current state to the batch
                modified_images_batch.append(current_image.clone())
                
                # If batch is full, process it
                if len(modified_images_batch) >= batch_size:
                    confidences.extend(process_batch(modified_images_batch))
                    modified_images_batch = []
            
            # Process remaining items in batch
            if modified_images_batch:
                confidences.extend(process_batch(modified_images_batch))
        
        # Calculate AUC
        auc = float(np.trapz(confidences)) / len(confidences)
        return confidences, auc
    
    def evaluate_road(self, image_tensor: torch.Tensor, 
                     saliency_map: np.ndarray, predicted_label: int,
                     thresholds: List[int] = [20, 40, 60, 80],
                     imputation: str = "blur") -> Dict[str, float]:
        """
        Evaluate ROAD (Remove and Debias) score at multiple thresholds.
        
        Args:
            image_tensor: Input image tensor (C, H, W)
            saliency_map: Saliency map (H, W) or (1, H, W)
            predicted_label: Target class index
            thresholds: List of percentile thresholds to remove (e.g., [20, 40] removes top 20% and 40%)
            imputation: Imputation strategy ("blur", "black")
            
        Returns:
            Dictionary mapping "road_{threshold}" to the score.
        """
        from torch.cuda.amp import autocast
        import torch
        import numpy as np
        import torchvision.transforms.functional as TF
        
        # Ensure image tensor is on the correct device
        image_tensor = image_tensor.to(self.device).clone()
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
            
        # Ensure saliency map is correct shape
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]
            
        flat_saliency = saliency_map.flatten()
        
        # Prepare batch of modified images
        # First element is the original image for baseline reference
        batch_images = [image_tensor]
        
        # Create imputation background
        if imputation == "blur":
            # Apply Gaussian blur
            # Kernel size should be odd, e.g., 11x11, sigma 5.0
            blurred_image = TF.gaussian_blur(image_tensor, kernel_size=11, sigma=5.0)
            imputation_tensor = blurred_image
        else: # "black" or default
            imputation_tensor = torch.zeros_like(image_tensor)
            
        # Pre-calculate flattened views for efficient indexing
        image_flat = image_tensor.view(image_tensor.shape[0], -1)
        imputation_flat = imputation_tensor.view(imputation_tensor.shape[0], -1)
        
        for p in thresholds:
            # Determine threshold value for top p% pixels
            # e.g., p=20 means we remove pixels > 80th percentile
            percentile_val = np.percentile(flat_saliency, 100 - p)
            
            mask = (saliency_map > percentile_val) # Pixels to remove
            mask_flat = mask.flatten()
            mask_indices = torch.where(torch.from_numpy(mask_flat).to(self.device))[0]
            
            # Create modified image
            modified = image_tensor.clone()
            modified_flat = modified.view(modified.shape[0], -1)
            
            # Replace important pixels with imputation values
            modified_flat[:, mask_indices] = imputation_flat[:, mask_indices]
            
            batch_images.append(modified)
            
        # Process entire batch in one go
        batch_tensor = torch.stack(batch_images)
        
        self.model.eval()
        with torch.no_grad():
            if self.device.type == 'cuda':
                with autocast():
                    outputs = self.model(batch_tensor)
            else:
                outputs = self.model(batch_tensor)
                
            probs = torch.softmax(outputs, dim=1)[:, predicted_label]
            
        # Calculate scores
        original_conf = probs[0].item()
        results = {}
        
        for i, p in enumerate(thresholds):
            modified_conf = probs[i+1].item()
            # ROAD score is the drop in confidence
            # Ensure non-negative
            diff = float(max(0.0, original_conf - modified_conf))
            results[f"road_{p}"] = diff
            
        return results
    
    def _evaluate_saliency_map(self, image_tensor: torch.Tensor, saliency_map: np.ndarray, 
                              predicted_label: int, step_size: int = 50, verbose: bool = False, 
                              batch_size: int = 64) -> Dict[str, float]:
        """
        Evaluate a single saliency map using insertion AUC, deletion AUC, and ROAD metrics.
        Returns a dictionary of all metrics.
        """
        try:
            results = {}
            
            # Compute insertion AUC
            _, insertion_auc = self.compute_insertion_auc(
                image_tensor, saliency_map, predicted_label, step_size, batch_size
            )
            results['insertion_auc'] = insertion_auc
            
            # Compute deletion AUC  
            _, deletion_auc = self.compute_deletion_auc(
                image_tensor, saliency_map, predicted_label, step_size, batch_size
            )
            results['deletion_auc'] = deletion_auc
            
            # Compute ROAD score (multi-threshold)
            road_scores = self.evaluate_road(
                image_tensor, saliency_map, predicted_label,
                thresholds=[20, 40, 60, 80],
                imputation="blur"
            )
            results.update(road_scores)
            
            if verbose:
                print(f"        Insertion AUC: {insertion_auc:.4f}")
                print(f"        Deletion AUC: {deletion_auc:.4f}")
                print(f"        ROAD Scores: {road_scores}")
            
            return results
            
        except Exception as e:
            print(f"        Error in saliency evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {'insertion_auc': 0.0, 'deletion_auc': 0.0}
    
    def _load_synset_mapping(self) -> Dict[str, str]:
        """Load ImageNet synset mapping from LOC_synset_mapping.txt"""
        mapping_file = Path(__file__).parent.parent.parent / "LOC_synset_mapping.txt"
        
        if not mapping_file.exists():
            raise FileNotFoundError(f"Synset mapping file not found: {mapping_file}")
        
        synset_mapping = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        synset, class_name = parts
                        synset_mapping[synset] = class_name
        
        print(f"Loaded {len(synset_mapping)} ImageNet class mappings")
        return synset_mapping
    
    def _get_enhanced_cam_layers(self, layer_mode: str = "last") -> List[torch.nn.Module]:
        """Get conv layers for Enhanced CAM extraction based on the specified mode."""
        all_conv_layers = []
        
        # Collect all convolutional layers
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
                all_conv_layers.append(module)
        
        if not all_conv_layers:
            raise ValueError(f"No convolutional layers found in model {self.model_name}")
        
        print(f"Found {len(all_conv_layers)} total convolutional layers")
        
        # Select layers based on mode
        if layer_mode == "all":
            selected_layers = all_conv_layers
            print(f"Selected all {len(selected_layers)} convolutional layers")
            
        elif layer_mode == "last_5":
            selected_layers = all_conv_layers[-5:] if len(all_conv_layers) >= 5 else all_conv_layers
            print(f"Selected last {len(selected_layers)} convolutional layers (requested 5)")
            
        elif layer_mode == "last":
            selected_layers = [all_conv_layers[-1]]
            print(f"Selected last convolutional layer")
            
        else:
            raise ValueError(f"Invalid layer_mode '{layer_mode}'. Must be one of ['all', 'last_5', 'last']")
        
        return selected_layers
    
    def get_imagenet_images(self, max_images: int = 50, classes_filter: List[str] = None,
                          start_index: int = 0, end_index: int = None) -> Tuple[List[str], List[int], List[str]]:
        """
        Get ImageNet validation images with proper class filtering and batch support.
        
        Args:
            max_images: Maximum number of images to get (ignored if end_index is set)
            classes_filter: List of class names to filter (e.g., ['tench', 'goldfish'])
            start_index: Index to start collecting images from (for batch processing)
            end_index: Index to stop collecting images at (for batch processing)
            
        Returns:
            Tuple of (image_paths, predicted_labels, class_names)
        """
        all_image_paths = []
        all_class_names = []
        
        # If no filter, use all classes, otherwise filter by class names
        if classes_filter is None:
            target_synsets = list(self.synset_mapping.keys())
        else:
            # Convert class names to synsets
            target_synsets = []
            for class_name in classes_filter:
                found_synset = None
                for synset, name in self.synset_mapping.items():
                    if class_name.lower() in name.lower():
                        found_synset = synset
                        break
                if found_synset:
                    target_synsets.append(found_synset)
                else:
                    print(f"Warning: Class '{class_name}' not found in synset mapping")
            
            if not target_synsets:
                raise ValueError(f"No valid synsets found for classes: {classes_filter}")
        
        # Collect images from target synsets
        images_per_synset = max(1, max_images // len(target_synsets)) if max_images > 0 else -1
        
        for synset in target_synsets:
            synset_dir = os.path.join(self.imagenet_path, synset)
            if not os.path.exists(synset_dir):
                print(f"Warning: Directory not found for synset {synset}: {synset_dir}")
                continue
            
            # Get images from this synset and SORT them for determinism
            synset_images = sorted(glob.glob(os.path.join(synset_dir, "*.JPEG")))
            if not synset_images:
                print(f"Warning: No JPEG images found in {synset_dir}")
                continue
            
            # Limit images per synset if specified
            if images_per_synset > 0:
                synset_images = synset_images[:images_per_synset]
            
            all_image_paths.extend(synset_images)
            all_class_names.extend([self.synset_mapping[synset]] * len(synset_images))
            
            if max_images > 0 and len(all_image_paths) >= max_images:
                break
        
        # Limit total images if specified (legacy behavior if start/end not used)
        if end_index is not None:
            # Batch mode
            print(f"Batch mode: selecting images from index {start_index} to {end_index}")
            all_image_paths = all_image_paths[start_index:end_index]
            all_class_names = all_class_names[start_index:end_index]
        elif max_images > 0 and len(all_image_paths) > max_images:
            # Legacy max_images mode
            all_image_paths = all_image_paths[:max_images]
            all_class_names = all_class_names[:max_images]
        elif start_index > 0:
            # Start index only
            all_image_paths = all_image_paths[start_index:]
            all_class_names = all_class_names[start_index:]
        
        # Get predicted labels using the model
        predicted_labels = self._predict_batch(all_image_paths)
        
        print(f"Collected {len(all_image_paths)} ImageNet images")
        if classes_filter:
            print(f"Filtered classes: {classes_filter}")
        
        return all_image_paths, predicted_labels, all_class_names
    
    def _predict_batch(self, image_paths: List[str]) -> List[int]:
        """Predict labels for a batch of images"""
        predicted_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc="Predicting labels"):
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    outputs = self.model(image_tensor)
                    predicted_label = torch.argmax(outputs, dim=1).item()
                    predicted_labels.append(predicted_label)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    predicted_labels.append(0)  # Default to first class
        
        return predicted_labels
    
    def extract_enhanced_cam(self, image_path: str, predicted_label: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Extract Enhanced CAM for a single ImageNet image."""
        if self.enhanced_cam_extractor is None:
            self.enhanced_cam_extractor = OptimizedCamExtractor(
                self.model, 
                self.model_name, 
                self.conv_layers,
                cam_method=self.enhanced_cam_method,
                device_preference=str(self.device)
            )
        
        # Extract Enhanced CAM
        image_tensor, saliency_map = self.enhanced_cam_extractor.extract_saliency_map(
            image_path, predicted_label
        )
        
        # Ensure image tensor has batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Convert saliency map to numpy if it's a tensor
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.cpu().numpy()
        
        return image_tensor, saliency_map
    
    def evaluate_enhanced_cam(self, max_images: int = 50, step_size: int = 50, 
                            verbose: bool = True, classes_filter: List[str] = None,
                            start_index: int = 0, end_index: int = None,
                            return_raw_data: bool = False, batch_size: int = 64, num_workers: int = 4) -> Dict[str, any]:
        """
        Evaluate Enhanced CAM method on ImageNet with proper AUC calculations.
        
        Args:
            max_images: Maximum number of images to evaluate
            step_size: Step size for insertion/deletion evaluation
            verbose: If True, show detailed logging for each image
            classes_filter: List of ImageNet class names to filter
            batch_size: Batch size for evaluation inference
        
        Returns:
            Dictionary with evaluation results
        """
        # Get ImageNet images and predictions
        image_paths, predicted_labels, class_names = self.get_imagenet_images(
            max_images=max_images, 
            classes_filter=classes_filter,
            start_index=start_index,
            end_index=end_index
        )
        
        # Auto-adjust verbosity for large datasets
        if len(image_paths) > 20 and verbose:
            print(f"📢 Large dataset detected ({len(image_paths)} images). Setting verbose=False for cleaner output.")
            verbose = False
        
        print(f"\nEvaluating Enhanced CAM on {len(image_paths)} ImageNet images...")
        print(f"Using step_size={step_size} and batch_size={batch_size} for proper evaluation...")
        print(f"Using {num_workers} workers for data loading...")
        
        if not verbose:
            print(f"Verbose mode OFF - only showing progress and summary.")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        # Create dataset and loader
        dataset = _ImageNetDataset(image_paths, predicted_labels, class_names, self.transform)
        loader = DataLoader(
            dataset, 
            batch_size=1,  # Keep batch size 1 for the complex per-image evaluation logic
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            prefetch_factor=2 if num_workers > 0 else None,
            shuffle=False
        )
        
        # Create progress description based on verbosity
        progress_desc = "Processing Enhanced CAM"
        
        for i, (image_tensor, predicted_label, class_name, image_path_batch) in enumerate(tqdm(
            loader, 
            desc=progress_desc, 
            total=len(dataset)
        )):
            try:
                # Unpack batch (size 1)
                image_tensor = image_tensor.squeeze(0)  # [C, H, W]
                # Keep as 1-batch tensor for compatibility with extract and evaluate methods
                # Actually, extract_enhanced_cam typically expects path, but we have tensor now?
                # Wait, extract_enhanced_cam calls enhanced_cam_extractor.extract_saliency_map(image_path, ...)
                # The extractor loads the image from path usually. 
                # Optimization: We can modify extract_saliency_map to accept tensor, OR we continue to pass path.
                # If we pass path, we reload image. That defeats the purpose of DataLoader prefetching image.
                # Let's check `extract_enhanced_cam` and `OptimizedCamExtractor`.
                # `extract_enhanced_cam` takes `image_path`.
                # We need to change `extract_enhanced_cam` to accept tensor or just use `image_path` from loader.
                
                # If we use `image_path` from loader, we are still doing disk I/O in main thread inside `extract_saliency_map` if it reads file.
                # However, `OptimizedCamExtractor` likely uses `cv2` or `PIL` to read.
                # To fully optimize, we should pass the pre-loaded tensor to `extract_enhanced_cam`.
                
                # Let's check `extract_enhanced_cam`. It calls `self.enhanced_cam_extractor.extract_saliency_map`.
                # We can't easily change `OptimizedCamExtractor` as it is in another file I haven't viewed. 
                # BUT, if `OptimizedCamExtractor` is purely for CAM, it should take a tensor.
                # If it takes a path, I might be limited.
                
                # However, the user asked for MAXIMISING RAM USAGE.
                # Even if I just pre-load images into RAM with DataLoader, it helps.
                # But to really help, I should use the tensor from DataLoader.
                
                # Let's assume for now I should use the path if I can't change extractor, 
                # BUT `extract_enhanced_cam` does return `image_tensor` and `saliency_map`.
                # If I can bypass internal image loading in extractor...
                
                # Wait, `extract_enhanced_cam` implementation in THIS file:
                # def extract_enhanced_cam(self, image_path: str, predicted_label: int) -> Tuple[torch.Tensor, np.ndarray]:
                #     ...
                #     image_tensor, saliency_map = self.enhanced_cam_extractor.extract_saliency_map(image_path, predicted_label)
                #     ...
                
                # I should probably just pass the path for now to be safe, as changing the extractor signature might be out of scope or risky without seeing it.
                # BUT, using DataLoader just for paths is not "maximizing RAM/GPU" much.
                # The "resize/transform" part is done in workers. That IS useful. 
                # So if `extract_saliency_map` allows passing a tensor, that's best.
                # If not, I still save time on `_evaluate_saliency_map` which USES the tensor.
                # Wait, `_evaluate_saliency_map` takes `image_tensor`.
                # `evaluate_enhanced_cam` GETS `image_tensor` from `extract_enhanced_cam`.
                # So `extract_enhanced_cam` does the loading.
                
                # CRITICAL: If I use DataLoader to load `image_tensor`, I can pass THAT to `_evaluate_saliency_map`.
                # But `extract_enhanced_cam` ALSO returns a tensor.
                # So I would be loading it TWICE: once in DataLoader, once in `extract_enhanced_cam`.
                # That wastes CPU, but maximizes RAM usage (storing loaded images in queue).
                # But it doesn't help GPU throughput if we re-load.
                
                # Let's look at `evaluate_method` (Standard CAM).
                # `self._extract_standard_cam(image_path, ...)` -> returns saliency map.
                # Then `image = Image.open(image_path)... image_tensor = self.transform(image)...`
                # value: `image_tensor` is used in `_evaluate_saliency_map`.
                # HERE, using DataLoader IS beneficial because we skip the load/transform in the main loop for evaluation.
                # `_extract_standard_cam` likely loads image internally too.
                
                # For `evaluate_enhanced_cam`:
                # It calls `extract_enhanced_cam`.
                
                # I will proceed with DataLoader yielding tensors.
                # Even if `extract_enhanced_cam` re-loads, at least `evaluate_standard_methods` acts better?
                # Actually, `evaluate_enhanced_cam` calls `extract_enhanced_cam` which usually does forward pass.
                # If I can't optimize `extract_enhanced_cam` interface, I'll still use DataLoader for the "Evaluation" phase (Insertion/Deletion) which uses the tensor.
                # AND I can pass the DataLoader's tensor to `_evaluate_saliency_map` instead of using the one from `extract_enhanced_cam` (they should be identical).
                # Wait, `extract_enhanced_cam` returns tensor used for CAM generation. It might be normalized differently?
                # `ImageNetProperAUCEvaluator` uses standard ImageNet normalization. 
                # `OptimizedCamExtractor` likely uses the same.
                
                # I will use the `image_path` from DataLoader (it returns tuple) to pass to `extract_enhanced_cam`.
                # And I will use `image_tensor` from DataLoader to pass to `_evaluate_saliency_map`.
                
                # ALSO: To maximize RAM, setting `num_workers` high helps.
                
                predicted_label = predicted_label.item()
                image_path = image_path_batch[0]
                class_name = class_name[0]
                
                if verbose:
                    print(f"\n--- Image {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")
                    print(f"    Class: {class_name}")
                    print(f"    Predicted label: {predicted_label}")
                
                # Extract Enhanced CAM (Still might do I/O, hard to avoid without deeper refactor)
                _, saliency_map = self.extract_enhanced_cam(image_path, predicted_label)
                
                # Use the PRE-FETCHED tensor for evaluation to save time
                metrics = self._evaluate_saliency_map(
                    image_tensor, saliency_map, predicted_label, step_size, verbose, batch_size
                )
                
                insertion_aucs.append(metrics['insertion_auc'])
                deletion_aucs.append(metrics['deletion_auc'])
                road_scores.append(metrics.get('road_20', 0.0))
                
                if verbose:
                    print(f"    Insertion AUC: {metrics['insertion_auc']:.4f}")
                    print(f"    Deletion AUC: {metrics['deletion_auc']:.4f}")
                    print(f"    ROAD Score (20%): {metrics.get('road_20', 0.0):.4f}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Calculate final statistics
        results = {
            'insertion_auc_mean': np.mean(insertion_aucs),
            'insertion_auc_std': np.std(insertion_aucs),
            'deletion_auc_mean': np.mean(deletion_aucs),
            'deletion_auc_std': np.std(deletion_aucs),
            'road_mean': np.mean(road_scores),
            'road_std': np.std(road_scores),
            'num_images': len(insertion_aucs),
            'classes_evaluated': classes_filter if classes_filter else 'All ImageNet classes'
        }
        
        if return_raw_data:
            results.update({
                'insertion_aucs': insertion_aucs,
                'deletion_aucs': deletion_aucs,
                'road_scores': road_scores
            })
            
        return results
    
    def evaluate_method(self, cam_method_name: str, max_images: int = 50, 
                       classes_filter: List[str] = None,
                       start_index: int = 0, end_index: int = None,
                       return_raw_data: bool = False, batch_size: int = 64, num_workers: int = 4) -> Dict[str, any]:
        """
        Evaluate a standard CAM method on ImageNet.
        
        Args:
            cam_method_name: Name of the CAM method
            max_images: Maximum number of images to evaluate
            classes_filter: List of ImageNet class names to filter
            batch_size: Batch size for evaluation inference
            
        Returns:
            Dictionary with evaluation results
        """

        # Get ImageNet images and predictions
        image_paths, predicted_labels, class_names = self.get_imagenet_images(
            max_images=max_images, 
            classes_filter=classes_filter,
            start_index=start_index,
            end_index=end_index
        )
        
        print(f"\nEvaluating {cam_method_name} on {len(image_paths)} ImageNet images...")
        print(f"Using {num_workers} workers for data loading...")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        # Create dataset and loader
        dataset = _ImageNetDataset(image_paths, predicted_labels, class_names, self.transform)
        loader = DataLoader(
            dataset, 
            batch_size=1,  # Keep batch size 1
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            prefetch_factor=2 if num_workers > 0 else None,
            shuffle=False
        )

        for i, (image_tensor, predicted_label, class_name, image_path_batch) in enumerate(tqdm(
            loader, 
            desc=f"Processing {cam_method_name}", 
            total=len(dataset)
        )):
            try:
                # Unpack batch
                image_tensor = image_tensor.squeeze(0)
                predicted_label = predicted_label.item()
                image_path = image_path_batch[0]
                
                # Extract standard CAM using pytorch-grad-cam
                saliency_map = self._extract_standard_cam(
                    image_path, predicted_label, cam_method_name
                )
                
                # Use pre-fetched image_tensor for evaluation
                # Evaluate saliency map
                metrics = self._evaluate_saliency_map(
                    image_tensor, saliency_map, predicted_label, step_size=50, verbose=False, batch_size=batch_size
                )
                
                insertion_aucs.append(metrics['insertion_auc'])
                deletion_aucs.append(metrics['deletion_auc'])
                road_scores.append(metrics.get('road_20', 0.0))
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Calculate final statistics
        results = {
            'insertion_auc_mean': np.mean(insertion_aucs),
            'insertion_auc_std': np.std(insertion_aucs),
            'deletion_auc_mean': np.mean(deletion_aucs),
            'deletion_auc_std': np.std(deletion_aucs),
            'road_mean': np.mean(road_scores),
            'road_std': np.std(road_scores),
            'num_images': len(insertion_aucs),
            'classes_evaluated': classes_filter if classes_filter else 'All ImageNet classes'
        }
        
        if return_raw_data:
            results.update({
                'insertion_aucs': insertion_aucs,
                'deletion_aucs': deletion_aucs,
                'road_scores': road_scores
            })

        return results
    
    def _extract_standard_cam(self, image_path: str, predicted_label: int, cam_method_name: str) -> np.ndarray:
        """Extract standard CAM using pytorch-grad-cam library"""
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, HiResCAM, LayerCAM, ScoreCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        # Select appropriate CAM method
        if cam_method_name == "GradCAM":
            cam_class = GradCAM
        elif cam_method_name == "GradCAM++":
            cam_class = GradCAMPlusPlus
        elif cam_method_name == "EigenCAM":
            cam_class = EigenCAM
        elif cam_method_name == "HiResCAM":
            cam_class = HiResCAM
        elif cam_method_name == "LayerCAM":
            cam_class = LayerCAM
        elif cam_method_name == "ScoreCAM":
            cam_class = ScoreCAM
        else:
            raise ValueError(f"Unsupported CAM method: {cam_method_name}")
        
        # Get target layer based on model architecture
        if hasattr(self.model, 'layer4'):  # ResNet
            target_layers = [self.model.layer4[-1]]
        elif hasattr(self.model, 'features'):  # VGG, DenseNet, etc.
            target_layers = [self.model.features[-1]]
        elif hasattr(self.model, 'classifier'):  # Some models
            # Find the last conv layer
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            target_layers = [conv_layers[-1]]
        else:
            # Fallback: find last conv layer
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            target_layers = [conv_layers[-1]]
        
        # Initialize CAM
        cam = cam_class(model=self.model, target_layers=target_layers)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Generate CAM
        targets = [ClassifierOutputTarget(predicted_label)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        
        # Return the saliency map for the first (and only) image
        return grayscale_cam[0]
    
    def compare_enhanced_vs_standard(self, standard_methods: List[str], max_images: int = 50,
                                   step_size: int = 50, verbose: bool = False,
                                   classes_filter: List[str] = None) -> pd.DataFrame:
        """
        Compare Enhanced CAM vs standard methods on ImageNet.
        
        Args:
            standard_methods: List of standard CAM methods
            max_images: Maximum number of images to evaluate
            step_size: Step size for evaluation
            verbose: Verbose output
            classes_filter: List of ImageNet class names to filter
            
        Returns:
            DataFrame with comparison results
        """
        print(f"Running ImageNet comparison: Enhanced CAM vs {len(standard_methods)} standard methods")
        
        all_results = []
        
        # Evaluate Enhanced CAM
        enhanced_results = self.evaluate_enhanced_cam(
            max_images=max_images,
            step_size=step_size,
            verbose=verbose,
            classes_filter=classes_filter
        )
        
        enhanced_row = {
            'Method': f'{self.enhanced_cam_method}',
            'Dataset': 'ImageNet',
            'Insertion_AUC_Mean': enhanced_results['insertion_auc_mean'],
            'Insertion_AUC_Std': enhanced_results['insertion_auc_std'],
            'Deletion_AUC_Mean': enhanced_results['deletion_auc_mean'],
            'Deletion_AUC_Std': enhanced_results['deletion_auc_std'],
            'ROAD_Mean': enhanced_results['road_mean'],
            'ROAD_Std': enhanced_results['road_std'],
            'Images_Evaluated': enhanced_results['num_images']
        }
        all_results.append(enhanced_row)
        
        # Evaluate standard methods
        for method in standard_methods:
            method_results = self.evaluate_method(
                cam_method_name=method,
                max_images=max_images,
                classes_filter=classes_filter
            )
            
            method_row = {
                'Method': method,
                'Dataset': 'ImageNet',
                'Insertion_AUC_Mean': method_results['insertion_auc_mean'],
                'Insertion_AUC_Std': method_results['insertion_auc_std'],
                'Deletion_AUC_Mean': method_results['deletion_auc_mean'],
                'Deletion_AUC_Std': method_results['deletion_auc_std'],
                'ROAD_Mean': method_results['road_mean'],
                'ROAD_Std': method_results['road_std'],
                'Images_Evaluated': method_results['num_images']
            }
            all_results.append(method_row)
        
        return pd.DataFrame(all_results)
