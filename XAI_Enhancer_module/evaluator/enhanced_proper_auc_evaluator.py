#!/usr/bin/env python3
"""
Enhanced Proper AUC Evaluator - Extended to support Enhanced CAM method.
This module uses the ProperAUCEvaluator as the base for all evaluations.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the base ProperAUCEvaluator
from XAI_Enhancer_module.evaluator.proper_auc_evaluation import ProperAUCEvaluator

# Local imports for Enhanced CAM
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
from XAI_Enhancer_module.utils.optimized_predictor import get_optimized_predictions
from XAI_Enhancer_module.utils.model_utils import get_validation_paths, TRAIN_DATA_PATH
from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, save_analysis_data


class EnhancedProperAUCEvaluator(ProperAUCEvaluator):
    """
    Extended evaluator that supports both standard CAM methods and Enhanced CAM.
    Uses ProperAUCEvaluator as the base for consistent AUC calculations.
    Supports different layer selection modes for Enhanced CAM.
    """
    
    def __init__(self, 
                 model, 
                 model_name: str, 
                 dataset_path: str, 
                 layer_mode: str = "last",
                 enhanced_cam_method: str = "GradCAMEnhanced"):
        """
        Initialize the Enhanced CAM evaluator.
        
        Args:
            model: Trained model for evaluation
            model_name: Name of the model (resnet18, resnet34, etc.)
            dataset_path: Path to ImageNet validation dataset
            layer_mode: Layer selection mode ("last", "last_5", "all")
            enhanced_cam_method: Enhanced CAM method to use
        """
        # Initialize attributes without calling super().__init__ to avoid loading model twice
        self.model_name = model_name
        self.device = next(model.parameters()).device
        self.dataset_path = dataset_path
        
        # Use the passed model instead of loading a new one
        self.model = model
        self.model.eval()
        
        # Get image size
        self.img_size = 224
        if model_name in ('b4',):
            self.img_size = 384
            
        print(f"ProperAUCEvaluator initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Image size: {self.img_size}")
        
        # Store enhanced CAM method
        self.enhanced_cam_method = enhanced_cam_method
        
        # Validate layer mode
        valid_modes = ["all", "last_5", "last"]
        if layer_mode not in valid_modes:
            raise ValueError(f"Invalid layer_mode '{layer_mode}'. Must be one of {valid_modes}")
        
        self.layer_mode = layer_mode
        
        # Initialize Enhanced CAM components
        self.conv_layers = self._get_enhanced_cam_layers(layer_mode)
        self.enhanced_cam_extractor = None
        
        print(f"EnhancedProperAUCEvaluator initialized with Enhanced CAM support")
        print(f"  Enhanced CAM method: {enhanced_cam_method}")
        print(f"  Layer mode: {layer_mode}")
        print(f"  Number of layers: {len(self.conv_layers)}")
    
    def _get_target_layers(self):
        """Get target layers for the model."""
        if 'resnet' in self.model_name:
            return [self.model.layer4[-1]]
        elif 'densenet' in self.model_name:
            return [self.model.features.norm5]
        elif self.model_name.startswith('b'):  # EfficientNet
            return [self.model.features[-1]]
        else:
            # Find last conv layer
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return [module]
            raise ValueError(f"Could not find target layer for model {self.model_name}")
    
    def extract_cam(self, image_path: str, predicted_label: int, 
                   cam_method_name: str = "GradCAM") -> Tuple[torch.Tensor, np.ndarray]:
        """Extract CAM for a single image."""
        from PIL import Image
        import torchvision.transforms as transforms
        from XAI_Enhancer_module.utils.model_utils import MEAN, STD
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Create transform
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get CAM method
        from pytorch_grad_cam import (
            GradCAM, GradCAMPlusPlus, EigenGradCAM, EigenCAM, 
            HiResCAM, LayerCAM, ScoreCAM
        )
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        cam_methods = {
            "GradCAM": GradCAM,
            "GradCAM++": GradCAMPlusPlus,
            "EigenGradCAM": EigenGradCAM,
            "EigenCAM": EigenCAM,
            "HiResCAM": HiResCAM,
            "LayerCAM": LayerCAM,
            "ScoreCAM": ScoreCAM,
        }
        
        if cam_method_name not in cam_methods:
            raise ValueError(f"Unknown CAM method: {cam_method_name}")
        
        cam_class = cam_methods[cam_method_name]
        target_layers = self._get_target_layers()
        
        # Generate CAM
        cam = cam_class(model=self.model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_label)]
        
        with cam:
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        
        return image_tensor, grayscale_cam[0]
    
    def compute_insertion_auc(self, image_tensor: torch.Tensor, 
                            saliency_map: np.ndarray, 
                            target_class: int, 
                            step_size: int = 224) -> Tuple[np.ndarray, float]:
        """Compute insertion AUC with proper normalization."""
        import torch.nn.functional as F
        
        # Flatten and sort pixels by importance (descending)
        flat_saliency = saliency_map.flatten()
        sorted_indices = np.argsort(-flat_saliency)  # Negative for descending
        
        total_pixels = self.img_size * self.img_size
        n_steps = (total_pixels + step_size - 1) // step_size
        
        scores = np.zeros(n_steps + 1)
        
        # Start with blurred baseline
        baseline_image = self._create_blurred_baseline(image_tensor)
        current_image = baseline_image.clone()
        
        # Initial score (blurred baseline)
        with torch.no_grad():
            output = self.model(current_image)
            prob = F.softmax(output, dim=1)[0, target_class].item()
        scores[0] = prob
        
        # Progressive insertion
        flat_original = image_tensor.view(-1, total_pixels)  # Shape: (C, H*W)
        flat_current = current_image.view(-1, total_pixels)
        
        for step in range(n_steps):
            start_idx = step * step_size
            end_idx = min((step + 1) * step_size, total_pixels)
            
            # Get pixel indices to insert in this step
            pixel_indices = sorted_indices[start_idx:end_idx]
            
            # Insert important pixels from original image
            flat_current[:, pixel_indices] = flat_original[:, pixel_indices]
            
            current_image = flat_current.view(image_tensor.shape)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(current_image)
                prob = F.softmax(output, dim=1)[0, target_class].item()
            scores[step + 1] = prob
        
        # Calculate AUC using trapezoidal rule, normalized to [0,1]
        x = np.linspace(0, 1, len(scores))
        auc = np.trapz(scores, x)
        
        return scores, auc
    
    def compute_deletion_auc(self, image_tensor: torch.Tensor, 
                           saliency_map: np.ndarray, 
                           target_class: int, 
                           step_size: int = 224) -> Tuple[np.ndarray, float]:
        """Compute deletion AUC with proper normalization."""
        import torch.nn.functional as F
        
        # Flatten and sort pixels by importance (descending)
        flat_saliency = saliency_map.flatten()
        sorted_indices = np.argsort(-flat_saliency)  # Negative for descending
        
        total_pixels = self.img_size * self.img_size
        n_steps = (total_pixels + step_size - 1) // step_size
        
        scores = np.zeros(n_steps + 1)
        current_image = image_tensor.clone()
        
        # Initial score (original image)
        with torch.no_grad():
            output = self.model(current_image)
            prob = F.softmax(output, dim=1)[0, target_class].item()
        scores[0] = prob
        
        # Progressive deletion
        flat_current = current_image.view(-1, total_pixels)  # Shape: (C, H*W)
        flat_baseline = self._create_blurred_baseline(image_tensor).view(-1, total_pixels)
        
        for step in range(n_steps):
            start_idx = step * step_size
            end_idx = min((step + 1) * step_size, total_pixels)
            
            # Get pixel indices to delete in this step
            pixel_indices = sorted_indices[start_idx:end_idx]
            
            # Replace important pixels with blurred baseline
            flat_current[:, pixel_indices] = flat_baseline[:, pixel_indices]
            
            current_image = flat_current.view(image_tensor.shape)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(current_image)
                prob = F.softmax(output, dim=1)[0, target_class].item()
            scores[step + 1] = prob
        
        # Calculate AUC using trapezoidal rule, normalized to [0,1]
        x = np.linspace(0, 1, len(scores))
        auc = np.trapz(scores, x)
        
        return scores, auc
        
    def _create_blurred_baseline(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Create blurred baseline image."""
        import torch.nn.functional as F
        
        # Ensure tensor has batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Ensure tensor is in correct format (batch, channels, height, width)
        if len(image_tensor.shape) == 4:
            batch, channels, height, width = image_tensor.shape
            if channels > 100:  # If channels is suspiciously large (like 224), likely wrong format
                # This shouldn't happen anymore with the fix above, but keep as safety
                print(f"Warning: Unexpected tensor shape {image_tensor.shape}, skipping blur")
                return image_tensor
        
        # Create Gaussian blur kernel
        def get_gaussian_kernel(size=11, sigma=5):
            """Create 2D Gaussian kernel."""
            x = torch.arange(-size//2 + 1., size//2 + 1.)
            x = torch.exp(-0.5 * (x / sigma).pow(2))
            kernel = x[:, None] * x[None, :]
            kernel = kernel / kernel.sum()
            return kernel
        
        kernel = get_gaussian_kernel().to(image_tensor.device)
        kernel = kernel.expand(image_tensor.size(1), 1, kernel.size(0), kernel.size(1))
        
        # Apply Gaussian blur
        blurred = F.conv2d(image_tensor, kernel, padding=kernel.size(-1)//2, groups=image_tensor.size(1))
        return blurred
    
    def evaluate_road(self, image_tensor: torch.Tensor, 
                     saliency_map: np.ndarray, 
                     target_class: int) -> float:
        """Evaluate using ROAD metric."""
        try:
            from pytorch_grad_cam.metrics.road import ROADCombined
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
            
            road_metric = ROADCombined(percentiles=[20, 40, 60, 80])
            
            # Ensure tensor has correct shape
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            # Fix saliency map dimensions for ROAD
            if saliency_map.ndim == 3:
                # Remove extra dimension if present (1, H, W) -> (H, W)
                saliency_map = saliency_map.squeeze(0)
            
            # ROAD expects saliency_map with shape (batch_size, H, W)
            saliency_tensor = np.expand_dims(saliency_map, axis=0)
            
            targets = [ClassifierOutputSoftmaxTarget(target_class)]
            
            scores = road_metric(image_tensor, saliency_tensor, targets, self.model)
            return scores[0]
        except Exception as e:
            print(f"ROAD evaluation failed: {e}")
            return 0.0
    
    def evaluate_method(self, cam_method_name: str, max_images: int = -1, sample_paths: list = None):
        """Evaluate a CAM method with proper AUC calculations."""
        from XAI_Enhancer_module.utils.model_utils import get_validation_paths, TRAIN_DATA_PATH
        from XAI_Enhancer_module.utils.optimized_predictor import get_optimized_predictions
        
        # Use dataset_path if available, otherwise fall back to TRAIN_DATA_PATH
        data_path = getattr(self, 'dataset_path', TRAIN_DATA_PATH) if hasattr(self, 'dataset_path') else TRAIN_DATA_PATH
        
        # Get image paths and predictions
        all_image_paths = get_validation_paths(data_path)
        
        # Handle special cases for max_images
        if max_images == -1 or max_images is None:
            # Use entire validation dataset
            image_paths = all_image_paths
        else:
            # Use specified number of images
            image_paths = all_image_paths[:max_images]
        predicted_labels, _, _ = get_optimized_predictions(
            self.model_name, image_paths, use_validation_set=False
        )
        
        print(f"\nEvaluating {cam_method_name} on {len(image_paths)} images...")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        for image_path, predicted_label in tqdm(zip(image_paths, predicted_labels), 
                                              desc=f"Processing {cam_method_name}"):
            try:
                # Extract CAM
                image_tensor, saliency_map = self.extract_cam(
                    image_path, predicted_label, cam_method_name
                )
                
                # Compute insertion AUC
                _, insertion_auc = self.compute_insertion_auc(
                    image_tensor, saliency_map, predicted_label
                )
                insertion_aucs.append(insertion_auc)
                
                # Compute deletion AUC
                _, deletion_auc = self.compute_deletion_auc(
                    image_tensor, saliency_map, predicted_label
                )
                deletion_aucs.append(deletion_auc)
                
                # Compute ROAD score
                road_score = self.evaluate_road(
                    image_tensor, saliency_map, predicted_label
                )
                road_scores.append(road_score)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Compile results
        results = {
            'cam_method': cam_method_name,
            'model_name': self.model_name,
            'num_images': len(insertion_aucs),
            'insertion_auc_mean': np.mean(insertion_aucs),
            'insertion_auc_std': np.std(insertion_aucs),
            'deletion_auc_mean': np.mean(deletion_aucs),
            'deletion_auc_std': np.std(deletion_aucs),
            'road_mean': np.mean(road_scores),
            'road_std': np.std(road_scores),
            'insertion_aucs': insertion_aucs,
            'deletion_aucs': deletion_aucs,
            'road_scores': road_scores
        }
        
        return results
    
    def _get_enhanced_cam_layers(self, layer_mode: str = "last") -> List[torch.nn.Module]:
        """
        Get conv layers for Enhanced CAM extraction based on the specified mode.
        
        Args:
            layer_mode: Mode for layer selection
                - "all": All convolutional layers
                - "last_5": Last 5 convolutional layers
                - "last": Only the last convolutional layer
                
        Returns:
            List of selected convolutional layers
        """
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
            # Fallback to model-specific selection for backward compatibility
            if 'resnet' in self.model_name:
                selected_layers = [self.model.layer4[-1]]
            elif 'densenet' in self.model_name:
                selected_layers = [self.model.features.norm5]
            elif self.model_name.startswith('b'):  # EfficientNet
                selected_layers = [self.model.features[-1]]
            else:
                # Find last conv layer
                for name, module in reversed(list(self.model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        selected_layers = [module]
                        break
                else:
                    raise ValueError(f"Could not find conv layers for model {self.model_name}")
        
        return selected_layers
    
    def extract_enhanced_cam(self, image_path: str, predicted_label: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Extract Enhanced CAM for a single image."""
        if self.enhanced_cam_extractor is None:
            self.enhanced_cam_extractor = OptimizedCamExtractor(
                self.model, 
                self.model_name, 
                self.conv_layers,
                cam_method=self.enhanced_cam_method
            )
        
        # Extract Enhanced CAM
        image_tensor, saliency_map = self.enhanced_cam_extractor.extract_saliency_map(
            image_path, predicted_label
        )
        
        # Ensure tensor has batch dimension for evaluation
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Ensure tensor is on the correct device
        image_tensor = image_tensor.to(self.device)
        
        # Convert saliency map to numpy if it's a tensor
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.cpu().numpy()
        
        return image_tensor, saliency_map
    
    def evaluate_enhanced_cam(self, max_images: int = 2, step_size: int = 50, 
                            verbose: bool = True) -> Dict[str, any]:
        """
        Evaluate Enhanced CAM method with proper AUC calculations.
        
        Args:
            max_images: Maximum number of images to evaluate
            step_size: Step size for insertion/deletion evaluation
            verbose: If True, show detailed logging for each image. 
                    If False, only show progress bar and summary.
        
        Returns:
            Dictionary with evaluation results
        """
        # Get image paths and predictions
        all_image_paths = get_validation_paths(TRAIN_DATA_PATH)
        
        # Handle special cases for max_images
        if max_images == -1 or max_images is None:
            # Use entire validation dataset
            image_paths = all_image_paths
        else:
            # Use specified number of images
            image_paths = all_image_paths[:max_images]
            
        predicted_labels, _, _ = get_optimized_predictions(
            self.model_name, image_paths, use_validation_set=False
        )
        
        # Auto-adjust verbosity for large datasets
        if len(image_paths) > 20 and verbose:
            print(f"📢 Large dataset detected ({len(image_paths)} images). Setting verbose=False for cleaner output.")
            verbose = False
        
        print(f"\nEvaluating Enhanced CAM on {len(image_paths)} images...")
        print(f"Using step_size={step_size} for proper evaluation...")
        if not verbose:
            print(f"Verbose mode OFF - only showing progress and summary.")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        # Create progress description based on verbosity
        progress_desc = "Processing Enhanced CAM"
        
        for i, (image_path, predicted_label) in enumerate(tqdm(zip(image_paths, predicted_labels), 
                                                             desc=progress_desc, 
                                                             total=len(image_paths))):
            try:
                # Extract Enhanced CAM
                image_tensor, saliency_map = self.extract_enhanced_cam(
                    image_path, predicted_label
                )
                
                # Detailed logging only if verbose
                if verbose:
                    print(f"  Processing: {Path(image_path).name}")
                    print(f"    Saliency range: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
                
                # Compute insertion AUC with proper step size
                _, insertion_auc = self.compute_insertion_auc(
                    image_tensor, saliency_map, predicted_label, step_size=step_size
                )
                insertion_aucs.append(insertion_auc)
                if verbose:
                    print(f"    Insertion AUC: {insertion_auc:.4f}")
                
                # Compute deletion AUC with proper step size
                _, deletion_auc = self.compute_deletion_auc(
                    image_tensor, saliency_map, predicted_label, step_size=step_size
                )
                deletion_aucs.append(deletion_auc)
                if verbose:
                    print(f"    Deletion AUC: {deletion_auc:.4f}")
                
                # Compute ROAD score
                road_score = self.evaluate_road(
                    image_tensor, saliency_map, predicted_label
                )
                road_scores.append(road_score)
                if verbose:
                    print(f"    ROAD Score: {road_score:.4f}")
                
                # Show periodic progress for non-verbose mode
                elif (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                    print(f"📊 Processed {i + 1}/{len(image_paths)} images. " +
                          f"Current averages: Ins={np.mean(insertion_aucs):.3f}, " +
                          f"Del={np.mean(deletion_aucs):.3f}, ROAD={np.mean(road_scores):.3f}")
                
            except Exception as e:
                error_msg = f"Error processing {Path(image_path).name}: {e}"
                if verbose:
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                else:
                    # Just log the error briefly
                    print(f"⚠️  {error_msg}")
                continue
        
        # Compile results
        results = {
            'cam_method': f'{self.enhanced_cam_method} ({self.layer_mode})',
            'model_name': self.model_name,
            'layer_mode': self.layer_mode,
            'num_layers': len(self.conv_layers),
            'num_images': len(insertion_aucs),
            'step_size': step_size,
            'insertion_auc_mean': np.mean(insertion_aucs) if insertion_aucs else 0.0,
            'insertion_auc_std': np.std(insertion_aucs) if insertion_aucs else 0.0,
            'deletion_auc_mean': np.mean(deletion_aucs) if deletion_aucs else 0.0,
            'deletion_auc_std': np.std(deletion_aucs) if deletion_aucs else 0.0,
            'road_mean': np.mean(road_scores) if road_scores else 0.0,
            'road_std': np.std(road_scores) if road_scores else 0.0,
            'insertion_aucs': insertion_aucs,
            'deletion_aucs': deletion_aucs,
            'road_scores': road_scores
        }
        
        # Save individual Enhanced CAM results to CSV
        try:
            # Create a simple DataFrame for individual results
            individual_df = pd.DataFrame([{
                'Method': results['cam_method'],
                'Model': results['model_name'],
                'Layer_Mode': results['layer_mode'],
                'Num_Layers': results['num_layers'],
                'Num_Images': results['num_images'],
                'Step_Size': results['step_size'],
                'Insertion_AUC_Mean': results['insertion_auc_mean'],
                'Insertion_AUC_Std': results['insertion_auc_std'],
                'Deletion_AUC_Mean': results['deletion_auc_mean'],
                'Deletion_AUC_Std': results['deletion_auc_std'],
                'ROAD_Mean': results['road_mean'],
                'ROAD_Std': results['road_std']
            }])
            
            saved_path = save_evaluation_results(
                individual_df,
                self.model_name,
                evaluation_type=f"enhanced_cam_{self.layer_mode}",
                add_timestamp=True
            )
            print(f"\n💾 Enhanced CAM results saved to: {saved_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save Enhanced CAM results to CSV: {e}")
        
        # Save detailed results to pickle file
        try:
            pickle_path = save_analysis_data(
                results,
                self.model_name,
                analysis_type=f"enhanced_cam_{self.layer_mode}_detailed",
                add_timestamp=True
            )
            print(f"💾 Detailed Enhanced CAM data saved to: {pickle_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save detailed results to pickle: {e}")
        
        return results
    
    def compare_enhanced_vs_standard(self, standard_methods: List[str] = None, 
                                   max_images: int = 2, step_size: int = 50,
                                   verbose: bool = None) -> pd.DataFrame:
        """
        Compare Enhanced CAM with standard CAM methods.
        
        Args:
            standard_methods: List of standard CAM methods to compare
            max_images: Maximum number of images to evaluate
            step_size: Step size for evaluation
            verbose: If None, auto-determine based on max_images
        
        Returns:
            DataFrame with comparison results
        """
        if standard_methods is None:
            standard_methods = ["GradCAM", "GradCAM++", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"]
        
        # Auto-determine verbosity if not specified
        if verbose is None:
            verbose = max_images <= 20
        
        results = []
        
        # Evaluate Enhanced CAM
        print(f"\n{'='*60}")
        print(f"Evaluating Enhanced CAM")
        print(f"{'='*60}")
        
        enhanced_results = self.evaluate_enhanced_cam(max_images, step_size, verbose)
        
        enhanced_summary = {
            'Method': f'Enhanced CAM ({self.layer_mode})',
            'Layers': f'{len(self.conv_layers)} layers',
            'Insertion AUC': f"{enhanced_results['insertion_auc_mean']:.4f} ± {enhanced_results['insertion_auc_std']:.4f}",
            'Deletion AUC': f"{enhanced_results['deletion_auc_mean']:.4f} ± {enhanced_results['deletion_auc_std']:.4f}",
            'ROAD Score': f"{enhanced_results['road_mean']:.4f} ± {enhanced_results['road_std']:.4f}",
            'Num Images': enhanced_results['num_images']
        }
        results.append(enhanced_summary)
        
        # Print Enhanced CAM results
        print(f"\nResults for Enhanced CAM:")
        print(f"  Insertion AUC: {enhanced_results['insertion_auc_mean']:.4f} ± {enhanced_results['insertion_auc_std']:.4f}")
        print(f"  Deletion AUC: {enhanced_results['deletion_auc_mean']:.4f} ± {enhanced_results['deletion_auc_std']:.4f}")
        print(f"  ROAD Score: {enhanced_results['road_mean']:.4f} ± {enhanced_results['road_std']:.4f}")
        
        # Evaluate standard methods
        for method in standard_methods:
            print(f"\n{'='*60}")
            print(f"Evaluating {method}")
            print(f"{'='*60}")
            
            method_results = self.evaluate_method(
                cam_method_name=method,
                max_images=max_images
            )
            
            # Extract summary statistics
            summary = {
                'Method': method,
                'Layers': '1 layer',  # Standard methods typically use 1 layer
                'Insertion AUC': f"{method_results['insertion_auc_mean']:.4f} ± {method_results['insertion_auc_std']:.4f}",
                'Deletion AUC': f"{method_results['deletion_auc_mean']:.4f} ± {method_results['deletion_auc_std']:.4f}",
                'ROAD Score': f"{method_results['road_mean']:.4f} ± {method_results['road_std']:.4f}",
                'Num Images': method_results['num_images']
            }
            results.append(summary)
            
            # Print detailed results
            print(f"\nResults for {method}:")
            print(f"  Insertion AUC: {method_results['insertion_auc_mean']:.4f} ± {method_results['insertion_auc_std']:.4f}")
            print(f"  Deletion AUC: {method_results['deletion_auc_mean']:.4f} ± {method_results['deletion_auc_std']:.4f}")
            print(f"  ROAD Score: {method_results['road_mean']:.4f} ± {method_results['road_std']:.4f}")
        
        # Create results DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Save results to model-specific CSV directory
        try:
            saved_path = save_evaluation_results(
                comparison_df,
                self.model_name,
                evaluation_type="enhanced_vs_standard_comparison",
                add_timestamp=True
            )
            print(f"\n💾 Results saved to: {saved_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save results to CSV: {e}")
        
        # Save detailed comparison data to pickle file
        try:
            # Create a comprehensive data structure for the comparison
            comparison_data = {
                'comparison_df': comparison_df,
                'model_name': self.model_name,
                'layer_mode': self.layer_mode,
                'num_layers': len(self.conv_layers),
                'max_images': max_images,
                'step_size': step_size,
                'enhanced_results': enhanced_results,
                'standard_results': {method: method_results for method, method_results in 
                                   zip(standard_methods, [summary for summary in results if summary['Method'] != f'Enhanced CAM ({self.layer_mode})'])}
            }
            
            pickle_path = save_analysis_data(
                comparison_data,
                self.model_name,
                analysis_type="enhanced_vs_standard_comparison_detailed",
                add_timestamp=True
            )
            print(f"💾 Detailed comparison data saved to: {pickle_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save detailed comparison data to pickle: {e}")
        
        return comparison_df


def main():
    """Test the enhanced evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Proper AUC Evaluation")
    parser.add_argument("--model", default="resnet18", help="Model name")
    parser.add_argument("--max-images", type=int, default=2, help="Maximum images to test")
    parser.add_argument("--step-size", type=int, default=224, help="Step size for evaluation")
    parser.add_argument("--standard-methods", nargs="+", 
                       default=["GradCAM", "GradCAM++", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"], 
                       choices=["GradCAM", "GradCAM++", "EigenGradCAM", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"],
                       help="Standard CAM methods to compare")
    parser.add_argument("--layer-mode", default="last", 
                       choices=["all", "last_5", "last"],
                       help="Layer selection mode for Enhanced CAM")
    parser.add_argument("--enhanced-cam-method", default="GradCAMEnhanced",
                       choices=["GradCAMEnhanced", "GradCAMPlusPlusEnhanced", "HiResCAMEnhanced", 
                               "ScoreCAMEnhanced", "AblationCAMEnhanced"],
                       help="Enhanced CAM method to use")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Enhanced Proper AUC Evaluation Test")
    print(f"Model: {args.model}, Max Images: {args.max_images}, Step Size: {args.step_size}")
    print(f"Layer Mode: {args.layer_mode}, Enhanced CAM Method: {args.enhanced_cam_method}")
    print(f"{'='*80}")
    
    evaluator = EnhancedProperAUCEvaluator(
        model_name=args.model, 
        layer_mode=args.layer_mode,
        enhanced_cam_method=args.enhanced_cam_method
    )
    
    # Compare Enhanced CAM vs Standard methods
    comparison_df = evaluator.compare_enhanced_vs_standard(
        standard_methods=args.standard_methods,
        max_images=args.max_images,
        step_size=args.step_size
    )
    
    print(f"\n{'='*80}")
    print("FINAL COMPARISON RESULTS:")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print("• All AUC values should be in [0, 1] range")
    print("• Higher insertion AUC = better (pixels added improve confidence)")
    print("• Lower deletion AUC = better (pixels removed decrease confidence)")
    print("• Lower ROAD score = better (more robust explanations)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
