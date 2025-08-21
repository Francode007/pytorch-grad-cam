#!/usr/bin/env python3
"""
Enhanced CAM evaluation with proper AUC calculations.
This script shows correct insertion/deletion AUC values in the [0,1] range.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Standard CAM imports
sys.path.append(str(project_root))  # Add pytorch-grad-cam root
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, EigenGradCAM, EigenCAM, 
    HiResCAM, LayerCAM, ScoreCAM
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.road import ROADCombined

# Local imports
from XAI_Enhancer_module.utils.model_utils import test_model, get_device, get_validation_paths, TRAIN_DATA_PATH, MEAN, STD
from XAI_Enhancer_module.utils.optimized_predictor import get_optimized_predictions


class ProperAUCEvaluator:
    """
    Evaluator that calculates proper AUC scores in [0,1] range for insertion/deletion metrics.
    """
    
    def __init__(self, model_name: str = "resnet18", device_preference: str = "auto"):
        self.model_name = model_name
        self.device = get_device(device_preference)
        
        # Load model
        self.model = test_model(model_name, device_preference=device_preference)
        self.model.eval()
        
        # Get image size
        self.img_size = 224
        if model_name in ('b4',):
            self.img_size = 384
        
        print(f"ProperAUCEvaluator initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Image size: {self.img_size}")
    
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
        """
        Compute insertion AUC by progressively adding pixels in order of importance.
        Returns scores array and AUC value in [0,1] range.
        """
        # Ensure tensor has correct shape
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        image_tensor = image_tensor.to(self.device)
        
        # Debug tensor shapes
        # print(f"    Image tensor shape: {image_tensor.shape}")
        # print(f"    Saliency map shape: {saliency_map.shape}")
        
        # Create blurred baseline
        blurred_image = self._create_blurred_baseline(image_tensor)
        
        # Get pixel importance order (most important first)
        flat_saliency = saliency_map.flatten()
        pixel_order = np.argsort(flat_saliency)[::-1]  # Descending order
        
        total_pixels = self.img_size * self.img_size
        n_steps = (total_pixels + step_size - 1) // step_size
        
        scores = np.zeros(n_steps + 1)
        current_image = blurred_image.clone()
        
        # Initial score (blurred image)
        with torch.no_grad():
            output = self.model(current_image)
            prob = F.softmax(output, dim=1)[0, target_class].item()
        scores[0] = prob
        
        # Progressive insertion
        for step in range(n_steps):
            start_idx = step * step_size
            end_idx = min((step + 1) * step_size, total_pixels)
            
            # Get pixel indices for this step
            pixel_indices = pixel_order[start_idx:end_idx]
            
            # Convert flat indices to 2D coordinates
            y_coords = pixel_indices // self.img_size
            x_coords = pixel_indices % self.img_size
            
            # Insert original pixels (fix tensor indexing)
            for y, x in zip(y_coords, x_coords):
                if y < current_image.shape[2] and x < current_image.shape[3]:
                    current_image[0, :, y, x] = image_tensor[0, :, y, x]
            
            # Get new score
            with torch.no_grad():
                output = self.model(current_image)
                prob = F.softmax(output, dim=1)[0, target_class].item()
            scores[step + 1] = prob
        
        # Calculate AUC using trapezoidal rule, normalized to [0,1]
        auc = np.trapz(scores) / len(scores)
        
        return scores, auc
    
    def compute_deletion_auc(self, image_tensor: torch.Tensor, 
                           saliency_map: np.ndarray, 
                           target_class: int, 
                           step_size: int = 224) -> Tuple[np.ndarray, float]:
        """
        Compute deletion AUC by progressively removing pixels in order of importance.
        Returns scores array and AUC value in [0,1] range.
        """
        # Ensure tensor has correct shape
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
        image_tensor = image_tensor.to(self.device)
        
        # Create baseline (black pixels)
        baseline_image = torch.zeros_like(image_tensor)
        
        # Get pixel importance order (most important first)
        flat_saliency = saliency_map.flatten()
        pixel_order = np.argsort(flat_saliency)[::-1]  # Descending order
        
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
        for step in range(n_steps):
            start_idx = step * step_size
            end_idx = min((step + 1) * step_size, total_pixels)
            
            # Get pixel indices for this step
            pixel_indices = pixel_order[start_idx:end_idx]
            
            # Convert flat indices to 2D coordinates
            y_coords = pixel_indices // self.img_size
            x_coords = pixel_indices % self.img_size
            
            # Delete pixels (replace with baseline) - fix tensor indexing
            for y, x in zip(y_coords, x_coords):
                if y < current_image.shape[2] and x < current_image.shape[3]:
                    current_image[0, :, y, x] = baseline_image[0, :, y, x]
            
            # Get new score
            with torch.no_grad():
                output = self.model(current_image)
                prob = F.softmax(output, dim=1)[0, target_class].item()
            scores[step + 1] = prob
        
        # Calculate AUC using trapezoidal rule, normalized to [0,1]
        auc = np.trapz(scores) / len(scores)
        
        return scores, auc
    
    def _create_blurred_baseline(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Create blurred baseline image."""
        import torchvision.transforms.functional as TF
        from PIL import ImageFilter
        
        # Convert to PIL, blur, and back to tensor
        pil_image = TF.to_pil_image(image_tensor.squeeze(0).cpu())
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=5))
        blurred_tensor = TF.to_tensor(blurred)
        
        # Normalize with same stats as original
        blurred_tensor = TF.normalize(blurred_tensor, mean=MEAN, std=STD)
        
        return blurred_tensor.unsqueeze(0).to(self.device)
    
    def evaluate_road(self, image_tensor: torch.Tensor, 
                     saliency_map: np.ndarray, 
                     target_class: int) -> float:
        """Evaluate using ROAD metric."""
        try:
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
            
            # print(f"    ROAD input shapes - Image: {image_tensor.shape}, Saliency: {saliency_tensor.shape}")
            
            targets = [ClassifierOutputSoftmaxTarget(target_class)]
            
            scores = road_metric(image_tensor, saliency_tensor, targets, self.model)
            return scores[0]
        except Exception as e:
            print(f"ROAD evaluation failed: {e}")
            return 0.0
    
    def evaluate_method(self, cam_method_name: str, max_images: int = -1, sample_paths: list = None):
        """Evaluate a CAM method with proper AUC calculations."""
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
    
    def compare_methods(self, methods: List[str] = None, 
                       max_images: int = 2) -> pd.DataFrame:
        """Compare multiple CAM methods."""
        if methods is None:
            methods = ["GradCAM", "GradCAM++", "EigenGradCAM"]
        
        results = []
        
        for method in methods:
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
            
            # Show individual scores for verification
            print(f"  Individual insertion AUCs: {method_results['insertion_aucs']}")
            print(f"  Individual deletion AUCs: {method_results['deletion_aucs']}")
            print(f"  Individual ROAD scores: {method_results['road_scores']}")
        
        return pd.DataFrame(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Proper AUC Evaluation")
    parser.add_argument("--model", default="resnet18", help="Model name")
    parser.add_argument("--max-images", type=int, default=2, help="Maximum images to test")
    parser.add_argument("--methods", nargs="+", 
                       default=["GradCAM", "GradCAM++", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"],
                       choices=["GradCAM", "GradCAM++", "EigenGradCAM", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"],
                       help="CAM methods to test")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Proper AUC Evaluation Test")
    print(f"Model: {args.model}, Max Images: {args.max_images}")
    print(f"{'='*80}")
    
    evaluator = ProperAUCEvaluator(model_name=args.model)
    
    # Compare methods
    comparison_df = evaluator.compare_methods(
        methods=args.methods, 
        max_images=args.max_images
    )
    
    print(f"\n{'='*80}")
    print("FINAL COMPARISON RESULTS:")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("EXPECTED RANGES:")
    print("• Insertion AUC: Should be in [0, 1] range")
    print("• Deletion AUC: Should be in [0, 1] range")
    print("• ROAD Score: Typically small positive values")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
