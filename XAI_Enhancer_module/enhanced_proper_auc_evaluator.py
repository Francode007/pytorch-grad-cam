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
from XAI_Enhancer_module.proper_auc_evaluation import ProperAUCEvaluator

# Local imports for Enhanced CAM
from XAI_Enhancer_module.optimized_cam_extractor import OptimizedCamExtractor
from XAI_Enhancer_module.optimized_predictor import get_optimized_predictions
from XAI_Enhancer_module.model_utils import get_validation_paths, TRAIN_DATA_PATH


class EnhancedProperAUCEvaluator(ProperAUCEvaluator):
    """
    Extended evaluator that supports both standard CAM methods and Enhanced CAM.
    Uses ProperAUCEvaluator as the base for consistent AUC calculations.
    """
    
    def __init__(self, model_name: str = "resnet18", device_preference: str = "auto"):
        super().__init__(model_name, device_preference)
        
        # Initialize Enhanced CAM components
        self.conv_layers = self._get_enhanced_cam_layers()
        self.enhanced_cam_extractor = None
        
        print(f"EnhancedProperAUCEvaluator initialized with Enhanced CAM support")
    
    def _get_enhanced_cam_layers(self) -> List[torch.nn.Module]:
        """Get conv layers for Enhanced CAM extraction."""
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
            raise ValueError(f"Could not find conv layers for model {self.model_name}")
    
    def extract_enhanced_cam(self, image_path: str, predicted_label: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Extract Enhanced CAM for a single image."""
        if self.enhanced_cam_extractor is None:
            self.enhanced_cam_extractor = OptimizedCamExtractor(
                self.model, self.model_name, self.conv_layers, 
                device_preference=str(self.device)
            )
        
        # Extract Enhanced CAM
        image_tensor, saliency_map = self.enhanced_cam_extractor.extract_saliency_map(
            image_path, predicted_label
        )
        
        # Convert saliency map to numpy if it's a tensor
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.cpu().numpy()
        
        return image_tensor, saliency_map
    
    def evaluate_enhanced_cam(self, max_images: int = 2, step_size: int = 50) -> Dict[str, any]:
        """Evaluate Enhanced CAM method with proper AUC calculations."""
        # Get image paths and predictions
        image_paths = get_validation_paths(TRAIN_DATA_PATH)[:max_images]
        predicted_labels, _, _ = get_optimized_predictions(
            self.model_name, image_paths, use_validation_set=False
        )
        
        print(f"\nEvaluating Enhanced CAM on {len(image_paths)} images...")
        print(f"Using step_size={step_size} for proper evaluation...")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        for image_path, predicted_label in tqdm(zip(image_paths, predicted_labels), 
                                              desc="Processing Enhanced CAM"):
            try:
                # Extract Enhanced CAM
                image_tensor, saliency_map = self.extract_enhanced_cam(
                    image_path, predicted_label
                )
                
                print(f"  Processing: {Path(image_path).name}")
                print(f"    Saliency range: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
                
                # Compute insertion AUC with proper step size
                _, insertion_auc = self.compute_insertion_auc(
                    image_tensor, saliency_map, predicted_label, step_size=step_size
                )
                insertion_aucs.append(insertion_auc)
                print(f"    Insertion AUC: {insertion_auc:.4f}")
                
                # Compute deletion AUC with proper step size
                _, deletion_auc = self.compute_deletion_auc(
                    image_tensor, saliency_map, predicted_label, step_size=step_size
                )
                deletion_aucs.append(deletion_auc)
                print(f"    Deletion AUC: {deletion_auc:.4f}")
                
                # Compute ROAD score
                road_score = self.evaluate_road(
                    image_tensor, saliency_map, predicted_label
                )
                road_scores.append(road_score)
                print(f"    ROAD Score: {road_score:.4f}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compile results
        results = {
            'cam_method': 'Enhanced CAM',
            'model_name': self.model_name,
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
        
        return results
    
    def compare_enhanced_vs_standard(self, standard_methods: List[str] = None, 
                                   max_images: int = 2, step_size: int = 50) -> pd.DataFrame:
        """Compare Enhanced CAM with standard CAM methods."""
        if standard_methods is None:
            standard_methods = ["GradCAM", "GradCAM++"]
        
        results = []
        
        # Evaluate Enhanced CAM
        print(f"\n{'='*60}")
        print(f"Evaluating Enhanced CAM")
        print(f"{'='*60}")
        
        enhanced_results = self.evaluate_enhanced_cam(max_images, step_size)
        
        enhanced_summary = {
            'Method': 'Enhanced CAM',
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
        
        return pd.DataFrame(results)


def main():
    """Test the enhanced evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Proper AUC Evaluation")
    parser.add_argument("--model", default="resnet18", help="Model name")
    parser.add_argument("--max-images", type=int, default=2, help="Maximum images to test")
    parser.add_argument("--step-size", type=int, default=50, help="Step size for evaluation")
    parser.add_argument("--standard-methods", nargs="+", default=["GradCAM"], 
                       help="Standard CAM methods to compare")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Enhanced Proper AUC Evaluation Test")
    print(f"Model: {args.model}, Max Images: {args.max_images}, Step Size: {args.step_size}")
    print(f"{'='*80}")
    
    evaluator = EnhancedProperAUCEvaluator(model_name=args.model)
    
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
