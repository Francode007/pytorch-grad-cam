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
                #  model, 
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
        super().__init__(model_name, device_preference="auto")
        
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
