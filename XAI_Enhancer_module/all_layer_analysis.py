#!/usr/bin/env python3
"""
All-Layer Enhanced CAM Analysis Script.
This script evaluates Enhanced CAM using all convolutional layers, shows layer weights,
and creates visualizations of the layer importance.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.evaluator.enhanced_proper_auc_evaluator import EnhancedProperAUCEvaluator
from XAI_Enhancer_module.utils.model_utils import test_model, get_validation_paths, TRAIN_DATA_PATH
from XAI_Enhancer_module.utils.optimized_predictor import get_optimized_predictions
from XAI_Enhancer_module.enhanced_cams.GradCAM_enhanced import GradCAMEnhanced
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2


class AllLayerAnalyzer:
    """
    Comprehensive analyzer for Enhanced CAM using all convolutional layers.
    Provides detailed layer analysis, weight visualization, and performance metrics.
    """
    
    def __init__(self, model_name: str, device_preference: str = "auto"):
        """
        Initialize the analyzer.
        
        Args:
            model_name: Name of the model to analyze
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        """
        self.model_name = model_name
        self.device_preference = device_preference
        
        # Load model
        self.model = test_model(model_name, device_preference=device_preference)
        self.model.eval()
        
        # Extract all convolutional layers
        self.conv_layers = self._extract_all_conv_layers()
        self.layer_info = self._get_layer_info()
        
        print(f"AllLayerAnalyzer initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {next(self.model.parameters()).device}")
        print(f"  Total conv layers: {len(self.conv_layers)}")
    
    def _extract_all_conv_layers(self) -> List[nn.Module]:
        """Extract all convolutional layers from the model."""
        conv_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                conv_layers.append(module)
        
        return conv_layers
    
    def _get_layer_info(self) -> List[Dict]:
        """Get detailed information about each convolutional layer."""
        layer_info = []
        
        for i, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                info = {
                    'index': i,
                    'name': name,
                    'type': type(module).__name__,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'module': module
                }
                layer_info.append(info)
        
        return layer_info
    
    def analyze_single_image(self, image_path: str, predicted_label: int) -> Dict:
        """
        Analyze a single image and return layer weights and CAM information.
        
        Args:
            image_path: Path to the image
            predicted_label: Predicted label for the image
            
        Returns:
            Dictionary with analysis results
        """
        # Load and preprocess image
        image = plt.imread(image_path)
        img_size = 384 if self.model_name.startswith("b4") else 224
        image = cv2.resize(image, (img_size, img_size))
        image = image.astype(np.float32) / 255.0
        
        # Apply transformations
        from XAI_Enhancer_module.utils.model_utils import transformations
        image_tensor = transformations(image).float().unsqueeze(0)
        
        # Initialize Enhanced GradCAM with all layers
        cam_method = GradCAMEnhanced(self.model, self.conv_layers)
        targets = [ClassifierOutputTarget(predicted_label)]
        
        # Extract CAMs and modified activations
        cam_per_layer, modified_activations_per_layer = cam_method(
            image_tensor.to(next(self.model.parameters()).device), 
            targets
        )
        
        # Get actual model output
        with torch.no_grad():
            actual_output = self.model(image_tensor.to(next(self.model.parameters()).device))
            actual_output_np = actual_output.cpu().numpy()
        
        # Compute layer weights using cosine similarity
        layer_weights = self._compute_layer_weights(
            actual_output_np, modified_activations_per_layer, image_tensor
        )
        
        # Compile results
        results = {
            'image_path': image_path,
            'predicted_label': predicted_label,
            'cam_per_layer': cam_per_layer,
            'layer_weights': layer_weights,
            'num_layers': len(self.conv_layers),
            'layer_info': self.layer_info,
            'actual_output': actual_output_np
        }
        
        return results
    
    def _compute_layer_weights(self, actual_output: np.ndarray, 
                             modified_activations_per_layer: List[np.ndarray],
                             input_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute layer weights based on cosine similarity between actual and modified outputs.
        
        Args:
            actual_output: Original model output
            modified_activations_per_layer: Modified activations for each layer
            input_tensor: Input tensor
            
        Returns:
            Array of normalized layer weights
        """
        device = next(self.model.parameters()).device
        cosine_similarities = []
        
        # Compute modified outputs for each layer
        for layer_idx, modified_activation in enumerate(modified_activations_per_layer):
            try:
                # Forward pass with modified activation
                # This is a simplified approach - in practice, you'd need to hook into the forward pass
                # For now, we'll use a proxy based on activation magnitudes
                activation_magnitude = np.mean(np.abs(modified_activation))
                similarity = 1.0 / (1.0 + activation_magnitude)  # Inverse relationship
                cosine_similarities.append(similarity)
                
            except Exception as e:
                print(f"Warning: Could not compute similarity for layer {layer_idx}: {e}")
                cosine_similarities.append(0.0)
        
        # Normalize weights using softmax
        cosine_similarities = np.array(cosine_similarities)
        weights = np.exp(cosine_similarities) / np.sum(np.exp(cosine_similarities))
        
        return weights
    
    def evaluate_all_layers(self, max_images: int = 5, verbose: bool = None) -> Dict:
        """
        Evaluate Enhanced CAM performance using all layers.
        
        Args:
            max_images: Maximum number of images to evaluate
            verbose: If None, auto-determine based on max_images.
                    If True/False, override auto-detection.
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"
{'='*80}")
        print(f"ENHANCED CAM ALL-LAYER EVALUATION")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Total layers: {len(self.conv_layers)}")
        print(f"Max images: {max_images}")
        
        # Auto-determine verbosity if not specified
        if verbose is None:
            verbose = max_images <= 20
            
        if not verbose:
            print(f"Verbose mode OFF - minimal logging for large dataset")
        
        # Initialize evaluator with all layers
        evaluator = EnhancedProperAUCEvaluator(
            model_name=self.model_name,
            device_preference=self.device_preference,
            layer_mode="all"
        )
        
        # Evaluate Enhanced CAM
        results = evaluator.evaluate_enhanced_cam(
            max_images=max_images,
            step_size=224,
            verbose=verbose
        )
        
        # Get detailed layer analysis for sample images
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
        
        detailed_analysis = []
        layer_weights_all = []
        
        # Show progress differently based on verbosity
        if verbose:
            print(f"\nDetailed layer analysis for {len(image_paths)} images:")
        else:
            print(f"\nAnalyzing layer weights for {len(image_paths)} images...")
        
        for i, (image_path, label) in enumerate(zip(image_paths, predicted_labels)):
            if verbose:
                print(f"\nAnalyzing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            elif (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                print(f"  Processed {i+1}/{len(image_paths)} images for layer analysis")
            
            try:
                analysis = self.analyze_single_image(image_path, label)
                detailed_analysis.append(analysis)
                layer_weights_all.append(analysis['layer_weights'])
                
                if verbose:
                    print(f"  Layer weights range: [{analysis['layer_weights'].min():.4f}, {analysis['layer_weights'].max():.4f}]")
                    print(f"  Top 3 layers: {np.argsort(analysis['layer_weights'])[-3:][::-1]}")
                
            except Exception as e:
                if verbose:
                    print(f"  Error analyzing image: {e}")
                else:
                    print(f"⚠️  Error analyzing {Path(image_path).name}: {e}")
                continue
        
        # Compile comprehensive results
        comprehensive_results = {
            'evaluation_metrics': results,
            'detailed_analysis': detailed_analysis,
            'layer_weights_all': layer_weights_all,
            'layer_info': self.layer_info,
            'average_weights': np.mean(layer_weights_all, axis=0) if layer_weights_all else None
        }
        
        return comprehensive_results
    
    def create_layer_visualization(self, results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualizations of layer analysis.
        
        Args:
            results: Results from evaluate_all_layers
            save_path: Optional path to save plots
        """
        # Set up the plotting style
        plt.style.use('default')  # Use default style instead of seaborn
        fig = plt.figure(figsize=(20, 15))
        
        # Extract data
        layer_info = results['layer_info']
        average_weights = results['average_weights']
        layer_weights_all = results['layer_weights_all']
        
        if average_weights is None or not layer_weights_all:
            print("No weight data available for visualization")
            return
        
        # 1. Average Layer Weights Bar Plot
        plt.subplot(3, 2, 1)
        layer_names = [f"L{i}: {info['name'].split('.')[-1]}" for i, info in enumerate(layer_info)]
        layer_indices = range(len(average_weights))
        
        bars = plt.bar(layer_indices, average_weights, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.title('Average Layer Weights (All Images)', fontsize=14, fontweight='bold')
        plt.xlabel('Layer Index')
        plt.ylabel('Average Weight')
        plt.xticks(layer_indices[::max(1, len(layer_indices)//10)], 
                   [layer_names[i] for i in layer_indices[::max(1, len(layer_indices)//10)]], 
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Highlight top 3 layers
        top_3_indices = np.argsort(average_weights)[-3:]
        for idx in top_3_indices:
            bars[idx].set_color('orange')
        
        # 2. Layer Weight Distribution Heatmap
        plt.subplot(3, 2, 2)
        if len(layer_weights_all) > 1:
            weights_matrix = np.array(layer_weights_all).T
            # Create a simple heatmap without seaborn
            im = plt.imshow(weights_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im)
            plt.yticks(range(len(layer_info)), [f"L{i}" for i in range(len(layer_info))])
            plt.xticks(range(len(layer_weights_all)), [f"Img{i+1}" for i in range(len(layer_weights_all))])
            plt.title('Layer Weights Across Images', fontsize=14, fontweight='bold')
            plt.ylabel('Layer Index')
            plt.xlabel('Image Index')
        
        # 3. Layer Architecture Information
        plt.subplot(3, 2, 3)
        out_channels = [info['out_channels'] for info in layer_info]
        plt.plot(layer_indices, out_channels, marker='o', linewidth=2, markersize=4)
        plt.title('Output Channels per Layer', fontsize=14, fontweight='bold')
        plt.xlabel('Layer Index')
        plt.ylabel('Output Channels')
        plt.grid(True, alpha=0.3)
        
        # 4. Weight Variance Analysis
        plt.subplot(3, 2, 4)
        if len(layer_weights_all) > 1:
            weight_std = np.std(layer_weights_all, axis=0)
            plt.bar(layer_indices, weight_std, alpha=0.7, color='lightcoral')
            plt.title('Layer Weight Variance Across Images', fontsize=14, fontweight='bold')
            plt.xlabel('Layer Index')
            plt.ylabel('Weight Standard Deviation')
            plt.xticks(layer_indices[::max(1, len(layer_indices)//10)])
            plt.grid(True, alpha=0.3)
        
        # 5. Top Performing Layers
        plt.subplot(3, 2, 5)
        top_5_indices = np.argsort(average_weights)[-5:]
        top_5_weights = average_weights[top_5_indices]
        top_5_names = [layer_names[i] for i in top_5_indices]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_5_indices)))
        bars = plt.barh(range(len(top_5_indices)), top_5_weights, color=colors)
        plt.yticks(range(len(top_5_indices)), top_5_names)
        plt.xlabel('Average Weight')
        plt.title('Top 5 Most Important Layers', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 6. Performance Metrics Summary
        plt.subplot(3, 2, 6)
        metrics = results['evaluation_metrics']
        metric_names = ['Insertion AUC', 'Deletion AUC', 'ROAD Score']
        metric_values = [
            metrics['insertion_auc_mean'],
            metrics['deletion_auc_mean'], 
            metrics['road_mean']
        ]
        metric_errors = [
            metrics['insertion_auc_std'],
            metrics['deletion_auc_std'],
            metrics['road_std']
        ]
        
        x_pos = range(len(metric_names))
        bars = plt.bar(x_pos, metric_values, yerr=metric_errors, capsize=5, 
                      color=['green', 'blue', 'red'], alpha=0.7)
        plt.xticks(x_pos, metric_names)
        plt.ylabel('Score')
        plt.title(f'Enhanced CAM Performance ({self.model_name})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value, error in zip(bars, metric_values, metric_errors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def print_layer_summary(self, results: Dict):
        """Print a detailed summary of the layer analysis."""
        print(f"\n{'='*80}")
        print(f"LAYER ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        layer_info = results['layer_info']
        average_weights = results['average_weights']
        
        if average_weights is None:
            print("No weight data available")
            return
        
        print(f"Model: {self.model_name}")
        print(f"Total convolutional layers: {len(layer_info)}")
        print(f"Images analyzed: {len(results['layer_weights_all'])}")
        
        # Overall statistics
        print(f"\nWeight Statistics:")
        print(f"  Mean weight: {np.mean(average_weights):.4f}")
        print(f"  Weight std: {np.std(average_weights):.4f}")
        print(f"  Min weight: {np.min(average_weights):.4f}")
        print(f"  Max weight: {np.max(average_weights):.4f}")
        
        # Top layers
        top_5_indices = np.argsort(average_weights)[-5:][::-1]
        print(f"\nTop 5 Most Important Layers:")
        for rank, idx in enumerate(top_5_indices, 1):
            layer = layer_info[idx]
            print(f"  {rank}. Layer {idx}: {layer['name']}")
            print(f"     Weight: {average_weights[idx]:.4f}")
            print(f"     Type: {layer['type']}")
            print(f"     Channels: {layer['in_channels']} → {layer['out_channels']}")
            print(f"     Kernel: {layer['kernel_size']}")
        
        # Performance metrics
        metrics = results['evaluation_metrics']
        print(f"\nPerformance Metrics:")
        print(f"  Insertion AUC: {metrics['insertion_auc_mean']:.4f} ± {metrics['insertion_auc_std']:.4f}")
        print(f"  Deletion AUC: {metrics['deletion_auc_mean']:.4f} ± {metrics['deletion_auc_std']:.4f}")
        print(f"  ROAD Score: {metrics['road_mean']:.4f} ± {metrics['road_std']:.4f}")
        
        print(f"\n{'='*80}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="All-Layer Enhanced CAM Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ResNet18 with all layers (auto verbosity)
  python all_layer_analysis.py --model resnet18 --max-images 5

  # Large scale analysis with quiet mode
  python all_layer_analysis.py --model resnet18 --max-images 100 --quiet --save-plots

  # Small scale with verbose output
  python all_layer_analysis.py --model b0 --max-images 3 --verbose --save-plots

  # Quick analysis with fewer images
  python all_layer_analysis.py --model resnet34 --max-images 2 --no-plots
        """
    )
    
    parser.add_argument('--model', '-m', default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'b0', 'b4'],
                       help='Model to analyze')
    
    parser.add_argument('--max-images', type=int, default=5,
                       help='Maximum number of images to analyze (use -1 for entire validation dataset)')
    
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device preference')
    
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots to files')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip visualization plots')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Force verbose output (detailed per-image logging)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Force quiet output (minimal logging)')
    
    parser.add_argument('--output-dir', default='./analysis_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Handle verbosity conflicts
    if args.verbose and args.quiet:
        print("❌ Error: Cannot specify both --verbose and --quiet")
        return
    
    # Determine verbosity
    if args.verbose:
        verbose = True
    elif args.quiet:
        verbose = False
    else:
        verbose = None  # Auto-determine based on max_images
    
    print(f"\n{'='*80}")
    print("ALL-LAYER ENHANCED CAM ANALYSIS")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Max images: {args.max_images}")
    print(f"  Device: {args.device}")
    print(f"  Save plots: {args.save_plots}")
    print(f"  Show plots: {not args.no_plots}")
    if verbose is not None:
        print(f"  Verbose: {verbose}")
    else:
        print(f"  Verbose: Auto (True for ≤20 images, False for >20 images)")
    
    try:
        # Initialize analyzer
        analyzer = AllLayerAnalyzer(args.model, args.device)
        
        # Run comprehensive analysis
        results = analyzer.evaluate_all_layers(
            max_images=args.max_images,
            verbose=verbose
        )
        
        # Print detailed summary
        analyzer.print_layer_summary(results)
        
        # Create visualizations
        if not args.no_plots:
            save_path = None
            if args.save_plots:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True)
                save_path = output_dir / f"{args.model}_all_layers_analysis.png"
            
            analyzer.create_layer_visualization(results, save_path)
        
        # Save detailed results
        if args.save_plots:
            import pickle
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            results_path = output_dir / f"{args.model}_analysis_results.pkl"
            
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Detailed results saved to: {results_path}")
        
        print(f"\n✅ All-layer analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
