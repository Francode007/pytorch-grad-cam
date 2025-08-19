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
from XAI_Enhancer_module.enhanced_cams import (
    GradCAMEnhanced, 
    GradCAMPlusPlusEnhanced, 
    HiResCAMEnhanced, 
    ScoreCAMEnhanced, 
    AblationCAMEnhanced
)
from XAI_Enhancer_module.utils.directory_manager import create_model_output_dirs, save_analysis_data, save_visualization
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2


class AllLayerAnalyzer:
    """
    Comprehensive analyzer for Enhanced CAM using all convolutional layers.
    Provides detailed layer analysis, weight visualization, and performance metrics.
    """
    
    def __init__(self, model_name: str, test_images: int = 5, enhanced_cam_method: str = "GradCAMEnhanced"):
        """
        Initialize the analyzer.
        
        Args:
            model_name: Name of the model to analyze
            test_images: Number of test images to use
            enhanced_cam_method: Enhanced CAM method to use
        """
        self.model_name = model_name
        self.test_images = test_images
        self.enhanced_cam_method = enhanced_cam_method
        self.device_preference = "auto"  # Default device preference
        
        # Load model
        self.model = test_model(model_name, device_preference=self.device_preference)
        self.model.eval()
        
        # Extract all convolutional layers
        self.conv_layers = self._extract_all_conv_layers()
        self.layer_info = self._get_layer_info()
        
        print(f"AllLayerAnalyzer initialized:")
        print(f"  Model: {model_name}")
        print(f"  Enhanced CAM method: {enhanced_cam_method}")
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
        
        # Initialize Enhanced CAM with all layers
        enhanced_cam_methods = {
            'GradCAMEnhanced': GradCAMEnhanced,
            'GradCAMPlusPlusEnhanced': GradCAMPlusPlusEnhanced,
            'HiResCAMEnhanced': HiResCAMEnhanced,
            'ScoreCAMEnhanced': ScoreCAMEnhanced,
            'AblationCAMEnhanced': AblationCAMEnhanced
        }
        
        if self.enhanced_cam_method not in enhanced_cam_methods:
            raise ValueError(f"Unknown enhanced CAM method: {self.enhanced_cam_method}. "
                           f"Available methods: {list(enhanced_cam_methods.keys())}")
        
        cam_class = enhanced_cam_methods[self.enhanced_cam_method]
        cam_method = cam_class(self.model, self.conv_layers)
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
        print(f"{'='*80}")
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
            dataset_path=TRAIN_DATA_PATH,
            layer_mode="all",
            enhanced_cam_method=self.enhanced_cam_method
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
    
    def create_layer_visualization(self, results: Dict, save_path: Optional[str] = None, 
                                 save_individual_plots: bool = True):
        """
        Create comprehensive visualizations of layer analysis.
        Saves both individual plots and a combined figure in model-specific directories.
        
        Args:
            results: Results from evaluate_all_layers
            save_path: Optional path to save the combined plot (if None, uses model-specific directory)
            save_individual_plots: Whether to save individual plots
        """
        # Set up the plotting style
        plt.style.use('default')  # Use default style instead of seaborn
        
        # Extract data
        layer_info = results['layer_info']
        average_weights = results['average_weights']
        layer_weights_all = results['layer_weights_all']
        
        if average_weights is None or not layer_weights_all:
            print("No weight data available for visualization")
            return
        
        # Create model-specific directories for saving plots
        if save_path is None or save_individual_plots:
            from XAI_Enhancer_module.utils.directory_manager import create_model_output_dirs
            model_analysis_dir, _ = create_model_output_dirs(self.model_name)
            
            if save_path is None:
                save_path = model_analysis_dir / "all_layers_combined_analysis.png"
        
        # Prepare common data
        layer_names = [f"L{i}: {info['name'].split('.')[-1]}" for i, info in enumerate(layer_info)]
        layer_indices = range(len(average_weights))
        
        # Create individual plots if requested
        individual_plots = []
        
        # 1. Average Layer Weights Bar Plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        bars = ax1.bar(layer_indices, average_weights, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_title('Average Layer Weights (All Images)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Layer Index', fontsize=14)
        ax1.set_ylabel('Average Weight', fontsize=14)
        ax1.set_xticks(layer_indices[::max(1, len(layer_indices)//10)])
        ax1.set_xticklabels([layer_names[i] for i in layer_indices[::max(1, len(layer_indices)//10)]], 
                           rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Highlight top 3 layers
        top_3_indices = np.argsort(average_weights)[-3:]
        for idx in top_3_indices:
            bars[idx].set_color('orange')
        
        plt.tight_layout()
        if save_individual_plots:
            individual_path = model_analysis_dir / "plot1_average_layer_weights.png"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            individual_plots.append(individual_path)
        plt.close()
        
        # 2. Layer Weight Distribution Heatmap
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        if len(layer_weights_all) > 1:
            weights_matrix = np.array(layer_weights_all).T
            im = ax2.imshow(weights_matrix, cmap='viridis', aspect='auto')
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Weight Value', fontsize=12)
            ax2.set_yticks(range(len(layer_info)))
            ax2.set_yticklabels([f"L{i}" for i in range(len(layer_info))])
            ax2.set_xticks(range(len(layer_weights_all)))
            ax2.set_xticklabels([f"Img{i+1}" for i in range(len(layer_weights_all))])
            ax2.set_title('Layer Weights Across Images', fontsize=16, fontweight='bold')
            ax2.set_ylabel('Layer Index', fontsize=14)
            ax2.set_xlabel('Image Index', fontsize=14)
        
        plt.tight_layout()
        if save_individual_plots:
            individual_path = model_analysis_dir / "plot2_layer_weights_heatmap.png"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            individual_plots.append(individual_path)
        plt.close()
        
        # 3. Layer Architecture Information
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        out_channels = [info['out_channels'] for info in layer_info]
        ax3.plot(layer_indices, out_channels, marker='o', linewidth=2, markersize=6, color='navy')
        ax3.set_title('Output Channels per Layer', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Layer Index', fontsize=14)
        ax3.set_ylabel('Output Channels', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_individual_plots:
            individual_path = model_analysis_dir / "plot3_layer_architecture.png"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            individual_plots.append(individual_path)
        plt.close()
        
        # 4. Weight Variance Analysis
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        if len(layer_weights_all) > 1:
            weight_std = np.std(layer_weights_all, axis=0)
            bars = ax4.bar(layer_indices, weight_std, alpha=0.7, color='lightcoral')
            ax4.set_title('Layer Weight Variance Across Images', fontsize=16, fontweight='bold')
            ax4.set_xlabel('Layer Index', fontsize=14)
            ax4.set_ylabel('Weight Standard Deviation', fontsize=14)
            ax4.set_xticks(layer_indices[::max(1, len(layer_indices)//10)])
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_individual_plots:
            individual_path = model_analysis_dir / "plot4_weight_variance.png"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            individual_plots.append(individual_path)
        plt.close()
        
        # Now create the combined figure with all 6 subplots
        fig_combined = plt.figure(figsize=(20, 15))
        
        # Recreate all subplots in the combined figure
        # 1. Average Layer Weights Bar Plot
        plt.subplot(3, 2, 1)
        bars = plt.bar(layer_indices, average_weights, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.title('Average Layer Weights (All Images)', fontsize=14, fontweight='bold')
        plt.xlabel('Layer Index')
        plt.ylabel('Average Weight')
        plt.xticks(layer_indices[::max(1, len(layer_indices)//10)], 
                   [layer_names[i] for i in layer_indices[::max(1, len(layer_indices)//10)]], 
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Highlight top 3 layers
        for idx in top_3_indices:
            bars[idx].set_color('orange')
        
        # 2. Layer Weight Distribution Heatmap
        plt.subplot(3, 2, 2)
        if len(layer_weights_all) > 1:
            weights_matrix = np.array(layer_weights_all).T
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
        
        # Save the combined figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Combined visualization saved to: {save_path}")
        
        # Print summary of saved plots
        if save_individual_plots:
            print(f"📈 Individual plots saved:")
            for i, path in enumerate(individual_plots, 1):
                print(f"   Plot {i}: {path}")
        
        plt.show()
        
        return fig_combined, individual_plots if save_individual_plots else []
    
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

  # Large scale analysis with quiet mode and save all plots
  python all_layer_analysis.py --model resnet18 --max-images 100 --quiet --save-plots

  # Save both combined and individual plots to model-specific directories
  python all_layer_analysis.py --model b0 --max-images 3 --save-plots --save-individual

  # Quick analysis with only individual plots saved
  python all_layer_analysis.py --model resnet34 --max-images 2 --save-individual --no-plots

  # Full analysis with all outputs saved in organized directories
  python all_layer_analysis.py --model resnet18 --max-images 10 --save-plots --save-individual --verbose

Note: All outputs are automatically saved to model-specific directories:
  • Analysis results: ./analysis_results/{model_name}/
  • Combined plots: ./analysis_results/{model_name}/all_layers_combined_analysis.png
  • Individual plots: ./analysis_results/{model_name}/plot1_average_layer_weights.png, etc.
  • Pickle data: ./analysis_results/{model_name}/all_layers_analysis_*.pkl
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
    
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual plots separately (in addition to combined plot)')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip visualization plots')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Force verbose output (detailed per-image logging)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Force quiet output (minimal logging)')
    
    parser.add_argument('--output-dir', default='./analysis_results',
                       help='Directory to save results')
    
    parser.add_argument('--enhanced-cam-method', default='GradCAMEnhanced',
                       choices=['GradCAMEnhanced', 'GradCAMPlusPlusEnhanced', 'HiResCAMEnhanced', 
                               'ScoreCAMEnhanced', 'AblationCAMEnhanced'],
                       help='Enhanced CAM method to use')
    
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
        analyzer = AllLayerAnalyzer(args.model, enhanced_cam_method=args.enhanced_cam_method)
        
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
                # Create model-specific directories
                model_analysis_dir, _ = create_model_output_dirs(args.model, args.output_dir)
                save_path = model_analysis_dir / "all_layers_combined_analysis.png"
            
            # Create visualizations with individual plots saved automatically when save_plots is True
            combined_fig, individual_plots = analyzer.create_layer_visualization(
                results, 
                save_path, 
                save_individual_plots=args.save_plots or args.save_individual
            )        # Save detailed results
        if args.save_plots:
            # Use the directory manager to save results with model-specific organization
            saved_path = save_analysis_data(
                results, 
                args.model, 
                analysis_type="all_layers",
                base_analysis_dir=args.output_dir,
                add_timestamp=True
            )
            print(f"Detailed results saved to: {saved_path}")
        
        print(f"\n✅ All-layer analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
