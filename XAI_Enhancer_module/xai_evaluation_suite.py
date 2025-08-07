"""
Comprehensive evaluation script for the novel XAI method.
This script evaluates explainability methods using optimized architectures.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import optimized modules
from XAI_Enhancer_module.optimized_cam_extractor import OptimizedCamExtractor, create_optimized_dataloader
from XAI_Enhancer_module.optimized_predictor import get_optimized_predictions, PredictionManager
from XAI_Enhancer_module.metrics.evaluation import get_metrics, CausalMetric
from XAI_Enhancer_module.model_utils import test_model, get_val_dataloader, get_device
from XAI_Enhancer_module.GradCAM_enhanced import GradCAMEnhanced


class XAIEvaluationSuite:
    """
    Comprehensive evaluation suite for XAI methods.
    """
    
    def __init__(self, 
                 model_name: str,
                 conv_layers: List[nn.Module] = None,
                 output_dir: str = "./evaluation_results",
                 device_preference: str = "auto"):
        """
        Initialize the XAI evaluation suite.
        
        Args:
            model_name: Name of the model to evaluate
            conv_layers: Optional list of specific convolutional layers
            output_dir: Directory to save results
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        """
        self.model_name = model_name
        self.device_preference = device_preference
        self.device = get_device(device_preference)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Initializing XAI Evaluation Suite")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Output: {output_dir}")
        
        # Load model
        self.model = test_model(model_name, device_preference=device_preference)
        
        # Set convolutional layers
        if conv_layers is None:
            self.conv_layers = self._get_default_conv_layers()
            print(f"   Using default conv layers: {len(self.conv_layers)} layers")
        else:
            self.conv_layers = conv_layers
            print(f"   Using custom conv layers: {len(self.conv_layers)} layers")
        
        # Initialize components
        self.cam_extractor = OptimizedCamExtractor(
            self.model, model_name, self.conv_layers, device_preference=device_preference
        )
        self.prediction_manager = PredictionManager(device_preference=device_preference)
        
        # Initialize metrics
        img_size = 224 if not model_name.startswith("b4") else 384
        print(f"   Calling get_metrics with model={type(self.model)}, model_name={model_name}, img_size={img_size}")
        self.insertion, self.deletion, self.road = get_metrics(
            self.model, model_name, img_size
        )
        
        # Results storage
        self.results = {}
        
    def _get_default_conv_layers(self) -> List[nn.Module]:
        """
        Get default convolutional layers based on model architecture.
        
        Returns:
            List of convolutional layers
        """
        if self.model_name.startswith('resnet'):
            return [
                self.model.layer1[-1],
                self.model.layer2[-1], 
                self.model.layer3[-1],
                self.model.layer4[-1]
            ]
        elif self.model_name.startswith('b'):  # EfficientNet
            # Get the last few blocks
            conv_layers = []
            for i, block in enumerate(self.model.features):
                if hasattr(block, 'conv_pw') or hasattr(block, 'conv_dw'):
                    conv_layers.append(block)
            return conv_layers[-4:]  # Last 4 blocks
        elif self.model_name == 'densenet':
            return [
                self.model.features.denseblock1,
                self.model.features.denseblock2,
                self.model.features.denseblock3,
                self.model.features.denseblock4
            ]
        elif self.model_name == 'xception':
            # For Xception, use the last few blocks
            return [
                self.model.block11,
                self.model.block12,
                self.model.conv3,
                self.model.conv4
            ]
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def get_all_conv_layers(self) -> Dict[str, List[nn.Module]]:
        """
        Extract all convolutional layers from the model for comprehensive analysis.
        
        Returns:
            Dictionary with categorized convolutional layers:
            - 'all_conv_layers': All Conv2d layers in the model
            - 'by_stage': Layers grouped by model stages (if applicable)
            - 'layer_names': Names/paths of the layers
        """
        all_conv_layers = []
        layer_names = []
        by_stage = {}
        
        def extract_conv_layers(module, prefix=""):
            """Recursively extract Conv2d layers from the module."""
            for name, child in module.named_children():
                current_path = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Conv2d):
                    all_conv_layers.append(child)
                    layer_names.append(current_path)
                else:
                    extract_conv_layers(child, current_path)
        
        # Extract all conv layers
        extract_conv_layers(self.model)
        
        # Group by stages based on model architecture
        if self.model_name.startswith('resnet'):
            by_stage = self._group_resnet_layers()
        elif self.model_name.startswith('b'):  # EfficientNet
            by_stage = self._group_efficientnet_layers()
        elif self.model_name == 'densenet':
            by_stage = self._group_densenet_layers()
        elif self.model_name == 'xception':
            by_stage = self._group_xception_layers()
        else:
            # Generic grouping for unsupported models
            by_stage = {'all_layers': all_conv_layers}
        
        return {
            'all_conv_layers': all_conv_layers,
            'layer_names': layer_names,
            'by_stage': by_stage,
            'total_count': len(all_conv_layers)
        }
    
    def _group_resnet_layers(self) -> Dict[str, List[nn.Module]]:
        """Group ResNet layers by stages."""
        stages = {}
        
        # Initial conv layer
        if hasattr(self.model, 'conv1'):
            stages['initial_conv'] = [self.model.conv1]
        
        # Main layers
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(self.model, layer_name):
                layer = getattr(self.model, layer_name)
                conv_layers = []
                
                for block in layer:
                    for name, module in block.named_modules():
                        if isinstance(module, nn.Conv2d):
                            conv_layers.append(module)
                
                stages[layer_name] = conv_layers
                
                # Also add the last block of each layer (commonly used for CAM)
                if layer:
                    stages[f'{layer_name}_last_block'] = [layer[-1]]
        
        return stages
    
    def _group_efficientnet_layers(self) -> Dict[str, List[nn.Module]]:
        """Group EfficientNet layers by stages."""
        stages = {}
        
        if hasattr(self.model, 'features'):
            # Initial conv
            if len(self.model.features) > 0:
                first_block = self.model.features[0]
                if hasattr(first_block, 'conv'):
                    stages['initial_conv'] = [first_block.conv]
                elif isinstance(first_block, nn.Conv2d):
                    stages['initial_conv'] = [first_block]
            
            # Group blocks by stages
            stage_blocks = []
            current_stage = []
            
            for i, block in enumerate(self.model.features[1:], 1):
                # Add all conv layers from this block
                block_convs = []
                for name, module in block.named_modules():
                    if isinstance(module, nn.Conv2d):
                        block_convs.append(module)
                
                if block_convs:
                    current_stage.extend(block_convs)
                    
                    # Create stages every few blocks or at the end
                    if i % 3 == 0 or i == len(self.model.features) - 1:
                        stages[f'stage_{len(stage_blocks) + 1}'] = current_stage.copy()
                        stage_blocks.append(current_stage.copy())
                        current_stage = []
        
        return stages
    
    def _group_densenet_layers(self) -> Dict[str, List[nn.Module]]:
        """Group DenseNet layers by dense blocks."""
        stages = {}
        
        if hasattr(self.model, 'features'):
            # Initial conv
            if hasattr(self.model.features, 'conv0'):
                stages['initial_conv'] = [self.model.features.conv0]
            
            # Dense blocks
            for name, module in self.model.features.named_children():
                if 'denseblock' in name:
                    block_convs = []
                    for subname, submodule in module.named_modules():
                        if isinstance(submodule, nn.Conv2d):
                            block_convs.append(submodule)
                    
                    if block_convs:
                        stages[name] = block_convs
                        # Also add the block itself for CAM
                        stages[f'{name}_module'] = [module]
        
        return stages
    
    def _group_xception_layers(self) -> Dict[str, List[nn.Module]]:
        """Group Xception layers by blocks."""
        stages = {}
        
        # Entry flow
        entry_convs = []
        for name in ['conv1', 'conv2']:
            if hasattr(self.model, name):
                entry_convs.append(getattr(self.model, name))
        if entry_convs:
            stages['entry_flow'] = entry_convs
        
        # Middle flow blocks
        middle_convs = []
        for i in range(1, 17):  # Xception typically has 16 middle blocks
            block_name = f'block{i}'
            if hasattr(self.model, block_name):
                block = getattr(self.model, block_name)
                block_convs = []
                for name, module in block.named_modules():
                    if isinstance(module, nn.Conv2d):
                        block_convs.append(module)
                middle_convs.extend(block_convs)
                stages[block_name] = block_convs
        
        if middle_convs:
            stages['middle_flow'] = middle_convs
        
        # Exit flow
        exit_convs = []
        for name in ['conv3', 'conv4']:
            if hasattr(self.model, name):
                exit_convs.append(getattr(self.model, name))
        if exit_convs:
            stages['exit_flow'] = exit_convs
        
        return stages
    
    def print_conv_layer_summary(self):
        """Print a comprehensive summary of all convolutional layers."""
        conv_info = self.get_all_conv_layers()
        
        print(f"\n{'='*60}")
        print(f"Convolutional Layer Summary for {self.model_name}")
        print(f"{'='*60}")
        print(f"Total Conv2d layers: {conv_info['total_count']}")
        
        print(f"\n{'-'*40}")
        print("Layers by Stage:")
        print(f"{'-'*40}")
        
        for stage_name, layers in conv_info['by_stage'].items():
            print(f"\n{stage_name}: {len(layers)} layers")
            for i, layer in enumerate(layers):
                if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                    print(f"  {i+1}. {layer.in_channels} -> {layer.out_channels} "
                          f"(kernel: {layer.kernel_size}, stride: {layer.stride})")
                else:
                    print(f"  {i+1}. {type(layer).__name__}")
        
        print(f"\n{'-'*40}")
        print("Recommended layers for CAM (last layers of each stage):")
        print(f"{'-'*40}")
        
        # Get recommended layers for CAM
        recommended = self._get_default_conv_layers()
        for i, layer in enumerate(recommended):
            stage_found = "Unknown"
            for stage_name, stage_layers in conv_info['by_stage'].items():
                if layer in stage_layers:
                    stage_found = stage_name
                    break
            
            if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                print(f"  {i+1}. {stage_found}: {layer.in_channels} -> {layer.out_channels}")
            else:
                print(f"  {i+1}. {stage_found}: {type(layer).__name__}")
    
    def get_layer_combinations_for_experimentation(self) -> List[List[nn.Module]]:
        """
        Generate different layer combinations for experimentation.
        
        Returns:
            List of different layer combinations to test
        """
        conv_info = self.get_all_conv_layers()
        combinations = []
        
        # Single layer combinations (last layer of each stage)
        default_layers = self._get_default_conv_layers()
        for i, layer in enumerate(default_layers):
            combinations.append([layer])
        
        # Progressive combinations (cumulative)
        for i in range(1, len(default_layers) + 1):
            combinations.append(default_layers[:i])
        
        # All combinations of 2 layers
        for i in range(len(default_layers)):
            for j in range(i + 1, len(default_layers)):
                combinations.append([default_layers[i], default_layers[j]])
        
        # Stage-based combinations
        by_stage = conv_info['by_stage']
        for stage_name, stage_layers in by_stage.items():
            if len(stage_layers) > 0 and 'last_block' in stage_name:
                combinations.append(stage_layers)
        
        # Remove duplicates
        unique_combinations = []
        for combo in combinations:
            if combo not in unique_combinations:
                unique_combinations.append(combo)
        
        return unique_combinations
    
    def experiment_all_individual_conv_layers(self, 
                                             image_paths: List[str] = None,
                                             max_layers: int = 20,
                                             batch_size: int = 2) -> pd.DataFrame:
        """
        Experiment with each individual convolutional layer to find the best performing single layer.
        
        Args:
            image_paths: Optional list of image paths for evaluation
            max_layers: Maximum number of individual layers to test
            batch_size: Batch size for evaluation (smaller for individual layer testing)
            
        Returns:
            DataFrame with results for each individual layer
        """
        print(f"\n{'='*60}")
        print(f"Individual Convolutional Layer Experimentation for {self.model_name}")
        print(f"{'='*60}")
        
        # Get all convolutional layers
        conv_info = self.get_all_conv_layers()
        all_layers = conv_info['all_conv_layers']
        layer_names = conv_info['layer_names']
        
        print(f"Found {len(all_layers)} convolutional layers")
        
        # Limit the number of layers to test for performance
        if len(all_layers) > max_layers:
            print(f"Testing only the last {max_layers} layers for performance...")
            test_layers = all_layers[-max_layers:]
            test_names = layer_names[-max_layers:]
        else:
            test_layers = all_layers
            test_names = layer_names
        
        results = []
        
        for i, (layer, layer_name) in enumerate(zip(test_layers, test_names)):
            print(f"\n{'-'*50}")
            print(f"Testing layer {i+1}/{len(test_layers)}: {layer_name}")
            
            # Get layer information
            if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                layer_info = f"{layer.in_channels}->{layer.out_channels}"
                print(f"Channels: {layer.in_channels} -> {layer.out_channels}")
                print(f"Kernel: {layer.kernel_size}, Stride: {layer.stride}")
            else:
                layer_info = f"{type(layer).__name__}"
                print(f"Type: {type(layer).__name__}")
            
            print(f"{'-'*50}")
            
            # Create evaluator with single layer
            temp_evaluator = XAIEvaluationSuite(
                model_name=self.model_name,
                conv_layers=[layer],  # Single layer
                output_dir=self.output_dir / f"single_layer_{i+1}"
            )
            
            try:
                # Run evaluation with smaller batch size
                eval_results = temp_evaluator.run_full_evaluation(
                    image_paths=image_paths,
                    batch_size=batch_size,
                    save_results=False  # Don't save individual results
                )
                
                results.append({
                    'layer_id': i + 1,
                    'layer_name': layer_name,
                    'layer_info': layer_info,
                    'insertion_auc': eval_results['insertion_auc'],
                    'deletion_auc': eval_results['deletion_auc'],
                    'road_mean': eval_results['road_mean'],
                    'road_std': eval_results['road_std'],
                    'num_images': eval_results['num_images'],
                    'status': 'success'
                })
                
                print(f"✅ Success - Insertion AUC: {eval_results['insertion_auc']:.4f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results.append({
                    'layer_id': i + 1,
                    'layer_name': layer_name,
                    'layer_info': layer_info,
                    'insertion_auc': np.nan,
                    'deletion_auc': np.nan,
                    'road_mean': np.nan,
                    'road_std': np.nan,
                    'num_images': 0,
                    'status': f'failed: {str(e)[:50]}'
                })
            
            # Clear cache after each layer
            temp_evaluator.cam_extractor.clear_cache()
            temp_evaluator.prediction_manager.clear_all_caches()
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        
        # Sort by insertion AUC (higher is better)
        results_df = results_df.sort_values('insertion_auc', ascending=False, na_position='last')
        
        # Save results
        output_file = self.output_dir / f"individual_layers_experiment_{self.model_name}.csv"
        results_df.to_csv(output_file, index=False)
        
        # Print results summary
        print(f"\n{'='*60}")
        print("Individual Layer Experimentation Results")
        print(f"{'='*60}")
        
        # Show top 10 performing layers
        top_layers = results_df.head(10)
        print("\n🏆 Top 10 Performing Individual Layers:")
        print(f"{'-'*60}")
        for idx, row in top_layers.iterrows():
            if not pd.isna(row['insertion_auc']):
                print(f"{row['layer_id']:2d}. {row['layer_name']:<25} | "
                      f"Ins AUC: {row['insertion_auc']:.4f} | "
                      f"Del AUC: {row['deletion_auc']:.4f} | "
                      f"ROAD: {row['road_mean']:.4f}")
        
        # Show statistics
        successful_results = results_df[results_df['status'] == 'success']
        if len(successful_results) > 0:
            print(f"\n📊 Statistics:")
            print(f"   Successful evaluations: {len(successful_results)}/{len(results_df)}")
            print(f"   Best Insertion AUC: {successful_results['insertion_auc'].max():.4f}")
            print(f"   Average Insertion AUC: {successful_results['insertion_auc'].mean():.4f}")
            print(f"   Best ROAD Score: {successful_results['road_mean'].min():.4f}")
            
            # Find the absolute best layer
            best_layer = successful_results.iloc[0]
            print(f"\n🥇 Best Performing Single Layer:")
            print(f"   Layer: {best_layer['layer_name']}")
            print(f"   Info: {best_layer['layer_info']}")
            print(f"   Insertion AUC: {best_layer['insertion_auc']:.4f}")
            print(f"   Deletion AUC: {best_layer['deletion_auc']:.4f}")
            print(f"   ROAD Mean: {best_layer['road_mean']:.4f}")
        
        print(f"\nResults saved to: {output_file}")
        return results_df
    
    def experiment_layer_depth_analysis(self, 
                                       image_paths: List[str] = None,
                                       step_size: int = 5) -> pd.DataFrame:
        """
        Analyze the effect of layer depth by testing layers at different depths.
        
        Args:
            image_paths: Optional list of image paths for evaluation
            step_size: Step size for selecting layers (every nth layer)
            
        Returns:
            DataFrame with results for layers at different depths
        """
        print(f"\n{'='*60}")
        print(f"Layer Depth Analysis for {self.model_name}")
        print(f"{'='*60}")
        
        # Get all convolutional layers
        conv_info = self.get_all_conv_layers()
        all_layers = conv_info['all_conv_layers']
        layer_names = conv_info['layer_names']
        
        # Select layers at different depths
        selected_indices = list(range(0, len(all_layers), step_size))
        if selected_indices[-1] != len(all_layers) - 1:
            selected_indices.append(len(all_layers) - 1)  # Always include the last layer
        
        print(f"Testing {len(selected_indices)} layers at different depths (every {step_size} layers)")
        
        results = []
        
        for i, layer_idx in enumerate(selected_indices):
            layer = all_layers[layer_idx]
            layer_name = layer_names[layer_idx]
            depth_percentage = (layer_idx / (len(all_layers) - 1)) * 100
            
            print(f"\n{'-'*50}")
            print(f"Testing depth {i+1}/{len(selected_indices)}: Layer {layer_idx+1}/{len(all_layers)} ({depth_percentage:.1f}%)")
            print(f"Layer: {layer_name}")
            print(f"{'-'*50}")
            
            # Create evaluator with single layer
            temp_evaluator = XAIEvaluationSuite(
                model_name=self.model_name,
                conv_layers=[layer],
                output_dir=self.output_dir / f"depth_analysis_{i+1}"
            )
            
            try:
                eval_results = temp_evaluator.run_full_evaluation(
                    image_paths=image_paths,
                    batch_size=2,
                    save_results=False
                )
                
                results.append({
                    'depth_rank': i + 1,
                    'layer_index': layer_idx + 1,
                    'layer_name': layer_name,
                    'depth_percentage': depth_percentage,
                    'insertion_auc': eval_results['insertion_auc'],
                    'deletion_auc': eval_results['deletion_auc'],
                    'road_mean': eval_results['road_mean'],
                    'road_std': eval_results['road_std'],
                    'num_images': eval_results['num_images']
                })
                
                print(f"✅ Insertion AUC: {eval_results['insertion_auc']:.4f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results.append({
                    'depth_rank': i + 1,
                    'layer_index': layer_idx + 1,
                    'layer_name': layer_name,
                    'depth_percentage': depth_percentage,
                    'insertion_auc': np.nan,
                    'deletion_auc': np.nan,
                    'road_mean': np.nan,
                    'road_std': np.nan,
                    'num_images': 0
                })
            
            # Clear cache
            temp_evaluator.cam_extractor.clear_cache()
            temp_evaluator.prediction_manager.clear_all_caches()
        
        # Create DataFrame
        depth_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / f"layer_depth_analysis_{self.model_name}.csv"
        depth_df.to_csv(output_file, index=False)
        
        # Analysis and visualization
        print(f"\n{'='*60}")
        print("Layer Depth Analysis Results")
        print(f"{'='*60}")
        
        valid_results = depth_df.dropna(subset=['insertion_auc'])
        if len(valid_results) > 0:
            print("\n📈 Depth vs Performance:")
            for _, row in valid_results.iterrows():
                print(f"Depth {row['depth_percentage']:5.1f}% | "
                      f"Ins AUC: {row['insertion_auc']:.4f} | "
                      f"Del AUC: {row['deletion_auc']:.4f} | "
                      f"Layer: {row['layer_name']}")
            
            # Find trends
            best_depth = valid_results.loc[valid_results['insertion_auc'].idxmax()]
            print(f"\n🎯 Best Depth:")
            print(f"   Depth: {best_depth['depth_percentage']:.1f}% through network")
            print(f"   Layer: {best_depth['layer_name']}")
            print(f"   Insertion AUC: {best_depth['insertion_auc']:.4f}")
        
        print(f"\nResults saved to: {output_file}")
        return depth_df
    
    def extract_saliency_maps(self, 
                            image_paths: List[str] = None,
                            batch_size: int = 8,
                            save_maps: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract saliency maps for evaluation.
        
        Args:
            image_paths: Optional list of image paths
            batch_size: Batch size for processing
            save_maps: Whether to save saliency maps
            
        Returns:
            Tuple of (images, saliency_maps, image_paths)
        """
        print(f"Extracting saliency maps for {self.model_name}...")
        
        # Get predictions
        if image_paths is None:
            predictions, class_names, paths = get_optimized_predictions(
                self.model_name, use_validation_set=True
            )
        else:
            predictions, class_names, paths = get_optimized_predictions(
                self.model_name, image_paths=image_paths, use_validation_set=False
            )
        
        # Create optimized dataloader
        dataloader = create_optimized_dataloader(
            paths, 
            predictions,
            self.model,
            self.model_name,
            self.conv_layers,
            batch_size=batch_size
        )
        
        # Extract saliency maps
        all_images = []
        all_saliency_maps = []
        all_paths = []
        
        for batch_images, batch_maps, batch_paths in tqdm(dataloader, desc="Extracting saliency maps"):
            all_images.extend(batch_images.cpu().numpy())
            all_saliency_maps.extend(batch_maps.cpu().numpy())
            all_paths.extend(batch_paths)
        
        images_array = np.array(all_images)
        saliency_maps_array = np.array(all_saliency_maps)
        
        if save_maps:
            # Save saliency maps
            maps_dir = self.output_dir / f"saliency_maps_{self.model_name}"
            maps_dir.mkdir(exist_ok=True)
            
            for i, (smap, path) in enumerate(zip(saliency_maps_array, all_paths)):
                filename = f"{i:04d}_{Path(path).stem}.npy"
                np.save(maps_dir / filename, smap)
        
        return images_array, saliency_maps_array, all_paths
    
    def evaluate_insertion_deletion(self, 
                                  images: np.ndarray,
                                  saliency_maps: np.ndarray,
                                  batch_size: int = 4) -> Dict[str, np.ndarray]:
        """
        Evaluate using insertion and deletion metrics.
        
        Args:
            images: Array of images
            saliency_maps: Array of saliency maps
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with insertion and deletion scores
        """
        print("Evaluating insertion/deletion metrics...")
        
        # Convert to torch tensors
        images_tensor = torch.from_numpy(images).float()
        
        # Evaluate insertion
        print("Computing insertion scores...")
        insertion_scores = self.insertion.evaluate(
            images_tensor, 
            saliency_maps, 
            batch_size
        )
        
        # Evaluate deletion
        print("Computing deletion scores...")
        deletion_scores = self.deletion.evaluate(
            images_tensor, 
            saliency_maps, 
            batch_size
        )
        
        results = {
            'insertion_scores': insertion_scores,
            'deletion_scores': deletion_scores,
            'insertion_auc': self._compute_auc(insertion_scores.mean(1)),
            'deletion_auc': self._compute_auc(deletion_scores.mean(1))
        }
        
        return results
    
    def evaluate_road_metric(self, 
                           images: np.ndarray,
                           saliency_maps: np.ndarray) -> Dict[str, float]:
        """
        Evaluate using ROAD metric.
        
        Args:
            images: Array of images
            saliency_maps: Array of saliency maps
            
        Returns:
            Dictionary with ROAD metric results
        """
        print("Computing ROAD metric...")
        
        # Convert to torch tensors
        images_tensor = torch.from_numpy(images).float().to(self.device)
        
        # Compute ROAD scores
        road_results = []
        
        for i in tqdm(range(len(images)), desc="ROAD evaluation"):
            image = images_tensor[i:i+1]
            saliency = saliency_maps[i]
            
            # Get model prediction and create targets
            with torch.no_grad():
                pred = self.model(image)
                pred_class = torch.argmax(pred, dim=1).item()
            
            # Import ClassifierOutputTarget
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            targets = [ClassifierOutputTarget(pred_class)]
            
            # Compute ROAD score
            road_score = self.road(
                input_tensor=image,
                cams=saliency[np.newaxis, :, :],  # Add batch dimension
                targets=targets,
                model=self.model
            )
            
            road_results.append(road_score)
        
        road_array = np.array(road_results)
        
        return {
            'road_scores': road_array,
            'road_mean': np.mean(road_array),
            'road_std': np.std(road_array)
        }
    
    def _compute_auc(self, scores: np.ndarray) -> float:
        """Compute normalized AUC."""
        return (scores.sum() - scores[0] / 2 - scores[-1] / 2) / (scores.shape[0] - 1)
    
    def run_full_evaluation(self, 
                          image_paths: List[str] = None,
                          batch_size: int = 8,
                          save_results: bool = True) -> Dict[str, any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            image_paths: Optional list of image paths
            batch_size: Batch size for processing
            save_results: Whether to save results
            
        Returns:
            Dictionary with all evaluation results
        """
        print(f"Starting full evaluation for {self.model_name}...")
        
        # Extract saliency maps
        images, saliency_maps, paths = self.extract_saliency_maps(
            image_paths, batch_size
        )
        
        # Run insertion/deletion evaluation  
        # Adjust batch size to ensure it divides evenly into the number of images
        eval_batch_size = min(batch_size, len(images))
        while len(images) % eval_batch_size != 0 and eval_batch_size > 1:
            eval_batch_size -= 1
        
        ins_del_results = self.evaluate_insertion_deletion(
            images, saliency_maps, batch_size=eval_batch_size
        )
        
        # Run ROAD evaluation
        road_results = self.evaluate_road_metric(images, saliency_maps)
        
        # Combine results
        all_results = {
            'model_name': self.model_name,
            'num_images': len(images),
            'insertion_auc': ins_del_results['insertion_auc'],
            'deletion_auc': ins_del_results['deletion_auc'],
            'road_mean': road_results['road_mean'],
            'road_std': road_results['road_std'],
            'detailed_results': {
                'insertion_deletion': ins_del_results,
                'road': road_results
            }
        }
        
        self.results[self.model_name] = all_results
        
        if save_results:
            self.save_results(all_results)
        
        # Clear caches to free memory
        self.cam_extractor.clear_cache()
        self.prediction_manager.clear_all_caches()
        
        return all_results
    
    def evaluate_layer_combinations(self, 
                                  layer_combinations: List[List[nn.Module]] = None,
                                  image_paths: List[str] = None,
                                  max_combinations: int = 5) -> pd.DataFrame:
        """
        Evaluate different layer combinations to find the optimal setup.
        
        Args:
            layer_combinations: List of layer combinations to test
            image_paths: Optional list of image paths for evaluation
            max_combinations: Maximum number of combinations to test
            
        Returns:
            DataFrame with comparison results for different layer combinations
        """
        if layer_combinations is None:
            layer_combinations = self.get_layer_combinations_for_experimentation()
        
        # Limit the number of combinations to test
        if len(layer_combinations) > max_combinations:
            layer_combinations = layer_combinations[:max_combinations]
            print(f"Testing first {max_combinations} layer combinations...")
        
        results = []
        
        for i, layers in enumerate(layer_combinations):
            print(f"\n{'='*50}")
            print(f"Testing combination {i+1}/{len(layer_combinations)}")
            print(f"Layers: {len(layers)} conv layers")
            print(f"{'='*50}")
            
            # Create a new evaluator with these specific layers
            temp_evaluator = XAIEvaluationSuite(
                model_name=self.model_name,
                conv_layers=layers,
                output_dir=self.output_dir / f"layer_combo_{i+1}"
            )
            
            try:
                # Run evaluation with smaller batch size for layer combinations
                eval_results = temp_evaluator.run_full_evaluation(
                    image_paths=image_paths,
                    batch_size=4,  # Smaller batch size for faster testing
                    save_results=False  # Don't save individual results
                )
                
                results.append({
                    'combination_id': i + 1,
                    'num_layers': len(layers),
                    'insertion_auc': eval_results['insertion_auc'],
                    'deletion_auc': eval_results['deletion_auc'],
                    'road_mean': eval_results['road_mean'],
                    'road_std': eval_results['road_std'],
                    'num_images': eval_results['num_images'],
                    'layer_description': f"{len(layers)} layers"
                })
                
                # Clear cache after each combination
                temp_evaluator.cam_extractor.clear_cache()
                temp_evaluator.prediction_manager.clear_all_caches()
                
            except Exception as e:
                print(f"Error evaluating combination {i+1}: {str(e)}")
                results.append({
                    'combination_id': i + 1,
                    'num_layers': len(layers),
                    'insertion_auc': np.nan,
                    'deletion_auc': np.nan,
                    'road_mean': np.nan,
                    'road_std': np.nan,
                    'num_images': 0,
                    'layer_description': f"{len(layers)} layers (failed)"
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Sort by insertion AUC (higher is better)
        comparison_df = comparison_df.sort_values('insertion_auc', ascending=False, na_position='last')
        
        # Save results
        comparison_df.to_csv(self.output_dir / "layer_combinations_comparison.csv", index=False)
        
        print(f"\n{'='*60}")
        print("Layer Combination Evaluation Results")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))
        
        # Find the best combination
        if not comparison_df['insertion_auc'].isna().all():
            best_combo = comparison_df.iloc[0]
            print(f"\n🏆 Best performing combination:")
            print(f"   Combination ID: {best_combo['combination_id']}")
            print(f"   Number of layers: {best_combo['num_layers']}")
            print(f"   Insertion AUC: {best_combo['insertion_auc']:.4f}")
            print(f"   Deletion AUC: {best_combo['deletion_auc']:.4f}")
            print(f"   ROAD Mean: {best_combo['road_mean']:.4f}")
        
        return comparison_df
    
    def comprehensive_layer_experimentation(self, 
                                          image_paths: List[str] = None,
                                          max_individual_layers: int = 15,
                                          max_combinations: int = 8,
                                          save_detailed_results: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive layer experimentation including individual layers, combinations, and depth analysis.
        
        Args:
            image_paths: Optional list of image paths for evaluation
            max_individual_layers: Maximum number of individual layers to test
            max_combinations: Maximum number of layer combinations to test
            save_detailed_results: Whether to save detailed results for each experiment
            
        Returns:
            Dictionary containing results from all experiments
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE LAYER EXPERIMENTATION FOR {self.model_name.upper()}")
        print(f"{'='*80}")
        
        all_results = {}
        
        # 1. Individual layer experimentation
        print(f"\n🔍 Phase 1: Individual Layer Analysis")
        print(f"{'='*50}")
        individual_results = self.experiment_all_individual_conv_layers(
            image_paths=image_paths,
            max_layers=max_individual_layers,
            batch_size=2
        )
        all_results['individual_layers'] = individual_results
        
        # 2. Layer combination experimentation
        print(f"\n🔍 Phase 2: Layer Combination Analysis")
        print(f"{'='*50}")
        combination_results = self.evaluate_layer_combinations(
            image_paths=image_paths,
            max_combinations=max_combinations
        )
        all_results['layer_combinations'] = combination_results
        
        # 3. Depth analysis
        print(f"\n🔍 Phase 3: Layer Depth Analysis")
        print(f"{'='*50}")
        depth_results = self.experiment_layer_depth_analysis(
            image_paths=image_paths,
            step_size=max(1, len(self.get_all_conv_layers()['all_conv_layers']) // 10)
        )
        all_results['depth_analysis'] = depth_results
        
        # 4. Generate comprehensive summary
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EXPERIMENTATION SUMMARY")
        print(f"{'='*80}")
        
        summary_stats = self._generate_experimentation_summary(all_results)
        all_results['summary'] = summary_stats
        
        # 5. Save all results
        if save_detailed_results:
            self._save_comprehensive_results(all_results)
        
        # 6. Generate recommendations
        recommendations = self._generate_layer_recommendations(all_results)
        all_results['recommendations'] = recommendations
        
        print(f"\n🎯 FINAL RECOMMENDATIONS:")
        print(f"{'='*50}")
        for category, rec in recommendations.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            if isinstance(rec, dict):
                for key, value in rec.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {rec}")
        
        return all_results
    
    def _generate_experimentation_summary(self, results: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Generate summary statistics from all experimentation results."""
        summary = {}
        
        # Individual layer stats
        if 'individual_layers' in results:
            individual_df = results['individual_layers']
            valid_individual = individual_df[individual_df['status'] == 'success']
            
            if len(valid_individual) > 0:
                summary['individual_layers'] = {
                    'total_tested': len(individual_df),
                    'successful': len(valid_individual),
                    'best_insertion_auc': valid_individual['insertion_auc'].max(),
                    'avg_insertion_auc': valid_individual['insertion_auc'].mean(),
                    'best_layer': valid_individual.loc[valid_individual['insertion_auc'].idxmax(), 'layer_name'],
                    'std_insertion_auc': valid_individual['insertion_auc'].std()
                }
        
        # Combination stats
        if 'layer_combinations' in results:
            combo_df = results['layer_combinations']
            valid_combo = combo_df.dropna(subset=['insertion_auc'])
            
            if len(valid_combo) > 0:
                summary['layer_combinations'] = {
                    'total_tested': len(combo_df),
                    'successful': len(valid_combo),
                    'best_insertion_auc': valid_combo['insertion_auc'].max(),
                    'avg_insertion_auc': valid_combo['insertion_auc'].mean(),
                    'best_num_layers': valid_combo.loc[valid_combo['insertion_auc'].idxmax(), 'num_layers'],
                    'optimal_layer_count': valid_combo.groupby('num_layers')['insertion_auc'].mean().idxmax()
                }
        
        # Depth analysis stats
        if 'depth_analysis' in results:
            depth_df = results['depth_analysis']
            valid_depth = depth_df.dropna(subset=['insertion_auc'])
            
            if len(valid_depth) > 0:
                best_depth_row = valid_depth.loc[valid_depth['insertion_auc'].idxmax()]
                summary['depth_analysis'] = {
                    'total_tested': len(depth_df),
                    'successful': len(valid_depth),
                    'best_insertion_auc': valid_depth['insertion_auc'].max(),
                    'optimal_depth_percentage': best_depth_row['depth_percentage'],
                    'optimal_depth_layer': best_depth_row['layer_name']
                }
        
        return summary
    
    def _generate_layer_recommendations(self, results: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Generate recommendations based on all experimentation results."""
        recommendations = {}
        
        # Best single layer recommendation
        if 'individual_layers' in results:
            individual_df = results['individual_layers']
            valid_individual = individual_df[individual_df['status'] == 'success']
            
            if len(valid_individual) > 0:
                best_single = valid_individual.iloc[0]
                recommendations['best_single_layer'] = {
                    'layer_name': best_single['layer_name'],
                    'insertion_auc': f"{best_single['insertion_auc']:.4f}",
                    'layer_info': best_single['layer_info']
                }
        
        # Best combination recommendation
        if 'layer_combinations' in results:
            combo_df = results['layer_combinations']
            valid_combo = combo_df.dropna(subset=['insertion_auc'])
            
            if len(valid_combo) > 0:
                best_combo = valid_combo.iloc[0]
                recommendations['best_layer_combination'] = {
                    'num_layers': best_combo['num_layers'],
                    'insertion_auc': f"{best_combo['insertion_auc']:.4f}",
                    'combination_id': best_combo['combination_id']
                }
        
        # Performance comparison
        single_best = 0
        combo_best = 0
        
        if 'individual_layers' in results:
            individual_df = results['individual_layers']
            valid_individual = individual_df[individual_df['status'] == 'success']
            if len(valid_individual) > 0:
                single_best = valid_individual['insertion_auc'].max()
        
        if 'layer_combinations' in results:
            combo_df = results['layer_combinations']
            valid_combo = combo_df.dropna(subset=['insertion_auc'])
            if len(valid_combo) > 0:
                combo_best = valid_combo['insertion_auc'].max()
        
        if single_best > 0 and combo_best > 0:
            if combo_best > single_best:
                improvement = ((combo_best - single_best) / single_best) * 100
                recommendations['strategy'] = f"Use layer combinations (improvement: {improvement:.1f}%)"
            else:
                recommendations['strategy'] = "Single layer performs as well as combinations"
        
        # Depth recommendation
        if 'depth_analysis' in results:
            depth_df = results['depth_analysis']
            valid_depth = depth_df.dropna(subset=['insertion_auc'])
            
            if len(valid_depth) > 0:
                best_depth = valid_depth.loc[valid_depth['insertion_auc'].idxmax()]
                recommendations['optimal_depth'] = {
                    'depth_percentage': f"{best_depth['depth_percentage']:.1f}%",
                    'recommendation': self._interpret_depth(best_depth['depth_percentage'])
                }
        
        return recommendations
    
    def _interpret_depth(self, depth_percentage: float) -> str:
        """Interpret the optimal depth percentage."""
        if depth_percentage < 25:
            return "Early layers (feature detection) work best"
        elif depth_percentage < 50:
            return "Mid-level layers (pattern recognition) work best"
        elif depth_percentage < 75:
            return "Late layers (complex features) work best"
        else:
            return "Final layers (high-level representations) work best"
    
    def _save_comprehensive_results(self, results: Dict[str, any]):
        """Save all comprehensive experimentation results."""
        # Create comprehensive results directory
        comp_dir = self.output_dir / "comprehensive_experimentation"
        comp_dir.mkdir(exist_ok=True)
        
        # Save individual DataFrames
        for key, data in results.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(comp_dir / f"{key}_{self.model_name}.csv", index=False)
        
        # Save summary as JSON
        if 'summary' in results:
            import json
            summary_file = comp_dir / f"summary_{self.model_name}.json"
            with open(summary_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                summary_clean = {}
                for key, value in results['summary'].items():
                    if isinstance(value, dict):
                        summary_clean[key] = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                            for k, v in value.items()}
                    else:
                        summary_clean[key] = value
                json.dump(summary_clean, f, indent=2)
        
        # Save recommendations
        if 'recommendations' in results:
            rec_file = comp_dir / f"recommendations_{self.model_name}.txt"
            with open(rec_file, 'w') as f:
                f.write(f"Layer Experimentation Recommendations for {self.model_name}\n")
                f.write("="*60 + "\n\n")
                
                for category, rec in results['recommendations'].items():
                    f.write(f"{category.replace('_', ' ').title()}:\n")
                    if isinstance(rec, dict):
                        for key, value in rec.items():
                            f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {rec}\n")
                    f.write("\n")
        
        print(f"\n💾 Comprehensive results saved to: {comp_dir}")
    
    def save_results(self, results: Dict[str, any]):
        """Save evaluation results to files."""
        # Save summary results
        summary = {
            'model_name': results['model_name'],
            'num_images': results['num_images'],
            'insertion_auc': results['insertion_auc'],
            'deletion_auc': results['deletion_auc'],
            'road_mean': results['road_mean'],
            'road_std': results['road_std']
        }
        
        df = pd.DataFrame([summary])
        df.to_csv(self.output_dir / f"summary_{self.model_name}.csv", index=False)
        
        # Save detailed results
        np.savez(
            self.output_dir / f"detailed_results_{self.model_name}.npz",
            **results['detailed_results']['insertion_deletion'],
            **results['detailed_results']['road']
        )
        
        print(f"Results saved to {self.output_dir}")
    
    def plot_results(self, save_plots: bool = True):
        """Plot evaluation results."""
        if not self.results:
            print("No results to plot. Run evaluation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for model_name, results in self.results.items():
            ins_scores = results['detailed_results']['insertion_deletion']['insertion_scores']
            del_scores = results['detailed_results']['insertion_deletion']['deletion_scores']
            
            # Plot insertion curve
            axes[0, 0].plot(ins_scores.mean(1), label=f"{model_name} (AUC: {results['insertion_auc']:.3f})")
            axes[0, 0].set_title("Insertion Metric")
            axes[0, 0].set_xlabel("Fraction of pixels inserted")
            axes[0, 0].set_ylabel("Model confidence")
            axes[0, 0].legend()
            
            # Plot deletion curve
            axes[0, 1].plot(del_scores.mean(1), label=f"{model_name} (AUC: {results['deletion_auc']:.3f})")
            axes[0, 1].set_title("Deletion Metric")
            axes[0, 1].set_xlabel("Fraction of pixels deleted")
            axes[0, 1].set_ylabel("Model confidence")
            axes[0, 1].legend()
            
            # Plot ROAD scores histogram
            road_scores = results['detailed_results']['road']['road_scores']
            axes[1, 0].hist(road_scores, bins=20, alpha=0.7, label=f"{model_name}")
            axes[1, 0].set_title("ROAD Score Distribution")
            axes[1, 0].set_xlabel("ROAD Score")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].legend()
            
            # Summary bar plot
            metrics = ['Insertion AUC', 'Deletion AUC', 'ROAD Mean']
            values = [results['insertion_auc'], results['deletion_auc'], results['road_mean']]
            axes[1, 1].bar(metrics, values, alpha=0.7, label=model_name)
            axes[1, 1].set_title("Summary Metrics")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / "evaluation_plots.png", dpi=300, bbox_inches='tight')
            print(f"Plots saved to {self.output_dir / 'evaluation_plots.png'}")
        
        plt.show()


def evaluate_multiple_models(model_names: List[str], 
                           image_paths: List[str] = None,
                           output_dir: str = "./evaluation_results",
                           device_preference: str = "auto") -> pd.DataFrame:
    """
    Evaluate multiple models and compare results.
    
    Args:
        model_names: List of model names to evaluate
        image_paths: Optional list of image paths
        output_dir: Output directory for results
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        DataFrame with comparison results
    """
    all_results = []
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        evaluator = XAIEvaluationSuite(
            model_name, 
            output_dir=f"{output_dir}/{model_name}",
            device_preference=device_preference
        )
        results = evaluator.run_full_evaluation(image_paths)
        
        all_results.append({
            'model_name': model_name,
            'insertion_auc': results['insertion_auc'],
            'deletion_auc': results['deletion_auc'],
            'road_mean': results['road_mean'],
            'road_std': results['road_std'],
            'num_images': results['num_images']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results)
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    print(f"\nComparison results saved to {output_dir}/model_comparison.csv")
    print("\nComparison Summary:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


# Example usage
if __name__ == "__main__":
    # Example: Evaluate a single model
    model_name = "resnet50"
    evaluator = XAIEvaluationSuite(model_name)
    results = evaluator.run_full_evaluation()
    evaluator.plot_results()
    
    # Example: Evaluate multiple models
    # model_names = ["resnet50", "b0", "densenet"]
    # comparison_df = evaluate_multiple_models(model_names)
