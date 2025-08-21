#!/usr/bin/env python3
"""
Multi-Dataset XAI Evaluation Suite.
Extended evaluation framework supporting both IBS and ImageNet datasets.
This script provides comprehensive evaluation using the modular ProperAUCEvaluator approach.
"""

import sys
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.evaluator.enhanced_proper_auc_evaluator import EnhancedProperAUCEvaluator
from XAI_Enhancer_module.evaluator.proper_auc_evaluation import ProperAUCEvaluator
from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, print_directory_structure
from XAI_Enhancer_module.utils.model_utils import (
    test_model, get_validation_paths, get_imagenet_validation_paths,
    TRAIN_DATA_PATH, IMAGENET_VAL_PATH, get_transformations
)
from XAI_Enhancer_module.utils.imagenet_utils import (
    sample_imagenet_images, get_imagenet_validation_stats, get_class_from_path
)


class MultiDatasetXAIEvaluationSuite:
    """
    Multi-dataset evaluation suite supporting both IBS and ImageNet datasets.
    Uses ProperAUCEvaluator as the base for all evaluations.
    """
    
    def __init__(self, model_name: str, dataset_type: str = "ibs",
                 device_preference: str = "auto", layer_mode: str = "last",
                 enhanced_cam_method: str = "GradCAMEnhanced",
                 base_model_path: Optional[str] = None):
        """
        Initialize the evaluation suite.
        
        Args:
            model_name: Name of the model
            dataset_type: "ibs" or "imagenet"
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
            layer_mode: Layer selection mode for Enhanced CAM
            enhanced_cam_method: Enhanced CAM method to use
            base_model_path: Optional path to the directory containing model weights
        """
        self.model_name = model_name
        self.dataset_type = dataset_type.lower()
        self.device_preference = device_preference
        self.layer_mode = layer_mode
        self.enhanced_cam_method = enhanced_cam_method
        self.base_model_path = base_model_path
        
        # Determine dataset path and configuration
        if self.dataset_type == "imagenet":
            self.dataset_path = IMAGENET_VAL_PATH
            self.num_classes = 1000
        else:  # IBS
            self.dataset_path = TRAIN_DATA_PATH
            self.num_classes = 2
        
        # Load the model
        self.model = test_model(
            model_name, 
            num_classes=self.num_classes,
            device_preference=device_preference,
            dataset_type=self.dataset_type,
            base_model_path=self.base_model_path
        )
        
        # Initialize evaluator
        self.evaluator = self._create_evaluator()
        
        print(f"MultiDatasetXAIEvaluationSuite initialized:")
        print(f"  Model: {model_name}")
        print(f"  Dataset: {dataset_type.upper()}")
        print(f"  Device: {device_preference}")
        print(f"  Layer mode: {layer_mode}")
        print(f"  Enhanced CAM method: {enhanced_cam_method}")
        print(f"  Number of classes: {self.num_classes}")
        if self.base_model_path:
            print(f"  Model base path: {self.base_model_path}")
    
    def _create_evaluator(self):
        """Create appropriate evaluator based on dataset type."""
        if self.dataset_type == "imagenet":
            return ImageNetProperAUCEvaluator(
                model=self.model,
                model_name=self.model_name,
                dataset_path=self.dataset_path,
                device_preference=self.device_preference
            )
        else:
            return EnhancedProperAUCEvaluator(
                model=self.model,
                model_name=self.model_name,
                dataset_path=self.dataset_path,
                layer_mode=self.layer_mode,
                enhanced_cam_method=self.enhanced_cam_method
            )
    
    def get_sample_images(self, max_images: int = 10) -> List[str]:
        """Get sample images based on dataset type."""
        if self.dataset_type == "imagenet":
            if max_images == -1:
                # Use all ImageNet validation images (50k images)
                return get_imagenet_validation_paths(self.dataset_path)
            else:
                # Sample images across classes
                num_classes = min(max_images, 50)  # Reasonable number of classes
                images_per_class = max(1, max_images // num_classes)
                return sample_imagenet_images(
                    self.dataset_path, 
                    num_classes=num_classes,
                    images_per_class=images_per_class
                )
        else:
            # IBS dataset
            if max_images == -1:
                return get_validation_paths(self.dataset_path)
            else:
                all_paths = get_validation_paths(self.dataset_path)
                return all_paths[:max_images]
    
    def print_dataset_info(self):
        """Print information about the current dataset."""
        print(f"\n{'='*60}")
        print(f"DATASET INFORMATION: {self.dataset_type.upper()}")
        print(f"{'='*60}")
        
        if self.dataset_type == "imagenet":
            stats = get_imagenet_validation_stats(self.dataset_path)
            print(f"Total classes: {stats['total_classes']}")
            print(f"Total images: {stats['total_images']}")
            print(f"Average images per class: {stats['avg_images_per_class']:.1f}")
            print(f"Min/Max images per class: {stats['min_images_per_class']}/{stats['max_images_per_class']}")
            print(f"Sample synsets: {', '.join(stats['sample_synsets'])}")
        else:
            all_paths = get_validation_paths(self.dataset_path)
            print(f"Total validation images: {len(all_paths)}")
            print(f"Classes: IBS, Normal")
            if all_paths:
                print(f"Sample path: {all_paths[0]}")
    
    def evaluate_enhanced_cam(self, max_images: int = 10, step_size: int = 50, 
                            verbose: bool = None) -> Dict:
        """Evaluate Enhanced CAM method."""
        print(f"\n{'='*60}")
        print("EVALUATING ENHANCED CAM")
        print(f"{'='*60}")
        
        if verbose is None:
            verbose = max_images <= 20
        
        results = self.evaluator.evaluate_enhanced_cam(
            max_images=max_images, 
            step_size=step_size,
            verbose=verbose
        )
        
        self._print_results("Enhanced CAM", results)
        return results
    
    def evaluate_standard_methods(self, methods: List[str] = None, max_images: int = 10,
                                base_csv_dir: str = "./csv_exports",
                                base_analysis_dir: str = "./analysis_results") -> Dict:
        """Evaluate standard CAM methods."""
        if methods is None:
            methods = ["GradCAM", "GradCAM++", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"]
        
        # Get sample images
        sample_images = self.get_sample_images(max_images)
        
        print(f"\n{'='*60}")
        print(f"EVALUATING STANDARD METHODS ON {self.dataset_type.upper()}")
        print(f"Using {len(sample_images)} images")
        print(f"{'='*60}")
        
        results = {}
        all_results_for_export = []
        
        for method in methods:
            print(f"\n{'-'*40}")
            print(f"EVALUATING {method}")
            print(f"{'-'*40}")
            
            method_results = self.evaluator.evaluate_method(
                cam_method_name=method,
                max_images=len(sample_images),
                sample_paths=sample_images
            )
            
            results[method] = method_results
            self._print_results(method, method_results)
            
            # Prepare data for CSV export
            export_row = {
                'Method': method,
                'Model': self.model_name,
                'Dataset': self.dataset_type.upper(),
                'Insertion_AUC_Mean': method_results['insertion_auc_mean'],
                'Insertion_AUC_Std': method_results['insertion_auc_std'],
                'Deletion_AUC_Mean': method_results['deletion_auc_mean'],
                'Deletion_AUC_Std': method_results['deletion_auc_std'],
                'ROAD_Mean': method_results['road_mean'],
                'ROAD_Std': method_results['road_std'],
                'Images_Evaluated': method_results['num_images']
            }
            all_results_for_export.append(export_row)
        
        # Save results
        if all_results_for_export:
            self._save_results(all_results_for_export, results, max_images, 
                             base_csv_dir, base_analysis_dir, methods)
        
        return results
    
    def _save_results(self, results_df_data, detailed_results, max_images, 
                     base_csv_dir, base_analysis_dir, methods):
        """Save evaluation results to files."""
        from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, save_analysis_data
        
        # Create DataFrame for CSV export
        results_df = pd.DataFrame(results_df_data)
        
        # Save CSV results
        evaluation_type = f"standard_methods_{self.dataset_type}"
        csv_path = save_evaluation_results(
            results_df, 
            self.model_name, 
            evaluation_type=evaluation_type,
            base_csv_dir=base_csv_dir
        )
        print(f"\n💾 Results saved to: {csv_path}")
        
        # Save detailed analysis data (pickle)
        analysis_data = {
            'model_name': self.model_name,
            'dataset_type': self.dataset_type,
            'evaluation_type': evaluation_type,
            'methods_evaluated': methods,
            'detailed_results': detailed_results,
            'summary_df': results_df,
            'max_images': max_images,
            'num_classes': self.num_classes
        }
        
        pickle_path = save_analysis_data(
            analysis_data,
            self.model_name,
            analysis_type=f"standard_methods_{self.dataset_type}_detailed",
            base_analysis_dir=base_analysis_dir
        )
        print(f"💾 Detailed analysis saved to: {pickle_path}")
    
    def _print_results(self, method_name: str, results: Dict):
        """Print evaluation results in a formatted way."""
        if not results:
            return
            
        print(f"\n📊 {method_name} Results:")
        print(f"  Insertion AUC: {results['insertion_auc_mean']:.4f} ± {results['insertion_auc_std']:.4f}")
        print(f"  Deletion AUC:  {results['deletion_auc_mean']:.4f} ± {results['deletion_auc_std']:.4f}")
        print(f"  ROAD Score:    {results['road_mean']:.6f} ± {results['road_std']:.6f}")
        print(f"  Images Used:   {results['num_images']}")
    
    def run_full_evaluation(self, methods: List[str] = None, max_images: int = 10,
                          include_enhanced: bool = True, step_size: int = 50,
                          base_csv_dir: str = "./csv_exports",
                          base_analysis_dir: str = "./analysis_results") -> Dict:
        """Run full evaluation with all methods."""
        self.print_dataset_info()
        
        all_results = {}
        
        # Standard methods
        standard_results = self.evaluate_standard_methods(
            methods=methods, 
            max_images=max_images,
            base_csv_dir=base_csv_dir,
            base_analysis_dir=base_analysis_dir
        )
        all_results['standard'] = standard_results
        
        # Enhanced CAM (only for IBS dataset)
        if include_enhanced and self.dataset_type != "imagenet":
            enhanced_results = self.evaluate_enhanced_cam(
                max_images=max_images, 
                step_size=step_size
            )
            all_results['enhanced'] = enhanced_results
        
        return all_results


class ImageNetProperAUCEvaluator(ProperAUCEvaluator):
    """
    Extended ProperAUCEvaluator for ImageNet dataset.
    """
    
    def __init__(self, model, model_name: str, dataset_path: str, device_preference: str = "auto"):
        # Initialize parent class (but skip the model loading since we already have the model)
        self.model_name = model_name
        self.device = next(model.parameters()).device
        self.dataset_path = dataset_path
        self.device_preference = device_preference
        
        # Set model (passed from outside)
        self.model = model
        self.model.eval()
        
        # Set image size based on model
        self.img_size = 224
        if model_name in ('b4',):
            self.img_size = 384
        
        # Get target layers
        self.target_layers = self._get_target_layers()
    
    def _get_target_layers(self):
        """Get target layers for CAM extraction."""
        # This is similar to the base class but adapted for different models
        if hasattr(self.model, 'features'):
            return [self.model.features[-1]]
        elif hasattr(self.model, 'layer4'):
            return [self.model.layer4[-1]]
        elif hasattr(self.model, 'blocks'):
            return [self.model.blocks[-1]]
        else:
            # Fallback: find the last convolutional layer
            conv_layers = []
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(module)
            return conv_layers[-1:] if conv_layers else []
    
    def evaluate_method(self, cam_method_name: str = "GradCAM", max_images: int = 10,
                       sample_paths: List[str] = None) -> Dict:
        """Evaluate a CAM method on ImageNet images."""
        import torch
        import numpy as np
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, HiResCAM, LayerCAM, ScoreCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from PIL import Image
        from XAI_Enhancer_module.utils.model_utils import get_transformations
        from XAI_Enhancer_module.utils.imagenet_utils import get_class_from_path
        
        # Get CAM method
        cam_methods = {
            "GradCAM": GradCAM,
            "GradCAM++": GradCAMPlusPlus, 
            "EigenCAM": EigenCAM,
            "HiResCAM": HiResCAM,
            "LayerCAM": LayerCAM,
            "ScoreCAM": ScoreCAM
        }
        
        if cam_method_name not in cam_methods:
            raise ValueError(f"Unknown CAM method: {cam_method_name}")
        
        cam_method = cam_methods[cam_method_name]
        
        if sample_paths is None:
            sample_paths = sample_imagenet_images(self.dataset_path, 
                                                num_classes=min(max_images, 10),
                                                images_per_class=max(1, max_images // 10))[:max_images]
        
        # Evaluation metrics storage
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        transform = get_transformations("imagenet")
        
        for i, image_path in enumerate(sample_paths):
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                img_size = 384 if self.model_name.startswith("b4") else 224
                image = image.resize((img_size, img_size))
                
                # Convert to tensor
                input_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Get predicted class (disable gradients only for prediction, not CAM)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()
                
                # Create CAM (gradients needed here)
                with cam_method(model=self.model, target_layers=self.target_layers) as cam:
                    targets = [ClassifierOutputTarget(predicted_class)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    saliency_map = grayscale_cam[0]  # Get first (and only) result
                    
                # Compute evaluation metrics
                insertion_auc = self.compute_insertion_auc(input_tensor, saliency_map, predicted_class)
                deletion_auc = self.compute_deletion_auc(input_tensor, saliency_map, predicted_class)
                road_score = self.evaluate_road(input_tensor, saliency_map, predicted_class)
                
                insertion_aucs.append(insertion_auc[1])  # Get AUC value
                deletion_aucs.append(deletion_auc[1])
                road_scores.append(road_score)
                
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(sample_paths)} images")
                    
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue
        
        # Compute statistics
        results = {
            'insertion_auc_mean': np.mean(insertion_aucs),
            'insertion_auc_std': np.std(insertion_aucs),
            'deletion_auc_mean': np.mean(deletion_aucs),
            'deletion_auc_std': np.std(deletion_aucs),
            'road_mean': np.mean(road_scores),
            'road_std': np.std(road_scores),
            'num_images': len(insertion_aucs)
        }
        
        return results

    def evaluate_enhanced_cam(self, max_images: int = 2, step_size: int = 50, 
                            verbose: bool = True) -> Dict[str, any]:
        """
        Evaluate Enhanced CAM method on ImageNet with proper AUC calculations.
        
        Args:
            max_images: Maximum number of images to evaluate
            step_size: Step size for insertion/deletion evaluation
            verbose: If True, show detailed logging for each image
        
        Returns:
            Dictionary with evaluation results
        """
        # Get sample ImageNet images
        sample_paths = sample_imagenet_images(self.dataset_path, 
                                            num_classes=min(max_images, 10),
                                            images_per_class=max(1, max_images // 10))[:max_images]
        
        # Get predictions using the model
        predicted_labels = []
        transform = get_transformations("imagenet")
        
        for image_path in sample_paths:
            # Load and preprocess image
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            img_size = 384 if self.model_name.startswith("b4") else 224
            image = image.resize((img_size, img_size))
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Get predicted class
            with torch.no_grad():
                output = self.model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                predicted_labels.append(predicted_class)
        
        print(f"\nEvaluating Enhanced CAM on {len(sample_paths)} ImageNet images...")
        print(f"Using step_size={step_size} for proper evaluation...")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        # Import Enhanced CAM components
        from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
        
        # Get enhanced layers for this model
        enhanced_layers = self._get_enhanced_cam_layers()
        
        for i, (image_path, predicted_label) in enumerate(zip(sample_paths, predicted_labels)):
            try:
                if verbose:
                    print(f"\n🔍 Processing image {i+1}/{len(sample_paths)}: {Path(image_path).name}")
                
                # Create Enhanced CAM extractor
                enhanced_cam_extractor = OptimizedCamExtractor(
                    self.model, 
                    self.model_name, 
                    enhanced_layers,
                    cam_method="GradCAMEnhanced"  # This should come from config
                )
                
                # Extract Enhanced CAM
                image_tensor, saliency_map = enhanced_cam_extractor.extract_saliency_map(
                    image_path, predicted_label
                )
                
                # Convert saliency map to numpy if needed
                if isinstance(saliency_map, torch.Tensor):
                    saliency_map = saliency_map.cpu().numpy()
                
                # Compute evaluation metrics
                insertion_auc = self.compute_insertion_auc(image_tensor, saliency_map, predicted_label)
                deletion_auc = self.compute_deletion_auc(image_tensor, saliency_map, predicted_label)
                road_score = self.evaluate_road(image_tensor, saliency_map, predicted_label)
                
                insertion_aucs.append(insertion_auc[1])  # Get AUC value
                deletion_aucs.append(deletion_auc[1])
                road_scores.append(road_score)
                
                if verbose:
                    print(f"   Insertion AUC: {insertion_auc[1]:.4f}")
                    print(f"   Deletion AUC:  {deletion_auc[1]:.4f}")
                    print(f"   ROAD Score:    {road_score:.6f}")
                
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue
        
        # Compile results
        results = {
            'cam_method': f"Enhanced CAM ({len(enhanced_layers)} layers)",
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
    
    def _get_enhanced_cam_layers(self) -> List[torch.nn.Module]:
        """Get enhanced CAM layers for ImageNet evaluation."""
        all_conv_layers = []
        
        # Collect all convolutional layers
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
                all_conv_layers.append(module)
        
        if not all_conv_layers:
            raise ValueError(f"No convolutional layers found in model {self.model_name}")
        
        # Return last 5 layers (or all if less than 5)
        return all_conv_layers[-5:] if len(all_conv_layers) >= 5 else all_conv_layers


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Dataset XAI Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # IBS dataset evaluation (default)
  python multi_dataset_xai_evaluation.py --model resnet18 --dataset ibs --max-images 10

  # ImageNet dataset evaluation
  python multi_dataset_xai_evaluation.py --model resnet18 --dataset imagenet --max-images 20

  # Large-scale ImageNet evaluation (quiet mode recommended)
  python multi_dataset_xai_evaluation.py --model resnet18 --dataset imagenet --max-images 100 --quiet

  # Full ImageNet validation set (50k images - use with caution!)
  python multi_dataset_xai_evaluation.py --model resnet18 --dataset imagenet --max-images -1 --quiet

  # Compare specific methods on ImageNet
  python multi_dataset_xai_evaluation.py --model resnet18 --dataset imagenet --methods GradCAM GradCAM++ --max-images 50

Dataset Notes:
  • IBS: Uses custom trained models from {BASE_MODEL_PATH}
  • ImageNet: Uses pretrained ImageNet1K models from torchvision/timm
  • ImageNet validation set contains ~50k images across 1000 classes
        """
    )
    
    parser.add_argument('--model', '-m', default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'b0', 'b4', 'densenet'],
                       help='Model to evaluate')
    
    parser.add_argument('--dataset', '-d', default='ibs',
                       choices=['ibs', 'imagenet'],
                       help='Dataset to use for evaluation')
    
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum images to evaluate (-1 for all validation images)')
    
    parser.add_argument('--methods', nargs='+',
                       default=['GradCAM', 'GradCAM++', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'ScoreCAM'],
                       choices=['GradCAM', 'GradCAM++', 'EigenGradCAM', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'ScoreCAM'],
                       help='CAM methods to evaluate')
    
    parser.add_argument('--step-size', type=int, default=50,
                       help='Step size for insertion/deletion evaluation')
    
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device preference')
    
    parser.add_argument('--layer-mode', default='last',
                       choices=['all', 'last_5', 'last'],
                       help='Layer selection mode for Enhanced CAM (IBS only)')
    
    parser.add_argument('--enhanced-cam-method', default='GradCAMEnhanced',
                       choices=['GradCAMEnhanced', 'GradCAMPlusPlusEnhanced', 'HiResCAMEnhanced', 
                               'ScoreCAMEnhanced', 'AblationCAMEnhanced'],
                       help='Enhanced CAM method (IBS only)')
    
    parser.add_argument('--no-enhanced', action='store_true',
                       help='Skip Enhanced CAM evaluation (even for IBS)')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output (recommended for large datasets)')
    
    parser.add_argument('--dataset-info', action='store_true',
                       help='Print dataset information and exit')
    
    args = parser.parse_args()
    
    # Create evaluation suite
    evaluator = MultiDatasetXAIEvaluationSuite(
        model_name=args.model,
        dataset_type=args.dataset,
        device_preference=args.device,
        layer_mode=args.layer_mode,
        enhanced_cam_method=args.enhanced_cam_method
    )
    
    if args.dataset_info:
        evaluator.print_dataset_info()
        return
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"MULTI-DATASET XAI EVALUATION")
    print(f"Model: {args.model} | Dataset: {args.dataset.upper()} | Max Images: {args.max_images}")
    print(f"Methods: {', '.join(args.methods)}")
    if args.dataset == 'ibs' and not args.no_enhanced:
        print(f"Enhanced CAM: {args.enhanced_cam_method} (Layer mode: {args.layer_mode})")
    print(f"{'='*80}")
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        methods=args.methods,
        max_images=args.max_images,
        include_enhanced=not args.no_enhanced,
        step_size=args.step_size
    )
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"Results saved to model-specific directories in ./csv_exports/{args.model}/")
    print(f"Detailed analysis saved to ./analysis_results/{args.model}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
