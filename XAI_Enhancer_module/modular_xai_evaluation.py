#!/usr/bin/env python3
"""
Updated XAI Evaluation Suite using ProperAUCEvaluator.
This script provides comprehensive evaluation using the modular ProperAUCEvaluator approach.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.evaluator.enhanced_proper_auc_evaluator import EnhancedProperAUCEvaluator
from XAI_Enhancer_module.evaluator.proper_auc_evaluation import ProperAUCEvaluator
from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, print_directory_structure

class ModularXAIEvaluationSuite:
    """
    Modular evaluation suite using ProperAUCEvaluator as the base.
    """
    
    def __init__(self, model_name: str, device_preference: str = "auto", 
                 layer_mode: str = "last"):
        """
        Initialize the evaluation suite.
        
        Args:
            model_name: Name of the model
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
            layer_mode: Layer selection mode for Enhanced CAM
        """
        self.model_name = model_name
        self.device_preference = device_preference
        self.layer_mode = layer_mode
        self.enhanced_evaluator = EnhancedProperAUCEvaluator(
            model_name, device_preference, layer_mode
        )
        
        print(f"ModularXAIEvaluationSuite initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {device_preference}")
        print(f"  Layer mode: {layer_mode}")
    
    def evaluate_enhanced_cam(self, max_images: int = 2, step_size: int = 50, 
                            verbose: bool = None) -> Dict:
        """
        Evaluate Enhanced CAM method.
        
        Args:
            max_images: Maximum number of images to evaluate
            step_size: Step size for evaluation
            verbose: If None, auto-determine based on max_images. 
                    If True/False, override auto-detection.
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print("EVALUATING ENHANCED CAM")
        print(f"{'='*60}")
        
        # Auto-determine verbosity if not specified
        if verbose is None:
            verbose = max_images <= 20
        
        results = self.enhanced_evaluator.evaluate_enhanced_cam(
            max_images=max_images, 
            step_size=step_size,
            verbose=verbose
        )
        
        self._print_results("Enhanced CAM", results)
        return results
    
    def evaluate_standard_methods(self, methods: List[str] = None, max_images: int = 2) -> Dict:
        """Evaluate standard CAM methods."""
        if methods is None:
            methods = ["GradCAM", "GradCAM++", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"]
        
        results = {}
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"EVALUATING {method}")
            print(f"{'='*60}")
            
            method_results = self.enhanced_evaluator.evaluate_method(
                cam_method_name=method,
                max_images=max_images
            )
            
            results[method] = method_results
            self._print_results(method, method_results)
        
        return results
    
    def run_full_comparison(self, standard_methods: List[str] = None, 
                          max_images: int = 2, step_size: int = 50, 
                          verbose: bool = None) -> pd.DataFrame:
        """
        Run full comparison between Enhanced CAM and standard methods.
        
        Args:
            standard_methods: List of standard CAM methods to compare
            max_images: Maximum number of images to evaluate
            step_size: Step size for evaluation
            verbose: If None, auto-determine based on max_images
        
        Returns:
            DataFrame with comparison results
        """
        print(f"\n{'='*80}")
        print("FULL COMPARISON EVALUATION")
        print(f"{'='*80}")
        
        if standard_methods is None:
            standard_methods = ["GradCAM", "GradCAM++", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"]
        
        # Auto-determine verbosity if not specified
        if verbose is None:
            verbose = max_images <= 20
        
        comparison_df = self.enhanced_evaluator.compare_enhanced_vs_standard(
            standard_methods=standard_methods,
            max_images=max_images,
            step_size=step_size,
            verbose=verbose
        )
        
        return comparison_df
    
    def _print_results(self, method_name: str, results: Dict):
        """Print formatted results."""
        print(f"\n📊 Results for {method_name}:")
        print(f"   Insertion AUC: {results['insertion_auc_mean']:.4f} ± {results['insertion_auc_std']:.4f}")
        print(f"   Deletion AUC: {results['deletion_auc_mean']:.4f} ± {results['deletion_auc_std']:.4f}")
        print(f"   ROAD Score: {results['road_mean']:.4f} ± {results['road_std']:.4f}")
        print(f"   Images evaluated: {results['num_images']}")
        
        # Validation check
        insertion_mean = results['insertion_auc_mean']
        deletion_mean = results['deletion_auc_mean']
        
        if 0 <= insertion_mean <= 1 and 0 <= deletion_mean <= 1:
            print(f"   ✅ AUC values are in valid [0,1] range")
        else:
            print(f"   ❌ AUC values are outside [0,1] range - check evaluation!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Modular XAI Evaluation Suite using ProperAUCEvaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Enhanced CAM only with last layer (auto verbosity)
  python modular_xai_evaluation.py --model resnet18 --eval-type enhanced-only --max-images 2 --layer-mode last

  # Large scale comparison with quiet mode (all standard methods)
  python modular_xai_evaluation.py --model resnet18 --eval-type comparison --max-images 100 --layer-mode all --quiet

  # Compare Enhanced CAM vs specific standard methods
  python modular_xai_evaluation.py --model resnet18 --eval-type comparison --max-images 5 --methods GradCAM EigenCAM HiResCAM

  # Evaluate all standard CAM methods and save to custom directories
  python modular_xai_evaluation.py --model resnet18 --eval-type standard-only --methods GradCAM GradCAM++ EigenCAM HiResCAM LayerCAM ScoreCAM --output-csv-dir ./custom_csv --output-analysis-dir ./custom_analysis

  # Quick test with new methods (results saved to ./csv_exports/resnet18/ and ./analysis_results/resnet18/)
  python modular_xai_evaluation.py --model resnet18 --eval-type comparison --max-images 2 --methods LayerCAM ScoreCAM

  # Evaluate Enhanced CAM with last 5 layers vs LayerCAM and ScoreCAM
  python modular_xai_evaluation.py --model resnet18 --eval-type comparison --layer-mode last_5 --methods LayerCAM ScoreCAM
  
  # All outputs are automatically organized by model name into subdirectories
        """
    )
    
    parser.add_argument('--model', '-m', default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'b0', 'b4'],
                       help='Model to evaluate')
    
    parser.add_argument('--eval-type', default='comparison',
                       choices=['enhanced-only', 'standard-only', 'comparison'],
                       help='Type of evaluation to run')
    
    parser.add_argument('--max-images', type=int, default=2,
                       help='Maximum number of images to evaluate (use -1 for entire validation dataset)')
    
    parser.add_argument('--step-size', type=int, default=224,
                       help='Step size for insertion/deletion evaluation')
    
    parser.add_argument('--methods', nargs='+', 
                       default=['GradCAM', 'GradCAM++', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'ScoreCAM'],
                       choices=['GradCAM', 'GradCAM++', 'EigenGradCAM', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'ScoreCAM'],
                       help='Standard CAM methods to evaluate')
    
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device preference')
    
    parser.add_argument('--layer-mode', default='last',
                       choices=['all', 'last_5', 'last'],
                       help='Layer selection mode for Enhanced CAM')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Force verbose output (detailed per-image logging)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Force quiet output (minimal logging)')
    
    parser.add_argument('--output-analysis-dir', default='./analysis_results',
                       help='Base directory for analysis results (model subdirs will be created)')
    
    parser.add_argument('--output-csv-dir', default='./csv_exports',
                       help='Base directory for CSV exports (model subdirs will be created)')
    
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
    print("MODULAR XAI EVALUATION SUITE")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Evaluation type: {args.eval_type}")
    print(f"  Max images: {args.max_images}")
    print(f"  Step size: {args.step_size}")
    print(f"  Device: {args.device}")
    print(f"  Layer mode: {args.layer_mode}")
    if verbose is not None:
        print(f"  Verbose: {verbose}")
    else:
        print(f"  Verbose: Auto (True for ≤20 images, False for >20 images)")
    
    try:
        # Initialize evaluation suite
        suite = ModularXAIEvaluationSuite(args.model, args.device, args.layer_mode)
        
        if args.eval_type == 'enhanced-only':
            # Evaluate Enhanced CAM only
            enhanced_results = suite.evaluate_enhanced_cam(
                args.max_images, args.step_size, verbose
            )
            
        elif args.eval_type == 'standard-only':
            # Evaluate standard methods only
            standard_results = suite.evaluate_standard_methods(args.methods, args.max_images)
            
        elif args.eval_type == 'comparison':
            # Full comparison
            comparison_df = suite.run_full_comparison(
                standard_methods=args.methods,
                max_images=args.max_images,
                step_size=args.step_size,
                verbose=verbose
            )
            
            print(f"\n{'='*80}")
            print("FINAL COMPARISON TABLE:")
            print(f"{'='*80}")
            print(comparison_df.to_string(index=False))
        
        print(f"\n✅ Evaluation completed successfully!")
        
        # Show output directory structure
        print(f"\n📁 OUTPUT SUMMARY:")
        print(f"{'='*50}")
        print(f"Results have been automatically saved to model-specific directories:")
        print(f"• Analysis results: {args.output_analysis_dir}/{args.model}/")
        print(f"• CSV exports: {args.output_csv_dir}/{args.model}/")
        print(f"\nFor detailed directory structure:")
        print_directory_structure(args.output_analysis_dir, args.output_csv_dir)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
