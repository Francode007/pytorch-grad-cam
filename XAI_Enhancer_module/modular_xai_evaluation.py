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

from XAI_Enhancer_module.enhanced_proper_auc_evaluator import EnhancedProperAUCEvaluator
from XAI_Enhancer_module.proper_auc_evaluation import ProperAUCEvaluator

class ModularXAIEvaluationSuite:
    """
    Modular evaluation suite using ProperAUCEvaluator as the base.
    """
    
    def __init__(self, model_name: str, device_preference: str = "auto"):
        self.model_name = model_name
        self.device_preference = device_preference
        self.enhanced_evaluator = EnhancedProperAUCEvaluator(model_name, device_preference)
        
        print(f"ModularXAIEvaluationSuite initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {device_preference}")
    
    def evaluate_enhanced_cam(self, max_images: int = 2, step_size: int = 50) -> Dict:
        """Evaluate Enhanced CAM method."""
        print(f"\n{'='*60}")
        print("EVALUATING ENHANCED CAM")
        print(f"{'='*60}")
        
        results = self.enhanced_evaluator.evaluate_enhanced_cam(
            max_images=max_images, 
            step_size=step_size
        )
        
        self._print_results("Enhanced CAM", results)
        return results
    
    def evaluate_standard_methods(self, methods: List[str] = None, max_images: int = 2) -> Dict:
        """Evaluate standard CAM methods."""
        if methods is None:
            methods = ["GradCAM", "GradCAM++"]
        
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
                          max_images: int = 2, step_size: int = 50) -> pd.DataFrame:
        """Run full comparison between Enhanced CAM and standard methods."""
        print(f"\n{'='*80}")
        print("FULL COMPARISON EVALUATION")
        print(f"{'='*80}")
        
        if standard_methods is None:
            standard_methods = ["GradCAM", "GradCAM++"]
        
        comparison_df = self.enhanced_evaluator.compare_enhanced_vs_standard(
            standard_methods=standard_methods,
            max_images=max_images,
            step_size=step_size
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
  # Evaluate Enhanced CAM only
  python modular_xai_evaluation.py --model resnet18 --eval-type enhanced-only --max-images 2

  # Compare Enhanced CAM vs standard methods
  python modular_xai_evaluation.py --model resnet18 --eval-type comparison --max-images 2

  # Evaluate standard methods only
  python modular_xai_evaluation.py --model resnet18 --eval-type standard-only --methods GradCAM GradCAM++
        """
    )
    
    parser.add_argument('--model', '-m', default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'b0', 'b4'],
                       help='Model to evaluate')
    
    parser.add_argument('--eval-type', default='comparison',
                       choices=['enhanced-only', 'standard-only', 'comparison'],
                       help='Type of evaluation to run')
    
    parser.add_argument('--max-images', type=int, default=2,
                       help='Maximum number of images to evaluate')
    
    parser.add_argument('--step-size', type=int, default=50,
                       help='Step size for insertion/deletion evaluation')
    
    parser.add_argument('--methods', nargs='+', default=['GradCAM', 'GradCAM++'],
                       choices=['GradCAM', 'GradCAM++', 'EigenGradCAM'],
                       help='Standard CAM methods to evaluate')
    
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device preference')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("MODULAR XAI EVALUATION SUITE")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Evaluation type: {args.eval_type}")
    print(f"  Max images: {args.max_images}")
    print(f"  Step size: {args.step_size}")
    print(f"  Device: {args.device}")
    
    try:
        # Initialize evaluation suite
        suite = ModularXAIEvaluationSuite(args.model, args.device)
        
        if args.eval_type == 'enhanced-only':
            # Evaluate Enhanced CAM only
            enhanced_results = suite.evaluate_enhanced_cam(args.max_images, args.step_size)
            
        elif args.eval_type == 'standard-only':
            # Evaluate standard methods only
            standard_results = suite.evaluate_standard_methods(args.methods, args.max_images)
            
        elif args.eval_type == 'comparison':
            # Full comparison
            comparison_df = suite.run_full_comparison(
                standard_methods=args.methods,
                max_images=args.max_images,
                step_size=args.step_size
            )
            
            print(f"\n{'='*80}")
            print("FINAL COMPARISON TABLE:")
            print(f"{'='*80}")
            print(comparison_df.to_string(index=False))
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
