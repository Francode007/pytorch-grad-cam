#!/usr/bin/env python3
"""
Comprehensive comparison between Enhanced CAM and Standard CAM methods.
This script uses the ProperAUCEvaluator for consistent evaluation.
"""

import sys
import numpy as np
import torch
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.enhanced_proper_auc_evaluator import EnhancedProperAUCEvaluator

def run_comprehensive_comparison(model_name="resnet18", max_images=2, step_size=50):
    """Run comprehensive comparison using the ProperAUCEvaluator."""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVALUATION COMPARISON")
    print(f"Using ProperAUCEvaluator for consistent AUC calculations")
    print(f"{'='*60}")
    
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Max Images: {max_images}")
    print(f"  Step Size: {step_size}")
    
    # Initialize the enhanced evaluator
    evaluator = EnhancedProperAUCEvaluator(model_name=model_name, device_preference="mps")
    
    # Compare Enhanced CAM vs Standard methods
    standard_methods = ["GradCAM", "GradCAM++"]
    comparison_df = evaluator.compare_enhanced_vs_standard(
        standard_methods=standard_methods,
        max_images=max_images,
        step_size=step_size
    )
    
    print(f"\n{'='*80}")
    print("FINAL COMPARISON RESULTS:")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print("• All AUC values should now be in [0, 1] range")
    print("• Higher insertion AUC = better (pixels added improve confidence)")
    print("• Lower deletion AUC = better (pixels removed decrease confidence)")
    print("• Lower ROAD score = better (more robust explanations)")
    print(f"{'='*80}")
    
    return comparison_df

def main():
    """Main comparison function."""
    print("🔍 COMPREHENSIVE EVALUATION COMPARISON")
    print("Using modular ProperAUCEvaluator approach")
    
    model_name = "resnet18"
    max_images = 2
    step_size = 50  # Proper step size instead of 224
    
    try:
        comparison_df = run_comprehensive_comparison(model_name, max_images, step_size)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"   The results should now show proper AUC values in [0,1] range.")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
