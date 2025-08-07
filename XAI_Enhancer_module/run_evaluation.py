#!/usr/bin/env python3
"""
Command-line interface for the XAI Evaluation Suite.
This script provides comprehensive evaluation of explainability methods with device selection and evaluation type options.
"""

import sys
import argparse
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.xai_evaluation_suite import XAIEvaluationSuite, evaluate_multiple_models
from XAI_Enhancer_module.model_utils import get_validation_paths, TRAIN_DATA_PATH


def setup_argument_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="XAI Evaluation Suite - Comprehensive evaluation of explainability methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model evaluation on auto-detected device
  python run_evaluation.py --model resnet50 --eval-type single

  # Multiple model comparison on MPS (Apple Silicon GPU)
  python run_evaluation.py --models resnet50 b0 resnet18 --eval-type comparison --device mps

  # Layer analysis on CUDA
  python run_evaluation.py --model resnet50 --eval-type layer-analysis --device cuda

  # Individual layer experimentation on CPU
  python run_evaluation.py --model resnet50 --eval-type individual-layers --device cpu --max-layers 15

  # Quick test with custom batch size
  python run_evaluation.py --model resnet18 --eval-type quick --batch-size 4 --max-images 10

Available Models:
  resnet18, resnet34, resnet50, b0, b4, densenet, xception

Available Devices:
  auto (default) - Auto-detect best available device
  cuda          - NVIDIA GPU with CUDA
  mps           - Apple Silicon GPU (Metal Performance Shaders)
  cpu           - CPU only
        """
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model', '-m',
        type=str,
        choices=['resnet18', 'resnet34', 'resnet50', 'b0', 'b4', 'densenet', 'xception'],
        help='Single model to evaluate'
    )
    model_group.add_argument(
        '--models', '-ms',
        nargs='+',
        choices=['resnet18', 'resnet34', 'resnet50', 'b0', 'b4', 'densenet', 'xception'],
        help='Multiple models to compare'
    )
    
    # Evaluation type
    parser.add_argument(
        '--eval-type', '-e',
        type=str,
        required=True,
        choices=[
            'single',           # Single model full evaluation
            'comparison',       # Multiple models comparison
            'layer-analysis',   # Layer combination analysis
            'individual-layers', # Individual layer experimentation
            'depth-analysis',   # Layer depth analysis
            'comprehensive',    # Comprehensive layer experimentation
            'step-by-step',     # Step-by-step evaluation
            'quick'            # Quick test with minimal data
        ],
        help='Type of evaluation to perform'
    )
    
    # Device selection
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for computation (default: auto)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )
    
    parser.add_argument(
        '--max-images', '-i',
        type=int,
        default=None,
        help='Maximum number of images to evaluate (default: all validation images)'
    )
    
    parser.add_argument(
        '--max-layers',
        type=int,
        default=20,
        help='Maximum number of layers to test in individual layer experiments (default: 20)'
    )
    
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=5,
        help='Maximum number of layer combinations to test (default: 5)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./evaluation_results',
        help='Output directory for results (default: ./evaluation_results)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save result plots'
    )
    
    parser.add_argument(
        '--save-detailed',
        action='store_true',
        help='Save detailed results including intermediate data'
    )
    
    # Custom image paths
    parser.add_argument(
        '--image-paths',
        nargs='+',
        type=str,
        help='Custom image paths to evaluate (overrides validation set)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def run_single_evaluation(args):
    """Run single model evaluation."""
    print(f"\n{'='*60}")
    print(f"Single Model Evaluation: {args.model}")
    print(f"{'='*60}")
    
    evaluator = XAIEvaluationSuite(
        model_name=args.model,
        output_dir=args.output_dir,
        device_preference=args.device
    )
    
    # Get image paths
    if args.image_paths:
        image_paths = args.image_paths
    else:
        val_paths = get_validation_paths(TRAIN_DATA_PATH)
        if args.max_images:
            image_paths = val_paths[:args.max_images]
        else:
            image_paths = None
    
    results = evaluator.run_full_evaluation(
        image_paths=image_paths,
        batch_size=args.batch_size,
        save_results=True
    )
    
    print(f"\n🎉 Evaluation completed!")
    print(f"Results for {args.model}:")
    print(f"  Insertion AUC: {results['insertion_auc']:.4f}")
    print(f"  Deletion AUC: {results['deletion_auc']:.4f}")
    print(f"  ROAD Mean: {results['road_mean']:.4f} ± {results['road_std']:.4f}")
    print(f"  Images evaluated: {results['num_images']}")
    
    if args.save_plots:
        evaluator.plot_results(save_plots=True)
    
    return results


def run_comparison_evaluation(args):
    """Run multiple models comparison."""
    print(f"\n{'='*60}")
    print(f"Multiple Models Comparison: {', '.join(args.models)}")
    print(f"{'='*60}")
    
    # Get image paths
    if args.image_paths:
        image_paths = args.image_paths
    elif args.max_images:
        val_paths = get_validation_paths(TRAIN_DATA_PATH)
        image_paths = val_paths[:args.max_images]
    else:
        image_paths = None
    
    comparison_df = evaluate_multiple_models(
        model_names=args.models,
        image_paths=image_paths,
        output_dir=args.output_dir,
        device_preference=args.device
    )
    
    print(f"\n🎉 Comparison completed!")
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def run_layer_analysis(args):
    """Run layer combination analysis."""
    print(f"\n{'='*60}")
    print(f"Layer Analysis: {args.model}")
    print(f"{'='*60}")
    
    evaluator = XAIEvaluationSuite(
        model_name=args.model,
        output_dir=args.output_dir,
        device_preference=args.device
    )
    
    # Print layer summary
    evaluator.print_conv_layer_summary()
    
    # Get combinations for experimentation
    combinations = evaluator.get_layer_combinations_for_experimentation()
    print(f"\nGenerated {len(combinations)} different layer combinations for testing")
    
    # Get image paths
    if args.image_paths:
        image_paths = args.image_paths
    else:
        val_paths = get_validation_paths(TRAIN_DATA_PATH)
        test_paths = val_paths[:args.max_images] if args.max_images else val_paths[:10]
        image_paths = test_paths
    
    print(f"\nTesting layer combinations on {len(image_paths)} images...")
    combo_results = evaluator.evaluate_layer_combinations(
        layer_combinations=combinations[:args.max_combinations],
        image_paths=image_paths,
        max_combinations=args.max_combinations
    )
    
    print(f"\n🎉 Layer combination evaluation completed!")
    return combo_results


def run_individual_layers_evaluation(args):
    """Run individual layer experimentation."""
    print(f"\n{'='*60}")
    print(f"Individual Layer Experimentation: {args.model}")
    print(f"{'='*60}")
    
    evaluator = XAIEvaluationSuite(
        model_name=args.model,
        output_dir=args.output_dir,
        device_preference=args.device
    )
    
    # Get image paths
    if args.image_paths:
        image_paths = args.image_paths
    else:
        val_paths = get_validation_paths(TRAIN_DATA_PATH)
        test_paths = val_paths[:args.max_images] if args.max_images else val_paths[:20]
        image_paths = test_paths
    
    results = evaluator.experiment_all_individual_conv_layers(
        image_paths=image_paths,
        max_layers=args.max_layers,
        batch_size=args.batch_size
    )
    
    print(f"\n🎉 Individual layer experimentation completed!")
    return results


def run_comprehensive_evaluation(args):
    """Run comprehensive layer experimentation."""
    print(f"\n{'='*60}")
    print(f"Comprehensive Layer Experimentation: {args.model}")
    print(f"{'='*60}")
    
    evaluator = XAIEvaluationSuite(
        model_name=args.model,
        output_dir=args.output_dir,
        device_preference=args.device
    )
    
    # Get image paths
    if args.image_paths:
        image_paths = args.image_paths
    else:
        val_paths = get_validation_paths(TRAIN_DATA_PATH)
        test_paths = val_paths[:args.max_images] if args.max_images else val_paths[:30]
        image_paths = test_paths
    
    results = evaluator.comprehensive_layer_experimentation(
        image_paths=image_paths,
        max_individual_layers=args.max_layers,
        max_combinations=args.max_combinations,
        save_detailed_results=args.save_detailed
    )
    
    print(f"\n🎉 Comprehensive experimentation completed!")
    return results


def run_quick_test(args):
    """Run quick test with minimal data."""
    print(f"\n{'='*60}")
    print(f"Quick Test: {args.model}")
    print(f"{'='*60}")
    
    evaluator = XAIEvaluationSuite(
        model_name=args.model,
        output_dir=args.output_dir,
        device_preference=args.device
    )
    
    # Use minimal data
    val_paths = get_validation_paths(TRAIN_DATA_PATH)
    test_paths = val_paths[:args.max_images] if args.max_images else val_paths[:5]
    
    results = evaluator.run_full_evaluation(
        image_paths=test_paths,
        batch_size=min(args.batch_size, 4),
        save_results=True
    )
    
    print(f"\n🎉 Quick test completed!")
    print(f"Results for {args.model} (on {len(test_paths)} images):")
    print(f"  Insertion AUC: {results['insertion_auc']:.4f}")
    print(f"  Deletion AUC: {results['deletion_auc']:.4f}")
    print(f"  ROAD Mean: {results['road_mean']:.4f}")
    
    return results


def main():
    """Main function to run the evaluation suite."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"XAI Evaluation Suite - Configuration")
    print(f"{'='*80}")
    print(f"Evaluation Type: {args.eval_type}")
    if args.model:
        print(f"Model: {args.model}")
    elif args.models:
        print(f"Models: {', '.join(args.models)}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Output Directory: {args.output_dir}")
    
    if args.max_images:
        print(f"Max Images: {args.max_images}")
    if args.image_paths:
        print(f"Custom Images: {len(args.image_paths)} images")
    
    try:
        # Route to appropriate evaluation function
        if args.eval_type == 'single':
            results = run_single_evaluation(args)
        elif args.eval_type == 'comparison':
            if not args.models:
                parser.error("--models is required for comparison evaluation")
            results = run_comparison_evaluation(args)
        elif args.eval_type == 'layer-analysis':
            results = run_layer_analysis(args)
        elif args.eval_type == 'individual-layers':
            results = run_individual_layers_evaluation(args)
        elif args.eval_type == 'comprehensive':
            results = run_comprehensive_evaluation(args)
        elif args.eval_type == 'quick':
            results = run_quick_test(args)
        else:
            parser.error(f"Evaluation type '{args.eval_type}' is not implemented yet")
        
        print(f"\n{'='*80}")
        print(f"✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print(f"\n\n❌ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Error during evaluation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
