#!/usr/bin/env python3
"""
PF-CAM Experiment Runner — Standalone.

Run PF-CAM evaluations on ImageNet without any dependency on enhanced_cams/
or OptimizedCamExtractor.

Usage:
    # Quick 5-image test
    python XAI_Enhancer_module/pf_cam/run_experiment.py \
        --model resnet50 --count 5 --device cpu --log-weights

    # Full evaluation
    python XAI_Enhancer_module/pf_cam/run_experiment.py \
        --model resnet50 --count 500 --device cuda \
        --beta 0.3 --k-percent 0.1 --temp 0.05 \
        --norm-strategy gradient_weighted --log-weights

    # Compare with standard methods
    python XAI_Enhancer_module/pf_cam/run_experiment.py \
        --model resnet50 --count 50 --compare-standard
"""

import argparse
import sys
import os
import time
import pandas as pd
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from XAI_Enhancer_module.pf_cam.extractor import PFCamExtractor
from XAI_Enhancer_module.pf_cam.normalization import NormStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description="PF-CAM (standalone) experiment runner"
    )
    # Model
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101",
                                 "densenet121", "efficientnet_b0"],
                        help="Pre-trained model")
    parser.add_argument("--model-cache-dir", type=str, default="../pytorch_models/",
                        help="Directory for cached model weights")
    # Dataset
    parser.add_argument("--imagenet-path", type=str,
                        default=str(project_root / "imagenet_val_sample"),
                        help="Path to ImageNet validation images")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of images to evaluate")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for batch processing")
    parser.add_argument("--end", type=int, default=None,
                        help="End index for batch processing")
    # PF-CAM hyperparameters
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Soft gating β (0=suppress shallow, 1=pass all)")
    parser.add_argument("--k-percent", type=float, default=0.1,
                        help="Top-K percent per stage")
    parser.add_argument("--k-min", type=int, default=2,
                        help="Minimum K layers per stage")
    parser.add_argument("--temp", type=float, default=0.05,
                        help="Temperature for softmax sharpening")
    # Normalization
    parser.add_argument("--norm-strategy", type=str, default="gradient_weighted",
                        choices=[s.value for s in NormStrategy],
                        help="Activation normalization strategy")
    # Layer mode
    parser.add_argument("--layer-mode", type=str, default="all",
                        choices=["all", "last_5", "last"],
                        help="Which conv layers to use (always 'all' for pyramid)")
    # Logging
    parser.add_argument("--log-weights", action="store_true",
                        help="Log per-image layer/stage weights")
    parser.add_argument("--output-dir", type=str, default="pf_cam_results",
                        help="Output directory")
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device preference")
    # Saliency sharpening (optional)
    parser.add_argument("--sharpen-gamma", type=float, default=1.0,
                        help="Power-law sharpening γ (>1 = sharper, 1.0 = disabled)")
    # Comparison
    parser.add_argument("--compare-standard", action="store_true",
                        help="Also run standard GradCAM, GradCAM++, HiResCAM for comparison")
    # Step size for evaluation
    parser.add_argument("--step-size", type=int, default=224,
                        help="Step size for insertion/deletion evaluation")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation inference")
    return parser.parse_args()


def main():
    args = parse_args()

    # Force layer_mode='all' for pyramid
    if args.layer_mode != "all":
        print(f"⚠️  PF-CAM requires all layers. Overriding layer_mode='{args.layer_mode}' → 'all'")
        args.layer_mode = "all"

    os.makedirs(args.output_dir, exist_ok=True)

    # Build aggregation config
    agg_config = {
        "type": "pyramid",
        "beta": args.beta,
        "k_percent": args.k_percent,
        "k_min": args.k_min,
        "temp": args.temp,
    }

    print("=" * 70)
    print("PF-CAM Standalone Experiment Runner")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Norm strategy:  {args.norm_strategy}")
    print(f"  Aggregation:    pyramid (β={args.beta}, k%={args.k_percent}, T={args.temp})")
    print(f"  Sharpen γ:      {args.sharpen_gamma}")
    print(f"  Device:         {args.device}")
    print(f"  Images:         {args.count} (start={args.start})")
    print(f"  Log weights:    {args.log_weights}")
    print(f"  Output dir:     {args.output_dir}")
    print("=" * 70)

    # Import evaluator
    from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import (
        ImageNetProperAUCEvaluator,
    )

    # Create evaluator with PFCamExtractor
    evaluator = ImageNetProperAUCEvaluator(
        model_name=args.model,
        imagenet_path=args.imagenet_path,
        device_preference=args.device,
        layer_mode=args.layer_mode,
        model_cache_dir=args.model_cache_dir,
        extractor_cls=PFCamExtractor,
        extractor_kwargs={
            "aggregation_config": agg_config,
            "norm_strategy": args.norm_strategy,
            "log_weights": args.log_weights,
            "weight_log_dir": args.output_dir,
            "sharpen_gamma": args.sharpen_gamma,
        },
    )

    # Run PF-CAM evaluation
    start_time = time.time()
    pf_results = evaluator.evaluate_enhanced_cam(
        max_images=args.count,
        step_size=args.step_size,
        verbose=False,
        start_index=args.start,
        end_index=args.end,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("PF-CAM Results")
    print("=" * 70)
    for key, value in pf_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/max(args.count,1):.1f}s/image)")

    # Save weight logs
    if args.log_weights and evaluator.enhanced_cam_extractor is not None:
        logger = evaluator.enhanced_cam_extractor.get_weight_logger()
        if logger is not None:
            logger.save()

            # Print summary
            summary = logger.get_summary()
            if summary:
                print(f"\n  Weight log: {summary['num_images']} images, "
                      f"{summary['num_layers']} layers, "
                      f"{summary['num_stages']} stages")
                print(f"  Mean stage weights: {[f'{w:.3f}' for w in summary['mean_stage_weight']]}")

    # Save results CSV
    results_path = os.path.join(args.output_dir, "pf_cam_results.csv")
    pd.DataFrame([pf_results]).to_csv(results_path, index=False)
    print(f"\n  Results saved to {results_path}")

    # Compare with standard methods
    if args.compare_standard:
        _run_standard_comparison(evaluator, args)


def _run_standard_comparison(evaluator, args):
    """Run standard CAM methods for comparison."""
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM

    standard_methods = {
        "GradCAM": GradCAM,
        "GradCAM++": GradCAMPlusPlus,
        "HiResCAM": HiResCAM,
    }

    print("\n" + "=" * 70)
    print("Standard Method Comparison")
    print("=" * 70)

    all_results = []

    for method_name, method_cls in standard_methods.items():
        print(f"\n  Running {method_name}...")
        results = evaluator.evaluate_method(
            method_name,
            max_images=args.count,
            step_size=args.step_size,
            start_index=args.start,
            end_index=args.end,
            batch_size=args.batch_size,
        )
        results["method"] = method_name
        all_results.append(results)

        for key, value in results.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")

    # Save comparison
    comparison_path = os.path.join(args.output_dir, "comparison_results.csv")
    pd.DataFrame(all_results).to_csv(comparison_path, index=False)
    print(f"\n  Comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
