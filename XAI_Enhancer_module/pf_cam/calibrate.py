#!/usr/bin/env python3
"""
PF-CAM Calibration (Grid Search) Script.

Performs a grid search over PF-CAM hyperparameters (beta, k_percent, temp)
to find the optimal configuration for a given dataset and model.
Optimized for long-running remote execution with result checkpointing.

Usage:
    python XAI_Enhancer_module/pf_cam/calibrate.py \
        --model resnet50 --count 50 --device cuda \
        --output-dir calibration_results
"""

import argparse
import sys
import os
import time
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from XAI_Enhancer_module.pf_cam.extractor import PFCamExtractor
from XAI_Enhancer_module.pf_cam.normalization import NormStrategy
from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import (
    ImageNetProperAUCEvaluator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PF-CAM Hyperparameter Grid Search")
    # Model & Data
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    parser.add_argument("--count", type=int, default=50, help="Images per trial")
    parser.add_argument("--start", type=int, default=0, help="Start image index")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--output-dir", type=str, default="pf_cam_calibration",
                        help="Output directory")

    # Grid Search Ranges
    parser.add_argument("--beta-list", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7],
                        help="List of beta values to test")
    parser.add_argument("--k-percent-list", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3],
                        help="List of k_percent values to test")
    parser.add_argument("--temp-list", type=float, nargs="+",
                        default=[0.05, 0.1, 0.5, 1.0],
                        help="List of temperature values to test")
    parser.add_argument("--norm-strategy", type=str, default="gradient_weighted",
                        help="Fixed normalization strategy for calibration")

    return parser.parse_args()


def run_single_trial(
    args,
    config: Dict,
    evaluator_cls,
    trial_idx: int,
    total_trials: int
) -> Dict:
    """Run a single configuration trial."""
    print(f"\n[Trial {trial_idx}/{total_trials}] Testing config: {config}")

    agg_config = {
        "type": "pyramid",
        "beta": config["beta"],
        "k_percent": config["k_percent"],
        "k_min": 2,
        "temp": config["temp"],
    }

    # Initialize evaluator for this config
    evaluator = evaluator_cls(
        model_name=args.model,
        device_preference=args.device,
        layer_mode="all",  # PF-CAM requires all layers
        extractor_cls=PFCamExtractor,
        extractor_kwargs={
            "aggregation_config": agg_config,
            "norm_strategy": args.norm_strategy,
            "log_weights": False,  # Disable logging for speed during calibration
        },
    )

    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate_enhanced_cam(
        max_images=args.count,
        step_size=50,  # Coarse step for speed
        verbose=False,
        start_index=args.start,
        batch_size=64,
    )
    elapsed = time.time() - start_time

    # Collect metrics
    trial_result = {
        "beta": config["beta"],
        "k_percent": config["k_percent"],
        "temp": config["temp"],
        "norm_strategy": args.norm_strategy,
        "insertion_auc": results.get("insertion_auc", 0.0),
        "deletion_auc": results.get("deletion_auc", 0.0),
        "road_combined_mean": results.get("road_combined_mean", 0.0),
        "time_sec": elapsed,
    }
    
    # Calculate composite score (higher is better)
    # Score = Ins AUC - Del AUC - ROAD (since lower ROAD is better)
    # Note: Traditional Ins-Del is Ins - Del. ROAD is separate.
    # Let's use Ins - Del as primary metric for now.
    trial_result["score_ins_del"] = trial_result["insertion_auc"] - trial_result["deletion_auc"]
    
    print(f"  Result: Ins={trial_result['insertion_auc']:.4f}, "
          f"Del={trial_result['deletion_auc']:.4f}, "
          f"ROAD={trial_result['road_combined_mean']:.4f} "
          f"-> Score={trial_result['score_ins_del']:.4f}")
    
    return trial_result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "calibration_results.csv")
    best_config_file = os.path.join(args.output_dir, "best_config.json")

    # Generate grid
    grid = list(itertools.product(
        args.beta_list,
        args.k_percent_list,
        args.temp_list
    ))
    configs = [
        {"beta": b, "k_percent": k, "temp": t} 
        for b, k, t in grid
    ]
    total_trials = len(configs)

    print("=" * 70)
    print("PF-CAM Calibration Grid Search")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Images per trial: {args.count}")
    print(f"  Total trials: {total_trials}")
    print(f"  Output: {results_file}")
    print("=" * 70)

    # Check for existing results to resume
    existing_results = []
    processed_configs = set()
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        existing_results = existing_df.to_dict("records")
        for res in existing_results:
            # Create a key for the processed config
            key = (res["beta"], res["k_percent"], res["temp"])
            processed_configs.add(key)
        print(f"  Resuming from {len(existing_results)} existing trials...")

    results = existing_results
    
    try:
        for idx, config in enumerate(configs, 1):
            key = (config["beta"], config["k_percent"], config["temp"])
            if key in processed_configs:
                continue

            trial_res = run_single_trial(
                args, config, ImageNetProperAUCEvaluator, idx, total_trials
            )
            results.append(trial_res)

            # Checkpoint results after each trial
            pd.DataFrame(results).to_csv(results_file, index=False)

    except KeyboardInterrupt:
        print("\n\ncalibration interrupted. Saving current progress...")
        pd.DataFrame(results).to_csv(results_file, index=False)
        sys.exit(0)

    # Analyze best result
    if results:
        df = pd.DataFrame(results)
        best_row = df.loc[df["score_ins_del"].idxmax()]
        
        print("\n" + "=" * 70)
        print("Calibration Completed - Best Configuration")
        print("=" * 70)
        print(f"  Beta:      {best_row['beta']}")
        print(f"  K-Percent: {best_row['k_percent']}")
        print(f"  Temp:      {best_row['temp']}")
        print(f"  Score:     {best_row['score_ins_del']:.4f} "
              f"(Ins: {best_row['insertion_auc']:.4f}, "
              f"Del: {best_row['deletion_auc']:.4f})")
        
        # Save best config
        best_config = {
            "beta": float(best_row["beta"]),
            "k_percent": float(best_row["k_percent"]),
            "temp": float(best_row["temp"]),
            "score": float(best_row["score_ins_del"])
        }
        with open(best_config_file, "w") as f:
            json.dump(best_config, f, indent=4)
        print(f"  Saved best config to {best_config_file}")

if __name__ == "__main__":
    main()
