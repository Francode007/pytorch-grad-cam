
import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2
from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator

def main():
    parser = argparse.ArgumentParser(description="Calibrate Pyramid Fusion Hyperparameters")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    parser.add_argument("--images-path", type=str, default="imagenet_val_sample", help="Path to images")
    parser.add_argument("--count", type=int, default=50, help="Number of images for calibration (keep small)")
    parser.add_argument("--gpu-batch-size", type=int, default=32, help="GPU Batch Size for metrics")
    parser.add_argument("--output-dir", type=str, default="calibration_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on")
    parser.add_argument("--step-size", type=int, default=1000, help="Step size for AUC valuation (larger = faster)")
    
    args = parser.parse_args()
    
    # Define Hyperparameter Grid
    # Focused on Pyramid Fusion keys
    # beta: 0.3, 0.4, 0.5, 0.6
    # k_percent: 0.1, 0.15, 0.2
    # temp: 0.05, 0.1, 0.2
    
    betas = [0.3, 0.4, 0.5, 0.6]
    k_percents = [0.1, 0.15, 0.2]
    temps = [0.05, 0.1, 0.2]
    
    combinations = list(itertools.product(betas, k_percents, temps))
    print(f"Total combinations to test: {len(combinations)}")
    
    # Setup Paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    model_cache_dir = str(project_root / "pytorch_models")
    
    if args.images_path == "imagenet_val_sample":
         imagenet_path = str(project_root / "XAI_Enhancer_module" / "imagenet_val_sample")
    else:
         imagenet_path = args.images_path

    if not os.path.exists(imagenet_path):
        print(f"Error: ImageNet path not found: {imagenet_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "calibration_grid_search.csv")
    
    results = []
    
    # Iterate through grid
    for idx, (beta, k_pct, temp) in enumerate(combinations):
        print(f"\n[{idx+1}/{len(combinations)}] Testing: beta={beta}, k_percent={k_pct}, temp={temp}")
        
        config = {
            "type": "pyramid",
            "beta": beta,
            "k_percent": k_pct,
            "temp": temp,
            "soft": True
        }
        
        # Initialize Evaluator
        # We re-init for cleanliness, though could optimize if extractor supports dynamic config
        # ImageNetProperAUCEvaluator re-loads model every time if init new.
        # BUT, since we only change config, maybe we can reuse? 
        # The evaluator takes 'extractor_kwargs' in init. 
        # So we must re-init or hack it.
        # Re-init is safer but slower (model load).
        # Optimization: Load model once outside? evaluator logic is tightly coupled.
        # Let's just re-init, model load is fast from cache/disk compared to evaluation?
        # Actually loading ResNet50 is fast.
        
        evaluator = ImageNetProperAUCEvaluator(
            model_name=args.model,
            imagenet_path=imagenet_path,
            device_preference=args.device,
            enhanced_cam_method="GradCAMEnhanced", # Using GradCAM as base
            model_cache_dir=model_cache_dir,
            extractor_cls=EnhancedExtractorV2,
            layer_mode="all", # Pyramid needs all layers
            extractor_kwargs={'aggregation_config': config}
        )
        
        # Run Evaluation on small subset
        # verbose=False to reduce noise
        metrics = evaluator.evaluate_enhanced_cam(
            max_images=args.count,
            step_size=args.step_size, # Fast step size for calibration
            batch_size=args.gpu_batch_size,
            verbose=False
        )
        
        # Store results
        res_entry = {
            "beta": beta,
            "k_percent": k_pct,
            "temp": temp,
            "Insertion_AUC": metrics['insertion_auc_mean'],
            "Deletion_AUC": metrics['deletion_auc_mean'],
            "ROAD": metrics['road_mean'],
            "Score": metrics['insertion_auc_mean'] - metrics['deletion_auc_mean'] # Simple combined score
        }
        results.append(res_entry)
        
        print(f"   -> Insertion: {res_entry['Insertion_AUC']:.4f} | Deletion: {res_entry['Deletion_AUC']:.4f} | ROAD: {res_entry['ROAD']:.4f}")
        
        # Save intermediate
        pd.DataFrame(results).to_csv(results_path, index=False)

    print("\n--- Calibration Complete ---")
    df = pd.DataFrame(results)
    df = df.sort_values(by="Insertion_AUC", ascending=False)
    print("Top 5 Configurations (by Insertion AUC):")
    print(df.head(5).to_string(index=False))
    
    best = df.iloc[0]
    print(f"\nBest Config: beta={best['beta']}, k_percent={best['k_percent']}, temp={best['temp']}")

if __name__ == "__main__":
    main()
