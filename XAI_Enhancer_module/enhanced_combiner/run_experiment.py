
import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import getpass

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2
from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator
from XAI_Enhancer_module.utils.notification_utils import send_email_notification
from huggingface_hub import login

def setup_credentials():
    print("\n--- Credential Setup ---")
    hf_token = os.environ.get("HF_TOKEN")
    
    if hf_token:
        print("Found HF_TOKEN in environment, logging in...")
        login(token=hf_token, add_to_git_credential=False)
    else:
        print("Please enter your Hugging Face Access Token (or press Enter if already logged in/not needed):")
        try:
             login(add_to_git_credential=False)
        except Exception as e:
             print(f"Login skipped/failed: {e}")

    email_to = "fnsaikia@gmail.com"
    email_sender = "fnsaikia@gmail.com"
    print(f"\nUsing default email recipient/sender: {email_to}")
    email_password = input("Enter Email Password (leave blank to skip email notifications): ").strip()
    return email_to, email_sender, email_password

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced XAI Experiments (Optimized)")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    parser.add_argument("--method", type=str, default="stagewise", 
                        choices=["standard", "stagewise", "topk", "temp", "pyramid"],
                        help="Aggregation method")
    parser.add_argument("--k", type=int, default=5, help="Top-K parameter")
    parser.add_argument("--k-percent", type=float, default=0.2, help="Top-K Percent for Pyramid method")
    parser.add_argument("--temp", type=float, default=0.05, help="Temperature parameter")
    parser.add_argument("--images-path", type=str, default="imagenet_val_sample", help="Path to images")
    parser.add_argument("--count", type=int, default=500, help="Total number of images to process")
    parser.add_argument("--batch-size", type=int, default=1000, help="Reporting/Restart chunk size")
    parser.add_argument("--gpu-batch-size", type=int, default=64, help="GPU Batch Size for metrics")
    parser.add_argument("--output-dir", type=str, default="enhanced_results", help="Output directory")
    parser.add_argument("--base-cam", type=str, default="GradCAM",
                        choices=["GradCAM", "GradCAM++", "HiResCAM", "ScoreCAM", "AblationCAM"],
                        help="Base CAM method to use for enhancement")
    parser.add_argument("--compare", action="store_true", help="Compare with standard methods")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on (cuda/cpu)")
    
    parser.add_argument("--layer-mode", type=str, default="last", choices=["last", "last_5", "all"],
                        help="Layer selection mode for CAM")
    parser.add_argument("--step-size", type=int, default=224, help="Step size for AUC evaluation")
    parser.add_argument("--beta", type=float, default=0.3, help="Soft Gating Beta for Pyramid method")
    
    args = parser.parse_args()
    
    # Credentials
    email_to, email_sender, email_password = setup_credentials()

    # Enforce layer_mode='all' for pyramid method if default 'last' is still set
    # Pyramid Fusion requires multiple layers to work.
    if args.method == "pyramid" and args.layer_mode == "last":
        print("INFO: 'pyramid' method requires multiple layers. Switching --layer-mode from 'last' to 'all'.")
        args.layer_mode = "all"
    
    # Map base-cam to enhanced class name
    base_cam_map = {
        "GradCAM": "GradCAMEnhanced",
        "GradCAM++": "GradCAMPlusPlusEnhanced",
        "HiResCAM": "HiResCAMEnhanced",
        "ScoreCAM": "ScoreCAMEnhanced",
        "AblationCAM": "AblationCAMEnhanced"
    }
    enhanced_cam_name = base_cam_map[args.base_cam]
    
    # Config
    metrics_config = {
        "type": args.method,
        "k": args.k,
        "k_percent": args.k_percent,
        "temp": args.temp,
        "beta": args.beta,
        "soft": True
    }
    
    print(f"Running Experiment: {args.model} | Method: {args.method} ({args.base_cam}) | Layers: {args.layer_mode} | Config: {metrics_config}")
    
    # Calculate paths relative to script location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    # Model Cache Directory (Project Root / pytorch_models)
    model_cache_dir = str(project_root / "pytorch_models")
    
    # ImageNet Path (Project Root / XAI_Enhancer_module / imagenet_val_sample)
    # Allow override via args, but default relative to project root if simple name
    if args.images_path == "imagenet_val_sample":
         imagenet_path = str(project_root / "XAI_Enhancer_module" / "imagenet_val_sample")
    else:
         imagenet_path = args.images_path
    
    # Path Validation
    if not os.path.exists(imagenet_path):
        print(f"Error: ImageNet path not found: {imagenet_path}")
        sys.exit(1)
        
    if not os.path.isdir(imagenet_path):
        print(f"Error: ImageNet path must be a directory: {imagenet_path}")
        sys.exit(1)

    # Check if directory is empty
    if not os.listdir(imagenet_path):
        print(f"Error: ImageNet directory is empty: {imagenet_path}")
        sys.exit(1)

    print(f"Model Cache Directory: {model_cache_dir}")
    print(f"ImageNet Path: {imagenet_path}")
    
    # Initialize Evaluator with Custom Extractor
    evaluator = ImageNetProperAUCEvaluator(
        model_name=args.model,
        imagenet_path=imagenet_path,
        device_preference=args.device,
        enhanced_cam_method=enhanced_cam_name,
        model_cache_dir=model_cache_dir,
        extractor_cls=EnhancedExtractorV2,
        layer_mode=args.layer_mode,  # Pass layer mode
        extractor_kwargs={'aggregation_config': metrics_config}
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_csv_path = os.path.join(args.output_dir, "comparison_report.csv")
    
    # Data Container
    all_results = []
    
    # Check if we are resuming (optional functionality, but good for large runs)
    # For now, we overwrite or append if careful.
    
    # CHUNKING LOOP
    for start_idx in range(0, args.count, args.batch_size):
        end_idx = min(start_idx + args.batch_size, args.count)
        print(f"\n{'='*20} Processing Chunk: {start_idx} to {end_idx} {'='*20}")
        
        chunk_results = []
        
        # 1. Enhanced Method
        print(f"--- Evaluating {args.method} (Mode: {args.layer_mode}) ---")
        enhanced_res = evaluator.evaluate_enhanced_cam(
            start_index=start_idx,
            end_index=end_idx,
            max_images=-1, # Controlled by start/end index
            batch_size=args.gpu_batch_size, # GPU batch size for metrics
            verbose=False,
            step_size=args.step_size # Faster evaluation
        )
        
        chunk_results.append({
            'Method': args.method,
            'Insertion_Mean': enhanced_res['insertion_auc_mean'],
            'Insertion_Std': enhanced_res['insertion_auc_std'],
            'Deletion_Mean': enhanced_res['deletion_auc_mean'],
            'Deletion_Std': enhanced_res['deletion_auc_std'],
            'ROAD_Mean': enhanced_res['road_mean'],
            'ROAD_Std': enhanced_res['road_std'],
            'Images_Evaluated': enhanced_res['num_images']
        })
        
        # 2. Standard Methods (Compare)
        if args.compare:
            print(f"--- Evaluating Standard Methods (Mode: last) ---")
            standard_methods = ["GradCAM", "GradCAM++", "HiResCAM"]
            
            # Switch to 'last' layer mode for standard methods (fair comparison)
            original_layer_mode = evaluator.layer_mode
            if evaluator.layer_mode != "last":
                evaluator.layer_mode = "last"
                evaluator.conv_layers = evaluator._get_enhanced_cam_layers("last")
                evaluator.enhanced_cam_extractor = None # Force re-init using new layers
            
            for method in standard_methods:
                print(f"--- Evaluating Standard {method} ---")
                
                # Hack: Update the evaluator's method to point to the standard method wrapper
                std_cam_name = base_cam_map.get(method, method + "Enhanced")
                evaluator.enhanced_cam_method = std_cam_name
                
                # Update extractor kwargs to be "standard" (no special config)
                evaluator.extractor_kwargs = {'aggregation_config': {"type": "standard"}}
                
                # Reset extractor instance to force re-initialization
                evaluator.enhanced_cam_extractor = None
                
                std_res = evaluator.evaluate_enhanced_cam(
                    start_index=start_idx,
                    end_index=end_idx,
                    max_images=-1,
                    batch_size=args.gpu_batch_size,
                    verbose=False,
                    step_size=args.step_size
                )
                
                chunk_results.append({
                    'Method': method,
                    'Insertion_Mean': std_res['insertion_auc_mean'],
                    'Insertion_Std': std_res['insertion_auc_std'],
                    'Deletion_Mean': std_res['deletion_auc_mean'],
                    'Deletion_Std': std_res['deletion_auc_std'],
                    'ROAD_Mean': std_res['road_mean'],
                    'ROAD_Std': std_res['road_std'],
                    'Images_Evaluated': std_res['num_images']
                })
                
            # Restore original configuration
            evaluator.enhanced_cam_method = enhanced_cam_name
            evaluator.extractor_kwargs = {'aggregation_config': metrics_config}
            # Restore layer mode if changed
            if original_layer_mode != "last":
                evaluator.layer_mode = original_layer_mode
                evaluator.conv_layers = evaluator._get_enhanced_cam_layers(original_layer_mode)
            evaluator.enhanced_cam_extractor = None
        
        # Append to CSV
        df_chunk = pd.DataFrame(chunk_results)
        if os.path.exists(results_csv_path) and start_idx > 0:
            df_chunk.to_csv(results_csv_path, mode='a', header=False, index=False)
        else:
            df_chunk.to_csv(results_csv_path, index=False)
            
        all_results.extend(chunk_results)
        
        # Email Notification
        if email_password:
            subject = f"Experiment Progress: {args.model} | Chunk {start_idx}-{end_idx}"
            body = f"Completed chunk {start_idx} to {end_idx}.\n\nResults:\n{df_chunk.to_string(index=False)}"
            send_email_notification(email_to, subject, body, email_sender, email_password)
            
    # Final Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    final_df = pd.DataFrame(all_results)
    print(final_df.to_string(index=False))
    
    # Final Email with Full CSV
    if email_password and not final_df.empty:
        subject = f"Experiment Complete: {args.model} | {args.method} vs Standard"
        body = f"Experiment completed for {len(final_df)} entries.\n\nSummary:\n{final_df.groupby('Method')[['Insertion_Mean', 'Deletion_Mean', 'ROAD_Mean']].mean().to_string()}\n\nFull CSV Data:\n{final_df.to_csv(index=False)}"
        try:
            send_email_notification(email_to, subject, body, email_sender, email_password)
            print("Final results email sent.")
        except Exception as e:
            print(f"Failed to send final email: {e}")

if __name__ == "__main__":
    main()
