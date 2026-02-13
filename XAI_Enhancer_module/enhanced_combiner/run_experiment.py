
import argparse
import sys
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.download_models import download_all_models
from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_imagenet_sample_path(base_path, count=500):
    """Finds or creates a list of image paths."""
    image_paths = []
    # Assumes hierarchical structure: val/n01440764/ILSVRC2012_val_00000293.JPEG
    # or flat structure depending on how `run_xai_enhancer.py` created it.
    # run_xai_enhancer creates: base_path / synset_id / val_X.JPEG
    
    base = Path(base_path)
    all_images = list(base.rglob("*.JPEG")) + list(base.rglob("*.jpg")) + list(base.rglob("*.png"))
    
    if len(all_images) == 0:
        print(f"No images found in {base_path}")
        return []
        
    # Sort for determinism
    all_images.sort()
    
    return all_images[:count]

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced XAI Experiments")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    parser.add_argument("--method", type=str, default="stagewise", 
                        choices=["standard", "stagewise", "topk", "temp", "pyramid"],
                        help="Aggregation method")
    parser.add_argument("--k", type=int, default=5, help="Top-K parameter")
    parser.add_argument("--k-percent", type=float, default=0.15, help="Top-K Percent for Pyramid method")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature parameter")
    parser.add_argument("--images-path", type=str, default="imagenet_val_sample", help="Path to images")
    parser.add_argument("--count", type=int, default=500, help="Number of images to process")
    parser.add_argument("--output-dir", type=str, default="enhanced_results", help="Output directory")
    parser.add_argument("--base-cam", type=str, default="GradCAM",
                        choices=["GradCAM", "GradCAM++", "HiResCAM", "ScoreCAM", "AblationCAM"],
                        help="Base CAM method to use for enhancement")
    parser.add_argument("--compare", action="store_true", help="Compare with standard methods")
    
    args = parser.parse_args()
    
    # Map base-cam to enhanced class name
    base_cam_map = {
        "GradCAM": "GradCAMEnhanced",
        "GradCAM++": "GradCAMPlusPlusEnhanced",
        "HiResCAM": "HiResCAMEnhanced",
        "ScoreCAM": "ScoreCAMEnhanced",
        "AblationCAM": "AblationCAMEnhanced"
    }
    enhanced_cam_name = base_cam_map[args.base_cam]
    
    # 1. Setup Config
    metrics_config = {
        "type": args.method,
        "k": args.k,
        "k_percent": args.k_percent,
        "temp": args.temp,
        "soft": True # Default to soft for now
    }
    
    print(f"Running Experiment: {args.model} | Method: {args.method} ({args.base_cam}) | Config: {metrics_config}")

    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Assuming load_model handles everything
    # We might need to look at `download_models.py` to see how `load_model` works or if we need `download_all_models`
    # In `download_models.py` there is `load_model_and_layers`? No, let's check `download_models.py` imports...
    # Actually `run_xai_enhancer.py` uses `download_all_models`.
    # Let's import `get_model` from `imagenet_model_utils` if it exists, or just use `torchvision`.
    
    from XAI_Enhancer_module.utils.imagenet_model_utils import load_pretrained_imagenet_model, get_model_target_layers
    
    # Load model and layers using available utilities
    model = load_pretrained_imagenet_model(args.model, device)
    # USE ALL LAYERS for Enhanced Method
    target_layers = get_model_target_layers(model, args.model, all_layers=True)
    print(f"Enhanced Extractor initialized with {len(target_layers)} layers/stages.")
    
    # 3. Setup Extractor
    extractor = EnhancedExtractorV2(
        model=model,
        model_name=args.model,
        conv_layers=target_layers, # all layers
        cam_method=enhanced_cam_name,
        device_preference=device,
        aggregation_config=metrics_config
    )
    
    # 4. Get Images
    image_paths = get_imagenet_sample_path(args.images_path, args.count)
    
    # Initialize metrics storage
    # Structure: {method_name: {'insertion': [], 'deletion': [], 'road': []}}
    methods_to_run = [args.method]
    if args.compare:
        methods_to_run.extend(["GradCAM", "GradCAM++", "HiResCAM"])
        
    all_metrics = {m: {'insertion': [], 'deletion': [], 'road': []} for m in methods_to_run}

    print(f"Found {len(image_paths)} images.")
    print(f"Methods to evaluate: {methods_to_run}")
    
    # 5. Run Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, img_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        try:
            # Predict Label (Top-1)
            img_tensor_in = extractor.preprocess_image(cv2.imread(str(img_path)))
            with torch.no_grad():
                out = model(img_tensor_in.to(device))
                pred_idx = torch.argmax(out, dim=1).item()
            
            # Initialize evaluator lazily (once)
            if i == 0:
                from XAI_Enhancer_module.evaluator.proper_auc_evaluation import ProperAUCEvaluator
                evaluator = ProperAUCEvaluator(model_name=args.model, device_preference=device, model=model)
                print("Initialized ProperAUCEvaluator for metrics...")

            # --- Evaluate Each Method ---
            for method in methods_to_run:
                # 1. Extract CAM
                if method == args.method:
                    # Enhanced Method
                    _, cam = extractor.extract_saliency_map(str(img_path), pred_idx)
                    
                    # Enhanced extractor returns None on failure
                    if cam is None:
                        continue
                    
                    # Ensure cam is float32 numpy (H, W)
                    cam_np = cam.cpu().numpy()
                    if cam_np.ndim == 3 and cam_np.shape[0] == 1:
                        cam_np = cam_np.squeeze(0)
                        
                else:
                    # Standard Method using Evaluator's extractor
                    # extract_cam returns (image_tensor, cam_numpy)
                    # cam_numpy is (H, W) or (1, H, W)
                    _, cam_np = evaluator.extract_cam(str(img_path), pred_idx, cam_method_name=method)
                    # No need for .cpu().numpy() as it is already numpy

                # 2. Save Visualization (only for enhanced or if debug needed, maybe all?)
                # Let's save all for comparison
                orig_img = cv2.imread(str(img_path))
                orig_img = cv2.resize(orig_img, (224, 224))
                orig_img = orig_img.astype(np.float32) / 255.0
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                
                visualization = show_cam_on_image(orig_img, cam_np, use_rgb=True)
                save_name = f"{i:03d}_{Path(img_path).stem}_{method}.jpg"
                cv2.imwrite(os.path.join(args.output_dir, save_name), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

                # 3. Compute Metrics
                try:
                    import gc
                    img_tensor_for_eval = img_tensor_in.unsqueeze(0) if img_tensor_in.dim() == 3 else img_tensor_in
                    
                    # Optimize: Use larger step_size for faster evaluation (e.g., 2240 pixels per step ~ 22 steps instead of 224)
                    _, ins_auc = evaluator.compute_insertion_auc(img_tensor_for_eval, cam_np, pred_idx, step_size=2240)
                    _, del_auc = evaluator.compute_deletion_auc(img_tensor_for_eval, cam_np, pred_idx, step_size=2240)
                    road_score = evaluator.evaluate_road(img_tensor_for_eval, cam_np, pred_idx)
                    
                    all_metrics[method]['insertion'].append(ins_auc)
                    all_metrics[method]['deletion'].append(del_auc)
                    all_metrics[method]['road'].append(road_score)
                    
                    # Force Cleanup
                    del img_tensor_for_eval
                    if method == args.method:
                         del cam
                    gc.collect()
                    
                except Exception as e:
                    tqdm.write(f"  Error computing metrics for {method}: {e}")

            # print metrics for enhanced method as progress
            if i % 1 == 0:
                 m = all_metrics[args.method]
                 if m['insertion']:
                    tqdm.write(f"  {args.method}: Ins={m['insertion'][-1]:.3f}, Del={m['deletion'][-1]:.3f}")

        except Exception as e:
            print(f"Failed on {img_path}: {e}")
            import traceback
            traceback.print_exc()

    # Print Summary Table
    print(f"\nExperiment Complete. Results saved to {args.output_dir}")
    print("\n" + "="*60)
    print(f"{'Method':<20} | {'Ins AUC':<12} | {'Del AUC':<12} | {'ROAD':<12}")
    print("-" * 60)
    
    summary_data = []
    
    for method in methods_to_run:
        m = all_metrics[method]
        if not m['insertion']:
            continue
            
        ins_mean = np.mean(m['insertion'])
        ins_std = np.std(m['insertion'])
        del_mean = np.mean(m['deletion'])
        del_std = np.std(m['deletion'])
        road_mean = np.mean(m['road'])
        road_std = np.std(m['road'])
        
        print(f"{method:<20} | {ins_mean:.4f}       | {del_mean:.4f}       | {road_mean:.4f}")
        
        summary_data.append({
            'Method': method,
            'Insertion_Mean': ins_mean, 'Insertion_Std': ins_std,
            'Deletion_Mean': del_mean, 'Deletion_Std': del_std,
            'ROAD_Mean': road_mean, 'ROAD_Std': road_std
        })
    print("-" * 60)
    
    # Save CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(args.output_dir, "comparison_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"Report saved to {csv_path}")

if __name__ == "__main__":
    main()
