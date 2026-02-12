
import argparse
import sys
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

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
                        choices=["standard", "stagewise", "topk", "temp"],
                        help="Aggregation method")
    parser.add_argument("--k", type=int, default=5, help="Top-K parameter")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature parameter")
    parser.add_argument("--images-path", type=str, default="imagenet_val_sample", help="Path to images")
    parser.add_argument("--count", type=int, default=10, help="Number of images to process")
    parser.add_argument("--output-dir", type=str, default="enhanced_results", help="Output directory")
    
    args = parser.parse_args()
    
    # 1. Setup Config
    metrics_config = {
        "type": args.method,
        "k": args.k,
        "temp": args.temp,
        "soft": True # Default to soft for now
    }
    
    print(f"Running Experiment: {args.model} | Method: {args.method} | Config: {metrics_config}")

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
    target_layers = get_model_target_layers(model, args.model)
    
    # 3. Setup Extractor
    extractor = EnhancedExtractorV2(
        model=model,
        model_name=args.model,
        conv_layers=target_layers, # all layers
        cam_method="GradCAMEnhanced", # or others
        device_preference=device,
        aggregation_config=metrics_config
    )
    
    # 4. Get Images
    image_paths = get_imagenet_sample_path(args.images_path, args.count)
    print(f"Found {len(image_paths)} images.")
    
    # 5. Run Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, img_path in tqdm(enumerate(image_paths)):
        try:
            # Predict Label First (or just use top-1)
            # Extractor needs a label.
            # Let's do a quick forward pass to get top-1.
            # Or assume we want the "True" label if available? 
            # For simplicity, let's explain the *predicted* class.
            
            # Use extractor helper for preprocessing just for prediction
            img_tensor_in = extractor.preprocess_image(cv2.imread(str(img_path)))
            with torch.no_grad():
                out = model(img_tensor_in.to(device))
                pred_idx = torch.argmax(out, dim=1).item()
            
            # Extract
            input_tensor, cam = extractor.extract_saliency_map(str(img_path), pred_idx)
            
            if cam is None:
                continue
                
            # Visualization
            # input_tensor is (3, H, W) normalized. Need to denormalize for vis?
            # show_cam_on_image expects (H, W, 3) float [0,1]
            rgb_img = input_tensor.permute(1, 2, 0).cpu().numpy()
            # If normalized with ImageNet mean/std, we need to undo.
            # But `OptimizedCamExtractor.preprocess_image` usually just divides by 255 and uses `transformations`.
            # If `transformations` includes Normalize, we have an issue.
            # `run_xai_enhancer` imports `transformations` from `model_utils`.
            # Let's assume rgb_img is roughly viewable or we just use original read image.
            
            # Better: Read original image again for Vis
            orig_img = cv2.imread(str(img_path))
            orig_img = cv2.resize(orig_img, (224, 224)) # assert 224
            orig_img = orig_img.astype(np.float32) / 255.0
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Ensure cam is on CPU and is a float32 numpy array
            cam_np = cam.cpu().numpy()
            
            # If cam has shape (1, H, W), squeeze it to (H, W)
            if cam_np.ndim == 3 and cam_np.shape[0] == 1:
                cam_np = cam_np.squeeze(0)
                
            visualization = show_cam_on_image(orig_img, cam_np, use_rgb=True)
            
            # Save
            save_name = f"{i:03d}_{Path(img_path).stem}_{args.method}.jpg"
            cv2.imwrite(os.path.join(args.output_dir, save_name), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Failed on {img_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
