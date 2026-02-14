
import argparse
import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pytorch_grad_cam.utils.image import show_cam_on_image
from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2
from XAI_Enhancer_module.utils.imagenet_model_utils import load_pretrained_imagenet_model, get_model_target_layers
from torchvision import transforms
from PIL import Image

def get_image_tensor(image_path, device):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return img.resize((224, 224)), tensor

def run_visualization(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    print(f"Loading model: {args.model}")
    model = load_pretrained_imagenet_model(args.model, device)
    
    # Get target layers (all layers for pyramid)
    # Start with all layers, let extractor filter if needed?
    # EnhancedExtractorV2 handles `conv_layers` list.
    # For standard cams, we might need just the last one, but Extractor usually handles list.
    # We pass ALL layers to Extractor, and let it decide?
    # No, EnhancedExtractor with "standard" config might expect specific layers?
    # Actually EnhancedExtractorV2 wrapper handles it.
    # BUT `get_model_target_layers` has `all_layers` flag.
    # For Pyramid/Stagewise we need `all_layers=True`.
    # For Standard CAMs, usually just last.
    # WE SHOULD ALWAYS PASS ALL LAYERS to EnhancedExtractorV2, because it likely filters or uses them based on method.
    target_layers = get_model_target_layers(model, args.model, all_layers=True)
    
    model.eval()
    
    # 2. Setup Extractors
    extractors = {}
    
    # Standard Methods to compare
    standard_methods = [m for m in args.methods if m != "pyramid"]
    # Enhanced Method (PF-CAM)
    use_pyramid = "pyramid" in args.methods
    
    # We use EnhancedExtractorV2 for everything as it wraps underlying logic
    # specific configs for each
    
    # 3. Process Images
    image_paths = args.image_paths
    if not image_paths and args.count > 0:
        # Pick random images from ImageNet Val if path not provided
        # This is complex to robustly do without proper dataset, so we expect paths or directory
        if os.path.isdir(args.images_dir):
             all_imgs = [os.path.join(args.images_dir, f) for f in os.listdir(args.images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
             image_paths = all_imgs[:args.count]
        else:
             print("Please provide specific --image-paths or a valid --images-dir")
             return

    os.makedirs(args.output_dir, exist_ok=True)
    
    for img_path in image_paths:
        print(f"Processing {img_path}...")
        try:
            pil_img, input_tensor = get_image_tensor(img_path, device)
            rgb_img = np.float32(pil_img) / 255.0
            
            # Predict
            outputs = model(input_tensor)
            predicted_label = outputs.argmax(dim=1).item()
            
            results = []
            method_names = []
            
            # Run requested methods
            for method in args.methods:
                print(f"  Running {method}...", end="\r")
                
                config = {"type": "standard"} # Default
                cam_method_name = "GradCAMEnhanced" # Default base
                
                if method == "pyramid":
                    config = {
                        "type": "pyramid",
                        "beta": args.beta,
                        "k_percent": 0.15, 
                        "temp": 0.1
                    }
                elif method in ["GradCAM", "GradCAM++", "HiResCAM", "ScoreCAM"]:
                   cam_method_name = method + "Enhanced" # Mapping to our classes
                
                extractor = EnhancedExtractorV2(
                    model=model,
                    model_name=args.model,
                    conv_layers=target_layers,
                    cam_method=cam_method_name,
                    device_preference=device,
                    aggregation_config=config
                )
                
                _, cam = extractor.extract_saliency_map(img_path, predicted_label, use_cache=False)
                
                if cam is None:
                    continue
                
                # Ensure cam is on CPU and correct shape
                cam_np = cam.cpu().numpy()
                print(f"  CAM shape: {cam_np.shape}, min: {cam_np.min():.3f}, max: {cam_np.max():.3f}")
                
                # Squeeze if needed
                if cam_np.ndim == 3 and cam_np.shape[0] == 1:
                    cam_np = cam_np.squeeze(0)
                
                # Visualization
                visualization = show_cam_on_image(rgb_img, cam_np, use_rgb=True)
                results.append(visualization)
                method_names.append(method)
                
            # Stitch images side-by-side
            # Add labels
            labeled_results = []
            for name, res in zip(method_names, results):
                # Add text label
                res_labeled = res.copy()
                cv2.putText(res_labeled, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                labeled_results.append(res_labeled)
                
            final_image = np.hstack([np.uint8(rgb_img * 255)] + labeled_results)
            
            save_name = os.path.basename(img_path).replace(".", "_compare.")
            save_path = os.path.join(args.output_dir, save_name)
            Image.fromarray(final_image).save(save_path)
            print(f"  Saved to {save_path}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Visualize PF-CAM vs Standard")
    parser.add_argument("--image-paths", nargs="*", help="List of image paths")
    parser.add_argument("--images-dir", type=str, default="imagenet_val_sample", help="Directory of images")
    parser.add_argument("--count", type=int, default=5, help="Number of images if using dir")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--methods", nargs="+", default=["pyramid", "GradCAM", "HiResCAM"])
    parser.add_argument("--output-dir", type=str, default="visualization_results")
    parser.add_argument("--beta", type=float, default=0.4, help="Beta for Soft Gating")
    
    args = parser.parse_args()
    run_visualization(args)

if __name__ == "__main__":
    main()
