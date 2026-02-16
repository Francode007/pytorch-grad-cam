import time
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image

# Add path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2
from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator
from XAI_Enhancer_module.utils.model_loader import ModelLoader

def profile_run():
    print("--- Performance Profiling ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Setup
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    model_cache_dir = str(project_root / "XAI_Enhancer_module" / "pytorch_models")
    imagenet_path = str(project_root / "XAI_Enhancer_module" / "imagenet_val_sample")
    
    # Check paths
    if not os.path.exists(imagenet_path):
        print(f"Error: {imagenet_path} not found")
        return

    # Load Model
    t0 = time.time()
    loader = ModelLoader(model_cache_dir)
    model = loader.load_pretrained_model("resnet50")
    model = model.to(device)
    model.eval()
    print(f"Model Load Time: {time.time() - t0:.4f}s")
    
    # Init Evaluator
    config = {
        "type": "pyramid",
        "k": 5,
        "k_percent": 0.2,
        "temp": 0.05,
        "beta": 0.3,
        "soft": True
    }
    
    evaluator = ImageNetProperAUCEvaluator(
        model_name="resnet50",
        imagenet_path=imagenet_path,
        device_preference="cuda",
        enhanced_cam_method="GradCAMEnhanced",
        model_cache_dir=model_cache_dir,
        extractor_cls=EnhancedExtractorV2,
        layer_mode="all",
        extractor_kwargs={'aggregation_config': config}
    )
    
    # Get 1 Image
    image_paths, labels, _ = evaluator.get_imagenet_images(max_images=1)
    if not image_paths:
        print("No images found")
        return
        
    img_path = image_paths[0]
    label = labels[0]
    
    print(f"\nProfiling Image: {img_path}")
    
    # 2. Profile CAM Extraction
    t_start = time.time()
    image_tensor, saliency_map = evaluator.extract_enhanced_cam(img_path, label)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_cam = time.time() - t_start
    print(f"CAM Extraction Time: {t_cam:.4f}s")
    
    # 3. Profile Insertion AUC
    print("\nProfiling Insertion AUC...")
    t_start = time.time()
    # Run 1: 50 steps (step_size=1000 approx)
    print("\nTest 1: 50 steps")
    t_start = time.time()
    _, _ = evaluator.compute_insertion_auc(image_tensor, saliency_map, label, step_size=1000, batch_size=2048)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"Time (50 steps): {time.time() - t_start:.4f}s")
    
    # Run 2: 224 steps (step_size=224)
    print("\nTest 2: 224 steps")
    t_start = time.time()
    _, _ = evaluator.compute_insertion_auc(image_tensor, saliency_map, label, step_size=224, batch_size=2048)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"Time (224 steps): {time.time() - t_start:.4f}s")
    
    # Run 3: 1000 steps (step_size=50)
    print("\nTest 3: 1000 steps")
    t_start = time.time()
    _, ins_auc = evaluator.compute_insertion_auc(image_tensor, saliency_map, label, step_size=50, batch_size=2048)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"Time (1000 steps): {time.time() - t_start:.4f}s")
    
    # 4. Profile Deletion AUC
    print("\nProfiling Deletion AUC...")
    t_start = time.time()
    _, del_auc = evaluator.compute_deletion_auc(image_tensor, saliency_map, label, step_size=224, batch_size=256)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_del = time.time() - t_start
    print(f"Deletion AUC Time: {t_del:.4f}s (Score: {del_auc:.4f})")

    # 5. Profile ROAD
    print("\nProfiling ROAD...")
    t_start = time.time()
    evaluator.evaluate_road(image_tensor, saliency_map, label)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_road = time.time() - t_start
    print(f"ROAD Time: {t_road:.4f}s")
    
    print(f"\nTotal Processing Time: {t_cam + t_del + t_road:.4f}s (excluding insertion tests)")

    print("--- Raw Model Benchmark ---")
    benchmark_data = torch.randn(256, 3, 224, 224).to(device)
    # Warmup
    _ = model(benchmark_data)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    t_bench = time.time()
    for _ in range(10):
        _ = model(benchmark_data)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    print(f"Raw ResNet50 Speed (batch 256): {(time.time() - t_bench)/10:.4f}s per batch")
    print(f"Throughput: {256 * 10 / (time.time() - t_bench):.2f} img/s")

if __name__ == "__main__":
    profile_run()
