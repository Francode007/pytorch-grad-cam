# Remote Optimization Plan: Pyramid Fusion CAM

**Goal**: Optimize the execution speed and GPU utilization of the Pyramid Fusion CAM method on the remote A100 server.
**Current Status**: 
- **Symptoms**: Processing is slow (~9s/image) and GPU utilization is very low (~8%) on an A100.
- **Hypothesis**: The bottleneck is likely **CPU-bound** (e.g., Python `for` loops in `enhanced_cams`), **I/O-bound** (NFS latency), or **inefficient batching** in `OptimizedCamExtractor`.

## 1. Profiling (Immediate Next Step)
Run the `profile_performance.py` script on the remote server to identify the bottleneck.

```bash
python3 XAI_Enhancer_module/enhanced_combiner/profile_performance.py
```

### Script Content (`profile_performance.py`)
If the script is missing, recreate it with this content:

```python
import time
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path

# Add path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.enhanced_combiner.extractor_v2 import EnhancedExtractorV2
from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator
from XAI_Enhancer_module.utils.model_loader import ModelLoader

def profile_run():
    print("--- Performance Profiling ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Setup
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    model_cache_dir = str(project_root / "XAI_Enhancer_module" / "pytorch_models")
    imagenet_path = str(project_root / "XAI_Enhancer_module" / "imagenet_val_sample")
    
    # Load Model
    t0 = time.time()
    try:
        loader = ModelLoader(model_cache_dir)
        model = loader.load_pretrained_model("resnet50")
        model = model.to(device)
        model.eval()
        print(f"Model Load Time: {time.time() - t0:.4f}s")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Init Evaluator
    config = {
        "type": "pyramid",
        "k": 5,
        "k_percent": 0.2,
        "temp": 0.05,
        "beta": 0.3,
        "soft": True
    }
    
    try:
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
    except Exception as e:
        print(f"Failed to init evaluator: {e}")
        return
    
    # Get 1 Image
    image_paths, labels, _ = evaluator.get_imagenet_images(max_images=1)
    if not image_paths:
        print("No images found")
        return
        
    img_path = image_paths[0]
    label = labels[0]
    
    print(f"\nProfiling Image: {img_path}")
    
    # Profile CAM Extraction
    t_start = time.time()
    image_tensor, saliency_map = evaluator.extract_enhanced_cam(img_path, label)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_cam = time.time() - t_start
    print(f"CAM Extraction Time: {t_cam:.4f}s")
    
    # Profile Insertion/Deletion metrics
    # (These use the GPU heavily for masking)
    t_start = time.time()
    _, ins_auc = evaluator.compute_insertion_auc(image_tensor, saliency_map, label, step_size=224, batch_size=256)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"Insertion AUC Time: {time.time() - t_start:.4f}s (Inference: {ins_auc:.4f})")

if __name__ == "__main__":
    profile_run()
```

## 2. Diagnosis & Fixes

Based on the profiling output, here is how to proceed:

### Scenario A: CAM Extraction is Slow (>1s)
**Diagnosis**: The `EnhancedExtractorV2` or `OptimizedCamExtractor` is inefficient.
- **Likely Cause**: `compute_modified_outputs_batch` loops through layers. For ResNet50 (`layer_mode='all'`), there are ~53 layers. Even with batching, overhead might be high.
- **Fix**: 
    1. Increase `layer_batch_size` in `OptimizedCamExtractor` (default is 32, try 64 or 128 on A100).
    2. Vectorize the hook registration or use `torch.func` (functorch) if available for faster per-sample gradients.

### Scenario B: Metric Computation is Slow
**Diagnosis**: `compute_insertion_auc` / `compute_deletion_auc`.
- **Likely Cause**: Too many small forward passes.
- **Fix**: 
    1. Ensure `gpu_batch_size` (for metrics) is large (try 512 or 1024 on A100).
    2. Increase `step_size`. `step_size=224` (default precision) involves thousands of steps. `step_size=1120` is ~5x faster.

### Scenario C: Model Loading / Data Loading is Slow
**Diagnosis**: NFS Latency.
- **Likely Cause**: Reading files from network storage.
- **Fix**: 
    1. Use `IMAGENET_Proper_val_set` (local SSD) if available, instead of NFS.
    2. Increase `num_workers` in `DataLoader` (currently 0 or 4).

## 3. Required Files
Ensure these files are present and updated on the remote server:
- `XAI_Enhancer_module/enhanced_combiner/run_experiment.py` (Updated with `layer_mode` fix and defaults)
- `XAI_Enhancer_module/enhanced_combiner/calibrate.py` (Updated model path)
- `XAI_Enhancer_module/evaluator/imagenet_proper_auc_evaluator.py`
- `XAI_Enhancer_module/utils/optimized_cam_extractor.py`
