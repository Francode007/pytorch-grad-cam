
import torch
import numpy as np
import time
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path("/Users/franchisnsaikia/IBS_Research/pytorch-grad-cam")
sys.path.append(str(project_root))

from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator

def benchmark_metal():
    print(f"{'='*60}")
    print("🚀 XAI OPTIMIZATION BENCHMARK (METAL GPU)")
    print(f"{'='*60}")

    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ MPS (Metal) acceleration detected and enabled.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✅ CUDA acceleration detected.")
    else:
        device = "cpu"
        print("⚠️ No GPU detected. Running on CPU (this will be slow).")

    # 2. Initialize Model (ResNet50)
    print("\n⏳ Initializing ResNet50 evaluator...")
    try:
        # We use a dummy path since we'll use synthetic data for pure speed benchmarking
        evaluator = ImageNetProperAUCEvaluator(
            model_name='resnet50',
            imagenet_path='/tmp', 
            device_preference=device
        )
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return

    # 3. Create Dummy Data
    # 300 images of size 224x224
    N_IMAGES_TOTAL = 300
    # For the SLOW benchmark, we only run a small subset to avoid waiting hours
    N_IMAGES_SLOW = 3
    
    print("\ngenerating dummy data...")
    dummy_images = torch.randn(N_IMAGES_TOTAL, 3, 224, 224).to(evaluator.device)
    dummy_saliency = np.random.rand(224, 224)
    predicted_label = 0
    step_size = 50

    print(f"\n📊 BENCHMARK CONFIGURATION:")
    print(f"   Model: ResNet50")
    print(f"   Device: {evaluator.device}")
    print(f"   Step Size: {step_size}")
    print(f"   Total Images for Optimization Test: {N_IMAGES_TOTAL}")

    # =========================================================================
    # BASELINE (Previous Implementation Simulation)
    # =========================================================================
    print(f"\n🐢 RUNNING BASELINE (Batch Size = 1)")
    print(f"   (Simulating previous sequential behavior on {N_IMAGES_SLOW} images to estimate time)")
    
    start_time = time.time()
    for i in tqdm(range(N_IMAGES_SLOW), desc="Baseline (Slow)"):
        # We use batch_size=1 to simulate the sequential overhead
        _, _ = evaluator.compute_insertion_auc(
            dummy_images[i], dummy_saliency, predicted_label, 
            step_size=step_size, batch_size=1
        )
    end_time = time.time()
    
    avg_time_slow = (end_time - start_time) / N_IMAGES_SLOW
    projected_time_total = avg_time_slow * N_IMAGES_TOTAL
    
    print(f"   ⏱️  Avg Time per Image (Baseline): {avg_time_slow:.2f} seconds")
    print(f"   📉 Projected Time for {N_IMAGES_TOTAL} images: {projected_time_total/60:.2f} minutes")

    # =========================================================================
    # OPTIMIZED (New Implementation)
    # =========================================================================
    BATCH_SIZE = 64
    print(f"\n🐇 RUNNING OPTIMIZED (Batch Size = {BATCH_SIZE})")
    print(f"   (Running on FULL {N_IMAGES_TOTAL} images)")
    
    start_time = time.time()
    for i in tqdm(range(N_IMAGES_TOTAL), desc="Optimized (Fast)"):
        _, _ = evaluator.compute_insertion_auc(
            dummy_images[i], dummy_saliency, predicted_label, 
            step_size=step_size, batch_size=BATCH_SIZE
        )
    end_time = time.time()
    
    total_time_fast = end_time - start_time
    avg_time_fast = total_time_fast / N_IMAGES_TOTAL
    
    print(f"   ⏱️  Avg Time per Image (Optimized): {avg_time_fast:.2f} seconds")
    print(f"   📉 Total Time for {N_IMAGES_TOTAL} images: {total_time_fast:.2f} seconds")

    # =========================================================================
    # RESULTS
    # =========================================================================
    speedup = avg_time_slow / avg_time_fast
    print(f"\n{'-'*60}")
    print(f"🏆 RESULTS SUITE")
    print(f"{'-'*60}")
    print(f"Baseline (Est):   {projected_time_total/60:.2f} minutes")
    print(f"Optimized (Act):  {total_time_fast/60:.2f} minutes")
    print(f"SPEEDUP FACTOR:   {speedup:.1f}x FASTER")
    print(f"{'-'*60}")

if __name__ == "__main__":
    benchmark_metal()
