
import torch
import numpy as np
import time
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path("/Users/franchisnsaikia/IBS_Research/pytorch-grad-cam")
sys.path.append(str(project_root))

from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator

def regression_test():
    print(f"{'='*60}")
    print("🧪 XAI REGRESSION & ENHANCEMENT TEST")
    print(f"{'='*60}")
    
    # Configuration
    N_IMAGES = 30
    BATCH_SIZE = 64 # Optimized Batch Size
    STEP_SIZE = 224 # Faster evaluation
    
    # Initialize Evaluator
    print("\n⏳ Initializing Evalutor (ResNet50) - Layer Mode: ALL...")
    evaluator = ImageNetProperAUCEvaluator(
        model_name='resnet50',
        imagenet_path='/tmp', 
        device_preference='auto',
        layer_mode='all' 
    )
    
    
    # Load Real Data
    print(f"\n📦 Loading {N_IMAGES} real images from imagenet_val_sample_test...")
    from PIL import Image
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensors = []
    labels = []
    
    # Read mapping
    with open("test_mapping.txt", "r") as f:
        lines = f.readlines()[:N_IMAGES]
        
    for line in lines:
        filename, label = line.strip().split()
        img_path = Path("imagenet_val_sample_test") / filename
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img)
        image_tensors.append(img_t)
        labels.append(int(label))
        
    images = torch.stack(image_tensors).to(evaluator.device)
    saliency_maps = [] # Will generate using evaluator
    
    print("\n🔍 Generating Saliency Maps using evaluator.extract_enhanced_cam...")
    
    for i in tqdm(range(N_IMAGES)):
        # Reconstruct path from mapping lines we read earlier
        line = lines[i]
        filename, label = line.strip().split()
        img_path = str(Path("imagenet_val_sample_test") / filename)
        
        # Use evaluator to get map. It handles lazy init of extractor.
        # This returns (tensor, map). We use the map. 
        # The tensor might be slightly different normalized/resized? 
        # evaluator.transform uses Resize(256), CenterCrop(224).
        # Our manual transform used Resize(224). 
        # We should stick to what evaluator returns to be consistent with the map!
        
        img_tensor_from_eval, saliency_map = evaluator.extract_enhanced_cam(img_path, int(label))
        saliency_maps.append(saliency_map)
        
        # Update our images tensor to match exactly what generated the map
        # img_tensor_from_eval is [1, C, H, W]
        images[i] = img_tensor_from_eval.squeeze(0)
        
    saliency_maps = np.array(saliency_maps)
    
    results_legacy = []
    results_enhanced = []
    
    print(f"\n🚀 Running Test on {N_IMAGES} images (Batch Size: {BATCH_SIZE})...")
    
    start_time = time.time()
    
    # We process in a loop to mimic the evaluation suite
    for i in tqdm(range(N_IMAGES)):
        img = images[i]
        saliency = saliency_maps[i]
        predicted_label = labels[i] # Use ground truth for metric calculation

        
        # ---------------------------------------------------------
        # 1. EMULATE "BEFORE" (Legacy Metrics)
        # ---------------------------------------------------------
        # Legacy ROAD: Imputation=Black (0), Threshold=90 only
        # Legacy AUC: We use the current batched implementation but it should produce same values
        
        # We manually call evaluate_road with legacy params
        road_legacy_dict = evaluator.evaluate_road(
            img, saliency, predicted_label,
            thresholds=[90],
            imputation="black" # OLD METHOD
        )
        road_legacy_val = road_legacy_dict['road_90']
        
        # ---------------------------------------------------------
        # 2. "AFTER" (Enhanced Metrics)
        # ---------------------------------------------------------
        # New ROAD: Imputation=Blur, Thresholds=[20,40,60,80]
        # This is what _evaluate_saliency_map now calls by default
        
        metrics = evaluator._evaluate_saliency_map(
            img, saliency, predicted_label, 
            step_size=STEP_SIZE, 
            batch_size=BATCH_SIZE, # Batched inference
            verbose=False
        )
        
        # Store results
        results_legacy.append({
            'Image_Idx': i,
            'ROAD_Legacy (Black_90)': road_legacy_val
        })
        
        results_enhanced.append({
            'Image_Idx': i,
            'Insertion_AUC': metrics['insertion_auc'],
            'Deletion_AUC': metrics['deletion_auc'],
            'ROAD_20 (Blur)': metrics.get('road_20', 0),
            'ROAD_40 (Blur)': metrics.get('road_40', 0),
            'ROAD_60 (Blur)': metrics.get('road_60', 0),
            'ROAD_80 (Blur)': metrics.get('road_80', 0)
        })

    end_time = time.time()
    total_time = end_time - start_time
    
    # Convert to DataFrames
    df_legacy = pd.DataFrame(results_legacy)
    df_enhanced = pd.DataFrame(results_enhanced)
    
    print(f"\n{'='*60}")
    print(f"✅ TEST COMPLETED in {total_time:.2f} seconds ({total_time/N_IMAGES:.2f}s/img)")
    print(f"{'='*60}")
    
    print("\n📊 METRICS SUMMARY (Average):")
    
    print("\n--- [BEFORE] Legacy Style ---")
    print(f"ROAD (Black, 90%):    {df_legacy['ROAD_Legacy (Black_90)'].mean():.4f}")
    
    print("\n--- [AFTER] Enhanced Style ---")
    print(f"Insertion AUC:        {df_enhanced['Insertion_AUC'].mean():.4f}")
    print(f"Deletion AUC:         {df_enhanced['Deletion_AUC'].mean():.4f}")
    print(f"ROAD (Blur, 20%):     {df_enhanced['ROAD_20 (Blur)'].mean():.4f}")
    print(f"ROAD (Blur, 40%):     {df_enhanced['ROAD_40 (Blur)'].mean():.4f}")
    print(f"ROAD (Blur, 60%):     {df_enhanced['ROAD_60 (Blur)'].mean():.4f}")
    print(f"ROAD (Blur, 80%):     {df_enhanced['ROAD_80 (Blur)'].mean():.4f}")
    
    print("\nNote: Using 30 real ImageNet validation images.")

if __name__ == "__main__":
    regression_test()
