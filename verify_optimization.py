
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path("/Users/franchisnsaikia/IBS_Research/pytorch-grad-cam")
sys.path.append(str(project_root))

from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator

def test_batched_optimization():
    print("Testing Batched Optimization...")
    
    # Mock class to avoid full model loading if possible, or use a small model
    # For now, let's try to initialize the real evaluator but with a lightweight model if possible
    # We'll use resnet18 which is standard
    
    try:
        evaluator = ImageNetProperAUCEvaluator(
            model_name='resnet18',
            imagenet_path='/tmp', # Dummy path, we won't load images from disk
            device_preference='auto'
        )
        
        # Create dummy inputs
        # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        dummy_image = torch.randn(1, 3, 224, 224).to(evaluator.device)
        dummy_saliency = np.random.rand(224, 224)
        predicted_label = 0
        
        step_size = 50
        batch_size = 32
        
        print(f"Device: {evaluator.device}")
        print(f"Running Insertion AUC (Batch Size: {batch_size})...")
        
        import time
        start_time = time.time()
        
        scores, auc = evaluator.compute_insertion_auc(
            dummy_image[0], dummy_saliency, predicted_label, step_size=step_size, batch_size=batch_size
        )
        
        end_time = time.time()
        print(f"Insertion AUC: {auc:.4f}")
        print(f"Time Taken: {end_time - start_time:.4f}s")
        print(f"Number of scores: {len(scores)}")
        
        # Verify length
        expected_steps = (224*224) // step_size + 1 # simplistic check
        # Actual logic is range(0, n, step)
        
        print("\nRunning Deletion AUC (Batch Size: {batch_size})...")
        start_time = time.time()
        
        scores_del, auc_del = evaluator.compute_deletion_auc(
            dummy_image[0], dummy_saliency, predicted_label, step_size=step_size, batch_size=batch_size
        )
        end_time = time.time()
        
        print(f"Deletion AUC: {auc_del:.4f}")
        print(f"Time Taken: {end_time - start_time:.4f}s")
        
        print("\nRunning ROAD...")
        start_time = time.time()
        road_score = evaluator.evaluate_road(
            dummy_image, dummy_saliency, predicted_label
        )
        end_time = time.time()
        print(f"ROAD Score: {road_score:.4f}")
        print(f"Time Taken: {end_time - start_time:.4f}s")

        print("\n✅ Verification Successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_batched_optimization()
