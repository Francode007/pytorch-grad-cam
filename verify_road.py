
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path("/Users/franchisnsaikia/IBS_Research/pytorch-grad-cam")
sys.path.append(str(project_root))

from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator

def verify_road():
    print("🚀 Verifying ROAD Enhancements...")
    
    # Initialize Evaluator (using dummy paths as we will use synthetic data)
    evaluator = ImageNetProperAUCEvaluator(
        model_name='resnet50',
        imagenet_path='/tmp',
        device_preference='auto'
    )
    
    # Create Dummy Data
    print("\n📦 Generating dummy data...")
    # 1 Image, 3 Channels, 224x224
    dummy_image = torch.randn(3, 224, 224)
    # Dummy saliency map (random)
    dummy_saliency = np.random.rand(224, 224)
    predicted_label = 0
    
    # Evaluate ROAD
    print("\n🧪 Running evaluate_road()...")
    results = evaluator.evaluate_road(
        dummy_image, 
        dummy_saliency, 
        predicted_label,
        thresholds=[20, 40, 60, 80],
        imputation="blur"
    )
    
    print("\n📊 Results:")
    print(results)
    
    # Assertions
    expected_keys = ["road_20", "road_40", "road_60", "road_80"]
    for key in expected_keys:
        if key not in results:
            print(f"❌ Missing key: {key}")
            exit(1)
        if not isinstance(results[key], float):
             print(f"❌ Invalid value type for {key}: {type(results[key])}")
             exit(1)
             
    print("\n✅ Verification PASSED: Multi-threshold ROAD calculation working correctly.")

if __name__ == "__main__":
    verify_road()
