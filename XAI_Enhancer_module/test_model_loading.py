#!/usr/bin/env python3
"""
Quick test script to verify the PyTorch loading fix.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model_utils import test_model

def test_model_loading():
    """Test if models can be loaded properly."""
    try:
        print("Testing model loading with PyTorch 2.6 compatibility...")
        
        # Test with the smallest model first
        model_name = "resnet18"
        print(f"Loading {model_name}...")
        model = test_model(model_name)
        print(f"✅ {model_name} loaded successfully!")
        
        # Test device placement
        device = next(model.parameters()).device
        print(f"✅ Model is on device: {device}")
        
        # Test model evaluation mode
        print(f"✅ Model training mode: {model.training}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test passed! You can now run the evaluation suite.")
    else:
        print("\n💥 Model loading test failed. Please check the error above.")
