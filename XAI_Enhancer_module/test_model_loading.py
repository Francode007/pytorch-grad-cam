#!/usr/bin/env python3
"""
Quick test script to verify the PyTorch loading fix with device selection.
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model_utils import test_model, get_device

def setup_argument_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(description='Test model loading with device selection')
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for testing (default: auto)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50', 'b0', 'b4', 'densenet', 'xception'],
        help='Model to test (default: resnet18)'
    )
    return parser

def test_model_loading(model_name="resnet18", device_preference="auto"):
    """Test if models can be loaded properly."""
    try:
        print(f"Testing model loading with PyTorch 2.6 compatibility...")
        print(f"Model: {model_name}")
        print(f"Device preference: {device_preference}")
        
        # Test device detection
        device = get_device(device_preference)
        print(f"✅ Device selected: {device}")
        
        # Test model loading
        print(f"Loading {model_name}...")
        model = test_model(model_name, device_preference=device_preference)
        print(f"✅ {model_name} loaded successfully!")
        
        # Test device placement
        actual_device = next(model.parameters()).device
        print(f"✅ Model is on device: {actual_device}")
        
        # Test model evaluation mode
        print(f"✅ Model training mode: {model.training}")
        
        # Test forward pass with dummy input
        import torch
        if model_name.startswith('b4'):
            dummy_input = torch.randn(1, 3, 384, 384).to(actual_device)
        else:
            dummy_input = torch.randn(1, 3, 224, 224).to(actual_device)
            
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    success = test_model_loading(args.model, args.device)
    if success:
        print(f"\n🎉 Model loading test passed!")
        print(f"You can now run the evaluation suite with:")
        print(f"python run_evaluation.py --model {args.model} --device {args.device} --eval-type quick")
    else:
        print(f"\n💥 Model loading test failed. Please check the error above.")
