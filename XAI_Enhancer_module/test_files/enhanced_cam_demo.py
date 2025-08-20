#!/usr/bin/env python3
"""
Enhanced CAM Methods Demonstration

This script demonstrates the usage of all enhanced CAM methods including:
- GradCAMEnhanced
- GradCAMPlusPlusEnhanced  
- HiResCAMEnhanced
- ScoreCAMEnhanced
- AblationCAMEnhanced

It shows how to use each method and compares their outputs.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import enhanced CAM methods
from XAI_Enhancer_module.enhanced_cams import (
    GradCAMEnhanced,
    GradCAMPlusPlusEnhanced, 
    HiResCAMEnhanced,
    ScoreCAMEnhanced,
    AblationCAMEnhanced
)

# Import utilities
from XAI_Enhancer_module.utils.model_utils import test_model, get_device, transformations
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def load_and_preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for model input."""
    image = plt.imread(image_path)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    
    # Apply transformations
    image_tensor = transformations(image).float().unsqueeze(0)
    return image_tensor


def get_conv_layers(model):
    """Extract convolutional layers from the model."""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
            conv_layers.append(module)
    return conv_layers


def test_enhanced_cam_method(model, conv_layers, image_tensor, target_label, 
                           cam_class, method_name, device):
    """Test a specific enhanced CAM method."""
    print(f"\n{'='*50}")
    print(f"Testing {method_name}")
    print(f"{'='*50}")
    
    try:
        # Initialize the CAM method
        if method_name == "AblationCAMEnhanced":
            # AblationCAM may need special parameters
            cam_method = cam_class(model, conv_layers, batch_size=16)
        else:
            cam_method = cam_class(model, conv_layers)
        
        # Set target
        targets = [ClassifierOutputTarget(target_label)]
        
        # Generate CAM
        print(f"Generating {method_name} visualization...")
        cam_per_layer, modified_activations = cam_method(
            image_tensor.to(device), 
            targets
        )
        
        print(f"✅ {method_name} successful!")
        print(f"   Number of layers processed: {len(cam_per_layer)}")
        print(f"   CAM shape: {cam_per_layer[0].shape if cam_per_layer else 'None'}")
        print(f"   Modified activations: {len(modified_activations)} layers")
        
        # Calculate some statistics
        if cam_per_layer:
            combined_cam = np.mean([cam.squeeze() for cam in cam_per_layer], axis=0)
            print(f"   Combined CAM range: [{combined_cam.min():.4f}, {combined_cam.max():.4f}]")
            print(f"   Combined CAM mean: {combined_cam.mean():.4f}")
        
        return cam_per_layer, modified_activations, True
        
    except Exception as e:
        print(f"❌ {method_name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, False


def load_model_safely(model_name: str, device_preference: str = "auto"):
    """Load model safely, trying different configurations."""
    device = get_device(device_preference)
    
    # First try with ImageNet classes (1000)
    try:
        print("Attempting to load model with ImageNet (1000 classes)...")
        model = test_model(model_name, num_classes=1000, device_preference=device_preference)
        model.eval()
        
        # Test the model to confirm it works
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Successfully loaded model with {output.shape[1]} classes")
            return model, output.shape[1]
            
    except Exception as e:
        print(f"⚠️  Failed to load ImageNet model: {e}")
        
    # Fallback to 2 classes
    try:
        print("Attempting to load model with 2 classes...")
        model = test_model(model_name, num_classes=2, device_preference=device_preference)
        model.eval()
        
        # Test the model to confirm it works
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Successfully loaded model with {output.shape[1]} classes")
            return model, output.shape[1]
            
    except Exception as e:
        print(f"❌ Failed to load model with 2 classes: {e}")
        raise RuntimeError(f"Could not load model {model_name} with any configuration")


def compare_enhanced_cam_methods(model_name: str = "resnet18", target_label: int = 281):
    """Compare all enhanced CAM methods on a sample image."""
    
    print(f"Enhanced CAM Methods Comparison")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Requested target label: {target_label}")
    
    # Load model safely
    model, num_classes = load_model_safely(model_name, device_preference="auto")
    device = next(model.parameters()).device
    
    # Get conv layers
    conv_layers = get_conv_layers(model)
    print(f"Found {len(conv_layers)} convolutional layers")
    
    # Create a simple test image (random noise for demonstration)
    img_size = 384 if model_name.startswith("b4") else 224
    test_image = np.random.rand(img_size, img_size, 3).astype(np.float32)
    image_tensor = transformations(test_image).float().unsqueeze(0)
    
    print(f"Test image shape: {image_tensor.shape}")
    print(f"Model has {num_classes} output classes")
    
    # Adjust target label if it's out of bounds
    if target_label >= num_classes:
        original_target = target_label
        target_label = num_classes - 1  # Use the last class
        print(f"⚠️  Target label {original_target} is out of bounds for {num_classes} classes")
        print(f"   Using target label {target_label} instead")
    
    # Test the model output and get prediction
    with torch.no_grad():
        test_output = model(image_tensor.to(device))
        predicted_class = torch.argmax(test_output, dim=1).item()
        print(f"Model prediction: class {predicted_class}")
        print(f"Using target: class {target_label}")
    
    # Enhanced CAM methods to test
    enhanced_methods = {
        'GradCAMEnhanced': GradCAMEnhanced,
        'GradCAMPlusPlusEnhanced': GradCAMPlusPlusEnhanced,
        'HiResCAMEnhanced': HiResCAMEnhanced,
        'ScoreCAMEnhanced': ScoreCAMEnhanced,
        'AblationCAMEnhanced': AblationCAMEnhanced
    }
    
    results = {}
    successful_methods = []
    
    # Test each method
    for method_name, cam_class in enhanced_methods.items():
        cam_per_layer, modified_activations, success = test_enhanced_cam_method(
            model, conv_layers, image_tensor, target_label, 
            cam_class, method_name, device
        )
        
        results[method_name] = {
            'cam_per_layer': cam_per_layer,
            'modified_activations': modified_activations,
            'success': success
        }
        
        if success:
            successful_methods.append(method_name)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Successful methods: {len(successful_methods)}/{len(enhanced_methods)}")
    
    for method_name in enhanced_methods.keys():
        status = "✅ SUCCESS" if results[method_name]['success'] else "❌ FAILED"
        print(f"  {method_name:<25}: {status}")
    
    if successful_methods:
        print(f"\n📊 Performance Comparison (for successful methods):")
        for method_name in successful_methods:
            cam_per_layer = results[method_name]['cam_per_layer']
            if cam_per_layer:
                combined_cam = np.mean([cam.squeeze() for cam in cam_per_layer], axis=0)
                print(f"  {method_name:<25}: mean={combined_cam.mean():.4f}, "
                      f"std={combined_cam.std():.4f}, "
                      f"range=[{combined_cam.min():.4f}, {combined_cam.max():.4f}]")
    
    print(f"\n🎯 All enhanced CAM methods use the same interface:")
    print(f"   cam_per_layer, modified_activations = method(image_tensor, targets)")
    print(f"   They return both CAM visualizations and modified activations for analysis.")
    
    return results


def demonstrate_method_switching():
    """Demonstrate how to easily switch between enhanced CAM methods."""
    
    print(f"\n{'='*80}")
    print(f"ENHANCED CAM METHOD SWITCHING DEMONSTRATION")
    print(f"{'='*80}")
    
    # This shows how the optimized CAM extractor can use different methods
    from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
    
    model_name = "resnet18"
    model = test_model(model_name, num_classes=1000)
    conv_layers = get_conv_layers(model)
    
    enhanced_methods = ['GradCAMEnhanced', 'GradCAMPlusPlusEnhanced', 'HiResCAMEnhanced']
    
    print(f"Creating OptimizedCamExtractor with different enhanced CAM methods:")
    
    for method in enhanced_methods:
        try:
            extractor = OptimizedCamExtractor(
                model, model_name, conv_layers, cam_method=method
            )
            print(f"✅ Successfully created extractor with {method}")
            print(f"   Method: {extractor.cam_method_name}")
            print(f"   CAM class: {type(extractor.cam_method).__name__}")
        except Exception as e:
            print(f"❌ Failed to create extractor with {method}: {e}")
    
    print(f"\n💡 Usage in evaluation:")
    print(f"   evaluator = EnhancedProperAUCEvaluator(")
    print(f"       model_name='resnet18',")
    print(f"       enhanced_cam_method='GradCAMPlusPlusEnhanced'  # <-- Choose method here")
    print(f"   )")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced CAM Methods Demo")
    parser.add_argument("--model", default="resnet18", 
                       choices=["resnet18", "resnet34", "resnet50", "b0", "b4"],
                       help="Model to test")
    parser.add_argument("--target-label", type=int, default=281,
                       help="Target label for CAM generation")
    parser.add_argument("--test-switching", action="store_true",
                       help="Test method switching functionality")
    
    args = parser.parse_args()
    
    try:
        # Run the comparison
        results = compare_enhanced_cam_methods(args.model, args.target_label)
        
        if args.test_switching:
            demonstrate_method_switching()
        
        print(f"\n🎉 Enhanced CAM methods demonstration completed!")
        print(f"\nTo use these methods in your own code:")
        print(f"  1. Import from XAI_Enhancer_module.enhanced_cams")
        print(f"  2. Initialize with your model and target layers")
        print(f"  3. Call with image tensor and targets")
        print(f"  4. Get both CAM visualizations and modified activations")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
