#!/usr/bin/env python3
"""
ImageNet model utilities for loading pre-trained models and handling predictions.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_transforms(model_name: str = "resnet50") -> transforms.Compose:
    """
    Get appropriate transforms for ImageNet models.
    
    Args:
        model_name: Name of the model (affects input size)
        
    Returns:
        Composed transforms
    """
    # Different models may require different input sizes
    if 'efficientnet' in model_name:
        if 'b0' in model_name:
            input_size = 224
        elif 'b4' in model_name:
            input_size = 380
        else:
            input_size = 224
    else:
        input_size = 224
    
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),  # Slightly larger for center crop
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_pretrained_imagenet_model(model_name: str, device: str = "cpu") -> nn.Module:
    """
    Load a pre-trained ImageNet model.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on
        
    Returns:
        Pre-trained model
    """
    print(f"Loading pre-trained {model_name} model...")
    
    # Map model names to torchvision models
    model_map = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b4': models.efficientnet_b4,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(model_map.keys())}")
    
    # Load model with pre-trained weights
    model = model_map[model_name](pretrained=True)
    model.eval()
    model = model.to(device)
    
    print(f"Successfully loaded {model_name} with pre-trained ImageNet weights")
    return model


def get_model_target_layers(model: nn.Module, model_name: str, all_layers: bool = False) -> List[nn.Module]:
    """
    Get appropriate target layers for CAM methods based on model architecture.
    
    Args:
        model: The loaded model
        model_name: Name of the model
        all_layers: If True, returns all convolutional layers (for enhanced method).
                   If False, returns only the last convolutional layer (for standard CAM).
        
    Returns:
        List of target layers for CAM
    """
    if all_layers:
        # Return all convolutional layers (or stages)
        layers = []
        if 'resnet' in model_name:
            # ResNet has 4 layers: layer1, layer2, layer3, layer4
            layers.extend([model.layer1[-1], model.layer2[-1], model.layer3[-1], model.layer4[-1]])
        elif 'vgg' in model_name:
             # VGG features are sequential, take every MaxPool or just all Convs?
             # Standard practice for layer-wise is usually the end of blocks.
             # VGG is flat. Let's return all Conv2d layers for now, or maybe selected ones to avoid too many?
             # For simplicity/correctness of "stagewise", we likely want the last conv of each block.
             # But determining blocks in VGG features list is tricky without hardcoding indices.
             # Let's return ALL conv layers for VGG if requested, or maybe every 2nd/3rd?
             # Safest is all modules that are Conv2d.
             layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
        elif 'densenet' in model_name:
             # DenseNet features: denseblock1, transition1, denseblock2...
             # We can pick the norm layers after each block?
             # model.features has keys like denseblock1, transition1...
             # Let's try to get all 'denseblock' outputs?
             # Accessing named children might be safer.
             for name, module in model.features.named_children():
                 if 'denseblock' in name:
                     layers.append(module) # or module[-1]? DenseBlock is a container.
             # If empty (fail safe), fallback to all convs
             if not layers:
                 layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        elif 'efficientnet' in model_name:
             # EfficientNet features are blocks.
             # Let's return the last layer of each stage?
             # Simplified: Return all Conv2d
             layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        else:
             layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
             
        return layers

    # Default: Last Layer Only
    if 'resnet' in model_name:
        # For ResNet models, use the last block of layer4
        return [model.layer4[-1]]
    
    elif 'vgg' in model_name:
        # For VGG models, use the last convolutional layer in features
        conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
        return [conv_layers[-1]]
    
    elif 'densenet' in model_name:
        # For DenseNet models, use the last layer in features
        return [model.features.norm5]
    
    elif 'mobilenet' in model_name:
        # For MobileNet models, find the last convolutional layer
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        return [conv_layers[-1]]
    
    elif 'efficientnet' in model_name:
        # For EfficientNet models, use the last layer in features
        conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
        return [conv_layers[-1]]
    
    else:
        # Generic approach: find the last convolutional layer
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            raise ValueError(f"No convolutional layers found in {model_name}")
        return [conv_layers[-1]]


def predict_imagenet_image(model: nn.Module, 
                          image_path: str, 
                          transforms: transforms.Compose,
                          device: str = "cpu",
                          top_k: int = 5) -> Tuple[List[int], List[float]]:
    """
    Predict ImageNet class for a single image.
    
    Args:
        model: Pre-trained model
        image_path: Path to image file
        transforms: Image transforms to apply
        device: Device for computation
        top_k: Number of top predictions to return
        
    Returns:
        Tuple of (predicted_indices, confidence_scores)
    """
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transforms(image).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top-k predictions
        top_prob, top_indices = torch.topk(probabilities, top_k)
        
        predicted_indices = top_indices.cpu().numpy().tolist()
        confidence_scores = top_prob.cpu().numpy().tolist()
    
    return predicted_indices, confidence_scores


def batch_predict_imagenet(model: nn.Module,
                          image_paths: List[str],
                          transforms: transforms.Compose,
                          device: str = "cpu",
                          batch_size: int = 32) -> Tuple[List[int], List[float]]:
    """
    Predict ImageNet classes for a batch of images.
    
    Args:
        model: Pre-trained model
        image_paths: List of image file paths
        transforms: Image transforms to apply
        device: Device for computation
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (predicted_labels, confidence_scores)
    """
    all_predictions = []
    all_confidences = []
    
    model.eval()
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        
        # Load and preprocess batch
        for image_path in batch_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = transforms(image)
                batch_tensors.append(image_tensor)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                # Use a dummy tensor for failed images
                dummy_tensor = torch.zeros(3, 224, 224)
                batch_tensors.append(dummy_tensor)
        
        if not batch_tensors:
            continue
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top prediction for each image
            max_probs, predicted_indices = torch.max(probabilities, dim=1)
            
            all_predictions.extend(predicted_indices.cpu().numpy().tolist())
            all_confidences.extend(max_probs.cpu().numpy().tolist())
    
    return all_predictions, all_confidences


def analyze_model_predictions(model: nn.Module,
                            image_paths: List[str],
                            transforms: transforms.Compose,
                            synset_mapping: Dict[str, str],
                            device: str = "cpu") -> Dict:
    """
    Analyze model predictions on a set of images.
    
    Args:
        model: Pre-trained model
        image_paths: List of image file paths
        transforms: Image transforms to apply
        synset_mapping: Mapping from synset IDs to class names
        device: Device for computation
        
    Returns:
        Dictionary with analysis results
    """
    predictions, confidences = batch_predict_imagenet(
        model, image_paths, transforms, device
    )
    
    # Convert indices to synsets and class names
    synset_list = list(synset_mapping.keys())
    predicted_synsets = [synset_list[idx] if idx < len(synset_list) else "unknown" 
                        for idx in predictions]
    predicted_classes = [synset_mapping.get(synset, "unknown") 
                        for synset in predicted_synsets]
    
    # Calculate statistics
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    
    # Count unique predictions
    unique_classes = set(predicted_classes)
    
    analysis = {
        'total_images': len(image_paths),
        'predictions': predictions,
        'predicted_synsets': predicted_synsets,
        'predicted_classes': predicted_classes,
        'confidences': confidences,
        'avg_confidence': avg_confidence,
        'min_confidence': min_confidence,
        'max_confidence': max_confidence,
        'unique_classes_predicted': len(unique_classes),
        'class_distribution': {cls: predicted_classes.count(cls) 
                             for cls in unique_classes}
    }
    
    return analysis


def get_model_info(model: nn.Module, model_name: str) -> Dict:
    """
    Get information about a loaded model.
    
    Args:
        model: Loaded model
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    
    # Get model device
    device = next(model.parameters()).device
    
    info = {
        'model_name': model_name,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'conv_layers_count': len(conv_layers),
        'linear_layers_count': len(linear_layers),
        'device': str(device),
        'is_training': model.training
    }
    
    return info


def print_model_summary(model: nn.Module, model_name: str):
    """Print a summary of the model"""
    info = get_model_info(model, model_name)
    
    print(f"\nModel Summary: {model_name}")
    print("=" * 50)
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Convolutional layers: {info['conv_layers_count']}")
    print(f"Linear layers: {info['linear_layers_count']}")
    print(f"Device: {info['device']}")
    print(f"Training mode: {info['is_training']}")


if __name__ == "__main__":
    """Demo usage of ImageNet model utilities"""
    print("ImageNet Model Utilities Demo")
    print("=" * 50)
    
    # Test model loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load a small model for demo
    model_name = "resnet18"
    model = load_pretrained_imagenet_model(model_name, device)
    
    # Print model summary
    print_model_summary(model, model_name)
    
    # Get target layers
    target_layers = get_model_target_layers(model, model_name)
    print(f"\nTarget layers for CAM: {len(target_layers)}")
    for i, layer in enumerate(target_layers):
        print(f"  {i+1}. {layer}")
    
    # Test transforms
    transforms = get_imagenet_transforms(model_name)
    print(f"\nImage transforms: {len(transforms.transforms)} steps")
    for i, transform in enumerate(transforms.transforms):
        print(f"  {i+1}. {transform}")
