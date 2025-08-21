import os
import glob
import random
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import timm
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Device configuration
def get_device(device_preference: str = "auto"):
    """
    Get the appropriate device based on preference and availability.
    
    Args:
        device_preference: Preferred device ("auto", "cuda", "mps", "cpu")
        
    Returns:
        Device string
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("Warning: CUDA not available, falling back to CPU")
            return "cpu"
    elif device_preference == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        else:
            print("Warning: MPS not available, falling back to CPU")
            return "cpu"
    elif device_preference == "cpu":
        return "cpu"
    else:
        print(f"Warning: Unknown device '{device_preference}', using auto detection")
        return get_device("auto")

# Paths and constants
BASE_MODEL_PATH = "/Users/f0s03xp/Desktop/IBS-research/models"
TRAIN_DATA_PATH = '/Users/f0s03xp/Desktop/IBS-research/og_data/IBS-preprocessed-dataset'
IMAGENET_VAL_PATH = '/Users/f0s03xp/Desktop/IBS-research/og_data/imagenet-val'

# Dataset-specific constants
IBS_MEAN = [0.6380, 0.3422, 0.2275]
IBS_STD = [0.2448, 0.2060, 0.1710]
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Standard ImageNet normalization
IMAGENET_STD = [0.229, 0.224, 0.225]   # Standard ImageNet normalization

# Legacy constants for backward compatibility
MEAN = IBS_MEAN
STD = IBS_STD
BATCH_SIZE = 16

# IBS Class mappings
IBS_CLASSES = ['IBS', 'Normal']
IBS_CLASSES.sort()
IBS_IDX_TO_CLASS = dict(enumerate(IBS_CLASSES))
IBS_CLASS_TO_IDX = {v: k for k, v in IBS_IDX_TO_CLASS.items()}

# Legacy class mappings for backward compatibility
CLASSES = IBS_CLASSES
IDX_TO_CLASS = IBS_IDX_TO_CLASS
CLASS_TO_IDX = IBS_CLASS_TO_IDX

# Image transformations
def get_transformations(dataset_type="ibs"):
    """Get appropriate transformations for the dataset type."""
    if dataset_type.lower() == "imagenet":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:  # IBS or default
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IBS_MEAN, std=IBS_STD)
        ])

# Legacy transformation for backward compatibility
transformations = get_transformations("ibs")

# Dataset
class IBSValDataset(Dataset):
    def __init__(self, image_paths, model_name):
        self.image_paths = image_paths
        self.model_name = model_name
        self.transformations = get_transformations("ibs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = plt.imread(image_filepath)
        img_size = 384 if self.model_name.startswith("b4") else 224
        image = cv2.resize(image, (img_size, img_size))
        image = image.astype(float) / 255.0
        image = self.transformations(image).float()
        image_name = os.path.splitext(os.path.basename(image_filepath))[0]
        return image, image_name, image_filepath


class ImageNetValDataset(Dataset):
    def __init__(self, image_paths, model_name):
        self.image_paths = image_paths
        self.model_name = model_name
        self.transformations = get_transformations("imagenet")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = plt.imread(image_filepath)
        img_size = 384 if self.model_name.startswith("b4") else 224
        image = cv2.resize(image, (img_size, img_size))
        image = image.astype(float) / 255.0
        image = self.transformations(image).float()
        image_name = os.path.splitext(os.path.basename(image_filepath))[0]
        return image, image_name, image_filepath

# Data loader functions
def get_validation_paths(data_path):
    """Get validation paths for IBS dataset (80/20 split)."""
    image_paths = []
    for class_dir in glob.glob(os.path.join(data_path, '*')):
        for img_path in glob.glob(os.path.join(class_dir, '*')):
            image_paths.append(img_path)
    random.shuffle(image_paths)
    split_idx = int(0.8 * len(image_paths))
    return image_paths[split_idx:]


def get_imagenet_validation_paths(data_path=IMAGENET_VAL_PATH, max_images_per_class=None):
    """
    Get ImageNet validation paths.
    
    Args:
        data_path: Path to ImageNet validation dataset
        max_images_per_class: Maximum images per class (None for all)
    
    Returns:
        List of image paths
    """
    image_paths = []
    class_dirs = glob.glob(os.path.join(data_path, 'n*'))
    
    for class_dir in sorted(class_dirs):
        class_images = glob.glob(os.path.join(class_dir, '*.JPEG'))
        if max_images_per_class is not None:
            class_images = class_images[:max_images_per_class]
        image_paths.extend(class_images)
    
    return image_paths


def get_val_dataloader(model_name, dataset_type="ibs"):
    """
    Get validation dataloader for specified dataset.
    
    Args:
        model_name: Name of the model
        dataset_type: "ibs" or "imagenet"
    
    Returns:
        DataLoader for the specified dataset
    """
    if dataset_type.lower() == "imagenet":
        valid_image_paths = get_imagenet_validation_paths()
        dataset = ImageNetValDataset(valid_image_paths, model_name)
    else:  # IBS or default
        valid_image_paths = get_validation_paths(TRAIN_DATA_PATH)
        dataset = IBSValDataset(valid_image_paths, model_name)
    
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model builder
def build_model_inf(model_name, num_classes=2, base_model_path=None, dataset_type="ibs"):
    """
    Build an inference model.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        base_model_path: Optional base path for loading models
        dataset_type: "ibs" or "imagenet"
        
    Returns:
        PyTorch model
    """
    # Use default path if none is provided
    if base_model_path is None:
        base_model_path = BASE_MODEL_PATH
        
    model_path = Path(base_model_path) / f'{model_name}_{dataset_type}.pth'
    
    print(f"Attempting to load model from: {model_path}")

    # Load the appropriate model based on its name
    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if dataset_type == "imagenet" else None)
        elif model_name == 'resnet34':
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if dataset_type == "imagenet" else None)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if dataset_type == "imagenet" else None)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name.startswith('efficientnet'):
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if dataset_type == "imagenet" else None)
        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if dataset_type == "imagenet" else None)
        else:
            raise ValueError(f"Unsupported EfficientNet model: {model_name}")
            
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Model {model_name} not supported yet.")

    # Load state dict if the model file exists
    if model_path.exists():
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    elif dataset_type != "imagenet":
        print(f"Warning: Model file not found at {model_path}. Using a randomly initialized model.")
    else:
        print("Using pre-trained ImageNet weights from torchvision.")
        
    return model

def pred_model(model_name, num_classes=2, base_model_path=BASE_MODEL_PATH, 
               device_preference="auto", dataset_type="ibs"):
    """
    Load prediction model.
    
    Args:
        model_name: Name of the model
        num_classes: Number of classes
        base_model_path: Path to model weights
        device_preference: Device preference
        dataset_type: "ibs" or "imagenet"
    
    Returns:
        Loaded model with softmax
    """
    device = get_device(device_preference)
    
    if dataset_type.lower() == "imagenet":
        num_classes = 1000  # Override for ImageNet
    
    model, model_path = build_model_inf(model_name, num_classes, base_model_path, dataset_type)
    
    # Load weights for IBS models, ImageNet models are already pretrained
    if model_path is not None:
        # Fix for PyTorch 2.6 weights_only default change
        try:
            state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        except Exception:
            # Fallback to weights_only=False for older model files
            state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    
    model = nn.Sequential(model, nn.Softmax(dim=1))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    return model


def test_model(model_name, num_classes=2, base_model_path=BASE_MODEL_PATH, 
               device_preference="auto", dataset_type="ibs"):
    """
    Load test model.
    
    Args:
        model_name: Name of the model
        num_classes: Number of classes
        base_model_path: Path to model weights
        device_preference: Device preference
        dataset_type: "ibs" or "imagenet"
    
    Returns:
        Loaded model without softmax
    """
    device = get_device(device_preference)
    
    if dataset_type.lower() == "imagenet":
        num_classes = 1000  # Override for ImageNet
    
    model, model_path = build_model_inf(model_name, num_classes, base_model_path, dataset_type)
    
    # Load weights for IBS models, ImageNet models are already pretrained
    if model_path is not None:
        # Fix for PyTorch 2.6 weights_only default change
        try:
            state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        except Exception:
            # Fallback to weights_only=False for older model files
            state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    return model

# Inference
# Inference
def inference(model, testloader, model_name, device_preference="auto"):
    print(f"Predicting labels for model: {model_name}")
    predictions, image_filepaths, image_names = [], [], []
    device = get_device(device_preference)
    with torch.no_grad():
        for images, names, filepaths in tqdm(testloader, total=len(testloader)):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            image_names.extend(names)
            image_filepaths.extend(filepaths)
    return predictions, image_filepaths, image_names

# Example usage (uncomment to run)
# model_name = 'resnet50'
# model = pred_model(model_name)
# val_loader = get_val_dataloader(model_name)
# preds, paths, names = inference(model, val_loader, model_name)