#!/usr/bin/env python3
"""
ImageNet Proper AUC Evaluator - Evaluator specialized for ImageNet dataset.
This module extends the base ProperAUCEvaluator to work with ImageNet validation dataset.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the base ProperAUCEvaluator
from XAI_Enhancer_module.evaluator.proper_auc_evaluation import ProperAUCEvaluator
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, save_analysis_data


class ImageNetProperAUCEvaluator(ProperAUCEvaluator):
    """
    Evaluator specialized for ImageNet dataset with proper synset mapping.
    Uses ProperAUCEvaluator as the base for consistent AUC calculations.
    """
    
    def __init__(self, 
                 model_name: str, 
                 imagenet_path: str,
                 device_preference: str = "auto",
                 layer_mode: str = "last",
                 enhanced_cam_method: str = "GradCAMEnhanced"):
        """
        Initialize the ImageNet evaluator.
        
        Args:
            model_name: Name of the pre-trained model
            imagenet_path: Path to ImageNet validation dataset
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
            layer_mode: Layer selection mode ("last", "last_5", "all")
            enhanced_cam_method: Enhanced CAM method to use
        """
        # Don't call super().__init__() to avoid loading custom models
        # Instead, initialize only what we need from the base class
        self.model_name = model_name
        
        # Get device
        from XAI_Enhancer_module.utils.model_utils import get_device
        self.device = get_device(device_preference)
        
        # Store ImageNet-specific parameters
        self.imagenet_path = imagenet_path
        self.layer_mode = layer_mode
        self.enhanced_cam_method = enhanced_cam_method
        
        # Load synset mapping
        self.synset_mapping = self._load_synset_mapping()
        self.class_names = list(self.synset_mapping.values())
        self.synset_to_idx = {synset: idx for idx, synset in enumerate(self.synset_mapping.keys())}
        self.idx_to_synset = {idx: synset for synset, idx in self.synset_to_idx.items()}
        
        # Load pre-trained ImageNet model
        self.model = self._load_pretrained_model(model_name)
        self.model.eval()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize Enhanced CAM components
        self.conv_layers = self._get_enhanced_cam_layers(layer_mode)
        self.enhanced_cam_extractor = None
        
        # ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"ImageNetProperAUCEvaluator initialized:")
        print(f"  Model: {model_name}")
        print(f"  ImageNet path: {imagenet_path}")
        print(f"  Device: {self.device}")
        print(f"  Layer mode: {layer_mode}")
        print(f"  Enhanced CAM method: {enhanced_cam_method}")
        print(f"  Number of classes: {len(self.synset_mapping)}")
        print(f"  Number of conv layers: {len(self.conv_layers)}")
    
    def compute_insertion_auc(self, image_tensor: torch.Tensor, 
                            saliency_map: np.ndarray, predicted_label: int,
                            step_size: int = 50) -> Tuple[List[float], float]:
        """
        Compute insertion AUC by progressively adding pixels in order of importance.
        This method is adapted from the base ProperAUCEvaluator.
        """
        # Import necessary functions
        import torch
        import numpy as np
        
        # Ensure image tensor is on the correct device
        image_tensor = image_tensor.to(self.device)
        
        # Ensure saliency map is the right shape
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]  # Take first channel if RGB
        
        # Flatten saliency map and get pixel indices sorted by importance
        flat_saliency = saliency_map.flatten()
        pixel_indices = np.argsort(flat_saliency)[::-1]  # Descending order
        
        # Create baseline image (black) on the correct device
        baseline_image = torch.zeros_like(image_tensor, device=self.device)
        
        # Progressively add pixels and measure confidence
        confidences = []
        n_pixels = len(pixel_indices)
        
        self.model.eval()
        with torch.no_grad():
            # Initial confidence with baseline
            baseline_output = self.model(baseline_image)
            baseline_confidence = torch.softmax(baseline_output, dim=1)[0, predicted_label].item()
            confidences.append(baseline_confidence)
            
            # Create modified image for insertion
            modified_image = baseline_image.clone()
            original_flat = image_tensor.view(image_tensor.shape[0], -1)
            modified_flat = modified_image.view(modified_image.shape[0], -1)
            
            for i in range(0, n_pixels, step_size):
                # Add pixels in order of importance
                end_idx = min(i + step_size, n_pixels)
                for j in range(i, end_idx):
                    pixel_idx = pixel_indices[j]
                    modified_flat[:, pixel_idx] = original_flat[:, pixel_idx]
                
                # Get model confidence
                output = self.model(modified_image)
                confidence = torch.softmax(output, dim=1)[0, predicted_label].item()
                confidences.append(confidence)
        
        # Calculate AUC
        auc = float(np.trapz(confidences)) / len(confidences)
        return confidences, auc
    
    def compute_deletion_auc(self, image_tensor: torch.Tensor, 
                           saliency_map: np.ndarray, predicted_label: int,
                           step_size: int = 50) -> Tuple[List[float], float]:
        """
        Compute deletion AUC by progressively removing pixels in order of importance.
        This method is adapted from the base ProperAUCEvaluator.
        """
        # Import necessary functions
        import torch
        import numpy as np
        
        # Ensure image tensor is on the correct device
        image_tensor = image_tensor.to(self.device)
        
        # Ensure saliency map is the right shape
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]  # Take first channel if RGB
        
        # Flatten saliency map and get pixel indices sorted by importance
        flat_saliency = saliency_map.flatten()
        pixel_indices = np.argsort(flat_saliency)[::-1]  # Descending order
        
        # Start with original image on the correct device
        modified_image = image_tensor.clone().to(self.device)
        
        # Progressively remove pixels and measure confidence
        confidences = []
        n_pixels = len(pixel_indices)
        
        self.model.eval()
        with torch.no_grad():
            # Initial confidence with original image
            original_output = self.model(modified_image)
            original_confidence = torch.softmax(original_output, dim=1)[0, predicted_label].item()
            confidences.append(original_confidence)
            
            # Create flattened view for easier manipulation
            modified_flat = modified_image.view(modified_image.shape[0], -1)
            
            for i in range(0, n_pixels, step_size):
                # Remove pixels in order of importance (set to 0)
                end_idx = min(i + step_size, n_pixels)
                for j in range(i, end_idx):
                    pixel_idx = pixel_indices[j]
                    modified_flat[:, pixel_idx] = 0
                
                # Get model confidence
                output = self.model(modified_image)
                confidence = torch.softmax(output, dim=1)[0, predicted_label].item()
                confidences.append(confidence)
        
        # Calculate AUC
        auc = float(np.trapz(confidences)) / len(confidences)
        return confidences, auc
    
    def evaluate_road(self, image_tensor: torch.Tensor, 
                     saliency_map: np.ndarray, predicted_label: int) -> float:
        """
        Evaluate ROAD (Remove and Debias) score.
        This method is adapted from the base ProperAUCEvaluator.
        """
        import torch
        import numpy as np
        
        # Ensure image tensor is on the correct device
        image_tensor = image_tensor.to(self.device)
        
        # Simple ROAD implementation: remove top 10% most important pixels
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]
        
        flat_saliency = saliency_map.flatten()
        threshold = np.percentile(flat_saliency, 90)  # Top 10%
        
        # Create masked image
        mask = (saliency_map > threshold)
        modified_image = image_tensor.clone().to(self.device)
        modified_flat = modified_image.view(modified_image.shape[0], -1)
        mask_flat = mask.flatten()
        
        # Use tensor indexing instead of numpy boolean indexing
        mask_indices = torch.where(torch.from_numpy(mask_flat))[0]
        modified_flat[:, mask_indices] = 0
        
        # Get confidence drop
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(image_tensor)
            modified_output = self.model(modified_image)
            
            original_conf = torch.softmax(original_output, dim=1)[0, predicted_label].item()
            modified_conf = torch.softmax(modified_output, dim=1)[0, predicted_label].item()
            
            road_score = original_conf - modified_conf
        
        return max(0, road_score)  # Ensure non-negative
    
    def _evaluate_saliency_map(self, image_tensor: torch.Tensor, saliency_map: np.ndarray, 
                              predicted_label: int, step_size: int = 50, verbose: bool = False) -> Tuple[float, float, float]:
        """
        Evaluate a single saliency map using insertion AUC, deletion AUC, and ROAD metrics.
        
        Args:
            image_tensor: Preprocessed image tensor
            saliency_map: Generated saliency map
            predicted_label: Predicted class label
            step_size: Step size for evaluation
            verbose: Whether to print detailed results
            
        Returns:
            Tuple of (insertion_auc, deletion_auc, road_score)
        """
        try:
            # Compute insertion AUC
            _, insertion_auc = self.compute_insertion_auc(
                image_tensor, saliency_map, predicted_label, step_size
            )
            
            # Compute deletion AUC  
            _, deletion_auc = self.compute_deletion_auc(
                image_tensor, saliency_map, predicted_label, step_size
            )
            
            # Compute ROAD score
            road_score = self.evaluate_road(
                image_tensor, saliency_map, predicted_label
            )
            
            if verbose:
                print(f"        Insertion AUC: {insertion_auc:.4f}")
                print(f"        Deletion AUC: {deletion_auc:.4f}")
                print(f"        ROAD Score: {road_score:.4f}")
            
            return insertion_auc, deletion_auc, road_score
            
        except Exception as e:
            print(f"        Error in saliency evaluation: {e}")
            return 0.0, 0.0, 0.0
    
    def _load_synset_mapping(self) -> Dict[str, str]:
        """Load ImageNet synset mapping from LOC_synset_mapping.txt"""
        mapping_file = Path(__file__).parent.parent.parent / "LOC_synset_mapping.txt"
        
        if not mapping_file.exists():
            raise FileNotFoundError(f"Synset mapping file not found: {mapping_file}")
        
        synset_mapping = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        synset, class_name = parts
                        synset_mapping[synset] = class_name
        
        print(f"Loaded {len(synset_mapping)} ImageNet class mappings")
        return synset_mapping
    
    def _load_pretrained_model(self, model_name: str) -> torch.nn.Module:
        """Load pre-trained ImageNet model"""
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=True)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif model_name == 'densenet169':
            model = models.densenet169(pretrained=True)
        elif model_name == 'densenet201':
            model = models.densenet201(pretrained=True)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
        elif model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=True)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model
    
    def _get_enhanced_cam_layers(self, layer_mode: str = "last") -> List[torch.nn.Module]:
        """Get conv layers for Enhanced CAM extraction based on the specified mode."""
        all_conv_layers = []
        
        # Collect all convolutional layers
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
                all_conv_layers.append(module)
        
        if not all_conv_layers:
            raise ValueError(f"No convolutional layers found in model {self.model_name}")
        
        print(f"Found {len(all_conv_layers)} total convolutional layers")
        
        # Select layers based on mode
        if layer_mode == "all":
            selected_layers = all_conv_layers
            print(f"Selected all {len(selected_layers)} convolutional layers")
            
        elif layer_mode == "last_5":
            selected_layers = all_conv_layers[-5:] if len(all_conv_layers) >= 5 else all_conv_layers
            print(f"Selected last {len(selected_layers)} convolutional layers (requested 5)")
            
        elif layer_mode == "last":
            selected_layers = [all_conv_layers[-1]]
            print(f"Selected last convolutional layer")
            
        else:
            raise ValueError(f"Invalid layer_mode '{layer_mode}'. Must be one of ['all', 'last_5', 'last']")
        
        return selected_layers
    
    def get_imagenet_images(self, max_images: int = 50, classes_filter: List[str] = None) -> Tuple[List[str], List[int], List[str]]:
        """
        Get ImageNet validation images with proper class filtering.
        
        Args:
            max_images: Maximum number of images to get
            classes_filter: List of class names to filter (e.g., ['tench', 'goldfish'])
            
        Returns:
            Tuple of (image_paths, predicted_labels, class_names)
        """
        all_image_paths = []
        all_class_names = []
        
        # If no filter, use all classes, otherwise filter by class names
        if classes_filter is None:
            target_synsets = list(self.synset_mapping.keys())
        else:
            # Convert class names to synsets
            target_synsets = []
            for class_name in classes_filter:
                found_synset = None
                for synset, name in self.synset_mapping.items():
                    if class_name.lower() in name.lower():
                        found_synset = synset
                        break
                if found_synset:
                    target_synsets.append(found_synset)
                else:
                    print(f"Warning: Class '{class_name}' not found in synset mapping")
            
            if not target_synsets:
                raise ValueError(f"No valid synsets found for classes: {classes_filter}")
        
        # Collect images from target synsets
        images_per_synset = max(1, max_images // len(target_synsets)) if max_images > 0 else -1
        
        for synset in target_synsets:
            synset_dir = os.path.join(self.imagenet_path, synset)
            if not os.path.exists(synset_dir):
                print(f"Warning: Directory not found for synset {synset}: {synset_dir}")
                continue
            
            # Get images from this synset
            synset_images = glob.glob(os.path.join(synset_dir, "*.JPEG"))
            if not synset_images:
                print(f"Warning: No JPEG images found in {synset_dir}")
                continue
            
            # Limit images per synset if specified
            if images_per_synset > 0:
                synset_images = synset_images[:images_per_synset]
            
            all_image_paths.extend(synset_images)
            all_class_names.extend([self.synset_mapping[synset]] * len(synset_images))
            
            if max_images > 0 and len(all_image_paths) >= max_images:
                break
        
        # Limit total images if specified
        if max_images > 0 and len(all_image_paths) > max_images:
            all_image_paths = all_image_paths[:max_images]
            all_class_names = all_class_names[:max_images]
        
        # Get predicted labels using the model
        predicted_labels = self._predict_batch(all_image_paths)
        
        print(f"Collected {len(all_image_paths)} ImageNet images")
        if classes_filter:
            print(f"Filtered classes: {classes_filter}")
        
        return all_image_paths, predicted_labels, all_class_names
    
    def _predict_batch(self, image_paths: List[str]) -> List[int]:
        """Predict labels for a batch of images"""
        predicted_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc="Predicting labels"):
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    outputs = self.model(image_tensor)
                    predicted_label = torch.argmax(outputs, dim=1).item()
                    predicted_labels.append(predicted_label)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    predicted_labels.append(0)  # Default to first class
        
        return predicted_labels
    
    def extract_enhanced_cam(self, image_path: str, predicted_label: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Extract Enhanced CAM for a single ImageNet image."""
        if self.enhanced_cam_extractor is None:
            self.enhanced_cam_extractor = OptimizedCamExtractor(
                self.model, 
                self.model_name, 
                self.conv_layers,
                cam_method=self.enhanced_cam_method,
                device_preference=str(self.device)
            )
        
        # Extract Enhanced CAM
        image_tensor, saliency_map = self.enhanced_cam_extractor.extract_saliency_map(
            image_path, predicted_label
        )
        
        # Ensure image tensor has batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Convert saliency map to numpy if it's a tensor
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.cpu().numpy()
        
        return image_tensor, saliency_map
    
    def evaluate_enhanced_cam(self, max_images: int = 50, step_size: int = 50, 
                            verbose: bool = True, classes_filter: List[str] = None) -> Dict[str, any]:
        """
        Evaluate Enhanced CAM method on ImageNet with proper AUC calculations.
        
        Args:
            max_images: Maximum number of images to evaluate
            step_size: Step size for insertion/deletion evaluation
            verbose: If True, show detailed logging for each image
            classes_filter: List of ImageNet class names to filter
        
        Returns:
            Dictionary with evaluation results
        """
        # Get ImageNet images and predictions
        image_paths, predicted_labels, class_names = self.get_imagenet_images(
            max_images=max_images, 
            classes_filter=classes_filter
        )
        
        # Auto-adjust verbosity for large datasets
        if len(image_paths) > 20 and verbose:
            print(f"📢 Large dataset detected ({len(image_paths)} images). Setting verbose=False for cleaner output.")
            verbose = False
        
        print(f"\nEvaluating Enhanced CAM on {len(image_paths)} ImageNet images...")
        print(f"Using step_size={step_size} for proper evaluation...")
        if not verbose:
            print(f"Verbose mode OFF - only showing progress and summary.")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        # Create progress description based on verbosity
        progress_desc = "Processing Enhanced CAM"
        
        for i, (image_path, predicted_label, class_name) in enumerate(tqdm(
            zip(image_paths, predicted_labels, class_names), 
            desc=progress_desc, 
            total=len(image_paths)
        )):
            try:
                if verbose:
                    print(f"\n--- Image {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")
                    print(f"    Class: {class_name}")
                    print(f"    Predicted label: {predicted_label}")
                
                # Extract Enhanced CAM
                image_tensor, saliency_map = self.extract_enhanced_cam(image_path, predicted_label)
                
                # Evaluate saliency map using base class methods
                insertion_auc, deletion_auc, road_score = self._evaluate_saliency_map(
                    image_tensor, saliency_map, predicted_label, step_size, verbose
                )
                
                insertion_aucs.append(insertion_auc)
                deletion_aucs.append(deletion_auc)
                road_scores.append(road_score)
                
                if verbose:
                    print(f"    Insertion AUC: {insertion_auc:.4f}")
                    print(f"    Deletion AUC: {deletion_auc:.4f}")
                    print(f"    ROAD Score: {road_score:.4f}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Calculate final statistics
        results = {
            'insertion_auc_mean': np.mean(insertion_aucs),
            'insertion_auc_std': np.std(insertion_aucs),
            'deletion_auc_mean': np.mean(deletion_aucs),
            'deletion_auc_std': np.std(deletion_aucs),
            'road_mean': np.mean(road_scores),
            'road_std': np.std(road_scores),
            'num_images': len(insertion_aucs),
            'classes_evaluated': classes_filter if classes_filter else 'All ImageNet classes'
        }
        
        return results
    
    def evaluate_method(self, cam_method_name: str, max_images: int = 50, 
                       classes_filter: List[str] = None) -> Dict[str, any]:
        """
        Evaluate a standard CAM method on ImageNet.
        
        Args:
            cam_method_name: Name of the CAM method
            max_images: Maximum number of images to evaluate
            classes_filter: List of ImageNet class names to filter
            
        Returns:
            Dictionary with evaluation results
        """
        # Get ImageNet images and predictions
        image_paths, predicted_labels, class_names = self.get_imagenet_images(
            max_images=max_images, 
            classes_filter=classes_filter
        )
        
        print(f"\nEvaluating {cam_method_name} on {len(image_paths)} ImageNet images...")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        
        for i, (image_path, predicted_label, class_name) in enumerate(tqdm(
            zip(image_paths, predicted_labels, class_names), 
            desc=f"Processing {cam_method_name}", 
            total=len(image_paths)
        )):
            try:
                # Extract standard CAM using pytorch-grad-cam
                saliency_map = self._extract_standard_cam(
                    image_path, predicted_label, cam_method_name
                )
                
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Evaluate saliency map
                insertion_auc, deletion_auc, road_score = self._evaluate_saliency_map(
                    image_tensor, saliency_map, predicted_label, step_size=50, verbose=False
                )
                
                insertion_aucs.append(insertion_auc)
                deletion_aucs.append(deletion_auc)
                road_scores.append(road_score)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Calculate final statistics
        results = {
            'insertion_auc_mean': np.mean(insertion_aucs),
            'insertion_auc_std': np.std(insertion_aucs),
            'deletion_auc_mean': np.mean(deletion_aucs),
            'deletion_auc_std': np.std(deletion_aucs),
            'road_mean': np.mean(road_scores),
            'road_std': np.std(road_scores),
            'num_images': len(insertion_aucs),
            'classes_evaluated': classes_filter if classes_filter else 'All ImageNet classes'
        }
        
        return results
    
    def _extract_standard_cam(self, image_path: str, predicted_label: int, cam_method_name: str) -> np.ndarray:
        """Extract standard CAM using pytorch-grad-cam library"""
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, HiResCAM, LayerCAM, ScoreCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        # Select appropriate CAM method
        if cam_method_name == "GradCAM":
            cam_class = GradCAM
        elif cam_method_name == "GradCAM++":
            cam_class = GradCAMPlusPlus
        elif cam_method_name == "EigenCAM":
            cam_class = EigenCAM
        elif cam_method_name == "HiResCAM":
            cam_class = HiResCAM
        elif cam_method_name == "LayerCAM":
            cam_class = LayerCAM
        elif cam_method_name == "ScoreCAM":
            cam_class = ScoreCAM
        else:
            raise ValueError(f"Unsupported CAM method: {cam_method_name}")
        
        # Get target layer based on model architecture
        if hasattr(self.model, 'layer4'):  # ResNet
            target_layers = [self.model.layer4[-1]]
        elif hasattr(self.model, 'features'):  # VGG, DenseNet, etc.
            target_layers = [self.model.features[-1]]
        elif hasattr(self.model, 'classifier'):  # Some models
            # Find the last conv layer
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            target_layers = [conv_layers[-1]]
        else:
            # Fallback: find last conv layer
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            target_layers = [conv_layers[-1]]
        
        # Initialize CAM
        cam = cam_class(model=self.model, target_layers=target_layers)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Generate CAM
        targets = [ClassifierOutputTarget(predicted_label)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        
        # Return the saliency map for the first (and only) image
        return grayscale_cam[0]
    
    def compare_enhanced_vs_standard(self, standard_methods: List[str], max_images: int = 50,
                                   step_size: int = 50, verbose: bool = False,
                                   classes_filter: List[str] = None) -> pd.DataFrame:
        """
        Compare Enhanced CAM vs standard methods on ImageNet.
        
        Args:
            standard_methods: List of standard CAM methods
            max_images: Maximum number of images to evaluate
            step_size: Step size for evaluation
            verbose: Verbose output
            classes_filter: List of ImageNet class names to filter
            
        Returns:
            DataFrame with comparison results
        """
        print(f"Running ImageNet comparison: Enhanced CAM vs {len(standard_methods)} standard methods")
        
        all_results = []
        
        # Evaluate Enhanced CAM
        enhanced_results = self.evaluate_enhanced_cam(
            max_images=max_images,
            step_size=step_size,
            verbose=verbose,
            classes_filter=classes_filter
        )
        
        enhanced_row = {
            'Method': f'{self.enhanced_cam_method}',
            'Dataset': 'ImageNet',
            'Insertion_AUC_Mean': enhanced_results['insertion_auc_mean'],
            'Insertion_AUC_Std': enhanced_results['insertion_auc_std'],
            'Deletion_AUC_Mean': enhanced_results['deletion_auc_mean'],
            'Deletion_AUC_Std': enhanced_results['deletion_auc_std'],
            'ROAD_Mean': enhanced_results['road_mean'],
            'ROAD_Std': enhanced_results['road_std'],
            'Images_Evaluated': enhanced_results['num_images']
        }
        all_results.append(enhanced_row)
        
        # Evaluate standard methods
        for method in standard_methods:
            method_results = self.evaluate_method(
                cam_method_name=method,
                max_images=max_images,
                classes_filter=classes_filter
            )
            
            method_row = {
                'Method': method,
                'Dataset': 'ImageNet',
                'Insertion_AUC_Mean': method_results['insertion_auc_mean'],
                'Insertion_AUC_Std': method_results['insertion_auc_std'],
                'Deletion_AUC_Mean': method_results['deletion_auc_mean'],
                'Deletion_AUC_Std': method_results['deletion_auc_std'],
                'ROAD_Mean': method_results['road_mean'],
                'ROAD_Std': method_results['road_std'],
                'Images_Evaluated': method_results['num_images']
            }
            all_results.append(method_row)
        
        return pd.DataFrame(all_results)
