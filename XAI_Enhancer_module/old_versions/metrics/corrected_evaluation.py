#!/usr/bin/env python3
"""
Corrected evaluation metrics that produce proper AUC scores in [0,1] range.
This module fixes the step size and AUC calculation issues.
"""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pytorch_grad_cam.metrics.road import ROADCombined

def gkern(klen, nsig):
    """Returns a Gaussian kernel array for blurring."""
    inp = np.zeros((klen, klen))
    inp[klen//2, klen//2] = 1
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc_corrected(scores):
    """Calculate normalized AUC that returns values in [0,1] range."""
    scores = np.array(scores)
    if len(scores) <= 1:
        return 0.0
    
    # Normalize scores to [0,1] range first
    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    # Calculate AUC using trapezoidal rule and normalize by number of intervals
    auc_value = np.trapz(scores_normalized) / (len(scores_normalized) - 1)
    return max(0.0, min(1.0, auc_value))  # Ensure [0,1] range

class CorrectedCausalMetric:
    """Corrected implementation of insertion/deletion metrics."""
    
    def __init__(self, model, model_name, mode, step_size, substrate_fn, device_preference="auto"):
        assert mode in ['del', 'ins']
        self.model = model
        self.model_name = model_name
        self.mode = mode
        self.step_size = step_size  # Use provided step_size, not img_size
        self.substrate_fn = substrate_fn
        
        # Import get_device here to avoid circular imports
        from XAI_Enhancer_module.utils.model_utils import get_device
        self.device = get_device(device_preference)
        
        # Calculate image dimensions
        self.img_size = 224
        if model_name in ('b4'):
            self.img_size = 384
        self.HW = self.img_size * self.img_size
    
    def single_run(self, img_tensor, explanation, verbose=0):
        """Run metric on one image-saliency pair with corrected step size."""
        
        # Get original prediction
        with torch.no_grad():
            pred = self.model(img_tensor.to(self.device))
            top_prob, predicted_class = torch.max(pred, 1)
            predicted_class = predicted_class.cpu().item()
        
        # Calculate number of steps based on actual step size
        n_steps = max(1, self.HW // self.step_size)
        
        if self.mode == 'del':
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()
        
        scores = []
        
        # Get pixel coordinates sorted by saliency (highest first)
        explanation_flat = explanation.reshape(-1)
        salient_order = np.argsort(explanation_flat)[::-1]  # Descending order
        
        # Evaluate at each step
        for i in range(n_steps + 1):
            with torch.no_grad():
                pred = self.model(start.to(self.device))
                prob = torch.softmax(pred, dim=1)[0, predicted_class].cpu().item()
                scores.append(prob)
            
            if i < n_steps:
                # Determine pixels to modify in this step
                start_idx = i * self.step_size
                end_idx = min((i + 1) * self.step_size, self.HW)
                coords_to_modify = salient_order[start_idx:end_idx]
                
                # Convert flat indices to 2D coordinates
                start_reshaped = start.view(1, 3, self.img_size, self.img_size)
                finish_reshaped = finish.view(1, 3, self.img_size, self.img_size)
                
                for coord in coords_to_modify:
                    h = coord // self.img_size
                    w = coord % self.img_size
                    start_reshaped[0, :, h, w] = finish_reshaped[0, :, h, w]
        
        return np.array(scores)
    
    def evaluate_batch(self, img_batch, exp_batch, batch_size=1):
        """Evaluate batch of images with corrected implementation."""
        n_samples = img_batch.shape[0]
        all_scores = []
        
        for i in tqdm(range(n_samples), desc=f'Evaluating {self.mode}'):
            scores = self.single_run(img_batch[i:i+1], exp_batch[i])
            all_scores.append(scores)
        
        # Convert to numpy array
        max_len = max(len(scores) for scores in all_scores)
        scores_array = np.zeros((max_len, n_samples))
        
        for i, scores in enumerate(all_scores):
            scores_array[:len(scores), i] = scores
        
        return scores_array

def get_corrected_metrics(model, model_name, step_size=50, device_preference="auto"):
    """Get corrected metrics with proper step size."""
    
    # Create blur kernel for insertion baseline
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
    
    # Create metrics with corrected step size
    insertion = CorrectedCausalMetric(
        model, model_name, 'ins', step_size, 
        substrate_fn=blur, device_preference=device_preference
    )
    
    deletion = CorrectedCausalMetric(
        model, model_name, 'del', step_size, 
        substrate_fn=torch.zeros_like, device_preference=device_preference
    )
    
    road_combined = ROADCombined(percentiles=[20, 40, 60, 80])
    
    return insertion, deletion, road_combined

def test_corrected_evaluation():
    """Test the corrected evaluation on a single image."""
    print("Testing corrected evaluation metrics...")
    
    # Load model and data
    from XAI_Enhancer_module.utils.model_utils import test_model, get_validation_paths, TRAIN_DATA_PATH
    from XAI_Enhancer_module.utils.optimized_predictor import get_optimized_predictions
    from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
    
    model_name = "resnet18"
    model = test_model(model_name, device_preference="mps")
    
    # Get one test image
    image_paths = get_validation_paths(TRAIN_DATA_PATH)[:1]
    predicted_labels, _, _ = get_optimized_predictions(
        model_name, image_paths, use_validation_set=False, device_preference="mps"
    )
    
    # Extract saliency map
    conv_layers = [model.layer4[-1]]
    cam_extractor = OptimizedCamExtractor(model, model_name, conv_layers, "mps")
    image_tensor, saliency_map = cam_extractor.extract_saliency_map(
        image_paths[0], predicted_labels[0]
    )
    
    print(f"Testing on: {image_paths[0]}")
    print(f"Predicted label: {predicted_labels[0]}")
    print(f"Saliency map range: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
    
    # Test with different step sizes
    step_sizes = [50, 100, 200]
    
    for step_size in step_sizes:
        print(f"\n--- Testing with step_size={step_size} ---")
        
        insertion, deletion, road = get_corrected_metrics(
            model, model_name, step_size=step_size, device_preference="mps"
        )
        
        # Run evaluation
        saliency_np = saliency_map.cpu().numpy()
        
        insertion_scores = insertion.single_run(image_tensor, saliency_np)
        deletion_scores = deletion.single_run(image_tensor, saliency_np)
        
        # Calculate AUC
        insertion_auc = auc_corrected(insertion_scores)
        deletion_auc = auc_corrected(deletion_scores)
        
        print(f"Insertion - Steps: {len(insertion_scores)}, AUC: {insertion_auc:.4f}")
        print(f"           Scores: {insertion_scores[:5]}... -> {insertion_scores[-5:]}")
        print(f"Deletion  - Steps: {len(deletion_scores)}, AUC: {deletion_auc:.4f}")
        print(f"           Scores: {deletion_scores[:5]}... -> {deletion_scores[-5:]}")

if __name__ == "__main__":
    test_corrected_evaluation()
