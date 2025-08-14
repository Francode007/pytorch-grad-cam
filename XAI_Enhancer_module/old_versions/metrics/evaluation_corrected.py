#!/usr/bin/env python3
"""
Corrected Evaluation Metrics
This file contains the fixed versions of your evaluation metrics to produce proper AUC scores.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

from XAI_Enhancer_module.old_versions.metrics.utils_metric import *
from pytorch_grad_cam.metrics.road import ROADCombined

n_classes = 2

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc_corrected(arr):
    """Returns normalized Area Under Curve of the array - CORRECTED VERSION."""
    # Normalize to [0,1] range first
    arr = np.array(arr)
    n_steps = len(arr)
    
    # Use trapezoidal rule and normalize by the number of intervals
    # This ensures AUC is in [0,1] range
    return np.trapz(arr) / (n_steps - 1) if n_steps > 1 else arr[0]

class CausalMetricCorrected():
    """Corrected version of the CausalMetric class."""

    def __init__(self, model, model_name, mode, step, substrate_fn, device_preference="auto"):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            device_preference (str): device preference for computation.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.model_name = model_name
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.device_preference = device_preference
        
        # Import get_device here to avoid circular imports
        from XAI_Enhancer_module.utils.model_utils import get_device
        self.device = get_device(device_preference)

    def HW_calc(self, model_name):
        HW = 224 * 224
        if model_name in ('b4'):
            HW = 384 * 384
        return HW

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair - CORRECTED VERSION.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        HW = self.HW_calc(self.model_name)
        
        # CORRECTED: Get initial prediction using SOFTMAX probabilities
        with torch.no_grad():
            pred = self.model(img_tensor.to(self.device))
            probs = F.softmax(pred, dim=1)
            top_prob, c = torch.max(probs, 1)
            c = c.cpu().numpy()[0]
        
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1).copy()
        
        for i in range(n_steps+1):
            # CORRECTED: Use softmax probabilities instead of raw logits
            with torch.no_grad():
                pred = self.model(start.to(self.device))
                probs = F.softmax(pred, dim=1)
                scores[i] = probs[0, c].cpu().numpy()  # Get probability for target class
            
            if verbose == 2:
                pr, cl = torch.topk(probs, 2)
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                # Convert coords to tensor indices for proper tensor assignment
                coords_tensor = torch.from_numpy(coords).long()
                start_flat = start.view(1, 3, HW)
                finish_flat = finish.view(1, 3, HW)
                start_flat[0, :, coords_tensor[0]] = finish_flat[0, :, coords_tensor[0]]
        
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images - CORRECTED VERSION.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        HW = self.HW_calc(self.model_name)
        n_samples = img_batch.shape[0]
        
        # CORRECTED: Predict probabilities instead of raw logits
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            with torch.no_grad():
                preds = self.model(img_batch[i*batch_size:(i+1)*batch_size].to(self.device))
                # CORRECTED: Convert to probabilities
                predictions[i*batch_size:(i+1)*batch_size] = F.softmax(preds, dim=1).cpu()

        top = torch.max(predictions, 1)[1]
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(n_samples, HW), axis=1), axis=-1)
        
        for i in tqdm(range(n_steps+1), desc=f'{self.mode.capitalize()} evaluation'):
            if self.mode == 'del':
                # Deletion: start with original, move towards substrate
                start = img_batch.clone()
                finish = torch.stack([self.substrate_fn(img) for img in img_batch])
            elif self.mode == 'ins':
                # Insertion: start with substrate, move towards original
                start = torch.stack([self.substrate_fn(img) for img in img_batch])
                finish = img_batch.clone()
            
            # Apply perturbation for current step
            if i > 0:
                coords = salient_order[:, self.step * (i-1):self.step * i]
                for j in range(n_samples):
                    coords_tensor = torch.from_numpy(coords[j]).long()
                    start_flat = start[j].view(3, HW)
                    finish_flat = finish[j].view(3, HW)
                    start_flat[:, coords_tensor] = finish_flat[:, coords_tensor]
            
            # Get scores for current step
            for j in tqdm(range(n_samples // batch_size), desc='Computing scores', leave=False):
                batch_start = j * batch_size
                batch_end = (j + 1) * batch_size
                
                with torch.no_grad():
                    batch_imgs = start[batch_start:batch_end].to(self.device)
                    preds = self.model(batch_imgs)
                    # CORRECTED: Use softmax probabilities
                    probs = F.softmax(preds, dim=1)
                    
                    for k in range(batch_size):
                        scores[i, batch_start + k] = probs[k, top[batch_start + k]].cpu().numpy()
        
        return scores


def get_metrics_corrected(model, model_name, img_size=224, device_preference="auto"):
    """Get corrected metrics that return proper AUC values in [0,1] range."""
    # substrate fn
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    # Function that blurs input image
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
    
    insertion = CausalMetricCorrected(model, model_name, 'ins', img_size, substrate_fn=blur, device_preference=device_preference)
    deletion = CausalMetricCorrected(model, model_name, 'del', img_size, substrate_fn=torch.zeros_like, device_preference=device_preference)
    road_combined = ROADCombined(percentiles=[20, 40, 60, 80])
    
    return insertion, deletion, road_combined


# Example usage function
def test_corrected_metrics(model, model_name, image_tensor, saliency_map, predicted_label):
    """Test the corrected metrics on a single image."""
    print("Testing corrected metrics...")
    
    # Get corrected metrics
    insertion, deletion, road_combined = get_metrics_corrected(model, model_name)
    
    # Test insertion
    insertion_scores = insertion.single_run(image_tensor, saliency_map)
    insertion_auc = auc_corrected(insertion_scores)
    print(f"Insertion AUC (corrected): {insertion_auc:.4f}")
    
    # Test deletion  
    deletion_scores = deletion.single_run(image_tensor, saliency_map)
    deletion_auc = auc_corrected(deletion_scores)
    print(f"Deletion AUC (corrected): {deletion_auc:.4f}")
    
    # Test ROAD
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
    targets = [ClassifierOutputSoftmaxTarget(predicted_label)]
    saliency_batch = np.expand_dims(saliency_map, axis=0)
    road_scores = road_combined(image_tensor, saliency_batch, targets, model)
    print(f"ROAD Score (corrected): {road_scores[0]:.4f}")
    
    return {
        'insertion_auc': insertion_auc,
        'deletion_auc': deletion_auc,
        'road_score': road_scores[0],
        'insertion_scores': insertion_scores,
        'deletion_scores': deletion_scores
    }


if __name__ == "__main__":
    print("Corrected evaluation metrics module loaded.")
    print("Key fixes:")
    print("1. Use softmax probabilities instead of raw logits")
    print("2. Proper AUC calculation with normalization")
    print("3. Correct score range [0,1] for probabilities")
    print("4. Fixed perturbation logic for insertion/deletion")
