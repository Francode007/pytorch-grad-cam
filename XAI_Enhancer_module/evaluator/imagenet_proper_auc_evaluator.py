#!/usr/bin/env python3
"""
ImageNet Proper AUC Evaluator - Evaluator specialized for ImageNet dataset.
This module extends the base ProperAUCEvaluator to work with ImageNet validation dataset.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the base ProperAUCEvaluator
from XAI_Enhancer_module.evaluator.proper_auc_evaluation import ProperAUCEvaluator
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor
from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, save_analysis_data



class _ImageNetDataset(Dataset):
    """Dataset wrapper for ImageNet images to enable multi-threaded loading."""
    def __init__(self, image_paths, predicted_labels, class_names, transform=None):
        self.image_paths = image_paths
        self.predicted_labels = predicted_labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        predicted_label = self.predicted_labels[idx]
        class_name = self.class_names[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image  # Should not happen if transform provided
            
        return image_tensor, predicted_label, class_name, image_path

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
                 enhanced_cam_method: str = "GradCAMEnhanced",
                 model_cache_dir: str = "../pytorch_models/",
                 extractor_cls=None,
                 extractor_kwargs: dict = None):
        """
        Initialize the ImageNet evaluator.
        
        Args:
            model_name: Name of the pre-trained model
            imagenet_path: Path to ImageNet validation dataset
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
            layer_mode: Layer selection mode ("last", "last_5", "all")
            enhanced_cam_method: Enhanced CAM method to use
            model_cache_dir: Directory containing pre-downloaded models (default: "../pytorch_models/")
            extractor_cls: Optional custom extractor class (default: OptimizedCamExtractor)
            extractor_kwargs: Optional kwargs for the extractor
        """
        # Don't call super().__init__() to avoid loading custom models
        # Instead, initialize only what we need from the base class
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        
        # Optimize global settings
        torch.backends.cudnn.benchmark = True
        
        # Get device
        from XAI_Enhancer_module.utils.model_utils import get_device
        device_val = get_device(device_preference)
        self.device = torch.device(device_val) if isinstance(device_val, str) else device_val
        
        # Store ImageNet-specific parameters
        self.imagenet_path = imagenet_path
        self.layer_mode = layer_mode
        self.enhanced_cam_method = enhanced_cam_method
        self.extractor_cls = extractor_cls or OptimizedCamExtractor
        self.extractor_kwargs = extractor_kwargs or {}
        
        # Load synset mapping
        self.synset_mapping = self._load_synset_mapping()
        self.class_names = list(self.synset_mapping.values())
        self.synset_to_idx = {synset: idx for idx, synset in enumerate(self.synset_mapping.keys())}
        self.idx_to_synset = {idx: synset for synset, idx in self.synset_to_idx.items()}
        
        # Initialize model loader with cache directory
        from XAI_Enhancer_module.utils.model_loader import ModelLoader
        self.model_loader = ModelLoader(model_cache_dir)
        
        # Load pre-trained ImageNet model using cached weights
        self.model = self.model_loader.load_pretrained_model(model_name)
        self.model.eval()
        
        # Load a CLEAN model for metrics (to avoid hook overhead from CAM extractor)
        self.clean_model = self.model_loader.load_pretrained_model(model_name)
        self.clean_model.eval()
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.clean_model = self.clean_model.to(self.device)
        
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
        print(f"  Model cache dir: {model_cache_dir}")
        print(f"  ImageNet path: {imagenet_path}")
        print(f"  Device: {self.device}")
        print(f"  Layer mode: {layer_mode}")
        print(f"  Enhanced CAM method: {enhanced_cam_method}")
        print(f"  Number of classes: {len(self.synset_mapping)}")
        print(f"  Number of conv layers: {len(self.conv_layers)}")
        
        # Print cache status for verification
        self.model_loader.print_cache_status()
    
    def compute_insertion_auc(self, image_tensor: torch.Tensor, 
                            saliency_map: np.ndarray, predicted_label: int,
                            step_size: int = 50, batch_size: int = 2048) -> Tuple[List[float], float]:
        """
        Compute insertion AUC by progressively adding pixels in order of importance.
        Optimized with full vectorization for large batch processing.
        """
        # Ensure image tensor is 3D [C, H, W]
        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
            
        # Ensure saliency map is the right shape
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]  # Take first channel if RGB
        
        # 1. Flatten saliency map and get pixel indices sorted by importance
        flat_saliency = saliency_map.flatten()
        pixel_indices_np = np.argsort(flat_saliency)[::-1]  # Descending order, high to low
        
        c, h, w = image_tensor.shape
        n_pixels = h * w
        pixel_indices = torch.from_numpy(pixel_indices_np.copy()).to(self.device).long()
        
        # 2. Determine steps
        steps = list(range(0, n_pixels, step_size))
        if steps[-1] != n_pixels:
            steps.append(n_pixels)
        
        num_steps = len(steps)
        
        # 3. Create a Rank Map for Vectorized Masking
        # We want to find pixels where rank < threshold
        rank_map_flat = torch.zeros(n_pixels, device=self.device, dtype=torch.long)
        rank_indices = pixel_indices
        rank_values = torch.arange(n_pixels, device=self.device)
        rank_map_flat[rank_indices] = rank_values
        
        # Reshape to 2D [H, W]
        rank_map = rank_map_flat.view(1, h, w) 
        
        # Create Thresholds Tensor [num_steps, 1, 1]
        thresholds = torch.tensor(steps, device=self.device).view(-1, 1, 1)
        
        # 4. Run Batched Inference - process in chunks to avoid materializing full [num_steps, C, H, W]
        confidences = []
        self.clean_model.eval()
        img_expanded = image_tensor.unsqueeze(0)  # [1, C, H, W]
        
        for i in range(0, num_steps, batch_size):
            end_idx = min(i + batch_size, num_steps)
            # Create only this chunk's masks [chunk_size, 1, H, W]
            chunk_thresholds = thresholds[i:end_idx]  # [chunk_size, 1, 1]
            chunk_masks = (rank_map < chunk_thresholds).unsqueeze(1).float()
            mini_batch = img_expanded * chunk_masks  # [chunk_size, C, H, W]
            
            with torch.no_grad():
                outputs = self.clean_model(mini_batch)
                
                # Get probabilities for target class
                batch_confidences = torch.softmax(outputs, dim=1)[:, predicted_label]
                confidences.extend(batch_confidences.tolist())
        
        # 5. Calculate AUC
        if not confidences:
            return [], 0.0
            
        auc = float(np.trapz(confidences)) / len(confidences)
        return confidences, auc
    def compute_deletion_auc(self, image_tensor: torch.Tensor, 
                            saliency_map: np.ndarray, predicted_label: int,
                            step_size: int = 50, batch_size: int = 2048) -> Tuple[List[float], float]:
        """
        Compute deletion AUC by progressively removing pixels in order of importance.
        Optimized with full vectorization for large batch processing.
        """
        # Ensure image tensor is 3D [C, H, W]
        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
            
        # Ensure saliency map is the right shape
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]  # Take first channel if RGB
        
        # 1. Flatten saliency map and get pixel indices sorted by importance
        flat_saliency = saliency_map.flatten()
        pixel_indices_np = np.argsort(flat_saliency)[::-1]  # Descending order, high to low
        
        c, h, w = image_tensor.shape
        n_pixels = h * w
        pixel_indices = torch.from_numpy(pixel_indices_np.copy()).to(self.device).long()
        
        # 2. Determine steps
        steps = list(range(0, n_pixels, step_size))
        if steps[-1] != n_pixels:
            steps.append(n_pixels)
        
        num_steps = len(steps)
        
        # 3. Create a Rank Map for Vectorized Masking
        rank_map_flat = torch.zeros(n_pixels, device=self.device, dtype=torch.long)
        rank_indices = pixel_indices
        rank_values = torch.arange(n_pixels, device=self.device)
        rank_map_flat[rank_indices] = rank_values
        
        # Reshape to 2D [H, W]
        rank_map = rank_map_flat.view(1, h, w) 
        
        # Create Thresholds Tensor [num_steps, 1, 1]
        thresholds = torch.tensor(steps, device=self.device).view(-1, 1, 1)
        
        # 4. Run Batched Inference - process in chunks to avoid materializing full [num_steps, C, H, W]
        confidences = []
        self.clean_model.eval()
        img_expanded = image_tensor.unsqueeze(0)  # [1, C, H, W]
        
        for i in range(0, num_steps, batch_size):
            end_idx = min(i + batch_size, num_steps)
            # Create only this chunk's masks (deletion: keep pixels where rank >= threshold)
            chunk_thresholds = thresholds[i:end_idx]  # [chunk_size, 1, 1]
            chunk_masks = (rank_map >= chunk_thresholds).unsqueeze(1).float()
            mini_batch = img_expanded * chunk_masks
            
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = self.clean_model(mini_batch)
                else:
                    outputs = self.clean_model(mini_batch)
                
                # Get probabilities for target class
                batch_confidences = torch.softmax(outputs, dim=1)[:, predicted_label]
                confidences.extend(batch_confidences.tolist())
        
        # 6. Calculate AUC
        if not confidences:
            return [], 0.0
            
        auc = float(np.trapz(confidences)) / len(confidences)
        return confidences, auc
    
    def evaluate_road(self, image_tensor: torch.Tensor, 
                     saliency_map: np.ndarray, predicted_label: int,
                     thresholds: List[int] = [20, 40, 60, 80],
                     imputation: str = "blur",
                     road_seed: int = 0) -> Dict[str, float]:
        """
        Evaluate ROAD at multiple thresholds for MoRF and LeRF directions.

        Returns keys:
          road_{p}_morf, road_{p}_lerf for each percentile p,
          road_morf / road_lerf (means), and road = (lerf - morf) / 2
          matching ROADCombined's combined score.
        """
        import torchvision.transforms.functional as TF

        # Seed before any stochastic imputation path (blur is deterministic today).
        torch.manual_seed(int(road_seed))
        np.random.seed(int(road_seed))

        image_tensor = image_tensor.to(self.device).clone()
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[0]

        if imputation == "blur":
            imputation_tensor = TF.gaussian_blur(image_tensor, kernel_size=11, sigma=5.0)
        else:
            imputation_tensor = torch.zeros_like(image_tensor)

        imputation_flat = imputation_tensor.view(imputation_tensor.shape[0], -1)
        batch_images = [image_tensor]
        directions = []  # (p, direction) parallel to batch_images[1:]

        for direction in ("morf", "lerf"):
            for p in thresholds:
                if direction == "morf":
                    # Remove top-p% most salient pixels
                    percentile_val = np.percentile(saliency_map, 100 - p)
                    mask = saliency_map >= percentile_val
                else:
                    # Remove bottom-p% least salient pixels
                    percentile_val = np.percentile(saliency_map, p)
                    mask = saliency_map <= percentile_val

                mask_indices = torch.where(
                    torch.from_numpy(mask.flatten()).to(self.device)
                )[0]
                modified = image_tensor.clone()
                modified_flat = modified.view(modified.shape[0], -1)
                modified_flat[:, mask_indices] = imputation_flat[:, mask_indices]
                batch_images.append(modified)
                directions.append((p, direction))

        batch_tensor = torch.stack(batch_images)
        self.clean_model.eval()
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = self.clean_model(batch_tensor)
            else:
                outputs = self.clean_model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)[:, predicted_label]

        original_conf = probs[0].item()
        results: Dict[str, float] = {"conf": float(original_conf)}
        morf_vals: List[float] = []
        lerf_vals: List[float] = []

        for i, (p, direction) in enumerate(directions):
            modified_conf = probs[i + 1].item()
            # Confidence drop (clipped); LeRF drop is often small / negative → clip
            diff = float(max(0.0, original_conf - modified_conf))
            key = f"road_{p}_{direction}"
            results[key] = diff
            # Keep legacy alias for MoRF so old summary code still works
            if direction == "morf":
                results[f"road_{p}"] = diff
                morf_vals.append(diff)
            else:
                lerf_vals.append(diff)

        results["road_morf"] = float(np.mean(morf_vals)) if morf_vals else 0.0
        results["road_lerf"] = float(np.mean(lerf_vals)) if lerf_vals else 0.0
        # Combined ROAD (higher is better), same convention as ROADCombined
        results["road"] = float((results["road_lerf"] - results["road_morf"]) / 2.0)
        return results

    def _evaluate_saliency_map(self, image_tensor: torch.Tensor, saliency_map: np.ndarray, 
                              predicted_label: int, step_size: int = 50, verbose: bool = False, 
                              batch_size: int = 64,
                              road_seed: int = 0,
                              road_imputation: str = "blur") -> Dict[str, float]:
        """
        Evaluate a single saliency map using insertion AUC, deletion AUC, and ROAD metrics.
        Returns a dictionary of all metrics (including per-threshold MoRF/LeRF).
        """
        try:
            results = {}

            _, insertion_auc = self.compute_insertion_auc(
                image_tensor, saliency_map, predicted_label, step_size, batch_size
            )
            results["insertion_auc"] = insertion_auc

            _, deletion_auc = self.compute_deletion_auc(
                image_tensor, saliency_map, predicted_label, step_size, batch_size
            )
            results["deletion_auc"] = deletion_auc

            road_scores = self.evaluate_road(
                image_tensor, saliency_map, predicted_label,
                thresholds=[20, 40, 60, 80],
                imputation=road_imputation,
                road_seed=road_seed,
            )
            results.update(road_scores)

            if verbose:
                print(f"        Insertion AUC: {insertion_auc:.4f}")
                print(f"        Deletion AUC: {deletion_auc:.4f}")
                print(f"        ROAD Scores: {road_scores}")

            return results

        except Exception as e:
            print(f"        Error in saliency evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {"insertion_auc": 0.0, "deletion_auc": 0.0, "road": 0.0}
    
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
    
    def _get_enhanced_cam_layers(self, layer_mode: str = "last") -> List[torch.nn.Module]:
        """Get target layers for CAM extraction (see common.layer_sets)."""
        from XAI_Enhancer_module.common.layer_sets import select_cam_layers

        arch = getattr(self, "model_name", None) or ""
        selected_layers = select_cam_layers(self.model, layer_mode, arch=arch)
        print(
            f"Layer set '{layer_mode}': selected {len(selected_layers)} "
            f"target module(s) (arch={arch})"
        )
        return selected_layers

    @staticmethod
    def _write_per_image_log(rows: List[Dict], output_path: str) -> str:
        """Persist per-image metrics to CSV (and Parquet when pyarrow/fastparquet is available)."""
        if not rows:
            return ""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df = pd.DataFrame(rows)
        csv_path = output_path if output_path.endswith(".csv") else f"{output_path}.csv"
        if not csv_path.endswith(".csv"):
            csv_path = output_path + ".csv"
        # Normalize: always write .csv; optionally sibling .parquet
        if output_path.endswith(".parquet"):
            csv_path = output_path[:-8] + ".csv"
            parquet_path = output_path
        elif output_path.endswith(".csv"):
            csv_path = output_path
            parquet_path = output_path[:-4] + ".parquet"
        else:
            csv_path = output_path + ".csv"
            parquet_path = output_path + ".parquet"

        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"Per-image metrics -> {csv_path} (+ {parquet_path})")
        except Exception:
            print(f"Per-image metrics -> {csv_path} (parquet skipped: no engine)")
        return csv_path

    def _row_from_metrics(
        self,
        *,
        image_id: str,
        method: str,
        predicted_label: int,
        metrics: Dict[str, float],
        layer_scores: Optional[np.ndarray] = None,
    ) -> Dict:
        road_keys = [k for k in metrics if k.startswith("road_")]
        row = {
            "image_id": image_id,
            "method": method,
            "pred": int(predicted_label),
            "conf": float(metrics.get("conf", np.nan)),
            "ins": float(metrics.get("insertion_auc", np.nan)),
            "del": float(metrics.get("deletion_auc", np.nan)),
            "road": float(metrics.get("road", np.nan)),
            "road_morf": float(metrics.get("road_morf", np.nan)),
            "road_lerf": float(metrics.get("road_lerf", np.nan)),
        }
        for k in sorted(road_keys):
            row[k] = float(metrics[k])
        if layer_scores is not None:
            for li, s in enumerate(np.asarray(layer_scores).reshape(-1)):
                row[f"S_l_{li}"] = float(s)
        return row
    
    def get_imagenet_images(self, max_images: int = 50, classes_filter: List[str] = None,
                          start_index: int = 0, end_index: int = None) -> Tuple[List[str], List[int], List[str]]:
        """
        Get ImageNet validation images with proper class filtering and batch support.
        
        Args:
            max_images: Maximum number of images to get (ignored if end_index is set)
            classes_filter: List of class names to filter (e.g., ['tench', 'goldfish'])
            start_index: Index to start collecting images from (for batch processing)
            end_index: Index to stop collecting images at (for batch processing)
            
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
            
            # Get images from this synset and SORT them for determinism
            synset_images = sorted(glob.glob(os.path.join(synset_dir, "*.JPEG")))
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
        
        # Limit total images if specified (legacy behavior if start/end not used)
        if end_index is not None:
            # Batch mode
            print(f"Batch mode: selecting images from index {start_index} to {end_index}")
            all_image_paths = all_image_paths[start_index:end_index]
            all_class_names = all_class_names[start_index:end_index]
        elif max_images > 0 and len(all_image_paths) > max_images:
            # Legacy max_images mode
            all_image_paths = all_image_paths[:max_images]
            all_class_names = all_class_names[:max_images]
        elif start_index > 0:
            # Start index only
            all_image_paths = all_image_paths[start_index:]
            all_class_names = all_class_names[start_index:]
        
        # Get predicted labels using the model
        predicted_labels = self._predict_batch(all_image_paths)
        
        print(f"Collected {len(all_image_paths)} ImageNet images")
        if classes_filter:
            print(f"Filtered classes: {classes_filter}")
        
        return all_image_paths, predicted_labels, all_class_names
    
    def _predict_batch(self, image_paths: List[str]) -> List[int]:
        """Predict labels for a batch of images"""
        predicted_labels = []
        
        self.clean_model.eval()
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc="Predicting labels"):
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    outputs = self.clean_model(image_tensor)
                    predicted_label = torch.argmax(outputs, dim=1).item()
                    predicted_labels.append(predicted_label)
                    
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    predicted_labels.append(0)  # Default to first class
        
        return predicted_labels
    
    def extract_enhanced_cam(self, input_data: Union[str, torch.Tensor], predicted_label: Union[int, List[int]]) -> Tuple[torch.Tensor, np.ndarray]:
        """Extract Enhanced CAM for a single ImageNet image or Batch."""
        if self.enhanced_cam_extractor is None:
            self.enhanced_cam_extractor = self.extractor_cls(
                self.model, 
                self.model_name, 
                self.conv_layers,
                cam_method=self.enhanced_cam_method,
                device_preference=str(self.device),
                **self.extractor_kwargs
            )
        
        # Extract Enhanced CAM
        # Now accepts batch tensor or path
        image_tensor, saliency_map = self.enhanced_cam_extractor.extract_saliency_map(
            input_data, predicted_label
        )
        
        # Ensure image tensor has batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Convert saliency map to numpy if it's a tensor
        # if isinstance(saliency_map, torch.Tensor):
        #     saliency_map = saliency_map.cpu().numpy()
        
        return image_tensor, saliency_map
    
    def evaluate_enhanced_cam(self, max_images: int = 50, step_size: int = 50, 
                            verbose: bool = True, classes_filter: List[str] = None,
                            start_index: int = 0, end_index: int = None,
                            return_raw_data: bool = False, batch_size: int = 64, num_workers: int = 4,
                            method_name: str = "EnhancedCAM",
                            per_image_path: Optional[str] = None,
                            road_seed: int = 0,
                            road_imputation: str = "blur") -> Dict[str, any]:
        """
        Evaluate Enhanced CAM method with proper AUC calculations.

        When ``per_image_path`` is set, writes one row per image (CSV + Parquet)
        so later stats (bootstrap / Wilcoxon) need no re-perturbation.
        """
        # Get ImageNet images and predictions
        image_paths, predicted_labels, class_names = self.get_imagenet_images(
            max_images=max_images, 
            classes_filter=classes_filter,
            start_index=start_index,
            end_index=end_index
        )
        
        # Auto-adjust verbosity for large datasets
        if len(image_paths) > 20 and verbose:
            print(f"📢 Large dataset detected ({len(image_paths)} images). Setting verbose=False for cleaner output.")
            verbose = False
        
        print(f"\nEvaluating {method_name} on {len(image_paths)} images...")
        print(f"Using step_size={step_size} and batch_size={batch_size} for proper evaluation...")
        print(f"Using {num_workers} workers for data loading...")
        
        if not verbose:
            print(f"Verbose mode OFF - only showing progress and summary.")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        per_image_rows: List[Dict] = []
        
        # Create dataset and loader (use passed batch_size; avoid large batches that cause OOM)
        dataset = _ImageNetDataset(image_paths, predicted_labels, class_names, self.transform)
        effective_batch_size = min(batch_size, len(dataset)) if len(dataset) > 0 else 1
        loader = DataLoader(
            dataset, 
            batch_size=effective_batch_size,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            prefetch_factor=2 if num_workers > 0 else None,
            shuffle=False
        )
        
        # Create progress description based on verbosity
        progress_desc = f"Processing {method_name}"
        
        start_time = time.time()
        images_processed = 0
        num_batches = len(loader)
        pbar = tqdm(loader, desc=progress_desc, total=num_batches, unit="batch")
        for i, (image_tensor, predicted_label, class_name, image_path_batch) in enumerate(pbar):
            try:
                current_batch_size = image_tensor.shape[0]
                
                if verbose:
                    print(f"\n--- Processing Batch {i+1} : {current_batch_size} images ---")
                
                pred_label_list = predicted_label.tolist()
                
                _, saliency_maps = self.extract_enhanced_cam(image_tensor, pred_label_list)
                layer_scores_batch = None
                if self.enhanced_cam_extractor is not None:
                    layer_scores_batch = getattr(
                        self.enhanced_cam_extractor, "last_layer_scores", None
                    )

                saliency_maps_np = saliency_maps.cpu().numpy()
                
                for b in range(current_batch_size):
                    img_t = image_tensor[b]
                    sal_np = saliency_maps_np[b]
                    lbl = pred_label_list[b]
                    img_id = image_path_batch[b]
                    if isinstance(img_id, (list, tuple)):
                        img_id = img_id[0]
                    
                    metrics = self._evaluate_saliency_map(
                        img_t, sal_np, lbl, step_size, verbose=False, batch_size=batch_size,
                        road_seed=road_seed, road_imputation=road_imputation,
                    )
                    
                    insertion_aucs.append(metrics['insertion_auc'])
                    deletion_aucs.append(metrics['deletion_auc'])
                    r_mean = metrics.get(
                        "road",
                        np.mean([v for k, v in metrics.items() if k.startswith("road_")]),
                    )
                    road_scores.append(r_mean)

                    ls = None
                    if layer_scores_batch is not None and b < len(layer_scores_batch):
                        ls = layer_scores_batch[b]
                    per_image_rows.append(
                        self._row_from_metrics(
                            image_id=str(img_id),
                            method=method_name,
                            predicted_label=lbl,
                            metrics=metrics,
                            layer_scores=ls,
                        )
                    )
                
                if verbose:
                    print(f"    Insertion AUC: {metrics['insertion_auc']:.4f}")
                    print(f"    Deletion AUC: {metrics['deletion_auc']:.4f}")
                    print(f"    ROAD Score (20%): {metrics.get('road_20', 0.0):.4f}")
                
                # Update progress bar with current means
                current_ins_mean = np.mean(insertion_aucs)
                current_del_mean = np.mean(deletion_aucs)
                current_road_mean = np.mean(road_scores)
                
                pbar.set_postfix({
                    'Ins': f"{current_ins_mean:.3f}",
                    'Del': f"{current_del_mean:.3f}",
                    'ROAD': f"{current_road_mean:.3f}"
                })
                
                images_processed += current_batch_size
                
                # Periodic logging for non-verbose mode (every 50 batches)
                if not verbose and (i + 1) % 50 == 0:
                     elapsed = time.time() - start_time
                     avg_time_per_img = elapsed / images_processed
                     remaining_imgs = len(dataset) - images_processed
                     eta = remaining_imgs * avg_time_per_img
                     
                     print(f"\n[Batch {i+1}/{num_batches}] {images_processed}/{len(dataset)} images - "
                           f"Ins: {current_ins_mean:.4f}, "
                           f"Del: {current_del_mean:.4f}, "
                           f"ROAD: {current_road_mean:.4f} | "
                           f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s ({1/avg_time_per_img:.2f} img/s)")
                
            except Exception as e:
                print(f"Error processing batch {i} (paths: {image_path_batch}): {e}")
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
            'classes_evaluated': classes_filter if classes_filter else 'All ImageNet classes',
            'step_size': step_size,
            'road_seed': road_seed,
            'road_imputation': road_imputation,
            'method': method_name,
        }
        
        if per_image_path:
            results["per_image_path"] = self._write_per_image_log(per_image_rows, per_image_path)
        elif return_raw_data:
            results["per_image_rows"] = per_image_rows

        if return_raw_data:
            results.update({
                'insertion_aucs': insertion_aucs,
                'deletion_aucs': deletion_aucs,
                'road_scores': road_scores
            })
            
        return results
    
    def evaluate_method(self, cam_method_name: str, max_images: int = 50, 
                       classes_filter: List[str] = None,
                       start_index: int = 0, end_index: int = None,
                       return_raw_data: bool = False, batch_size: int = 64, num_workers: int = 4,
                       step_size: int = 224,
                       per_image_path: Optional[str] = None,
                       road_seed: int = 0,
                       road_imputation: str = "blur") -> Dict[str, any]:
        """
        Evaluate a standard CAM method.

        When ``per_image_path`` is set, writes one row per image (CSV + Parquet).
        """

        # Get ImageNet images and predictions
        image_paths, predicted_labels, class_names = self.get_imagenet_images(
            max_images=max_images, 
            classes_filter=classes_filter,
            start_index=start_index,
            end_index=end_index
        )
        
        print(f"\nEvaluating {cam_method_name} on {len(image_paths)} images...")
        print(f"Using step_size={step_size}, {num_workers} workers for data loading...")
        
        insertion_aucs = []
        deletion_aucs = []
        road_scores = []
        per_image_rows: List[Dict] = []
        
        # Create dataset and loader
        dataset = _ImageNetDataset(image_paths, predicted_labels, class_names, self.transform)
        loader = DataLoader(
            dataset, 
            batch_size=1,  # Keep batch size 1
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            prefetch_factor=2 if num_workers > 0 else None,
            shuffle=False
        )

        start_time = time.time()
        pbar = tqdm(loader, desc=f"Processing {cam_method_name}", total=len(dataset))
        for i, (image_tensor, predicted_label, class_name, image_path_batch) in enumerate(pbar):
            try:
                # Unpack batch
                image_tensor = image_tensor.squeeze(0)
                predicted_label = predicted_label.item()
                image_path = image_path_batch[0]
                
                # Extract standard CAM using pytorch-grad-cam
                saliency_map = self._extract_standard_cam(
                    image_path, predicted_label, cam_method_name
                )
                
                # Use pre-fetched image_tensor for evaluation
                # Evaluate saliency map
                metrics = self._evaluate_saliency_map(
                    image_tensor, saliency_map, predicted_label,
                    step_size=step_size, verbose=False, batch_size=batch_size,
                    road_seed=road_seed, road_imputation=road_imputation,
                )
                
                insertion_aucs.append(metrics['insertion_auc'])
                deletion_aucs.append(metrics['deletion_auc'])
                road_scores.append(metrics.get('road', metrics.get('road_20', 0.0)))
                per_image_rows.append(
                    self._row_from_metrics(
                        image_id=str(image_path),
                        method=cam_method_name,
                        predicted_label=predicted_label,
                        metrics=metrics,
                    )
                )
                
                # Update progress bar with current means
                current_ins_mean = np.mean(insertion_aucs)
                current_del_mean = np.mean(deletion_aucs)
                current_road_mean = np.mean(road_scores)
                
                pbar.set_postfix({
                    'Ins': f"{current_ins_mean:.3f}",
                    'Del': f"{current_del_mean:.3f}",
                    'ROAD': f"{current_road_mean:.3f}"
                })
                
                # Periodic logging for non-verbose mode (every 50 images)
                # Note: evaluate_method doesn't have a verbose arg exposed in the loop same as above, but we can default check
                if (i + 1) % 50 == 0:
                     elapsed = time.time() - start_time
                     avg_time_per_img = elapsed / (i + 1)
                     remaining_imgs = len(dataset) - (i + 1)
                     eta = remaining_imgs * avg_time_per_img
                     
                     print(f"\n[Batch {i+1}/{len(dataset)}] Means - "
                           f"Ins: {current_ins_mean:.4f}, "
                           f"Del: {current_del_mean:.4f}, "
                           f"ROAD: {current_road_mean:.4f} | "
                           f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s ({1/avg_time_per_img:.2f} img/s)")
                
            except Exception as e:
                print(f"Error processing batch {i} (paths: {image_path_batch}): {e}")
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
            'classes_evaluated': classes_filter if classes_filter else 'All ImageNet classes',
            'step_size': step_size,
            'road_seed': road_seed,
            'road_imputation': road_imputation,
            'method': cam_method_name,
        }
        
        if per_image_path:
            results["per_image_path"] = self._write_per_image_log(per_image_rows, per_image_path)
        elif return_raw_data:
            results["per_image_rows"] = per_image_rows

        if return_raw_data:
            results.update({
                'insertion_aucs': insertion_aucs,
                'deletion_aucs': deletion_aucs,
                'road_scores': road_scores
            })

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
