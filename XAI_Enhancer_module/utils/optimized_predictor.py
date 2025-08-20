"""
Optimized prediction module for efficient label prediction and caching.
This module pre-computes predictions to avoid redundant forward passes.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import os
from tqdm import tqdm

from XAI_Enhancer_module.utils.model_utils import get_val_dataloader, test_model, IDX_TO_CLASS, CLASS_TO_IDX


class OptimizedPredictor:
    """
    Optimized predictor that pre-computes and caches model predictions.
    """
    
    def __init__(self, model, model_name: str, device_preference: str = "auto"):
        """
        Initialize the predictor.
        
        Args:
            model: The trained model for prediction
            model_name: Name of the model
            device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        """
        self.model = model
        self.model_name = model_name
        self.device_preference = device_preference
        self.device = next(model.parameters()).device
        
        # Cache for predictions
        self._prediction_cache = {}
        
    def predict_batch(self, 
                     image_paths: List[str],
                     use_cache: bool = True) -> Tuple[List[int], List[str]]:
        """
        Predict labels for a batch of images efficiently.
        
        Args:
            image_paths: List of image file paths
            use_cache: Whether to use caching
            
        Returns:
            Tuple of (predicted_labels, predicted_class_names)
        """
        if use_cache:
            # Check cache first
            uncached_paths = []
            cached_predictions = {}
            
            for path in image_paths:
                if path in self._prediction_cache:
                    cached_predictions[path] = self._prediction_cache[path]
                else:
                    uncached_paths.append(path)
        else:
            uncached_paths = image_paths
            cached_predictions = {}
        
        # Predict uncached images
        if uncached_paths:
            new_predictions = self._predict_images(uncached_paths)
            if use_cache:
                # Update cache
                for path, pred in zip(uncached_paths, new_predictions):
                    self._prediction_cache[path] = pred
        else:
            new_predictions = []
        
        # Combine cached and new predictions in original order
        predicted_labels = []
        predicted_class_names = []
        
        new_pred_iter = iter(new_predictions)
        for path in image_paths:
            if path in cached_predictions:
                pred_idx = cached_predictions[path]
            else:
                pred_idx = next(new_pred_iter)
            
            predicted_labels.append(pred_idx)
            predicted_class_names.append(IDX_TO_CLASS[pred_idx])
        
        return predicted_labels, predicted_class_names
    
    def _predict_images(self, image_paths: List[str]) -> List[int]:
        """
        Internal method to predict labels for uncached images.
        
        Args:
            image_paths: List of image paths to predict
            
        Returns:
            List of predicted label indices
        """
        # Create a temporary dataset for these images
        from XAI_Enhancer_module.utils.model_utils import IBSValDataset
        
        dataset = IBSValDataset(image_paths, self.model_name)
        dataloader = DataLoader(
            dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=0
        )
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for images, names, filepaths in tqdm(dataloader, desc="Predicting labels"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs.data, 1)
                predictions.extend(preds.cpu().numpy().tolist())
        
        return predictions
    
    def predict_dataloader(self, dataloader: DataLoader) -> Tuple[List[int], List[str], List[str]]:
        """
        Predict labels for an entire dataloader.
        
        Args:
            dataloader: DataLoader with images
            
        Returns:
            Tuple of (predicted_labels, predicted_class_names, image_paths)
        """
        predictions = []
        image_paths = []
        
        self.model.eval()
        
        with torch.no_grad():
            for images, names, filepaths in tqdm(dataloader, desc="Batch prediction"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs.data, 1)
                
                predictions.extend(preds.cpu().numpy().tolist())
                image_paths.extend(filepaths)
        
        predicted_class_names = [IDX_TO_CLASS[pred] for pred in predictions]
        return predictions, predicted_class_names, image_paths
    
    def clear_cache(self):
        """Clear the prediction cache to free memory."""
        self._prediction_cache.clear()


def get_optimized_predictions(model_name: str, 
                            image_paths: List[str] = None,
                            use_validation_set: bool = True,
                            device_preference: str = "auto",
                            dataset_type: str = "ibs") -> Tuple[List[int], List[str], List[str]]:
    """
    Get optimized predictions for images.
    
    Args:
        model_name: Name of the model to use
        image_paths: Optional list of specific image paths
        use_validation_set: Whether to use the default validation set
        device_preference: Device preference ("auto", "cuda", "mps", "cpu")
        dataset_type: "ibs" or "imagenet"
        
    Returns:
        Tuple of (predicted_labels, predicted_class_names, image_paths)
    """
    # Load the model
    model = test_model(model_name, device_preference=device_preference, dataset_type=dataset_type)
    predictor = OptimizedPredictor(model, model_name, device_preference=device_preference)
    
    if use_validation_set and image_paths is None:
        # Use the validation dataloader
        val_dataloader = get_val_dataloader(model_name, dataset_type=dataset_type)
        predictions, class_names, paths = predictor.predict_dataloader(val_dataloader)
    elif image_paths is not None:
        # Use specific image paths
        predictions, class_names = predictor.predict_batch(image_paths)
        paths = image_paths
    else:
        raise ValueError("Either use_validation_set must be True or image_paths must be provided")
    
    return predictions, class_names, paths


class PredictionManager:
    """
    Manager class for handling predictions across multiple models and datasets.
    """
    
    def __init__(self, device_preference: str = "auto"):
        self.device_preference = device_preference
        self.predictors = {}
        self.cached_results = {}
    
    def get_predictor(self, model_name: str) -> OptimizedPredictor:
        """
        Get or create a predictor for the specified model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            OptimizedPredictor instance
        """
        if model_name not in self.predictors:
            model = test_model(model_name, device_preference=self.device_preference)
            self.predictors[model_name] = OptimizedPredictor(model, model_name, self.device_preference)
        
        return self.predictors[model_name]
    
    def predict_for_model(self, 
                         model_name: str,
                         image_paths: List[str] = None,
                         use_cache: bool = True) -> Tuple[List[int], List[str], List[str]]:
        """
        Get predictions for a specific model.
        
        Args:
            model_name: Name of the model
            image_paths: Optional list of image paths
            use_cache: Whether to use caching
            
        Returns:
            Tuple of (predicted_labels, predicted_class_names, image_paths)
        """
        cache_key = f"{model_name}_{hash(tuple(image_paths) if image_paths else 'validation')}"
        
        if use_cache and cache_key in self.cached_results:
            return self.cached_results[cache_key]
        
        result = get_optimized_predictions(model_name, image_paths)
        
        if use_cache:
            self.cached_results[cache_key] = result
        
        return result
    
    def clear_all_caches(self):
        """Clear all caches to free memory."""
        for predictor in self.predictors.values():
            predictor.clear_cache()
        self.cached_results.clear()
