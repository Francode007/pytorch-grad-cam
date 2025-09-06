#!/usr/bin/env python3
"""
Model Loader Utility - Utility for loading pre-downloaded models from cache directory.
This module provides functionality to load PyTorch models from a specified cache directory.
"""

import os
import torch
import torchvision.models as models
from pathlib import Path
from typing import Optional


class ModelLoader:
    """Utility class for loading pre-downloaded models from cache directory."""
    
    def __init__(self, model_cache_dir: str = "../pytorch_models/"):
        """
        Initialize the model loader.
        
        Args:
            model_cache_dir: Base directory where pre-downloaded models are stored.
                           Should contain hub/checkpoints/ subdirectory.
        """
        self.model_cache_dir = Path(model_cache_dir).resolve()
        self.hub_dir = self.model_cache_dir / "hub"
        self.checkpoints_dir = self.hub_dir / "checkpoints"
        
        print(f"ModelLoader initialized with cache directory: {self.model_cache_dir}")
        
        # Verify cache directory exists
        if not self.model_cache_dir.exists():
            raise FileNotFoundError(f"Model cache directory does not exist: {self.model_cache_dir}")
        
        if not self.hub_dir.exists():
            print(f"Warning: hub directory not found at {self.hub_dir}")
        elif not self.checkpoints_dir.exists():
            print(f"Warning: checkpoints directory not found at {self.checkpoints_dir}")
        else:
            print(f"✅ Model cache directory structure verified: {self.checkpoints_dir}")
            # List some of the cached files for verification
            pth_files = list(self.checkpoints_dir.glob("*.pth"))
            if pth_files:
                print(f"   Found {len(pth_files)} cached model files")
            else:
                print(f"   No .pth files found in checkpoints directory")
    
    def load_pretrained_model(self, model_name: str) -> torch.nn.Module:
        """
        Load a pre-trained ImageNet model using the cached weights.
        
        Args:
            model_name: Name of the model to load (e.g., 'resnet50', 'vgg16')
            
        Returns:
            torch.nn.Module: The loaded model
            
        Raises:
            ValueError: If the model name is not supported
            FileNotFoundError: If the model cache is not accessible
        """
        # Set TORCH_HOME to use our cached models (the base directory, not hub/checkpoints)
        # PyTorch will automatically look in hub/checkpoints/ subdirectory
        original_torch_home = os.environ.get('TORCH_HOME')
        os.environ['TORCH_HOME'] = str(self.model_cache_dir)
        
        try:
            # Get the model loader function dynamically
            if not hasattr(models, model_name):
                raise ValueError(f"Unsupported model: {model_name}")
            
            model_loader = getattr(models, model_name)
            
            # Load with modern weights parameter (replaces deprecated pretrained=True)
            try:
                # Try modern approach first
                model = model_loader(weights='IMAGENET1K_V1')
                print(f"✅ Loaded {model_name} using cached weights from {self.model_cache_dir}")
            except TypeError:
                # Fallback for older models that might not support weights parameter
                model = model_loader(pretrained=True)
                print(f"✅ Loaded {model_name} using cached weights (fallback method)")
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name} from cache: {e}")
            print(f"Attempting to download from internet...")
            
            # Fallback: try to load without cache (will download)
            if original_torch_home:
                os.environ['TORCH_HOME'] = original_torch_home
            else:
                os.environ.pop('TORCH_HOME', None)
            
            model_loader = getattr(models, model_name)
            try:
                model = model_loader(weights='IMAGENET1K_V1')
            except TypeError:
                model = model_loader(pretrained=True)
            
            print(f"⚠️ Downloaded {model_name} from internet (cache not used)")
            return model
            
        finally:
            # Restore original TORCH_HOME
            if original_torch_home:
                os.environ['TORCH_HOME'] = original_torch_home
            elif 'TORCH_HOME' in os.environ:
                os.environ.pop('TORCH_HOME')
    
    def list_available_models(self) -> list:
        """
        List models that should be available in the cache.
        
        Returns:
            list: List of supported model names
        """
        supported_models = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'vgg16', 'vgg19',
            'densenet121', 'densenet169', 'densenet201',
            'mobilenet_v2', 'mobilenet_v3_large',
            'efficientnet_b0', 'efficientnet_b4'
        ]
        return supported_models
    
    def check_cache_status(self) -> dict:
        """
        Check the status of the model cache directory.
        
        Returns:
            dict: Status information about the cache
        """
        status = {
            'cache_dir_exists': self.model_cache_dir.exists(),
            'hub_dir_exists': self.hub_dir.exists(),
            'checkpoints_dir_exists': self.checkpoints_dir.exists(),
            'cache_dir_path': str(self.model_cache_dir),
            'hub_dir_path': str(self.hub_dir),
            'checkpoints_dir_path': str(self.checkpoints_dir)
        }
        
        if self.checkpoints_dir.exists():
            # Count .pth files in checkpoints directory
            pth_files = list(self.checkpoints_dir.glob("*.pth"))
            status['num_cached_files'] = len(pth_files)
            status['cached_files'] = [f.name for f in pth_files]
        else:
            status['num_cached_files'] = 0
            status['cached_files'] = []
        
        return status
    
    def print_cache_status(self):
        """Print a human-readable status of the model cache."""
        status = self.check_cache_status()
        
        print(f"\n📁 Model Cache Status:")
        print(f"{'='*50}")
        print(f"Cache directory: {status['cache_dir_path']}")
        print(f"  ✅ Exists: {status['cache_dir_exists']}")
        
        if status['cache_dir_exists']:
            print(f"Hub directory: {status['hub_dir_path']}")
            print(f"  ✅ Exists: {status['hub_dir_exists']}")
            
            if status['hub_dir_exists']:
                print(f"Checkpoints directory: {status['checkpoints_dir_path']}")
                print(f"  ✅ Exists: {status['checkpoints_dir_exists']}")
                
                if status['checkpoints_dir_exists']:
                    print(f"  📦 Cached model files: {status['num_cached_files']}")
                    if status['cached_files']:
                        print(f"  Files: {', '.join(status['cached_files'][:5])}")
                        if len(status['cached_files']) > 5:
                            print(f"         ... and {len(status['cached_files']) - 5} more")
