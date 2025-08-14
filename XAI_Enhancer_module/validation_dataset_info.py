#!/usr/bin/env python3
"""
Demonstration script showing how to use the entire validation dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.utils.model_utils import get_validation_paths, TRAIN_DATA_PATH

def show_validation_dataset_info():
    """Show information about the validation dataset."""
    try:
        # Get all validation paths
        all_paths = get_validation_paths(TRAIN_DATA_PATH)
        
        print("=== VALIDATION DATASET INFORMATION ===")
        print(f"Total validation images: {len(all_paths)}")
        print(f"First 3 image paths:")
        for i, path in enumerate(all_paths[:3]):
            print(f"  {i+1}. {path}")
        
        if len(all_paths) > 3:
            print(f"  ... and {len(all_paths) - 3} more images")
            
        print(f"\nTo use the ENTIRE validation dataset:")
        print(f"  python modular_xai_evaluation.py --model resnet18 --max-images -1")
        print(f"  python all_layer_analysis.py --model resnet18 --max-images -1")
        
        print(f"\nRecommended usage for different dataset sizes:")
        if len(all_paths) <= 20:
            print(f"  Small dataset ({len(all_paths)} images): Use default verbose mode")
            print(f"  python modular_xai_evaluation.py --model resnet18 --max-images -1")
        elif len(all_paths) <= 100:
            print(f"  Medium dataset ({len(all_paths)} images): Use quiet mode")
            print(f"  python modular_xai_evaluation.py --model resnet18 --max-images -1 --quiet")
        else:
            print(f"  Large dataset ({len(all_paths)} images): Definitely use quiet mode")
            print(f"  python modular_xai_evaluation.py --model resnet18 --max-images -1 --quiet")
            
    except Exception as e:
        print(f"Error accessing validation dataset: {e}")
        print(f"Make sure your data path is correctly configured in model_utils.py")

if __name__ == "__main__":
    show_validation_dataset_info()
