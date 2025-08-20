#!/usr/bin/env python3
"""
ImageNet utilities for class mappings and dataset handling.
This module provides utilities specific to ImageNet dataset evaluation.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ImageNet class mapping (synset ID to class index)
IMAGENET_SYNSET_TO_IDX = {}
IMAGENET_IDX_TO_SYNSET = {}
IMAGENET_IDX_TO_LABEL = {}

def load_imagenet_class_mappings():
    """
    Load ImageNet class mappings from the LOC_synset_mapping.txt file.
    
    Returns:
        Tuple of (synset_to_idx, idx_to_synset, idx_to_label) dictionaries
    """
    global IMAGENET_SYNSET_TO_IDX, IMAGENET_IDX_TO_SYNSET, IMAGENET_IDX_TO_LABEL
    
    # Check if already loaded
    if IMAGENET_SYNSET_TO_IDX:
        return IMAGENET_SYNSET_TO_IDX, IMAGENET_IDX_TO_SYNSET, IMAGENET_IDX_TO_LABEL
    
    synset_to_idx = {}
    idx_to_synset = {}
    idx_to_label = {}
    
    # Get the path to the synset mapping file
    current_dir = Path(__file__).parent
    synset_file = current_dir / "LOC_synset_mapping.txt"
    
    try:
        with open(synset_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)  # Split on first space only
                    if len(parts) >= 2:
                        synset_id = parts[0]
                        label = parts[1]
                        
                        synset_to_idx[synset_id] = idx
                        idx_to_synset[idx] = synset_id
                        idx_to_label[idx] = label
                    else:
                        print(f"Warning: Malformed line {idx + 1}: {line}")
        
        print(f"✅ Loaded {len(synset_to_idx)} ImageNet class mappings from {synset_file}")
        
    except FileNotFoundError:
        print(f"⚠️  Warning: {synset_file} not found. Creating fallback mappings.")
        # Fallback: create basic mappings
        for i in range(1000):
            synset_id = f"n{i:08d}"
            synset_to_idx[synset_id] = i
            idx_to_synset[i] = synset_id
            idx_to_label[i] = f"class_{i}"
    
    except Exception as e:
        print(f"⚠️  Error loading synset mappings: {e}")
        # Fallback: create basic mappings
        for i in range(1000):
            synset_id = f"n{i:08d}"
            synset_to_idx[synset_id] = i
            idx_to_synset[i] = synset_id
            idx_to_label[i] = f"class_{i}"
    
    # Store in global variables
    IMAGENET_SYNSET_TO_IDX = synset_to_idx
    IMAGENET_IDX_TO_SYNSET = idx_to_synset
    IMAGENET_IDX_TO_LABEL = idx_to_label
    
    return synset_to_idx, idx_to_synset, idx_to_label


def get_class_from_path(image_path: str) -> int:
    """
    Extract class index from ImageNet image path using proper synset mappings.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Class index (0-999)
    """
    # Extract synset from path (e.g., .../n01440764/image.JPEG -> n01440764)
    synset = os.path.basename(os.path.dirname(image_path))
    
    # Ensure mappings are loaded
    if not IMAGENET_SYNSET_TO_IDX:
        load_imagenet_class_mappings()
    
    # Get class index from synset
    if synset in IMAGENET_SYNSET_TO_IDX:
        return IMAGENET_SYNSET_TO_IDX[synset]
    
    # Fallback: if synset not found in mapping, try to get all synsets and find position
    try:
        from XAI_Enhancer_module.utils.model_utils import IMAGENET_VAL_PATH
        import glob
        
        # Get all class directories and sort them
        class_dirs = sorted([os.path.basename(d) for d in glob.glob(os.path.join(IMAGENET_VAL_PATH, 'n*'))])
        
        if synset in class_dirs:
            class_idx = class_dirs.index(synset)
            print(f"⚠️  Synset {synset} not in mapping file, using directory position: {class_idx}")
            return class_idx
        
    except Exception as e:
        print(f"⚠️  Error finding class for synset {synset}: {e}")
    
    # Final fallback
    print(f"⚠️  Unknown synset {synset}, returning class 0")
    return 0


def get_imagenet_class_info(class_idx: int) -> Tuple[str, str]:
    """
    Get synset and label for a class index.
    
    Args:
        class_idx: Class index (0-999)
        
    Returns:
        Tuple of (synset, label)
    """
    if not IMAGENET_IDX_TO_SYNSET:
        load_imagenet_class_mappings()
    
    synset = IMAGENET_IDX_TO_SYNSET.get(class_idx, f"unknown_{class_idx}")
    label = IMAGENET_IDX_TO_LABEL.get(class_idx, f"unknown_class_{class_idx}")
    
    return synset, label


def get_readable_class_name(class_idx: int) -> str:
    """
    Get a readable class name for a given class index.
    
    Args:
        class_idx: Class index (0-999)
        
    Returns:
        Readable class name
    """
    synset, label = get_imagenet_class_info(class_idx)
    
    # Clean up the label (take the first part before comma for shorter names)
    if ',' in label:
        primary_name = label.split(',')[0].strip()
    else:
        primary_name = label
    
    return primary_name


def validate_synset_mapping() -> Dict:
    """
    Validate the synset mapping by checking against actual directories.
    
    Returns:
        Dictionary with validation results
    """
    if not IMAGENET_SYNSET_TO_IDX:
        load_imagenet_class_mappings()
    
    try:
        from XAI_Enhancer_module.utils.model_utils import IMAGENET_VAL_PATH
        import glob
        
        # Get actual directories
        actual_dirs = set([os.path.basename(d) for d in glob.glob(os.path.join(IMAGENET_VAL_PATH, 'n*'))])
        mapped_synsets = set(IMAGENET_SYNSET_TO_IDX.keys())
        
        # Find matches and mismatches
        matched = actual_dirs.intersection(mapped_synsets)
        missing_from_mapping = actual_dirs - mapped_synsets
        extra_in_mapping = mapped_synsets - actual_dirs
        
        results = {
            'total_actual_dirs': len(actual_dirs),
            'total_mapped_synsets': len(mapped_synsets),
            'matched_synsets': len(matched),
            'missing_from_mapping': len(missing_from_mapping),
            'extra_in_mapping': len(extra_in_mapping),
            'coverage_percent': (len(matched) / len(actual_dirs)) * 100 if actual_dirs else 0,
            'sample_missing': list(missing_from_mapping)[:10],
            'sample_extra': list(extra_in_mapping)[:10]
        }
        
        return results
        
    except Exception as e:
        return {'error': str(e)}


def sample_imagenet_images(data_path: str, 
                          num_classes: int = 10, 
                          images_per_class: int = 5,
                          random_seed: Optional[int] = None) -> List[str]:
    """
    Sample a subset of ImageNet images for evaluation with proper class information.
    
    Args:
        data_path: Path to ImageNet validation directory
        num_classes: Number of classes to sample
        images_per_class: Number of images per class
        random_seed: Random seed for reproducible sampling
        
    Returns:
        List of sampled image paths with class information
    """
    import glob
    import random
    
    if random_seed is not None:
        random.seed(random_seed)
    
    # Ensure mappings are loaded
    if not IMAGENET_SYNSET_TO_IDX:
        load_imagenet_class_mappings()
    
    class_dirs = sorted(glob.glob(os.path.join(data_path, 'n*')))
    sampled_dirs = random.sample(class_dirs, min(num_classes, len(class_dirs)))
    sampled_paths = []
    
    print(f"📸 Sampling {images_per_class} images from {len(sampled_dirs)} classes:")
    
    for class_dir in sampled_dirs:
        synset = os.path.basename(class_dir)
        class_idx = get_class_from_path(class_dir + "/dummy.jpg")  # Get class index
        class_name = get_readable_class_name(class_idx)
        
        images = glob.glob(os.path.join(class_dir, '*.JPEG'))
        sampled_images = random.sample(images, min(images_per_class, len(images)))
        sampled_paths.extend(sampled_images)
        
        print(f"  {synset} ({class_idx}): {class_name} - {len(sampled_images)} images")
    
    print(f"✅ Total sampled: {len(sampled_paths)} images")
    return sampled_paths


def get_imagenet_validation_stats(data_path: str) -> Dict:
    """
    Get statistics about the ImageNet validation dataset.
    
    Args:
        data_path: Path to ImageNet validation directory
        
    Returns:
        Dictionary with dataset statistics
    """
    import glob
    
    class_dirs = glob.glob(os.path.join(data_path, 'n*'))
    total_images = 0
    images_per_class = []
    
    for class_dir in class_dirs:
        images = glob.glob(os.path.join(class_dir, '*.JPEG'))
        class_image_count = len(images)
        images_per_class.append(class_image_count)
        total_images += class_image_count
    
    stats = {
        'total_classes': len(class_dirs),
        'total_images': total_images,
        'avg_images_per_class': total_images / len(class_dirs) if class_dirs else 0,
        'min_images_per_class': min(images_per_class) if images_per_class else 0,
        'max_images_per_class': max(images_per_class) if images_per_class else 0,
        'sample_synsets': [os.path.basename(d) for d in sorted(class_dirs)[:5]]
    }
    
    return stats


# Initialize mappings when module is imported
load_imagenet_class_mappings()


if __name__ == "__main__":
    # Test the utilities
    print("🧪 Testing ImageNet utilities...")
    
    # Test synset mapping loading
    print("\n1. Testing synset mapping loading...")
    synset_to_idx, idx_to_synset, idx_to_label = load_imagenet_class_mappings()
    print(f"   Loaded {len(synset_to_idx)} synset mappings")
    
    # Test some specific mappings
    test_synsets = ['n01440764', 'n01443537', 'n02123045']  # tench, goldfish, tabby cat
    for synset in test_synsets:
        if synset in synset_to_idx:
            idx = synset_to_idx[synset]
            label = idx_to_label[idx]
            print(f"   {synset} -> {idx}: {label}")
    
    # Test class info functions
    print("\n2. Testing class info functions...")
    for test_idx in [0, 1, 281]:  # Test a few indices
        synset, label = get_imagenet_class_info(test_idx)
        readable_name = get_readable_class_name(test_idx)
        print(f"   Class {test_idx}: {synset} -> {readable_name}")
    
    print("\n✅ ImageNet utilities testing complete!")
    }
    
    return stats


# Initialize mappings when module is imported
load_imagenet_class_mappings()


# Initialize mappings when module is imported
load_imagenet_class_mappings()


if __name__ == "__main__":
    # Test the utilities
    from XAI_Enhancer_module.utils.model_utils import IMAGENET_VAL_PATH
    
    print("🧪 Testing ImageNet utilities...")
    
    # Test synset mapping loading
    print("\n1. Testing synset mapping loading...")
    synset_to_idx, idx_to_synset, idx_to_label = load_imagenet_class_mappings()
    print(f"   Loaded {len(synset_to_idx)} synset mappings")
    
    # Test some specific mappings
    test_synsets = ['n01440764', 'n01443537', 'n02123045']  # tench, goldfish, tabby cat
    for synset in test_synsets:
        if synset in synset_to_idx:
            idx = synset_to_idx[synset]
            label = idx_to_label[idx]
            print(f"   {synset} -> {idx}: {label}")
    
    # Test class info functions
    print("\n2. Testing class info functions...")
    for test_idx in [0, 1, 281]:  # Test a few indices
        synset, label = get_imagenet_class_info(test_idx)
        readable_name = get_readable_class_name(test_idx)
        print(f"   Class {test_idx}: {synset} -> {readable_name}")
    
    # Validate mapping if dataset exists
    if os.path.exists(IMAGENET_VAL_PATH):
        print("\n3. Validating synset mapping against actual dataset...")
        validation_results = validate_synset_mapping()
        
        if 'error' not in validation_results:
            print(f"   Dataset coverage: {validation_results['coverage_percent']:.1f}%")
            print(f"   Matched synsets: {validation_results['matched_synsets']}/{validation_results['total_actual_dirs']}")
            
            if validation_results['missing_from_mapping'] > 0:
                print(f"   ⚠️  Missing from mapping: {validation_results['missing_from_mapping']}")
                print(f"      Sample missing: {validation_results['sample_missing']}")
            
            if validation_results['extra_in_mapping'] > 0:
                print(f"   ⚠️  Extra in mapping: {validation_results['extra_in_mapping']}")
        else:
            print(f"   ❌ Validation error: {validation_results['error']}")
        
        # Test dataset stats
        print("\n4. Testing dataset statistics...")
        stats = get_imagenet_validation_stats(IMAGENET_VAL_PATH)
        print(f"   Total classes: {stats['total_classes']}")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Avg images per class: {stats['avg_images_per_class']:.1f}")
        print(f"   Sample synsets: {', '.join(stats['sample_synsets'])}")
        
        # Test image sampling with class info
        print("\n5. Testing image sampling...")
        sample_paths = sample_imagenet_images(
            IMAGENET_VAL_PATH, 
            num_classes=3, 
            images_per_class=2,
            random_seed=42  # For reproducible results
        )
        
        print("\n6. Testing path-to-class conversion...")
        for path in sample_paths[:5]:
            class_idx = get_class_from_path(path)
            class_name = get_readable_class_name(class_idx)
            synset = os.path.basename(os.path.dirname(path))
            print(f"   {synset} -> Class {class_idx}: {class_name}")
    
    else:
        print(f"\n⚠️  ImageNet dataset not found at: {IMAGENET_VAL_PATH}")
        print("    Testing with synthetic data...")
        
        # Test with synthetic paths
        test_paths = [
            "/path/to/n01440764/image1.JPEG",
            "/path/to/n01443537/image2.JPEG", 
            "/path/to/n02123045/image3.JPEG"
        ]
        
        for path in test_paths:
            class_idx = get_class_from_path(path)
            class_name = get_readable_class_name(class_idx)
            synset = os.path.basename(os.path.dirname(path))
            print(f"   {synset} -> Class {class_idx}: {class_name}")
    
    print("\n✅ ImageNet utilities testing complete!")
            IMAGENET_VAL_PATH, 
            num_classes=3, 
            images_per_class=2,
            random_seed=42  # For reproducible results
        )
        
        print(f"
6. Testing path-to-class conversion...")
        for path in sample_paths[:5]:
            class_idx = get_class_from_path(path)
            class_name = get_readable_class_name(class_idx)
            synset = os.path.basename(os.path.dirname(path))
            print(f"   {synset} -> Class {class_idx}: {class_name}")
    
    else:
        print(f"
⚠️  ImageNet dataset not found at: {IMAGENET_VAL_PATH}")
        print("    Testing with synthetic data...")
        
        # Test with synthetic paths
        test_paths = [
            "/path/to/n01440764/image1.JPEG",
            "/path/to/n01443537/image2.JPEG", 
            "/path/to/n02123045/image3.JPEG"
        ]
        
        for path in test_paths:
            class_idx = get_class_from_path(path)
            class_name = get_readable_class_name(class_idx)
            synset = os.path.basename(os.path.dirname(path))
            print(f"   {synset} -> Class {class_idx}: {class_name}")
    
    print("
✅ ImageNet utilities testing complete!")
