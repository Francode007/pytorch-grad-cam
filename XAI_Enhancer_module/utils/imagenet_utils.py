#!/usr/bin/env python3
"""
ImageNet utilities for synset mapping and dataset handling.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_imagenet_synset_mapping(mapping_file_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load ImageNet synset mapping from LOC_synset_mapping.txt
    
    Args:
        mapping_file_path: Path to synset mapping file. If None, uses default location.
        
    Returns:
        Dictionary mapping synset IDs to class names
    """
    if mapping_file_path is None:
        # Default location relative to project root
        project_root = Path(__file__).parent.parent.parent
        mapping_file_path = project_root / "LOC_synset_mapping.txt"
    
    mapping_file = Path(mapping_file_path)
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Synset mapping file not found: {mapping_file}")
    
    synset_mapping = {}
    with open(mapping_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    synset, class_name = parts
                    synset_mapping[synset] = class_name
                else:
                    print(f"Warning: Invalid line {line_num} in synset mapping: {line}")
    
    print(f"Loaded {len(synset_mapping)} ImageNet class mappings")
    return synset_mapping


def get_imagenet_class_names() -> List[str]:
    """Get list of all ImageNet class names"""
    synset_mapping = load_imagenet_synset_mapping()
    return list(synset_mapping.values())


def find_classes_by_name(search_terms: List[str], 
                        synset_mapping: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """
    Find ImageNet classes by searching class names.
    
    Args:
        search_terms: List of terms to search for in class names
        synset_mapping: Pre-loaded synset mapping. If None, loads from file.
        
    Returns:
        Dictionary mapping search terms to lists of matching class names
    """
    if synset_mapping is None:
        synset_mapping = load_imagenet_synset_mapping()
    
    results = {}
    
    for search_term in search_terms:
        matches = []
        search_lower = search_term.lower()
        
        for synset, class_name in synset_mapping.items():
            if search_lower in class_name.lower():
                matches.append(class_name)
        
        results[search_term] = matches
        print(f"Found {len(matches)} matches for '{search_term}': {matches[:3]}{'...' if len(matches) > 3 else ''}")
    
    return results


def get_synset_from_class_name(class_name: str, 
                              synset_mapping: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Get synset ID from class name (exact or partial match).
    
    Args:
        class_name: Class name to search for
        synset_mapping: Pre-loaded synset mapping. If None, loads from file.
        
    Returns:
        Synset ID if found, None otherwise
    """
    if synset_mapping is None:
        synset_mapping = load_imagenet_synset_mapping()
    
    class_name_lower = class_name.lower()
    
    # First try exact match
    for synset, name in synset_mapping.items():
        if name.lower() == class_name_lower:
            return synset
    
    # Then try partial match
    for synset, name in synset_mapping.items():
        if class_name_lower in name.lower():
            return synset
    
    return None


def validate_imagenet_path(imagenet_path: str) -> Tuple[bool, str, Dict[str, int]]:
    """
    Validate ImageNet dataset path and provide statistics.
    
    Args:
        imagenet_path: Path to ImageNet validation dataset
        
    Returns:
        Tuple of (is_valid, message, synset_counts)
    """
    path = Path(imagenet_path)
    
    if not path.exists():
        return False, f"Path does not exist: {imagenet_path}", {}
    
    if not path.is_dir():
        return False, f"Path is not a directory: {imagenet_path}", {}
    
    # Check for synset directories
    synset_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith('n')]
    
    if not synset_dirs:
        return False, f"No synset directories found in: {imagenet_path}", {}
    
    # Count images in each synset
    synset_counts = {}
    total_images = 0
    
    for synset_dir in synset_dirs:
        synset = synset_dir.name
        image_files = list(synset_dir.glob("*.JPEG")) + list(synset_dir.glob("*.jpg"))
        image_count = len(image_files)
        synset_counts[synset] = image_count
        total_images += image_count
    
    message = (f"Valid ImageNet dataset found:\n"
              f"  - {len(synset_dirs)} synset directories\n"
              f"  - {total_images} total images\n"
              f"  - Average {total_images/len(synset_dirs):.1f} images per synset")
    
    return True, message, synset_counts


def create_imagenet_subset_config(target_classes: List[str], 
                                 output_file: str = "imagenet_subset_config.json") -> str:
    """
    Create a configuration file for ImageNet subset evaluation.
    
    Args:
        target_classes: List of class names to include in subset
        output_file: Output configuration file path
        
    Returns:
        Path to created configuration file
    """
    synset_mapping = load_imagenet_synset_mapping()
    
    subset_config = {
        "subset_name": f"ImageNet_{len(target_classes)}_classes",
        "total_classes": len(target_classes),
        "classes": [],
        "synsets": []
    }
    
    not_found = []
    
    for class_name in target_classes:
        synset = get_synset_from_class_name(class_name, synset_mapping)
        if synset:
            subset_config["classes"].append({
                "name": class_name,
                "synset": synset,
                "full_name": synset_mapping[synset]
            })
            subset_config["synsets"].append(synset)
        else:
            not_found.append(class_name)
    
    if not_found:
        print(f"Warning: Could not find synsets for: {not_found}")
    
    # Save configuration
    with open(output_file, 'w') as f:
        json.dump(subset_config, f, indent=2)
    
    print(f"Created subset configuration: {output_file}")
    print(f"  - {len(subset_config['classes'])} classes configured")
    print(f"  - {len(not_found)} classes not found")
    
    return output_file


def print_imagenet_sample_classes(num_samples: int = 20):
    """Print a sample of ImageNet classes for reference"""
    synset_mapping = load_imagenet_synset_mapping()
    
    print(f"\nSample of {num_samples} ImageNet classes:")
    print("=" * 60)
    
    items = list(synset_mapping.items())[:num_samples]
    for i, (synset, class_name) in enumerate(items, 1):
        print(f"{i:2d}. {synset}: {class_name}")
    
    print(f"\n... and {len(synset_mapping) - num_samples} more classes")
    print(f"Total ImageNet classes: {len(synset_mapping)}")


def suggest_classes_for_evaluation() -> List[str]:
    """Suggest a diverse set of ImageNet classes for evaluation"""
    suggestions = [
        "tench",  # Fish
        "goldfish",  # Fish
        "great white shark",  # Predator
        "tiger shark",  # Predator
        "cock",  # Bird
        "hen",  # Bird
        "ostrich",  # Large bird
        "bald eagle",  # Bird of prey
        "bullfrog",  # Amphibian
        "loggerhead",  # Turtle
        "American alligator",  # Reptile
        "green lizard",  # Reptile
        "Komodo dragon",  # Large reptile
        "African crocodile",  # Large reptile
        "barn spider",  # Arachnid
        "black widow",  # Dangerous spider
        "tarantula",  # Large spider
        "peacock",  # Colorful bird
        "macaw",  # Colorful bird
        "hummingbird",  # Small bird
        "toucan",  # Distinctive bird
        "jellyfish",  # Marine
        "sea anemone",  # Marine
        "tabby cat",  # Domestic
        "tiger cat",  # Wild cat
        "Persian cat",  # Domestic breed
        "cougar",  # Wild cat
        "lynx",  # Wild cat
        "leopard",  # Big cat
        "lion",  # Big cat
        "tiger",  # Big cat
        "cheetah",  # Fast cat
        "brown bear",  # Large mammal
        "polar bear",  # Arctic mammal
        "elephant",  # Large mammal
        "zebra",  # Striped mammal
        "horse",  # Domestic animal
        "dog",  # Domestic animal
    ]
    
    return suggestions


if __name__ == "__main__":
    """Demo usage of ImageNet utilities"""
    print("ImageNet Utilities Demo")
    print("=" * 50)
    
    # Load synset mapping
    mapping = load_imagenet_synset_mapping()
    print(f"Loaded {len(mapping)} classes")
    
    # Show sample classes
    print_imagenet_sample_classes(15)
    
    # Search for specific classes
    search_terms = ["shark", "eagle", "cat"]
    matches = find_classes_by_name(search_terms)
    
    print("\nSearch results:")
    for term, results in matches.items():
        print(f"'{term}': {len(results)} matches")
        for result in results[:3]:
            print(f"  - {result}")
    
    # Suggest classes for evaluation
    suggestions = suggest_classes_for_evaluation()
    print(f"\nSuggested classes for evaluation ({len(suggestions)} classes):")
    for i, class_name in enumerate(suggestions[:10], 1):
        print(f"{i:2d}. {class_name}")
    print("... and more")
