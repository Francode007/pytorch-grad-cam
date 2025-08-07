#!/usr/bin/env python3
"""
Quick fix script for the evaluation suite errors.
Run this script to automatically fix the pandas and multiprocessing issues.
"""

import re
from pathlib import Path

def fix_pandas_sort_values():
    """Fix pandas sort_values na_last parameter."""
    files_to_fix = [
        "xai_evaluation_suite.py"
    ]
    
    for file_name in files_to_fix:
        file_path = Path(file_name)
        if file_path.exists():
            content = file_path.read_text()
            
            # Replace na_last=True with na_position='last'
            content = content.replace("na_last=True", "na_position='last'")
            
            file_path.write_text(content)
            print(f"✅ Fixed pandas sort_values in {file_path}")

def fix_multiprocessing():
    """Fix multiprocessing issues by setting num_workers=0."""
    fixes = [
        ("optimized_cam_extractor.py", "num_workers=num_workers", "num_workers=0"),
        ("optimized_cam_extractor.py", "pin_memory=True if torch.cuda.is_available() else False", "pin_memory=False"),
        ("optimized_predictor.py", "num_workers=2", "num_workers=0"),
        ("model_utils.py", "num_workers=2", "num_workers=0"),
    ]
    
    for file_name, old_pattern, new_pattern in fixes:
        file_path = Path(file_name)
        if file_path.exists():
            content = file_path.read_text()
            content = content.replace(old_pattern, new_pattern)
            file_path.write_text(content)
            print(f"✅ Fixed multiprocessing in {file_path}")

if __name__ == "__main__":
    print("Applying fixes for evaluation suite errors...")
    fix_pandas_sort_values()
    fix_multiprocessing()
    print("\n🎉 All fixes applied successfully!")
    print("\nYou can now run the evaluation suite without these errors.")
