#!/usr/bin/env python3
"""
Directory management utilities for organizing XAI evaluation outputs.
This module provides functions to create model-specific output directories.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime


def create_model_output_dirs(model_name: str, 
                           base_analysis_dir: str = "./analysis_results",
                           base_csv_dir: str = "./csv_exports") -> tuple[Path, Path]:
    """
    Create model-specific output directories for analysis results and CSV exports.
    
    Args:
        model_name: Name of the model (e.g., 'resnet18', 'b4', etc.)
        base_analysis_dir: Base directory for analysis results
        base_csv_dir: Base directory for CSV exports
        
    Returns:
        Tuple of (analysis_dir, csv_dir) paths
    """
    # Create base directories if they don't exist
    base_analysis_path = Path(base_analysis_dir)
    base_csv_path = Path(base_csv_dir)
    
    # Create model-specific subdirectories
    model_analysis_dir = base_analysis_path / model_name
    model_csv_dir = base_csv_path / model_name
    
    # Create directories
    model_analysis_dir.mkdir(parents=True, exist_ok=True)
    model_csv_dir.mkdir(parents=True, exist_ok=True)
    
    return model_analysis_dir, model_csv_dir


def get_timestamped_filename(base_name: str, extension: str = "") -> str:
    """
    Generate a timestamped filename.
    
    Args:
        base_name: Base name for the file
        extension: File extension (with or without dot)
        
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    return f"{base_name}_{timestamp}{extension}"


def save_evaluation_results(results_df, model_name: str, 
                          evaluation_type: str = "comparison",
                          base_csv_dir: str = "./csv_exports",
                          add_timestamp: bool = True) -> Path:
    """
    Save evaluation results DataFrame to a model-specific CSV file.
    
    Args:
        results_df: Pandas DataFrame with evaluation results
        model_name: Name of the model
        evaluation_type: Type of evaluation (e.g., 'comparison', 'enhanced_only', etc.)
        base_csv_dir: Base directory for CSV exports
        add_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to the saved CSV file
    """
    # Create model-specific directory
    _, model_csv_dir = create_model_output_dirs(model_name, base_csv_dir=base_csv_dir)
    
    # Generate filename
    base_filename = f"{evaluation_type}_results"
    if add_timestamp:
        filename = get_timestamped_filename(base_filename, "csv")
    else:
        filename = f"{base_filename}.csv"
    
    # Save file
    output_path = model_csv_dir / filename
    results_df.to_csv(output_path, index=False)
    
    return output_path


def save_analysis_data(data, model_name: str, 
                      analysis_type: str = "all_layers",
                      base_analysis_dir: str = "./analysis_results",
                      add_timestamp: bool = True) -> Path:
    """
    Save analysis data (pickle, JSON, etc.) to a model-specific directory.
    
    Args:
        data: Data to save (will be pickled)
        model_name: Name of the model
        analysis_type: Type of analysis
        base_analysis_dir: Base directory for analysis results
        add_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to the saved file
    """
    import pickle
    
    # Create model-specific directory
    model_analysis_dir, _ = create_model_output_dirs(model_name, base_analysis_dir=base_analysis_dir)
    
    # Generate filename
    base_filename = f"{analysis_type}_analysis"
    if add_timestamp:
        filename = get_timestamped_filename(base_filename, "pkl")
    else:
        filename = f"{base_filename}.pkl"
    
    # Save file
    output_path = model_analysis_dir / filename
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    return output_path


def save_visualization(fig, model_name: str, 
                      viz_type: str = "analysis",
                      base_analysis_dir: str = "./analysis_results",
                      add_timestamp: bool = True) -> Path:
    """
    Save matplotlib figure to a model-specific directory.
    
    Args:
        fig: Matplotlib figure object
        model_name: Name of the model
        viz_type: Type of visualization
        base_analysis_dir: Base directory for analysis results
        add_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to the saved figure
    """
    # Create model-specific directory
    model_analysis_dir, _ = create_model_output_dirs(model_name, base_analysis_dir=base_analysis_dir)
    
    # Generate filename
    base_filename = f"{viz_type}_visualization"
    if add_timestamp:
        filename = get_timestamped_filename(base_filename, "png")
    else:
        filename = f"{viz_type}_visualization.png"
    
    # Save figure
    output_path = model_analysis_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return output_path


def list_model_outputs(base_analysis_dir: str = "./analysis_results",
                      base_csv_dir: str = "./csv_exports") -> dict:
    """
    List all outputs for each model.
    
    Args:
        base_analysis_dir: Base directory for analysis results
        base_csv_dir: Base directory for CSV exports
        
    Returns:
        Dictionary with model names as keys and lists of files as values
    """
    results = {}
    
    # Check analysis results
    analysis_path = Path(base_analysis_dir)
    if analysis_path.exists():
        for model_dir in analysis_path.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                if model_name not in results:
                    results[model_name] = {'analysis': [], 'csv': []}
                
                results[model_name]['analysis'] = [
                    str(f.relative_to(analysis_path)) for f in model_dir.iterdir() if f.is_file()
                ]
    
    # Check CSV exports
    csv_path = Path(base_csv_dir)
    if csv_path.exists():
        for model_dir in csv_path.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                if model_name not in results:
                    results[model_name] = {'analysis': [], 'csv': []}
                
                results[model_name]['csv'] = [
                    str(f.relative_to(csv_path)) for f in model_dir.iterdir() if f.is_file()
                ]
    
    return results


def print_directory_structure(base_analysis_dir: str = "./analysis_results",
                             base_csv_dir: str = "./csv_exports"):
    """
    Print the current directory structure for outputs.
    
    Args:
        base_analysis_dir: Base directory for analysis results
        base_csv_dir: Base directory for CSV exports
    """
    print("📁 OUTPUT DIRECTORY STRUCTURE:")
    print("=" * 50)
    
    model_outputs = list_model_outputs(base_analysis_dir, base_csv_dir)
    
    if not model_outputs:
        print("No model-specific outputs found.")
        return
    
    for model_name, files in model_outputs.items():
        print(f"\n📊 {model_name.upper()}:")
        
        if files['analysis']:
            print(f"  📈 Analysis Results ({base_analysis_dir}/{model_name}/):")
            for file in files['analysis']:
                print(f"    • {file}")
        
        if files['csv']:
            print(f"  📋 CSV Exports ({base_csv_dir}/{model_name}/):")
            for file in files['csv']:
                print(f"    • {file}")
        
        if not files['analysis'] and not files['csv']:
            print("    (No files found)")


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    
    print("Testing directory management utilities...")
    
    # Test directory creation
    print("\n1. Testing directory creation:")
    analysis_dir, csv_dir = create_model_output_dirs("resnet18")
    print(f"   Analysis dir: {analysis_dir}")
    print(f"   CSV dir: {csv_dir}")
    
    # Test timestamped filename generation
    print("\n2. Testing filename generation:")
    filename = get_timestamped_filename("test_results", "csv")
    print(f"   Timestamped filename: {filename}")
    
    # Test with sample data
    print("\n3. Testing with sample data:")
    sample_df = pd.DataFrame({
        'method': ['GradCAM', 'Enhanced CAM'],
        'insertion_auc': [0.75, 0.82],
        'deletion_auc': [0.65, 0.58]
    })
    
    try:
        saved_path = save_evaluation_results(sample_df, "resnet18", "test_comparison")
        print(f"   Sample results saved to: {saved_path}")
    except Exception as e:
        print(f"   Error saving sample results: {e}")
    
    # Show directory structure
    print("\n4. Current directory structure:")
    print_directory_structure()
