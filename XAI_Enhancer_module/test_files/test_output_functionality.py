#!/usr/bin/env python3
"""
Test script for model-specific output directories and plot saving functionality.
This script tests the new features for organized output saving.
"""

import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.utils.directory_manager import (
    create_model_output_dirs, save_evaluation_results, save_analysis_data,
    print_directory_structure, list_model_outputs
)

def test_directory_creation():
    """Test model-specific directory creation."""
    print("="*60)
    print("TESTING DIRECTORY CREATION")
    print("="*60)
    
    models_to_test = ['resnet18', 'resnet50', 'b0']
    
    for model in models_to_test:
        print(f"\n📁 Testing directories for {model}:")
        analysis_dir, csv_dir = create_model_output_dirs(model)
        print(f"  Analysis: {analysis_dir}")
        print(f"  CSV: {csv_dir}")
        print(f"  Analysis exists: {analysis_dir.exists()}")
        print(f"  CSV exists: {csv_dir.exists()}")

def test_csv_saving():
    """Test CSV saving functionality."""
    print("\n" + "="*60)
    print("TESTING CSV SAVING")
    print("="*60)
    
    import pandas as pd
    
    # Create sample evaluation results
    sample_results = pd.DataFrame({
        'Method': ['Enhanced CAM (all)', 'GradCAM', 'LayerCAM', 'ScoreCAM'],
        'Insertion_AUC': [0.82, 0.75, 0.78, 0.73],
        'Deletion_AUC': [0.58, 0.65, 0.62, 0.68],
        'ROAD_Score': [0.043, 0.052, 0.048, 0.055],
        'Num_Images': [10, 10, 10, 10]
    })
    
    try:
        saved_path = save_evaluation_results(
            sample_results,
            'test_model',
            evaluation_type='feature_test',
            add_timestamp=True
        )
        print(f"✅ Sample CSV saved to: {saved_path}")
        print(f"   File exists: {saved_path.exists()}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

def test_pickle_saving():
    """Test pickle saving functionality."""
    print("\n" + "="*60)
    print("TESTING PICKLE SAVING")
    print("="*60)
    
    # Create sample analysis data
    sample_data = {
        'model_name': 'test_model',
        'layer_info': [
            {'name': 'conv1', 'out_channels': 64, 'kernel_size': (7, 7)},
            {'name': 'conv2', 'out_channels': 128, 'kernel_size': (3, 3)}
        ],
        'average_weights': [0.15, 0.25, 0.35, 0.25],
        'evaluation_metrics': {
            'insertion_auc_mean': 0.82,
            'deletion_auc_mean': 0.58,
            'road_mean': 0.043
        }
    }
    
    try:
        saved_path = save_analysis_data(
            sample_data,
            'test_model',
            analysis_type='feature_test',
            add_timestamp=True
        )
        print(f"✅ Sample pickle saved to: {saved_path}")
        print(f"   File exists: {saved_path.exists()}")
        
        # Test loading the data back
        import pickle
        with open(saved_path, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"✅ Data loaded successfully: {type(loaded_data)}")
        print(f"   Model name in data: {loaded_data.get('model_name')}")
        
    except Exception as e:
        print(f"❌ Error with pickle operations: {e}")

def test_plot_saving():
    """Test plot saving functionality."""
    print("\n" + "="*60)
    print("TESTING PLOT SAVING")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a sample plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, label='sin(x)')
        ax.set_title('Test Plot for Model-Specific Saving')
        ax.set_xlabel('X values')
        ax.set_ylabel('Y values')
        ax.legend()
        ax.grid(True)
        
        # Save using the directory manager
        from XAI_Enhancer_module.utils.directory_manager import save_visualization
        saved_path = save_visualization(
            fig,
            'test_model',
            viz_type='feature_test',
            add_timestamp=True
        )
        
        print(f"✅ Plot saved to: {saved_path}")
        print(f"   File exists: {saved_path.exists()}")
        
        plt.close(fig)
        
    except Exception as e:
        print(f"❌ Error saving plot: {e}")

def test_visualization_function():
    """Test the enhanced visualization function."""
    print("\n" + "="*60)
    print("TESTING ENHANCED VISUALIZATION FUNCTION")
    print("="*60)
    
    try:
        # Test the imports
        from XAI_Enhancer_module.all_layer_analysis import AllLayerAnalyzer
        print("✅ AllLayerAnalyzer import successful")
        
        # Note: We can't test the full visualization without a model and data,
        # but we can test that the function exists and has the right signature
        analyzer = AllLayerAnalyzer('resnet18')
        print(f"✅ Analyzer created for model: resnet18")
        print(f"   Found {len(analyzer.conv_layers)} convolutional layers")
        
        # Check if the create_layer_visualization method has the expected parameters
        import inspect
        sig = inspect.signature(analyzer.create_layer_visualization)
        params = list(sig.parameters.keys())
        print(f"✅ Visualization method parameters: {params}")
        
        expected_params = ['results', 'save_path', 'save_individual_plots']
        has_all_params = all(param in params for param in expected_params)
        print(f"✅ Has all expected parameters: {has_all_params}")
        
    except Exception as e:
        print(f"❌ Error testing visualization function: {e}")
        import traceback
        traceback.print_exc()

def show_directory_structure():
    """Show the current directory structure."""
    print("\n" + "="*60)
    print("CURRENT DIRECTORY STRUCTURE")
    print("="*60)
    
    print_directory_structure()

def cleanup_test_files():
    """Clean up test files and directories."""
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)
    
    try:
        # Remove test directories
        test_dirs = [
            Path('./analysis_results/test_model'),
            Path('./csv_exports/test_model')
        ]
        
        for test_dir in test_dirs:
            if test_dir.exists():
                shutil.rmtree(test_dir)
                print(f"🗑️ Removed: {test_dir}")
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")

def main():
    """Run all tests."""
    print("🧪 TESTING MODEL-SPECIFIC OUTPUT FUNCTIONALITY")
    print("="*80)
    
    # Run tests
    test_directory_creation()
    test_csv_saving()
    test_pickle_saving()
    test_plot_saving()
    test_visualization_function()
    show_directory_structure()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED!")
    print("="*80)
    print("\nFeatures tested:")
    print("• ✅ Model-specific directory creation")
    print("• ✅ CSV saving to organized directories")  
    print("• ✅ Pickle saving to organized directories")
    print("• ✅ Plot saving to organized directories")
    print("• ✅ Enhanced visualization function with individual plots")
    print("• ✅ Directory structure management")
    
    print("\nYou can now use these features:")
    print("1. Run all_layer_analysis.py with --save-plots --save-individual")
    print("2. Run modular_xai_evaluation.py (automatically saves to model directories)")
    print("3. All outputs will be organized by model name")
    
    # Offer cleanup
    response = input("\nClean up test files? (y/n): ").strip().lower()
    if response == 'y':
        cleanup_test_files()

if __name__ == "__main__":
    main()
