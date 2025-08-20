#!/usr/bin/env python3
"""
Simple Pickle Reader Examples.
This script shows various simple ways to read and use the pickle file data.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def simple_read_example():
    """Simple example of reading the pickle file."""
    print("=== SIMPLE PICKLE READING EXAMPLE ===\n")
    
    # Load the pickle file
    pkl_path = "analysis_results/resnet18_analysis_results.pkl"
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded pickle file: {pkl_path}")
    print(f"📊 Main keys: {list(data.keys())}")
    
    return data


def extract_layer_weights(data):
    """Extract and analyze layer weights."""
    print("\n=== LAYER WEIGHTS ANALYSIS ===\n")
    
    # Get average weights
    avg_weights = data['average_weights']
    print(f"Average weights shape: {avg_weights.shape}")
    print(f"Average weights: {avg_weights}")
    
    # Find most important layers
    top_3_indices = np.argsort(avg_weights)[-3:][::-1]
    print(f"\nTop 3 most important layers:")
    for i, idx in enumerate(top_3_indices, 1):
        layer_name = data['layer_info'][idx]['name']
        weight = avg_weights[idx]
        print(f"  {i}. Layer {idx} ({layer_name}): {weight:.6f}")
    
    return avg_weights, top_3_indices


def extract_performance_metrics(data):
    """Extract performance metrics."""
    print("\n=== PERFORMANCE METRICS ===\n")
    
    metrics = data['evaluation_metrics']
    
    print(f"Model: {metrics['model_name']}")
    print(f"Method: {metrics['cam_method']}")
    print(f"Layers used: {metrics['num_layers']}")
    print(f"Images evaluated: {metrics['num_images']}")
    
    print(f"\nPerformance scores:")
    print(f"  Insertion AUC: {metrics['insertion_auc_mean']:.4f} ± {metrics['insertion_auc_std']:.4f}")
    print(f"  Deletion AUC:  {metrics['deletion_auc_mean']:.4f} ± {metrics['deletion_auc_std']:.4f}")
    print(f"  ROAD Score:    {metrics['road_mean']:.4f} ± {metrics['road_std']:.4f}")
    
    return metrics


def extract_layer_info(data):
    """Extract detailed layer information."""
    print("\n=== LAYER INFORMATION ===\n")
    
    layer_info = data['layer_info']
    print(f"Total layers: {len(layer_info)}")
    
    # Create a simple summary
    print(f"\nLayer summary:")
    for i, layer in enumerate(layer_info[:5]):  # Show first 5 layers
        print(f"  Layer {i}: {layer['name']} ({layer['type']})")
        print(f"    Channels: {layer['in_channels']} → {layer['out_channels']}")
    
    if len(layer_info) > 5:
        print(f"  ... and {len(layer_info) - 5} more layers")
    
    return layer_info


def analyze_per_image_weights(data):
    """Analyze weights for individual images."""
    print("\n=== PER-IMAGE WEIGHT ANALYSIS ===\n")
    
    layer_weights_all = data['layer_weights_all']
    print(f"Number of images analyzed: {len(layer_weights_all)}")
    
    if layer_weights_all:
        # Convert to numpy array for easier analysis
        weights_matrix = np.array(layer_weights_all)
        print(f"Weights matrix shape: {weights_matrix.shape}")
        print(f"  (rows = images, columns = layers)")
        
        # Find most consistent layer across images
        weight_std = np.std(weights_matrix, axis=0)
        most_consistent = np.argmin(weight_std)
        most_variable = np.argmax(weight_std)
        
        print(f"\nLayer consistency analysis:")
        print(f"  Most consistent layer: {most_consistent} (std: {weight_std[most_consistent]:.6f})")
        print(f"  Most variable layer: {most_variable} (std: {weight_std[most_variable]:.6f})")
        
        return weights_matrix
    
    return None


def extract_individual_results(data):
    """Extract results for individual images."""
    print("\n=== INDIVIDUAL IMAGE RESULTS ===\n")
    
    detailed_analysis = data['detailed_analysis']
    print(f"Number of images with detailed analysis: {len(detailed_analysis)}")
    
    for i, analysis in enumerate(detailed_analysis):
        image_name = Path(analysis['image_path']).name
        predicted_label = analysis['predicted_label']
        layer_weights = analysis['layer_weights']
        
        print(f"\nImage {i+1}: {image_name}")
        print(f"  Predicted label: {predicted_label}")
        print(f"  Top 3 layers for this image: {np.argsort(layer_weights)[-3:][::-1]}")
        print(f"  Weight range: [{layer_weights.min():.6f}, {layer_weights.max():.6f}]")


def create_simple_dataframe(data):
    """Create a simple DataFrame for analysis."""
    print("\n=== CREATING DATAFRAME FOR ANALYSIS ===\n")
    
    # Combine layer info with weights
    layer_info = data['layer_info']
    avg_weights = data['average_weights']
    
    df_data = []
    for i, (layer, weight) in enumerate(zip(layer_info, avg_weights)):
        df_data.append({
            'layer_index': i,
            'layer_name': layer['name'],
            'layer_type': layer['type'],
            'in_channels': layer['in_channels'],
            'out_channels': layer['out_channels'],
            'average_weight': weight
        })
    
    df = pd.DataFrame(df_data)
    
    print(f"Created DataFrame with {len(df)} rows")
    print(f"\nTop 5 most important layers:")
    top_layers = df.nlargest(5, 'average_weight')
    print(top_layers[['layer_index', 'layer_name', 'average_weight']].to_string(index=False))
    
    return df


def save_summary_to_text(data, output_file="analysis_summary.txt"):
    """Save a text summary of the analysis."""
    print(f"\n=== SAVING SUMMARY TO {output_file} ===\n")
    
    with open(output_file, 'w') as f:
        f.write("Enhanced CAM All-Layer Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic info
        metrics = data['evaluation_metrics']
        f.write(f"Model: {metrics['model_name']}\n")
        f.write(f"Method: {metrics['cam_method']}\n")
        f.write(f"Layers: {metrics['num_layers']}\n")
        f.write(f"Images: {metrics['num_images']}\n\n")
        
        # Performance
        f.write("Performance Metrics:\n")
        f.write(f"  Insertion AUC: {metrics['insertion_auc_mean']:.4f} ± {metrics['insertion_auc_std']:.4f}\n")
        f.write(f"  Deletion AUC:  {metrics['deletion_auc_mean']:.4f} ± {metrics['deletion_auc_std']:.4f}\n")
        f.write(f"  ROAD Score:    {metrics['road_mean']:.4f} ± {metrics['road_std']:.4f}\n\n")
        
        # Top layers
        avg_weights = data['average_weights']
        layer_info = data['layer_info']
        top_5_indices = np.argsort(avg_weights)[-5:][::-1]
        
        f.write("Top 5 Most Important Layers:\n")
        for rank, idx in enumerate(top_5_indices, 1):
            layer_name = layer_info[idx]['name']
            weight = avg_weights[idx]
            f.write(f"  {rank}. Layer {idx} ({layer_name}): {weight:.6f}\n")
    
    print(f"✅ Summary saved to {output_file}")


def main():
    """Main function demonstrating various ways to read the pickle file."""
    print("PICKLE FILE READING EXAMPLES")
    print("=" * 50)
    
    # Check if file exists
    pkl_path = "analysis_results/resnet18_analysis_results.pkl"
    if not Path(pkl_path).exists():
        print(f"❌ Pickle file not found: {pkl_path}")
        print("Please run the all_layer_analysis.py script first.")
        return
    
    # 1. Simple reading
    data = simple_read_example()
    
    # 2. Extract different components
    weights, top_layers = extract_layer_weights(data)
    metrics = extract_performance_metrics(data)
    layer_info = extract_layer_info(data)
    
    # 3. Analyze per-image data
    weights_matrix = analyze_per_image_weights(data)
    extract_individual_results(data)
    
    # 4. Create DataFrame for further analysis
    df = create_simple_dataframe(data)
    
    # 5. Save summary
    save_summary_to_text(data)
    
    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print("\nYou can now:")
    print("  - Use the 'data' variable to access all pickle data")
    print("  - Use the 'df' DataFrame for pandas analysis")
    print("  - Use the 'weights' array for numpy operations")
    print("  - Read the 'analysis_summary.txt' file")


if __name__ == "__main__":
    main()
