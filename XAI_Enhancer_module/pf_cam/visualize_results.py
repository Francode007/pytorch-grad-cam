#!/usr/bin/env python3
"""
PF-CAM Result Visualization & Analysis Script.

Generates plots and analysis reports from PF-CAM experiment results and weight logs.
1. Comparison bar charts (PF-CAM vs Standard Methods)
2. AUC/ROAD metric summaries
3. Weight distribution histograms (if logs available)
4. Stage contribution analysis
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="PF-CAM Result Visualization")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing pf_cam_results.csv and/or weight logs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: results-dir/plots)")
    parser.add_argument("--comparison-file", type=str, default="comparison_results.csv",
                        help="Filename of standard method comparison results")
    return parser.parse_args()


def plot_method_comparison(pf_results_df, comparison_df, output_dir):
    """Plot metric comparison between PF-CAM and standard methods."""
    if comparison_df is None:
        print("No comparison metrics found. Skipping comparison plot.")
        return

    # Combine PF-CAM and standard results for plotting
    # Assumes pf_results_df has 1 row (the aggregate)
    pf_metrics = pf_results_df.iloc[0].to_dict()
    pf_metrics["method"] = "PF-CAM"
    
    # Format comparison_df
    all_data = []
    all_data.append(pf_metrics)
    
    if not comparison_df.empty:
        standard_data = comparison_df.to_dict("records")
        all_data.extend(standard_data)
    
    df = pd.DataFrame(all_data)
    
    # Metrics to plot
    metrics = ["insertion_auc", "deletion_auc", "road_mean"]
    metric_names = ["Insertion AUC (↑)", "Deletion AUC (↓)", "ROAD Score (↓)"]
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_style("whitegrid")
    
    colors = sns.color_palette("muted")
    pf_color = colors[3]  # Redish/Purple often distinct
    std_color = colors[0] # Blue
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_names)):
        if metric not in df.columns:
            continue
            
        ax = axes[idx]
        # Color PF-CAM differently
        palette = {m: pf_color if m == "PF-CAM" else std_color for m in df["method"]}
        
        sns.barplot(data=df, x="method", y=metric, ax=ax, palette=palette)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        
        # Add value labels
        for i, row in df.iterrows():
            # Get bar object (tricky with seaborn order, assuming sorted)
            val = row[metric]
            # Simple text annotation
            # We can't easily map row back to bar without stricter control, 
            # so we'll just loop through patches
            pass

        # Improve labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "method_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison plot to {save_path}")


def analyze_weight_distribution(weight_log_file, output_dir):
    """Analyze stage and layer weight distributions."""
    if not os.path.exists(weight_log_file):
        print("No weight log CSV found. Skipping weight analysis.")
        return

    print("Loading weight logs (this may take a moment)...")
    df = pd.read_csv(weight_log_file)
    
    # 1. Stage Weight Distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="stage_id", y="stage_weight", inner="quartile")
    plt.title("Distribution of Stage Weights (0=Deepest, Higher=Shallower)")
    plt.xlabel("Stage ID")
    plt.ylabel("Softmax Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_weights_violin.png"))
    
    # 2. Top-K Selection Frequency per Stage
    # Calculate % of times layers in each stage were selected
    # Group by stage_id, then calculate mean of top_k_selected
    stage_selection = df.groupby("stage_id")["top_k_selected"].mean() * 100
    
    plt.figure(figsize=(8, 5))
    stage_selection.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Layer Selection Frequency by Stage (Top-K)")
    plt.xlabel("Stage ID")
    plt.ylabel("% Layers Selected")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_selection_freq.png"))
    
    # 3. Layer Contribution (Mean Softmax Weight)
    # Get unique layer names sorted by appearance
    df_sorted = df.sort_values("stage_id")
    layer_order = df_sorted["layer_name"].unique()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x="softmax_weight", y="layer_name", order=layer_order, errorbar=None)
    plt.title("Mean Attribution Weight per Layer")
    plt.xlabel("Mean Softmax Weight")
    plt.ylabel("Layer Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_impact_mean.png"))
    
    print(f"Saved weight analysis plots to {output_dir}")


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load comparison results
    comp_path = os.path.join(args.results_dir, args.comparison_file)
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    
    # Load PF-CAM results
    pf_res_path = os.path.join(args.results_dir, "pf_cam_results.csv")
    pf_df = pd.read_csv(pf_res_path) if os.path.exists(pf_res_path) else None
    
    if pf_df is not None:
        plot_method_comparison(pf_df, comp_df, output_dir)
    else:
        print("PF-CAM results CSV not found.")
        
    # Analyze weights
    weight_log_path = os.path.join(args.results_dir, "pf_cam_weight_log.csv")
    analyze_weight_distribution(weight_log_path, output_dir)
    
    print("\nVisualization Complete.")

if __name__ == "__main__":
    main()
