#!/usr/bin/env python3
"""
Comprehensive Visualization Script for ResNet Analysis Results.
Creates comparison plots and exports data for ResNet18, ResNet34, and ResNet50.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_all_results(base_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load results for all ResNet models.
    
    Args:
        base_path: Base path to the analysis results directory
        
    Returns:
        Dictionary with model results
    """
    models = {
        'resnet18': 'enhanced_cam_all_detailed_analysis_20250826_115323.pkl',
        'resnet34': 'enhanced_cam_all_detailed_analysis_20250826_202038.pkl',
        'resnet50': 'enhanced_cam_all_detailed_analysis_20250826_214812.pkl'
    }
    
    results = {}
    base_path = Path(base_path)
    
    for model_name, pkl_file in models.items():
        pkl_path = base_path / model_name / pkl_file
        
        try:
            with open(pkl_path, 'rb') as f:
                model_results = pickle.load(f)
                results[model_name] = model_results
                print(f"✅ Loaded {model_name} results from {pkl_path}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
    
    return results

def create_summary_comparison(results: Dict[str, Dict[str, Any]], save_path: str):
    """
    Create a summary comparison plot of all models.
    
    Args:
        results: Dictionary containing results for all models
        save_path: Path to save the plot
    """
    # Prepare data for comparison
    model_names = []
    insertion_means = []
    insertion_stds = []
    deletion_means = []
    deletion_stds = []
    road_means = []
    road_stds = []
    num_layers = []
    num_images = []
    
    for model_name, model_data in results.items():
        model_names.append(model_name.upper())
        insertion_means.append(model_data['insertion_auc_mean'])
        insertion_stds.append(model_data['insertion_auc_std'])
        deletion_means.append(model_data['deletion_auc_mean'])
        deletion_stds.append(model_data['deletion_auc_std'])
        road_means.append(model_data['road_mean'])
        road_stds.append(model_data['road_std'])
        num_layers.append(model_data['num_layers'])
        num_images.append(model_data['num_images'])
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ResNet Models Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Insertion AUC
    axes[0, 0].bar(model_names, insertion_means, yerr=insertion_stds, 
                   capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('Insertion AUC (Higher is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('AUC Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(insertion_means):
        axes[0, 0].text(i, v + insertion_stds[i] + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Deletion AUC
    axes[0, 1].bar(model_names, deletion_means, yerr=deletion_stds, 
                   capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 1].set_title('Deletion AUC (Lower is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('AUC Score')
    axes[0, 1].set_ylim(0, max(deletion_means) + max(deletion_stds) + 0.1)
    for i, v in enumerate(deletion_means):
        axes[0, 1].text(i, v + deletion_stds[i] + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: ROAD Score
    axes[0, 2].bar(model_names, road_means, yerr=road_stds, 
                   capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 2].set_title('ROAD Score (Higher is Better)', fontweight='bold')
    axes[0, 2].set_ylabel('ROAD Score')
    for i, v in enumerate(road_means):
        axes[0, 2].text(i, v + road_stds[i] + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Model Architecture Info
    x_pos = range(len(model_names))
    axes[1, 0].bar(x_pos, num_layers, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 0].set_title('Number of Layers', fontweight='bold')
    axes[1, 0].set_ylabel('Layer Count')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names)
    for i, v in enumerate(num_layers):
        axes[1, 0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Dataset Info
    axes[1, 1].bar(model_names, num_images, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Number of Images Evaluated', fontweight='bold')
    axes[1, 1].set_ylabel('Image Count')
    for i, v in enumerate(num_images):
        axes[1, 1].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Combined Performance Score
    # Create a combined score (higher insertion + lower deletion + higher road)
    combined_scores = []
    for i in range(len(model_names)):
        # Normalize scores (insertion and road positive, deletion negative)
        score = insertion_means[i] - deletion_means[i] + road_means[i]
        combined_scores.append(score)
    
    axes[1, 2].bar(model_names, combined_scores, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 2].set_title('Combined Performance Score*\n(*Insertion - Deletion + ROAD)', fontweight='bold')
    axes[1, 2].set_ylabel('Combined Score')
    for i, v in enumerate(combined_scores):
        axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Summary comparison saved to: {save_path}")
    return fig

def create_distribution_plots(results: Dict[str, Dict[str, Any]], save_path: str):
    """
    Create distribution plots for individual image scores.
    
    Args:
        results: Dictionary containing results for all models
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Score Distributions Across Images', fontsize=16, fontweight='bold')
    
    metrics = ['insertion_aucs', 'deletion_aucs', 'road_scores']
    titles = ['Insertion AUC Distribution', 'Deletion AUC Distribution', 'ROAD Score Distribution']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        for model_name, model_data in results.items():
            scores = model_data[metric]
            axes[idx].hist(scores, alpha=0.6, label=model_name.upper(), bins=30, density=True)
        
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_xlabel('Score')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Distribution plots saved to: {save_path}")
    return fig

def create_violin_plots(results: Dict[str, Dict[str, Any]], save_path: str):
    """
    Create violin plots for detailed distribution analysis.
    
    Args:
        results: Dictionary containing results for all models
        save_path: Path to save the plot
    """
    # Prepare data for violin plots
    data_for_violin = []
    
    for model_name, model_data in results.items():
        for insertion, deletion, road in zip(model_data['insertion_aucs'], 
                                           model_data['deletion_aucs'], 
                                           model_data['road_scores']):
            data_for_violin.append({
                'Model': model_name.upper(),
                'Insertion_AUC': insertion,
                'Deletion_AUC': deletion,
                'ROAD_Score': road
            })
    
    df = pd.DataFrame(data_for_violin)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Detailed Score Distributions (Violin Plots)', fontsize=16, fontweight='bold')
    
    # Insertion AUC violin plot
    sns.violinplot(data=df, x='Model', y='Insertion_AUC', ax=axes[0])
    axes[0].set_title('Insertion AUC Distribution', fontweight='bold')
    axes[0].set_ylabel('Insertion AUC')
    
    # Deletion AUC violin plot
    sns.violinplot(data=df, x='Model', y='Deletion_AUC', ax=axes[1])
    axes[1].set_title('Deletion AUC Distribution', fontweight='bold')
    axes[1].set_ylabel('Deletion AUC')
    
    # ROAD Score violin plot
    sns.violinplot(data=df, x='Model', y='ROAD_Score', ax=axes[2])
    axes[2].set_title('ROAD Score Distribution', fontweight='bold')
    axes[2].set_ylabel('ROAD Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Violin plots saved to: {save_path}")
    return fig

def create_correlation_matrix(results: Dict[str, Dict[str, Any]], save_path: str):
    """
    Create correlation matrix between different metrics.
    
    Args:
        results: Dictionary containing results for all models
        save_path: Path to save the plot
    """
    # Combine all data
    all_data = []
    
    for model_name, model_data in results.items():
        for i in range(len(model_data['insertion_aucs'])):
            all_data.append({
                'Model': model_name,
                'Insertion_AUC': model_data['insertion_aucs'][i],
                'Deletion_AUC': model_data['deletion_aucs'][i],
                'ROAD_Score': model_data['road_scores'][i]
            })
    
    df = pd.DataFrame(all_data)
    
    # Create correlation matrix for each model
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Correlation Matrices Between Metrics', fontsize=16, fontweight='bold')
    
    models = ['resnet18', 'resnet34', 'resnet50']
    
    for idx, model in enumerate(models):
        model_df = df[df['Model'] == model][['Insertion_AUC', 'Deletion_AUC', 'ROAD_Score']]
        corr_matrix = model_df.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[idx], cbar_kws={'shrink': 0.8})
        axes[idx].set_title(f'{model.upper()} Correlations', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Correlation matrices saved to: {save_path}")
    return fig

def export_comprehensive_csv(results: Dict[str, Dict[str, Any]], output_dir: str):
    """
    Export comprehensive CSV files for analysis.
    
    Args:
        results: Dictionary containing results for all models
        output_dir: Directory to save CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary statistics
    summary_data = []
    for model_name, model_data in results.items():
        summary_data.append({
            'Model': model_name,
            'CAM_Method': model_data['cam_method'],
            'Num_Layers': model_data['num_layers'],
            'Num_Images': model_data['num_images'],
            'Step_Size': model_data['step_size'],
            'Insertion_AUC_Mean': model_data['insertion_auc_mean'],
            'Insertion_AUC_Std': model_data['insertion_auc_std'],
            'Deletion_AUC_Mean': model_data['deletion_auc_mean'],
            'Deletion_AUC_Std': model_data['deletion_auc_std'],
            'ROAD_Mean': model_data['road_mean'],
            'ROAD_Std': model_data['road_std']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / 'model_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary comparison exported to: {summary_path}")
    
    # 2. Detailed individual scores
    detailed_data = []
    for model_name, model_data in results.items():
        for i in range(len(model_data['insertion_aucs'])):
            detailed_data.append({
                'Model': model_name,
                'Image_Index': i,
                'Insertion_AUC': model_data['insertion_aucs'][i],
                'Deletion_AUC': model_data['deletion_aucs'][i],
                'ROAD_Score': model_data['road_scores'][i]
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_path = output_path / 'detailed_individual_scores.csv'
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✅ Detailed scores exported to: {detailed_path}")
    
    # 3. Statistical analysis
    stats_data = []
    for model_name, model_data in results.items():
        for metric in ['insertion_aucs', 'deletion_aucs', 'road_scores']:
            scores = model_data[metric]
            stats_data.append({
                'Model': model_name,
                'Metric': metric,
                'Count': len(scores),
                'Mean': np.mean(scores),
                'Std': np.std(scores),
                'Min': np.min(scores),
                'Q25': np.percentile(scores, 25),
                'Median': np.median(scores),
                'Q75': np.percentile(scores, 75),
                'Max': np.max(scores)
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_path = output_path / 'statistical_analysis.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"✅ Statistical analysis exported to: {stats_path}")

def main():
    """Main function to create all visualizations."""
    parser = argparse.ArgumentParser(description='Create comprehensive visualizations for ResNet analysis')
    parser.add_argument('--results-dir', default='XAI_Enhancer_module/analysis_results',
                       help='Directory containing analysis results')
    parser.add_argument('--output-dir', default='visualization_results',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directories
    output_path = Path(args.output_dir)
    plots_dir = output_path / 'plots'
    csv_dir = output_path / 'csv_exports'
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting comprehensive ResNet analysis visualization...")
    
    # Load all results
    results = load_all_results(args.results_dir)
    
    if not results:
        print("❌ No results loaded. Exiting.")
        return
    
    print(f"\n📊 Creating visualizations for {len(results)} models...")
    
    # Create all visualizations
    try:
        # Summary comparison
        create_summary_comparison(results, plots_dir / 'resnet_models_comparison.png')
        
        # Distribution plots
        create_distribution_plots(results, plots_dir / 'score_distributions.png')
        
        # Violin plots
        create_violin_plots(results, plots_dir / 'detailed_distributions_violin.png')
        
        # Correlation matrices
        create_correlation_matrix(results, plots_dir / 'correlation_matrices.png')
        
        # Export CSV data
        export_comprehensive_csv(results, csv_dir)
        
        print(f"\n✅ All visualizations completed successfully!")
        print(f"📁 Plots saved in: {plots_dir}")
        print(f"📊 CSV files saved in: {csv_dir}")
        
    except Exception as e:
        print(f"❌ Error during visualization creation: {e}")

if __name__ == "__main__":
    main()
