"""
ImageNet XAI Evaluation Suite using the Enhanced CAM framework.
This script provides comprehensive evaluation for ImageNet dataset using the modular approach.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.evaluator.imagenet_proper_auc_evaluator import ImageNetProperAUCEvaluator
from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, print_directory_structure
from XAI_Enhancer_module.utils.notification_utils import send_email_notification
import json
import glob
import numpy as np
import datetime
import os

class ImageNetXAIEvaluationSuite:
    """
    ImageNet evaluation suite using ProperAUCEvaluator as the base.
    """
    
    def __init__(self, model_name: str, imagenet_path: str, device_preference: str = "auto", 
                 layer_mode: str = "last", enhanced_cam_method: str = "GradCAMEnhanced",
                 model_cache_dir: str = "/Users/f0s03xp/pytorch_models/"):
        self.model_name = model_name
        self.imagenet_path = imagenet_path
        self.device_preference = device_preference
        self.layer_mode = layer_mode
        self.enhanced_cam_method = enhanced_cam_method
        self.model_cache_dir = model_cache_dir
        self.evaluator = ImageNetProperAUCEvaluator(
            model_name=model_name,
            imagenet_path=imagenet_path,
            device_preference=device_preference,
            layer_mode=layer_mode,
            enhanced_cam_method=enhanced_cam_method,
            model_cache_dir=model_cache_dir
        )
        print(f"ImageNetXAIEvaluationSuite initialized:")
        print(f"  Model: {model_name}")
        print(f"  Model cache dir: {model_cache_dir}")
        print(f"  ImageNet path: {imagenet_path}")
        print(f"  Device: {device_preference}")
        print(f"  Layer mode: {layer_mode}")
        print(f"  Enhanced CAM method: {enhanced_cam_method}")
    
    def evaluate_enhanced_cam(self, max_images: int = 50, step_size: int = 50, 
                            verbose: bool = None, classes_filter: List[str] = None,
                            start_index: int = 0, end_index: int = None,
                            save_intermediate: bool = False,
                            output_dir: str = ".",
                            batch_size: int = 64) -> Dict:
        print(f"\n{'='*60}")
        print("EVALUATING ENHANCED CAM ON IMAGENET")
        print(f"{'='*60}")
        if verbose is None:
            verbose = max_images <= 20
        results = self.evaluator.evaluate_enhanced_cam(
            max_images=max_images, 
            step_size=step_size,
            verbose=verbose,
            classes_filter=classes_filter,
            start_index=start_index,
            end_index=end_index,
            return_raw_data=save_intermediate,
            batch_size=batch_size
        )
        self._print_results("Enhanced CAM", results)
        
        if save_intermediate:
            self._save_intermediate_results("EnhancedCAM", results, start_index, end_index, output_dir)
            
        return results
        
    def _save_intermediate_results(self, method_name: str, results: Dict, start: int, end_index: int, output_dir: str):
        """Save raw intermediate results to JSON"""
        filename = f"partial_results_{self.model_name}_{method_name}_{start}_{end_index}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert numpy types to native Python types for JSON
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, int)):
                serializable_results[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                serializable_results[k] = float(v)
            elif isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif isinstance(v, list):
                serializable_results[k] = v
            else:
                serializable_results[k] = str(v)
                
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"✅ Saved intermediate results to {filepath}")

    def evaluate_standard_methods(self, methods: List[str] = None, max_images: int = 50,
                                base_csv_dir: str = "./csv_exports",
                                base_analysis_dir: str = "./analysis_results",
                                classes_filter: List[str] = None,
                                start_index: int = 0, end_index: int = None,
                                save_intermediate: bool = False,
                                batch_size: int = 64) -> Dict:
        if methods is None:
            methods = ["GradCAM", "GradCAM++", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"]
        results = {}
        all_results_for_export = []
        for method in methods:
            print(f"\n{'='*60}")
            print(f"EVALUATING {method} ON IMAGENET")
            print(f"{'='*60}")
            method_results = self.evaluator.evaluate_method(
                cam_method_name=method,
                max_images=max_images,
                classes_filter=classes_filter,
                start_index=start_index,
                end_index=end_index,
                return_raw_data=save_intermediate,
                batch_size=batch_size
            )
            results[method] = method_results
            self._print_results(method, method_results)
            
            if save_intermediate:
                self._save_intermediate_results(method, method_results, start_index, end_index, base_analysis_dir)
            export_row = {
                'Method': method,
                'Model': self.model_name,
                'Dataset': 'ImageNet',
                'Insertion_AUC_Mean': method_results['insertion_auc_mean'],
                'Insertion_AUC_Std': method_results['insertion_auc_std'],
                'Deletion_AUC_Mean': method_results['deletion_auc_mean'],
                'Deletion_AUC_Std': method_results['deletion_auc_std'],
                'ROAD_Mean': method_results['road_mean'],
                'ROAD_Std': method_results['road_std'],
                'Images_Evaluated': method_results['num_images'],
                'Classes_Filter': str(classes_filter) if classes_filter else 'All'
            }
            all_results_for_export.append(export_row)
        if all_results_for_export:
            from XAI_Enhancer_module.utils.directory_manager import save_evaluation_results, save_analysis_data
            results_df = pd.DataFrame(all_results_for_export)
            csv_path = save_evaluation_results(
                results_df, 
                f"{self.model_name}_imagenet", 
                evaluation_type="standard_methods",
                base_csv_dir=base_csv_dir
            )
            print(f"\n💾 ImageNet standard methods results saved to: {csv_path}")
            analysis_data = {
                'model_name': self.model_name,
                'dataset': 'ImageNet',
                'imagenet_path': self.imagenet_path,
                'evaluation_type': 'standard_methods',
                'methods_evaluated': methods,
                'detailed_results': results,
                'summary_df': results_df,
                'max_images': max_images,
                'classes_filter': classes_filter
            }
            pickle_path = save_analysis_data(
                analysis_data,
                f"{self.model_name}_imagenet",
                analysis_type="standard_methods_detailed",
                base_analysis_dir=base_analysis_dir
            )
            print(f"💾 Detailed analysis data saved to: {pickle_path}")
        return results
    
    def run_full_comparison(self, standard_methods: List[str] = None, 
                          max_images: int = 50, step_size: int = 50, 
                          verbose: bool = None, classes_filter: List[str] = None) -> pd.DataFrame:
        print(f"\n{'='*80}")
        print("FULL COMPARISON EVALUATION ON IMAGENET")
        print(f"{'='*80}")
        if standard_methods is None:
            standard_methods = ["GradCAM", "GradCAM++", "EigenCAM", "HiResCAM", "LayerCAM", "ScoreCAM"]
        if verbose is None:
            verbose = max_images <= 20
        comparison_df = self.evaluator.compare_enhanced_vs_standard(
            standard_methods=standard_methods,
            max_images=max_images,
            step_size=step_size,
            verbose=verbose,
            classes_filter=classes_filter
        )
        return comparison_df
            
    def aggregate_results(self, results_dir: str) -> pd.DataFrame:
        """Aggregate all JSON files in the directory and calculate final metrics."""
        print(f"Searching for partial result files in {results_dir}...")
        files = glob.glob(os.path.join(results_dir, "partial_results_*.json"))
        
        if not files:
            print("❌ No partial result files found.")
            return pd.DataFrame()
            
        print(f"Found {len(files)} files to aggregate.")
        
        aggregated_data = {} # method -> {metric -> [values]}
        
        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                # Extract method name from filename or assume from content
                # Filename format: partial_results_{model}_{method}_{start}_{end}.json
                parts = os.path.basename(file).replace("partial_results_", "").replace(".json", "").split("_")
                method_name_parts = parts[1:-2] # Skip model name (0), and start/end (-2, -1)
                method_name = "_".join(method_name_parts)
                
                if method_name not in aggregated_data:
                    aggregated_data[method_name] = {}
                
                # Append lists dynamically for all keys found in data
                for key, values in data.items():
                    if key not in aggregated_data[method_name]:
                        aggregated_data[method_name][key] = []
                    aggregated_data[method_name][key].extend(values)
                    
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        # Calculate final stats
        final_rows = []
        for method, metrics in aggregated_data.items():
            row = {
                'Method': method,
                'Dataset': 'ImageNet',
                'Images_Evaluated': len(metrics['insertion_auc']) if 'insertion_auc' in metrics else 0
            }
            
            # Calculate mean/std for each metric type found
            for metric_name, values in metrics.items():
                if values:
                    row[f"{metric_name}_Mean"] = np.mean(values)
                    row[f"{metric_name}_Std"] = np.std(values)
            
            final_rows.append(row)
            
        df = pd.DataFrame(final_rows)
        return df
    
    def evaluate_class_specific(self, target_classes: List[str], max_images_per_class: int = 10,
                              methods: List[str] = None) -> Dict:
        if methods is None:
            methods = ["GradCAMEnhanced", "GradCAM", "GradCAM++", "HiResCAM"]
        print(f"\n{'='*80}")
        print("CLASS-SPECIFIC IMAGENET EVALUATION")
        print(f"{'='*80}")
        print(f"Target classes: {target_classes}")
        print(f"Max images per class: {max_images_per_class}")
        print(f"Methods: {methods}")
        class_results = {}
        for target_class in target_classes:
            print(f"\n--- Evaluating class: {target_class} ---")
            class_results[target_class] = {}
            for method in methods:
                print(f"  Evaluating {method} for {target_class}...")
                if method == "GradCAMEnhanced":
                    results = self.evaluator.evaluate_enhanced_cam(
                        max_images=max_images_per_class,
                        classes_filter=[target_class],
                        verbose=False
                    )
                else:
                    results = self.evaluator.evaluate_method(
                        cam_method_name=method,
                        max_images=max_images_per_class,
                        classes_filter=[target_class]
                    )
                class_results[target_class][method] = results
                print(f"    {method}: Ins={results['insertion_auc_mean']:.3f}, "
                      f"Del={results['deletion_auc_mean']:.3f}, "
                      f"ROAD={results['road_mean']:.3f} "
                      f"({results['num_images']} imgs)")
        return class_results
    
    def _print_results(self, method_name: str, results: Dict):
        print(f"\n📊 Results for {method_name}:")
        print(f"   Insertion AUC: {results['insertion_auc_mean']:.4f} ± {results['insertion_auc_std']:.4f}")
        print(f"   Deletion AUC: {results['deletion_auc_mean']:.4f} ± {results['deletion_auc_std']:.4f}")
        print(f"   ROAD Score: {results['road_mean']:.4f} ± {results['road_std']:.4f}")
        print(f"   Images evaluated: {results['num_images']}")
        insertion_mean = results['insertion_auc_mean']
        deletion_mean = results['deletion_auc_mean']
        if 0 <= insertion_mean <= 1 and 0 <= deletion_mean <= 1:
            print(f"   ✅ AUC values are in valid [0,1] range")
        else:
            print(f"   ❌ AUC values are outside [0,1] range - check evaluation!")

def main():
    parser = argparse.ArgumentParser(
        description="ImageNet XAI Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Enhanced CAM on ImageNet with ResNet50 (using pre-downloaded models)
  python imagenet_evaluation.py --model resnet50 --imagenet-path /path/to/imagenet/val --eval-type enhanced-only --max-images 100

  # Compare methods on specific classes with custom model cache
  python imagenet_evaluation.py --model resnet50 --imagenet-path /path/to/imagenet/val --eval-type comparison --classes tench goldfish "great white shark" --max-images 20 --model-cache-dir /custom/path/to/models

  # Large scale evaluation with quiet mode
  python imagenet_evaluation.py --model resnet50 --imagenet-path /path/to/imagenet/val --eval-type comparison --max-images 1000 --quiet

  # Class-specific detailed analysis
  python imagenet_evaluation.py --model resnet50 --imagenet-path /path/to/imagenet/val --eval-type class-specific --classes tench goldfish --max-images-per-class 15

Note: Models will be loaded from --model-cache-dir (default: /Users/f0s03xp/pytorch_models/). 
      Use the download_models.py script to pre-download models to this directory.
        """
    )
    parser.add_argument('--model', '-m', default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                               'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201',
                               'mobilenet_v2', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b4'],
                       help='Model to evaluate (pre-trained ImageNet models)')
    parser.add_argument('--imagenet-path', required=True,
                       help='Path to ImageNet validation dataset')
    parser.add_argument('--eval-type', default='comparison',
                       choices=['enhanced-only', 'standard-only', 'comparison', 'class-specific'],
                       help='Type of evaluation to run')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Maximum number of images to evaluate (use -1 for entire validation dataset)')
    parser.add_argument('--max-images-per-class', type=int, default=10,
                       help='Maximum images per class for class-specific evaluation')
    parser.add_argument('--step-size', type=int, default=50,
                       help='Step size for insertion/deletion evaluation')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation (default: 64)')
    parser.add_argument('--methods', nargs='+', 
                       default=['GradCAM', 'GradCAM++', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'ScoreCAM'],
                       choices=['GradCAM', 'GradCAM++', 'EigenGradCAM', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'ScoreCAM'],
                       help='Standard CAM methods to evaluate')
    parser.add_argument('--classes', nargs='+',
                       help='Specific ImageNet class names to filter (e.g., "tench" "goldfish")')
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device preference')
    parser.add_argument('--layer-mode', default='last',
                       choices=['all', 'last_5', 'last'],
                       help='Layer selection mode for Enhanced CAM')
    parser.add_argument('--verbose', action='store_true',
                       help='Force verbose output (detailed per-image logging)')
    parser.add_argument('--quiet', action='store_true',
                       help='Force quiet output (minimal logging)')
    parser.add_argument('--output-analysis-dir', default='./analysis_results',
                       help='Base directory for analysis results')
    parser.add_argument('--output-csv-dir', default='./csv_exports',
                       help='Base directory for CSV exports')
    parser.add_argument('--enhanced-cam-method', default='GradCAMEnhanced',
                       choices=['GradCAMEnhanced', 'GradCAMPlusPlusEnhanced', 'HiResCAMEnhanced', 
                               'ScoreCAMEnhanced', 'AblationCAMEnhanced'],
                       help='Enhanced CAM method to use')
    parser.add_argument('--model-cache-dir', default='/Users/f0s03xp/pytorch_models/',
                       help='Directory containing pre-downloaded models (default: /Users/f0s03xp/pytorch_models/)')
    
    # New arguments for batch processing and aggregation
    parser.add_argument('--start-index', type=int, default=0, help='Start index for batch processing')
    parser.add_argument('--end-index', type=int, default=None, help='End index for batch processing')
    parser.add_argument('--save-intermediate', action='store_true', help='Save raw intermediate results to JSON')
    parser.add_argument('--aggregate-dir', help='Directory to aggregate results from')
    
    # Email notifications
    parser.add_argument('--email-to', help='Recipient email for notifications')
    parser.add_argument('--email-sender', help='Sender email address')
    parser.add_argument('--email-password', help='Sender email password (app password)')
    
    args = parser.parse_args()
    
    # Handle aggregation mode
    if args.aggregate_dir:
        print(f"Running in AGGREGATION MODE on {args.aggregate_dir}")
        # Dummy init just to get access to helper methods if needed, or simple static aggregation
        # We need a minimal suite instance to use `_print_results`? No, we created `aggregate_results` on the suite.
        # We'll just instantiate with dummy values as we don't need the model for aggregation.
        try:
            # Must provide required args even if dummy
            suite = ImageNetXAIEvaluationSuite(
                model_name=args.model, imagenet_path=args.imagenet_path, model_cache_dir=args.model_cache_dir
            )
            df = suite.aggregate_results(args.aggregate_dir)
            if not df.empty:
                print(f"\n{'='*80}")
                print("AGGREGATED FINAL RESULTS:")
                print(f"{'='*80}")
                print(df.to_string(index=False))
                
                # Check for Valid Range
                for _, row in df.iterrows():
                    print(f"\n{row['Method']}:")
                    if 0 <= row['Insertion_AUC_Mean'] <= 1 and 0 <= row['Deletion_AUC_Mean'] <= 1:
                        print("   ✅ AUC values are in valid [0,1] range")
                    else:
                        print("   ❌ AUC values are outside [0,1] range")
                        
                csv_path = os.path.join(args.output_csv_dir, f"{args.model}_imagenet_aggregated.csv")
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df.to_csv(csv_path, index=False)
                print(f"\n💾 Saved aggregated CSV to {csv_path}")
                
                if args.email_to:
                    subject = f"ImageNet Evaluation Completed: {args.model}"
                    body = f"Evaluation finished.\n\nAggregated Results:\n\n{df.to_string(index=False)}"
                    send_email_notification(args.email_to, subject, body, args.email_sender, args.email_password)

            return
        except Exception as e:
            print(f"Error during aggregation: {e}")
            import traceback
            traceback.print_exc()
            return
            
    if args.verbose and args.quiet:
        print("❌ Error: Cannot specify both --verbose and --quiet")
        return
    if args.verbose:
        verbose = True
    elif args.quiet:
        verbose = False
    else:
        verbose = None
    print(f"\n{'='*80}")
    print("IMAGENET XAI EVALUATION SUITE")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Model cache dir: {args.model_cache_dir}")
    print(f"  ImageNet path: {args.imagenet_path}")
    print(f"  Evaluation type: {args.eval_type}")
    print(f"  Max images: {args.max_images}")
    print(f"  Step size: {args.step_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Layer mode: {args.layer_mode}")
    if args.classes:
        print(f"  Classes filter: {args.classes}")
    if verbose is not None:
        print(f"  Verbose: {verbose}")
    else:
        print(f"  Verbose: Auto (True for ≤20 images, False for >20 images)")
    try:
        suite = ImageNetXAIEvaluationSuite(
            model_name=args.model,
            imagenet_path=args.imagenet_path,
            device_preference=args.device,
            layer_mode=args.layer_mode,
            enhanced_cam_method=args.enhanced_cam_method,
            model_cache_dir=args.model_cache_dir
        )
        if args.eval_type == 'enhanced-only':
            enhanced_results = suite.evaluate_enhanced_cam(
                max_images=args.max_images,
                step_size=args.step_size,
                verbose=verbose,
                classes_filter=args.classes,
                start_index=args.start_index,
                end_index=args.end_index,
                save_intermediate=args.save_intermediate,
                output_dir=args.output_analysis_dir,
                batch_size=args.batch_size
            )
            
            if args.email_to:
                subject = f"ImageNet Batch {args.start_index}-{args.end_index}: Enhanced CAM"
                body = f"Evaluation completed for {args.model} on indices {args.start_index} to {args.end_index}.\n\n"
                body += f"Insertion AUC: {enhanced_results['insertion_auc_mean']:.4f}\n"
                body += f"Deletion AUC: {enhanced_results['deletion_auc_mean']:.4f}\n"
                body += f"ROAD Score: {enhanced_results['road_mean']:.4f}\n"
                send_email_notification(args.email_to, subject, body, args.email_sender, args.email_password)
                
        elif args.eval_type == 'standard-only':
            standard_results = suite.evaluate_standard_methods(
                methods=args.methods,
                max_images=args.max_images,
                base_csv_dir=args.output_csv_dir,
                base_analysis_dir=args.output_analysis_dir,
                classes_filter=args.classes,
                start_index=args.start_index,
                end_index=args.end_index,
                save_intermediate=args.save_intermediate,
                batch_size=args.batch_size
            )
            if args.email_to:
                subject = f"ImageNet Batch {args.start_index}-{args.end_index}: Standard Methods"
                body = f"Evaluation completed for {args.model} on indices {args.start_index} to {args.end_index}.\n\nMethods: {args.methods}"
                send_email_notification(args.email_to, subject, body, args.email_sender, args.email_password)
        elif args.eval_type == 'comparison':
            # Note: comparison mode doesn't cleanly support batching/intermediate saving as cleanly in the original script
            # because it wraps other calls. We'll simplify for now by NOT heavily refactoring comparison mode 
            # but printing a warning if users try to use batching with it, or relying on manual sequential calls.
            # Actually, the user's request is satisfied by running separate/sequential calls in the notebook.
            # But let's defer this specific update or handle it if easy.
            # For now, let's assume the user will use 'enhanced-only' or 'standard-only' in batches, 
            # Or we can just adapt comparison to call the new signatures.
            
            # Since the user wants to run 300 samples each time, they will likely run:
            # 1. Enhanced CAM pass
            # 2. Standard methods pass
            # OR we can just allow comparison to run both sequentially.
            
            # Let's just print a warning if they try comparison with batching features
            if args.start_index > 0 or args.end_index is not None:
                print("⚠️ Warning: 'comparison' mode runs Enhanced AND Standard methods sequentially.")
                print("   If you want to save intermediate results for aggregation, it's safer to run 'enhanced-only' and 'standard-only' separately.")
            
            comparison_df = suite.run_full_comparison(
                standard_methods=args.methods,
                max_images=args.max_images,
                step_size=args.step_size,
                verbose=verbose,
                classes_filter=args.classes
            )
            print(f"\n{'='*80}")
            print("FINAL COMPARISON TABLE:")
            print(f"{'='*80}")
            print(comparison_df.to_string(index=False))
        elif args.eval_type == 'class-specific':
            if not args.classes:
                print("❌ Error: --classes must be specified for class-specific evaluation")
                return
            class_results = suite.evaluate_class_specific(
                target_classes=args.classes,
                max_images_per_class=args.max_images_per_class,
                methods=['GradCAMEnhanced'] + args.methods
            )
            print(f"\n{'='*80}")
            print("CLASS-SPECIFIC EVALUATION SUMMARY:")
            print(f"{'='*80}")
            for class_name, class_data in class_results.items():
                print(f"\n{class_name}:")
                for method, results in class_data.items():
                    print(f"  {method}: Ins={results['insertion_auc_mean']:.3f}, "
                          f"Del={results['deletion_auc_mean']:.3f}, "
                          f"ROAD={results['road_mean']:.3f}")
        print(f"\n✅ ImageNet evaluation completed successfully!")
        print(f"\n📁 OUTPUT SUMMARY:")
        print(f"{'='*50}")
        print(f"Results have been automatically saved to model-specific directories:")
        print(f"• Analysis results: {args.output_analysis_dir}/{args.model}_imagenet/")
        print(f"• CSV exports: {args.output_csv_dir}/{args.model}_imagenet/")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
