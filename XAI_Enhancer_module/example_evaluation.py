"""
Example script demonstrating the usage of the optimized XAI evaluation suite.
This script shows how to evaluate your novel XAI method efficiently.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XAI_Enhancer_module.xai_evaluation_suite import XAIEvaluationSuite, evaluate_multiple_models
from XAI_Enhancer_module.model_utils import get_validation_paths, TRAIN_DATA_PATH


def example_single_model_evaluation():
    """Example of evaluating a single model."""
    print("Example 1: Single Model Evaluation")
    print("-" * 40)
    
    # Choose a model to evaluate
    model_name = "resnet50"  # You can change this to any supported model
    
    # Create evaluation suite
    evaluator = XAIEvaluationSuite(
        model_name=model_name,
        output_dir=f"./evaluation_results_{model_name}"
    )
    
    # Run full evaluation (uses validation set by default)
    results = evaluator.run_full_evaluation(
        batch_size=8,  # Adjust based on your GPU memory
        save_results=True
    )
    
    # Print summary results
    print(f"\nResults for {model_name}:")
    print(f"Insertion AUC: {results['insertion_auc']:.4f}")
    print(f"Deletion AUC: {results['deletion_auc']:.4f}")
    print(f"ROAD Mean: {results['road_mean']:.4f} ± {results['road_std']:.4f}")
    print(f"Number of images evaluated: {results['num_images']}")
    
    # Generate plots
    evaluator.plot_results(save_plots=True)
    
    return results


def example_multiple_models_comparison():
    """Example of comparing multiple models."""
    print("\nExample 2: Multiple Models Comparison")
    print("-" * 40)
    
    # List of models to compare
    model_names = ["resnet50", "b0", "resnet18"]  # Add more models as needed
    
    # Run comparison evaluation
    comparison_df = evaluate_multiple_models(
        model_names=model_names,
        output_dir="./evaluation_results_comparison"
    )
    
    # Print comparison results
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def example_custom_image_set_evaluation():
    """Example of evaluating on a custom set of images."""
    print("\nExample 3: Custom Image Set Evaluation")
    print("-" * 40)
    
    # Get a subset of validation images for faster testing
    val_paths = get_validation_paths(TRAIN_DATA_PATH)
    
    # Use only first 50 images for quick testing
    custom_image_paths = val_paths[:50]
    
    print(f"Evaluating on {len(custom_image_paths)} custom images")
    
    # Evaluate on custom image set
    model_name = "resnet50"
    evaluator = XAIEvaluationSuite(
        model_name=model_name,
        output_dir=f"./evaluation_results_custom_{model_name}"
    )
    
    results = evaluator.run_full_evaluation(
        image_paths=custom_image_paths,
        batch_size=4,  # Smaller batch for testing
        save_results=True
    )
    
    print(f"\nCustom evaluation results for {model_name}:")
    print(f"Insertion AUC: {results['insertion_auc']:.4f}")
    print(f"Deletion AUC: {results['deletion_auc']:.4f}")
    print(f"ROAD Mean: {results['road_mean']:.4f}")
    
    return results


def example_step_by_step_evaluation():
    """Example showing step-by-step evaluation process."""
    print("\nExample 4: Step-by-step Evaluation")
    print("-" * 40)
    
    model_name = "resnet50"
    
    # Initialize evaluator
    evaluator = XAIEvaluationSuite(model_name=model_name)
    
    # Step 1: Extract saliency maps
    print("Step 1: Extracting saliency maps...")
    images, saliency_maps, image_paths = evaluator.extract_saliency_maps(
        batch_size=8
    )
    print(f"Extracted {len(saliency_maps)} saliency maps")
    
    # Step 2: Evaluate insertion/deletion metrics
    print("Step 2: Evaluating insertion/deletion metrics...")
    ins_del_results = evaluator.evaluate_insertion_deletion(
        images, saliency_maps, batch_size=4
    )
    print(f"Insertion AUC: {ins_del_results['insertion_auc']:.4f}")
    print(f"Deletion AUC: {ins_del_results['deletion_auc']:.4f}")
    
    # Step 3: Evaluate ROAD metric
    print("Step 3: Evaluating ROAD metric...")
    road_results = evaluator.evaluate_road_metric(images, saliency_maps)
    print(f"ROAD Mean: {road_results['road_mean']:.4f}")
    
    # Step 4: Save results
    print("Step 4: Saving results...")
    all_results = {
        'model_name': model_name,
        'num_images': len(images),
        'insertion_auc': ins_del_results['insertion_auc'],
        'deletion_auc': ins_del_results['deletion_auc'],
        'road_mean': road_results['road_mean'],
        'road_std': road_results['road_std'],
        'detailed_results': {
            'insertion_deletion': ins_del_results,
            'road': road_results
        }
    }
    
    evaluator.save_results(all_results)
    print("Results saved successfully!")
    
    return all_results


def quick_test():
    """Quick test with minimal data for debugging."""
    print("\nQuick Test: Minimal Evaluation")
    print("-" * 40)
    
    # Get just a few images for quick testing
    val_paths = get_validation_paths(TRAIN_DATA_PATH)
    test_paths = val_paths[:5]  # Only 5 images for quick test
    
    model_name = "resnet18"  # Smaller model for faster testing
    
    evaluator = XAIEvaluationSuite(
        model_name=model_name,
        output_dir="./quick_test_results"
    )
    
    # Run with minimal settings
    results = evaluator.run_full_evaluation(
        image_paths=test_paths,
        batch_size=2,
        save_results=True
    )
    
    print(f"Quick test completed!")
    print(f"Evaluated {results['num_images']} images")
    print(f"Insertion AUC: {results['insertion_auc']:.4f}")
    
    return results


def example_layer_analysis():
    """Example of analyzing convolutional layers and testing combinations."""
    print("\nExample 5: Layer Analysis and Combination Testing")
    print("-" * 50)
    
    model_name = "resnet50"
    
    # Initialize evaluator
    evaluator = XAIEvaluationSuite(model_name=model_name)
    
    # Print comprehensive layer summary
    evaluator.print_conv_layer_summary()
    
    # Get layer information
    conv_info = evaluator.get_all_conv_layers()
    print(f"\nFound {conv_info['total_count']} convolutional layers in {model_name}")
    
    # Get combinations for experimentation
    combinations = evaluator.get_layer_combinations_for_experimentation()
    print(f"Generated {len(combinations)} different layer combinations for testing")
    
    # Test a few combinations (using small dataset for demo)
    val_paths = get_validation_paths(TRAIN_DATA_PATH)
    test_paths = val_paths[:10]  # Only 10 images for quick testing
    
    print(f"\nTesting layer combinations on {len(test_paths)} images...")
    combo_results = evaluator.evaluate_layer_combinations(
        layer_combinations=combinations[:3],  # Test only first 3 combinations
        image_paths=test_paths,
        max_combinations=3
    )
    
    print(f"\nLayer combination evaluation completed!")
    return combo_results


def example_individual_layer_experimentation():
    """Example of testing all individual convolutional layers."""
    print("\nExample 6: Individual Layer Experimentation")
    print("-" * 50)
    
    model_name = "resnet18"  # Use smaller model for faster testing
    
    # Initialize evaluator
    evaluator = XAIEvaluationSuite(model_name=model_name)
    
    # Get a small test set
    val_paths = get_validation_paths(TRAIN_DATA_PATH)
    test_paths = val_paths[:8]  # Very small set for individual layer testing
    
    print(f"Testing individual convolutional layers on {len(test_paths)} images...")
    
    # Test individual layers
    individual_results = evaluator.experiment_all_individual_conv_layers(
        image_paths=test_paths,
        max_layers=10,  # Test only 10 layers for demo
        batch_size=1   # Small batch size
    )
    
    print(f"\nIndividual layer experimentation completed!")
    print(f"Best performing layer: {individual_results.iloc[0]['layer_name']}")
    print(f"Best Insertion AUC: {individual_results.iloc[0]['insertion_auc']:.4f}")
    
    return individual_results


def example_comprehensive_experimentation():
    """Example of comprehensive layer experimentation."""
    print("\nExample 7: Comprehensive Layer Experimentation")
    print("-" * 50)
    
    model_name = "resnet18"  # Use smaller model for comprehensive analysis
    
    # Initialize evaluator
    evaluator = XAIEvaluationSuite(model_name=model_name)
    
    # Get a test set
    val_paths = get_validation_paths(TRAIN_DATA_PATH)
    test_paths = val_paths[:12]  # Small set for comprehensive testing
    
    print(f"Running comprehensive experimentation on {len(test_paths)} images...")
    print("This will test individual layers, combinations, and depth analysis...")
    
    # Run comprehensive experimentation
    comprehensive_results = evaluator.comprehensive_layer_experimentation(
        image_paths=test_paths,
        max_individual_layers=8,  # Test 8 individual layers
        max_combinations=5,       # Test 5 combinations
        save_detailed_results=True
    )
    
    print(f"\n🎉 Comprehensive experimentation completed!")
    
    # Print key findings
    if 'summary' in comprehensive_results:
        summary = comprehensive_results['summary']
        print(f"\n📊 Key Findings:")
        
        if 'individual_layers' in summary:
            ind_summary = summary['individual_layers']
            print(f"   Best single layer AUC: {ind_summary['best_insertion_auc']:.4f}")
        
        if 'layer_combinations' in summary:
            combo_summary = summary['layer_combinations']
            print(f"   Best combination AUC: {combo_summary['best_insertion_auc']:.4f}")
            print(f"   Optimal layer count: {combo_summary['optimal_layer_count']}")
    
    return comprehensive_results


def example_comprehensive_analysis():
    """Example of comprehensive model and layer analysis."""
    print("\nExample 8: Comprehensive Analysis")
    print("-" * 40)
    
    model_name = "resnet18"  # Use smaller model for comprehensive analysis
    
    # Step 1: Initialize and analyze layers
    print("Step 1: Analyzing model architecture...")
    evaluator = XAIEvaluationSuite(model_name=model_name)
    evaluator.print_conv_layer_summary()
    
    # Step 2: Test different layer combinations
    print("\nStep 2: Testing layer combinations...")
    val_paths = get_validation_paths(TRAIN_DATA_PATH)
    test_paths = val_paths[:15]  # Use 15 images for testing
    
    combo_results = evaluator.evaluate_layer_combinations(
        image_paths=test_paths,
        max_combinations=4  # Test 4 different combinations
    )
    
    # Step 3: Run full evaluation with best combination
    print("\nStep 3: Running full evaluation with default layers...")
    full_results = evaluator.run_full_evaluation(
        image_paths=test_paths,
        save_results=True
    )
    
    # Step 4: Generate plots
    print("\nStep 4: Generating evaluation plots...")
    evaluator.plot_results()
    
    print(f"\nComprehensive analysis completed!")
    print(f"Results saved to: {evaluator.output_dir}")
    
    return {
        'layer_combinations': combo_results,
        'full_evaluation': full_results
    }


if __name__ == "__main__":
    print("XAI Evaluation Suite Examples")
    print("=" * 50)
    
    # Choose which example to run
    choice = input("""
Choose an example to run:
1. Single model evaluation (recommended for first run)
2. Multiple models comparison
3. Custom image set evaluation
4. Step-by-step evaluation
5. Quick test (minimal data)
6. Layer analysis and combinations
7. Individual layer experimentation
8. Comprehensive experimentation
9. Comprehensive analysis

Enter choice (1-9): """).strip()
    
    try:
        if choice == "1":
            example_single_model_evaluation()
        elif choice == "2":
            example_multiple_models_comparison()
        elif choice == "3":
            example_custom_image_set_evaluation()
        elif choice == "4":
            example_step_by_step_evaluation()
        elif choice == "5":
            quick_test()
        elif choice == "6":
            example_layer_analysis()
        elif choice == "7":
            example_individual_layer_experimentation()
        elif choice == "8":
            example_comprehensive_experimentation()
        elif choice == "9":
            example_comprehensive_analysis()
        else:
            print("Invalid choice. Running quick test...")
            quick_test()
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Running quick test instead...")
        quick_test()
    
    print("\nEvaluation completed!")
