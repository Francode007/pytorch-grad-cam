#!/usr/bin/env python3
"""
Usage examples for the XAI Evaluation Suite with argument parsing.
This file demonstrates how to use the new command-line interface.
"""

import subprocess
import sys
from pathlib import Path


def print_examples():
    """Print usage examples for the evaluation suite."""
    
    print("""
🚀 XAI Evaluation Suite - Usage Examples
========================================

The evaluation suite now supports command-line arguments for device selection and evaluation types.

BASIC USAGE:
-----------

1. Single model evaluation (auto device detection):
   python run_evaluation.py --model resnet50 --eval-type single

2. Quick test with minimal data:
   python run_evaluation.py --model resnet18 --eval-type quick --max-images 5

3. Individual layer experimentation:
   python run_evaluation.py --model resnet50 --eval-type individual-layers --max-layers 10

DEVICE SELECTION:
----------------

4. Use Apple Silicon GPU (MPS):
   python run_evaluation.py --model resnet50 --eval-type single --device mps

5. Force CPU usage:
   python run_evaluation.py --model resnet50 --eval-type single --device cpu

6. Use CUDA GPU:
   python run_evaluation.py --model resnet50 --eval-type single --device cuda

ADVANCED EVALUATIONS:
--------------------

7. Multiple model comparison:
   python run_evaluation.py --models resnet50 b0 resnet18 --eval-type comparison --device mps

8. Layer analysis with custom parameters:
   python run_evaluation.py --model resnet50 --eval-type layer-analysis --max-combinations 3 --max-images 20

9. Comprehensive layer experimentation:
   python run_evaluation.py --model resnet50 --eval-type comprehensive --save-detailed --max-images 50

CUSTOM PARAMETERS:
-----------------

10. Custom batch size and output directory:
    python run_evaluation.py --model resnet50 --eval-type single --batch-size 4 --output-dir ./my_results

11. Save plots and detailed results:
    python run_evaluation.py --model resnet50 --eval-type single --save-plots --save-detailed

12. Verbose output for debugging:
    python run_evaluation.py --model resnet18 --eval-type quick --verbose

EVALUATION TYPES:
----------------

Available evaluation types:
- single:           Full evaluation of a single model
- comparison:       Compare multiple models  
- layer-analysis:   Test different layer combinations
- individual-layers: Test each layer individually
- depth-analysis:   Analyze layer depth effects
- comprehensive:    Complete layer experimentation
- step-by-step:     Manual step-by-step evaluation
- quick:           Quick test with minimal data

AVAILABLE MODELS:
----------------

Supported models: resnet18, resnet34, resnet50, b0, b4, densenet, xception

DEVICE OPTIONS:
--------------

- auto (default): Automatically detect best available device
- cuda:          NVIDIA GPU with CUDA support
- mps:           Apple Silicon GPU (Metal Performance Shaders)
- cpu:           CPU only

GETTING HELP:
------------

For complete help and all options:
python run_evaluation.py --help

EXAMPLE WORKFLOW:
----------------

# 1. Start with a quick test to verify everything works
python run_evaluation.py --model resnet18 --eval-type quick --device auto

# 2. Run a full evaluation on your preferred device
python run_evaluation.py --model resnet50 --eval-type single --device mps --save-plots

# 3. Compare different models
python run_evaluation.py --models resnet50 b0 densenet --eval-type comparison --device mps

# 4. Analyze layer performance
python run_evaluation.py --model resnet50 --eval-type individual-layers --max-layers 15 --device mps

# 5. Comprehensive analysis with all details
python run_evaluation.py --model resnet50 --eval-type comprehensive --save-detailed --device mps

""")


def run_example(command_args):
    """Run an example command."""
    cmd = ["python", "run_evaluation.py"] + command_args
    print(f"🔧 Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print("✅ Success!")
            print(result.stdout)
        else:
            print("❌ Error!")
            print(result.stderr)
    except Exception as e:
        print(f"💥 Failed to run command: {e}")


def interactive_mode():
    """Interactive mode to run examples."""
    examples = [
        {
            'name': 'Quick Test (ResNet18)',
            'args': ['--model', 'resnet18', '--eval-type', 'quick', '--max-images', '3']
        },
        {
            'name': 'Single Model Evaluation (ResNet50)',
            'args': ['--model', 'resnet50', '--eval-type', 'single', '--max-images', '10']
        },
        {
            'name': 'Layer Analysis (ResNet50)',
            'args': ['--model', 'resnet50', '--eval-type', 'layer-analysis', '--max-combinations', '2', '--max-images', '5']
        },
        {
            'name': 'Individual Layers (ResNet50)',
            'args': ['--model', 'resnet50', '--eval-type', 'individual-layers', '--max-layers', '5', '--max-images', '5']
        },
        {
            'name': 'Model Comparison (ResNet18 vs B0)',
            'args': ['--models', 'resnet18', 'b0', '--eval-type', 'comparison', '--max-images', '5']
        }
    ]
    
    print("\n📋 Interactive Examples")
    print("=" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
    
    print(f"{len(examples) + 1}. Show all usage examples")
    print(f"{len(examples) + 2}. Exit")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(examples) + 2}): ").strip()
            
            if choice == str(len(examples) + 2):
                print("👋 Goodbye!")
                break
            elif choice == str(len(examples) + 1):
                print_examples()
                continue
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(examples):
                example = examples[choice_idx]
                print(f"\n🚀 Running: {example['name']}")
                run_example(example['args'])
            else:
                print("❌ Invalid choice. Please try again.")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        print_examples()
