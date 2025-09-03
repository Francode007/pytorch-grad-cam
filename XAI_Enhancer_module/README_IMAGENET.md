# ImageNet XAI Evaluation System

This module provides a comprehensive evaluation framework for XAI (Explainable AI) methods on the ImageNet dataset, extending the existing Enhanced CAM evaluation framework to work with pre-trained ImageNet models.

## Overview

The ImageNet XAI Evaluation System allows you to:

- **Evaluate Enhanced CAM methods** on ImageNet with different layer selection modes
- **Compare multiple CAM methods** (GradCAM, GradCAM++, HiResCAM, etc.) side-by-side
- **Filter by specific ImageNet classes** for targeted evaluation
- **Handle large-scale evaluation** with efficient batch processing
- **Generate comprehensive reports** with AUC metrics and statistical analysis

## Architecture

```
ImageNet XAI Evaluation System
├── imagenet_evaluation.py          # Main evaluation script
├── evaluator/
│   └── imagenet_proper_auc_evaluator.py  # Core evaluation logic
├── utils/
│   ├── imagenet_utils.py           # Synset mapping and dataset utilities
│   └── imagenet_model_utils.py     # Model loading and prediction utilities
├── imagenet_demo.py                # Demo and tutorial script
└── README_IMAGENET.md             # This documentation
```

## Key Features

### 1. Synset Mapping and Class Filtering
- **Automatic synset mapping**: Maps ImageNet synset IDs (e.g., `n01440764`) to human-readable names (e.g., `tench, Tinca tinca`)
- **Class filtering**: Evaluate specific classes by name (e.g., `'tiger'`, `'elephant'`, `'airplane'`)
- **Search functionality**: Find classes by partial name matching

### 2. Pre-trained Model Support
Supports popular ImageNet pre-trained models:
- **ResNet family**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **VGG family**: VGG16, VGG19
- **DenseNet family**: DenseNet121, DenseNet169, DenseNet201
- **MobileNet family**: MobileNetV2, MobileNetV3-Large
- **EfficientNet family**: EfficientNet-B0, EfficientNet-B4

### 3. Enhanced CAM Integration
- **Layer mode selection**: Choose from `last`, `last_5`, or `all` convolutional layers
- **Multiple Enhanced CAM methods**: GradCAMEnhanced, GradCAMPlusPlusEnhanced, HiResCAMEnhanced, etc.
- **Optimized extraction**: Efficient saliency map generation with proper layer targeting

### 4. Comprehensive Evaluation Metrics
- **Insertion AUC**: Measures how well saliency maps identify important regions
- **Deletion AUC**: Measures how well saliency maps identify regions to remove
- **ROAD Score**: Relative Remove-and-Debias metric for fairness evaluation
- **Statistical analysis**: Mean, standard deviation, and confidence intervals

## Quick Start

### 1. Setup Requirements

```bash
# Install required packages
pip install torch torchvision pillow numpy pandas tqdm

# Ensure you have pytorch-grad-cam installed
pip install grad-cam
```

### 2. Prepare ImageNet Dataset

Download the ImageNet validation dataset and organize it as:
```
imagenet_val/
├── n01440764/  # synset directories
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
│   └── ...
├── n01443537/
│   └── ...
└── ...
```

### 3. Run Demo

```bash
# Run the demo to explore capabilities
python imagenet_demo.py --imagenet-path /path/to/imagenet/val

# Test dataset validation
python imagenet_demo.py --demo validation --imagenet-path /path/to/imagenet/val
```

### 4. Basic Evaluation

```bash
# Quick comparison on a few classes
python imagenet_evaluation.py \
  --model resnet50 \
  --imagenet-path /path/to/imagenet/val \
  --eval-type comparison \
  --max-images 50 \
  --classes "tiger" "elephant" "airplane"
```

## Usage Examples

### Example 1: Enhanced CAM Only
Evaluate only the Enhanced CAM method with different layer modes:

```bash
# Using last layer only
python imagenet_evaluation.py \
  --model resnet50 \
  --imagenet-path /path/to/imagenet/val \
  --eval-type enhanced-only \
  --max-images 100 \
  --layer-mode last

# Using last 5 layers
python imagenet_evaluation.py \
  --model resnet50 \
  --imagenet-path /path/to/imagenet/val \
  --eval-type enhanced-only \
  --max-images 100 \
  --layer-mode last_5
```

### Example 2: Method Comparison
Compare Enhanced CAM against standard methods:

```bash
python imagenet_evaluation.py \
  --model resnet50 \
  --imagenet-path /path/to/imagenet/val \
  --eval-type comparison \
  --max-images 200 \
  --methods GradCAM GradCAM++ HiResCAM ScoreCAM \
  --classes "tench" "goldfish" "great white shark"
```

### Example 3: Large-Scale Evaluation
Run comprehensive evaluation with quiet mode:

```bash
python imagenet_evaluation.py \
  --model resnet50 \
  --imagenet-path /path/to/imagenet/val \
  --eval-type comparison \
  --max-images 1000 \
  --quiet \
  --output-csv-dir ./results/csv \
  --output-analysis-dir ./results/analysis
```

### Example 4: Class-Specific Analysis
Detailed per-class evaluation:

```bash
python imagenet_evaluation.py \
  --model resnet50 \
  --imagenet-path /path/to/imagenet/val \
  --eval-type class-specific \
  --classes "tiger" "lion" "cheetah" "leopard" \
  --max-images-per-class 15 \
  --methods GradCAMEnhanced GradCAM HiResCAM
```

### Example 5: Multiple Model Comparison
Compare different model architectures:

```bash
# ResNet50
python imagenet_evaluation.py --model resnet50 --imagenet-path /path/to/imagenet/val --eval-type comparison --max-images 100

# VGG16
python imagenet_evaluation.py --model vgg16 --imagenet-path /path/to/imagenet/val --eval-type comparison --max-images 100

# DenseNet121
python imagenet_evaluation.py --model densenet121 --imagenet-path /path/to/imagenet/val --eval-type comparison --max-images 100
```

## Command Line Options

### Required Arguments
- `--imagenet-path`: Path to ImageNet validation dataset

### Model Selection
- `--model`: Model to evaluate (default: `resnet50`)
  - Options: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `vgg16`, `vgg19`, `densenet121`, `densenet169`, `densenet201`, `mobilenet_v2`, `mobilenet_v3_large`, `efficientnet_b0`, `efficientnet_b4`

### Evaluation Type
- `--eval-type`: Type of evaluation (default: `comparison`)
  - `enhanced-only`: Evaluate only Enhanced CAM
  - `standard-only`: Evaluate only standard methods
  - `comparison`: Compare Enhanced CAM vs standard methods
  - `class-specific`: Detailed per-class analysis

### Dataset Configuration
- `--max-images`: Maximum number of images to evaluate (default: `50`, use `-1` for entire dataset)
- `--max-images-per-class`: For class-specific evaluation (default: `10`)
- `--classes`: Specific ImageNet class names to filter (e.g., `"tiger" "elephant"`)

### Method Selection
- `--methods`: Standard CAM methods to evaluate (default: `GradCAM GradCAM++ EigenCAM HiResCAM LayerCAM ScoreCAM`)
- `--enhanced-cam-method`: Enhanced CAM method (default: `GradCAMEnhanced`)
- `--layer-mode`: Layer selection for Enhanced CAM (default: `last`)
  - `last`: Only last convolutional layer
  - `last_5`: Last 5 convolutional layers
  - `all`: All convolutional layers

### Output Control
- `--verbose`: Force verbose output (detailed per-image logging)
- `--quiet`: Force quiet output (minimal logging)
- `--output-csv-dir`: Directory for CSV exports (default: `./csv_exports`)
- `--output-analysis-dir`: Directory for analysis results (default: `./analysis_results`)

### Hardware
- `--device`: Device preference (default: `auto`)
  - Options: `auto`, `cuda`, `mps`, `cpu`
- `--step-size`: Step size for insertion/deletion evaluation (default: `50`)

## Output Files

The evaluation system automatically organizes results by model:

```
csv_exports/
└── resnet50_imagenet/
    ├── standard_methods_20240830_143022.csv
    └── comparison_20240830_143145.csv

analysis_results/
└── resnet50_imagenet/
    ├── standard_methods_detailed_20240830_143022.pkl
    └── comparison_detailed_20240830_143145.pkl
```

### CSV Output Format
Each CSV file contains:
- `Method`: CAM method name
- `Model`: Model architecture
- `Dataset`: "ImageNet"
- `Insertion_AUC_Mean/Std`: Insertion AUC statistics
- `Deletion_AUC_Mean/Std`: Deletion AUC statistics
- `ROAD_Mean/Std`: ROAD score statistics
- `Images_Evaluated`: Number of images processed
- `Classes_Filter`: Applied class filter (if any)

### Pickle Output Format
Detailed analysis files contain:
- Raw AUC scores for each image
- Model configuration
- Dataset information
- Evaluation parameters
- Class-specific results (for class-specific evaluation)

## Class Selection Guide

### Popular ImageNet Classes for Evaluation

**Animals:**
- Marine: `"tench"`, `"goldfish"`, `"great white shark"`, `"tiger shark"`
- Birds: `"bald eagle"`, `"peacock"`, `"ostrich"`, `"hummingbird"`
- Mammals: `"tiger"`, `"lion"`, `"elephant"`, `"polar bear"`
- Domestic: `"tabby cat"`, `"golden retriever"`, `"beagle"`

**Objects:**
- Vehicles: `"sports car"`, `"airliner"`, `"motorcycle"`, `"bicycle"`
- Technology: `"laptop"`, `"cellular telephone"`, `"television"`
- Tools: `"chain saw"`, `"hammer"`, `"screwdriver"`

**Natural:**
- Plants: `"daisy"`, `"sunflower"`, `"mushroom"`
- Food: `"pizza"`, `"hamburger"`, `"ice cream"`

### Finding Classes
Use the search functionality:

```python
from XAI_Enhancer_module.utils.imagenet_utils import find_classes_by_name

# Search for classes containing specific terms
matches = find_classes_by_name(["shark", "eagle", "cat"])
```

## Performance Considerations

### Memory Usage
- **Small evaluation** (≤50 images): ~2-4 GB GPU memory
- **Medium evaluation** (≤500 images): ~4-8 GB GPU memory  
- **Large evaluation** (≥1000 images): ~8+ GB GPU memory

### Time Estimates
- **Enhanced CAM**: ~2-5 seconds per image
- **Standard methods**: ~1-3 seconds per image
- **50 images, 4 methods**: ~10-15 minutes
- **500 images, 4 methods**: ~1-2 hours
- **1000 images, 4 methods**: ~3-4 hours

### Optimization Tips
1. **Use quiet mode** for large evaluations: `--quiet`
2. **Filter classes** to reduce dataset size: `--classes "tiger" "lion"`
3. **Use GPU** when available: `--device cuda`
4. **Batch processing**: The system automatically batches predictions
5. **Layer mode**: Use `last` instead of `all` for faster Enhanced CAM

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install torch torchvision pillow numpy pandas tqdm grad-cam
```

**2. CUDA Out of Memory**
```bash
# Reduce batch size or use CPU
python imagenet_evaluation.py --device cpu --max-images 50
```

**3. Dataset Path Issues**
```bash
# Validate dataset first
python imagenet_demo.py --demo validation --imagenet-path /path/to/imagenet/val
```

**4. Class Name Not Found**
```bash
# Search for correct class names
python imagenet_demo.py --demo classes
```

### Performance Issues

**Slow evaluation:**
- Use `--device cuda` if available
- Reduce `--max-images` 
- Use `--layer-mode last` instead of `all`
- Filter specific classes with `--classes`

**High memory usage:**
- Reduce batch size in the code
- Use `--device cpu`
- Process fewer images at once

## Integration with Existing Framework

This ImageNet evaluation system is designed to integrate seamlessly with the existing Enhanced CAM framework:

1. **Consistent API**: Uses the same evaluation patterns as the original system
2. **Compatible metrics**: Same AUC calculation methods
3. **Unified output**: CSV and pickle files follow the same format
4. **Modular design**: Can be extended to other datasets

## Extending the System

### Adding New Models
To add support for new models, modify `imagenet_model_utils.py`:

```python
def load_pretrained_imagenet_model(model_name: str, device: str = "cpu"):
    model_map = {
        # ... existing models ...
        'your_new_model': your_model_function,
    }
```

### Adding New Datasets
The system can be adapted for other large-scale datasets by:
1. Creating dataset-specific utilities (similar to `imagenet_utils.py`)
2. Modifying the evaluator to handle different class mappings
3. Updating the dataset validation logic

### Custom CAM Methods
Enhanced CAM methods can be added by:
1. Implementing the method in the `enhanced_cams/` directory
2. Updating the method selection in the evaluator
3. Adding the method to the command-line options

## Citation

If you use this ImageNet evaluation system in your research, please cite:

```bibtex
@misc{imagenet_xai_evaluator,
  title={ImageNet XAI Evaluation System for Enhanced CAM Methods},
  author={Your Name},
  year={2024},
  note={Extension of Enhanced CAM evaluation framework for ImageNet dataset}
}
```

## License

This code is released under the same license as the original Enhanced CAM framework.
