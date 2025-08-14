# Using the Entire Validation Dataset

## Problem
You want to evaluate Enhanced CAM on your **entire validation dataset** instead of just a subset of images.

## Solution
Use `--max-images -1` to process the entire validation dataset.

## Quick Answer
```bash
# Use entire validation dataset with quiet mode (recommended for >20 images)
python modular_xai_evaluation.py --model resnet18 --max-images -1 --quiet

# Use entire validation dataset with all layers
python all_layer_analysis.py --model resnet18 --max-images -1 --quiet --save-plots
```

## Detailed Usage

### 1. **Enhanced CAM Evaluation on Entire Dataset**
```bash
# Basic usage - auto-detects verbosity based on dataset size
python modular_xai_evaluation.py --model resnet18 --max-images -1

# Force quiet mode for clean output (recommended for large datasets)
python modular_xai_evaluation.py --model resnet18 --max-images -1 --quiet

# Enhanced CAM only (no comparison with standard methods)
python modular_xai_evaluation.py --model resnet18 --max-images -1 --eval-type enhanced-only --quiet

# All layers mode with entire dataset
python modular_xai_evaluation.py --model resnet18 --max-images -1 --layer-mode all --quiet
```

### 2. **All-Layer Analysis on Entire Dataset**
```bash
# Basic all-layer analysis
python all_layer_analysis.py --model resnet18 --max-images -1 --quiet

# With visualization plots saved
python all_layer_analysis.py --model resnet18 --max-images -1 --quiet --save-plots

# Different models
python all_layer_analysis.py --model resnet50 --max-images -1 --quiet
python all_layer_analysis.py --model b0 --max-images -1 --quiet  # EfficientNet-B0
```

### 3. **Check Your Dataset Size First**
```bash
# See how many validation images you have
python validation_dataset_info.py
```

## What Happens When You Use `-1`

### **Before (Limited to N images):**
```bash
python modular_xai_evaluation.py --model resnet18 --max-images 100
# Only processes first 100 images from validation set
```

### **After (Entire dataset):**
```bash
python modular_xai_evaluation.py --model resnet18 --max-images -1
# Processes ALL images in your validation set
```

## Output Examples

### Small Dataset (≤20 images) - Auto Verbose
```bash
$ python modular_xai_evaluation.py --model resnet18 --max-images -1

Total validation images: 15
Evaluating Enhanced CAM on 15 images...

Processing: image_001.jpg
  Saliency range: [0.0234, 0.8765]
  Insertion AUC: 0.9234
  Deletion AUC: 0.1876
  ROAD Score: 0.4521
...
```

### Large Dataset (>20 images) - Auto Quiet
```bash
$ python modular_xai_evaluation.py --model resnet18 --max-images -1

Total validation images: 1000
📢 Large dataset detected (1000 images). Setting verbose=False for cleaner output.
Evaluating Enhanced CAM on 1000 images...
Verbose mode OFF - only showing progress and summary.

📊 Processed 100/1000 images. Current averages: Ins=0.923, Del=0.187, ROAD=0.451
📊 Processed 200/1000 images. Current averages: Ins=0.931, Del=0.182, ROAD=0.446
...
```

## Performance Considerations

### **Dataset Size Guidelines:**
- **≤20 images**: Use default verbose mode
- **21-100 images**: Use `--quiet` flag  
- **>100 images**: Always use `--quiet` flag
- **>500 images**: Consider running in background

### **Memory Usage:**
- Large datasets may require significant GPU memory
- Monitor GPU memory usage during evaluation
- Consider batch processing for very large datasets (>1000 images)

### **Time Estimates:**
- **Enhanced CAM**: ~2-5 seconds per image (depending on model)
- **100 images**: ~5-10 minutes
- **500 images**: ~20-40 minutes  
- **1000 images**: ~1-2 hours

## Recommended Commands for Production

### **Quick Evaluation (Enhanced CAM only):**
```bash
python modular_xai_evaluation.py \
  --model resnet18 \
  --max-images -1 \
  --eval-type enhanced-only \
  --quiet
```

### **Full Comparison (Enhanced vs Standard):**
```bash
python modular_xai_evaluation.py \
  --model resnet18 \
  --max-images -1 \
  --eval-type comparison \
  --quiet
```

### **Comprehensive Analysis with All Layers:**
```bash
python all_layer_analysis.py \
  --model resnet18 \
  --max-images -1 \
  --quiet \
  --save-plots
```

### **Background Processing (for very large datasets):**
```bash
nohup python modular_xai_evaluation.py \
  --model resnet18 \
  --max-images -1 \
  --quiet > evaluation_results.log 2>&1 &
```

## Alternative Methods (if -1 doesn't work)

If `-1` doesn't work for some reason, you can use a very large number:

```bash
# Use a large number (e.g., 10000) to effectively get all images
python modular_xai_evaluation.py --model resnet18 --max-images 10000 --quiet
```

## Troubleshooting

### **Error: "list index out of range"**
- Your validation dataset might be empty
- Check your data path configuration in `model_utils.py`

### **Memory errors with large datasets**
- Use `--quiet` mode to reduce memory usage
- Consider processing in smaller batches
- Monitor GPU memory usage

### **Very slow processing**
- Ensure you're using GPU if available (`--device cuda`)
- Use `--quiet` mode to reduce I/O overhead
- Consider using fewer layers (`--layer-mode last` instead of `all`)

## Summary

**To use your entire validation dataset:**
1. Add `--max-images -1` to your command
2. Add `--quiet` for large datasets (>20 images)
3. Monitor memory and time requirements
4. Use appropriate device settings (`--device auto`)

**Most common usage:**
```bash
python modular_xai_evaluation.py --model resnet18 --max-images -1 --quiet
```
