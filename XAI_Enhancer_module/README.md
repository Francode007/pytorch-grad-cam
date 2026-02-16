# XAI Enhancer Module

An extended Class Activation Mapping (CAM) framework built on top of [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), featuring **PF-CAM** (Pyramid Fusion CAM) — a multi-layer saliency map aggregation method with hierarchical top-down fusion.

## 📁 Module Structure

```
XAI_Enhancer_module/
├── pf_cam/                  # ★ PF-CAM standalone module
│   ├── extractor.py         #   CAM extraction + layer scoring
│   ├── aggregator.py        #   Pyramid fusion aggregation
│   ├── normalization.py     #   Normalization strategies
│   ├── weight_logger.py     #   Diagnostics (CSV/JSON)
│   ├── run_experiment.py    #   CLI experiment runner
│   ├── test_pf_cam.py       #   Unit tests
│   └── README.md            #   Detailed architecture & usage
├── evaluator/               # Evaluation framework
│   ├── imagenet_proper_auc_evaluator.py
│   ├── enhanced_proper_auc_evaluator.py
│   └── proper_auc_evaluation.py
├── utils/                   # Shared utilities
│   ├── model_loader.py      #   Model loading (torchvision/timm)
│   ├── model_utils.py       #   Device selection, paths
│   ├── imagenet_utils.py    #   ImageNet data utilities
│   ├── imagenet_model_utils.py
│   ├── optimized_cam_extractor.py
│   ├── optimized_predictor.py
│   ├── directory_manager.py
│   └── notification_utils.py
├── create_dataset.py        # Download ImageNet validation subset
├── download_models.py       # Download pre-trained model weights
├── XAI_ANALYSIS_REPORT.md   # Analysis report
├── XAI_ENHANCER_ELS.pdf     # Paper reference
└── imagenet_val_sample/     # Local test images
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd pytorch-grad-cam
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy pandas tqdm matplotlib pillow
pip install -e .  # Install pytorch-grad-cam
```

### 2. Download Models & Data

```bash
# Download pre-trained models
python XAI_Enhancer_module/download_models.py

# Download ImageNet validation subset (requires HF_TOKEN)
export HF_TOKEN="your_huggingface_token"
python XAI_Enhancer_module/create_dataset.py
```

### 3. Run PF-CAM

```bash
# Quick test (5 images, CPU)
python XAI_Enhancer_module/pf_cam/run_experiment.py \
    --model resnet50 --count 5 --device cpu --log-weights

# Full evaluation (GPU)
python XAI_Enhancer_module/pf_cam/run_experiment.py \
    --model resnet50 --count 500 --device cuda \
    --beta 0.3 --k-percent 0.1 --temp 0.05 \
    --norm-strategy gradient_weighted \
    --log-weights --compare-standard

# Run unit tests
python -m pytest XAI_Enhancer_module/pf_cam/test_pf_cam.py -v
```

→ See **[pf_cam/README.md](pf_cam/README.md)** for full architecture, all CLI arguments, and hyperparameter tuning.

## 📊 Evaluation Metrics

| Metric | What It Measures | Better Score |
|:-------|:-----------------|:-------------|
| **Insertion AUC** | Model confidence as top pixels are added | Higher ↑ |
| **Deletion AUC** | Model confidence as top pixels are removed | Lower ↓ |
| **ROAD** | Prediction change when important regions are removed | Lower ↓ |

## 🔧 Supported Models

`resnet18`, `resnet34`, `resnet50`, `resnet101`, `densenet121`, `efficientnet_b0`

## 📋 Requirements

```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
pandas>=1.2.0
tqdm>=4.60.0
matplotlib>=3.3.0
pillow>=8.0.0
pytorch-grad-cam (this repo)
```
