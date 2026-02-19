# IBS Pre-processed dataset pipeline (XAI_Enhancer_module)

Dataset preprocessing, training, classification evaluation, and CAM evaluation (Standard vs Enhanced) for **IBS Pre-processed** (2-class: IBS vs Normal), using the same insertion/deletion AUC and ROAD metrics as the ImageNet and Kvasir pipelines.

## Setup

**All commands below must be run from the repository root** (the `pytorch-grad-cam` directory that contains `XAI_Enhancer_module`), not from inside `XAI_Enhancer_module`. Otherwise you get `ModuleNotFoundError: No module named 'XAI_Enhancer_module'`.

Use the repository virtual environment and install dependencies:

```bash
cd /path/to/pytorch-grad-cam    # repository root
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Download IBS dataset and prepare splits (80:20)

**Recommended (works on remote server):** use the Kaggle dataset. Install dependencies (`pip install -r requirements.txt` includes `kaggle`), then set Kaggle API credentials and run:

```bash
cd /path/to/pytorch-grad-cam

# Kaggle credentials (one of):
# 1) Environment variables (good for remote server):
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key

# 2) Or place kaggle.json: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json
#    (Create API token at https://www.kaggle.com/settings)

python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root data --source kaggle --val-ratio 0.2 --seed 42
```

Dataset: [Kaggle – Pre-processed IBS](https://www.kaggle.com/datasets/franchisn/pre-processed-ibs). The script downloads and extracts it to `data/IBS-preprocessed-dataset` and creates `data/IBS-preprocessed-dataset/splits/train.txt` and `val.txt`.

**If the zip file is already on disk:** extract from local zip (no Kaggle credentials needed):

```bash
python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root data --source zip --val-ratio 0.2 --seed 42
```

**If dataset is already on disk:** create only the splits (no download):

```bash
python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root data --skip-download --val-ratio 0.2 --seed 42
```

Expected layout: `data/IBS-preprocessed-dataset/<class_name>/*.jpg` with classes: `IBS`, `Normal`.

## 2. Train

Training uses ImageNet-pretrained backbones and ImageNet mean/std normalization (transfer learning). Optimized batching options: `--num-workers`, `--pin-memory`, `--persistent-workers`, `--prefetch-factor`, `--amp`, `--grad-accum-steps`.

```bash
python -m XAI_Enhancer_module.ibs.train \
  --data-root data/IBS-preprocessed-dataset \
  --arch resnet50 \
  --batch-size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --optimizer adamw \
  --lr-scheduler cosine \
  --num-workers 4 \
  --amp \
  --output-dir runs/ibs
```

Checkpoints: `runs/ibs/resnet50/best.pth`, `last.pth`, and `metrics.json`.

### A100 40GB and sequential multi-model training

For an A100 40GB GPU you can use the `--a100` preset (batch 128, AMP bfloat16, `torch.compile`, 8 workers):

```bash
python -m XAI_Enhancer_module.ibs.train \
  --data-root data/IBS-preprocessed-dataset --arch resnet50 --epochs 50 \
  --output-dir runs/ibs --a100
```

To train several architectures **sequentially** on the server and save everything under one output directory (e.g. on a remote path), use:

```bash
python -m XAI_Enhancer_module.ibs.train_models_sequential \
  --archs resnet18 resnet34 resnet50 densenet121 vgg16 vgg19 \
  --data-root data/IBS-preprocessed-dataset \
  --output-dir /path/on/server/ibs_runs \
  --epochs 50 --a100
```

Each model is trained one after another; checkpoints go to `--output-dir/<arch>/` (e.g. `best.pth`, `last.pth`, `metrics.json`). A manifest is written to `--output-dir/sequential_training_manifest.json` with status and paths for each run.

## 3. Classification evaluation (accuracy and F1)

Uses the existing codebase (no new metric code); reports accuracy and F1 (macro/weighted) on the validation set.

```bash
python -m XAI_Enhancer_module.ibs.eval_classification \
  --data-root data/IBS-preprocessed-dataset \
  --arch resnet50 \
  --checkpoint runs/ibs/resnet50/best.pth \
  --output runs/ibs/resnet50/val_metrics.json
```

## 4. CAM evaluation (Standard vs Enhanced)

Reuses the **same** insertion AUC, deletion AUC, and ROAD metrics as in `enhanced_combiner/run_experiment.py` (no additional metric code). Compares Enhanced CAM against standard methods (GradCAM, GradCAM++, etc.) on the IBS validation set.

```bash
python -m XAI_Enhancer_module.ibs.eval_cams \
  --data-root data/IBS-preprocessed-dataset \
  --arch resnet50 \
  --checkpoint runs/ibs/resnet50/best.pth \
  --methods gradcam,gradcam++,enhancedcam \
  --enhanced-method stagewise \
  --layer-mode last \
  --output-dir runs/ibs/cam_eval
```

Output: `runs/ibs/cam_eval/comparison_report.csv` with Insertion_Mean, Deletion_Mean, ROAD_Mean per method.

**If you hit OOM** (out of memory on RAM or VRAM), use smaller batches and disable workers:
```bash
python -m XAI_Enhancer_module.ibs.eval_cams \
  --batch-size 4 \
  --layer-batch-size 4 \
  --num-workers 0 \
  --max-images 100 \
  ...
```

## .gitignore

The repo `.gitignore` already includes `/data` and `.venv` so the dataset and virtual environment are not committed.
