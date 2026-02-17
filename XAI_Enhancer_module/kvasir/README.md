# Kvasir-v2 pipeline (XAI_Enhancer_module)

Dataset preprocessing, training, classification evaluation, and CAM evaluation (Standard vs Enhanced) for **Kvasir-v2** (8-class GI tract images), using the same insertion/deletion AUC and ROAD metrics as the ImageNet pipeline.

## Setup

**All commands below must be run from the repository root** (the `pytorch-grad-cam` directory that contains `XAI_Enhancer_module`), not from inside `XAI_Enhancer_module`. Otherwise you get `ModuleNotFoundError: No module named 'XAI_Enhancer_module'`.

Use the repository virtual environment and install dependencies:

```bash
cd /path/to/pytorch-grad-cam    # repository root
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Download Kvasir-v2 and prepare splits (80:20)

The automatic download URL for Kvasir-v2 often returns 404, so **manual download is recommended**:

1. Open **[datasets.simula.no/kvasir](https://datasets.simula.no/kvasir/)** and use the **"Download Kvasir version 2"** link (kvasir-v2.zip, ~2.3GB).
2. Extract the zip so that the folder **`data/kvasir-v2`** contains the 8 class subfolders (e.g. `dyed-lifted-polyp`, `polyps`, …). If the zip has one top-level folder inside, you can put that folder as `data/kvasir-v2` or move its contents into `data/kvasir-v2`.
3. **From the repository root**, create the 80:20 train/val splits:

```bash
cd /path/to/pytorch-grad-cam
python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data --skip-download --val-ratio 0.2 --seed 42
```

If you prefer to try automatic download first (may fail with 404):

```bash
python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data --val-ratio 0.2 --seed 42
```

Expected layout: `data/kvasir-v2/<class_name>/*.jpg` with class names e.g. `dyed-lifted-polyp`, `dyed-resection-margins`, `esophagitis`, `normal-cecum`, `normal-pylorus`, `normal-z-line`, `polyps`, `ulcerative-colitis`. Splits are written to `data/kvasir-v2/splits/train.txt` and `val.txt`.

## 2. Train

Training uses ImageNet-pretrained backbones and ImageNet mean/std normalization (same as agreed for medical transfer). Optimized batching options: `--num-workers`, `--pin-memory`, `--persistent-workers`, `--prefetch-factor`, `--amp`, `--grad-accum-steps`.

```bash
python -m XAI_Enhancer_module.kvasir.train \
  --data-root data/kvasir-v2 \
  --arch resnet50 \
  --batch-size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --optimizer adamw \
  --lr-scheduler cosine \
  --num-workers 4 \
  --amp \
  --output-dir runs/kvasir
```

Checkpoints: `runs/kvasir/resnet50/best.pth`, `last.pth`, and `metrics.json`.

## 3. Classification evaluation (accuracy and F1)

Uses the existing codebase (no new metric code); reports accuracy and F1 (macro/weighted) on the validation set.

```bash
python -m XAI_Enhancer_module.kvasir.eval_classification \
  --data-root data/kvasir-v2 \
  --arch resnet50 \
  --checkpoint runs/kvasir/resnet50/best.pth \
  --output runs/kvasir/resnet50/val_metrics.json
```

## 4. CAM evaluation (Standard vs Enhanced)

Reuses the **same** insertion AUC, deletion AUC, and ROAD metrics as in `enhanced_combiner/run_experiment.py` (no additional metric code). Compares Enhanced CAM against standard methods (GradCAM, GradCAM++, etc.) on the Kvasir validation set.

```bash
python -m XAI_Enhancer_module.kvasir.eval_cams \
  --data-root data/kvasir-v2 \
  --arch resnet50 \
  --checkpoint runs/kvasir/resnet50/best.pth \
  --methods gradcam,gradcam++,enhancedcam \
  --enhanced-method stagewise \
  --layer-mode last \
  --output-dir runs/kvasir/cam_eval
```

Output: `runs/kvasir/cam_eval/comparison_report.csv` with Insertion_Mean, Deletion_Mean, ROAD_Mean per method.

## .gitignore

The repo `.gitignore` already includes `/data`, `/kvasir_data`, and `.venv` so the dataset and virtual environment are not committed.
