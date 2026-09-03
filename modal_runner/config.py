"""
Modal runner for the XAI-Enhancer CMPB revision pipeline.

Volume layout (mounted at /vol):
  /vol/data/kvasir-v2
  /vol/data/IBS-patient-dataset      # patient-level (revision default)
  /vol/data/IBS-preprocessed-dataset # legacy numeric dump (optional)
  /vol/data/ibs_groups.csv
  /vol/data/Kvasir-SEG
  /vol/models          # TORCH_HOME (ImageNet pretrained weights)
  /vol/runs/kvasir
  /vol/runs/ibs
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Identifiers (Modal dashboard / CLI)
# ---------------------------------------------------------------------------

APP_NAME = "xai-enhancer"
VOLUME_NAME = "xai-enhancer-vol"
SECRET_KAGGLE = "kaggle-credentials"  # keys: KAGGLE_USERNAME, KAGGLE_KEY

# ---------------------------------------------------------------------------
# Paths inside Modal containers
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/root/repo")
VOL_ROOT = Path("/vol")

DATA_ROOT = VOL_ROOT / "data"
KVASIR_ROOT = DATA_ROOT / "kvasir-v2"
# Patient-aware IBS tree (franchisn/ibs-dataset flattened) — used for R3-1 folds / train
IBS_ROOT = DATA_ROOT / "IBS-patient-dataset"
# Legacy flat numeric dump (franchisn/pre-processed-ibs); no recoverable exam IDs
IBS_PREPROCESSED_ROOT = DATA_ROOT / "IBS-preprocessed-dataset"
KVASIR_SEG_ROOT = DATA_ROOT / "Kvasir-SEG"
# Bundled exam map (also copied onto the volume under /vol/data/)
IBS_GROUPS_CSV_REPO = REPO_ROOT / "XAI_Enhancer_module" / "ibs" / "metadata" / "ibs_groups.csv"
IBS_GROUPS_CSV_VOL = DATA_ROOT / "ibs_groups.csv"

MODELS_ROOT = VOL_ROOT / "models"  # TORCH_HOME
RUNS_ROOT = VOL_ROOT / "runs"
KVASIR_RUNS = RUNS_ROOT / "kvasir"
IBS_RUNS = RUNS_ROOT / "ibs"

# ---------------------------------------------------------------------------
# Compute presets
# ---------------------------------------------------------------------------

GPU_TRAIN = "A100"  # training + CAM eval
GPU_LIGHT = "T4"  # light GPU jobs (optional)
TIMEOUT_DOWNLOAD_S = 4 * 60 * 60
TIMEOUT_TRAIN_S = 8 * 60 * 60
TIMEOUT_EVAL_S = 12 * 60 * 60

# Architectures used in the revision matrix
KVASIR_ARCHS = ("resnet18", "resnet34", "resnet50", "densenet121", "vgg16")
IBS_ARCHS = KVASIR_ARCHS

# Local files / dirs that must not be uploaded into the Modal image mount
LOCAL_IGNORE = [
    ".git",
    ".git/**",
    "**/.DS_Store",
    "**/__pycache__",
    "**/*.pyc",
    ".venv",
    ".venv/**",
    "XAI_Enhancer_module/.venv",
    "XAI_Enhancer_module/.venv/**",
    "data",
    "data/**",
    "runs",
    "runs/**",
    "**/*.pth",
    "**/*.pt",
    "**/*.ckpt",
    "**/*.zip",
    "**/*.pdf",
    "**/imagenet_val_sample",
    "**/imagenet_val_sample/**",
    "**/pytorch_models",
    "**/pytorch_models/**",
    "**/analysis_results",
    "**/analysis_results/**",
    "**/enhanced_results",
    "**/enhanced_results/**",
]
