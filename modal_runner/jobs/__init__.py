"""Job package for Modal-hosted XAI-Enhancer pipeline steps."""

from modal_runner.jobs.download import download_ibs, download_kvasir, download_models
from modal_runner.jobs.evaluate import (
    eval_ibs_cams,
    eval_ibs_classification,
    eval_kvasir_cams,
    eval_kvasir_classification,
)
from modal_runner.jobs.splits import prepare_ibs_folds, prepare_kvasir_splits, smoke_splits
from modal_runner.jobs.train import (
    train_ibs,
    train_ibs_matrix,
    train_kvasir,
    train_kvasir_matrix,
)

__all__ = [
    "download_ibs",
    "download_kvasir",
    "download_models",
    "eval_ibs_cams",
    "eval_ibs_classification",
    "eval_kvasir_cams",
    "eval_kvasir_classification",
    "prepare_ibs_folds",
    "prepare_kvasir_splits",
    "smoke_splits",
    "train_ibs",
    "train_ibs_matrix",
    "train_kvasir",
    "train_kvasir_matrix",
]
