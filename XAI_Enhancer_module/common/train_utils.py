"""Shared training helpers for Phase 2 (seed / fold / val macro-F1)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(output_dir: str | Path, arch: str, *, seed: Optional[int] = None, fold: Optional[int] = None) -> Path:
    """
    Layout:
      Kvasir: {output_dir}/{arch}/seed{seed}/
      IBS:    {output_dir}/{arch}/fold{fold}/
    """
    base = Path(output_dir) / arch
    if fold is not None:
        return base / f"fold{fold}"
    if seed is not None:
        return base / f"seed{seed}"
    return base


def write_args_json(out_dir: Path, args: Any, extra: Optional[Dict[str, Any]] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(vars(args)) if hasattr(args, "__dict__") else dict(args)
    if extra:
        payload.update(extra)
    path = out_dir / "args.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, preds, probs) as numpy arrays."""
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    for images, labels, _ in loader:
        images = images.to(device)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(images)
        else:
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu().numpy())
    y = np.asarray(all_labels, dtype=np.int64)
    pred = np.asarray(all_preds, dtype=np.int64)
    prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0))
    return y, pred, prob


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    class_names: Optional[Sequence[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Accuracy, macro/weighted F1, AUROC (OVR), ECE, per-class report."""
    n_classes = num_classes or int(y_prob.shape[1]) if y_prob.size else int(y_true.max() + 1)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        if len(y_true)
        else 0.0,
        "n_samples": int(len(y_true)),
        "n_classes": int(n_classes),
    }
    metrics["ece"] = float(_expected_calibration_error(y_true, y_prob, n_bins=15))

    try:
        if n_classes == 2 and y_prob.shape[1] >= 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auroc_macro_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            metrics["auroc"] = metrics["auroc_macro_ovr"]
    except ValueError:
        metrics["auroc"] = None

    labels = list(range(n_classes))
    target_names = list(class_names) if class_names is not None else [str(i) for i in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    metrics["per_class"] = report
    return metrics


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    if len(y_true) == 0 or y_prob.size == 0:
        return 0.0
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if not np.any(mask):
            continue
        ece += abs(accuracies[mask].mean() - confidences[mask].mean()) * (mask.sum() / len(y_true))
    return float(ece)
