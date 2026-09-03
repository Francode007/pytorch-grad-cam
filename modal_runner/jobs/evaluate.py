"""Evaluation jobs: classifier metrics and CAM faithfulness."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from modal_runner.config import IBS_ROOT, IBS_RUNS, KVASIR_ROOT, KVASIR_RUNS
from modal_runner.runtime import configure_torch_home, ensure_layout, run_module


def eval_kvasir_classification(
    *,
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    batch_size: int = 64,
    output: Optional[str] = None,
) -> str:
    ensure_layout()
    configure_torch_home()
    ckpt = checkpoint or str(KVASIR_RUNS / arch / "best.pth")
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    out = output or str(KVASIR_RUNS / arch / f"cls_{split}.json")
    args = [
        "--data-root",
        str(KVASIR_ROOT),
        "--arch",
        arch,
        "--checkpoint",
        ckpt,
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
        "--output",
        out,
    ]
    run_module("XAI_Enhancer_module.kvasir.eval_classification", args)
    return f"Classification metrics -> {out}"


def eval_ibs_classification(
    *,
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    batch_size: int = 64,
    output: Optional[str] = None,
) -> str:
    ensure_layout()
    configure_torch_home()
    ckpt = checkpoint or str(IBS_RUNS / arch / "best.pth")
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    out = output or str(IBS_RUNS / arch / f"cls_{split}.json")
    args = [
        "--data-root",
        str(IBS_ROOT),
        "--arch",
        arch,
        "--checkpoint",
        ckpt,
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
        "--output",
        out,
    ]
    run_module("XAI_Enhancer_module.ibs.eval_classification", args)
    return f"IBS classification metrics -> {out}"


def eval_kvasir_cams(
    *,
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    methods: Optional[Sequence[str]] = None,
    enhanced_method: str = "standard",
    layer_mode: str = "all",
    max_images: int = -1,
    batch_size: int = 16,
    layer_batch_size: int = 8,
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> str:
    ensure_layout()
    configure_torch_home()
    ckpt = checkpoint or str(KVASIR_RUNS / arch / "best.pth")
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    out = output_dir or str(KVASIR_RUNS / arch / "cam_eval")
    args: List[str] = [
        "--data-root",
        str(KVASIR_ROOT),
        "--arch",
        arch,
        "--checkpoint",
        ckpt,
        "--split",
        split,
        "--enhanced-method",
        enhanced_method,
        "--layer-mode",
        layer_mode,
        "--batch-size",
        str(batch_size),
        "--layer-batch-size",
        str(layer_batch_size),
        "--max-images",
        str(max_images),
        "--output-dir",
        out,
        "--device",
        "cuda",
    ]
    if methods:
        args.extend(["--methods", *methods])
    if extra_args:
        args.extend(extra_args)
    run_module("XAI_Enhancer_module.kvasir.eval_cams", args)
    return f"Kvasir CAM eval -> {out}"


def eval_ibs_cams(
    *,
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    enhanced_method: str = "standard",
    layer_mode: str = "all",
    max_images: int = -1,
    batch_size: int = 16,
    layer_batch_size: int = 8,
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> str:
    ensure_layout()
    configure_torch_home()
    ckpt = checkpoint or str(IBS_RUNS / arch / "best.pth")
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    out = output_dir or str(IBS_RUNS / arch / "cam_eval")
    args: List[str] = [
        "--data-root",
        str(IBS_ROOT),
        "--arch",
        arch,
        "--checkpoint",
        ckpt,
        "--split",
        split,
        "--enhanced-method",
        enhanced_method,
        "--layer-mode",
        layer_mode,
        "--batch-size",
        str(batch_size),
        "--layer-batch-size",
        str(layer_batch_size),
        "--max-images",
        str(max_images),
        "--output-dir",
        out,
        "--device",
        "cuda",
    ]
    if extra_args:
        args.extend(extra_args)
    run_module("XAI_Enhancer_module.ibs.eval_cams", args)
    return f"IBS CAM eval -> {out}"
