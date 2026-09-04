"""Evaluation jobs: classifier metrics and CAM faithfulness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

from modal_runner.config import IBS_ROOT, IBS_RUNS, KVASIR_ROOT, KVASIR_RUNS
from modal_runner.runtime import configure_torch_home, ensure_layout, run_module


DEFAULT_CAM_METHODS = (
    "gradcam",
    "gradcampp",
    "hirescam",
    "enhancedcam",
    "uniform",
)


def _method_slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def eval_kvasir_classification(
    *,
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    seed: int = 42,
    batch_size: int = 64,
    output: Optional[str] = None,
) -> str:
    ensure_layout()
    configure_torch_home()
    ckpt = checkpoint or str(KVASIR_RUNS / arch / f"seed{seed}" / "best.pth")
    if not Path(ckpt).exists():
        legacy = KVASIR_RUNS / arch / "best.pth"
        if legacy.exists() and not checkpoint:
            ckpt = str(legacy)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    out = output or str(Path(ckpt).with_name(f"cls_{split}.json"))
    args = [
        "--data-root", str(KVASIR_ROOT),
        "--arch", arch,
        "--checkpoint", ckpt,
        "--split", split,
        "--batch-size", str(batch_size),
        "--device", "cuda",
        "--output", out,
    ]
    run_module("XAI_Enhancer_module.kvasir.eval_classification", args)
    return f"Classification metrics -> {out}"


def eval_ibs_classification(
    *,
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    fold: int = 0,
    batch_size: int = 64,
    output: Optional[str] = None,
) -> str:
    ensure_layout()
    configure_torch_home()
    ckpt = checkpoint or str(IBS_RUNS / arch / f"fold{fold}" / "best.pth")
    if not Path(ckpt).exists():
        legacy = IBS_RUNS / arch / "best.pth"
        if legacy.exists() and not checkpoint:
            ckpt = str(legacy)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    out = output or str(Path(ckpt).with_name(f"cls_{split}.json"))
    args = [
        "--data-root", str(IBS_ROOT),
        "--arch", arch,
        "--fold", str(fold),
        "--checkpoint", ckpt,
        "--split", split,
        "--batch-size", str(batch_size),
        "--device", "cuda",
        "--output", out,
    ]
    run_module("XAI_Enhancer_module.ibs.eval_classification", args)
    return f"IBS classification metrics -> {out}"


def eval_kvasir_cams(
    *,
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    seed: int = 42,
    methods: Optional[Sequence[str] | str] = None,
    enhanced_method: str = "standard",
    layer_set: str = "all",
    max_images: int = -1,
    batch_size: int = 16,
    layer_batch_size: int = 8,
    step_size: int = 224,
    road_seed: int = 0,
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> str:
    from XAI_Enhancer_module.common.resource_monitor import ResourceMonitor

    ensure_layout()
    configure_torch_home()
    ckpt = checkpoint or str(KVASIR_RUNS / arch / f"seed{seed}" / "best.pth")
    if not Path(ckpt).exists():
        legacy = KVASIR_RUNS / arch / "best.pth"
        if legacy.exists() and not checkpoint:
            ckpt = str(legacy)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    out = output_dir or str(Path(ckpt).parent / "cam_eval")
    Path(out).mkdir(parents=True, exist_ok=True)
    args: List[str] = [
        "--data-root", str(KVASIR_ROOT),
        "--arch", arch,
        "--checkpoint", ckpt,
        "--split", split,
        "--enhanced-method", enhanced_method,
        "--layer-set", layer_set,
        "--batch-size", str(batch_size),
        "--layer-batch-size", str(layer_batch_size),
        "--step-size", str(step_size),
        "--road-seed", str(road_seed),
        "--max-images", str(max_images),
        "--output-dir", out,
        "--device", "cuda",
    ]
    if methods:
        method_str = methods if isinstance(methods, str) else ",".join(methods)
        args.extend(["--methods", method_str])
    if extra_args:
        args.extend(extra_args)
    label = f"kvasir_cams arch={arch} seed={seed} methods={methods}"
    with ResourceMonitor(label=label) as mon:
        run_module("XAI_Enhancer_module.kvasir.eval_cams", args)
    mon.write(Path(out) / "resources.json")
    return (
        f"Kvasir CAM eval -> {out} "
        f"(wall={mon.report.wall_s:.1f}s RAM={mon.report.ram_peak_mb:.0f}MB "
        f"GPU_used={mon.report.gpu_peak_used_mb:.0f}MB "
        f"GPU_alloc={mon.report.gpu_peak_alloc_mb:.0f}MB)"
    )


def eval_ibs_cams(
    *,
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    fold: int = 0,
    methods: Optional[Sequence[str] | str] = None,
    enhanced_method: str = "standard",
    layer_set: str = "all",
    max_images: int = -1,
    batch_size: int = 16,
    layer_batch_size: int = 8,
    step_size: int = 224,
    road_seed: int = 0,
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> str:
    from XAI_Enhancer_module.common.resource_monitor import ResourceMonitor

    ensure_layout()
    configure_torch_home()
    ckpt = checkpoint or str(IBS_RUNS / arch / f"fold{fold}" / "best.pth")
    if not Path(ckpt).exists():
        legacy = IBS_RUNS / arch / "best.pth"
        if legacy.exists() and not checkpoint:
            ckpt = str(legacy)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    out = output_dir or str(Path(ckpt).parent / "cam_eval")
    Path(out).mkdir(parents=True, exist_ok=True)
    args: List[str] = [
        "--data-root", str(IBS_ROOT),
        "--arch", arch,
        "--fold", str(fold),
        "--checkpoint", ckpt,
        "--split", split,
        "--enhanced-method", enhanced_method,
        "--layer-set", layer_set,
        "--batch-size", str(batch_size),
        "--layer-batch-size", str(layer_batch_size),
        "--step-size", str(step_size),
        "--road-seed", str(road_seed),
        "--max-images", str(max_images),
        "--output-dir", out,
        "--device", "cuda",
    ]
    if methods:
        method_str = methods if isinstance(methods, str) else ",".join(methods)
        args.extend(["--methods", method_str])
    if extra_args:
        args.extend(extra_args)
    label = f"ibs_cams arch={arch} fold={fold} methods={methods}"
    with ResourceMonitor(label=label) as mon:
        run_module("XAI_Enhancer_module.ibs.eval_cams", args)
    mon.write(Path(out) / "resources.json")
    return (
        f"IBS CAM eval -> {out} "
        f"(wall={mon.report.wall_s:.1f}s RAM={mon.report.ram_peak_mb:.0f}MB "
        f"GPU_used={mon.report.gpu_peak_used_mb:.0f}MB "
        f"GPU_alloc={mon.report.gpu_peak_alloc_mb:.0f}MB)"
    )


def merge_cam_wave_reports(base_dir: str, methods: Sequence[str]) -> str:
    """Concatenate comparison_report.csv + resources.json from by_method/*."""
    base = Path(base_dir)
    rows = []
    resources = {}
    for m in methods:
        slug = _method_slug(m)
        d = base / "by_method" / slug
        rep = d / "comparison_report.csv"
        if rep.exists():
            import pandas as pd

            df = pd.read_csv(rep)
            df["method_cli"] = m
            rows.append(df)
        res = d / "resources.json"
        if res.exists():
            resources[m] = json.loads(res.read_text())
    if rows:
        import pandas as pd

        merged = pd.concat(rows, ignore_index=True)
        merged.to_csv(base / "comparison_report.csv", index=False)
    with open(base / "wave_resources.json", "w") as f:
        json.dump(resources, f, indent=2, sort_keys=True)
    return f"Merged CAM wave reports -> {base} ({len(methods)} methods)"
