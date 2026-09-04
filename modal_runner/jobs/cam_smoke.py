"""
CAM eval smoke / runtime benchmark (GPU vs CPU, RAM variants).

Writes a JSON report with measured wall times and extrapolated full-matrix
estimates. Intended to be driven by Modal orchestrator ``smoke-cam-benchmark``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from modal_runner.config import (
    DEFAULT_CAM_METHODS,
    IBS_ARCHS,
    IBS_FOLDS,
    IBS_ROOT,
    KVASIR_ARCHS,
    KVASIR_ROOT,
    KVASIR_SEEDS,
)
from modal_runner.jobs.evaluate import eval_kvasir_cams
from modal_runner.runtime import ensure_layout


def _count_split_lines(split_file: Path) -> int:
    if not split_file.exists():
        return -1
    return sum(1 for line in split_file.read_text().splitlines() if line.strip())


def _read_resources(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "resources.json"
    if path.exists():
        return json.loads(path.read_text())
    # Prefer child process resources if present
    for child in sorted(out_dir.rglob("resources.json")):
        return json.loads(child.read_text())
    return {}


def run_one_smoke(
    *,
    label: str,
    arch: str,
    seed: int,
    methods: Sequence[str],
    max_images: int,
    device: str,
    output_dir: str,
    layer_set: str = "all",
    enhanced_method: str = "standard",
) -> Dict[str, Any]:
    """Run sequential multi-method CAM eval and return timing/resources."""
    ensure_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    msg = eval_kvasir_cams(
        arch=arch,
        seed=seed,
        methods=",".join(methods),
        max_images=max_images,
        enhanced_method=enhanced_method,
        layer_set=layer_set,
        output_dir=str(out),
        device=device,
    )
    res = _read_resources(out)
    # Prefer in-process eval_cams resources if nested
    nested = list(out.glob("resources.json"))
    if nested:
        res = json.loads(nested[0].read_text())
    return {
        "label": label,
        "device": device,
        "arch": arch,
        "seed": seed,
        "methods": list(methods),
        "max_images": max_images,
        "message": msg,
        "resources": res,
        "wall_s": float(res.get("wall_s", 0.0)),
        "ram_peak_mb": float(res.get("ram_peak_mb", 0.0)),
        "gpu_peak_used_mb": float(res.get("gpu_peak_used_mb", 0.0)),
        "gpu_peak_alloc_mb": float(res.get("gpu_peak_alloc_mb", 0.0)),
        "used_gpu": bool(res.get("used_gpu", False)),
    }


def build_estimate(
    *,
    smoke: Dict[str, Any],
    n_smoke: int,
    methods: Sequence[str],
    parallel_methods: bool = True,
) -> Dict[str, Any]:
    """
    Extrapolate full revision CAM matrix from one smoke timing.

    Assumptions
    -----------
    - Wall scales ~linearly with image count.
    - With ``parallel_methods=True`` (recommended Modal wave), wall per
      (arch,seed|fold) ≈ smoke_wall * (N_full / N_smoke)
      because the smoke already ran all methods (sequentially). For a fair
      parallel estimate we use ``smoke_wall / n_methods`` as a proxy for the
      slowest-method time when smoke was sequential, then multiply by scale.
    """
    n_kvasir_test = _count_split_lines(KVASIR_ROOT / "splits" / "test.txt")
    n_ibs_test = _count_split_lines(IBS_ROOT / "splits" / "fold0" / "test.txt")
    if n_kvasir_test < 0:
        n_kvasir_test = 1588
    if n_ibs_test < 0:
        n_ibs_test = 1107

    wall = float(smoke.get("wall_s") or 0.0)
    n_methods = max(1, len(methods))
    # Sequential smoke measured sum of methods; parallel wave ≈ max ≈ sum/n * 1.1 slack
    if parallel_methods:
        wall_per_run_smoke = (wall / n_methods) * 1.15
    else:
        wall_per_run_smoke = wall

    scale_k = n_kvasir_test / max(1, n_smoke)
    scale_i = n_ibs_test / max(1, n_smoke)
    t_kvasir_run = wall_per_run_smoke * scale_k
    t_ibs_run = wall_per_run_smoke * scale_i

    n_kvasir_runs = len(KVASIR_ARCHS) * len(KVASIR_SEEDS)  # 5×3
    n_ibs_runs = len(IBS_ARCHS) * len(IBS_FOLDS)  # 5×5

    # One wave at a time (current CLI pattern)
    total_seq_waves_s = n_kvasir_runs * t_kvasir_run + n_ibs_runs * t_ibs_run
    # If 5 method GPUs always busy: already baked into parallel_methods estimate
    stats_s = 600.0  # rough CPU stats over full logs

    return {
        "n_smoke": n_smoke,
        "n_methods": n_methods,
        "parallel_methods": parallel_methods,
        "n_kvasir_test": n_kvasir_test,
        "n_ibs_test": n_ibs_test,
        "n_kvasir_runs": n_kvasir_runs,
        "n_ibs_runs": n_ibs_runs,
        "sec_per_kvasir_run": round(t_kvasir_run, 1),
        "sec_per_ibs_run": round(t_ibs_run, 1),
        "hours_kvasir_all": round(n_kvasir_runs * t_kvasir_run / 3600.0, 2),
        "hours_ibs_all": round(n_ibs_runs * t_ibs_run / 3600.0, 2),
        "hours_cam_matrix_waves_serial": round(total_seq_waves_s / 3600.0, 2),
        "hours_stats_approx": round(stats_s / 3600.0, 2),
        "hours_total_approx": round((total_seq_waves_s + stats_s) / 3600.0, 2),
        "notes": [
            "Estimate assumes wall ∝ N_images and method-parallel waves (≈1.15× mean method time).",
            "ResNet-50 / VGG are slower than resnet18 smoke; multiply by ~1.5–2.5 for heavy arches.",
            "HR-CAM / Score-CAM / Opti-CAM not in DEFAULT_CAM_METHODS — add separately.",
        ],
    }


def write_benchmark_report(
    *,
    output_dir: str | Path,
    variants: List[Dict[str, Any]],
    estimate: Dict[str, Any],
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "variants": variants,
        "estimate": estimate,
        "recommendation": _recommend(variants),
    }
    path = out / "smoke_benchmark.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    # Human-readable summary
    lines = ["# CAM smoke benchmark", ""]
    for v in variants:
        lines.append(
            f"- **{v['label']}**: wall={v.get('wall_s', 0):.1f}s "
            f"RAM={v.get('ram_peak_mb', 0):.0f}MB "
            f"GPU_used={v.get('gpu_peak_used_mb', 0):.0f}MB "
            f"device={v.get('device')}"
        )
    lines += [
        "",
        "## Extrapolation (method-parallel waves)",
        f"- Kvasir test N={estimate['n_kvasir_test']}, runs={estimate['n_kvasir_runs']} "
        f"→ ~{estimate['hours_kvasir_all']} h",
        f"- IBS test N={estimate['n_ibs_test']}, runs={estimate['n_ibs_runs']} "
        f"→ ~{estimate['hours_ibs_all']} h",
        f"- **Total CAM waves (serial over runs): ~{estimate['hours_cam_matrix_waves_serial']} h**",
        f"- Stats (CPU): ~{estimate['hours_stats_approx']} h",
        f"- **Grand total ≈ {estimate['hours_total_approx']} h**",
        "",
        f"Recommendation: {report['recommendation']}",
    ]
    (out / "smoke_benchmark.md").write_text("\n".join(lines) + "\n")
    return path


def _recommend(variants: List[Dict[str, Any]]) -> str:
    by_label = {v["label"]: v for v in variants}
    gpu = by_label.get("gpu_a100")
    cpu16 = by_label.get("cpu_ram16g")
    cpu64 = by_label.get("cpu_ram64g")
    parts = []
    if gpu and cpu16 and gpu["wall_s"] > 0 and cpu16["wall_s"] > 0:
        speedup = cpu16["wall_s"] / gpu["wall_s"]
        parts.append(
            f"GPU is ~{speedup:.1f}× faster than CPU@16GB for this smoke; "
            "keep A100 for CAM eval."
        )
    if cpu16 and cpu64 and cpu16["wall_s"] > 0 and cpu64["wall_s"] > 0:
        ratio = cpu16["wall_s"] / cpu64["wall_s"]
        if ratio > 1.15:
            parts.append(
                f"Raising RAM 16→64GB sped CPU by ~{ratio:.2f}× "
                f"(likely less swapping); helpful for CPU-only jobs like stats."
            )
        else:
            parts.append(
                f"Raising RAM 16→64GB did not materially speed CPU CAM "
                f"(ratio={ratio:.2f}); CAM is compute-bound, not RAM-bound. "
                "Extra RAM still good for Phase-4 stats over large CSVs."
            )
    if not parts:
        return "Insufficient variant timings to recommend."
    return " ".join(parts)
