"""Training jobs for Kvasir-v2 and IBS classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from modal_runner.config import (
    IBS_ARCHS,
    IBS_ROOT,
    IBS_RUNS,
    KVASIR_ARCHS,
    KVASIR_BATCH_SIZE_DEFAULT,
    KVASIR_ROOT,
    KVASIR_RUNS,
)
from modal_runner.runtime import configure_torch_home, ensure_layout, run_module


def _gpu_train_args(
    *,
    data_root: str,
    arch: str,
    seed: int,
    epochs: int,
    batch_size: int,
    output_dir: str,
    amp: bool,
    amp_dtype: str,
    compile_model: bool,
    fold: Optional[int] = None,
    resume: str = "",
    auto_batch: bool = False,
    a100: bool = True,
    locked_batch_file: Optional[str] = None,
) -> List[str]:
    args: List[str] = [
        "--data-root", data_root,
        "--arch", arch,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--output-dir", output_dir,
        "--device", "cuda",
        "--num-workers", "8",
    ]
    if fold is not None:
        args.extend(["--fold", str(fold)])
    if amp:
        args.extend(["--amp", "--amp-dtype", amp_dtype])
    if compile_model:
        args.append("--compile")
    if a100:
        args.append("--a100")
    # Prefer locked_batch_sizes.json when present; only probe if unlocked.
    if auto_batch:
        args.append("--auto-batch-size")
    if resume:
        args.extend(["--resume", resume])
    if locked_batch_file:
        args.extend(["--locked-batch-file", locked_batch_file])
    return args


def train_kvasir(
    *,
    arch: str = "resnet50",
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = KVASIR_BATCH_SIZE_DEFAULT,
    amp: bool = True,
    amp_dtype: str = "bfloat16",
    compile_model: bool = True,
    output_dir: Optional[str] = None,
    resume: str = "",
    auto_batch: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> str:
    """Train one Kvasir classifier → /vol/runs/kvasir/{arch}/seed{seed}/."""
    ensure_layout()
    configure_torch_home()
    if not KVASIR_ROOT.exists():
        raise FileNotFoundError(f"Missing {KVASIR_ROOT}. Run download-kvasir first.")

    out = output_dir or str(KVASIR_RUNS)
    args = _gpu_train_args(
        data_root=str(KVASIR_ROOT),
        arch=arch,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=out,
        amp=amp,
        amp_dtype=amp_dtype,
        compile_model=compile_model,
        resume=resume,
        auto_batch=auto_batch,
        a100=True,
        locked_batch_file=str(Path(out) / "locked_batch_sizes.json"),
    )
    if extra_args:
        args.extend(extra_args)
    run_module("XAI_Enhancer_module.kvasir.train", args)
    return f"Kvasir train done: arch={arch} seed={seed} -> {out}/{arch}/seed{seed}"


def train_kvasir_matrix(
    *,
    archs: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    epochs: int = 50,
    batch_size: int = KVASIR_BATCH_SIZE_DEFAULT,
) -> str:
    """Train Kvasir arches × seeds sequentially (legacy; prefer parallel seed fan-out)."""
    chosen = list(archs) if archs else list(KVASIR_ARCHS)
    seed_list = list(seeds) if seeds else [42, 43, 44]
    lines = []
    for arch in chosen:
        for seed in seed_list:
            lines.append(
                train_kvasir(arch=arch, seed=seed, epochs=epochs, batch_size=batch_size)
            )
    return "\n".join(lines)


def train_ibs(
    *,
    arch: str = "resnet50",
    fold: int = 0,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
    amp: bool = True,
    amp_dtype: str = "bfloat16",
    compile_model: bool = True,
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> str:
    """Train one IBS fold → /vol/runs/ibs/{arch}/fold{fold}/."""
    ensure_layout()
    configure_torch_home()
    if not IBS_ROOT.exists():
        raise FileNotFoundError(f"Missing {IBS_ROOT}. Run download-ibs-patient first.")

    out = output_dir or str(IBS_RUNS)
    args = _gpu_train_args(
        data_root=str(IBS_ROOT),
        arch=arch,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=out,
        amp=amp,
        amp_dtype=amp_dtype,
        compile_model=compile_model,
        fold=fold,
        auto_batch=False,
        a100=False,
    )
    if extra_args:
        args.extend(extra_args)
    run_module("XAI_Enhancer_module.ibs.train", args)
    return f"IBS train done: arch={arch} fold={fold} -> {out}/{arch}/fold{fold}"


def train_ibs_matrix(
    *,
    archs: Optional[Sequence[str]] = None,
    folds: Optional[Sequence[int]] = None,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
) -> str:
    """Train IBS arches × folds sequentially."""
    chosen = list(archs) if archs else list(IBS_ARCHS)
    fold_list = list(folds) if folds else [0, 1, 2, 3, 4]
    lines = []
    for arch in chosen:
        for fold in fold_list:
            lines.append(
                train_ibs(
                    arch=arch,
                    fold=fold,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                )
            )
    return "\n".join(lines)
