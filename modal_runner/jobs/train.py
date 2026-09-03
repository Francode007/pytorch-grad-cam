"""Training jobs for Kvasir-v2 and IBS classifiers."""

from __future__ import annotations

from typing import List, Optional, Sequence

from modal_runner.config import IBS_ARCHS, IBS_ROOT, IBS_RUNS, KVASIR_ARCHS, KVASIR_ROOT, KVASIR_RUNS
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
    if amp:
        args.extend(["--amp", "--amp-dtype", amp_dtype])
    if compile_model:
        args.append("--compile")
    return args


def train_kvasir(
    *,
    arch: str = "resnet50",
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
    amp: bool = True,
    amp_dtype: str = "bfloat16",
    compile_model: bool = True,
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> str:
    """Train one Kvasir classifier; checkpoints land under /vol/runs/kvasir."""
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
    )
    if extra_args:
        args.extend(extra_args)
    run_module("XAI_Enhancer_module.kvasir.train", args)
    return f"Kvasir train done: arch={arch} seed={seed} -> {out}/{arch}"


def train_kvasir_matrix(
    *,
    archs: Optional[Sequence[str]] = None,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
) -> str:
    """Train several Kvasir architectures sequentially."""
    chosen = list(archs) if archs else list(KVASIR_ARCHS)
    return "\n".join(
        train_kvasir(arch=arch, seed=seed, epochs=epochs, batch_size=batch_size)
        for arch in chosen
    )


def train_ibs(
    *,
    arch: str = "resnet50",
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
    amp: bool = True,
    amp_dtype: str = "bfloat16",
    compile_model: bool = True,
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> str:
    """Train one IBS classifier; checkpoints land under /vol/runs/ibs."""
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
    )
    if extra_args:
        args.extend(extra_args)
    run_module("XAI_Enhancer_module.ibs.train", args)
    return f"IBS train done: arch={arch} seed={seed} -> {out}/{arch}"


def train_ibs_matrix(
    *,
    archs: Optional[Sequence[str]] = None,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
) -> str:
    """Train several IBS architectures sequentially."""
    chosen = list(archs) if archs else list(IBS_ARCHS)
    return "\n".join(
        train_ibs(arch=arch, seed=seed, epochs=epochs, batch_size=batch_size)
        for arch in chosen
    )
