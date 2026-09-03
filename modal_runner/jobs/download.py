"""Download jobs: datasets (Kaggle) and ImageNet pretrained backbones."""

from __future__ import annotations

from pathlib import Path

from modal_runner.config import DATA_ROOT, IBS_ROOT, KVASIR_ROOT
from modal_runner.runtime import (
    configure_torch_home,
    ensure_kaggle_credentials,
    ensure_layout,
    run_module,
)


def download_kvasir(*, skip_if_present: bool = True, source: str = "kaggle") -> str:
    """Download + organise Kvasir-v2 under /vol/data/kvasir-v2 (also writes splits)."""
    ensure_layout()
    ensure_kaggle_credentials()
    if skip_if_present and _has_kvasir():
        return f"Kvasir-v2 already present at {KVASIR_ROOT} ({_count_images(KVASIR_ROOT)} images)"

    run_module(
        "XAI_Enhancer_module.kvasir.download_and_prepare",
        ["--data-root", str(DATA_ROOT), "--source", source],
    )
    return f"Kvasir ready at {KVASIR_ROOT} ({_count_images(KVASIR_ROOT)} images)"


def download_ibs(*, skip_if_present: bool = True, source: str = "kaggle") -> str:
    """
    Download IBS pre-processed dataset only (no patient folds).

    Patient-level folds require a groups CSV — use ``prepare_ibs_folds`` afterwards.
    """
    ensure_layout()
    ensure_kaggle_credentials()
    if skip_if_present and _has_ibs():
        return f"IBS already present at {IBS_ROOT} ({_count_images(IBS_ROOT)} images)"

    from XAI_Enhancer_module.ibs.data import download_ibs as _download_ibs

    path = _download_ibs(data_root=str(DATA_ROOT), source=source)
    return f"IBS ready at {path} ({_count_images(Path(path))} images)"


def download_models() -> str:
    """Cache torchvision ImageNet weights under /vol/models (TORCH_HOME)."""
    ensure_layout()
    torch_home = configure_torch_home()
    from XAI_Enhancer_module.download_models import download_all_models

    download_all_models(custom_folder=str(torch_home))
    ckpt_dir = torch_home / "hub" / "checkpoints"
    n = len(list(ckpt_dir.glob("*.pth"))) if ckpt_dir.exists() else 0
    return f"Models cached under {torch_home} ({n} .pth files)"


def _has_kvasir() -> bool:
    return KVASIR_ROOT.is_dir() and _count_images(KVASIR_ROOT) > 1000


def _has_ibs() -> bool:
    return IBS_ROOT.is_dir() and _count_images(IBS_ROOT) > 1000


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        n += sum(1 for _ in root.rglob(ext))
    return n
