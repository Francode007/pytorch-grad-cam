"""Download jobs: datasets (Kaggle) and ImageNet pretrained backbones."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from modal_runner.config import DATA_ROOT, IBS_ROOT, KVASIR_ROOT
from modal_runner.runtime import (
    configure_torch_home,
    ensure_kaggle_credentials,
    ensure_layout,
    run_module,
)

_KAGGLE_403_HINT = """
Kaggle returned 403 Forbidden on dataset download.

Most common fixes (in order):
  1. Open the dataset in a browser while logged into the SAME Kaggle account
     as your Modal secret, and click Download once (accepts terms):
       https://www.kaggle.com/datasets/franchisn/pre-processed-ibs
       https://www.kaggle.com/datasets/plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset
  2. Recreate the API token (Kaggle → Settings → Create New Token) and update
     the Modal secret:
       modal secret delete kaggle-credentials
       modal secret create kaggle-credentials KAGGLE_USERNAME=... KAGGLE_KEY=...
  3. Bypass Kaggle: upload your local zip to the Modal volume, then ingest:
       modal volume put xai-enhancer-vol data/IBS-preprocessed-dataset.zip \\
         /data/IBS-preprocessed-dataset.zip
       modal run -m modal_runner.app -- ingest-ibs-zip
"""


def download_kvasir(*, skip_if_present: bool = True, source: str = "kaggle") -> str:
    """Download + organise Kvasir-v2 under /vol/data/kvasir-v2 (also writes splits)."""
    ensure_layout()
    ensure_kaggle_credentials()
    if skip_if_present and _has_kvasir():
        return f"Kvasir-v2 already present at {KVASIR_ROOT} ({_count_images(KVASIR_ROOT)} images)"

    try:
        run_module(
            "XAI_Enhancer_module.kvasir.download_and_prepare",
            ["--data-root", str(DATA_ROOT), "--source", source],
        )
    except Exception as e:
        if "403" in str(e):
            raise RuntimeError(_KAGGLE_403_HINT) from e
        raise
    return f"Kvasir ready at {KVASIR_ROOT} ({_count_images(KVASIR_ROOT)} images)"


def download_ibs(*, skip_if_present: bool = True, source: str = "kaggle") -> str:
    """
    Download IBS pre-processed dataset only (no patient folds).

    Prefer ``ingest_ibs_zip`` if Kaggle returns 403 and you already have the zip locally.
    """
    ensure_layout()
    if source != "zip":
        ensure_kaggle_credentials()
    if skip_if_present and _has_ibs():
        return f"IBS already present at {IBS_ROOT} ({_count_images(IBS_ROOT)} images)"

    from XAI_Enhancer_module.ibs.data import download_ibs as _download_ibs

    try:
        path = _download_ibs(data_root=str(DATA_ROOT), source=source)
    except SystemExit as e:
        raise RuntimeError(
            "IBS Kaggle download failed (see logs above)." + _KAGGLE_403_HINT
        ) from e
    except Exception as e:
        if "403" in str(e):
            raise RuntimeError(_KAGGLE_403_HINT) from e
        raise
    return f"IBS ready at {path} ({_count_images(Path(path))} images)"


def ingest_ibs_zip(zip_path: Optional[str] = None) -> str:
    """
    Extract IBS from a zip already on the Modal volume.

    Default zip location: ``/vol/data/IBS-preprocessed-dataset.zip``

    Upload from your laptop first::

        modal volume put xai-enhancer-vol data/IBS-preprocessed-dataset.zip \\
          /data/IBS-preprocessed-dataset.zip
    """
    ensure_layout()
    if _has_ibs():
        return f"IBS already present at {IBS_ROOT} ({_count_images(IBS_ROOT)} images)"

    zp = Path(zip_path) if zip_path else DATA_ROOT / "IBS-preprocessed-dataset.zip"
    if not zp.exists():
        raise FileNotFoundError(
            f"Zip not found on volume: {zp}\n"
            "Upload it first (from the repo root on your laptop):\n"
            "  modal volume put xai-enhancer-vol data/IBS-preprocessed-dataset.zip "
            "/data/IBS-preprocessed-dataset.zip\n"
            "Then:\n"
            "  modal run -m modal_runner.app -- ingest-ibs-zip"
        )

    from XAI_Enhancer_module.ibs.data import _ensure_ibs_structure

    print(f"Extracting {zp} ({zp.stat().st_size / 1e9:.2f} GB) ...", flush=True)
    with tempfile.TemporaryDirectory(prefix="ibs_zip_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(tmp_path)
        path = _ensure_ibs_structure(DATA_ROOT, tmp_path)

    # Flatten accidental double nesting: .../IBS-preprocessed-dataset/IBS-preprocessed-dataset/IBS
    nested = IBS_ROOT / "IBS-preprocessed-dataset"
    if nested.is_dir() and not (IBS_ROOT / "IBS").is_dir():
        for child in nested.iterdir():
            dest = IBS_ROOT / child.name
            if not dest.exists():
                shutil.move(str(child), str(dest))
        try:
            nested.rmdir()
        except OSError:
            pass

    if not _has_ibs():
        raise RuntimeError(
            f"Extracted {zp} into {path} but could not find enough images under {IBS_ROOT}. "
            "Check: modal volume ls xai-enhancer-vol /data"
        )

    n = _count_images(IBS_ROOT)
    return f"IBS ingested from {zp} -> {IBS_ROOT} ({n} images)"


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
