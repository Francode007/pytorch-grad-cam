"""Download jobs: datasets (Kaggle) and ImageNet pretrained backbones."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from modal_runner.config import (
    DATA_ROOT,
    IBS_GROUPS_CSV_REPO,
    IBS_GROUPS_CSV_VOL,
    IBS_PREPROCESSED_ROOT,
    IBS_ROOT,
    KVASIR_ROOT,
)
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
       https://www.kaggle.com/datasets/franchisn/ibs-dataset
       https://www.kaggle.com/datasets/franchisn/pre-processed-ibs
       https://www.kaggle.com/datasets/plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset
  2. Recreate the API token (Kaggle → Settings → Create New Token) and update
     the Modal secret:
       modal secret delete kaggle-credentials
       modal secret create kaggle-credentials KAGGLE_USERNAME=... KAGGLE_KEY=...
  3. Bypass Kaggle for the legacy numeric dump: upload zip then ingest-ibs-zip.
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


def download_ibs_patient(*, skip_if_present: bool = True, force: bool = False) -> str:
    """
    Download ``franchisn/ibs-dataset``, flatten to ``IBS_ROOT``, install groups CSV.

    This is the revision default (patient/exam IDs). The numeric pre-processed
    dump is handled separately by ``download_ibs`` / ``ingest_ibs_zip``.
    """
    ensure_layout()
    ensure_kaggle_credentials()
    if skip_if_present and not force and _has_ibs_patient():
        _ensure_groups_csv_on_volume()
        return (
            f"IBS patient dataset already at {IBS_ROOT} "
            f"({_count_images(IBS_ROOT)} images); groups CSV -> {IBS_GROUPS_CSV_VOL}"
        )

    raw_root = DATA_ROOT / "ibs-raw-unnormalized"
    if force and raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print("Downloading franchisn/ibs-dataset ...", flush=True)
        api.dataset_download_files(
            "franchisn/ibs-dataset",
            path=str(raw_root),
            unzip=True,
            quiet=False,
        )
    except Exception as e:
        if "403" in str(e):
            raise RuntimeError(_KAGGLE_403_HINT) from e
        raise

    nested = raw_root / "Endoscope-Normal-IBS-Classification-Data"
    if not nested.is_dir():
        # zip may extract with an extra top folder
        candidates = [p for p in raw_root.rglob("Endoscope-Normal-IBS-Classification-Data") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"Expected Endoscope-Normal-IBS-Classification-Data under {raw_root}"
            )
        nested = candidates[0]

    args = [
        "--raw-root",
        str(nested),
        "--out-root",
        str(IBS_ROOT),
        "--out-csv",
        str(IBS_GROUPS_CSV_VOL),
        "--force",
        "--copy",
    ]
    run_module("XAI_Enhancer_module.ibs.metadata.build_ibs_groups_csv", args)
    _ensure_groups_csv_on_volume(prefer_built=IBS_GROUPS_CSV_VOL)

    n = _count_images(IBS_ROOT)
    return f"IBS patient dataset ready at {IBS_ROOT} ({n} images); groups={IBS_GROUPS_CSV_VOL}"


def download_ibs(*, skip_if_present: bool = True, source: str = "kaggle") -> str:
    """
    Download legacy IBS pre-processed (numeric) dataset only (no patient folds).

    Prefer ``download_ibs_patient`` for R3-1. Prefer ``ingest_ibs_zip`` if Kaggle
    returns 403 and you already have the zip locally.
    """
    ensure_layout()
    if source != "zip":
        ensure_kaggle_credentials()
    if skip_if_present and _has_ibs_preprocessed():
        return (
            f"IBS preprocessed already at {IBS_PREPROCESSED_ROOT} "
            f"({_count_images(IBS_PREPROCESSED_ROOT)} images)"
        )

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
    return f"IBS preprocessed ready at {path} ({_count_images(Path(path))} images)"


def ingest_ibs_zip(zip_path: Optional[str] = None) -> str:
    """
    Extract legacy numeric IBS from a zip already on the Modal volume.

    Default zip location: ``/vol/data/IBS-preprocessed-dataset.zip``
    """
    ensure_layout()
    if _has_ibs_preprocessed():
        return (
            f"IBS preprocessed already at {IBS_PREPROCESSED_ROOT} "
            f"({_count_images(IBS_PREPROCESSED_ROOT)} images)"
        )

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

    nested = IBS_PREPROCESSED_ROOT / "IBS-preprocessed-dataset"
    if nested.is_dir() and not (IBS_PREPROCESSED_ROOT / "IBS").is_dir():
        for child in nested.iterdir():
            dest = IBS_PREPROCESSED_ROOT / child.name
            if not dest.exists():
                shutil.move(str(child), str(dest))
        try:
            nested.rmdir()
        except OSError:
            pass

    if not _has_ibs_preprocessed():
        raise RuntimeError(
            f"Extracted {zp} into {path} but could not find enough images under "
            f"{IBS_PREPROCESSED_ROOT}. Check: modal volume ls xai-enhancer-vol /data"
        )

    n = _count_images(IBS_PREPROCESSED_ROOT)
    return f"IBS ingested from {zp} -> {IBS_PREPROCESSED_ROOT} ({n} images)"


def download_models() -> str:
    """Cache torchvision ImageNet weights under /vol/models (TORCH_HOME)."""
    ensure_layout()
    torch_home = configure_torch_home()
    from XAI_Enhancer_module.download_models import download_all_models

    download_all_models(custom_folder=str(torch_home))
    ckpt_dir = torch_home / "hub" / "checkpoints"
    n = len(list(ckpt_dir.glob("*.pth"))) if ckpt_dir.exists() else 0
    return f"Models cached under {torch_home} ({n} .pth files)"


def _ensure_groups_csv_on_volume(prefer_built: Optional[Path] = None) -> Path:
    """Copy the bundled repo CSV onto the volume if missing."""
    ensure_layout()
    if prefer_built and prefer_built.exists():
        if prefer_built.resolve() != IBS_GROUPS_CSV_VOL.resolve():
            shutil.copy2(prefer_built, IBS_GROUPS_CSV_VOL)
        return IBS_GROUPS_CSV_VOL
    if IBS_GROUPS_CSV_VOL.exists():
        return IBS_GROUPS_CSV_VOL
    if not IBS_GROUPS_CSV_REPO.exists():
        raise FileNotFoundError(
            f"Bundled groups CSV missing: {IBS_GROUPS_CSV_REPO}. "
            "Rebuild with build_ibs_groups_csv or check the git checkout."
        )
    shutil.copy2(IBS_GROUPS_CSV_REPO, IBS_GROUPS_CSV_VOL)
    return IBS_GROUPS_CSV_VOL


def _has_kvasir() -> bool:
    return KVASIR_ROOT.is_dir() and _count_images(KVASIR_ROOT) > 1000


def _has_ibs_patient() -> bool:
    return IBS_ROOT.is_dir() and _count_images(IBS_ROOT) > 1000


def _has_ibs_preprocessed() -> bool:
    return IBS_PREPROCESSED_ROOT.is_dir() and _count_images(IBS_PREPROCESSED_ROOT) > 1000


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        n += sum(1 for _ in root.rglob(ext))
    return n
