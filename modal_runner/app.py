"""
Modal App for the XAI-Enhancer full pipeline (branch ``modal_kvasir``).

Run from the repository root after ``modal setup``:

  modal run -m modal_runner.app -- --help
  modal run -m modal_runner.app -- download-models
  modal run -m modal_runner.app -- download-kvasir
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import modal

from modal_runner.config import (
    APP_NAME,
    GPU_TRAIN,
    SECRET_KAGGLE,
    TIMEOUT_DOWNLOAD_S,
    TIMEOUT_EVAL_S,
    TIMEOUT_TRAIN_S,
    VOL_ROOT,
    VOLUME_NAME,
)
from modal_runner.image import build_image
from modal_runner.runtime import ensure_layout, volume_summary

# ---------------------------------------------------------------------------
# Shared Modal resources
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)
image = build_image()
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
kaggle_secret = modal.Secret.from_name(SECRET_KAGGLE)

_VOLUMES = {str(VOL_ROOT): volume}


def _commit() -> None:
    volume.commit()


# ---------------------------------------------------------------------------
# CPU jobs — data / models
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes=_VOLUMES,
    secrets=[kaggle_secret],
    timeout=TIMEOUT_DOWNLOAD_S,
    cpu=4.0,
    memory=8192,
)
def download_kvasir(skip_if_present: bool = True, source: str = "kaggle") -> str:
    from modal_runner.jobs.download import download_kvasir as _job

    msg = _job(skip_if_present=skip_if_present, source=source)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    secrets=[kaggle_secret],
    timeout=TIMEOUT_DOWNLOAD_S,
    cpu=4.0,
    memory=8192,
)
def download_ibs(skip_if_present: bool = True, source: str = "kaggle") -> str:
    from modal_runner.jobs.download import download_ibs as _job

    msg = _job(skip_if_present=skip_if_present, source=source)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=TIMEOUT_DOWNLOAD_S,
    cpu=4.0,
    memory=8192,
)
def download_models() -> str:
    from modal_runner.jobs.download import download_models as _job

    msg = _job()
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=TIMEOUT_DOWNLOAD_S,
    cpu=8.0,
    memory=16384,
)
def prepare_kvasir_splits(dedupe: bool = True, seed: int = 42) -> str:
    from modal_runner.jobs.splits import prepare_kvasir_splits as _job

    msg = _job(dedupe=dedupe, seed=seed)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=TIMEOUT_DOWNLOAD_S,
    cpu=4.0,
    memory=8192,
)
def prepare_ibs_folds(
    groups_csv: Optional[str] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> str:
    from modal_runner.jobs.splits import prepare_ibs_folds as _job

    msg = _job(groups_csv=groups_csv, n_folds=n_folds, seed=seed)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=TIMEOUT_DOWNLOAD_S,
    cpu=8.0,
    memory=16384,
)
def smoke_splits(dedupe: bool = False) -> str:
    from modal_runner.jobs.splits import smoke_splits as _job

    msg = _job(dedupe=dedupe)
    _commit()
    return msg


@app.function(image=image, volumes=_VOLUMES, timeout=600, cpu=1.0, memory=1024)
def status() -> str:
    ensure_layout()
    return volume_summary()


# ---------------------------------------------------------------------------
# GPU jobs — train / eval
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes=_VOLUMES,
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN_S,
    memory=65536,
)
def train_kvasir(
    arch: str = "resnet50",
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
    smoke: bool = False,
) -> str:
    from modal_runner.jobs.train import train_kvasir as _job

    if smoke:
        epochs = min(epochs, 2)
        batch_size = min(batch_size, 32)
    msg = _job(arch=arch, seed=seed, epochs=epochs, batch_size=batch_size)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN_S,
    memory=65536,
)
def train_kvasir_matrix(
    archs: Optional[List[str]] = None,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
) -> str:
    from modal_runner.jobs.train import train_kvasir_matrix as _job

    msg = _job(archs=archs, seed=seed, epochs=epochs, batch_size=batch_size)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN_S,
    memory=65536,
)
def train_ibs(
    arch: str = "resnet50",
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
    smoke: bool = False,
) -> str:
    from modal_runner.jobs.train import train_ibs as _job

    if smoke:
        epochs = min(epochs, 2)
        batch_size = min(batch_size, 32)
    msg = _job(arch=arch, seed=seed, epochs=epochs, batch_size=batch_size)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_EVAL_S,
    memory=65536,
)
def eval_kvasir_classification(
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
) -> str:
    from modal_runner.jobs.evaluate import eval_kvasir_classification as _job

    msg = _job(arch=arch, checkpoint=checkpoint, split=split)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_EVAL_S,
    memory=65536,
)
def eval_ibs_classification(
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
) -> str:
    from modal_runner.jobs.evaluate import eval_ibs_classification as _job

    msg = _job(arch=arch, checkpoint=checkpoint, split=split)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_EVAL_S,
    memory=65536,
)
def eval_kvasir_cams(
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    enhanced_method: str = "standard",
    layer_mode: str = "all",
    max_images: int = -1,
) -> str:
    from modal_runner.jobs.evaluate import eval_kvasir_cams as _job

    msg = _job(
        arch=arch,
        checkpoint=checkpoint,
        split=split,
        enhanced_method=enhanced_method,
        layer_mode=layer_mode,
        max_images=max_images,
    )
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_EVAL_S,
    memory=65536,
)
def eval_ibs_cams(
    arch: str = "resnet50",
    checkpoint: Optional[str] = None,
    split: str = "test",
    enhanced_method: str = "standard",
    layer_mode: str = "all",
    max_images: int = -1,
) -> str:
    from modal_runner.jobs.evaluate import eval_ibs_cams as _job

    msg = _job(
        arch=arch,
        checkpoint=checkpoint,
        split=split,
        enhanced_method=enhanced_method,
        layer_mode=layer_mode,
        max_images=max_images,
    )
    _commit()
    return msg


# ---------------------------------------------------------------------------
# Local CLI dispatcher
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modal run -m modal_runner.app --",
        description="XAI-Enhancer Modal pipeline (modal_kvasir)",
    )
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("help", help="Show this help")
    sub.add_parser("status", help="List key paths on the Modal volume")

    dmod = sub.add_parser("download-models", help="Cache ImageNet torchvision weights")
    _ = dmod

    dk = sub.add_parser("download-kvasir", help="Download Kvasir-v2 from Kaggle")
    dk.add_argument("--force", action="store_true", help="Re-download even if present")
    dk.add_argument("--source", default="kaggle", choices=("kaggle", "simula", "manual"))

    di = sub.add_parser("download-ibs", help="Download IBS pre-processed dataset from Kaggle")
    di.add_argument("--force", action="store_true")
    di.add_argument("--source", default="kaggle", choices=("kaggle", "zip", "manual"))

    pk = sub.add_parser("prepare-kvasir-splits", help="Write 70/10/20 (+ optional pHash dedupe)")
    pk.add_argument("--seed", type=int, default=42)
    pk.add_argument("--no-dedupe", action="store_true")

    pi = sub.add_parser("prepare-ibs-folds", help="Patient-level 5-fold CV (needs groups CSV)")
    pi.add_argument("--groups-csv", required=True, help="Path on volume, e.g. /vol/data/ibs_groups.csv")
    pi.add_argument("--n-folds", type=int, default=5)
    pi.add_argument("--seed", type=int, default=42)

    sm = sub.add_parser("smoke-splits", help="Phase 1 split smoke test")
    sm.add_argument("--dedupe", action="store_true")

    tk = sub.add_parser("train-kvasir", help="Train one Kvasir classifier on A100")
    tk.add_argument("--arch", default="resnet50")
    tk.add_argument("--seed", type=int, default=42)
    tk.add_argument("--epochs", type=int, default=50)
    tk.add_argument("--batch-size", type=int, default=128)
    tk.add_argument("--smoke", action="store_true", help="2 epochs, small batch")

    tkm = sub.add_parser("train-kvasir-matrix", help="Train all revision arches sequentially")
    tkm.add_argument("--seed", type=int, default=42)
    tkm.add_argument("--epochs", type=int, default=50)
    tkm.add_argument("--batch-size", type=int, default=128)
    tkm.add_argument("--archs", nargs="+", default=None)

    ti = sub.add_parser("train-ibs", help="Train one IBS classifier on A100")
    ti.add_argument("--arch", default="resnet50")
    ti.add_argument("--seed", type=int, default=42)
    ti.add_argument("--epochs", type=int, default=50)
    ti.add_argument("--batch-size", type=int, default=128)
    ti.add_argument("--smoke", action="store_true")

    ekc = sub.add_parser("eval-kvasir-cls", help="Classifier metrics on a Kvasir split")
    ekc.add_argument("--arch", default="resnet50")
    ekc.add_argument("--split", default="test")
    ekc.add_argument("--checkpoint", default="")

    eic = sub.add_parser("eval-ibs-cls", help="Classifier metrics on an IBS split")
    eic.add_argument("--arch", default="resnet50")
    eic.add_argument("--split", default="test")
    eic.add_argument("--checkpoint", default="")

    ekcam = sub.add_parser("eval-kvasir-cams", help="CAM faithfulness eval (Kvasir)")
    ekcam.add_argument("--arch", default="resnet50")
    ekcam.add_argument("--split", default="test")
    ekcam.add_argument("--checkpoint", default="")
    ekcam.add_argument("--enhanced-method", default="standard")
    ekcam.add_argument("--layer-mode", default="all")
    ekcam.add_argument("--max-images", type=int, default=-1)

    eicam = sub.add_parser("eval-ibs-cams", help="CAM faithfulness eval (IBS)")
    eicam.add_argument("--arch", default="resnet50")
    eicam.add_argument("--split", default="test")
    eicam.add_argument("--checkpoint", default="")
    eicam.add_argument("--enhanced-method", default="standard")
    eicam.add_argument("--layer-mode", default="all")
    eicam.add_argument("--max-images", type=int, default=-1)

    return p


@app.local_entrypoint()
def main(*cli_args: str) -> None:
    """
    Dispatch a pipeline action.

    Examples
    --------
    modal run -m modal_runner.app -- download-models
    modal run -m modal_runner.app -- download-kvasir
    modal run -m modal_runner.app -- train-kvasir --arch resnet18 --smoke
    """
    parser = _build_parser()
    if not cli_args or cli_args[0] in ("-h", "--help", "help"):
        parser.print_help()
        return

    args = parser.parse_args(list(cli_args))
    action = args.action

    if action == "status":
        print(status.remote())
    elif action == "download-models":
        print(download_models.remote())
    elif action == "download-kvasir":
        print(download_kvasir.remote(skip_if_present=not args.force, source=args.source))
    elif action == "download-ibs":
        print(download_ibs.remote(skip_if_present=not args.force, source=args.source))
    elif action == "prepare-kvasir-splits":
        print(prepare_kvasir_splits.remote(dedupe=not args.no_dedupe, seed=args.seed))
    elif action == "prepare-ibs-folds":
        print(
            prepare_ibs_folds.remote(
                groups_csv=args.groups_csv, n_folds=args.n_folds, seed=args.seed,
            )
        )
    elif action == "smoke-splits":
        print(smoke_splits.remote(dedupe=args.dedupe))
    elif action == "train-kvasir":
        print(
            train_kvasir.remote(
                arch=args.arch,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                smoke=args.smoke,
            )
        )
    elif action == "train-kvasir-matrix":
        print(
            train_kvasir_matrix.remote(
                archs=args.archs,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        )
    elif action == "train-ibs":
        print(
            train_ibs.remote(
                arch=args.arch,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                smoke=args.smoke,
            )
        )
    elif action == "eval-kvasir-cls":
        print(
            eval_kvasir_classification.remote(
                arch=args.arch,
                checkpoint=args.checkpoint or None,
                split=args.split,
            )
        )
    elif action == "eval-ibs-cls":
        print(
            eval_ibs_classification.remote(
                arch=args.arch,
                checkpoint=args.checkpoint or None,
                split=args.split,
            )
        )
    elif action == "eval-kvasir-cams":
        print(
            eval_kvasir_cams.remote(
                arch=args.arch,
                checkpoint=args.checkpoint or None,
                split=args.split,
                enhanced_method=args.enhanced_method,
                layer_mode=args.layer_mode,
                max_images=args.max_images,
            )
        )
    elif action == "eval-ibs-cams":
        print(
            eval_ibs_cams.remote(
                arch=args.arch,
                checkpoint=args.checkpoint or None,
                split=args.split,
                enhanced_method=args.enhanced_method,
                layer_mode=args.layer_mode,
                max_images=args.max_images,
            )
        )
    else:
        parser.print_help()
