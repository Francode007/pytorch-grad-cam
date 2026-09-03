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
    KVASIR_ARCHS,
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
    memory=16384,
)
def download_ibs_patient(skip_if_present: bool = True, force: bool = False) -> str:
    """Download franchisn/ibs-dataset → flat IBS-patient-dataset + groups CSV."""
    from modal_runner.jobs.download import download_ibs_patient as _job

    msg = _job(skip_if_present=skip_if_present, force=force)
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
    """Legacy numeric pre-processed dump (no exam IDs). Prefer download-ibs-patient."""
    from modal_runner.jobs.download import download_ibs as _job

    msg = _job(skip_if_present=skip_if_present, source=source)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=TIMEOUT_DOWNLOAD_S,
    cpu=4.0,
    memory=16384,
)
def ingest_ibs_zip(zip_path: str = "") -> str:
    """Extract legacy numeric IBS from a zip already on the volume (bypass Kaggle)."""
    from modal_runner.jobs.download import ingest_ibs_zip as _job

    msg = _job(zip_path=zip_path or None)
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


@app.function(image=image, volumes=_VOLUMES, timeout=600, cpu=1.0, memory=2048)
def summarize_kvasir_seed(
    seed: int = 42,
    archs: Optional[List[str]] = None,
    train_results: Optional[List[str]] = None,
) -> str:
    """
    After a seed wave: print cost/metrics table, lock batch sizes, save
    /vol/runs/kvasir/waves/seed{seed}/wave_summary.{json,txt} + wave.log.
    """
    from modal_runner.jobs.summarize import summarize_kvasir_seed as _job

    ensure_layout()
    volume.reload()
    msg = _job(seed=seed, archs=archs, train_results=train_results)
    _commit()
    return msg


@app.function(image=image, volumes=_VOLUMES, timeout=600, cpu=1.0, memory=2048)
def reset_kvasir_seed(
    seed: int = 42,
    archs: Optional[List[str]] = None,
    clear_locks: bool = True,
    wipe_runs: bool = True,
) -> str:
    """Clear locked batch sizes and wipe ``{arch}/seed{seed}`` run dirs."""
    from modal_runner.jobs.reset import reset_kvasir_seed as _job

    ensure_layout()
    volume.reload()
    msg = _job(seed=seed, archs=archs, clear_locks=clear_locks, wipe_runs=wipe_runs)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=TIMEOUT_TRAIN_S,
    cpu=2.0,
    memory=4096,
)
def train_kvasir_seed_wave(
    seed: int = 42,
    archs: Optional[List[str]] = None,
    epochs: int = 50,
    batch_size: int = 0,
    smoke: bool = False,
    resume: str = "",
    auto_batch: bool = True,
    reset: bool = False,
) -> str:
    """
    Orchestrator: optional reset → parallel train_kvasir.map → summarize.

    Running this as a *single* Modal function is required for ``modal run --detach``:
    detach only keeps the last triggered function alive; mapping from the local
    entrypoint would drop child GPUs after disconnect.
    """
    chosen = list(archs) if archs else list(KVASIR_ARCHS)
    lines: List[str] = [
        f"train_kvasir_seed_wave seed={seed} archs={chosen} epochs={epochs} reset={reset}"
    ]

    if reset:
        from modal_runner.jobs.reset import reset_kvasir_seed as _reset

        volume.reload()
        lines.append(
            _reset(seed=seed, archs=chosen, clear_locks=True, wipe_runs=True)
        )
        _commit()

    # Fan out child A100 containers from this remote orchestrator.
    results = list(
        train_kvasir.map(
            chosen,
            kwargs={
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "smoke": smoke,
                "resume": resume,
                "auto_batch": auto_batch,
            },
            order_outputs=True,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    train_lines: List[str] = []
    for arch, res in zip(chosen, results):
        if isinstance(res, BaseException):
            line = f"FAIL  arch={arch} seed={seed}: {res}"
            print(line, flush=True)
            train_lines.append(line)
        else:
            line = f"OK    {res}"
            print(line, flush=True)
            train_lines.append(str(res))
        lines.append(line)

    from modal_runner.jobs.summarize import summarize_kvasir_seed as _summarize

    volume.reload()
    summary = _summarize(seed=seed, archs=chosen, train_results=train_lines)
    _commit()
    lines.append(summary)
    n_fail = sum(1 for r in results if isinstance(r, BaseException))
    if n_fail:
        raise RuntimeError(
            f"{n_fail}/{len(chosen)} arch(s) failed for seed={seed}. "
            f"Resume: modal run --detach -m modal_runner.app -- "
            f"train-kvasir --arch <arch> --seed {seed} --resume auto"
        )
    return "\n".join(lines)


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
    batch_size: int = 0,
    smoke: bool = False,
    resume: str = "",
    auto_batch: bool = True,
) -> str:
    """One A100: train a single Kvasir (arch, seed). Logs/ckpts on the volume."""
    from modal_runner.jobs.train import train_kvasir as _job

    if smoke:
        epochs = min(epochs, 2)
        batch_size = 32 if batch_size <= 0 else min(batch_size, 32)
        auto_batch = False
    msg = _job(
        arch=arch,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        resume=resume,
        auto_batch=auto_batch and not smoke,
    )
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
    seeds: Optional[List[int]] = None,
    epochs: int = 50,
    batch_size: int = 0,
) -> str:
    """Legacy sequential arches×seeds on one GPU. Prefer train-kvasir-seed."""
    from modal_runner.jobs.train import train_kvasir_matrix as _job

    msg = _job(archs=archs, seeds=seeds, epochs=epochs, batch_size=batch_size)
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
    fold: int = 0,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
    smoke: bool = False,
) -> str:
    from modal_runner.jobs.train import train_ibs as _job

    if smoke:
        epochs = min(epochs, 2)
        batch_size = min(batch_size, 32)
    msg = _job(arch=arch, fold=fold, seed=seed, epochs=epochs, batch_size=batch_size)
    _commit()
    return msg


@app.function(
    image=image,
    volumes=_VOLUMES,
    gpu=GPU_TRAIN,
    timeout=TIMEOUT_TRAIN_S,
    memory=65536,
)
def train_ibs_matrix(
    archs: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
) -> str:
    from modal_runner.jobs.train import train_ibs_matrix as _job

    msg = _job(archs=archs, folds=folds, seed=seed, epochs=epochs, batch_size=batch_size)
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
    seed: int = 42,
) -> str:
    from modal_runner.jobs.evaluate import eval_kvasir_classification as _job

    msg = _job(arch=arch, checkpoint=checkpoint, split=split, seed=seed)
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
    fold: int = 0,
) -> str:
    from modal_runner.jobs.evaluate import eval_ibs_classification as _job

    msg = _job(arch=arch, checkpoint=checkpoint, split=split, fold=fold)
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

    di = sub.add_parser(
        "download-ibs",
        help="Legacy: download numeric pre-processed IBS (no exam IDs)",
    )
    di.add_argument("--force", action="store_true")
    di.add_argument("--source", default="kaggle", choices=("kaggle", "zip", "manual"))

    dip = sub.add_parser(
        "download-ibs-patient",
        help="Download franchisn/ibs-dataset → IBS-patient-dataset + ibs_groups.csv",
    )
    dip.add_argument("--force", action="store_true", help="Re-download and rebuild")

    iiz = sub.add_parser(
        "ingest-ibs-zip",
        help="Extract legacy numeric IBS from volume zip (bypass Kaggle 403)",
    )
    iiz.add_argument(
        "--zip-path",
        default="",
        help="Zip path on the volume (default: /vol/data/IBS-preprocessed-dataset.zip)",
    )

    pk = sub.add_parser("prepare-kvasir-splits", help="Write 70/10/20 (+ optional pHash dedupe)")
    pk.add_argument("--seed", type=int, default=42)
    pk.add_argument("--no-dedupe", action="store_true")

    pi = sub.add_parser(
        "prepare-ibs-folds",
        help="Patient-level 5-fold CV (defaults to bundled ibs_groups.csv)",
    )
    pi.add_argument(
        "--groups-csv",
        default="",
        help="Optional override (default: /vol/data/ibs_groups.csv from bundled metadata)",
    )
    pi.add_argument("--n-folds", type=int, default=5)
    pi.add_argument("--seed", type=int, default=42)

    sm = sub.add_parser("smoke-splits", help="Phase 1 split smoke test")
    sm.add_argument("--dedupe", action="store_true")

    tk = sub.add_parser("train-kvasir", help="Train one Kvasir classifier on A100")
    tk.add_argument("--arch", default="resnet50")
    tk.add_argument("--seed", type=int, default=42)
    tk.add_argument("--epochs", type=int, default=50)
    tk.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 + auto-batch (default): probe ~82%% VRAM; or set explicitly",
    )
    tk.add_argument("--no-auto-batch", action="store_true", help="Disable VRAM batch probe")
    tk.add_argument(
        "--resume",
        default="",
        help="Checkpoint path, or auto|latest|mid under the run dir",
    )
    tk.add_argument("--smoke", action="store_true", help="2 epochs, small batch")

    tks = sub.add_parser(
        "train-kvasir-seed",
        help="Parallel: 5 A100s (one per arch) for a single --seed (logout-safe with --detach)",
    )
    tks.add_argument("--seed", type=int, required=True, help="Run this seed only (42, then 43, then 44)")
    tks.add_argument("--epochs", type=int, default=50)
    tks.add_argument("--batch-size", type=int, default=0, help="0 = auto-batch per arch")
    tks.add_argument("--no-auto-batch", action="store_true")
    tks.add_argument("--archs", nargs="+", default=None, help="Default: all 5 matrix arches")
    tks.add_argument(
        "--resume",
        default="",
        help="Pass to every arch (e.g. auto to continue from checkpoint_latest)",
    )
    tks.add_argument("--smoke", action="store_true")
    tks.add_argument(
        "--reset",
        action="store_true",
        help="Wipe seed run dirs + locked_batch_sizes before training (clean relaunch)",
    )

    sks = sub.add_parser(
        "summarize-kvasir-seed",
        help="Print/save cost+metrics table for a finished seed wave (volume)",
    )
    sks.add_argument("--seed", type=int, required=True)
    sks.add_argument("--archs", nargs="+", default=None)

    rks = sub.add_parser(
        "reset-kvasir-seed",
        help="Clear batch locks and wipe /vol/runs/kvasir/{arch}/seed{N}/",
    )
    rks.add_argument("--seed", type=int, required=True)
    rks.add_argument("--archs", nargs="+", default=None)
    rks.add_argument("--keep-runs", action="store_true", help="Only clear locks, keep run dirs")
    rks.add_argument("--keep-locks", action="store_true", help="Only wipe runs, keep locks")

    tkm = sub.add_parser(
        "train-kvasir-matrix",
        help="Legacy sequential arches×seeds on ONE GPU (prefer train-kvasir-seed)",
    )
    tkm.add_argument("--epochs", type=int, default=50)
    tkm.add_argument("--batch-size", type=int, default=0)
    tkm.add_argument("--archs", nargs="+", default=None)
    tkm.add_argument("--seeds", nargs="+", type=int, default=None)

    ti = sub.add_parser("train-ibs", help="Train one IBS patient fold on A100")
    ti.add_argument("--arch", default="resnet50")
    ti.add_argument("--fold", type=int, default=0)
    ti.add_argument("--seed", type=int, default=42)
    ti.add_argument("--epochs", type=int, default=50)
    ti.add_argument("--batch-size", type=int, default=128)
    ti.add_argument("--smoke", action="store_true")

    tim = sub.add_parser("train-ibs-matrix", help="Train IBS arches × folds 0..4")
    tim.add_argument("--epochs", type=int, default=50)
    tim.add_argument("--batch-size", type=int, default=128)
    tim.add_argument("--archs", nargs="+", default=None)
    tim.add_argument("--folds", nargs="+", type=int, default=None)
    tim.add_argument("--seed", type=int, default=42)

    ekc = sub.add_parser("eval-kvasir-cls", help="Classifier metrics on a Kvasir split")
    ekc.add_argument("--arch", default="resnet50")
    ekc.add_argument("--seed", type=int, default=42)
    ekc.add_argument("--split", default="test")
    ekc.add_argument("--checkpoint", default="")

    eic = sub.add_parser("eval-ibs-cls", help="Classifier metrics on an IBS fold split")
    eic.add_argument("--arch", default="resnet50")
    eic.add_argument("--fold", type=int, default=0)
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
    modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42
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
    elif action == "download-ibs-patient":
        print(download_ibs_patient.remote(skip_if_present=not args.force, force=args.force))
    elif action == "download-ibs":
        print(download_ibs.remote(skip_if_present=not args.force, source=args.source))
    elif action == "ingest-ibs-zip":
        print(ingest_ibs_zip.remote(zip_path=args.zip_path))
    elif action == "prepare-kvasir-splits":
        print(prepare_kvasir_splits.remote(dedupe=not args.no_dedupe, seed=args.seed))
    elif action == "prepare-ibs-folds":
        print(
            prepare_ibs_folds.remote(
                groups_csv=args.groups_csv or None,
                n_folds=args.n_folds,
                seed=args.seed,
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
                resume=args.resume,
                auto_batch=not args.no_auto_batch,
            )
        )
    elif action == "train-kvasir-seed":
        # Single remote orchestrator so --detach keeps the whole wave (map+summary).
        archs = list(args.archs) if args.archs else list(KVASIR_ARCHS)
        print(
            f"Dispatching train_kvasir_seed_wave seed={args.seed} archs={archs} "
            f"reset={args.reset} (detach-safe single remote)",
            flush=True,
        )
        print(
            train_kvasir_seed_wave.remote(
                seed=args.seed,
                archs=archs,
                epochs=args.epochs,
                batch_size=args.batch_size,
                smoke=args.smoke,
                resume=args.resume,
                auto_batch=not args.no_auto_batch,
                reset=args.reset,
            )
        )
    elif action == "reset-kvasir-seed":
        print(
            reset_kvasir_seed.remote(
                seed=args.seed,
                archs=args.archs,
                clear_locks=not args.keep_locks,
                wipe_runs=not args.keep_runs,
            )
        )
    elif action == "summarize-kvasir-seed":
        print(
            summarize_kvasir_seed.remote(
                seed=args.seed,
                archs=args.archs,
            )
        )
    elif action == "train-kvasir-matrix":
        print(
            train_kvasir_matrix.remote(
                archs=args.archs,
                seeds=args.seeds,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        )
    elif action == "train-ibs":
        print(
            train_ibs.remote(
                arch=args.arch,
                fold=args.fold,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                smoke=args.smoke,
            )
        )
    elif action == "train-ibs-matrix":
        print(
            train_ibs_matrix.remote(
                archs=args.archs,
                folds=args.folds,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        )
    elif action == "eval-kvasir-cls":
        print(
            eval_kvasir_classification.remote(
                arch=args.arch,
                checkpoint=args.checkpoint or None,
                split=args.split,
                seed=args.seed,
            )
        )
    elif action == "eval-ibs-cls":
        print(
            eval_ibs_classification.remote(
                arch=args.arch,
                checkpoint=args.checkpoint or None,
                split=args.split,
                fold=args.fold,
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
