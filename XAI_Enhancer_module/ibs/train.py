"""
IBS training (Phase 2): patient folds, val macro-F1 ckpts, A100 auto-batch,
volume logs, mid-run resume — same machinery as Kvasir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.common.train_utils import (
    classification_metrics,
    collect_logits,
    run_dir,
    seed_everything,
)
from XAI_Enhancer_module.ibs.data import (
    IBS_CLASSES,
    IBS_NUM_CLASSES,
    IBSDataset,
    get_train_transforms,
    get_val_transforms,
)
from XAI_Enhancer_module.ibs.models import build_ibs_model, load_ibs_checkpoint
from XAI_Enhancer_module.kvasir.train import (
    A100_DEFAULT_BATCH,
    _Tee,
    _commit_volume,
    _gpu_mem_stats,
    _load_locked_batch,
    _locked_batch_file,
    _log_gpu,
    _resolve_resume,
    _save_trainer_ckpt,
    _unwrap,
    _update_locked_batch_file,
    _write_args,
    find_max_batch_size,
    train_epoch,
)
from XAI_Enhancer_module.utils.model_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Train IBS classifier on patient folds (val macro-F1)")
    p.add_argument("--data-root", type=str, default="data/IBS-patient-dataset")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"],
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--auto-batch-size", action="store_true")
    p.add_argument("--force-auto-batch", action="store_true")
    p.add_argument("--locked-batch-file", type=str, default="")
    p.add_argument("--target-mem-frac", type=float, default=0.82)
    p.add_argument("--min-steps-per-epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true", default=True)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="runs/ibs")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--a100", action="store_true")
    p.add_argument("--commit-every-epochs", type=int, default=5)
    args = p.parse_args()
    if args.a100:
        args.amp = True
        args.amp_dtype = "bfloat16"
        args.compile = True
        args.num_workers = 8
        args.prefetch_factor = 4
    return args


def create_loaders(args, data_root: Path, batch_size: int):
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    fold_dir = data_root / "splits" / f"fold{args.fold}"
    if not (fold_dir / "train.txt").exists():
        raise FileNotFoundError(
            f"Missing {fold_dir / 'train.txt'}. Run prepare-ibs-folds first."
        )
    train_ds = IBSDataset(str(data_root), split="train", transform=train_tf, fold=args.fold)
    val_ds = IBSDataset(str(data_root), split="val", transform=val_tf, fold=args.fold)
    kw = dict(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader


def main():
    args = parse_args()
    if args.no_pin_memory:
        args.pin_memory = False
    seed_everything(args.seed)
    device = torch.device(get_device(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    out_dir = run_dir(args.output_dir, args.arch, fold=args.fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_f = log_path.open("a", buffering=1)
    log_f.write(f"\n===== start {datetime.now(timezone.utc).isoformat()} =====\n")
    sys.stdout = _Tee(sys.__stdout__, log_f)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, log_f)  # type: ignore[assignment]
    print(f"Logging to {log_path}", flush=True)

    model = build_ibs_model(args.arch, num_classes=IBS_NUM_CLASSES, pretrained=True)
    model = model.to(device)

    amp_dtype = getattr(torch, args.amp_dtype, torch.bfloat16)
    lock_path = _locked_batch_file(args.output_dir, args.locked_batch_file)
    args.locked_batch_file = str(lock_path)

    fold_train = data_root / "splits" / f"fold{args.fold}" / "train.txt"
    n_train = sum(1 for _ in fold_train.open()) if fold_train.exists() else 0
    max_bs_by_steps = max(16, n_train // max(1, args.min_steps_per_epoch)) if n_train else 128
    print(
        f"[batch] fold={args.fold} n_train={n_train} min_steps/epoch={args.min_steps_per_epoch} "
        f"→ max_bs_by_steps={max_bs_by_steps}",
        flush=True,
    )

    batch_size = int(args.batch_size)
    batch_size_source = "cli"
    if batch_size > 0:
        if batch_size > max_bs_by_steps:
            batch_size = max_bs_by_steps
            batch_size_source = "cli-capped"
    else:
        locked = None if args.force_auto_batch else _load_locked_batch(lock_path, args.arch)
        if locked is not None:
            batch_size = min(int(locked), max_bs_by_steps)
            batch_size_source = "locked" if batch_size == int(locked) else "locked-capped"
            print(f"[batch] using locked {args.arch}={locked} from {lock_path}", flush=True)
        elif args.auto_batch_size and device.type == "cuda":
            print(
                f"[auto-batch] probing arch={args.arch} max_bs={max_bs_by_steps}",
                flush=True,
            )
            batch_size = find_max_batch_size(
                model,
                device,
                amp=args.amp,
                amp_dtype=amp_dtype,
                start=max(16, min(32, max_bs_by_steps // 4)),
                max_bs=max_bs_by_steps,
                target_mem_frac=args.target_mem_frac,
                weight_decay=args.weight_decay,
            )
            batch_size_source = "auto-probe"
        else:
            batch_size = min(A100_DEFAULT_BATCH.get(args.arch, 128), max_bs_by_steps)
            batch_size_source = "default"

    args.batch_size = batch_size
    print(f"Using batch_size={batch_size} (source={batch_size_source})", flush=True)

    _write_args(
        out_dir,
        args,
        batch_size=batch_size,
        batch_size_source=batch_size_source,
        batch_size_locked=(batch_size_source in {"locked", "locked-capped"}),
        extra={
            "dataset": "ibs",
            "fold": args.fold,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "n_train": n_train,
            "max_bs_by_steps": max_bs_by_steps,
        },
    )

    train_loader, val_loader = create_loaders(args, data_root, batch_size)
    print(f"train steps/epoch≈{len(train_loader)}  val batches={len(val_loader)}", flush=True)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    else:
        scheduler = None

    use_scaler = args.amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    start_epoch = 1
    best_f1 = -1.0
    metrics_log: List[Dict[str, Any]] = []

    resume_path = _resolve_resume(args.resume, out_dir)
    if resume_path is not None:
        print(f"Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        load_ibs_checkpoint(model, str(resume_path), device)
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", ckpt.get("val_f1_macro", -1.0)))
        metrics_log = list(ckpt.get("metrics_log") or [])
        print(f"Resume: next_epoch={start_epoch} best_f1={best_f1:.4f}", flush=True)

    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile(mode='reduce-overhead')", flush=True)
        except Exception as e:
            print(f"torch.compile skipped: {e}", flush=True)

    _commit_volume()
    _log_gpu("init")
    mid_epoch = max(1, args.epochs // 2)
    t0 = time.time()
    batch_locked = batch_size_source in {"locked", "locked-capped"}

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args, epoch)
        y, pred, prob = collect_logits(model, val_loader, device, use_amp=args.amp, amp_dtype=amp_dtype)
        val_m = classification_metrics(y, pred, prob, class_names=IBS_CLASSES, num_classes=IBS_NUM_CLASSES)
        if scheduler:
            scheduler.step()
        row = {
            "epoch": epoch,
            "fold": args.fold,
            "train_loss": train_loss,
            "val_acc": val_m["accuracy"],
            "val_f1_macro": val_m["f1_macro"],
            "val_auroc": val_m.get("auroc"),
            "elapsed_s": time.time() - t0,
            "batch_size": batch_size,
        }
        mem = _gpu_mem_stats()
        if mem:
            row["gpu_used_frac"] = mem["used_frac"]
        metrics_log.append(row)
        print(
            f"Fold {args.fold} Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_acc={val_m['accuracy']:.4f}  val_f1_macro={val_m['f1_macro']:.4f}",
            flush=True,
        )
        _log_gpu(f"epoch{epoch}")

        if epoch == 1 or (not batch_locked and epoch == start_epoch):
            batch_locked = True
            _write_args(
                out_dir,
                args,
                batch_size=batch_size,
                batch_size_source=batch_size_source,
                batch_size_locked=True,
                extra={
                    "dataset": "ibs",
                    "fold": args.fold,
                    "locked_after_epoch": epoch,
                    "locked_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_used_frac_at_lock": mem.get("used_frac") if mem else None,
                },
            )
            _update_locked_batch_file(
                lock_path,
                arch=args.arch,
                batch_size=batch_size,
                seed=args.seed,
                source=f"fold{args.fold}-epoch{epoch}-confirmed",
            )
            print(f"[batch] locked arch={args.arch} batch_size={batch_size}", flush=True)
            _commit_volume()

        _save_trainer_ckpt(
            out_dir / "checkpoint_latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_f1=max(best_f1, val_m["f1_macro"]),
            metrics_log=metrics_log,
            args=args,
            batch_size=batch_size,
            extra={"fold": args.fold},
        )
        if epoch == mid_epoch:
            _save_trainer_ckpt(
                out_dir / "checkpoint_mid.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_f1=max(best_f1, val_m["f1_macro"]),
                metrics_log=metrics_log,
                args=args,
                batch_size=batch_size,
                extra={"mid": True, "fold": args.fold},
            )
            print(f"Saved mid-run checkpoint at epoch {epoch}", flush=True)
            _commit_volume()

        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            torch.save(
                {
                    "model_state_dict": _unwrap(model).state_dict(),
                    "epoch": epoch,
                    "val_f1_macro": best_f1,
                    "val_acc": val_m["accuracy"],
                    "arch": args.arch,
                    "fold": args.fold,
                    "seed": args.seed,
                    "batch_size": batch_size,
                },
                out_dir / "best.pth",
            )
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model_state_dict": _unwrap(model).state_dict(), "epoch": epoch}, out_dir / f"epoch_{epoch}.pth")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)
        if args.commit_every_epochs and epoch % args.commit_every_epochs == 0:
            _commit_volume()

    torch.save(
        {
            "model_state_dict": _unwrap(model).state_dict(),
            "epoch": args.epochs,
            "arch": args.arch,
            "fold": args.fold,
            "seed": args.seed,
        },
        out_dir / "last.pth",
    )
    _save_trainer_ckpt(
        out_dir / "checkpoint_latest.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=args.epochs,
        best_f1=best_f1,
        metrics_log=metrics_log,
        args=args,
        batch_size=batch_size,
        extra={"finished": True, "fold": args.fold},
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Best val_f1_macro: {best_f1:.4f}  Checkpoints in {out_dir}", flush=True)
    _log_gpu("final")
    _commit_volume()


if __name__ == "__main__":
    main()
