"""
Kvasir-v2 training (Phase 2): seed-aware runs, val macro-F1 checkpoints,
volume-backed logs, mid-run resume, and A100-oriented auto batch sizing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.common.train_utils import (
    classification_metrics,
    collect_logits,
    run_dir,
    seed_everything,
    write_args_json,
)
from XAI_Enhancer_module.kvasir.data import (
    KVASIR_CLASSES,
    KVASIR_NUM_CLASSES,
    KvasirDataset,
    get_train_transforms,
    get_val_transforms,
    prepare_splits,
)
from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.utils.model_utils import get_device

# Conservative A100-40GB defaults (224², AMP bf16) used when --auto-batch-size is off
# or as the search start / fallback after OOM.
A100_DEFAULT_BATCH = {
    "resnet18": 512,
    "resnet34": 384,
    "resnet50": 256,
    "densenet121": 192,
    "vgg16": 128,
    "vgg19": 96,
}


class _Tee:
    """Write to multiple streams (stdout + train.log)."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _commit_volume() -> None:
    """Persist /vol writes mid-run so detach / preemption keeps checkpoints + logs."""
    try:
        import modal

        from modal_runner.config import VOLUME_NAME

        modal.Volume.from_name(VOLUME_NAME).commit()
        print("[volume] committed", flush=True)
    except Exception as e:
        print(f"[volume] commit skipped: {e}", flush=True)


def _gpu_mem_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    used = total - free
    return {
        "total_gb": total / (1024**3),
        "used_gb": used / (1024**3),
        "free_gb": free / (1024**3),
        "alloc_gb": alloc / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "used_frac": used / max(total, 1),
    }


def _log_gpu(tag: str) -> None:
    s = _gpu_mem_stats()
    if not s:
        return
    print(
        f"[gpu {tag}] used={s['used_gb']:.2f}/{s['total_gb']:.2f} GiB "
        f"({100 * s['used_frac']:.1f}%)  alloc={s['alloc_gb']:.2f}  "
        f"reserved={s['reserved_gb']:.2f}",
        flush=True,
    )


def find_max_batch_size(
    model: nn.Module,
    device: torch.device,
    *,
    amp: bool,
    amp_dtype: torch.dtype,
    start: int = 32,
    max_bs: int = 1024,
    target_mem_frac: float = 0.82,
) -> int:
    """
    Binary-search the largest power-of-two-ish batch that fits, then pick the
    largest candidate whose post-step memory use is <= target_mem_frac of VRAM.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    candidates: List[int] = []
    bs = start
    while bs <= max_bs:
        candidates.append(bs)
        bs *= 2

    def _try(batch: int) -> Tuple[bool, float]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(batch, 3, 224, 224, device=device)
            y = torch.zeros(batch, dtype=torch.long, device=device)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            if amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss = criterion(model(x), y)
                loss.backward()
            else:
                loss = criterion(model(x), y)
                loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            used_frac = (total - free) / max(total, 1)
            del x, y, loss, optimizer
            torch.cuda.empty_cache()
            return True, float(used_frac)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            return False, 1.0

    lo, hi = start, start
    ok_frac = 0.0
    for c in candidates:
        fits, frac = _try(c)
        print(f"[auto-batch] try bs={c} fits={fits} mem={100 * frac:.1f}%", flush=True)
        if fits:
            lo, hi, ok_frac = c, c, frac
        else:
            break
    # Refine between hi and 2*hi if we never OOM'd at max
    if hi == candidates[-1] and ok_frac < target_mem_frac:
        # probe upward in +25% steps until OOM or target
        probe = hi
        while probe < max_bs:
            nxt = min(max_bs, int(probe * 1.25))
            if nxt <= probe:
                break
            fits, frac = _try(nxt)
            print(f"[auto-batch] try bs={nxt} fits={fits} mem={100 * frac:.1f}%", flush=True)
            if not fits:
                break
            lo, ok_frac = nxt, frac
            probe = nxt
            if frac >= target_mem_frac:
                break

    # If far below target, we already took the max that fits — good.
    # If overshot target slightly, step down until under target.
    chosen = lo
    if ok_frac > target_mem_frac + 0.05:
        for c in reversed([c for c in candidates if c <= lo] or [lo]):
            fits, frac = _try(c)
            if fits and frac <= target_mem_frac:
                chosen = c
                ok_frac = frac
                break

    print(
        f"[auto-batch] selected bs={chosen} (post-step mem ≈ {100 * ok_frac:.1f}% "
        f"target≤{100 * target_mem_frac:.0f}%)",
        flush=True,
    )
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return max(chosen, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train Kvasir-v2 classifier (val macro-F1 checkpoint)")
    p.add_argument("--data-root", type=str, default="data/kvasir-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"],
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = use A100_DEFAULT_BATCH[arch] (or --auto-batch-size)",
    )
    p.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="Probe max batch that fits; target ~82%% GPU memory",
    )
    p.add_argument("--target-mem-frac", type=float, default=0.82)
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
    p.add_argument("--output-dir", type=str, default="runs/kvasir")
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Also save epoch_N.pth every N epochs (0 = only mid + latest + best)",
    )
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint path, or 'auto' / 'latest' / 'mid' under the run dir",
    )
    p.add_argument("--a100", action="store_true", help="AMP bf16 + compile + workers=8")
    p.add_argument(
        "--commit-every-epochs",
        type=int,
        default=5,
        help="Commit Modal volume every N epochs (keeps logs/ckpts if job dies)",
    )
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
    splits_dir = data_root / "splits"
    if not (splits_dir / "train.txt").exists() or not (splits_dir / "test.txt").exists():
        print("Creating 70/10/20 train/val/test split...")
        prepare_splits(str(data_root), seed=args.seed)
    train_ds = KvasirDataset(str(data_root), split="train", transform=train_tf, splits_dir=str(splits_dir))
    val_ds = KvasirDataset(str(data_root), split="val", transform=val_tf, splits_dir=str(splits_dir))
    kw = dict(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device, scaler, args, epoch):
    model.train()
    amp_dtype = getattr(torch, args.amp_dtype, torch.float16)
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for i, (images, labels, _) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if args.amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss = criterion(model(images), labels) / args.grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            loss = criterion(model(images), labels) / args.grad_accum_steps
            loss.backward()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        if (i + 1) % args.grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if (i + 1) % args.log_interval == 0:
            pbar.set_postfix(loss=total_loss / max(n, 1))
    return total_loss / n if n else 0.0


def _resolve_resume(path_arg: str, out_dir: Path) -> Optional[Path]:
    if not path_arg:
        return None
    key = path_arg.strip().lower()
    if key in {"auto", "latest"}:
        for name in ("checkpoint_latest.pth", "checkpoint_mid.pth", "best.pth", "last.pth"):
            p = out_dir / name
            if p.exists():
                return p
        return None
    if key == "mid":
        p = out_dir / "checkpoint_mid.pth"
        return p if p.exists() else None
    p = Path(path_arg)
    return p if p.exists() else None


def _save_trainer_ckpt(
    path: Path,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_f1: float,
    metrics_log: List[Dict[str, Any]],
    args,
    batch_size: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_f1": best_f1,
        "metrics_log": metrics_log,
        "arch": args.arch,
        "seed": args.seed,
        "batch_size": batch_size,
        "val_f1_macro": best_f1,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


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

    out_dir = run_dir(args.output_dir, args.arch, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_f = log_path.open("a", buffering=1)
    log_f.write(f"\n===== start {datetime.now(timezone.utc).isoformat()} =====\n")
    sys.stdout = _Tee(sys.__stdout__, log_f)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, log_f)  # type: ignore[assignment]
    print(f"Logging to {log_path}", flush=True)

    model = build_kvasir_model(args.arch, num_classes=KVASIR_NUM_CLASSES, pretrained=True)
    model = model.to(device)

    amp_dtype = getattr(torch, args.amp_dtype, torch.bfloat16)
    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = A100_DEFAULT_BATCH.get(args.arch, 128)

    if args.auto_batch_size and device.type == "cuda":
        print(f"[auto-batch] probing for arch={args.arch} target_mem={args.target_mem_frac}", flush=True)
        batch_size = find_max_batch_size(
            model,
            device,
            amp=args.amp,
            amp_dtype=amp_dtype,
            start=max(32, A100_DEFAULT_BATCH.get(args.arch, 128) // 4),
            max_bs=1024,
            target_mem_frac=args.target_mem_frac,
        )
    args.batch_size = batch_size
    print(f"Using batch_size={batch_size}", flush=True)

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
        load_kvasir_checkpoint(model, str(resume_path), device)
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", ckpt.get("val_f1_macro", -1.0)))
        metrics_log = list(ckpt.get("metrics_log") or [])
        print(f"Resume: next_epoch={start_epoch} best_f1={best_f1:.4f}", flush=True)

    # Compile AFTER resume load so state_dict keys stay clean on disk
    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile(mode='reduce-overhead')", flush=True)
        except Exception as e:
            print(f"torch.compile skipped: {e}", flush=True)

    write_args_json(
        out_dir,
        args,
        extra={
            "dataset": "kvasir",
            "checkpoint_metric": "val_f1_macro",
            "batch_size_resolved": batch_size,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _commit_volume()
    _log_gpu("init")

    mid_epoch = max(1, args.epochs // 2)
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args, epoch)
        y, pred, prob = collect_logits(model, val_loader, device, use_amp=args.amp, amp_dtype=amp_dtype)
        val_m = classification_metrics(
            y, pred, prob, class_names=KVASIR_CLASSES, num_classes=KVASIR_NUM_CLASSES
        )
        if scheduler:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_m["accuracy"],
            "val_f1_macro": val_m["f1_macro"],
            "val_auroc": val_m.get("auroc"),
            "elapsed_s": time.time() - t0,
        }
        mem = _gpu_mem_stats()
        if mem:
            row["gpu_used_frac"] = mem["used_frac"]
        metrics_log.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_acc={val_m['accuracy']:.4f}  val_f1_macro={val_m['f1_macro']:.4f}",
            flush=True,
        )
        _log_gpu(f"epoch{epoch}")

        # Always refresh latest (full trainer state → resume)
        _save_trainer_ckpt(
            out_dir / "checkpoint_latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_f1=best_f1 if best_f1 > val_m["f1_macro"] else val_m["f1_macro"],
            metrics_log=metrics_log,
            args=args,
            batch_size=batch_size,
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
                extra={"mid": True},
            )
            print(f"Saved mid-run checkpoint at epoch {epoch} → {out_dir / 'checkpoint_mid.pth'}", flush=True)
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
                    "seed": args.seed,
                    "batch_size": batch_size,
                },
                out_dir / "best.pth",
            )

        if args.save_every and epoch % args.save_every == 0:
            torch.save(
                {"model_state_dict": _unwrap(model).state_dict(), "epoch": epoch},
                out_dir / f"epoch_{epoch}.pth",
            )

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

        if args.commit_every_epochs and epoch % args.commit_every_epochs == 0:
            _commit_volume()

    torch.save(
        {
            "model_state_dict": _unwrap(model).state_dict(),
            "epoch": args.epochs,
            "arch": args.arch,
            "seed": args.seed,
        },
        out_dir / "last.pth",
    )
    # Final full state snapshot
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
        extra={"finished": True},
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Best val_f1_macro: {best_f1:.4f}  Checkpoints in {out_dir}", flush=True)
    _log_gpu("final")
    _commit_volume()


if __name__ == "__main__":
    main()
