"""
IBS training (Phase 2): patient-level folds, best ckpt by val macro-F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from XAI_Enhancer_module.ibs.data import (
    IBS_CLASSES,
    IBS_NUM_CLASSES,
    IBSDataset,
    get_train_transforms,
    get_val_transforms,
)
from XAI_Enhancer_module.ibs.models import build_ibs_model, load_ibs_checkpoint
from XAI_Enhancer_module.utils.model_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Train IBS classifier on patient folds (val macro-F1)")
    p.add_argument(
        "--data-root",
        type=str,
        default="data/IBS-patient-dataset",
        help="IBS root with IBS/ + Normal/ (patient-aware tree)",
    )
    p.add_argument("--fold", type=int, required=True, help="Patient fold index (0..n_folds-1)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (fold splits are fixed separately)")
    p.add_argument("--arch", type=str, default="resnet50",
                   choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true", default=True)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="runs/ibs")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--a100", action="store_true")
    args = p.parse_args()
    if args.a100:
        args.batch_size = 128
        args.amp = True
        args.amp_dtype = "bfloat16"
        args.compile = True
        args.num_workers = 8
        args.prefetch_factor = 4
    return args


def create_loaders(args, data_root: Path):
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    fold_dir = data_root / "splits" / f"fold{args.fold}"
    if not (fold_dir / "train.txt").exists():
        raise FileNotFoundError(
            f"Missing {fold_dir / 'train.txt'}. Run prepare-ibs-folds / "
            "download_and_prepare first."
        )
    train_ds = IBSDataset(
        str(data_root), split="train", transform=train_tf, fold=args.fold,
    )
    val_ds = IBSDataset(
        str(data_root), split="val", transform=val_tf, fold=args.fold,
    )
    kw = dict(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device, scaler, args, epoch):
    model.train()
    amp_dtype = getattr(torch, args.amp_dtype, torch.float16)
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Fold{args.fold} Epoch {epoch}", leave=False)
    optimizer.zero_grad()
    for i, (images, labels, _) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
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
            optimizer.zero_grad()
        if (i + 1) % args.log_interval == 0:
            pbar.set_postfix(loss=total_loss / max(n, 1))
    return total_loss / n if n else 0.0


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
    train_loader, val_loader = create_loaders(args, data_root)

    model = build_ibs_model(args.arch, num_classes=IBS_NUM_CLASSES, pretrained=True)
    if args.resume:
        load_ibs_checkpoint(model, args.resume, device)
    model = model.to(device)
    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile(mode='reduce-overhead')")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

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

    amp_dtype = getattr(torch, args.amp_dtype, torch.float16)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda" and amp_dtype == torch.float16) else None

    out_dir = run_dir(args.output_dir, args.arch, fold=args.fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_args_json(
        out_dir,
        args,
        extra={"dataset": "ibs", "checkpoint_metric": "val_f1_macro", "fold": args.fold},
    )

    best_f1 = -1.0
    metrics_log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args, epoch)
        y, pred, prob = collect_logits(model, val_loader, device, use_amp=args.amp, amp_dtype=amp_dtype)
        val_m = classification_metrics(y, pred, prob, class_names=IBS_CLASSES, num_classes=IBS_NUM_CLASSES)
        if scheduler:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_m["accuracy"],
            "val_f1_macro": val_m["f1_macro"],
            "val_auroc": val_m.get("auroc"),
        }
        metrics_log.append(row)
        print(
            f"Fold {args.fold} Epoch {epoch}  train_loss={train_loss:.4f}  "
            f"val_acc={val_m['accuracy']:.4f}  val_f1_macro={val_m['f1_macro']:.4f}"
        )
        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1_macro": best_f1,
                    "val_acc": val_m["accuracy"],
                    "arch": args.arch,
                    "fold": args.fold,
                    "seed": args.seed,
                },
                out_dir / "best.pth",
            )
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, out_dir / f"epoch_{epoch}.pth")

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs}, out_dir / "last.pth")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Best val_f1_macro: {best_f1:.4f}  Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
