"""
Kvasir-v2 training script with 80:20 train/val, optimized batching, and standard args.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.kvasir.data import (
    KvasirDataset,
    get_train_transforms,
    get_val_transforms,
    prepare_splits,
    KVASIR_NUM_CLASSES,
)
from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.utils.model_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Train Kvasir-v2 classifier")
    p.add_argument("--data-root", type=str, default="data/kvasir-v2", help="Kvasir-v2 root (or parent of kvasir-v2)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation fraction (80:20 default)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    p.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true", default=True, help="Pin memory for DataLoader (disable for MPS if you see errors)")
    p.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory (use for MPS/CPU)")
    p.add_argument("--persistent-workers", action="store_true", default=True)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--amp", action="store_true", default=False, help="Use automatic mixed precision (recommended for A100)")
    p.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="AMP dtype (bfloat16 often better on A100)")
    p.add_argument("--compile", action="store_true", default=False, help="Use torch.compile(model) for speed (PyTorch 2+, CUDA)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="runs/kvasir")
    p.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N epochs (0 = only best/last)")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    p.add_argument("--a100", action="store_true", help="Preset for A100 40GB: batch_size=128, amp, bfloat16, compile, num_workers=8")
    args = p.parse_args()
    if getattr(args, "a100", False):
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
    splits_dir = data_root / "splits"
    if not (splits_dir / "train.txt").exists():
        print("Creating 80:20 train/val split...")
        prepare_splits(str(data_root), val_ratio=args.val_ratio, seed=args.seed)
    train_ds = KvasirDataset(str(data_root), split="train", transform=train_tf, splits_dir=str(splits_dir))
    val_ds = KvasirDataset(str(data_root), split="val", transform=val_tf, splits_dir=str(splits_dir))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device, scaler, args, epoch):
    model.train()
    amp_dtype = getattr(torch, getattr(args, "amp_dtype", "float16"), torch.float16)
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    optimizer.zero_grad()
    for i, (images, labels, _) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        if args.amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels) / args.grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            logits = model(images)
            loss = criterion(logits, labels) / args.grad_accum_steps
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
            pbar.set_postfix(loss=total_loss / n)
    return total_loss / n if n else 0.0


@torch.no_grad()
def validate(model, loader, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    correct, total = 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(images)
        else:
            logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def main():
    args = parse_args()
    if getattr(args, "no_pin_memory", False):
        args.pin_memory = False
    torch.manual_seed(args.seed)
    device = torch.device(get_device(args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}. Run download and prepare_splits first.")
    train_loader, val_loader = create_loaders(args, data_root)

    model = build_kvasir_model(args.arch, num_classes=KVASIR_NUM_CLASSES, pretrained=True)
    if args.resume:
        load_kvasir_checkpoint(model, args.resume, device)
    model = model.to(device)
    if getattr(args, "compile", False) and device.type == "cuda":
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
    amp_dtype = getattr(torch, getattr(args, "amp_dtype", "float16"), torch.float16)
    use_scaler = args.amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    out_dir = Path(args.output_dir) / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    metrics_log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args, epoch)
        val_acc = validate(model, val_loader, device, use_amp=args.amp, amp_dtype=amp_dtype)
        if scheduler:
            scheduler.step()
        metrics_log.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"Epoch {epoch}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, out_dir / "best.pth")
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, out_dir / f"epoch_{epoch}.pth")
    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs}, out_dir / "last.pth")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Best val_acc: {best_acc:.4f}  Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
