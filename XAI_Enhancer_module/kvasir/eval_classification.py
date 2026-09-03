"""
Evaluate a Kvasir classifier: accuracy, F1, AUROC, ECE on a chosen split (default test).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.common.train_utils import classification_metrics, collect_logits
from XAI_Enhancer_module.kvasir.data import (
    KVASIR_CLASSES,
    KVASIR_NUM_CLASSES,
    KvasirDataset,
    get_val_transforms,
)
from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.utils.model_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Kvasir model on a split")
    p.add_argument("--data-root", type=str, default="data/kvasir-v2")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--arch", type=str, default="resnet50",
                   choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="", help="JSON path for metrics")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    data_root = Path(args.data_root)
    splits_dir = data_root / "splits"
    if not (splits_dir / f"{args.split}.txt").exists():
        raise FileNotFoundError(f"Missing {splits_dir / args.split}.txt")

    ds = KvasirDataset(
        str(data_root), split=args.split, transform=get_val_transforms(), splits_dir=str(splits_dir),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_kvasir_model(args.arch, num_classes=KVASIR_NUM_CLASSES, pretrained=False)
    load_kvasir_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    y, pred, prob = collect_logits(model, loader, device)
    metrics = classification_metrics(
        y, pred, prob, class_names=KVASIR_CLASSES, num_classes=KVASIR_NUM_CLASSES,
    )
    metrics.update({"arch": args.arch, "split": args.split, "checkpoint": args.checkpoint})

    print(f"Split:       {args.split} (n={metrics['n_samples']})")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"AUROC:       {metrics.get('auroc')}")
    print(f"ECE:         {metrics['ece']:.4f}")

    out = Path(args.output) if args.output else Path(args.checkpoint).with_name(f"eval_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
