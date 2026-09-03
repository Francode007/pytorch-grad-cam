"""
Evaluate an IBS classifier: accuracy, F1, AUROC, ECE on a fold split (default test).
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
from XAI_Enhancer_module.ibs.data import (
    IBS_CLASSES,
    IBS_NUM_CLASSES,
    IBSDataset,
    get_val_transforms,
)
from XAI_Enhancer_module.ibs.models import build_ibs_model, load_ibs_checkpoint
from XAI_Enhancer_module.utils.model_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate IBS model on a patient-fold split")
    p.add_argument("--data-root", type=str, default="data/IBS-patient-dataset")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--arch", type=str, default="resnet50",
                   choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    data_root = Path(args.data_root)
    fold_dir = data_root / "splits" / f"fold{args.fold}"
    if not (fold_dir / f"{args.split}.txt").exists():
        raise FileNotFoundError(f"Missing {fold_dir / args.split}.txt")

    ds = IBSDataset(
        str(data_root), split=args.split, transform=get_val_transforms(), fold=args.fold,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_ibs_model(args.arch, num_classes=IBS_NUM_CLASSES, pretrained=False)
    load_ibs_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    y, pred, prob = collect_logits(model, loader, device)
    metrics = classification_metrics(
        y, pred, prob, class_names=IBS_CLASSES, num_classes=IBS_NUM_CLASSES,
    )
    metrics.update({
        "arch": args.arch,
        "fold": args.fold,
        "split": args.split,
        "checkpoint": args.checkpoint,
    })

    print(f"Fold/Split:  {args.fold}/{args.split} (n={metrics['n_samples']})")
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
