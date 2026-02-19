"""
Evaluate IBS-trained model: accuracy and F1 (macro/weighted) on validation set.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.ibs.data import (
    IBSDataset,
    get_val_transforms,
    IBS_NUM_CLASSES,
)
from XAI_Enhancer_module.ibs.models import build_ibs_model, load_ibs_checkpoint
from XAI_Enhancer_module.utils.model_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate IBS model: accuracy and F1 on validation set")
    p.add_argument("--data-root", type=str, default="data/IBS-preprocessed-dataset", help="IBS dataset root")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--arch", type=str, default="resnet50",
                    choices=["resnet50", "resnet18", "resnet34", "densenet121", "vgg16", "vgg19"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (e.g. best.pth)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="", help="Optional JSON path to save metrics")
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, device):
    all_preds, all_labels = [], []
    for images, labels, _ in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    data_root = Path(args.data_root)
    splits_dir = data_root / "splits"
    split_file = splits_dir / f"{args.split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run prepare_splits first.")
    val_ds = IBSDataset(str(data_root), split=args.split, transform=get_val_transforms(), splits_dir=str(splits_dir))
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_ibs_model(args.arch, num_classes=IBS_NUM_CLASSES, pretrained=False)
    load_ibs_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    preds, labels = run_eval(model, loader, device)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    metrics = {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
