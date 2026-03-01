#!/usr/bin/env python3
"""
Raw + Softmax Logit-Similarity Extraction (all Conv2d layers, long format).

For every validation image, extracts the pre-softmax raw cosine
similarities (alpha_l) and the post-softmax Enhancer weights for
**every** Conv2d layer in the network using HiResCAMEnhanced.

Runs across 5 architectures (vgg16, vgg19, resnet18, resnet34, resnet50)
and 2 datasets (IBS, Kvasir-v2).

Long-format CSV columns:
    Dataset, Model, Image_ID, Layer_Index, Raw_Similarity, Softmax_Weight

Usage:
    python -m XAI_Enhancer_module.ablation.raw_softmax_extraction \
        --kvasir-data-root data/kvasir-v2 \
        --ibs-data-root data/IBS-preprocessed-dataset \
        --checkpoints kvasir:resnet50:./kvasir_runs/resnet50/best.pth \
                      ibs:vgg16:./ibs_runs/vgg16/best.pth  ... \
        --output-csv runs/ablation/raw_softmax_scores.csv \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from XAI_Enhancer_module.ablation.enhancer_weight_extraction import (
    get_enhancer_raw_and_softmax_all_layers,
    _SplitDataset,
    _get_all_conv2d,
    _load_model,
)
from XAI_Enhancer_module.kvasir.data import (
    load_split_file as kvasir_load_split,
    get_val_transforms as kvasir_val_transforms,
)
from XAI_Enhancer_module.ibs.data import (
    load_split_file as ibs_load_split,
    get_val_transforms as ibs_val_transforms,
)
from XAI_Enhancer_module.utils.model_utils import get_device
from torch.utils.data import DataLoader

MODELS = ["vgg16", "vgg19", "resnet18", "resnet34", "resnet50"]
DATASETS = ["kvasir", "ibs"]
CSV_COLUMNS = ["Dataset", "Model", "Image_ID", "Layer_Index",
               "Raw_Similarity", "Softmax_Weight"]


def _build_dataloader(
    dataset: str, data_root: str, max_images: int, device: torch.device,
) -> DataLoader:
    root = Path(data_root)
    splits_dir = root / "splits"
    if dataset == "kvasir":
        samples = kvasir_load_split(splits_dir / "val.txt", root)
        transform = kvasir_val_transforms()
    else:
        samples = ibs_load_split(splits_dir / "val.txt", root)
        transform = ibs_val_transforms()
    if 0 < max_images < len(samples):
        samples = samples[:max_images]
    return DataLoader(
        _SplitDataset(samples, transform),
        batch_size=1, shuffle=False, num_workers=4,
        pin_memory=(device.type == "cuda"),
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract raw + softmax scores for ALL Conv2d layers (long format).",
    )
    p.add_argument("--kvasir-data-root", default="data/kvasir-v2")
    p.add_argument("--ibs-data-root", default="data/IBS-preprocessed-dataset")
    p.add_argument("--checkpoints", nargs="+", required=True, metavar="DS:ARCH:PATH")
    p.add_argument("--output-csv", default="runs/ablation/raw_softmax_scores.csv")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(get_device(args.device))
    print(f"Device: {device}")

    ckpt_map: dict[tuple[str, str], str] = {}
    for spec in args.checkpoints:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad checkpoint spec '{spec}'")
        ckpt_map[(parts[0].lower(), parts[1].lower())] = parts[2]

    data_roots = {"kvasir": args.kvasir_data_root, "ibs": args.ibs_data_root}

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: reload existing rows and skip completed (dataset, model) pairs
    if output_path.exists():
        existing = pd.read_csv(output_path)
        rows: list[dict] = existing.to_dict("records")
        done = set()
        for r in rows:
            done.add((r["Dataset"], r["Model"]))
        print(f"Resuming: {len(rows)} rows loaded, "
              f"{len(done)} (dataset,model) pairs done")
    else:
        rows = []
        done = set()

    def _save():
        pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(output_path, index=False)

    for ds in DATASETS:
        print(f"\n{'='*60}\nDataset: {ds.upper()}\n{'='*60}")
        loader = _build_dataloader(ds, data_roots[ds], args.max_images, device)

        for arch in MODELS:
            key = (ds, arch)
            if key not in ckpt_map:
                print(f"  [SKIP] No checkpoint for {ds}/{arch}")
                continue
            if key in done:
                print(f"  [CACHED] {ds}/{arch}")
                continue

            print(f"\n  Model: {arch}")
            model = _load_model(ds, arch, ckpt_map[key], device)
            target_layers = _get_all_conv2d(model)
            n_layers = len(target_layers)
            print(f"    All Conv2d layers: {n_layers}")

            for img_id, (img_tensor, label) in enumerate(
                tqdm(loader, desc=f"    {ds}/{arch}", leave=False)
            ):
                img_tensor = img_tensor.squeeze(0).to(device)
                label_int = label.item()

                layer_results = get_enhancer_raw_and_softmax_all_layers(
                    img_tensor, model, target_layers, device, label_int,
                )

                for layer_idx, raw_sim, softmax_w in layer_results:
                    rows.append({
                        "Dataset": ds,
                        "Model": arch,
                        "Image_ID": img_id,
                        "Layer_Index": layer_idx,
                        "Raw_Similarity": round(raw_sim, 6),
                        "Softmax_Weight": round(softmax_w, 6),
                    })

            _save()
            print(f"    [SAVED] incremental results to {output_path}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _save()
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
