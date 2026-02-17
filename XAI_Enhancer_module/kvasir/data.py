"""
Kvasir-v2 dataset: load, split (80:20), and download utilities.
Uses ImageNet mean/std normalization and ImageNet-style resize + crop as agreed.
"""

import os
import random
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# Kvasir-v2 has 8 classes (alphabetical folder names for stable indices)
KVASIR_CLASSES = [
    "dyed-lifted-polyp",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]
KVASIR_NUM_CLASSES = len(KVASIR_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(KVASIR_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(KVASIR_CLASSES)}

# ImageNet normalization (used for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(crop_size: int = 224, resize_size: int = 256):
    """Train transforms: RandomResizedCrop + flip + ImageNet norm."""
    return T.Compose([
        T.Resize(resize_size),
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(crop_size: int = 224, resize_size: int = 256):
    """Val/test transforms: Resize + CenterCrop + ImageNet norm."""
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_image_paths(root: Path, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[Tuple[Path, int]]:
    """Scan root for class folders and return (path, label_index)."""
    out: List[Tuple[Path, int]] = []
    root = Path(root)
    for class_name in KVASIR_CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in extensions:
                out.append((p, CLASS_TO_IDX[class_name]))
    return out


def prepare_splits(
    data_root: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    splits_dir: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Create 80:20 stratified train/val split files under data_root/splits.
    Returns paths to train.txt and val.txt (each line: relative_path label_index).
    """
    data_root = Path(data_root)
    if splits_dir is None:
        splits_dir = data_root / "splits"
    else:
        splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_image_paths(data_root)
    if not pairs:
        raise FileNotFoundError(f"No images found under {data_root} (expected class folders: {KVASIR_CLASSES})")

    # Stratified split per class
    rng = random.Random(seed)
    train_pairs: List[Tuple[Path, int]] = []
    val_pairs: List[Tuple[Path, int]] = []
    for class_idx in range(KVASIR_NUM_CLASSES):
        class_pairs = [(p, l) for p, l in pairs if l == class_idx]
        rng.shuffle(class_pairs)
        n_val = max(1, int(len(class_pairs) * val_ratio))
        val_pairs.extend(class_pairs[:n_val])
        train_pairs.extend(class_pairs[n_val:])

    def rel(p: Path) -> str:
        return str(p.relative_to(data_root))

    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    with open(train_file, "w") as f:
        for p, lbl in train_pairs:
            f.write(f"{rel(p)}\t{lbl}\n")
    with open(val_file, "w") as f:
        for p, lbl in val_pairs:
            f.write(f"{rel(p)}\t{lbl}\n")

    return train_file, val_file


def load_split_file(split_path: Path, data_root: Path) -> List[Tuple[Path, int]]:
    """Load paths and labels from a split file (relative path + label per line)."""
    out = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            rel_path = parts[0]
            label = int(parts[1])
            out.append((data_root / rel_path, label))
    return out


class KvasirDataset(Dataset):
    """Kvasir-v2 dataset using a splits file (train.txt / val.txt)."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[T.Compose] = None,
        splits_dir: Optional[str] = None,
    ):
        """
        Args:
            data_root: Root directory containing class-named folders (or where split paths are relative to).
            split: 'train' or 'val'.
            transform: Applied to PIL image.
            splits_dir: Directory containing train.txt/val.txt; default data_root/splits.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        if splits_dir is None:
            splits_dir = self.data_root / "splits"
        else:
            splits_dir = Path(splits_dir)
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. Run prepare_splits() first."
            )
        self.samples = load_split_file(split_file, self.data_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(path)


def download_kvasir_v2(
    data_root: str = "data",
    url: Optional[str] = None,
) -> Path:
    """
    Download Kvasir-v2 zip from Simula and extract to data_root/kvasir-v2.
    If the dataset is already present (kvasir-v2/<class> folders exist), skip.
    """
    data_root = Path(data_root)
    extract_dir = data_root / "kvasir-v2"
    if extract_dir.exists():
        # Check that we have class folders
        if any((extract_dir / c).is_dir() for c in KVASIR_CLASSES):
            print(f"Kvasir-v2 already present at {extract_dir}, skipping download.")
            return extract_dir

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "kvasir-v2.zip"
    if url is None:
        # Official page: https://datasets.simula.no/kvasir/ — use the "Download Kvasir version 2" link there
        url = "https://datasets.simula.no/kvasir/kvasir-v2.zip"
    try:
        import urllib.request
        print(f"Downloading Kvasir-v2 from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(
            "Automatic download failed (the direct zip URL may not be available).\n"
            "Download Kvasir v2 manually:\n"
            "  1. Open https://datasets.simula.no/kvasir/\n"
            "  2. Use the 'Download Kvasir version 2' link (kvasir-v2.zip, ~2.3GB)\n"
            f"  3. Save the zip as: {zip_path}\n"
            "  4. Run this script again, or run with --skip-download after extracting the zip to:\n"
            f"     {extract_dir}"
        )
        raise SystemExit(1) from e
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir
