"""
IBS (Pre-processed) dataset: load, split (80:20), and download utilities.
Uses IBS dataset-specific mean/std normalization with ImageNet-style resize + crop.
"""

import os
import random
import threading
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# IBS pre-processed dataset has 2 classes (alphabetical folder names for stable indices)
IBS_CLASSES = [
    "IBS",
    "Normal",
]
IBS_NUM_CLASSES = len(IBS_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(IBS_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(IBS_CLASSES)}

# IBS dataset-specific normalization (computed from training images)
IBS_MEAN = [0.6380, 0.3422, 0.2275]
IBS_STD = [0.2448, 0.2060, 0.1710]


def get_train_transforms(crop_size: int = 224, resize_size: int = 256):
    """Train transforms: Resize + RandomCrop + flip + ImageNet norm."""
    return T.Compose([
        T.Resize(resize_size),
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IBS_MEAN, std=IBS_STD),
    ])


def get_val_transforms(crop_size: int = 224, resize_size: int = 256):
    """Val/test transforms: Resize + CenterCrop + ImageNet norm."""
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IBS_MEAN, std=IBS_STD),
    ])


def _find_image_paths(root: Path, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[Tuple[Path, int]]:
    """Scan root for class folders and return (path, label_index)."""
    out: List[Tuple[Path, int]] = []
    root = Path(root)
    for cls_name in IBS_CLASSES:
        class_dir = root / cls_name
        if not class_dir.is_dir():
            continue
        label = CLASS_TO_IDX[cls_name]
        search_dirs = [class_dir]
        direct_files = [p for p in class_dir.iterdir() if p.suffix.lower() in extensions]
        if not direct_files and class_dir.is_dir():
            subdirs = [p for p in class_dir.iterdir() if p.is_dir()]
            if subdirs:
                search_dirs = subdirs
        for search_dir in search_dirs:
            for p in search_dir.iterdir():
                if p.suffix.lower() in extensions:
                    out.append((p, label))
    return out


def prepare_splits(
    data_root: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    splits_dir: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Create 80:20 stratified train/val split files under data_root/splits.
    Returns paths to train.txt and val.txt (each line: relative_path\tlabel_index).
    """
    data_root = Path(data_root)
    if splits_dir is None:
        splits_dir = data_root / "splits"
    else:
        splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_image_paths(data_root)
    if not pairs:
        raise FileNotFoundError(f"No images found under {data_root} (expected class folders: {IBS_CLASSES})")

    rng = random.Random(seed)
    train_pairs: List[Tuple[Path, int]] = []
    val_pairs: List[Tuple[Path, int]] = []
    for class_idx in range(IBS_NUM_CLASSES):
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


class IBSDataset(Dataset):
    """IBS pre-processed dataset using a splits file (train.txt / val.txt)."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[T.Compose] = None,
        splits_dir: Optional[str] = None,
    ):
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


# Kaggle dataset: https://www.kaggle.com/datasets/franchisn/pre-processed-ibs
IBS_KAGGLE_SLUG = "franchisn/pre-processed-ibs"


def _ensure_ibs_structure(data_root: Path, download_dir: Path) -> Path:
    """
    After Kaggle unzip, the zip may have one top-level folder or class folders at root.
    Move contents so that data_root/IBS-preprocessed-dataset contains the 2 class subdirs.
    """
    import shutil
    extract_dir = data_root / "IBS-preprocessed-dataset"
    extract_dir.mkdir(parents=True, exist_ok=True)
    subdirs = [p for p in download_dir.iterdir() if p.is_dir()]
    if any((download_dir / c).is_dir() for c in IBS_CLASSES):
        for p in download_dir.iterdir():
            dest = extract_dir / p.name
            if not dest.exists():
                shutil.move(str(p), str(dest))
        return extract_dir
    for d in subdirs:
        if any((d / c).is_dir() for c in IBS_CLASSES):
            for p in d.iterdir():
                dest = extract_dir / p.name
                if not dest.exists():
                    shutil.move(str(p), str(dest))
            try:
                d.rmdir()
            except OSError:
                pass
            return extract_dir
    raise FileNotFoundError(
        f"Could not find IBS class folders under {download_dir}. "
        f"Expected one of: {IBS_CLASSES}"
    )


def download_ibs_kaggle(data_root: str = "data") -> Path:
    """
    Download IBS pre-processed dataset from Kaggle and extract to data_root/IBS-preprocessed-dataset.
    Requires Kaggle API credentials: KAGGLE_USERNAME and KAGGLE_KEY env vars,
    or ~/.kaggle/kaggle.json.
    """
    data_root = Path(data_root)
    extract_dir = data_root / "IBS-preprocessed-dataset"
    if extract_dir.exists() and any((extract_dir / c).is_dir() for c in IBS_CLASSES):
        print(f"IBS dataset already present at {extract_dir}, skipping download.")
        return extract_dir

    data_root.mkdir(parents=True, exist_ok=True)
    download_dir = data_root / "_ibs_kaggle_dl"
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        download_error = [None]

        def _do_download():
            try:
                api.dataset_download_files(
                    IBS_KAGGLE_SLUG,
                    path=str(download_dir),
                    unzip=True,
                )
            except Exception as e:
                download_error[0] = e

        thread = threading.Thread(target=_do_download, daemon=False)
        thread.start()
        with tqdm(
            desc="Downloading IBS dataset from Kaggle",
            total=None,
            unit="",
            dynamic_ncols=True,
            bar_format="{desc}: {elapsed} elapsed",
            mininterval=1.0,
            file=__import__("sys").stdout,
        ) as pbar:
            while thread.is_alive():
                pbar.update(0)
                time.sleep(0.5)
            thread.join()
        if download_error[0]:
            raise download_error[0]
        with tqdm(desc="Organising dataset", total=1, bar_format="{desc}...") as pbar:
            result = _ensure_ibs_structure(data_root, download_dir)
            pbar.update(1)
        return result
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print(
            "Set up Kaggle API credentials:\n"
            "  1. Go to https://www.kaggle.com/settings -> Create New Token (downloads kaggle.json)\n"
            "  2. On server: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
            "  Or set env: KAGGLE_USERNAME=your_user KAGGLE_KEY=your_key"
        )
        raise SystemExit(1) from e
    finally:
        if download_dir.exists():
            import shutil
            try:
                shutil.rmtree(download_dir)
            except OSError:
                pass


def download_ibs_from_zip(data_root: str = "data", zip_path: Optional[str] = None) -> Path:
    """Extract IBS dataset from a local zip file."""
    data_root = Path(data_root)
    extract_dir = data_root / "IBS-preprocessed-dataset"
    if extract_dir.exists() and any((extract_dir / c).is_dir() for c in IBS_CLASSES):
        print(f"IBS dataset already present at {extract_dir}, skipping extraction.")
        return extract_dir

    if zip_path is None:
        zip_path = data_root / "IBS-preprocessed-dataset.zip"
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    print(f"Extracting {zip_path} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def download_ibs(
    data_root: str = "data",
    source: str = "kaggle",
) -> Path:
    """
    Download/extract IBS dataset to data_root/IBS-preprocessed-dataset.
    If the dataset is already present (class folders exist), skip.

    source: "kaggle" (default, uses Kaggle API),
            "zip" (extract from local zip at data_root/IBS-preprocessed-dataset.zip),
            "manual" (print instructions only).
    """
    data_root = Path(data_root)
    extract_dir = data_root / "IBS-preprocessed-dataset"
    if extract_dir.exists() and any((extract_dir / c).is_dir() for c in IBS_CLASSES):
        print(f"IBS dataset already present at {extract_dir}, skipping download.")
        return extract_dir

    if source == "kaggle":
        return download_ibs_kaggle(data_root=str(data_root))

    if source == "zip":
        return download_ibs_from_zip(data_root=str(data_root))

    if source == "manual":
        print(
            "Manual download options:\n"
            "  • Kaggle: https://www.kaggle.com/datasets/franchisn/pre-processed-ibs\n"
            f"Extract so that class folders appear under: {extract_dir}"
        )
        raise SystemExit(1)

    raise ValueError(f"Unknown source: {source}. Use 'kaggle', 'zip', or 'manual'.")
