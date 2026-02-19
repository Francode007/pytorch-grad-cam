"""
Download IBS pre-processed dataset and create 80:20 train/val splits.

Run from the repository root (pytorch-grad-cam), e.g.:
  cd /path/to/pytorch-grad-cam
  python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root data
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from XAI_Enhancer_module.ibs.data import (
        download_ibs,
        prepare_splits,
        IBS_CLASSES,
        _find_image_paths,
    )
except ModuleNotFoundError:
    print("Error: Could not import XAI_Enhancer_module. Run this script from the repository root:")
    print("  cd /path/to/pytorch-grad-cam")
    print("  python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root data")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Download IBS pre-processed dataset and prepare 80:20 splits. "
        "Use --source kaggle for remote/server (requires Kaggle API credentials)."
    )
    p.add_argument("--data-root", type=str, default="data", help="Parent directory for IBS-preprocessed-dataset")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--source",
        type=str,
        default="kaggle",
        choices=("kaggle", "zip", "manual"),
        help="Download source: kaggle (default), zip (local .zip), manual (print instructions only)",
    )
    p.add_argument("--skip-download", action="store_true", help="Only run prepare_splits (dataset already present)")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not args.skip_download:
        try:
            extract_dir = download_ibs(data_root=str(data_root), source=args.source)
        except SystemExit:
            print("\nTo only create train/val splits (after you have extracted the dataset):")
            print(f"  python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root {data_root} --skip-download")
            raise
    else:
        extract_dir = data_root / "IBS-preprocessed-dataset"
    extract_dir = Path(extract_dir)
    if args.skip_download and not extract_dir.exists():
        print(f"Error: --skip-download was used but {extract_dir} does not exist.")
        print("Download first (e.g. with --source kaggle) or from https://www.kaggle.com/datasets/franchisn/pre-processed-ibs")
        print(f"Then extract so class folders are under: {extract_dir}")
        sys.exit(1)

    def count_class_folders(root):
        return len({c for c in IBS_CLASSES if (Path(root) / c).is_dir()})

    n_at_root = count_class_folders(extract_dir)
    if n_at_root == len(IBS_CLASSES):
        use_root = extract_dir
    else:
        best_root, best_count = extract_dir, n_at_root
        for d in extract_dir.iterdir():
            if d.is_dir():
                n = count_class_folders(d)
                if n > best_count:
                    best_root, best_count = d, n
        use_root = best_root
    pairs = _find_image_paths(use_root)
    if not pairs:
        for d in extract_dir.iterdir():
            if d.is_dir():
                pairs = _find_image_paths(d)
                if pairs:
                    use_root = d
                    break
    if not pairs:
        raise FileNotFoundError(
            f"No images found under {extract_dir}. Expected class folders: {IBS_CLASSES}. "
            "If the zip extracted with a different structure, move class folders to "
            f"{extract_dir}/<class_name>/."
        )
    print(f"Using data root: {use_root} ({len(pairs)} images)")
    train_file, val_file = prepare_splits(str(use_root), val_ratio=args.val_ratio, seed=args.seed)
    print(f"Split files: {train_file}, {val_file}")


if __name__ == "__main__":
    main()
