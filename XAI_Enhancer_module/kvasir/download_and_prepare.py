"""
Download Kvasir-v2 (version 2 only) and create train/val/test splits (70/10/20 by default).

Run from the repository root (pytorch-grad-cam), e.g.:
  cd /path/to/pytorch-grad-cam
  python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path (so "from XAI_Enhancer_module..." works)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from XAI_Enhancer_module.kvasir.data import (
        download_kvasir_v2,
        KVASIR_CLASSES,
        FOLDER_TO_CLASS,
        _find_image_paths,
    )
    from XAI_Enhancer_module.common.splits import (
        prepare_kvasir_splits,
        write_split_summary,
        deduplicate_across_splits,
    )
except ModuleNotFoundError:
    print("Error: Could not import XAI_Enhancer_module. Run this script from the repository root:")
    print("  cd /path/to/pytorch-grad-cam")
    print("  python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Download Kvasir-v2 and prepare train/val/test splits (revision: 70/10/20)."
    )
    p.add_argument("--data-root", type=str, default="data", help="Parent directory for kvasir-v2")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dedupe", action="store_true", help="Reassign near-duplicate images across splits (imagehash)")
    p.add_argument(
        "--source",
        type=str,
        default="kaggle",
        choices=("kaggle", "simula", "manual"),
        help="Download source: kaggle (default, works on server with KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json), simula (direct URL), manual (print instructions only)",
    )
    p.add_argument("--skip-download", action="store_true", help="Only run prepare_splits (dataset already present)")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not args.skip_download:
        try:
            extract_dir = download_kvasir_v2(data_root=str(data_root), source=args.source)
        except SystemExit:
            print("\nTo only create train/val splits (after you have extracted the dataset):")
            print(f"  python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root {data_root} --skip-download")
            raise
    else:
        extract_dir = data_root / "kvasir-v2"
    extract_dir = Path(extract_dir)
    if args.skip_download and not extract_dir.exists():
        print(f"Error: --skip-download was used but {extract_dir} does not exist.")
        print("Download first (e.g. with --source kaggle) or from https://www.kaggle.com/datasets/plhalvorsen/KVASIR-v2-a-gastrointestinal-tract-dataset")
        print(f"Then extract so class folders are under: {extract_dir}")
        sys.exit(1)
    # Pick the directory that has the most (ideally all 8) class folders as direct children
    def count_class_folders(root):
        # Count canonical classes that have at least one folder (canonical or alias) present
        return len({FOLDER_TO_CLASS[f] for f in FOLDER_TO_CLASS if (Path(root) / f).is_dir()})

    n_at_root = count_class_folders(extract_dir)
    if n_at_root == len(KVASIR_CLASSES):
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
        # Fallback: any subdir that has at least one class folder
        for d in extract_dir.iterdir():
            if d.is_dir():
                pairs = _find_image_paths(d)
                if pairs:
                    use_root = d
                    break
    if not pairs:
        raise FileNotFoundError(
            f"No images found under {extract_dir}. Expected class folders: {KVASIR_CLASSES}. "
            "If the zip extracted with a different structure, move class folders to "
            f"{extract_dir}/<class_name>/."
        )
    print(f"Using data root: {use_root} ({len(pairs)} images)")
    train_file, val_file, test_file = prepare_kvasir_splits(
        str(use_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.dedupe:
        moved = deduplicate_across_splits(str(use_root))
        print(f"Near-duplicate reassignment: {moved} images moved to match canonical split")
    summary = write_split_summary(str(use_root), dataset="kvasir", class_names=KVASIR_CLASSES)
    print(f"Split files: {train_file}, {val_file}, {test_file}")
    print(f"Split summary: {summary}")


if __name__ == "__main__":
    main()
