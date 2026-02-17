"""
Download Kvasir-v2 (version 2 only) and create 80:20 train/val splits.
Update .gitignore so dataset and .venv are not committed.

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
        prepare_splits,
        KVASIR_CLASSES,
        _find_image_paths,
    )
except ModuleNotFoundError:
    print("Error: Could not import XAI_Enhancer_module. Run this script from the repository root:")
    print("  cd /path/to/pytorch-grad-cam")
    print("  python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Download Kvasir-v2 and prepare 80:20 splits")
    p.add_argument("--data-root", type=str, default="data", help="Parent directory for kvasir-v2")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true", help="Only run prepare_splits (dataset already present)")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not args.skip_download:
        try:
            extract_dir = download_kvasir_v2(data_root=str(data_root))
        except SystemExit:
            print("\nTo only create train/val splits (after you have extracted the dataset):")
            print(f"  python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root {data_root} --skip-download")
            raise
    else:
        extract_dir = data_root / "kvasir-v2"
    extract_dir = Path(extract_dir)
    if args.skip_download and not extract_dir.exists():
        print(f"Error: --skip-download was used but {extract_dir} does not exist.")
        print("Download the zip from https://datasets.simula.no/kvasir/ (Kvasir version 2), then extract it to:")
        print(f"  {extract_dir}")
        sys.exit(1)
    # Handle nested extract: sometimes zip contains one top-level folder
    candidates = [extract_dir]
    for d in extract_dir.iterdir():
        if d.is_dir() and any((d / c).exists() for c in KVASIR_CLASSES):
            candidates.append(d)
            break
    use_root = candidates[-1] if len(candidates) > 1 else extract_dir
    pairs = _find_image_paths(use_root)
    if not pairs:
        # Try one level down
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
    train_file, val_file = prepare_splits(str(use_root), val_ratio=args.val_ratio, seed=args.seed)
    print(f"Split files: {train_file}, {val_file}")


if __name__ == "__main__":
    main()
