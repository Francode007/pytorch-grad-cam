"""
Prepare patient-level IBS k-fold splits.

Preferred data: flattened ``franchisn/ibs-dataset`` under ``IBS-patient-dataset``
with bundled ``ibs/metadata/ibs_groups.csv``.

Run from the repository root::

  python -m XAI_Enhancer_module.ibs.metadata.build_ibs_groups_csv --force
  python -m XAI_Enhancer_module.ibs.download_and_prepare --skip-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from XAI_Enhancer_module.ibs.data import (
        download_ibs,
        IBS_CLASSES,
        _find_image_paths,
    )
    from XAI_Enhancer_module.common.splits import (
        GroupIdError,
        diagnose_ibs_grouping,
        load_group_map,
        prepare_ibs_patient_folds,
        write_split_summary,
    )
except ModuleNotFoundError:
    print("Error: Could not import XAI_Enhancer_module. Run this script from the repository root:")
    print("  cd /path/to/pytorch-grad-cam")
    print("  python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root data")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare patient-level IBS k-fold splits (R3-1)."
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Parent of IBS-patient-dataset / IBS-preprocessed-dataset",
    )
    p.add_argument(
        "--patient-dataset-root",
        type=str,
        default="",
        help="Explicit class-folder root (IBS/ + Normal/). "
        "Default: <data-root>/IBS-patient-dataset, else IBS-preprocessed-dataset",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--inner-val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--source",
        type=str,
        default="kaggle",
        choices=("kaggle", "zip", "manual"),
        help="Download source for legacy numeric dump only (when not --skip-download)",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Only prepare folds (dataset already on disk)",
    )
    p.add_argument(
        "--groups-csv",
        type=str,
        default="",
        help="CSV mapping image → exam id (default: bundled ibs/metadata/ibs_groups.csv)",
    )
    return p.parse_args()


def _default_groups_csv() -> Optional[str]:
    bundled = _REPO_ROOT / "XAI_Enhancer_module" / "ibs" / "metadata" / "ibs_groups.csv"
    return str(bundled) if bundled.exists() else None


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    if args.patient_dataset_root:
        extract_dir = Path(args.patient_dataset_root)
    elif args.skip_download:
        patient = data_root / "IBS-patient-dataset"
        legacy = data_root / "IBS-preprocessed-dataset"
        extract_dir = patient if patient.is_dir() else legacy
    else:
        try:
            extract_dir = Path(download_ibs(data_root=str(data_root), source=args.source))
        except SystemExit:
            print("\nTo only create folds after building the patient tree:")
            print(
                "  python -m XAI_Enhancer_module.ibs.metadata.build_ibs_groups_csv --force"
            )
            print(
                f"  python -m XAI_Enhancer_module.ibs.download_and_prepare "
                f"--data-root {data_root} --skip-download"
            )
            raise

    extract_dir = Path(extract_dir)
    if not extract_dir.exists():
        print(f"Error: dataset root does not exist: {extract_dir}")
        print("Build from franchisn/ibs-dataset:")
        print("  python -m XAI_Enhancer_module.ibs.metadata.build_ibs_groups_csv --force")
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
            f"No images found under {extract_dir}. Expected class folders: {IBS_CLASSES}."
        )

    groups_csv = args.groups_csv or _default_groups_csv()
    group_map = load_group_map(groups_csv) if groups_csv else None
    print(f"Using data root: {use_root} ({len(pairs)} images)")
    print(f"groups_csv: {groups_csv or '(none — filename heuristics only)'}")
    diag = diagnose_ibs_grouping(use_root, group_map=group_map)
    print(
        f"IBS grouping: resolved={diag['n_resolved']} unresolved={diag['n_unresolved']} "
        f"exif={diag['exif_present']}"
    )
    print("Filename samples:", diag["samples"][:8])

    try:
        fold_paths = prepare_ibs_patient_folds(
            str(use_root),
            n_folds=args.n_folds,
            seed=args.seed,
            inner_val_ratio=args.inner_val_ratio,
            groups_csv=groups_csv,
        )
    except GroupIdError as e:
        print(f"\n{e}")
        print(
            "\nNext step: use franchisn/ibs-dataset + bundled "
            "XAI_Enhancer_module/ibs/metadata/ibs_groups.csv "
            "(see metadata/README.md)."
        )
        sys.exit(1)
    summary = write_split_summary(str(use_root), dataset="ibs", class_names=IBS_CLASSES)
    print(f"Created {len(fold_paths)} folds under {use_root / 'splits'}")
    print(f"Split summary: {summary}")


if __name__ == "__main__":
    main()
