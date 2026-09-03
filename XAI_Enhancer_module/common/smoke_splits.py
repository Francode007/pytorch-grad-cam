"""
Phase 1 smoke test: IBS filename diagnosis, synthetic patient folds, Kvasir 70/10/20 summary.

Run from the repository root:
  python -m XAI_Enhancer_module.common.smoke_splits
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from XAI_Enhancer_module.common.splits import (
    GroupIdError,
    diagnose_ibs_grouping,
    extract_group_id,
    prepare_ibs_patient_folds,
    prepare_kvasir_splits,
    write_split_summary,
)
from XAI_Enhancer_module.ibs.data import IBS_CLASSES
from XAI_Enhancer_module.kvasir.data import KVASIR_CLASSES


def _print_csv(path: Path) -> None:
    print(f"\n===== {path} =====")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            print("(empty)")
            return
        cols = list(rows[0].keys())
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
        print("  ".join(c.ljust(widths[c]) for c in cols))
        for r in rows:
            print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def _ibs_filename_audit(ibs_root: Path) -> None:
    print("===== IBS filename samples (Kaggle pre-processed) =====")
    diag = diagnose_ibs_grouping(ibs_root, n_samples=8)
    print(f"n_images={diag['n_images']} resolved={diag['n_resolved']} "
          f"unresolved={diag['n_unresolved']} exif_present={diag['exif_present']}")
    for s in diag["samples"]:
        p = ibs_root / s
        try:
            gid = extract_group_id(p, ibs_root)
            print(f"  {s}  ->  group={gid}")
        except GroupIdError as e:
            print(f"  {s}  ->  UNRESOLVED ({e})")
    if diag["resolvable"]:
        print("extract_group_id succeeded on all images.")
    else:
        print(
            "\nNo recoverable exam/patient id (numeric stems, class folders only, no EXIF). "
            "Patient-level folds require --groups-csv (R3-1 / D8)."
        )


def _synthetic_ibs_folds() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="ibs_fold_smoke_"))
    (tmp / "IBS").mkdir()
    (tmp / "Normal").mkdir()
    pairs = []
    for i in range(20):
        for f in range(1, 9):
            p = tmp / "IBS" / f"examI{i:02d}_{f:02d}.jpg"
            p.write_bytes(b"")
            pairs.append((p, 0))
    for i in range(40):
        for f in range(1, 9):
            p = tmp / "Normal" / f"examN{i:02d}_{f:02d}.jpg"
            p.write_bytes(b"")
            pairs.append((p, 1))

    def finder(_root):
        return pairs

    print("\n===== Synthetic IBS folds (exam prefix filenames) =====")
    print("samples:", [str(p.relative_to(tmp)) for p, _ in pairs[:6]])
    print("extract_group_id:", [extract_group_id(p, tmp) for p, _ in pairs[:6]])
    prepare_ibs_patient_folds(str(tmp), n_folds=5, seed=42, find_pairs_fn=finder)
    summary = write_split_summary(str(tmp), dataset="ibs", class_names=IBS_CLASSES)
    _print_csv(summary)
    return summary


def _kvasir_summary(kvasir_root: Path, dedupe: bool) -> Path:
    print("\n===== Kvasir-v2 70/10/20 =====")
    prepare_kvasir_splits(str(kvasir_root), seed=42)
    if dedupe:
        from XAI_Enhancer_module.common.splits import deduplicate_across_splits
        print("Near-duplicate scan (pHash Hamming ≤ 6)...")
        moved = deduplicate_across_splits(str(kvasir_root), hamming_threshold=6)
        print(f"Near-duplicate reassignment: {moved} images moved")
    summary = write_split_summary(str(kvasir_root), dataset="kvasir", class_names=KVASIR_CLASSES)
    _print_csv(summary)
    return summary


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Phase 1 split smoke test")
    p.add_argument("--ibs-root", type=str, default="data/IBS-preprocessed-dataset")
    p.add_argument("--kvasir-root", type=str, default="data/kvasir-v2")
    p.add_argument("--dedupe", action="store_true", help="Run pHash near-duplicate pass on Kvasir (slow)")
    p.add_argument("--no-kvasir", action="store_true")
    args = p.parse_args()

    ibs_root = Path(args.ibs_root)
    if ibs_root.exists():
        _ibs_filename_audit(ibs_root)
    else:
        print(f"IBS root not found: {ibs_root} (skipping filename audit)")

    _synthetic_ibs_folds()

    kvasir_root = Path(args.kvasir_root)
    if args.no_kvasir:
        return 0
    if not kvasir_root.exists():
        print(f"Kvasir root not found: {kvasir_root}")
        return 1
    _kvasir_summary(kvasir_root, dedupe=args.dedupe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
