#!/usr/bin/env python3
"""
Build a flat IBS/Normal tree + ``ibs_groups.csv`` from the patient-structured
Kaggle dump ``franchisn/ibs-dataset`` (unnormalized).

The numeric ``franchisn/pre-processed-ibs`` release cannot be fully remapped
back to these exam IDs (pHash/ORB only recover ~20%); revision experiments
should train on this patient-aware tree instead.

Usage (from repo root)::

  # after: kaggle datasets download -d franchisn/ibs-dataset --unzip -p data/ibs-raw-unnormalized
  python -m XAI_Enhancer_module.ibs.metadata.build_ibs_groups_csv \\
    --raw-root data/ibs-raw-unnormalized/Endoscope-Normal-IBS-Classification-Data \\
    --out-root data/IBS-patient-dataset
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROC_RE = re.compile(r"(Proc\d{8}\d+|CNVP\d+)", re.I)
EXTS = {".jpg", ".jpeg", ".png"}


def fw(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def binary_label(shard: str) -> str:
    return "IBS" if shard.upper().startswith("IBS") else "Normal"


def exam_id(name: str) -> str:
    m = PROC_RE.search(name)
    return m.group(1) if m else Path(name).stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/ibs-raw-unnormalized/Endoscope-Normal-IBS-Classification-Data"),
    )
    p.add_argument("--out-root", type=Path, default=Path("data/IBS-patient-dataset"))
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("XAI_Enhancer_module/ibs/metadata/ibs_groups.csv"),
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hardlinking (default: hardlink when possible)",
    )
    p.add_argument("--force", action="store_true", help="Delete out-root if it exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.raw_root.is_dir():
        print(f"Missing raw root: {args.raw_root}", file=sys.stderr)
        return 1

    if args.out_root.exists():
        if not args.force:
            print(f"Refusing to overwrite {args.out_root} (pass --force)", file=sys.stderr)
            return 1
        shutil.rmtree(args.out_root)

    (args.out_root / "IBS").mkdir(parents=True)
    (args.out_root / "Normal").mkdir(parents=True)

    rows: list[dict] = []
    collisions = 0
    for p in sorted(args.raw_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        parts = p.relative_to(args.raw_root).parts
        if len(parts) < 3:
            continue
        shard, patient, fname = parts[0], fw(parts[1]), parts[2]
        lab = binary_label(shard)
        dest_name = fname
        dest = args.out_root / lab / dest_name
        if dest.exists():
            dest_name = f"{shard}_p{patient}_{fname}"
            dest = args.out_root / lab / dest_name
            collisions += 1
        if args.copy:
            shutil.copy2(p, dest)
        else:
            try:
                os.link(p, dest)
            except OSError:
                shutil.copy2(p, dest)
        rows.append(
            {
                "rel_path": f"{lab}/{dest_name}",
                "group_id": exam_id(fname),
                "label": lab,
                "raw_rel": str(p.relative_to(args.raw_root.parent)).replace("\\", "/"),
                "shard": shard,
                "patient_folder": patient,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rel_path", "group_id", "label", "raw_rel", "shard", "patient_folder"],
        )
        w.writeheader()
        w.writerows(rows)

    n_groups = len({r["group_id"] for r in rows})
    by_lab = Counter(r["label"] for r in rows)
    print(
        f"Wrote {len(rows)} images -> {args.out_root} "
        f"(IBS={by_lab['IBS']}, Normal={by_lab['Normal']}, groups={n_groups}, "
        f"renamed_collisions={collisions})"
    )
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
