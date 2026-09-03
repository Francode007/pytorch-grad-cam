"""
Stratified dataset splits for the CMPB revision.

- Kvasir-v2: train / val / test (default 70/10/20), image-level, stratified per class.
- IBS: patient-level k-fold CV (StratifiedGroupKFold) with an inner validation split per fold.

Split files: <data_root>/splits/{train,val,test}.txt or splits/fold{k}/{train,val,test}.txt
Each line: relative_path<TAB>label_index
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# Type alias: (absolute_path, label_index)
Sample = Tuple[Path, int]


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _write_split_file(path: Path, pairs: Sequence[Sample], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p, lbl in pairs:
            f.write(f"{_rel(p, root)}\t{lbl}\n")


def _stratified_split_pairs(
    pairs: List[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    labels = [lbl for _, lbl in pairs]
    train_pairs, holdout = train_test_split(
        pairs, test_size=(1.0 - train_ratio), stratify=labels, random_state=seed,
    )
    if val_ratio + test_ratio <= 0:
        return train_pairs, [], holdout
    relative_test = test_ratio / (val_ratio + test_ratio)
    hold_labels = [lbl for _, lbl in holdout]
    val_pairs, test_pairs = train_test_split(
        holdout, test_size=relative_test, stratify=hold_labels, random_state=seed,
    )
    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# IBS patient / exam grouping
# ---------------------------------------------------------------------------

_GROUP_PATTERNS = [
  # exam/session prefix before frame index: exam123_01.jpg, exam123-frame-01.jpg
    re.compile(r"^(?P<gid>[^_/\\-]+)[_-]\d+\.(jpe?g|png)$", re.I),
    re.compile(r"^(?P<gid>[^_/\\-]+)[_-]frame[_-]?\d+", re.I),
    # parent folder as group when images live one level down
]


def extract_group_id(image_path: Path, data_root: Optional[Path] = None) -> str:
    """
    Infer a patient/exam group id from path structure or filename.

    Heuristics (first match wins):
    1. Parent directory name if it is not a class folder (IBS, Normal, IBS-C, ...).
    2. Filename prefix before _NN or -NN frame index.
    3. Hash of parent directory (fallback — keeps same-folder images together).
    """
    path = Path(image_path)
    class_names = {"IBS", "Normal", "IBS-C", "IBS-D", "normal", "ibs"}
    parts = path.parts
    if len(parts) >= 2:
        parent = parts[-2]
        if parent not in class_names and not parent.lower().startswith("ibs"):
            return parent
    stem = path.stem
    for pat in _GROUP_PATTERNS:
        m = pat.match(path.name) or pat.match(stem)
        if m:
            return m.group("gid")
    # Stable fallback: group by immediate parent folder
    return f"dir_{hashlib.md5(str(path.parent).encode()).hexdigest()[:12]}"


def prepare_ibs_patient_folds(
    data_root: str,
    n_folds: int = 5,
    seed: int = 42,
    inner_val_ratio: float = 0.15,
    find_pairs_fn: Optional[Callable[[Path], List[Sample]]] = None,
    splits_dir: Optional[str] = None,
) -> Dict[int, Dict[str, Path]]:
    """
    Create patient-disjoint k-fold splits for IBS.

    Returns mapping fold_idx -> {train, val, test} split file paths.
    """
    from XAI_Enhancer_module.ibs.data import _find_image_paths  # lazy import

    root = Path(data_root)
    finder = find_pairs_fn or _find_image_paths
    pairs = finder(root)
    if not pairs:
        raise FileNotFoundError(f"No IBS images under {root}")

    groups = [extract_group_id(p, root) for p, _ in pairs]
    labels = [lbl for _, lbl in pairs]

    base = Path(splits_dir) if splits_dir else root / "splits"
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_paths: Dict[int, Dict[str, Path]] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(pairs, labels, groups)):
        train_pool = [pairs[i] for i in train_idx]
        test_pairs = [pairs[i] for i in test_idx]
        train_groups = [groups[i] for i in train_idx]

        # Inner val: hold out whole groups from train_pool
        if inner_val_ratio > 0 and len(train_pool) > 1:
            unique_groups = sorted(set(train_groups))
            n_val_groups = max(1, int(round(len(unique_groups) * inner_val_ratio)))
            rng = __import__("random").Random(seed + fold_idx)
            val_group_set = set(rng.sample(unique_groups, min(n_val_groups, len(unique_groups))))
            train_pairs = [s for s, g in zip(train_pool, train_groups) if g not in val_group_set]
            val_pairs = [s for s, g in zip(train_pool, train_groups) if g in val_group_set]
        else:
            train_pairs, val_pairs = train_pool, []

        fold_dir = base / f"fold{fold_idx}"
        paths = {
            "train": fold_dir / "train.txt",
            "val": fold_dir / "val.txt",
            "test": fold_dir / "test.txt",
        }
        _write_split_file(paths["train"], train_pairs, root)
        _write_split_file(paths["val"], val_pairs, root)
        _write_split_file(paths["test"], test_pairs, root)
        fold_paths[fold_idx] = paths

    return fold_paths


# ---------------------------------------------------------------------------
# Kvasir-v2 image-level splits
# ---------------------------------------------------------------------------


def prepare_kvasir_splits(
    data_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    find_pairs_fn: Optional[Callable[[Path], List[Sample]]] = None,
    splits_dir: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Create stratified train/val/test split files for Kvasir-v2."""
    from XAI_Enhancer_module.kvasir.data import _find_image_paths

    root = Path(data_root)
    finder = find_pairs_fn or _find_image_paths
    pairs = finder(root)
    if not pairs:
        raise FileNotFoundError(f"No Kvasir images under {root}")

    train_pairs, val_pairs, test_pairs = _stratified_split_pairs(
        pairs, train_ratio, val_ratio, test_ratio, seed,
    )
    base = Path(splits_dir) if splits_dir else root / "splits"
    train_f, val_f, test_f = base / "train.txt", base / "val.txt", base / "test.txt"
    _write_split_file(train_f, train_pairs, root)
    _write_split_file(val_f, val_pairs, root)
    _write_split_file(test_f, test_pairs, root)
    return train_f, val_f, test_f


def deduplicate_across_splits(
    data_root: str,
    hamming_threshold: int = 6,
    splits_dir: Optional[str] = None,
) -> int:
    """
    Move near-duplicate images (perceptual hash) into the same split as the first seen image.
    Requires ``imagehash`` and ``Pillow``. Returns number of reassigned images.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError as e:
        raise ImportError("Install imagehash and Pillow for deduplicate_across_splits") from e

    root = Path(data_root)
    base = Path(splits_dir) if splits_dir else root / "splits"
    split_names = ["train", "val", "test"]
    # Load all paths with assigned split
    assigned: Dict[str, str] = {}  # rel_path -> split
    all_pairs: Dict[str, Tuple[Path, int]] = {}
    for split in split_names:
        sf = base / f"{split}.txt"
        if not sf.exists():
            continue
        for line in sf.read_text().splitlines():
            if not line.strip():
                continue
            rel, lbl = line.split("\t", 1)
            assigned[rel] = split
            all_pairs[rel] = (root / rel, int(lbl))

    # Compute hashes
    hashes: Dict[str, imagehash.ImageHash] = {}
    for rel, (path, _) in all_pairs.items():
        try:
            hashes[rel] = imagehash.phash(Image.open(path).convert("RGB"))
        except OSError:
            continue

    # Union-find style: map rel -> canonical rel in same split
    canonical: Dict[str, str] = {rel: rel for rel in hashes}
    rel_list = list(hashes.keys())
    for i, rel_a in enumerate(rel_list):
        for rel_b in rel_list[i + 1 :]:
            if hashes[rel_a] - hashes[rel_b] <= hamming_threshold:
                # merge: keep earlier split assignment
                ca, cb = canonical[rel_a], canonical[rel_b]
                target = ca if assigned.get(ca, "train") <= assigned.get(cb, "val") else cb
                other = cb if target == ca else ca
                canonical[other] = target
                # move other to target's split
                assigned[rel_b] = assigned.get(
                    canonical[rel_b], assigned.get(rel_b, "train"),
                )

    moved = 0
    new_splits: Dict[str, List[Sample]] = {s: [] for s in split_names}
    for rel, (path, lbl) in all_pairs.items():
        can = canonical.get(rel, rel)
        split = assigned.get(can, assigned.get(rel, "train"))
        if assigned.get(rel) != split:
            moved += 1
        new_splits[split].append((path, lbl))

    for split in split_names:
        _write_split_file(base / f"{split}.txt", new_splits[split], root)
    return moved


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _count_by_label(pairs: Sequence[Sample]) -> Counter:
    return Counter(lbl for _, lbl in pairs)


def write_split_summary(
    data_root: str,
    dataset: str = "kvasir",
    splits_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    fold: Optional[int] = None,
) -> Path:
    """
    Write CSV summarising images (and patients for IBS) per class per split.

    For IBS with folds, pass ``fold`` or scans all ``fold*`` directories.
    """
    root = Path(data_root)
    base = Path(splits_dir) if splits_dir else root / "splits"
    out = Path(output_csv) if output_csv else base / f"split_summary_{dataset}.csv"

    rows: List[dict] = []

    def _process_split_file(split_name: str, split_file: Path, fold_id: str = "") -> None:
        if not split_file.exists():
            return
        pairs: List[Sample] = []
        groups: List[str] = []
        for line in split_file.read_text().splitlines():
            if not line.strip():
                continue
            rel, lbl = line.split("\t", 1)
            path = root / rel
            pairs.append((path, int(lbl)))
            if dataset == "ibs":
                groups.append(extract_group_id(path, root))
        counts = _count_by_label(pairs)
        row = {
            "dataset": dataset,
            "fold": fold_id,
            "split": split_name,
            "n_images": len(pairs),
            "n_patients": len(set(groups)) if groups else "",
        }
        for idx, count in sorted(counts.items()):
            name = class_names[idx] if class_names and idx < len(class_names) else str(idx)
            row[f"n_images_{name}"] = count
            if groups:
                g_by_lbl = defaultdict(set)
                for g, (_, lbl) in zip(groups, pairs):
                    g_by_lbl[lbl].add(g)
                row[f"n_patients_{name}"] = len(g_by_lbl.get(idx, set()))
        rows.append(row)

    if dataset == "ibs" and (fold is not None or any(base.glob("fold*"))):
        fold_dirs = [base / f"fold{fold}"] if fold is not None else sorted(base.glob("fold*"))
        for fd in fold_dirs:
            fid = fd.name.replace("fold", "")
            for split in ("train", "val", "test"):
                _process_split_file(split, fd / f"{split}.txt", fold_id=fid)
    else:
        for split in ("train", "val", "test"):
            _process_split_file(split, base / f"{split}.txt")

    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise FileNotFoundError(f"No split files found under {base}")
    fieldnames = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return out
