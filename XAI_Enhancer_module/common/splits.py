"""
Stratified dataset splits for the CMPB revision (R3-1).

- Kvasir-v2: train / val / test (default 70/10/20), image-level, stratified per class.
- IBS: patient-level k-fold CV (StratifiedGroupKFold) with an inner validation split per fold.

Split files: <data_root>/splits/{train,val,test}.txt or splits/fold{k}/{train,val,test}.txt
Each line: relative_path<TAB>label_index
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# Type alias: (absolute_path, label_index)
Sample = Tuple[Path, int]

# Class-folder names that must NOT be treated as patient/exam IDs.
_CLASS_FOLDERS = frozenset({
    "ibs", "normal", "ibs-c", "ibs-d", "ibs_c", "ibs_d",
    "ibsc", "ibsd",
})

# exam/session prefix before a frame index: exam123_01.jpg, P12-frame-03.jpg
# Proc/CNVP IDs from franchisn/ibs-dataset: Proc202001290027_1_1.JPG
_GROUP_PATTERNS = [
    re.compile(
        r"^(?P<gid>Proc\d{8}\d+|CNVP\d+)(?:_\d+)+\.(jpe?g|png)$",
        re.I,
    ),
    re.compile(r"^(?P<gid>.+)[_-]frame[_-]?(?P<frame>\d+)\.(jpe?g|png)$", re.I),
    re.compile(r"^(?P<gid>[^_/\\]+)[_-](?P<frame>\d+)\.(jpe?g|png)$", re.I),
]


class GroupIdError(ValueError):
    """Raised when a patient/exam group id cannot be inferred from an image path."""


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _write_split_file(path: Path, pairs: Sequence[Sample], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p, lbl in pairs:
            f.write(f"{_rel(p, root)}\t{lbl}\n")


def _write_protocol(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


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


def load_group_map(csv_path: str | Path) -> Dict[str, str]:
    """
    Load image → group_id mapping.

    Accepts a header with a group column (``group_id`` / ``group`` / ``patient``)
    and a path column (``rel_path`` / ``path`` / ``filename`` / ``image`` / ``stem``),
    or a two-column file ``path,group_id`` with or without a header.
    Keys are stored as given plus basename and stem so lookups are forgiving.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Group map not found: {path}")

    mapping: Dict[str, str] = {}

    def _store(key: str, gid: str) -> None:
        key = key.strip().replace("\\", "/")
        gid = gid.strip()
        if not key or not gid:
            return
        mapping[key] = gid
        mapping[Path(key).name] = gid
        mapping[Path(key).stem] = gid

    with path.open(newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        first = sample.splitlines()[0] if sample else ""
        headerish = first.lower().replace(" ", "")
        has_header = any(
            token in headerish
            for token in ("group_id", "group", "patient", "rel_path", "filename", "image", "stem")
        )
        if has_header:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty group map: {path}")
            orig = list(reader.fieldnames)
            col = {name.lower(): name for name in orig}
            path_col = next(
                (col[c] for c in ("rel_path", "path", "filename", "image", "stem") if c in col),
                orig[0],
            )
            gid_col = next(
                (col[c] for c in ("group_id", "group", "patient", "patient_id", "exam_id") if c in col),
                orig[1] if len(orig) > 1 else None,
            )
            if gid_col is None:
                raise ValueError(f"No group_id column in {path}: {orig}")
            for row in reader:
                _store(row[path_col], row[gid_col])
        else:
            f.seek(0)
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.replace("\t", ",").split(",", 1)
                if len(parts) != 2:
                    continue
                _store(parts[0], parts[1])
    if not mapping:
        raise ValueError(f"No rows loaded from group map: {path}")
    return mapping


def _lookup_group_map(path: Path, data_root: Optional[Path], group_map: Mapping[str, str]) -> Optional[str]:
    keys: List[str] = []
    if data_root is not None:
        try:
            keys.append(_rel(path, Path(data_root)).replace("\\", "/"))
        except ValueError:
            pass
    keys.extend([str(path).replace("\\", "/"), path.name, path.stem])
    for key in keys:
        if key in group_map:
            return group_map[key]
    return None


def extract_group_id(
    image_path: Path | str,
    data_root: Optional[Path] = None,
    group_map: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Infer a patient/exam group id from a mapping, path structure, or filename.

    Heuristics (first match wins):
    1. ``group_map`` keyed by relative path, filename, or stem.
    2. Parent directory name if it is not a class folder (IBS, Normal, IBS-C, ...).
    3. Filename prefix before ``_NN`` / ``-NN`` / ``_frame_NN`` frame index.

    Raises GroupIdError if nothing matches. Numeric-only names such as ``2882.jpg``
    (Kaggle ``pre-processed-ibs``) have no recoverable exam id — do **not** fall
    back to the class folder (that collapses all IBS images into one group).
    """
    path = Path(image_path)
    if group_map:
        hit = _lookup_group_map(path, data_root, group_map)
        if hit is not None:
            return hit

    if len(path.parts) >= 2:
        parent = path.parts[-2]
        if parent.lower() not in _CLASS_FOLDERS:
            return parent

    for pat in _GROUP_PATTERNS:
        m = pat.match(path.name)
        if m:
            gid = m.group("gid")
            if gid.lower() not in _CLASS_FOLDERS:
                return gid

    raise GroupIdError(
        f"Cannot infer patient/exam id from {path.name!r}. "
        "Expected a non-class parent folder, a prefix like exam123_01.jpg, "
        "or an entry in groups_csv. The Kaggle pre-processed IBS release uses "
        "globally unique integer names (0.jpg … 5546.jpg) with no EXIF."
    )


def sample_ibs_filenames(data_root: str | Path, n: int = 12) -> List[str]:
    """Return up to ``n`` example relative paths per class folder (for diagnostics)."""
    root = Path(data_root)
    out: List[str] = []
    for cls in ("IBS", "Normal", "IBS-C", "IBS-D"):
        d = root / cls
        if not d.is_dir():
            continue
        files = sorted(
            [p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.name),
        )
        for p in files[:n]:
            out.append(str(p.relative_to(root)))
    return out


def diagnose_ibs_grouping(
    data_root: str | Path,
    n_samples: int = 12,
    group_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    """Inspect filenames / EXIF and report whether extract_group_id can run."""
    from XAI_Enhancer_module.ibs.data import _find_image_paths

    root = Path(data_root)
    pairs = _find_image_paths(root)
    samples = sample_ibs_filenames(root, n=n_samples)
    ok, failed = 0, 0
    for p, _ in pairs:
        try:
            extract_group_id(p, root, group_map=group_map)
            ok += 1
        except GroupIdError:
            failed += 1

    exif_present = False
    try:
        from PIL import Image
        for rel in samples[:6]:
            with Image.open(root / rel) as im:
                if im.getexif():
                    exif_present = True
                    break
    except Exception:
        pass

    return {
        "n_images": len(pairs),
        "n_resolved": ok,
        "n_unresolved": failed,
        "exif_present": exif_present,
        "samples": samples,
        "resolvable": failed == 0 and len(pairs) > 0,
    }


def _group_label(groups: Sequence[str], labels: Sequence[int]) -> Dict[str, int]:
    """Map each group to a single label; error if a group mixes classes."""
    seen: Dict[str, int] = {}
    for g, lbl in zip(groups, labels):
        prev = seen.get(g)
        if prev is None:
            seen[g] = lbl
        elif prev != lbl:
            raise GroupIdError(
                f"Group {g!r} has mixed labels ({prev} and {lbl}); "
                "refusing to split. Check groups_csv."
            )
    return seen


def _inner_val_by_group(
    train_pool: List[Sample],
    train_groups: List[str],
    inner_val_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    if inner_val_ratio <= 0 or len(train_pool) <= 1:
        return train_pool, []
    g_to_lbl = _group_label(train_groups, [lbl for _, lbl in train_pool])
    unique = sorted(g_to_lbl)
    if len(unique) < 2:
        return train_pool, []
    g_lbls = [g_to_lbl[g] for g in unique]
    try:
        g_train, g_val = train_test_split(
            unique,
            test_size=inner_val_ratio,
            stratify=g_lbls,
            random_state=seed,
        )
    except ValueError:
        rng = random.Random(seed)
        n_val = max(1, int(round(len(unique) * inner_val_ratio)))
        g_val = rng.sample(unique, min(n_val, len(unique) - 1))
        g_train = [g for g in unique if g not in set(g_val)]
    val_set = set(g_val)
    train_pairs = [s for s, g in zip(train_pool, train_groups) if g not in val_set]
    val_pairs = [s for s, g in zip(train_pool, train_groups) if g in val_set]
    if not train_pairs or not val_pairs:
        return train_pool, []
    return train_pairs, val_pairs


def _assert_fold_invariants(
    train_pairs: Sequence[Sample],
    val_pairs: Sequence[Sample],
    test_pairs: Sequence[Sample],
    train_g: Sequence[str],
    val_g: Sequence[str],
    test_g: Sequence[str],
    n_total: int,
    fold_idx: int,
) -> None:
    n = len(train_pairs) + len(val_pairs) + len(test_pairs)
    if n != n_total:
        raise AssertionError(
            f"Fold {fold_idx}: train+val+test = {n} != {n_total} images"
        )
    sets = {
        "train": set(train_g),
        "val": set(val_g),
        "test": set(test_g),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        leak = sets[a] & sets[b]
        if leak:
            raise AssertionError(
                f"Fold {fold_idx}: group leakage {a}/{b}: {sorted(leak)[:8]}"
            )


def prepare_ibs_patient_folds(
    data_root: str,
    n_folds: int = 5,
    seed: int = 42,
    inner_val_ratio: float = 0.15,
    find_pairs_fn: Optional[Callable[[Path], List[Sample]]] = None,
    splits_dir: Optional[str] = None,
    groups_csv: Optional[str] = None,
    group_map: Optional[Mapping[str, str]] = None,
) -> Dict[int, Dict[str, Path]]:
    """
    Create patient-disjoint k-fold splits for IBS.

    Returns mapping fold_idx -> {train, val, test} split file paths.
    Requires recoverable group ids (filename/folder heuristic or ``groups_csv``).
    """
    from XAI_Enhancer_module.ibs.data import _find_image_paths  # lazy import

    root = Path(data_root)
    finder = find_pairs_fn or _find_image_paths
    pairs = finder(root)
    if not pairs:
        raise FileNotFoundError(f"No IBS images under {root}")

    mapping: Optional[Dict[str, str]] = dict(group_map) if group_map else None
    if groups_csv:
        loaded = load_group_map(groups_csv)
        mapping = {**loaded, **(mapping or {})}

    groups: List[str] = []
    unresolved: List[Path] = []
    for p, _ in pairs:
        try:
            groups.append(extract_group_id(p, root, group_map=mapping))
        except GroupIdError:
            unresolved.append(p)
    if unresolved:
        samples = sample_ibs_filenames(root, n=8) or [p.name for p in unresolved[:8]]
        raise GroupIdError(
            f"Cannot build patient-level IBS folds: {len(unresolved)}/{len(pairs)} "
            f"images have no exam/patient id.\n"
            f"Filename samples: {samples}\n"
            "The Kaggle pre-processed release is a flat IBS/ and Normal/ dump of "
            "integer names (e.g. IBS/2882.jpg) with no EXIF. Pass --groups-csv "
            "from the Dryad archives or the data owner (H. Mihara). "
            "Refusing to group by class folder (that would be 2 groups and leak patients)."
        )

    labels = [lbl for _, lbl in pairs]
    _group_label(groups, labels)
    n_groups = len(set(groups))
    if n_groups < n_folds:
        raise GroupIdError(
            f"Only {n_groups} patient/exam groups for {n_folds}-fold CV."
        )

    base = Path(splits_dir) if splits_dir else root / "splits"
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_paths: Dict[int, Dict[str, Path]] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(pairs, labels, groups)):
        train_pool = [pairs[i] for i in train_idx]
        test_pairs = [pairs[i] for i in test_idx]
        train_g_pool = [groups[i] for i in train_idx]
        test_g = [groups[i] for i in test_idx]
        train_pairs, val_pairs = _inner_val_by_group(
            train_pool, train_g_pool, inner_val_ratio, seed=seed + fold_idx,
        )
        train_g = [extract_group_id(p, root, group_map=mapping) for p, _ in train_pairs]
        val_g = [extract_group_id(p, root, group_map=mapping) for p, _ in val_pairs]
        _assert_fold_invariants(
            train_pairs, val_pairs, test_pairs, train_g, val_g, test_g, len(pairs), fold_idx,
        )

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

    group_file = base / "group_ids.csv"
    group_file.parent.mkdir(parents=True, exist_ok=True)
    with group_file.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "group_id", "label"])
        for (p, lbl), g in zip(pairs, groups):
            w.writerow([_rel(p, root), g, lbl])

    _write_protocol(base / "PROTOCOL.txt", [
        f"dataset=ibs",
        f"created_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"n_folds={n_folds}",
        f"seed={seed}",
        f"inner_val_ratio={inner_val_ratio}",
        f"n_images={len(pairs)}",
        f"n_groups={n_groups}",
        f"groups_csv={groups_csv or ''}",
        "splitter=StratifiedGroupKFold",
        "reviewer=R3-1",
    ])
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
    """Create stratified train/val/test split files for Kvasir-v2 (default 70/10/20)."""
    from XAI_Enhancer_module.kvasir.data import _find_image_paths

    root = Path(data_root)
    finder = find_pairs_fn or _find_image_paths
    pairs = finder(root)
    if not pairs:
        raise FileNotFoundError(f"No Kvasir images under {root}")

    train_pairs, val_pairs, test_pairs = _stratified_split_pairs(
        pairs, train_ratio, val_ratio, test_ratio, seed,
    )
    n = len(train_pairs) + len(val_pairs) + len(test_pairs)
    if n != len(pairs):
        raise AssertionError(f"train+val+test = {n} != {len(pairs)}")

    base = Path(splits_dir) if splits_dir else root / "splits"
    train_f, val_f, test_f = base / "train.txt", base / "val.txt", base / "test.txt"
    _write_split_file(train_f, train_pairs, root)
    _write_split_file(val_f, val_pairs, root)
    _write_split_file(test_f, test_pairs, root)
    _write_protocol(base / "PROTOCOL.txt", [
        f"dataset=kvasir",
        f"created_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"train_ratio={train_ratio}",
        f"val_ratio={val_ratio}",
        f"test_ratio={test_ratio}",
        f"seed={seed}",
        f"n_images={len(pairs)}",
        f"n_train={len(train_pairs)}",
        f"n_val={len(val_pairs)}",
        f"n_test={len(test_pairs)}",
        "split=stratified_image_level",
        "reviewer=R3-1",
        "note=Kvasir-v2 has no patient IDs in the public release (D8).",
    ])
    return train_f, val_f, test_f


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: attach the lexicographically larger root.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def deduplicate_across_splits(
    data_root: str,
    hamming_threshold: int = 6,
    splits_dir: Optional[str] = None,
    log_csv: Optional[str] = None,
) -> int:
    """
    Move near-duplicate images (pHash Hamming ≤ threshold) into the same split.

    Each connected component is assigned to the split of its lexicographically
    first relative path. Requires ``imagehash`` and ``Pillow``.
    Returns the number of reassigned images.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError as e:
        raise ImportError("Install imagehash and Pillow for deduplicate_across_splits") from e

    root = Path(data_root)
    base = Path(splits_dir) if splits_dir else root / "splits"
    split_names = ["train", "val", "test"]
    assigned: Dict[str, str] = {}
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

    hashes: Dict[str, int] = {}
    rel_list = list(all_pairs)
    for i, rel in enumerate(rel_list):
        path, _ = all_pairs[rel]
        try:
            h = imagehash.phash(Image.open(path).convert("RGB"))
            hashes[rel] = int(str(h), 16)
        except OSError:
            continue
        if (i + 1) % 1000 == 0:
            print(f"  pHash {i + 1}/{len(rel_list)}")

    hashed = [rel for rel in rel_list if rel in hashes]
    uf = _UnionFind(hashed)
    pair_log: List[Tuple[str, str, int]] = []
    n = len(hashed)
    import numpy as np
    arr = np.array([hashes[r] for r in hashed], dtype=np.uint64)
    for i in range(n - 1):
        xor = np.bitwise_xor(arr[i], arr[i + 1 :])
        raw = np.ascontiguousarray(xor).view(np.uint8).reshape(xor.shape[0], 8)
        dist = np.unpackbits(raw, axis=1).sum(axis=1)
        hits = np.flatnonzero(dist <= hamming_threshold)
        for j in hits:
            jj = i + 1 + int(j)
            d = int(dist[j])
            uf.union(hashed[i], hashed[jj])
            pair_log.append((hashed[i], hashed[jj], d))
        if (i + 1) % 1000 == 0:
            print(f"  Hamming {i + 1}/{n} ({len(pair_log)} pairs)")

    components: Dict[str, List[str]] = defaultdict(list)
    for rel in hashed:
        components[uf.find(rel)].append(rel)

    new_assigned = dict(assigned)
    moved = 0
    for members in components.values():
        members_sorted = sorted(members)
        canonical = members_sorted[0]
        target_split = assigned[canonical]
        for rel in members:
            if new_assigned[rel] != target_split:
                moved += 1
            new_assigned[rel] = target_split

    new_splits: Dict[str, List[Sample]] = {s: [] for s in split_names}
    for rel, (path, lbl) in all_pairs.items():
        new_splits[new_assigned.get(rel, assigned[rel])].append((path, lbl))
    for split in split_names:
        _write_split_file(base / f"{split}.txt", new_splits[split], root)

    out_log = Path(log_csv) if log_csv else base / "near_duplicates.csv"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    with out_log.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_a", "rel_b", "hamming", "split_a", "split_b", "assigned_split"])
        for a, b, dist in pair_log:
            w.writerow([a, b, dist, assigned[a], assigned[b], new_assigned[a]])

    proto = base / "PROTOCOL.txt"
    extra = [
        f"dedupe_hamming_le={hamming_threshold}",
        f"dedupe_pairs={len(pair_log)}",
        f"dedupe_moved={moved}",
        f"dedupe_components={sum(1 for m in components.values() if len(m) > 1)}",
        f"n_train_after_dedupe={len(new_splits['train'])}",
        f"n_val_after_dedupe={len(new_splits['val'])}",
        f"n_test_after_dedupe={len(new_splits['test'])}",
    ]
    if proto.exists():
        proto.write_text(proto.read_text().rstrip() + "\n" + "\n".join(extra) + "\n")
    else:
        _write_protocol(proto, extra)
    return moved


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _count_by_label(pairs: Sequence[Sample]) -> Counter:
    return Counter(lbl for _, lbl in pairs)


def _load_group_ids_file(base: Path) -> Dict[str, str]:
    """Read splits/group_ids.csv (written next to fold* dirs)."""
    candidates = [base / "group_ids.csv"]
    if base.name.startswith("fold"):
        candidates.append(base.parent / "group_ids.csv")
    gf = next((p for p in candidates if p.exists()), None)
    if gf is None:
        return {}
    out: Dict[str, str] = {}
    with gf.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["rel_path"].replace("\\", "/")] = row["group_id"]
    return out


def write_split_summary(
    data_root: str,
    dataset: str = "kvasir",
    splits_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    fold: Optional[int] = None,
    group_map: Optional[Mapping[str, str]] = None,
) -> Path:
    """
    Write CSV summarising images (and patients for IBS) per class per split.

    For IBS with folds, pass ``fold`` or scan all ``fold*`` directories.
    """
    root = Path(data_root)
    base = Path(splits_dir) if splits_dir else root / "splits"
    out = Path(output_csv) if output_csv else base / f"split_summary_{dataset}.csv"
    stored_groups = _load_group_ids_file(base)

    rows: List[dict] = []

    def _group_for(path: Path) -> Optional[str]:
        rel = _rel(path, root).replace("\\", "/")
        if group_map:
            hit = _lookup_group_map(path, root, group_map)
            if hit is not None:
                return hit
        if rel in stored_groups:
            return stored_groups[rel]
        if dataset != "ibs":
            return None
        return extract_group_id(path, root, group_map=group_map)

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
                groups.append(_group_for(path) or "")
        counts = _count_by_label(pairs)
        g_by_lbl: Dict[int, set] = defaultdict(set)
        for g, (_, lbl) in zip(groups, pairs):
            if g:
                g_by_lbl[lbl].add(g)
        row = {
            "dataset": dataset,
            "fold": fold_id,
            "split": split_name,
            "n_images": len(pairs),
            "n_patients": len(set(groups) - {""}) if groups else "",
        }
        names = list(class_names) if class_names else [str(i) for i in sorted(counts)]
        for idx, name in enumerate(names):
            row[f"n_images_{name}"] = counts.get(idx, 0)
            if groups:
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
    preferred = ["dataset", "fold", "split", "n_images", "n_patients"]
    all_keys = {k for r in rows for k in r}
    fieldnames = [c for c in preferred if c in all_keys] + sorted(all_keys - set(preferred))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out
