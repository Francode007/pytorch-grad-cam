"""Shared CAM-eval protocol helpers (Tier 1.4 / Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def enforce_eval_split(split: str, allow_val: bool = False) -> str:
    """Reported metrics use test; val requires an explicit opt-in (D-M6)."""
    split = (split or "test").lower().strip()
    if split == "val" and not allow_val:
        raise SystemExit(
            "Refusing to run CAM eval on --split val without --allow-val "
            "(revision protocol: report on test only)."
        )
    if split not in ("train", "val", "test"):
        raise SystemExit(f"Unknown split '{split}'. Expected train|val|test.")
    return split


def write_protocol_header(output_dir: str | Path, protocol: Dict[str, Any]) -> Path:
    """Write protocol.json beside CAM eval outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    with open(path, "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
    print(f"Protocol header -> {path}")
    return path


def per_image_stem(output_dir: str | Path, method_slug: str) -> str:
    """Path stem (no extension) for per-image logs of one method."""
    safe = (
        method_slug.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return str(Path(output_dir) / "per_image" / safe)
