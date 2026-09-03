"""Shared runtime helpers used inside Modal containers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from modal_runner.config import (
    DATA_ROOT,
    IBS_RUNS,
    KVASIR_RUNS,
    MODELS_ROOT,
    REPO_ROOT,
    RUNS_ROOT,
    VOL_ROOT,
)


def ensure_layout() -> None:
    """Create the standard volume directory tree."""
    for p in (
        DATA_ROOT,
        MODELS_ROOT,
        RUNS_ROOT,
        KVASIR_RUNS,
        IBS_RUNS,
    ):
        p.mkdir(parents=True, exist_ok=True)


def configure_torch_home() -> Path:
    """Point PyTorch / torchvision caches at the persistent volume."""
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(MODELS_ROOT)
    os.environ["TORCH_HUB_DIR"] = str(MODELS_ROOT / "hub")
    return MODELS_ROOT


def ensure_kaggle_credentials() -> None:
    """
    Materialise ~/.kaggle/kaggle.json from Modal secret env vars.

    Expected secret name: ``kaggle-credentials``
    Keys: ``KAGGLE_USERNAME``, ``KAGGLE_KEY``
    """
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not user or not key:
        raise RuntimeError(
            "Kaggle credentials missing. Create a Modal secret:\n"
            "  modal secret create kaggle-credentials "
            "KAGGLE_USERNAME=... KAGGLE_KEY=...\n"
            "See modal_runner/README.md."
        )
    # Also export for libraries that read env vars directly.
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    cred_path = kaggle_dir / "kaggle.json"
    cred_path.write_text(json.dumps({"username": user, "key": key}))
    cred_path.chmod(0o600)
    print(f"Kaggle credentials loaded for user={user!r}", flush=True)


def run_module(module: str, args: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    """Run ``python -m <module> ...`` inside the container (repo on PYTHONPATH)."""
    cmd: List[str] = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or REPO_ROOT))


def volume_summary(paths: Optional[Iterable[Path]] = None) -> str:
    """Return a short human-readable listing of key volume paths."""
    targets = list(paths) if paths is not None else [
        VOL_ROOT,
        DATA_ROOT,
        MODELS_ROOT,
        RUNS_ROOT,
    ]
    lines: List[str] = []
    for root in targets:
        if not root.exists():
            lines.append(f"{root}: MISSING")
            continue
        lines.append(f"{root}:")
        try:
            children = sorted(root.iterdir())[:40]
        except OSError as e:
            lines.append(f"  (error: {e})")
            continue
        if not children:
            lines.append("  (empty)")
        for child in children:
            suffix = "/" if child.is_dir() else ""
            lines.append(f"  {child.name}{suffix}")
    return "\n".join(lines)
