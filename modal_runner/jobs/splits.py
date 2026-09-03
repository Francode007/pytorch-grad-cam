"""Split preparation jobs (Kvasir 70/10/20, IBS patient folds, smoke)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from modal_runner.config import DATA_ROOT, IBS_ROOT, KVASIR_ROOT
from modal_runner.runtime import ensure_layout, run_module


def prepare_kvasir_splits(*, dedupe: bool = True, seed: int = 42) -> str:
    """Create stratified 70/10/20 splits (+ optional pHash near-dup pass)."""
    ensure_layout()
    if not KVASIR_ROOT.exists():
        raise FileNotFoundError(f"Missing {KVASIR_ROOT}. Run download-kvasir first.")
    args = [
        "--data-root",
        str(DATA_ROOT),
        "--skip-download",
        "--seed",
        str(seed),
    ]
    if dedupe:
        args.append("--dedupe")
    run_module("XAI_Enhancer_module.kvasir.download_and_prepare", args)
    summary = KVASIR_ROOT / "splits" / "split_summary_kvasir.csv"
    return f"Kvasir splits written under {KVASIR_ROOT / 'splits'}; summary={summary}"


def prepare_ibs_folds(
    *,
    groups_csv: Optional[str] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> str:
    """
    Patient-level 5-fold CV for IBS.

    ``groups_csv`` must live on the volume (e.g. /vol/data/ibs_groups.csv).
    """
    ensure_layout()
    if not IBS_ROOT.exists():
        raise FileNotFoundError(f"Missing {IBS_ROOT}. Run download-ibs first.")
    args = [
        "--data-root",
        str(DATA_ROOT),
        "--skip-download",
        "--n-folds",
        str(n_folds),
        "--seed",
        str(seed),
    ]
    if groups_csv:
        path = Path(groups_csv)
        if not path.exists():
            raise FileNotFoundError(
                f"groups_csv not found: {path}. Upload it to the volume first:\n"
                "  modal volume put xai-enhancer-vol ./ibs_groups.csv /data/ibs_groups.csv"
            )
        args.extend(["--groups-csv", str(path)])
    run_module("XAI_Enhancer_module.ibs.download_and_prepare", args)
    return f"IBS folds under {IBS_ROOT / 'splits'}"


def smoke_splits(*, dedupe: bool = False) -> str:
    """Run the Phase 1 smoke script (IBS audit + Kvasir summary)."""
    ensure_layout()
    args = [
        "--ibs-root",
        str(IBS_ROOT),
        "--kvasir-root",
        str(KVASIR_ROOT),
    ]
    if dedupe:
        args.append("--dedupe")
    run_module("XAI_Enhancer_module.common.smoke_splits", args)
    return "smoke_splits finished"
