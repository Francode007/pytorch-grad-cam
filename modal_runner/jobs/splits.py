"""Split preparation jobs (Kvasir 70/10/20, IBS patient folds, smoke)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from modal_runner.config import (
    DATA_ROOT,
    IBS_GROUPS_CSV_REPO,
    IBS_GROUPS_CSV_VOL,
    IBS_ROOT,
    KVASIR_ROOT,
)
from modal_runner.jobs.download import _ensure_groups_csv_on_volume
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
    Patient-level 5-fold CV for IBS (``IBS-patient-dataset``).

    Defaults to the bundled ``ibs_groups.csv`` (copied onto the volume).
    """
    ensure_layout()
    if not IBS_ROOT.exists():
        raise FileNotFoundError(
            f"Missing {IBS_ROOT}. Run download-ibs-patient first "
            "(franchisn/ibs-dataset → flat IBS/Normal with Proc filenames)."
        )

    if groups_csv:
        path = Path(groups_csv)
        if not path.exists():
            raise FileNotFoundError(
                f"groups_csv not found: {path}. Bundled default is "
                f"{IBS_GROUPS_CSV_REPO} (also /vol/data/ibs_groups.csv)."
            )
    else:
        path = _ensure_groups_csv_on_volume()

    # prepare_ibs_patient_folds expects data_root = class-folder parent
    args = [
        "--data-root",
        str(DATA_ROOT),
        "--skip-download",
        "--n-folds",
        str(n_folds),
        "--seed",
        str(seed),
        "--groups-csv",
        str(path),
        "--patient-dataset-root",
        str(IBS_ROOT),
    ]
    run_module("XAI_Enhancer_module.ibs.download_and_prepare", args)
    return f"IBS folds under {IBS_ROOT / 'splits'} (groups_csv={path})"


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
    if IBS_GROUPS_CSV_VOL.exists() or IBS_GROUPS_CSV_REPO.exists():
        csv_path = IBS_GROUPS_CSV_VOL if IBS_GROUPS_CSV_VOL.exists() else IBS_GROUPS_CSV_REPO
        args.extend(["--groups-csv", str(csv_path)])
    run_module("XAI_Enhancer_module.common.smoke_splits", args)
    return "smoke_splits finished"
