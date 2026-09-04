"""CPU statistics job over per-image CAM CSVs (no GPU)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from modal_runner.config import RUNS_ROOT
from modal_runner.runtime import ensure_layout, run_module


def run_stats(
    *,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> str:
    ensure_layout()
    inp = input_dir or str(RUNS_ROOT)
    out = output_dir or str(Path(inp) / "stats")
    args = [
        "--input-dir", inp,
        "--output-dir", out,
        "--n-boot", str(n_boot),
        "--seed", str(seed),
    ]
    run_module("XAI_Enhancer_module.analysis.stats", args)
    return f"Stats -> {out}"
