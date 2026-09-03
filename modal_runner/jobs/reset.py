"""Reset incomplete Kvasir seed runs / batch locks on the volume."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Sequence

from modal_runner.config import KVASIR_ARCHS, KVASIR_RUNS


def reset_kvasir_seed(
    *,
    seed: int,
    archs: Optional[Sequence[str]] = None,
    runs_root: Path | None = None,
    clear_locks: bool = True,
    wipe_runs: bool = True,
) -> str:
    """
    Clear bad auto-batch locks and/or wipe incomplete ``{arch}/seed{seed}`` dirs
    so a relaunch starts clean (no stale smoke metrics.json).
    """
    root = runs_root or KVASIR_RUNS
    chosen = list(archs) if archs else list(KVASIR_ARCHS)
    lines: List[str] = [f"reset_kvasir_seed seed={seed} root={root}"]

    if clear_locks:
        lock = root / "locked_batch_sizes.json"
        shard_dir = root / "locked_batch_sizes.d"
        if lock.exists():
            lock.unlink()
            lines.append(f"removed {lock}")
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
            lines.append(f"removed {shard_dir}")

    if wipe_runs:
        for arch in chosen:
            run_dir = root / arch / f"seed{seed}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
                lines.append(f"removed {run_dir}")
            else:
                lines.append(f"absent  {run_dir}")

    wave_dir = root / "waves" / f"seed{seed}"
    if wipe_runs and wave_dir.exists():
        # Keep wave.log history but replace summaries on next complete wave;
        # remove stale summary that claimed success from smoke leftovers.
        for name in ("wave_summary.json", "wave_summary.txt"):
            p = wave_dir / name
            if p.exists():
                p.unlink()
                lines.append(f"removed {p}")

    msg = "\n".join(lines)
    print(msg, flush=True)
    return msg
