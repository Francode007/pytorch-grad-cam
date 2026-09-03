"""Summarize a completed Kvasir seed wave from volume metrics/args."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from modal_runner.config import IBS_ARCHS, IBS_RUNS, KVASIR_ARCHS, KVASIR_RUNS, VOLUME_NAME

# Modal A100 40GB list price used for rough post-wave estimates (see PHASE2_CODEBASE.md)
A100_USD_PER_HOUR = 2.10
CPU_RAM_OVERHEAD = 1.15


def locked_batch_path(runs_root: Path | None = None) -> Path:
    return (runs_root or KVASIR_RUNS) / "locked_batch_sizes.json"


def load_locked_batches(runs_root: Path | None = None) -> Dict[str, int]:
    root = runs_root or KVASIR_RUNS
    path = locked_batch_path(root)
    out: Dict[str, int] = {}
    shard_dir = root / "locked_batch_sizes.d"
    if shard_dir.exists():
        for p in sorted(shard_dir.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                out[str(d["arch"])] = int(d["batch_size"])
            except Exception:
                continue
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
        batches = data.get("batch_sizes") if isinstance(data, dict) else data
        if isinstance(batches, dict):
            for k, v in batches.items():
                try:
                    out[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
    return out


def save_locked_batches(
    batch_sizes: Dict[str, int],
    *,
    runs_root: Path | None = None,
    source_seed: Optional[int] = None,
    merge: bool = True,
) -> Path:
    """Write / merge /vol/runs/kvasir/locked_batch_sizes.json."""
    root = runs_root or KVASIR_RUNS
    root.mkdir(parents=True, exist_ok=True)
    path = locked_batch_path(root)
    existing = load_locked_batches(root) if merge else {}
    existing.update({str(k): int(v) for k, v in batch_sizes.items()})
    payload = {
        "batch_sizes": existing,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source_seed": source_seed,
        "note": "Reuse these batch sizes for seeds 43/44 (fair mean±SD). "
        "train.py prefers this file over re-probing when --auto-batch-size is set.",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _best_from_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}
    best = max(metrics, key=lambda r: float(r.get("val_f1_macro") or -1.0))
    last = metrics[-1]
    return {
        "epochs_done": int(last.get("epoch") or len(metrics)),
        "best_epoch": int(best.get("epoch") or 0),
        "best_val_f1_macro": float(best.get("val_f1_macro") or 0.0),
        "best_val_acc": float(best.get("val_acc") or 0.0),
        "final_val_f1_macro": float(last.get("val_f1_macro") or 0.0),
        "elapsed_s": float(last.get("elapsed_s") or 0.0),
        "gpu_used_frac_last": last.get("gpu_used_frac"),
    }


def collect_seed_rows(
    seed: int,
    *,
    archs: Optional[Sequence[str]] = None,
    runs_root: Path | None = None,
) -> List[Dict[str, Any]]:
    root = runs_root or KVASIR_RUNS
    chosen = list(archs) if archs else list(KVASIR_ARCHS)
    rows: List[Dict[str, Any]] = []
    for arch in chosen:
        run_dir = root / arch / f"seed{seed}"
        metrics_path = run_dir / "metrics.json"
        args_path = run_dir / "args.json"
        row: Dict[str, Any] = {
            "arch": arch,
            "seed": seed,
            "run_dir": str(run_dir),
            "status": "missing",
        }
        if args_path.exists():
            try:
                args = _read_json(args_path)
                row["batch_size"] = args.get("batch_size_resolved", args.get("batch_size"))
                row["batch_size_source"] = args.get("batch_size_source")
                row["batch_size_locked"] = args.get("batch_size_locked")
                row["epochs_planned"] = args.get("epochs")
                row["status"] = "started"
            except Exception as e:
                row["args_error"] = str(e)
        if (run_dir / "best.pth").exists():
            row["has_best_ckpt"] = True
        if metrics_path.exists():
            try:
                metrics = _read_json(metrics_path)
                if isinstance(metrics, list) and metrics:
                    row.update(_best_from_metrics(metrics))
                    planned = int(row.get("epochs_planned") or 0)
                    done = int(row.get("epochs_done") or 0)
                    if planned and done >= planned and (run_dir / "last.pth").exists():
                        row["status"] = "ok"
                    elif done > 0 and (run_dir / "checkpoint_latest.pth").exists():
                        row["status"] = "partial"
                    elif done > 0:
                        row["status"] = "stale_metrics"
                    else:
                        row["status"] = "empty_metrics"
                else:
                    row["status"] = "bad_metrics"
            except Exception as e:
                row["status"] = "metrics_error"
                row["metrics_error"] = str(e)
        elif row.get("status") == "started":
            row["status"] = "crashed"
        rows.append(row)
    return rows


def _read_json_safe(path: Path) -> Any:
    try:
        return _read_json(path)
    except Exception:
        return None


def format_summary_table(
    rows: List[Dict[str, Any]],
    *,
    seed: int,
    runs_root: Path | None = None,
) -> str:
    headers = [
        "arch",
        "status",
        "batch",
        "src",
        "epochs",
        "best_f1",
        "best_acc",
        "gpu%",
        "hours",
        "est_$",
    ]
    lines = [
        f"Kvasir wave summary — seed={seed}",
        f"generated_utc={datetime.now(timezone.utc).isoformat()}",
        "",
        "  ".join(f"{h:>10}" for h in headers),
        "  ".join("-" * 10 for _ in headers),
    ]
    for r in rows:
        hours = float(r.get("elapsed_s") or 0.0) / 3600.0
        est = hours * A100_USD_PER_HOUR * CPU_RAM_OVERHEAD
        gpu = r.get("gpu_used_frac_last")
        gpu_s = f"{100 * float(gpu):.0f}" if isinstance(gpu, (int, float)) else "-"
        lines.append(
            "  ".join(
                [
                    f"{r.get('arch', ''):>10}",
                    f"{r.get('status', ''):>10}",
                    f"{str(r.get('batch_size', '-')):>10}",
                    f"{str(r.get('batch_size_source') or '-'):>10}",
                    f"{str(r.get('epochs_done', '-')):>10}",
                    f"{float(r.get('best_val_f1_macro') or 0):>10.4f}"
                    if r.get("best_val_f1_macro") is not None and r.get("status") == "ok"
                    else f"{'-':>10}",
                    f"{float(r.get('best_val_acc') or 0):>10.4f}"
                    if r.get("best_val_acc") is not None and r.get("status") == "ok"
                    else f"{'-':>10}",
                    f"{gpu_s:>10}",
                    f"{hours:>10.2f}",
                    f"{est:>10.2f}",
                ]
            )
        )
    ok = [r for r in rows if r.get("status") == "ok"]
    gpu_hours = sum(float(r.get("elapsed_s") or 0.0) for r in ok) / 3600.0
    wall_hours = max((float(r.get("elapsed_s") or 0.0) for r in ok), default=0.0) / 3600.0
    cost = gpu_hours * A100_USD_PER_HOUR * CPU_RAM_OVERHEAD
    lines.extend(
        [
            "",
            f"arches_ok={len(ok)}/{len(rows)}",
            f"sum_gpu_hours={gpu_hours:.2f}  (billable ≈ sum of per-arch elapsed)",
            f"wall_clock_hours≈{wall_hours:.2f}  (max arch elapsed; parallel wave)",
            f"est_cost_usd≈{cost:.2f}  (@ ${A100_USD_PER_HOUR}/GPU-hr × {CPU_RAM_OVERHEAD} CPU/RAM)",
            f"locked_batch_file={locked_batch_path(runs_root)}",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_kvasir_seed(
    *,
    seed: int,
    archs: Optional[Sequence[str]] = None,
    runs_root: Path | None = None,
    train_results: Optional[Sequence[str]] = None,
) -> str:
    """
    Build cost/metrics table for one seed wave, lock batch sizes from args.json,
    and persist artifacts under /vol/runs/kvasir/waves/seed{seed}/.
    """
    root = runs_root or KVASIR_RUNS
    rows = collect_seed_rows(seed, archs=archs, runs_root=root)
    table = format_summary_table(rows, seed=seed, runs_root=root)
    print(table, flush=True)

    locked: Dict[str, int] = {}
    for r in rows:
        bs = r.get("batch_size")
        if r.get("arch") and isinstance(bs, (int, float)) and int(bs) > 0:
            locked[str(r["arch"])] = int(bs)
    lock_path = save_locked_batches(locked, runs_root=root, source_seed=seed, merge=True)

    wave_dir = root / "waves" / f"seed{seed}"
    wave_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "volume": VOLUME_NAME,
        "a100_usd_per_hour": A100_USD_PER_HOUR,
        "cpu_ram_overhead": CPU_RAM_OVERHEAD,
        "rows": rows,
        "locked_batch_sizes": locked,
        "locked_batch_file": str(lock_path),
        "train_results": list(train_results) if train_results else [],
        "totals": {
            "arches_ok": sum(1 for r in rows if r.get("status") == "ok"),
            "arches_total": len(rows),
            "sum_gpu_hours": sum(float(r.get("elapsed_s") or 0.0) for r in rows) / 3600.0,
            "wall_clock_hours": max((float(r.get("elapsed_s") or 0.0) for r in rows), default=0.0)
            / 3600.0,
            "est_cost_usd": (
                sum(float(r.get("elapsed_s") or 0.0) for r in rows) / 3600.0
            )
            * A100_USD_PER_HOUR
            * CPU_RAM_OVERHEAD,
        },
    }
    (wave_dir / "wave_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (wave_dir / "wave_summary.txt").write_text(table)
    # Append-only wave log (mirrors train.log pattern)
    with (wave_dir / "wave.log").open("a") as f:
        f.write(f"\n===== wave complete {payload['generated_utc']} =====\n")
        f.write(table)
        if train_results:
            f.write("\ntrain_results:\n")
            for line in train_results:
                f.write(f"  {line}\n")

    msg = (
        f"Wave summary saved → {wave_dir}/wave_summary.{{json,txt}} + wave.log; "
        f"locked batches → {lock_path}"
    )
    print(msg, flush=True)
    return table + "\n" + msg


def collect_ibs_fold_rows(
    fold: int,
    *,
    archs: Optional[Sequence[str]] = None,
    runs_root: Path | None = None,
) -> List[Dict[str, Any]]:
    """Same status logic as Kvasir, under {arch}/fold{fold}/."""
    root = runs_root or IBS_RUNS
    chosen = list(archs) if archs else list(IBS_ARCHS)
    rows: List[Dict[str, Any]] = []
    for arch in chosen:
        run_dir = root / arch / f"fold{fold}"
        metrics_path = run_dir / "metrics.json"
        args_path = run_dir / "args.json"
        row: Dict[str, Any] = {
            "arch": arch,
            "fold": fold,
            "run_dir": str(run_dir),
            "status": "missing",
        }
        if args_path.exists():
            try:
                args = _read_json(args_path)
                row["batch_size"] = args.get("batch_size_resolved", args.get("batch_size"))
                row["batch_size_source"] = args.get("batch_size_source")
                row["batch_size_locked"] = args.get("batch_size_locked")
                row["epochs_planned"] = args.get("epochs")
                row["status"] = "started"
            except Exception as e:
                row["args_error"] = str(e)
        if metrics_path.exists():
            try:
                metrics = _read_json(metrics_path)
                if isinstance(metrics, list) and metrics:
                    row.update(_best_from_metrics(metrics))
                    planned = int(row.get("epochs_planned") or 0)
                    done = int(row.get("epochs_done") or 0)
                    if planned and done >= planned and (run_dir / "last.pth").exists():
                        row["status"] = "ok"
                    elif done > 0 and (run_dir / "checkpoint_latest.pth").exists():
                        row["status"] = "partial"
                    elif done > 0:
                        row["status"] = "stale_metrics"
                    else:
                        row["status"] = "empty_metrics"
                else:
                    row["status"] = "bad_metrics"
            except Exception as e:
                row["status"] = "metrics_error"
                row["metrics_error"] = str(e)
        elif row.get("status") == "started":
            row["status"] = "crashed"
        rows.append(row)
    return rows


def summarize_ibs_fold(
    *,
    fold: int,
    archs: Optional[Sequence[str]] = None,
    runs_root: Path | None = None,
    train_results: Optional[Sequence[str]] = None,
) -> str:
    root = runs_root or IBS_RUNS
    rows = collect_ibs_fold_rows(fold, archs=archs, runs_root=root)
    table = format_summary_table(rows, seed=fold, runs_root=root).replace(
        f"Kvasir wave summary — seed={fold}",
        f"IBS fold wave summary — fold={fold}",
    )
    print(table, flush=True)
    locked: Dict[str, int] = {}
    for r in rows:
        bs = r.get("batch_size")
        if r.get("arch") and isinstance(bs, (int, float)) and int(bs) > 0:
            locked[str(r["arch"])] = int(bs)
    lock_path = save_locked_batches(locked, runs_root=root, source_seed=fold, merge=True)
    wave_dir = root / "waves" / f"fold{fold}"
    wave_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fold": fold,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "volume": VOLUME_NAME,
        "rows": rows,
        "locked_batch_sizes": locked,
        "locked_batch_file": str(lock_path),
        "train_results": list(train_results) if train_results else [],
    }
    (wave_dir / "wave_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (wave_dir / "wave_summary.txt").write_text(table)
    with (wave_dir / "wave.log").open("a") as f:
        f.write(f"\n===== fold{fold} complete {payload['generated_utc']} =====\n")
        f.write(table)
        if train_results:
            f.write("\ntrain_results:\n")
            for line in train_results:
                f.write(f"  {line}\n")
    msg = f"IBS fold summary saved → {wave_dir}; locked batches → {lock_path}"
    print(msg, flush=True)
    return table + "\n" + msg
