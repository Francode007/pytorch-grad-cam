"""
Orchestrate Phase 2 training matrices.

Kvasir: 5 arches × seeds (default 42/43/44)
IBS:    5 arches × folds (default 0..4)

Usage::

  python -m XAI_Enhancer_module.common.train_matrix --dataset kvasir --epochs 50
  python -m XAI_Enhancer_module.common.train_matrix --dataset ibs --epochs 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

_REPO = Path(__file__).resolve().parent.parent.parent
KVASIR_ARCHS = ("resnet18", "resnet34", "resnet50", "vgg19", "vgg16")
IBS_ARCHS = KVASIR_ARCHS
KVASIR_SEEDS = (42, 43, 44)
IBS_FOLDS = (0, 1, 2, 3, 4)


def parse_args():
    p = argparse.ArgumentParser(description="Train revision matrix (Kvasir seeds / IBS folds)")
    p.add_argument("--dataset", choices=("kvasir", "ibs", "both"), default="both")
    p.add_argument(
        "--archs",
        nargs="+",
        default=None,
        help="Default: KVASIR_ARCHS for kvasir, IBS_ARCHS for ibs",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(KVASIR_SEEDS))
    p.add_argument("--folds", nargs="+", type=int, default=list(IBS_FOLDS))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--kvasir-root", type=str, default="data/kvasir-v2")
    p.add_argument("--ibs-root", type=str, default="data/IBS-patient-dataset")
    p.add_argument("--kvasir-out", type=str, default="runs/kvasir")
    p.add_argument("--ibs-out", type=str, default="runs/ibs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="bfloat16")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run(cmd: List[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(_REPO))


def train_kvasir_matrix(args) -> None:
    amp = args.amp and not args.no_amp
    compile_model = args.compile and not args.no_compile
    archs = list(args.archs) if args.archs else list(KVASIR_ARCHS)
    for arch in archs:
        for seed in args.seeds:
            cmd = [
                sys.executable, "-m", "XAI_Enhancer_module.kvasir.train",
                "--data-root", args.kvasir_root,
                "--arch", arch,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--output-dir", args.kvasir_out,
                "--device", args.device,
                "--num-workers", "8",
            ]
            if amp:
                cmd.extend(["--amp", "--amp-dtype", args.amp_dtype])
            if compile_model:
                cmd.append("--compile")
            _run(cmd, args.dry_run)


def train_ibs_matrix(args) -> None:
    amp = args.amp and not args.no_amp
    compile_model = args.compile and not args.no_compile
    archs = list(args.archs) if args.archs else list(IBS_ARCHS)
    for arch in archs:
        for fold in args.folds:
            cmd = [
                sys.executable, "-m", "XAI_Enhancer_module.ibs.train",
                "--data-root", args.ibs_root,
                "--arch", arch,
                "--fold", str(fold),
                "--seed", "42",
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--output-dir", args.ibs_out,
                "--device", args.device,
                "--num-workers", "8",
            ]
            if amp:
                cmd.extend(["--amp", "--amp-dtype", args.amp_dtype])
            if compile_model:
                cmd.append("--compile")
            _run(cmd, args.dry_run)


def main():
    args = parse_args()
    if args.dataset in ("kvasir", "both"):
        train_kvasir_matrix(args)
    if args.dataset in ("ibs", "both"):
        train_ibs_matrix(args)


if __name__ == "__main__":
    main()
