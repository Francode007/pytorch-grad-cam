"""
Train multiple Kvasir models sequentially on a single GPU (e.g. A100).
Saves all checkpoints under --output-dir on the (remote) server.
Use for running one job that trains arch1, then arch2, ... in order.

Example (A100 40GB, from repo root):
  python -m XAI_Enhancer_module.kvasir.train_models_sequential \\
    --archs resnet18 resnet34 resnet50 densenet121 \\
    --data-root data/kvasir-v2 --output-dir /path/on/server/kvasir_runs \\
    --a100
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

AVAILABLE_ARCHS = ["resnet18", "resnet34", "resnet50", "densenet121"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train multiple Kvasir models sequentially. All checkpoints saved under --output-dir (e.g. on remote server)."
    )
    p.add_argument(
        "--archs",
        nargs="+",
        default=AVAILABLE_ARCHS,
        choices=AVAILABLE_ARCHS,
        metavar="ARCH",
        help=f"Architectures to train in order. Default: {AVAILABLE_ARCHS}",
    )
    p.add_argument("--data-root", type=str, default="data/kvasir-v2")
    p.add_argument("--output-dir", type=str, default="runs/kvasir", help="Base dir for checkpoints (e.g. on server)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128, help="A100 40GB: 128–256 for resnet50, 256+ for resnet18")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--lr-scheduler", type=str, default="cosine")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True, help="Mixed precision (recommended for A100)")
    p.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--compile", action="store_true", default=True, help="torch.compile for speed on A100")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--a100", action="store_true", help="Preset: batch 128, amp bfloat16, compile, 8 workers")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    train_module = "XAI_Enhancer_module.kvasir.train"
    log = {"archs": args.archs, "output_dir": args.output_dir, "started": datetime.utcnow().isoformat() + "Z", "runs": []}

    for i, arch in enumerate(args.archs):
        print(f"\n{'='*60}\n[{i+1}/{len(args.archs)}] Training {arch}\n{'='*60}\n")
        cmd = [
            sys.executable,
            "-m",
            train_module,
            "--arch", arch,
            "--data-root", args.data_root,
            "--output-dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--optimizer", args.optimizer,
            "--lr-scheduler", args.lr_scheduler,
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--seed", str(args.seed),
            "--val-ratio", str(args.val_ratio),
            "--log-interval", str(args.log_interval),
        ]
        if args.amp:
            cmd += ["--amp", "--amp-dtype", args.amp_dtype]
        if args.compile:
            cmd += ["--compile"]
        if getattr(args, "a100", False):
            cmd += ["--a100"]
        if args.save_every:
            cmd += ["--save-every", str(args.save_every)]

        try:
            subprocess.run(cmd, check=True, cwd=str(project_root))
            log["runs"].append({"arch": arch, "status": "ok", "checkpoint_dir": f"{args.output_dir}/{arch}"})
        except subprocess.CalledProcessError as e:
            log["runs"].append({"arch": arch, "status": "failed", "returncode": e.returncode})
            print(f"Training {arch} failed with return code {e.returncode}. Stopping.")
            break

    log["finished"] = datetime.utcnow().isoformat() + "Z"
    manifest_path = Path(args.output_dir) / "sequential_training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    return 0 if all(r.get("status") == "ok" for r in log["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
