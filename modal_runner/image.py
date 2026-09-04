"""Modal container image for the XAI-Enhancer pipeline."""

from __future__ import annotations

from pathlib import Path

import modal

from modal_runner.config import LOCAL_IGNORE, REPO_ROOT

_REPO_LOCAL = Path(__file__).resolve().parent.parent


def build_image() -> modal.Image:
    """
    Debian + CUDA PyTorch stack, with the repo mounted at /root/repo.

    Code is attached via ``add_local_dir`` so local edits are picked up on the
    next ``modal run`` without rebuilding the whole image.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "git",
            "wget",
            "curl",
            "libgl1",
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "libxrender1",
        )
        # CUDA wheels (Modal GPUs expose the driver; containers need matching torch)
        .pip_install(
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "numpy",
            "Pillow",
            "ttach",
            "tqdm",
            "opencv-python-headless",
            "matplotlib",
            "scikit-learn",
            "scipy",
            "psutil",
            "imagehash",
            "timm",
            "pandas",
            "pyarrow",
            "datasets",
            "kaggle",
        )
        .env(
            {
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONUNBUFFERED": "1",
            }
        )
        .add_local_dir(
            local_path=str(_REPO_LOCAL),
            remote_path=str(REPO_ROOT),
            ignore=LOCAL_IGNORE,
        )
    )
