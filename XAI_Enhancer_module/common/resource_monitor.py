"""Peak GPU / host-RAM resource monitoring for Modal jobs."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ResourceReport:
    wall_s: float = 0.0
    ram_peak_mb: float = 0.0
    ram_end_mb: float = 0.0
    gpu_peak_alloc_mb: float = 0.0
    gpu_peak_reserved_mb: float = 0.0
    gpu_peak_used_mb: float = 0.0  # nvidia-smi used
    gpu_device: str = ""
    used_gpu: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        try:
            import resource

            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if rss > 1e9:
                return rss / (1024.0 * 1024.0)
            return rss / 1024.0
        except Exception:
            return 0.0


def _nvidia_smi_used_mb() -> float:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return max(vals) if vals else 0.0
    except Exception:
        return 0.0


def _gpu_torch_stats() -> tuple[bool, str, float, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "", 0.0, 0.0
        device = torch.cuda.get_device_name(0)
        alloc = float(torch.cuda.max_memory_allocated(0)) / (1024.0 * 1024.0)
        reserved = float(torch.cuda.max_memory_reserved(0)) / (1024.0 * 1024.0)
        return True, device, alloc, reserved
    except Exception:
        return False, "", 0.0, 0.0


class ResourceMonitor:
    """Context manager that records wall time and peak RAM / GPU memory."""

    def __init__(self, label: str = "", poll_s: float = 1.0):
        self.label = label
        self.poll_s = poll_s
        self._t0 = 0.0
        self._ram_peak = 0.0
        self._gpu_used_peak = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.report = ResourceReport()

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._ram_peak = max(self._ram_peak, _rss_mb())
            self._gpu_used_peak = max(self._gpu_used_peak, _nvidia_smi_used_mb())

    def __enter__(self) -> "ResourceMonitor":
        self._t0 = time.perf_counter()
        self._ram_peak = _rss_mb()
        self._gpu_used_peak = _nvidia_smi_used_mb()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def sample(self) -> None:
        self._ram_peak = max(self._ram_peak, _rss_mb())
        self._gpu_used_peak = max(self._gpu_used_peak, _nvidia_smi_used_mb())

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.sample()
        used_gpu, device, alloc, reserved = _gpu_torch_stats()
        if self._gpu_used_peak > 0:
            used_gpu = True
            if not device:
                device = "nvidia-smi"
        self.report = ResourceReport(
            wall_s=float(time.perf_counter() - self._t0),
            ram_peak_mb=float(self._ram_peak),
            ram_end_mb=float(_rss_mb()),
            gpu_peak_alloc_mb=float(alloc),
            gpu_peak_reserved_mb=float(reserved),
            gpu_peak_used_mb=float(self._gpu_used_peak),
            gpu_device=device,
            used_gpu=used_gpu,
            notes=[self.label] if self.label else [],
        )
        print(
            f"[resources] label={self.label!r} wall={self.report.wall_s:.1f}s "
            f"RAM_peak={self.report.ram_peak_mb:.0f}MB "
            f"GPU_used_peak={self.report.gpu_peak_used_mb:.0f}MB "
            f"GPU_alloc_peak={self.report.gpu_peak_alloc_mb:.0f}MB "
            f"GPU={self.report.gpu_device or 'none'}",
            flush=True,
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2, sort_keys=True)
        return path
