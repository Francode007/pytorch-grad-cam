import csv
import gc
import sys
from pathlib import Path
from typing import Callable, Dict, List

# Project root so "XAI_Enhancer_module" can be imported when script is run as
# python XAI_Enhancer_module/benchmark_xai_overhead.py (append keeps venv's pytorch_grad_cam first)
_TOP = Path(__file__).resolve().parent.parent
if _TOP not in [Path(p).resolve() for p in sys.path]:
    sys.path.append(str(_TOP))

import torch
import torch.nn as nn

try:
    from torchvision import models
    from torchvision.models import ResNet50_Weights
    _HAS_TORCHVISION_WEIGHTS = True
except Exception:  # pragma: no cover - fallback for older torchvision
    from torchvision import models  # type: ignore
    _HAS_TORCHVISION_WEIGHTS = False

from pytorch_grad_cam import GradCAM, LayerCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from XAI_Enhancer_module.utils.model_utils import get_device
from XAI_Enhancer_module.utils.optimized_cam_extractor import OptimizedCamExtractor


def load_resnet50(device: torch.device) -> nn.Module:
    """
    Load a pre-trained ResNet-50 model on the given device.
    """
    if _HAS_TORCHVISION_WEIGHTS:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        # Fallback for older torchvision versions
        model = models.resnet50(pretrained=True)

    model.eval()
    model.to(device)
    return model


def get_resnet50_target_layers(model: nn.Module) -> Dict[str, List[nn.Module]]:
    """
    Select target layers for CAM methods:

    - Standard CAMs (GradCAM, LayerCAM, HiResCAM): last convolutional layer only.
    - XAI-Enhancer: all convolutional layers (full-depth aggregation).
    """
    all_conv_layers: List[nn.Module] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            all_conv_layers.append(module)

    if not all_conv_layers:
        raise ValueError("No convolutional layers found in ResNet-50.")

    standard_target = [all_conv_layers[-1]]
    enhancer_targets = all_conv_layers

    return {"standard": standard_target, "enhancer": enhancer_targets}


def benchmark_method(
    name: str,
    forward_fn: Callable[[], None],
    device: torch.device,
    warmup_iters: int = 50,
    benchmark_iters: int = 1000,
    log_progress: bool = True,
) -> Dict[str, float]:
    """
    Benchmark a single XAI method.

    - Runs warm-up iterations (un-timed).
    - Measures average latency over benchmark_iters using CUDA events.
    - Measures peak VRAM allocation using torch.cuda.max_memory_allocated.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if log_progress:
        print(f"  Warm-up ({warmup_iters} iters)...", flush=True)
    for i in range(warmup_iters):
        forward_fn()
        if log_progress and (i + 1) % 25 == 0 and i + 1 < warmup_iters:
            print(f"    warm-up {i + 1}/{warmup_iters}", flush=True)
    torch.cuda.synchronize(device)

    if log_progress:
        print(f"  Timed run ({benchmark_iters} iters)...", flush=True)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(benchmark_iters):
        forward_fn()
        if log_progress and (i + 1) % 200 == 0 and i + 1 < benchmark_iters:
            print(f"    timed {i + 1}/{benchmark_iters}", flush=True)
    end_event.record()

    torch.cuda.synchronize(device)
    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / float(benchmark_iters)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024.0 ** 2)

    if log_progress:
        print(f"  Done. Peak VRAM: {peak_mb:.1f} MB, Avg latency: {avg_ms:.2f} ms", flush=True)

    return {
        "Method": name,
        "Latency_ms": avg_ms,
        "Peak_VRAM_MB": peak_mb,
    }


def _run_benchmark_one_method(
    method_name: str,
    model: nn.Module,
    target_layers: Dict[str, List[nn.Module]],
    dummy_input: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    benchmark_iters: int,
) -> Dict[str, float]:
    """
    Create only this method's CAM/extractor, run benchmark, return result.
    Caller must ensure no other CAM objects are alive so memory is bounded.
    """
    targets = [ClassifierOutputTarget(0)]

    if method_name == "Base GradCAM":
        cam = GradCAM(model=model, target_layers=target_layers["standard"])
        def forward_fn() -> None:
            _ = cam(input_tensor=dummy_input, targets=targets)
    elif method_name == "LayerCAM":
        cam = LayerCAM(model=model, target_layers=target_layers["standard"])
        def forward_fn() -> None:
            _ = cam(input_tensor=dummy_input, targets=targets)
    elif method_name == "HR-CAM":
        cam = HiResCAM(model=model, target_layers=target_layers["standard"])
        def forward_fn() -> None:
            _ = cam(input_tensor=dummy_input, targets=targets)
    elif method_name == "XAI-Enhancer":
        extractor = OptimizedCamExtractor(
            model=model,
            model_name="resnet50",
            conv_layers=target_layers["enhancer"],
            cam_method="HiResCAMEnhanced",
            device_preference="cuda",
            layer_batch_size=4,
        )
        def forward_fn() -> None:
            _, _ = extractor.extract_saliency_map(
                input_data=dummy_input, predicted_label=0, use_cache=False
            )
        cam = extractor  # for cleanup below
    else:
        raise ValueError(f"Unknown method: {method_name}")

    result = benchmark_method(
        method_name, forward_fn, device,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        log_progress=True,
    )
    # Release CAM/extractor and hooks so next method doesn't add to peak memory
    del cam
    if method_name == "XAI-Enhancer":
        try:
            del extractor
        except NameError:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark XAI methods: latency and peak VRAM.")
    parser.add_argument("--warmup", type=int, default=50, help="Warm-up iterations per method.")
    parser.add_argument("--iters", type=int, default=1000,
                        help="Timed iterations per method. Use 100–200 if OOM on XAI-Enhancer.")
    parser.add_argument("--skip-enhancer", action="store_true",
                        help="Skip XAI-Enhancer (use when testing standard methods only).")
    parser.add_argument("--output", "-o", type=str, default="benchmark_xai_results.csv",
                        help="Output CSV path for results (default: benchmark_xai_results.csv).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available. Please run on a CUDA-enabled GPU.")

    device = torch.device(get_device("cuda"))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    print("Loading ResNet-50 and building target layers...", flush=True)
    model = load_resnet50(device)
    target_layers = get_resnet50_target_layers(model)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    n_std = len(target_layers["standard"])
    n_enh = len(target_layers["enhancer"])
    print(f"  Standard (last layer): {n_std} layer(s). Enhancer (all conv): {n_enh} layers.", flush=True)
    print("", flush=True)

    results: List[Dict[str, float]] = []
    method_names = ["Base GradCAM", "LayerCAM", "HR-CAM"]
    if not args.skip_enhancer:
        method_names.append("XAI-Enhancer")

    for name in method_names:
        print(f"Benchmarking {name} (warmup={args.warmup}, iters={args.iters})...", flush=True)
        res = _run_benchmark_one_method(
            name, model, target_layers, dummy_input, device,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
        )
        results.append(res)
        print("", flush=True)

    print("=== Results ===", flush=True)
    fieldnames = ["Method", "Latency_ms", "Peak_VRAM_MB"]
    out_stdout = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    out_stdout.writeheader()
    for row in results:
        out_stdout.writerow({
            "Method": row["Method"],
            "Latency_ms": f"{row['Latency_ms']:.4f}",
            "Peak_VRAM_MB": f"{row['Peak_VRAM_MB']:.2f}",
        })
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow({
                "Method": row["Method"],
                "Latency_ms": row["Latency_ms"],
                "Peak_VRAM_MB": row["Peak_VRAM_MB"],
            })
    print(f"Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

