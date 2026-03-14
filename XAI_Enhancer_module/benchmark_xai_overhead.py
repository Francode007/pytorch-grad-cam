import csv
import sys
from typing import Callable, Dict, List

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

    # Warm-up
    for _ in range(warmup_iters):
        forward_fn()
    torch.cuda.synchronize(device)

    # Timed run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(benchmark_iters):
        forward_fn()
    end_event.record()

    torch.cuda.synchronize(device)
    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / float(benchmark_iters)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024.0 ** 2)

    return {
        "Method": name,
        "Latency_ms": avg_ms,
        "Peak_VRAM_MB": peak_mb,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available. Please run on a CUDA-enabled GPU.")

    # Use the project's device utility to keep behavior consistent
    device = get_device("cuda")

    # Enable cuDNN benchmarking for more stable, optimized timings
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    # Load model and prepare dummy input
    model = load_resnet50(device)
    target_layers = get_resnet50_target_layers(model)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    # Use a fixed target class; for overhead this is sufficient and reproducible
    targets = [ClassifierOutputTarget(0)]

    # --- Base GradCAM ---
    gradcam = GradCAM(model=model, target_layers=target_layers["standard"], use_cuda=True)

    def run_gradcam() -> None:
        _ = gradcam(input_tensor=dummy_input, targets=targets)

    # --- LayerCAM ---
    layercam = LayerCAM(model=model, target_layers=target_layers["standard"], use_cuda=True)

    def run_layercam() -> None:
        _ = layercam(input_tensor=dummy_input, targets=targets)

    # --- HiResCAM (HR-CAM) ---
    hrcam = HiResCAM(model=model, target_layers=target_layers["standard"], use_cuda=True)

    def run_hrcam() -> None:
        _ = hrcam(input_tensor=dummy_input, targets=targets)

    # --- Custom XAI-Enhancer ---
    # We use the optimized extractor with enhanced HiResCAM underneath,
    # aggregating explanations across multiple convolutional stages.
    enhancer_extractor = OptimizedCamExtractor(
        model=model,
        model_name="resnet50",
        conv_layers=target_layers["enhancer"],
        cam_method="HiResCAMEnhanced",
        device_preference="cuda",
        layer_batch_size=4,  # small layer batch size for this single-image benchmark
    )

    def run_xai_enhancer() -> None:
        # We bypass caching and use a fixed label to isolate computational overhead.
        _input_tensor, _saliency_map = enhancer_extractor.extract_saliency_map(
            input_data=dummy_input,
            predicted_label=0,
            use_cache=False,
        )

    results: List[Dict[str, float]] = []

    results.append(benchmark_method("Base GradCAM", run_gradcam, device))
    results.append(benchmark_method("LayerCAM", run_layercam, device))
    results.append(benchmark_method("HR-CAM", run_hrcam, device))
    results.append(benchmark_method("XAI-Enhancer", run_xai_enhancer, device))

    # Output as formatted CSV to stdout
    fieldnames = ["Method", "Latency_ms", "Peak_VRAM_MB"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        # Round for readability but keep numeric precision reasonable
        row_out = {
            "Method": row["Method"],
            "Latency_ms": f"{row['Latency_ms']:.4f}",
            "Peak_VRAM_MB": f"{row['Peak_VRAM_MB']:.2f}",
        }
        writer.writerow(row_out)


if __name__ == "__main__":
    main()

