"""
PF-CAM Extractor — Standalone multi-layer CAM extraction and scoring.

Uses standard pytorch-grad-cam GradCAM (or any BaseCAM subclass) directly.
NO dependency on GradCAMEnhanced, OptimizedCamExtractor, or enhanced_cams/.

Fixes three issues from the original XAI-Enhancer:
1. Shape mismatch: Uses sequential per-layer forward passes (correct, no silent failures)
2. Negative gradients: Applies ReLU to weighted activations before masking
3. Normalization: Supports multiple strategies via NormStrategy enum
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

from XAI_Enhancer_module.pf_cam.normalization import NormStrategy, normalize_activations
from XAI_Enhancer_module.pf_cam.aggregator import PFCamAggregator
from XAI_Enhancer_module.pf_cam.weight_logger import WeightLogger, ImageWeightRecord


class PFCamExtractor:
    """
    Standalone PF-CAM extractor.

    Extracts per-layer CAMs using standard pytorch-grad-cam GradCAM,
    computes layer importance scores via sequential modified forward passes,
    and aggregates using PF-CAM (Pyramid Fusion with soft gating).

    Compatible with ImageNetProperAUCEvaluator via the extract_saliency_map interface.

    Args:
        model: Pre-trained CNN model
        model_name: Model identifier string
        conv_layers: List of convolutional layer modules to extract CAMs from
        cam_method: Ignored (kept for interface compatibility). Always uses GradCAM.
        device_preference: Device string ("auto", "cuda", "mps", "cpu")
        layer_batch_size: Ignored (kept for interface compatibility).
        aggregation_config: Dict with PF-CAM config (type, beta, k_percent, temp, etc.)
        norm_strategy: Normalization strategy name (default: "gradient_weighted")
        log_weights: Whether to log per-image weights (default: False)
        weight_log_dir: Directory for weight logs (default: ".")
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        conv_layers: List[nn.Module],
        cam_method: str = "GradCAM",        # Ignored: always uses standard GradCAM
        device_preference: str = "auto",
        layer_batch_size: int = 32,          # Ignored: uses sequential approach
        aggregation_config: Optional[Dict] = None,
        norm_strategy: str = "gradient_weighted",
        log_weights: bool = False,
        weight_log_dir: str = ".",
    ):
        self.model = model
        self.model_name = model_name
        self.conv_layers = conv_layers
        self.device = self._resolve_device(device_preference)
        self.aggregation_config = aggregation_config or {
            "type": "pyramid", "beta": 0.3, "k_percent": 0.1, "temp": 0.05
        }
        self.norm_strategy = NormStrategy(norm_strategy)
        self.log_weights = log_weights

        # Weight logger
        self.weight_logger = WeightLogger(output_dir=weight_log_dir) if log_weights else None

        # Build layer name map
        self._layer_names = self._build_layer_name_map()

        # ImageNet preprocessing
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Cache for actual model outputs
        self._actual_output_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public Interface (compatible with ImageNetProperAUCEvaluator)
    # ------------------------------------------------------------------

    def extract_saliency_map(
        self,
        image_path: str,
        predicted_label: int,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract PF-CAM saliency map for a single image.

        Args:
            image_path: Path to the input image
            predicted_label: Target class index
            use_cache: Whether to cache model outputs

        Returns:
            Tuple of (input_tensor [C,H,W], saliency_map [H,W])
        """
        # 1. Load and preprocess
        input_tensor = self._load_image(image_path)
        if input_tensor is None:
            return None, None

        # 2. Get actual model output (for cosine similarity)
        cache_key = image_path if use_cache else None
        actual_output = self._get_actual_output(input_tensor, cache_key)

        # 3. Extract per-layer CAMs, activations, and gradients
        targets = [ClassifierOutputTarget(predicted_label)]
        cam_per_layer, layer_activations, layer_grads, layer_shapes = \
            self._extract_all_layer_cams(input_tensor, targets)

        # 4. Compute masked activations (with ReLU fix) and per-layer scores
        scores = self._compute_layer_scores(
            input_tensor, actual_output, cam_per_layer,
            layer_activations, layer_grads, predicted_label
        )

        # 5. Aggregate using PF-CAM
        cams_tensor = [torch.from_numpy(c[0, :]).float() for c in cam_per_layer]

        final_cam, diagnostics = PFCamAggregator.aggregate(
            cams_tensor, scores, layer_shapes, self.aggregation_config
        )

        # 6. Normalize to [0, 1]
        final_cam = final_cam - final_cam.min()
        final_cam = final_cam / (final_cam.max() + 1e-7)

        # 7. Log weights if enabled
        if self.log_weights and self.weight_logger is not None:
            self._log_weights(
                image_path, predicted_label, scores,
                layer_shapes, diagnostics
            )

        return input_tensor.squeeze(0), final_cam

    def get_weight_logger(self) -> Optional[WeightLogger]:
        """Return the weight logger for saving after all images."""
        return self.weight_logger

    # ------------------------------------------------------------------
    # Core Extraction Logic
    # ------------------------------------------------------------------

    def _extract_all_layer_cams(
        self,
        input_tensor: torch.Tensor,
        targets: List,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract CAMs, activations, and gradients for ALL target layers
        using a single forward+backward pass with standard GradCAM.

        Returns:
            cam_per_layer: List of scaled CAM maps [1, H_out, W_out]
            activations: List of raw activations [1, C, H, W]
            grads: List of gradients [1, C, H, W]
            layer_shapes: List of (H, W) tuples (original spatial dims)
        """
        # Create ActivationsAndGradients to hook into ALL target layers
        act_and_grad = ActivationsAndGradients(
            self.model, self.conv_layers, reshape_transform=None, detach=True
        )

        try:
            # Forward pass — captures activations for all layers
            input_on_device = input_tensor.to(self.device)
            outputs = act_and_grad(input_on_device)

            # Backward pass — captures gradients for all layers
            self.model.zero_grad()
            loss = sum(
                [target(output) for target, output in zip(targets, outputs)]
            )
            loss.backward(retain_graph=True)

            # Collect activations and gradients
            activations_list = [a.cpu().data.numpy() for a in act_and_grad.activations]
            grads_list = [g.cpu().data.numpy() for g in act_and_grad.gradients]

        finally:
            act_and_grad.release()

        # Compute per-layer CAMs (standard GradCAM: mean(grads) * activations)
        target_size = (input_tensor.shape[-2], input_tensor.shape[-1])  # H, W
        cam_per_layer = []
        layer_shapes = []

        for i in range(len(self.conv_layers)):
            act = activations_list[i]   # (1, C, H, W)
            grad = grads_list[i]        # (1, C, H, W)

            # Record original spatial shape
            layer_shapes.append((act.shape[-2], act.shape[-1]))

            # GradCAM weights: mean of gradients over spatial dims
            if grad.ndim == 4:
                weights = np.mean(grad, axis=(2, 3))  # (1, C)
            elif grad.ndim == 5:
                weights = np.mean(grad, axis=(2, 3, 4))
            else:
                raise ValueError(f"Unexpected gradient shape: {grad.shape}")

            # Weighted activations
            if act.ndim == 4:
                weighted_act = weights[:, :, None, None] * act
            else:
                weighted_act = weights[:, :, None, None, None] * act

            # CAM = ReLU(sum across channels)
            cam = weighted_act.sum(axis=1)
            cam = np.maximum(cam, 0)

            # Scale to input size
            scaled = scale_cam_image(cam, target_size)
            cam_per_layer.append(scaled[:, None, :])

        return cam_per_layer, activations_list, grads_list, layer_shapes

    def _compute_layer_scores(
        self,
        input_tensor: torch.Tensor,
        actual_output: np.ndarray,
        cam_per_layer: List[np.ndarray],
        layer_activations: List[np.ndarray],
        layer_grads: List[np.ndarray],
        predicted_label: int,
    ) -> np.ndarray:
        """
        Compute per-layer importance scores via cosine similarity
        of modified vs actual model outputs.

        Fixes:
        - ReLU applied before masking (Issue 2)
        - Per-layer sequential forward pass (Issue 1)
        - Cosine similarity on softmax probabilities (Issue 4)
        """
        n_layers = len(self.conv_layers)
        scores = np.zeros(n_layers, dtype=np.float64)
        actual_probs = torch.softmax(
            torch.from_numpy(actual_output).float(), dim=0
        )

        input_on_device = input_tensor.to(self.device)

        for i in range(n_layers):
            act = layer_activations[i]    # (1, C, H, W)
            grad = layer_grads[i]         # (1, C, H, W)

            # GradCAM weights
            if grad.ndim == 4:
                weights = np.mean(grad, axis=(2, 3))
            else:
                weights = np.mean(grad, axis=(2, 3, 4))

            # Weighted activations
            if act.ndim == 4:
                weighted_act = weights[:, :, None, None] * act
            else:
                weighted_act = weights[:, :, None, None, None] * act

            # *** FIX Issue 2: ReLU before masking ***
            weighted_act_pos = np.maximum(weighted_act, 0)

            # *** FIX Issue 3: Configurable normalization ***
            masked_act = normalize_activations(
                weighted_act_pos, act, grads=grad, strategy=self.norm_strategy
            )

            # *** FIX Issue 1: Sequential per-layer forward pass ***
            # Replace activation at layer i with masked version, compute output
            masked_act_tensor = torch.from_numpy(masked_act).float().to(self.device)

            modified_output = self._forward_with_replacement(
                input_on_device, self.conv_layers[i], masked_act_tensor
            )

            if modified_output is not None:
                # *** FIX: Cosine similarity on softmax probabilities ***
                modified_probs = torch.softmax(
                    torch.from_numpy(modified_output).float(), dim=0
                )
                sim = F.cosine_similarity(
                    actual_probs.unsqueeze(0), modified_probs.unsqueeze(0), dim=1
                )
                scores[i] = sim.item()
            else:
                scores[i] = 0.0

        return scores

    def _forward_with_replacement(
        self,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        replacement: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """
        Run a single forward pass, replacing the output of target_layer
        with the given replacement tensor.

        This is the Issue 1 fix: one forward pass per layer, no batching,
        no shape mismatches, no silent failures.
        """
        replaced = [False]

        def hook_fn(module, inp, output):
            # Verify shape match
            if replacement.shape == output.shape:
                replaced[0] = True
                return replacement
            else:
                # Only happens if the activation shape truly doesn't match
                # (e.g., model architecture change). Log and skip.
                print(
                    f"[PFCam] WARNING: Shape mismatch at {module.__class__.__name__}: "
                    f"replacement {replacement.shape} vs output {output.shape}. Skipping."
                )
                return output

        handle = target_layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                output = self.model(input_tensor)
            return output[0].cpu().numpy()
        except Exception as e:
            print(f"[PFCam] ERROR in modified forward pass: {e}")
            return None
        finally:
            handle.remove()

    # ------------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------------

    def _load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess an image, returning [1, C, H, W] tensor."""
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self._transform(image)
            return tensor.unsqueeze(0)  # add batch dim
        except Exception as e:
            print(f"[PFCam] Error reading {image_path}: {e}")
            return None

    def _get_actual_output(
        self, input_tensor: torch.Tensor, cache_key: Optional[str] = None
    ) -> np.ndarray:
        """Get the model's output for the original image."""
        if cache_key and cache_key in self._actual_output_cache:
            return self._actual_output_cache[cache_key]

        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            output_np = output[0].cpu().numpy()

        if cache_key:
            self._actual_output_cache[cache_key] = output_np
        return output_np

    # ------------------------------------------------------------------
    # Weight Logging
    # ------------------------------------------------------------------

    def _log_weights(
        self,
        image_path: str,
        predicted_label: int,
        scores: np.ndarray,
        layer_shapes: List[Tuple[int, int]],
        diagnostics: Dict,
    ):
        """Log per-image weight information."""
        temp = self.aggregation_config.get("temp", 0.1)
        softmax_weights = torch.softmax(
            torch.tensor(scores, dtype=torch.float32) / temp, dim=0
        ).tolist()

        # Determine stage assignment per layer
        shape_to_stage = {}
        stage_counter = 0
        stage_ids = []
        for shape in layer_shapes:
            if shape not in shape_to_stage:
                shape_to_stage[shape] = stage_counter
                stage_counter += 1
            stage_ids.append(shape_to_stage[shape])

        # Top-K selection status
        top_k_selected = [False] * len(self.conv_layers)
        if "stage_selected_indices" in diagnostics:
            for stage_indices in diagnostics["stage_selected_indices"]:
                for idx in stage_indices:
                    if idx < len(top_k_selected):
                        top_k_selected[idx] = True

        record = ImageWeightRecord(
            image_path=image_path,
            predicted_label=predicted_label,
            layer_names=self._layer_names,
            raw_scores=scores.tolist(),
            softmax_weights=softmax_weights,
            layer_shapes=layer_shapes,
            stage_ids=stage_ids,
            stage_scores=diagnostics.get("stage_scores", []),
            stage_weights=diagnostics.get("stage_weights", []),
            top_k_selected=top_k_selected,
        )
        self.weight_logger.log(record)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_layer_name_map(self) -> List[str]:
        """Build human-readable names for conv layers."""
        layer_to_name = {}
        for name, module in self.model.named_modules():
            layer_to_name[id(module)] = name

        names = []
        for i, layer in enumerate(self.conv_layers):
            name = layer_to_name.get(id(layer), f"conv_{i}")
            names.append(name)
        return names

    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        """Resolve device preference to actual device."""
        if preference == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(preference)
