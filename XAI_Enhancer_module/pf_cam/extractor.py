"""
PF-CAM Extractor — Standalone multi-layer CAM extraction and scoring.

Uses standard pytorch-grad-cam GradCAM (or any BaseCAM subclass) directly.
NO dependency on GradCAMEnhanced, OptimizedCamExtractor, or enhanced_cams/.

Fixes three issues from the original XAI-Enhancer:
1. Shape mismatch: Uses sequential per-layer forward passes (correct, no silent failures)
2. Negative gradients: Applies ReLU to weighted activations before masking
3. Normalization: Supports multiple strategies via NormStrategy enum

GPU Optimisation (from pf_cam_v0):
- Batch layer scoring: groups layers by activation shape (stage) and batches
  all layers within the same stage into a single mega-forward pass.
- Batch tensor input: extract_saliency_map() accepts pre-loaded tensors
  from DataLoader, skipping redundant I/O.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
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
        layer_batch_size: Number of layers to batch per GPU forward pass (within same stage)
        aggregation_config: Dict with PF-CAM config (type, beta, k_percent, temp, etc.)
        norm_strategy: Normalization strategy name (default: "gradient_weighted")
        log_weights: Whether to log per-image weights (default: False)
        weight_log_dir: Directory for weight logs (default: ".")
        sharpen_gamma: Power-law sharpening exponent (default: 1.0 = no sharpening)
        scoring_method: "cosine" (fidelity only) or "localization_aware" (composite)
        loc_weight: Balance between fidelity and localization (0=all localization, 1=all fidelity)
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        conv_layers: List[nn.Module],
        cam_method: str = "GradCAM",        # Ignored: always uses standard GradCAM
        device_preference: str = "auto",
        layer_batch_size: int = 16,          # Layers batched per forward pass (within stage)
        aggregation_config: Optional[Dict] = None,
        norm_strategy: str = "gradient_weighted",
        log_weights: bool = False,
        weight_log_dir: str = ".",
        sharpen_gamma: float = 1.0,
        scoring_mode: str = "batched",       # "batched" or "sequential"
        scoring_method: str = "localization_aware",  # "cosine" or "localization_aware"
        loc_weight: float = 0.5,            # α: 0=all localization, 1=all fidelity
    ):
        self.model = model
        self.model_name = model_name
        self.conv_layers = conv_layers
        self.device = self._resolve_device(device_preference)
        self.layer_batch_size = layer_batch_size
        self.aggregation_config = aggregation_config or {
            "type": "pyramid", "beta": 0.3, "k_percent": 0.1, "temp": 0.05
        }
        self.norm_strategy = NormStrategy(norm_strategy)
        self.log_weights = log_weights
        self.sharpen_gamma = sharpen_gamma
        self.scoring_mode = scoring_mode
        self.scoring_method = scoring_method
        self.loc_weight = loc_weight

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
        input_data: Union[str, torch.Tensor],
        predicted_label: Union[int, List[int]],
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract PF-CAM saliency map for a single image or a batch.

        Args:
            input_data: Either a file path (str) or pre-loaded tensor [B, C, H, W]
            predicted_label: Target class index (int) or list of ints for batch
            use_cache: Whether to cache model outputs (only for path input)

        Returns:
            Tuple of (input_tensor [B, C, H, W], saliency_map [B, H, W])
        """
        # 1. Handle input
        if isinstance(input_data, str):
            # Single image path
            input_tensor = self._load_image(input_data)
            if input_tensor is None:
                return None, None
            batch_size = 1
            predicted_labels = [predicted_label]
            cache_key = input_data if use_cache else None
        elif isinstance(input_data, torch.Tensor):
            # Batch tensor
            input_tensor = input_data
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            batch_size = input_tensor.shape[0]
            if isinstance(predicted_label, int):
                predicted_labels = [predicted_label] * batch_size
            else:
                predicted_labels = list(predicted_label)
            cache_key = None
        else:
            raise ValueError("input_data must be image path (str) or Tensor")

        # Process each image in the batch
        all_cams = []
        for b in range(batch_size):
            single_tensor = input_tensor[b:b+1]  # [1, C, H, W]
            lbl = predicted_labels[b]

            # 2. Get actual model output
            actual_output = self._get_actual_output(single_tensor, cache_key if b == 0 else None)

            # 3. Extract per-layer CAMs, activations, and gradients
            targets = [ClassifierOutputTarget(lbl)]
            cam_per_layer, layer_activations, layer_grads, layer_shapes = \
                self._extract_all_layer_cams(single_tensor, targets)

            # 4. Compute per-layer importance scores
            if self.scoring_method == "localization_aware":
                scores = self._compute_layer_scores_localization_aware(
                    single_tensor, actual_output, cam_per_layer,
                    layer_activations, layer_grads, layer_shapes, lbl
                )
            elif self.scoring_mode == "sequential":
                scores = self._compute_layer_scores_sequential(
                    single_tensor, actual_output, cam_per_layer,
                    layer_activations, layer_grads, lbl
                )
            else:
                scores = self._compute_layer_scores_batched(
                    single_tensor, actual_output, cam_per_layer,
                    layer_activations, layer_grads, layer_shapes, lbl
                )

            # 5. Aggregate using PF-CAM
            cams_tensor = [torch.from_numpy(c[0, :]).float() for c in cam_per_layer]

            final_cam, diagnostics = PFCamAggregator.aggregate(
                cams_tensor, scores, layer_shapes, self.aggregation_config
            )

            # 6. Normalize to [0, 1]
            final_cam = final_cam - final_cam.min()
            final_cam = final_cam / (final_cam.max() + 1e-7)

            # 6b. Optional sharpening
            if self.sharpen_gamma != 1.0:
                final_cam = final_cam ** self.sharpen_gamma
                final_cam = final_cam / (final_cam.max() + 1e-7)

            # 7. Log weights if enabled
            if self.log_weights and self.weight_logger is not None and isinstance(input_data, str):
                self._log_weights(
                    input_data, lbl, scores,
                    layer_shapes, diagnostics
                )

            all_cams.append(final_cam)

        # Stack batch results
        final_cam_batch = torch.stack(all_cams)  # [B, H, W]

        if batch_size == 1 and isinstance(input_data, str):
            return input_tensor.squeeze(0), final_cam_batch.squeeze(0)
        return input_tensor, final_cam_batch

    def get_weight_logger(self) -> Optional[WeightLogger]:
        """Return the weight logger for saving after all images."""
        return self.weight_logger

    def verify_scoring(self, image_path: str, predicted_label: int) -> Dict:
        """
        Run BOTH batched and sequential scoring on a single image and compare.

        Returns dict with per-layer scores from each method, max absolute diff,
        and a boolean indicating equivalence (within tolerance 1e-4).
        """
        import time

        input_tensor = self._load_image(image_path)
        if input_tensor is None:
            return {"error": f"Could not load {image_path}"}

        actual_output = self._get_actual_output(input_tensor)

        targets = [ClassifierOutputTarget(predicted_label)]
        cam_per_layer, layer_activations, layer_grads, layer_shapes = \
            self._extract_all_layer_cams(input_tensor, targets)

        # --- Batched scoring ---
        t0 = time.time()
        scores_batched = self._compute_layer_scores_batched(
            input_tensor, actual_output, cam_per_layer,
            layer_activations, layer_grads, layer_shapes, predicted_label
        )
        t_batched = time.time() - t0

        # Need a clean forward state — extract CAMs again to reset model
        cam_per_layer2, layer_activations2, layer_grads2, _ = \
            self._extract_all_layer_cams(input_tensor, targets)

        # --- Sequential scoring ---
        t0 = time.time()
        scores_sequential = self._compute_layer_scores_sequential(
            input_tensor, actual_output, cam_per_layer2,
            layer_activations2, layer_grads2, predicted_label
        )
        t_sequential = time.time() - t0

        # Compare
        diff = np.abs(scores_batched - scores_sequential)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        equivalent = max_diff < 1e-4

        print("\n" + "=" * 70)
        print("SCORING VERIFICATION: Batched vs Sequential")
        print("=" * 70)
        print(f"  Image: {image_path}")
        print(f"  Label: {predicted_label}")
        print(f"  Layers: {len(self.conv_layers)}")
        print(f"  Time — Batched: {t_batched:.2f}s, Sequential: {t_sequential:.2f}s")
        print(f"  Max abs score diff:  {max_diff:.6f}")
        print(f"  Mean abs score diff: {mean_diff:.6f}")

        if not equivalent:
            print(f"\n  ⚠️  SCORES DIVERGE (max diff={max_diff:.6f} > 1e-4)")
            print(f"  {'Layer':<40s}  {'Batched':>10s}  {'Sequential':>10s}  {'Diff':>10s}")
            print(f"  {'─'*40}  {'─'*10}  {'─'*10}  {'─'*10}")
            # Show top 10 most divergent layers
            top_divergent = np.argsort(diff)[::-1][:10]
            for idx in top_divergent:
                name = self._layer_names[idx] if idx < len(self._layer_names) else f"layer_{idx}"
                print(f"  {name:<40s}  {scores_batched[idx]:>10.6f}  "
                      f"{scores_sequential[idx]:>10.6f}  {diff[idx]:>10.6f}")
        else:
            print(f"\n  ✅ Scores are equivalent (max diff={max_diff:.6f})")

        print("=" * 70)

        return {
            "scores_batched": scores_batched,
            "scores_sequential": scores_sequential,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "equivalent": equivalent,
            "time_batched": t_batched,
            "time_sequential": t_sequential,
        }

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

    # ------------------------------------------------------------------
    # Localization-aware scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_spatial_concentration(
        cam_per_layer: List[np.ndarray],
        top_fraction: float = 0.1,
    ) -> np.ndarray:
        """
        Compute spatial concentration (energy ratio) for each layer's CAM.

        Measures what fraction of the CAM's total energy is concentrated in
        the top `top_fraction` of pixels. Sharp, focused CAMs score high
        (≈0.9); diffuse, uniform CAMs score low (≈0.1).

        Args:
            cam_per_layer: List of CAM arrays, each [1, H, W] or [1, 1, H, W]
            top_fraction: Fraction of pixels considered "top" (default: 10%)

        Returns:
            np.ndarray of energy-ratio scores, shape (n_layers,)
        """
        n_layers = len(cam_per_layer)
        loc_scores = np.zeros(n_layers, dtype=np.float64)

        for i, cam in enumerate(cam_per_layer):
            cam_flat = cam.flatten().astype(np.float64)
            total_energy = cam_flat.sum()

            if total_energy < 1e-10:
                # Dead CAM — no localization signal
                loc_scores[i] = 0.0
                continue

            # Sort descending and take top fraction
            cam_sorted = np.sort(cam_flat)[::-1]
            n_top = max(1, int(len(cam_flat) * top_fraction))
            top_energy = cam_sorted[:n_top].sum()

            loc_scores[i] = top_energy / total_energy

        return loc_scores

    def _compute_layer_scores_localization_aware(
        self,
        input_tensor: torch.Tensor,
        actual_output: np.ndarray,
        cam_per_layer: List[np.ndarray],
        layer_activations: List[np.ndarray],
        layer_grads: List[np.ndarray],
        layer_shapes: List[Tuple[int, int]],
        predicted_label: int,
    ) -> np.ndarray:
        """
        Composite scoring: fidelity^α × localization^(1-α).

        Combines the existing cosine-similarity fidelity score (how much
        a layer preserves the prediction) with a spatial concentration
        score (how focused the CAM is), using a geometric mean weighted
        by self.loc_weight (α).

        α = 1.0 → pure fidelity (identical to cosine scoring)
        α = 0.0 → pure localization (ignore prediction relevance)
        α = 0.5 → balanced (default)
        """
        # 1. Fidelity scores from existing batched method
        fidelity_scores = self._compute_layer_scores_batched(
            input_tensor, actual_output, cam_per_layer,
            layer_activations, layer_grads, layer_shapes, predicted_label
        )

        # 2. Localization scores from spatial concentration
        loc_scores = self._compute_spatial_concentration(cam_per_layer)

        # 3. Composite: fidelity^α × localization^(1-α)
        alpha = self.loc_weight
        # Clamp to avoid log(0)
        fidelity_safe = np.clip(fidelity_scores, 1e-10, None)
        loc_safe = np.clip(loc_scores, 1e-10, None)

        composite = (fidelity_safe ** alpha) * (loc_safe ** (1.0 - alpha))

        return composite

    def _compute_layer_scores_batched(
        self,
        input_tensor: torch.Tensor,
        actual_output: np.ndarray,
        cam_per_layer: List[np.ndarray],
        layer_activations: List[np.ndarray],
        layer_grads: List[np.ndarray],
        layer_shapes: List[Tuple[int, int]],
        predicted_label: int,
    ) -> np.ndarray:
        """
        Compute per-layer importance scores via cosine similarity,
        batching layers with the SAME activation shape in a single
        GPU forward pass (stage-grouped mega-batch).

        Fixes maintained:
        - ReLU applied before masking (Issue 2)
        - Cosine similarity on softmax probabilities (Issue 4)
        - Shape-safe: only layers with identical shapes are batched (Issue 1)
        """
        n_layers = len(self.conv_layers)
        scores = np.zeros(n_layers, dtype=np.float64)
        actual_probs = torch.softmax(
            torch.from_numpy(actual_output).float(), dim=0
        )

        input_on_device = input_tensor.to(self.device)

        # Prepare masked activations for all layers
        masked_activations = []
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
            masked_activations.append(masked_act)

        # Group layers by activation shape for batched forward passes
        shape_groups: Dict[Tuple[int, ...], List[int]] = {}
        for i in range(n_layers):
            act_shape = tuple(masked_activations[i].shape)
            if act_shape not in shape_groups:
                shape_groups[act_shape] = []
            shape_groups[act_shape].append(i)

        # Process each shape group
        for act_shape, layer_indices in shape_groups.items():
            # Process in sub-batches within the group
            for chunk_start in range(0, len(layer_indices), self.layer_batch_size):
                chunk_indices = layer_indices[chunk_start:chunk_start + self.layer_batch_size]
                num_in_chunk = len(chunk_indices)

                # Build mega-batch: replicate input for each layer in the chunk
                mega_input = input_on_device.expand(num_in_chunk, -1, -1, -1)

                # Register hooks for each layer in the chunk
                hooks = []
                for local_idx, global_idx in enumerate(chunk_indices):
                    replacement = torch.from_numpy(
                        masked_activations[global_idx]
                    ).float().to(self.device)

                    def create_hook(batch_idx, repl):
                        def hook_fn(module, inp, output):
                            # Only replace the slice for this layer's batch index
                            if repl.shape == output[batch_idx:batch_idx+1].shape:
                                output[batch_idx:batch_idx+1] = repl
                            else:
                                # Shape mismatch — skip this layer (safety)
                                pass
                            return output
                        return hook_fn

                    hook = self.conv_layers[global_idx].register_forward_hook(
                        create_hook(local_idx, replacement)
                    )
                    hooks.append(hook)

                # Run batched forward pass
                try:
                    with torch.no_grad():
                        batch_output = self.model(mega_input)

                    # Compute cosine similarity for each layer in the chunk
                    for local_idx, global_idx in enumerate(chunk_indices):
                        modified_probs = torch.softmax(
                            batch_output[local_idx].float().cpu(), dim=0
                        )
                        sim = F.cosine_similarity(
                            actual_probs.unsqueeze(0),
                            modified_probs.unsqueeze(0),
                            dim=1
                        )
                        scores[global_idx] = sim.item()
                except Exception as e:
                    # Fallback: score all layers in this chunk as 0
                    print(f"[PFCam] WARNING: Batch forward failed ({e}), scoring chunk as 0")
                    for global_idx in chunk_indices:
                        scores[global_idx] = 0.0
                finally:
                    for h in hooks:
                        h.remove()

        return scores

    # ------------------------------------------------------------------
    # Legacy sequential scoring (kept for reference / fallback)
    # ------------------------------------------------------------------

    def _compute_layer_scores_sequential(
        self,
        input_tensor: torch.Tensor,
        actual_output: np.ndarray,
        cam_per_layer: List[np.ndarray],
        layer_activations: List[np.ndarray],
        layer_grads: List[np.ndarray],
        predicted_label: int,
    ) -> np.ndarray:
        """Sequential per-layer scoring (original Issue 1 fix). Kept as fallback."""
        n_layers = len(self.conv_layers)
        scores = np.zeros(n_layers, dtype=np.float64)
        actual_probs = torch.softmax(
            torch.from_numpy(actual_output).float(), dim=0
        )

        input_on_device = input_tensor.to(self.device)

        for i in range(n_layers):
            act = layer_activations[i]
            grad = layer_grads[i]

            if grad.ndim == 4:
                weights = np.mean(grad, axis=(2, 3))
            else:
                weights = np.mean(grad, axis=(2, 3, 4))

            if act.ndim == 4:
                weighted_act = weights[:, :, None, None] * act
            else:
                weighted_act = weights[:, :, None, None, None] * act

            weighted_act_pos = np.maximum(weighted_act, 0)
            masked_act = normalize_activations(
                weighted_act_pos, act, grads=grad, strategy=self.norm_strategy
            )

            masked_act_tensor = torch.from_numpy(masked_act).float().to(self.device)
            modified_output = self._forward_with_replacement(
                input_on_device, self.conv_layers[i], masked_act_tensor
            )

            if modified_output is not None:
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
        with the given replacement tensor. (Sequential fallback)
        """
        replaced = [False]

        def hook_fn(module, inp, output):
            if replacement.shape == output.shape:
                replaced[0] = True
                return replacement
            else:
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
