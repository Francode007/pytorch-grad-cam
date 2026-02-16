"""
Normalization strategies for activation masking in PF-CAM.

Provides multiple approaches for normalizing weighted activations before
masking, each with different tradeoffs between channel independence and
cross-channel importance preservation.
"""

import numpy as np
from enum import Enum
from typing import Optional


class NormStrategy(str, Enum):
    """Available normalization strategies for activation masking."""

    CHANNEL_SPATIAL = "channel_spatial"
    """Per-channel spatial normalization to [0,1]. Current baseline.
    Treats each channel independently — all channels get equal weight
    regardless of their actual contribution to classification."""

    GLOBAL = "global"
    """Global normalization across all channels.
    Preserves cross-channel magnitude ratios but dominant channels
    suppress weaker informative channels."""

    GRADIENT_WEIGHTED = "gradient_weighted"
    """Hybrid: per-channel spatial norm + gradient magnitude weighting.
    Normalizes each channel to [0,1] (preserving independence), then
    scales by gradient importance so relevant channels matter more.
    RECOMMENDED DEFAULT."""

    L2_CHANNEL = "l2_channel"
    """L2-norm per channel. Normalizes each channel to unit L2 norm.
    Preserves spatial pattern shape but loses magnitude info."""


def normalize_activations(
    weighted_activations: np.ndarray,
    activations: np.ndarray,
    grads: Optional[np.ndarray] = None,
    strategy: NormStrategy = NormStrategy.GRADIENT_WEIGHTED,
) -> np.ndarray:
    """
    Normalize weighted activations and apply activation masking.

    Implements Eq (2) from the paper: Ā^c_l = N(Q^c_l) ⊙ A^c_l
    with multiple normalization strategies for N().

    Args:
        weighted_activations: Q^c_l, shape (N, C, H, W) or (N, C, D, H, W)
        activations: A^c_l, raw activations from the layer
        grads: Gradients (required for GRADIENT_WEIGHTED strategy)
        strategy: Normalization strategy to use

    Returns:
        masked_activations: Ā^c_l, shape same as activations
    """
    if strategy == NormStrategy.CHANNEL_SPATIAL:
        return _norm_channel_spatial(weighted_activations, activations)
    elif strategy == NormStrategy.GLOBAL:
        return _norm_global(weighted_activations, activations)
    elif strategy == NormStrategy.GRADIENT_WEIGHTED:
        return _norm_gradient_weighted(weighted_activations, activations, grads)
    elif strategy == NormStrategy.L2_CHANNEL:
        return _norm_l2_channel(weighted_activations, activations)
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")


def _spatial_axes(ndim: int) -> tuple:
    """Return spatial axes for a given tensor dimensionality."""
    if ndim == 4:
        return (2, 3)       # (N, C, H, W)
    elif ndim == 5:
        return (2, 3, 4)    # (N, C, D, H, W)
    else:
        raise ValueError(f"Expected 4D or 5D array, got {ndim}D")


def _norm_channel_spatial(
    weighted_activations: np.ndarray, activations: np.ndarray
) -> np.ndarray:
    """
    Per-channel spatial normalization to [0, 1].

    Each channel is independently normalized so its spatial values span [0, 1].
    This preserves channel independence but treats all channels as equally important.
    """
    axes = _spatial_axes(weighted_activations.ndim)
    min_val = weighted_activations.min(axis=axes, keepdims=True)
    max_val = weighted_activations.max(axis=axes, keepdims=True)
    denom = max_val - min_val
    denom = np.where(denom < 1e-8, 1.0, denom)
    norm_weighted = (weighted_activations - min_val) / denom
    return norm_weighted * activations


def _norm_global(
    weighted_activations: np.ndarray, activations: np.ndarray
) -> np.ndarray:
    """
    Global normalization across all channels and spatial dimensions.

    Preserves cross-channel magnitude ratios — high-magnitude channels
    remain high, low-magnitude channels remain low. However, informative
    but low-magnitude channels may be suppressed.
    """
    # Normalize per sample (keep batch dim)
    for b in range(weighted_activations.shape[0]):
        sample = weighted_activations[b]
        min_val = sample.min()
        max_val = sample.max()
        denom = max_val - min_val
        if denom < 1e-8:
            denom = 1.0
        weighted_activations[b] = (sample - min_val) / denom
    return weighted_activations * activations


def _norm_gradient_weighted(
    weighted_activations: np.ndarray,
    activations: np.ndarray,
    grads: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Hybrid: per-channel spatial normalization + gradient magnitude weighting.

    1. Normalizes each channel spatially to [0, 1] (preserves channel independence)
    2. Scales each channel by its gradient importance (mean |gradient|)
    3. Applies the weighted normalized activations as a mask

    This captures relative channel importance while respecting independence —
    channels the model cares about more contribute more to the mask.
    """
    if grads is None:
        # Fall back to channel_spatial if no gradients provided
        return _norm_channel_spatial(weighted_activations, activations)

    axes = _spatial_axes(weighted_activations.ndim)

    # Step 1: Per-channel spatial normalization
    min_val = weighted_activations.min(axis=axes, keepdims=True)
    max_val = weighted_activations.max(axis=axes, keepdims=True)
    denom = max_val - min_val
    denom = np.where(denom < 1e-8, 1.0, denom)
    norm_weighted = (weighted_activations - min_val) / denom

    # Step 2: Compute gradient importance per channel
    # Mean absolute gradient across spatial dims → shape (N, C, 1, 1) or (N, C, 1, 1, 1)
    grad_importance = np.mean(np.abs(grads), axis=axes, keepdims=True)
    # Normalize to [0, 1] relative scale (within each sample)
    for b in range(grad_importance.shape[0]):
        max_imp = grad_importance[b].max()
        if max_imp > 1e-8:
            grad_importance[b] = grad_importance[b] / max_imp

    # Step 3: Importance-weighted mask
    return (norm_weighted * grad_importance) * activations


def _norm_l2_channel(
    weighted_activations: np.ndarray, activations: np.ndarray
) -> np.ndarray:
    """
    L2-norm per channel normalization.

    Each channel is normalized to unit L2 norm. This preserves the spatial
    pattern shape but loses magnitude information. Useful when only the
    pattern of activation matters, not its strength.
    """
    axes = _spatial_axes(weighted_activations.ndim)
    # Compute L2 norm per channel
    l2_norm = np.sqrt(np.sum(weighted_activations ** 2, axis=axes, keepdims=True))
    l2_norm = np.where(l2_norm < 1e-8, 1.0, l2_norm)
    norm_weighted = weighted_activations / l2_norm
    return norm_weighted * activations
