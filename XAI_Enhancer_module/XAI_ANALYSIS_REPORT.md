# XAI-Enhancer → PF-CAM: Comprehensive Analysis Report

## Executive Summary

This report documents a thorough code review and methodology analysis of the XAI-Enhancer framework, covering:
1. Why the original XAI-Enhancer underperforms standard CAM methods on ImageNet
2. How the Pyramid Fusion CAM (PF-CAM) addresses these issues
3. Layer selection guidance for ImageNet vs medical imaging tasks
4. Further improvement recommendations

---

## Part A — Original XAI-Enhancer Issues

### Key Results: EnhancedCAM vs Standard Methods on ImageNet (5000 images, ResNet50)

| Metric | EnhancedCAM | GradCAM | HiResCAM | GradCAM++ |
|--------|:-----------:|:-------:|:--------:|:---------:|
| Insertion AUC ↑ | **0.486** | 0.564 | 0.564 | 0.552 |
| Deletion AUC ↓ | 0.156 | **0.152** | **0.152** | 0.156 |
| ROAD ↑ | 0.276 | **0.295** | **0.295** | 0.289 |

> **⚠️ EnhancedCAM has 14% lower insertion AUC and slightly worse deletion and ROAD scores compared to standard methods on ImageNet.**

### 4 Critical Issues Identified

#### Issue 1: Shape Mismatch in Forward Hooks (Silent Failure)

In `optimized_cam_extractor.py`, the `compute_modified_outputs_batch` method uses hooks to replace layer outputs during forward propagation. When masked activations from early layers (e.g., conv1: 64 channels, 112×112) don't match the expected output shape, the `except` block silently returns unmodified output—meaning many layers contribute **zero** information to the weight calculation.

```python
# The hook silently fails on shape mismatch:
def hook_fn(module, input, output):
    if replacement.shape != output[idx_in_batch].shape:
         try:
             reshaped_replacement = replacement.view(output[idx_in_batch].shape)
         except:
             return output  # ← SILENT FAILURE!
```

#### Issue 2: Negative Gradients Retained in Masking

In `GradCAM_enhanced.py`, the CAM map is ReLU'd (`np.maximum(cam, 0)`) but the masked activations retain influence from negative-weighted channels after normalisation, creating a mismatch between what the CAM highlights vs what the masked activations emphasise.

#### Issue 3: Channel-wise vs Global Normalisation

The paper's Eq (2) states `N(Q^c_l)` — normalisation of the feature map. The code normalises **per-channel spatially**, which gives all channels equal relative importance regardless of their actual importance.

#### Issue 4: Cosine Similarity on Raw Logits

For ImageNet with 1000 classes, cosine similarity on raw logits yields values very close to 1.0 for all layers. The softmax weights become approximately uniform, effectively degrading to a simple average of all CAM maps.

### Root Cause: IBS vs ImageNet

| Property | IBS Dataset | ImageNet |
|----------|:-----------:|:--------:|
| Classes | 4 (binary-like) | 1000 |
| Distinguishing features | **Textures** (low/mid-level) | **Objects** (high-level semantics) |
| Early layers discriminative? | ✅ Yes | ❌ Mostly no |
| Multi-layer aggregation helps? | ✅ Yes (+50%) | ❌ No (−14%) |

---

## Part B — PF-CAM Analysis

### Architecture

PF-CAM introduces three key innovations over the original XAI-Enhancer:

1. **Stage-Based Grouping**: Layers are grouped by spatial resolution (e.g., 7×7, 14×14, 28×28, 56×56)
2. **Intra-Stage Top-K Selection**: Only the best layers within each stage contribute (controlled by `k_percent`)
3. **Top-Down Soft Gating**: Deep semantic layers create a "gate" that controls how much detail from shallow layers passes through

```
Soft Gate Formula:
    gate = normalized(fused_cam)  # from deeper stages
    soft_gate = β + (1 - β) × gate
    masked_details = shallow_stage_cam × soft_gate
    fused_cam = fused_cam + masked_details
```

### Issue-by-Issue Verification

| # | Original Issue | Fixed by PF-CAM? | Details |
|:-:|:---|:-:|:---|
| 1 | Shape mismatch in forward hooks | ⚠️ Partially | Still uses `OptimizedCamExtractor` underneath, but PF-CAM's structural guidance (stage grouping + soft gating) makes it more resilient to noisy weights |
| 2 | Negative gradients in masking | ❌ No | `GradCAM_enhanced.py` is unchanged |
| 3 | Channel-wise normalisation | ❌ No | Same code, same issue |
| 4 | Cosine similarity near-uniform | ✅ Mitigated | Temperature scaling (temp=0.05) dramatically sharpens weights; Top-K filters out low-scoring layers |
| 5 | Early layer noise dilution | ✅ Addressed | Soft gating ensures deep layers anchor the explanation; shallow layers only refine within the object mask |

### PF-CAM Results (Best Configuration: β=0.3, k_percent=0.1, temp=0.05)

| Metric | PF-CAM | Original EnhancedCAM | GradCAM | vs GradCAM |
|--------|:------:|:-------------------:|:-------:|:----------:|
| Insertion AUC ↑ | **0.617** | 0.486 | 0.564 | **+9.4%** |
| Deletion AUC ↓ | **0.136** | 0.156 | 0.152 | **−10.5%** |
| ROAD ↑ | 0.232 | 0.276 | **0.295** | −21.4% |

PF-CAM significantly improves over both the original EnhancedCAM and standard GradCAM on Insertion and Deletion metrics. ROAD score is lower, likely due to spatially broader activation regions from multi-resolution fusion.

---

## Part C — Layer Selection Guidance

### ImageNet (Semantic/Object Classification)

**Recommendation: Use `layer_mode='all'` with PF-CAM only (not with original XAI-Enhancer).**

PF-CAM handles all layers safely via:
- Stage grouping separates noisy early layers from semantic late layers
- Top-K filters worst layers per stage
- Soft gating lets deep layers control shallow layer contributions
- Temperature scaling sharpens weights

| Parameter | Recommended for ImageNet |
|:--|:--|
| **Layer mode** | `all` |
| **β (soft gating)** | 0.2–0.3 (strict gating) |
| **k_percent** | 0.1 (10%, selective) |
| **temp** | 0.05–0.1 (sharp) |

**Alternative:** `layer_mode='last_5'` for ~10× faster evaluation with comparable quality.

### Medical Imaging (IBS/Texture Classification)

**Recommendation: Use `layer_mode='all'` with both PF-CAM and XAI-Enhancer.**

Medical imaging benefits from all layers because discriminative texture features are encoded across early-to-mid layers.

| Parameter | Recommended for Medical Imaging |
|:--|:--|
| **Layer mode** | `all` |
| **β (soft gating)** | 0.4–0.6 (permissive, allow textures) |
| **k_percent** | 0.2–0.3 (keep more layers) |
| **temp** | 0.5–1.0 (distributed weights) |

---

## Part D — Further Improvement Recommendations

### Priority 1: Fix Remaining Bugs in `GradCAM_enhanced.py`

```python
# Apply ReLU before normalize_and_mask:
weighted_activations_pos = np.maximum(weighted_activations, 0)
masked_activations = self.normalize_and_mask_activations(weighted_activations_pos, activations)
```

### Priority 2: Fix `compute_modified_outputs_batch` Silent Failures

The batched hook approach silently returns unmodified outputs when shapes don't match. Options:
- Fall back to per-layer forward passes (correct but slower)
- Use `torch.nn.functional.interpolate` to resize masked activations
- Only inject at matching layers and skip mismatched ones

### Priority 3: Add Resolution-Aware Stage Weighting

Weight stage contributions by their cosine similarity scores:

```python
stage_importance = F.softmax(torch.tensor(stage_scores_list) / temp, dim=0)
masked_details = next_stage_cam * soft_gate * stage_importance[i]
```

### Priority 4: Improve ROAD Score

- Lower β (0.1–0.2) to reduce background leakage
- Apply spatial entropy thresholding on final PF-CAM (remove regions < 5th percentile)
- Consider bilateral filtering to smooth noise while preserving edges

### Priority 5: Use Softmax Probabilities for Cosine Similarity

```python
actual_probs = torch.softmax(torch.from_numpy(actual_output), dim=0)
modified_probs = torch.softmax(torch.from_numpy(modified_output), dim=0)
similarity = F.cosine_similarity(actual_probs.unsqueeze(0), modified_probs.unsqueeze(0))
```

---

*Analysis date: 2026-02-16 | Branch: `pf_cam_v0` | Model: ResNet50 | Dataset: ImageNet (5000 images)*
