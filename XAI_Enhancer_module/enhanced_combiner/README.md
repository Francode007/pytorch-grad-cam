
# Enhanced Combiner Module

This module implements enhanced aggregation strategies for XAI CAM generation.

## Components

- `aggregator.py`: Core logic for combining CAMs.
    - **Stagewise**: Groups layers by resolution/stage before aggregation.
    - **Top-K**: Keeps only the top-K highest scoring layers.
    - **Pyramid Fusion**: Hierarchical top-down fusion with **Soft Gating** and stage-wise Top-K.
    - **Temperature**: Applies temperature scaling to the softmax weights.
- `extractor_v2.py`: Wraps `OptimizedCamExtractor` to use the new aggregator. Supports batch processing.
- `run_experiment.py`: Script to run comparisons on ImageNet samples.

## Usage

### Running Experiments

```bash
# Standard Run
python run_experiment.py --model resnet50 --method pyramid --count 50

# Comparison Mode (Pyramid vs GradCAM vs GradCAM++ vs HiResCAM)
python run_experiment.py --model resnet50 --method pyramid --compare --count 50

# Custom Pyramid Configuration
python run_experiment.py --model resnet50 --method pyramid --k-percent 0.2 --count 50

# Full 5000-Image Experiment with Comparison
python run_experiment.py \
  --model resnet50 \
  --method pyramid \
  --layer-mode all \
  --count 5000 \
  --gpu-batch-size 64 \
  --batch-size 1000 \
  --step-size 224 \
  --compare \
  --output-dir pyramid_5000_results \
  --images-path /path/to/imagenet_val_sample
```

### Methods

- `standard`: Original Cosine Similarity + Softmax.
- `stagewise`: Hierarchical aggregation by stage.
- `topk`: Top-K layer selection.
- `temp`: Temperature scaling on softmax.
- `pyramid`: **(New)** State-of-the-art hierarchical fusion.
    - **Top-K Selection**: Selects best layers per stage (controlled by `--k-percent`).
    - **Top-Down Soft Gating**: Uses deeper, semantic layers to guide the activation of shallower layers.

---

## Pyramid Fusion Details (PF-CAM)

The Pyramid Fusion method operates on the principle that deep layers contain semantic information (what/where roughly) while shallow layers contain fine details (edges/boundaries).

### Data Flow

For ResNet50 with `--layer-mode all`, all **53 convolutional layers** are used:

```
run_experiment.py (--method pyramid --layer-mode all)
  → ImageNetProperAUCEvaluator (loads 53 conv layers)
    → DataLoader (batch_size=32)
      → extract_enhanced_cam() — passes [B,C,H,W] tensor
        → EnhancedExtractorV2.extract_saliency_map()
          1. get_actual_output_batch → [B, Classes]
          2. cam_method() → CAMs per layer [B, H, W] × 53 layers
          3. compute_modified_outputs_batch (layer batching, 16 at a time)
          4. compute_cosine_similarities → [53, B] scores
          5. aggregate_hybrid(type=pyramid) — per-image aggregation loop
            → aggregate_pyramid_fusion()
              → Group 53 layers → 5 stages by resolution
              → Top-K selection per stage
              → Top-down soft gating fusion
              → Final CAM [B, H, W]
```

### Layer Stages (ResNet50)

| Stage | Resolution | Approx. Layers | Role |
|-------|-----------|----------------|------|
| C1 | 112×112 | ~3 | Edge/texture features |
| C2 | 56×56 | ~12 | Low-level patterns |
| C3 | 28×28 | ~16 | Mid-level features |
| C4 | 14×14 | ~18 | Semantic features |
| C5 | 7×7 | ~4 | High-level abstractions |

> **Note:** The pyramid method **requires** `--layer-mode all` (or `last_5`). The script auto-switches from `last` to `all` if you use `--method pyramid`.

### Soft Gating Mechanism

Unlike strict multiplication (Masking), Soft Gating allows a fraction of the shallow layer's detailed information to pass through even if the deep layer's activation is low.

```python
# Gate comes from the Deep Layer (fused_cam)
soft_gate = beta + (1.0 - beta) * gate
# Apply to Shallow Layer (next_stage_cam)
masked_details = next_stage_cam * soft_gate
# Add to accumulation
fused_cam = fused_cam + masked_details
```

---

## Evaluation Metrics

All metrics use a **clean model** (without CAM hooks) for unbiased evaluation:

| Metric | Method | What It Measures |
|--------|--------|-----------------|
| Insertion AUC | Progressively reveal pixels by importance | How quickly the model recovers confidence (higher = better) |
| Deletion AUC | Progressively remove pixels by importance | How quickly confidence drops (lower = better) |
| ROAD | Remove top-p% pixels, replace with blur | Confidence drop from removing important regions |

The `--compare` flag runs the enhanced method against standard GradCAM, GradCAM++, and HiResCAM and outputs a side-by-side CSV (`comparison_report.csv`).

---

## GPU Utilization Parameters

| Parameter | CLI Flag | Default | Effect |
|-----------|---------|---------|--------|
| **GPU Batch Size** | `--gpu-batch-size` | 64 | Images processed simultaneously for metric evaluation |
| **DataLoader Batch** | Hardcoded | 32 | Images loaded and sent to CAM extraction at once |
| **Layer Batch Size** | Hardcoded | 16 | Layers processed simultaneously per forward pass. Higher = more GPU memory, faster processing |

For A100 40GB with ResNet50, `layer_batch_size=16` is optimal. Estimated speed: **~1-1.5s/image** for pyramid with 53 layers.

---

## Hyperparameter Analysis

| Parameter | Meaning | Effect on Output | Recommended Range |
| :--- | :--- | :--- | :--- |
| **beta** | **Soft Gating Factor** | Controls how much "unguided" detail from shallow layers is included.<br>**Higher Beta (e.g., 0.6)**: More detailed, larger coverage. Improves Insertion, hurts Deletion (more noise).<br>**Lower Beta (e.g., 0.2)**: Stricter masking. Clean, focused maps. Improves Deletion, hurts Insertion. | `0.3` - `0.5` (Default: `0.4`) |
| **k-percent** | **Top-K Selection** | Percentage of layers to keep per stage.<br>**Higher**: Smoother maps, more context.<br>**Lower**: Sharper maps, focuses on most active filters. | `0.1` - `0.2` (10-20%) |
| **temp** | **Softmax Temperature** | Controls the sharpness of layer weighting.<br>**Lower (<1.0)**: Only the absolute best layers contribute.<br>**Higher (>1.0)**: Averages more layers together. | `0.1` (Sharp validation) |
