# PF-CAM — Pyramid Fusion Class Activation Mapping

A standalone multi-layer CAM aggregation method that combines saliency maps from multiple convolutional layers using hierarchical top-down fusion with soft gating.

**Zero dependency** on `GradCAMEnhanced` or `OptimizedCamExtractor` — uses standard [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) directly.

---

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Standard pytorch-grad-cam GradCAM               │
│ (ActivationsAndGradients hooks on ALL conv layers)│
│                                                   │
│ Single forward + backward pass                   │
│    → Per-layer activations A_l  (1, C, H, W)    │
│    → Per-layer gradients   G_l  (1, C, H, W)    │
│    → Per-layer CAM maps    M_l  (H_out, W_out)  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Activation Masking (per layer)                   │
│                                                   │
│ 1. weights = mean(G_l, spatial_dims)             │
│ 2. weighted_act = weights * A_l                  │
│ 3. weighted_act = ReLU(weighted_act)    ← Fix #2│
│ 4. masked_act = Normalize(weighted_act) ⊙ A_l   │
│                  ↑ configurable strategy  ← Fix #3│
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Layer Scoring (per layer, sequential)    ← Fix #1│
│                                                   │
│ For each layer l:                                │
│   1. Hook: replace A_l → masked_A_l             │
│   2. Forward pass → modified logits              │
│   3. score_l = CosSim(softmax(orig), softmax(mod))│
│                                                   │
│ → scores: [s_1, s_2, ..., s_L]                  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ PF-CAM Aggregation                               │
│                                                   │
│ Stage 1: Group by Resolution                     │
│   layers → {56×56: [l1..l4], 14×14: [l5..l7],  │
│             7×7: [l8..l10]}                      │
│                                                   │
│ Stage 2: Intra-Stage Top-K Selection             │
│   Keep top k% layers per stage (min k_min)       │
│   Weighted sum within stage (temp-scaled softmax) │
│                                                   │
│ Stage 3: Top-Down Soft Gating                    │
│   fused = deepest_stage_cam                      │
│   for each shallower stage:                      │
│     gate = normalize(fused)                      │
│     soft_gate = β + (1 - β) × gate              │
│     fused = fused + shallow_cam × soft_gate      │
│     fused = normalize(fused)                     │
│                                                   │
│ → Final saliency map (H_out × W_out)            │
└─────────────────────────────────────────────────┘
```

### Soft Gating Explained

The `β` parameter controls how much detail from shallower (higher-resolution) layers passes through:

| β value | Effect |
|:--------|:-------|
| `0.0` | Only regions highlighted by deeper layers are visible from shallower layers |
| `0.3` | 30% baseline + 70% gated by deeper layers (recommended for ImageNet) |
| `0.5` | 50/50 split — balanced fusion |
| `1.0` | All shallow detail passes through unfiltered |

---

## Normalization Strategies

Set via `--norm-strategy` argument.

| Strategy | Flag | Description |
|:---------|:-----|:------------|
| **Gradient Weighted** *(default)* | `gradient_weighted` | Per-channel spatial [0,1] normalization, then scale by gradient magnitude. Preserves channel independence while weighting by importance. |
| Channel Spatial | `channel_spatial` | Per-channel spatial [0,1]. All channels weighted equally. |
| Global | `global` | Global [0,1] across all channels. Preserves magnitude ratios. |
| L2 Channel | `l2_channel` | Unit L2 norm per channel. Preserves spatial patterns. |

---

## Files

| File | Description |
|:-----|:------------|
| `extractor.py` | Core CAM extraction + layer scoring using standard GradCAM |
| `aggregator.py` | PF-CAM pyramid fusion + standard/temperature/top-k aggregation |
| `normalization.py` | Four normalization strategy implementations |
| `weight_logger.py` | Per-image layer/stage weight diagnostics (CSV + JSON) |
| `run_experiment.py` | CLI experiment runner |
| `test_pf_cam.py` | Unit tests (18 tests) |
| `calibrate.py` | Hyperparameter grid search for optimal β, k%, and temperature |
| `visualize_results.py` | Plotting suite for method comparison and weight analysis |

---

## Usage

### Quick Start

```bash
cd /path/to/pytorch-grad-cam
source .venv_new/bin/activate  # or your virtualenv

# Run on 5 images (CPU, quick test)
python XAI_Enhancer_module/pf_cam/run_experiment.py \
    --model resnet50 \
    --count 5 \
    --device cpu \
    --log-weights
```

### Full Evaluation

```bash
# Full run with weight logging and standard method comparison
python XAI_Enhancer_module/pf_cam/run_experiment.py \
    --model resnet50 \
    --imagenet-path /path/to/imagenet/val \
    --count 500 \
    --device cuda \
    --beta 0.3 \
    --k-percent 0.1 \
    --temp 0.05 \
    --norm-strategy gradient_weighted \
    --log-weights \
    --compare-standard \
    --output-dir pf_cam_results
```

### Hyperparameter Calibration (Grid Search)

Systematically search for the best `beta`, `k_percent`, and `temp` values:

```bash
python XAI_Enhancer_module/pf_cam/calibrate.py \
    --model resnet50 \
    --count 50 \
    --device cuda \
    --output-dir calibration_results
```
*Supports resuming if interrupted.*

### Result Visualization

Generate plots from your results:

```bash
python XAI_Enhancer_module/pf_cam/visualize_results.py \
    --results-dir pf_cam_results \
    --comparison-file comparison_results.csv
```
Generates:
- `method_comparison.png` (Insertion/Deletion/ROAD bar charts)
- `stage_weights_violin.png` (Distribution of weights per stage)
- `stage_selection_freq.png` (How often layers in each stage are selected)
- `layer_impact_mean.png` (Mean impact of each individual layer)

### Batch Processing (Large Datasets)

```bash
# Process images 0-999
python XAI_Enhancer_module/pf_cam/run_experiment.py \
    --model resnet50 --count 5000 --start 0 --end 1000 \
    --device cuda --output-dir results_batch_0

# Process images 1000-1999
python XAI_Enhancer_module/pf_cam/run_experiment.py \
    --model resnet50 --count 5000 --start 1000 --end 2000 \
    --device cuda --output-dir results_batch_1
```

### Unit Tests

```bash
python -m pytest XAI_Enhancer_module/pf_cam/test_pf_cam.py -v
```

---

## CLI Arguments

### Model & Data

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--model` | `resnet50` | Pre-trained model (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `densenet121`, `efficientnet_b0`) |
| `--model-cache-dir` | `../pytorch_models/` | Directory for cached model weights |
| `--imagenet-path` | `imagenet_val_images/` | Path to ImageNet validation images |
| `--count` | `50` | Number of images to evaluate |
| `--start` | `0` | Start index for batch processing |
| `--end` | `None` | End index for batch processing |

### PF-CAM Hyperparameters

| Argument | Default | Range | Description |
|:---------|:--------|:------|:------------|
| `--beta` | `0.3` | `0.0 – 1.0` | Soft gating strength. Lower = more suppression of shallow layers |
| `--k-percent` | `0.1` | `0.05 – 1.0` | Fraction of layers to keep per stage |
| `--k-min` | `2` | `1 – N` | Minimum layers to keep per stage |
| `--temp` | `0.05` | `0.01 – 2.0` | Temperature for softmax sharpening. Lower = sharper weights |

### Normalization & Logging

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--norm-strategy` | `gradient_weighted` | Normalization strategy (see table above) |
| `--log-weights` | `false` | Enable per-image weight logging to CSV/JSON |
| `--output-dir` | `pf_cam_results/` | Output directory for results and logs |

### Evaluation

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--step-size` | `50` | Step size for insertion/deletion AUC curves |
| `--batch-size` | `64` | Batch size for evaluation model inference |
| `--compare-standard` | `false` | Also run GradCAM, GradCAM++, HiResCAM for comparison |
| `--device` | `auto` | Device (`auto`, `cuda`, `mps`, `cpu`) |
| `--layer-mode` | `all` | Which conv layers to use (always forced to `all` for PF-CAM) |

---

## Output Files

When `--log-weights` is enabled:

| File | Format | Content |
|:-----|:-------|:--------|
| `pf_cam_weight_log.csv` | Flat CSV | One row per layer per image: layer name, raw score, softmax weight, stage ID, stage score, stage weight, top-K selected |
| `pf_cam_weight_log.json` | Structured JSON | One entry per image with nested layer and stage arrays |
| `pf_cam_results.csv` | CSV | Aggregated metrics (insertion AUC, deletion AUC, ROAD scores) |
| `comparison_results.csv` | CSV | Standard method comparison results (if `--compare-standard`) |

---

## Recommended Hyperparameters

| Task Type | β | k_percent | temp | norm_strategy |
|:----------|:--|:----------|:-----|:--------------|
| **ImageNet (semantic)** | 0.2 – 0.3 | 0.1 | 0.05 | `gradient_weighted` |
| **Medical imaging (texture)** | 0.4 – 0.6 | 0.2 – 0.3 | 0.5 – 1.0 | `gradient_weighted` |

- **Semantic tasks**: Deep layers carry most signal → low β suppresses noisy shallow details
- **Texture tasks**: Early/mid layers carry useful signal → higher β, higher k%, warmer temperature

---

## Issues Fixed (vs Original XAI-Enhancer)

1. **Shape mismatch** — Sequential per-layer forward passes with explicit shape checks (no silent `except` failures)
2. **Negative gradients** — ReLU applied before normalization/masking
3. **Channel normalization** — Gradient-weighted hybrid preserves channel independence
4. **Cosine similarity** — Computed on softmax probabilities, not raw logits
