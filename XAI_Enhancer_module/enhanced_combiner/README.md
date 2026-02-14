
# Enhanced Combiner Module

This module implements enhanced aggregation strategies for XAI CAM generation.

## Components

- `aggregator.py`: Core logic for combining CAMs.
    - **Stagewise**: Groups layers by resolution/stage before aggregation.
    - **Top-K**: Keeps only the top-K highest scoring layers.
    - **Pyramid Fusion**: Hierarchical top-down fusion with **Soft Gating** and stage-wise Top-K.
    - **Temperature**: Applies temperature scaling to the softmax weights.
- `extractor_v2.py`: Wraps `OptimizedCamExtractor` to use the new aggregator.
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
```

### Methods

- `standard`: Original Cosine Similarity + Softmax.
- `stagewise`: Hierarchical aggregation by stage.
- `topk`: Top-K layer selection.
- `temp`: Temperature scaling on softmax.
- `pyramid`: **(New)** State-of-the-art hierarchical fusion.
    - **Top-K Selection**: Selects best layers per stage (controlled by `--k-percent`).
    - **Top-Down Soft Gating**: Uses deeper, semantic layers to guide the activation of shallower layers.

## Pyramid Fusion Details (PF-CAM)

The Pyramid Fusion method operates on the principle that deep layers contain semantic information (what/where roughly) while shallow layers contain fine details (edges/boundaries).

**Soft Gating Mechanism**:
Unlike strict multiplication (Masking), Soft Gating allows a fraction of the shallow layer's detailed information to pass through even if the deep layer's activation is low. This prevents the map from becoming too concentrated and improves coverage (Insertion Score).

**Formula**:
```python
# Gate comes from the Deep Layer (fused_cam)
soft_gate = beta + (1.0 - beta) * gate
# Apply to Shallow Layer (next_stage_cam)
masked_details = next_stage_cam * soft_gate
# Add to accumulation
fused_cam = fused_cam + masked_details
```

## Hyperparameter Analysis

| Parameter | Meaning | Effect on Output | Recommended Range |
| :--- | :--- | :--- | :--- |
| **beta** | **Soft Gating Factor** | Controls how much "unguided" detail from shallow layers is included.<br>**Higher Beta (e.g., 0.6)**: More detailed, larger coverage. Improves Insertion, hurts Deletion (more noise).<br>**Lower Beta (e.g., 0.2)**: Stricter masking. Clean, focused maps. Improves Deletion, hurts Insertion. | `0.3` - `0.5` (Default: `0.4`) |
| **k-percent** | **Top-K Selection** | Percentage of layers to keep per stage.<br>**Higher**: Smoother maps, more context.<br>**Lower**: Sharper maps, focuses on most active filters. | `0.1` - `0.2` (10-20%) |
| **temp** | **Softmax Temperature** | Controls the sharpness of layer weighting.<br>**Lower (<1.0)**: Only the absolute best layers contribute.<br>**Higher (>1.0)**: Averages more layers together. | `0.1` (Sharp validation) |
