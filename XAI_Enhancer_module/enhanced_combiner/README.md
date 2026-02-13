
# Enhanced Combiner Module

This module implements enhanced aggregation strategies for XAI CAM generation.

## Components

- `aggregator.py`: Core logic for combining CAMs.
    - **Stagewise**: Groups layers by resolution/stage before aggregation.
    - **Top-K**: Keeps only the top-K highest scoring layers.
    - **Pyramid Fusion**: Hierarchical top-down fusion with gating and stage-wise Top-K.
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
    - **Top-Down Gating**: Uses deeper, semantic layers to gate/filter noise from shallower, high-resolution layers.
