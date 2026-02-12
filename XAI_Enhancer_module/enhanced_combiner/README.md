
# Enhanced Combiner Module

This module implements enhanced aggregation strategies for XAI CAM generation.

## Components

- `aggregator.py`: Core logic for combining CAMs.
    - **Stagewise**: Groups layers by resolution/stage before aggregation.
    - **Top-K**: Keeps only the top-K highest scoring layers.
    - **Temperature**: Applies temperature scaling to the softmax weights.
- `extractor_v2.py`: Wraps `OptimizedCamExtractor` to use the new aggregator.
- `run_experiment.py`: Script to run comparisons on ImageNet samples.

## Usage

### Running Experiments

```bash
python run_experiment.py --model resnet50 --method stagewise --count 50
python run_experiment.py --model resnet50 --method topk --k 5 --soft 1 --count 50
```

### Methods

- `standard`: Original Cosine Similarity + Softmax.
- `stagewise`: Hierarchical aggregation by stage.
- `topk`: Top-K layer selection.
- `temp`: Temperature scaling on softmax.
