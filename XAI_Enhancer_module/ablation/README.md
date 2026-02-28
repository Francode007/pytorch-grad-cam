# Layer-wise Ablation Study Module

This module contains three scripts for running a layer-wise ROAD ablation
study across five architectures (VGG-16, VGG-19, ResNet-18, ResNet-34,
ResNet-50) on the Kvasir-v2 and IBS datasets.

## Prerequisites

| Requirement | Purpose |
|---|---|
| Python 3.10+ | `list[...]` type hints used throughout |
| PyTorch with CUDA | GPU-accelerated inference |
| pytorch-grad-cam | GradCAM saliency extraction (already in this repo) |
| pandas, numpy | Data wrangling and CSV export |
| matplotlib, seaborn | Publication-quality plots |
| tqdm | Progress bars |

All dependencies are already satisfied by the project's existing
environment.  Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Directory layout

```
XAI_Enhancer_module/ablation/
├── __init__.py
├── README.md                          # this file
├── layerwise_road_extraction.py       # Task 1 – ROAD metric extraction
├── plot_layerwise_results.py          # Task 2 – publication figures
└── bottleneck_visualization.py        # Task 3 – side-by-side heatmaps
```

## Before you start

Make sure:

1. **Datasets** are downloaded and split files exist:
   - Kvasir-v2 at `data/kvasir-v2/` (with `splits/val.txt`)
   - IBS at `data/IBS-preprocessed-dataset/` (with `splits/val.txt`)

   If split files don't exist yet, generate them:
   ```bash
   python -c "from XAI_Enhancer_module.kvasir.data import prepare_splits; prepare_splits('data/kvasir-v2')"
   python -c "from XAI_Enhancer_module.ibs.data   import prepare_splits; prepare_splits('data/IBS-preprocessed-dataset')"
   ```

2. **Trained checkpoints** are available for every (dataset, architecture)
   combination you want to evaluate.  The extraction script accepts these
   as `dataset:arch:path` triples so you only run the combos you have
   weights for.

## Step 1 – Extract layer-wise ROAD scores

Run everything from the **project root** (`pytorch-grad-cam/`).

### Minimal example (one model, one dataset)

```bash
python -m XAI_Enhancer_module.ablation.layerwise_road_extraction \
    --kvasir-data-root data/kvasir-v2 \
    --ibs-data-root data/IBS-preprocessed-dataset \
    --checkpoints kvasir:resnet50:/path/to/kvasir_resnet50.pth \
    --output-csv runs/ablation/layerwise_road.csv \
    --max-images 50 \
    --device cuda
```

### Full sweep (all 5 models, both datasets)

```bash
python -m XAI_Enhancer_module.ablation.layerwise_road_extraction \
    --kvasir-data-root data/kvasir-v2 \
    --ibs-data-root data/IBS-preprocessed-dataset \
    --checkpoints \
        kvasir:vgg16:/weights/kvasir_vgg16.pth \
        kvasir:vgg19:/weights/kvasir_vgg19.pth \
        kvasir:resnet18:/weights/kvasir_resnet18.pth \
        kvasir:resnet34:/weights/kvasir_resnet34.pth \
        kvasir:resnet50:/weights/kvasir_resnet50.pth \
        ibs:vgg16:/weights/ibs_vgg16.pth \
        ibs:vgg19:/weights/ibs_vgg19.pth \
        ibs:resnet18:/weights/ibs_resnet18.pth \
        ibs:resnet34:/weights/ibs_resnet34.pth \
        ibs:resnet50:/weights/ibs_resnet50.pth \
    --output-csv runs/ablation/layerwise_road.csv \
    --device cuda
```

> **Tip:** Use `--max-images 100` for a quick sanity-check run before
> committing to the full validation set.  Remove the flag (or pass `-1`)
> to evaluate all images.

### Output

A CSV file at the path given by `--output-csv` with columns:

| Dataset | Model | Layer_Index | ROAD_Score |
|---|---|---|---|
| kvasir | resnet50 | -5 | 0.0342 |
| kvasir | resnet50 | -4 | 0.0510 |
| ... | ... | ... | ... |

`Layer_Index` uses negative indexing: `-1` is the last Conv2d layer,
`-5` is the fifth-from-last.

### Resource estimates

| Component | GPU VRAM | Wall-time |
|---|---|---|
| VGG-16/19 per image | ~2 GB | ~0.3 s |
| ResNet-18/34 per image | ~1.5 GB | ~0.2 s |
| ResNet-50 per image | ~2 GB | ~0.25 s |

For 100 images x 5 layers x 5 models x 2 datasets, expect roughly
30-45 minutes on a single GPU.

## Step 2 – Generate publication figures

```bash
python -m XAI_Enhancer_module.ablation.plot_layerwise_results \
    --input-csv runs/ablation/layerwise_road.csv \
    --output-dir runs/ablation/figures
```

This creates four files:

| File | Description |
|---|---|
| `plot_a_architectural_contrast.png` | All 5 models on Kvasir-v2 |
| `plot_a_architectural_contrast.pdf` | Same, vector format for LaTeX |
| `plot_b_cross_dataset.png` | ResNet-50 on IBS vs Kvasir-v2 |
| `plot_b_cross_dataset.pdf` | Same, vector format |

All figures are saved at 300 DPI with a clean serif-font academic
aesthetic (seaborn `whitegrid` + `paper` context).

## Step 3 – Bottleneck heatmap comparison

Visualise the raw GradCAM activation from the -4 Conv2d layer of both
ResNet-34 and ResNet-50 for a single image:

```bash
python -m XAI_Enhancer_module.ablation.bottleneck_visualization \
    --image-path data/kvasir-v2/polyps/cju0s2wabakbh0878vhdq59tn.jpg \
    --dataset kvasir \
    --resnet34-ckpt /weights/kvasir_resnet34.pth \
    --resnet50-ckpt /weights/kvasir_resnet50.pth \
    --output-path runs/ablation/bottleneck_vis.png \
    --device cuda
```

The output is a three-panel figure:

```
┌───────────────┬──────────────────┬──────────────────┐
│ Original      │ ResNet-34 (jet)  │ ResNet-50 (jet)  │
│ Image         │ GradCAM @ -4     │ GradCAM @ -4     │
└───────────────┴──────────────────┴──────────────────┘
```

### Programmatic usage

You can also call the function directly from Python (e.g. in a notebook):

```python
from XAI_Enhancer_module.ablation.bottleneck_visualization import visualize_bottleneck
from XAI_Enhancer_module.kvasir.data import get_val_transforms
from PIL import Image

transform = get_val_transforms()
image_tensor = transform(Image.open("my_image.jpg").convert("RGB"))

fig = visualize_bottleneck(
    image_tensor=image_tensor,
    resnet34_ckpt="/weights/kvasir_resnet34.pth",
    resnet50_ckpt="/weights/kvasir_resnet50.pth",
    dataset="kvasir",
    device="cuda",
    output_path="bottleneck.png",   # optional, omit to just display
)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: splits/val.txt` | Run `prepare_splits()` as shown above |
| `CUDA out of memory` | Reduce `--max-images` or use `--device cpu` |
| `No checkpoint provided for ...` | Add the missing `dataset:arch:path` to `--checkpoints` |
| Plot script says "No Kvasir data" | Ensure the CSV has rows where `Dataset == "kvasir"` |
| `ModuleNotFoundError` | Run from the project root so `sys.path` resolves correctly |
