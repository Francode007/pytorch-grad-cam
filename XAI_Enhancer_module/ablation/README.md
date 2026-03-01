# Layer-wise Ablation Study Module

This module provides two experiment suites for the XAI-Enhancer research:

**Part I -- ROAD Ablation Study:** Layer-wise ROAD metric extraction
across five architectures (VGG-16, VGG-19, ResNet-18, ResNet-34,
ResNet-50) on Kvasir-v2 and IBS, with publication figures and a
bottleneck heatmap comparison.

**Part II -- Enhancer Weight Analysis:** Extraction and visualisation of
the logit-similarity weights that the XAI-Enhancer assigns to each
layer, plus a Spearman correlation proof linking those weights to ROAD
scores.

## Prerequisites

| Requirement | Purpose |
|---|---|
| Python 3.9+ | `list[...]` type hints (PEP 585) |
| PyTorch with CUDA | GPU-accelerated inference |
| pytorch-grad-cam | GradCAM saliency extraction (already in this repo) |
| pandas, numpy | Data wrangling and CSV export |
| matplotlib, seaborn | Publication-quality plots |
| scipy | Spearman correlation (weight analysis) |
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
├── README.md                            # this file
│
│   Part I – ROAD Ablation Study
├── layerwise_road_extraction.py         # Step 1 – ROAD metric extraction
├── plot_layerwise_results.py            # Step 2 – publication figures
├── bottleneck_visualization.py          # Step 3 – side-by-side heatmaps
│
│   Part II – Enhancer Weight Analysis (softmax only)
├── enhancer_weight_extraction.py        # Step 4 – weight extraction pipeline
├── plot_enhancer_weights.py             # Step 5 – adaptive-shift bar charts
├── weight_road_correlation.py           # Step 6 – Spearman correlation proof
│
│   Part III – Raw vs Softmax Analysis
├── raw_softmax_extraction.py            # Step 7 – raw + softmax extraction
├── plot_raw_scores.py                   # Step 8 – bottleneck & diffusion plots
└── regularization_proof.py              # Step 9 – variance reduction table
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

---

# Part II -- Enhancer Weight Analysis

These scripts analyse the logit-similarity weights that the
XAI-Enhancer module assigns to each of the last 5 Conv2d layers.
The weights are derived by running **HiResCAMEnhanced**, injecting
per-layer masked activations, computing cosine similarity between
original and modified logits, and applying softmax.

## Step 4 -- Extract Enhancer weights

### Minimal example

```bash
python -m XAI_Enhancer_module.ablation.enhancer_weight_extraction \
    --kvasir-data-root data/kvasir-v2 \
    --ibs-data-root data/IBS-preprocessed-dataset \
    --checkpoints kvasir:resnet50:./kvasir_runs/resnet50/best.pth \
    --output-csv runs/ablation/enhancer_weights.csv \
    --max-images 50 \
    --device cuda
```

### Full run (VGG-16 + ResNet-50, both datasets)

```bash
python -m XAI_Enhancer_module.ablation.enhancer_weight_extraction \
    --kvasir-data-root data/kvasir-v2 \
    --ibs-data-root data/IBS-preprocessed-dataset \
    --checkpoints \
        kvasir:vgg16:./kvasir_runs/vgg16/best.pth \
        kvasir:resnet50:./kvasir_runs/resnet50/best.pth \
        ibs:vgg16:./ibs_runs/vgg16/best.pth \
        ibs:resnet50:./ibs_runs/resnet50/best.pth \
    --output-csv runs/ablation/enhancer_weights.csv \
    --device cuda
```

### Output

A per-image CSV with columns:

| Dataset | Model | Layer_5_Weight | Layer_4_Weight | Layer_3_Weight | Layer_2_Weight | Layer_1_Weight |
|---|---|---|---|---|---|---|
| kvasir | resnet50 | 0.1821 | 0.2034 | 0.2105 | 0.1987 | 0.2053 |
| ... | ... | ... | ... | ... | ... | ... |

`Layer_5_Weight` corresponds to Conv2d index -5 (deepest of the five),
`Layer_1_Weight` to index -1 (shallowest / last).

## Step 5 -- Adaptive-shift visualisation

```bash
python -m XAI_Enhancer_module.ablation.plot_enhancer_weights \
    --input-csv runs/ablation/enhancer_weights.csv \
    --output-dir runs/ablation/figures
```

This creates four files:

| File | Description |
|---|---|
| `weight_plot_a_cross_dataset.png` | ResNet-50 weights: IBS vs Kvasir-v2 |
| `weight_plot_a_cross_dataset.pdf` | Same, vector format |
| `weight_plot_b_cross_architecture.png` | VGG-16 vs ResNet-50 on Kvasir-v2 |
| `weight_plot_b_cross_architecture.pdf` | Same, vector format |

Both plots are grouped bar charts with error bars (std) showing how
the Enhancer re-distributes attention across layers depending on the
dataset or architecture.

## Step 6 -- ROAD-weight Spearman correlation

This script proves that the Enhancer weights are correlated with
the actual faithfulness (ROAD) of each layer's explanation.

```bash
python -m XAI_Enhancer_module.ablation.weight_road_correlation \
    --kvasir-data-root data/kvasir-v2 \
    --checkpoint ./kvasir_runs/resnet50/best.pth \
    --num-images 100 \
    --output-csv runs/ablation/weight_road_corr.csv \
    --device cuda
```

For each of the 100 randomly sampled images (seed=42), the script:
1. Computes standalone ROAD scores for layers -5 through -1.
2. Extracts the Enhancer weights for those same layers.
3. Computes Spearman rank correlation between the two 5-element arrays.

Output is printed to the console and saved to a per-image CSV:

| image | label | spearman_rho | p_value | road_5 | ... | weight_1 |
|---|---|---|---|---|---|---|

The final line reports the **average Spearman rho** across all valid
samples.

### Programmatic usage

```python
from XAI_Enhancer_module.ablation.enhancer_weight_extraction import get_enhancer_weights
from XAI_Enhancer_module.kvasir.models import build_kvasir_model, load_kvasir_checkpoint
from XAI_Enhancer_module.kvasir.data import get_val_transforms, KVASIR_NUM_CLASSES
from PIL import Image
import torch, torch.nn as nn

device = torch.device("cuda")
model = build_kvasir_model("resnet50", num_classes=KVASIR_NUM_CLASSES, pretrained=False)
load_kvasir_checkpoint(model, "kvasir_runs/resnet50/best.pth", device)
model.to(device).eval()

layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-5:]
transform = get_val_transforms()
img = transform(Image.open("my_image.jpg").convert("RGB")).to(device)

weights = get_enhancer_weights(img, model, layers, device, predicted_label=3)
print(weights)  # array of 5 floats summing to ~1.0
```

---

# Part III -- Raw vs Softmax Analysis

These scripts extract **both** the pre-softmax raw cosine similarities
(alpha\_l) and the post-softmax weights, enabling analysis of the
softmax regularisation effect and architectural bottleneck behaviour.

## Step 7 -- Extract raw + softmax scores

Runs across all 5 architectures and both datasets.  Each image
produces **two** rows in the CSV (one Raw, one Softmax).

```bash
python -m XAI_Enhancer_module.ablation.raw_softmax_extraction \
    --kvasir-data-root data/kvasir-v2 \
    --ibs-data-root data/IBS-preprocessed-dataset \
    --checkpoints \
        kvasir:vgg16:./kvasir_runs/vgg16/best.pth \
        kvasir:vgg19:./kvasir_runs/vgg19/best.pth \
        kvasir:resnet18:./kvasir_runs/resnet18/best.pth \
        kvasir:resnet34:./kvasir_runs/resnet34/best.pth \
        kvasir:resnet50:./kvasir_runs/resnet50/best.pth \
        ibs:vgg16:./ibs_runs/vgg16/best.pth \
        ibs:vgg19:./ibs_runs/vgg19/best.pth \
        ibs:resnet18:./ibs_runs/resnet18/best.pth \
        ibs:resnet34:./ibs_runs/resnet34/best.pth \
        ibs:resnet50:./ibs_runs/resnet50/best.pth \
    --output-csv runs/ablation/raw_softmax_scores.csv \
    --device cuda
```

### Output CSV

| Dataset | Model | Metric_Type | Layer_5_Val | Layer_4_Val | Layer_3_Val | Layer_2_Val | Layer_1_Val |
|---|---|---|---|---|---|---|---|
| kvasir | resnet50 | Raw | 0.9821 | 0.9734 | 0.9905 | 0.9887 | 0.9853 |
| kvasir | resnet50 | Softmax | 0.1921 | 0.1834 | 0.2105 | 0.2087 | 0.2053 |

The script supports crash recovery: it reads any existing CSV on
restart and skips already-completed (dataset, model) pairs.

## Step 8 -- Bottleneck and diffusion plots

```bash
python -m XAI_Enhancer_module.ablation.plot_raw_scores \
    --input-csv runs/ablation/raw_softmax_scores.csv \
    --output-dir runs/ablation/figures
```

Creates four files:

| File | Description |
|---|---|
| `raw_plot_a_bottleneck_proof.png/pdf` | ResNet-18 vs -34 vs -50 raw scores on IBS (with error bands) |
| `raw_plot_b_information_diffusion.png/pdf` | VGG-16 vs ResNet-50 raw scores on Kvasir-v2 (with error bands) |

Plot A demonstrates how deeper ResNet variants exhibit different
raw similarity profiles across layers (the bottleneck effect).
Plot B contrasts the flat VGG profile against the structured ResNet
profile (information diffusion patterns).

## Step 9 -- Regularisation proof (variance table)

```bash
python -m XAI_Enhancer_module.ablation.regularization_proof \
    --input-csv runs/ablation/raw_softmax_scores.csv \
    --output-txt runs/ablation/regularization_table.md
```

Prints a Markdown table to the console (and optionally saves it):

| Model | Raw Avg Std | Softmax Avg Std | Reduction (%) | Raw Median Std | Softmax Median Std |
|---|---|---|---|---|---|
| VGG-16 | 0.0123 | 0.0045 | 63.41 | 0.0118 | 0.0042 |
| ResNet-50 | 0.0098 | 0.0031 | 68.37 | 0.0091 | 0.0028 |

The **Reduction (%)** column quantifies how much the softmax
operation compresses the layer-wise variance, proving its
regularising effect on the weight distribution.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: splits/val.txt` | Run `prepare_splits()` as shown above |
| `CUDA out of memory` | Reduce `--max-images` or use `--device cpu` |
| `No checkpoint provided for ...` | Add the missing `dataset:arch:path` to `--checkpoints` |
| Plot script says "No Kvasir data" | Ensure the CSV has rows where `Dataset == "kvasir"` |
| `ModuleNotFoundError` | Run from the project root so `sys.path` resolves correctly |
| `No module named 'scipy'` | `pip install scipy` in your environment |
