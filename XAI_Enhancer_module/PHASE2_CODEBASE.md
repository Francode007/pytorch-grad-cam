# Phase 2 codebase guide — training matrices & classifier metrics

Supplementary map of the **Phase 2** code added for the CMPB revision
(R3-1 patient/seed protocol, D-M5 val macro-F1 checkpoints).

Use this alongside:

- Operator checklist: [`MODAL_EXPERIMENT_PLAN.md`](MODAL_EXPERIMENT_PLAN.md)
- Short repro commands: [`REPRODUCE.md`](REPRODUCE.md)
- Locked decisions: [`../cmpb_revision/DECISIONS.md`](../cmpb_revision/DECISIONS.md)

---

## 1. What the matrix commands are

### Kvasir (recommended): parallel per seed

`train-kvasir-seed` fans out **one A100 per architecture** for a **single**
`--seed` (5 GPUs in parallel). Run seed 42 first, estimate cost, then 43 / 44.

```bash
# Survives Mac sleep / logout / closed terminal
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42

# Later:
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 43
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 44
```

| Modal CLI | Behaviour | Count per invocation |
|-----------|-----------|---------------------:|
| `train-kvasir-seed --seed S` | **5 A100s in parallel** (all arches, one seed) | **5** |
| `train-kvasir` | Single arch × seed (resume / debug) | **1** |
| `train-kvasir-matrix` | Legacy **sequential** arches×seeds on one GPU | up to **15** |

### IBS (unchanged for now)

| Modal CLI | Local equivalent | Cartesian product (defaults) | Count |
|-----------|------------------|------------------------------|------:|
| `train-ibs-matrix` | `python -m XAI_Enhancer_module.common.train_matrix --dataset ibs` | **5 arches × 5 folds** sequential | **25** |

### Default axes

| Axis | Values | Defined in |
|------|--------|------------|
| Architectures | `resnet18`, `resnet34`, `resnet50`, `densenet121`, `vgg16` | `modal_runner/config.py` → `KVASIR_ARCHS` / `IBS_ARCHS` |
| Kvasir seeds | `42`, `43`, `44` | `config.KVASIR_SEEDS`; pass one at a time via `--seed` |
| IBS folds | `0`, `1`, `2`, `3`, `4` | Patient CV from Phase 1B |
| IBS RNG seed | `42` (fixed; fold identity comes from `--fold`) | `ibs/train.py --seed` |

### Shared train hyperparameters (each cell)

Unless overridden on the CLI:

| Flag | Default (Modal Kvasir seed run) | Smoke (`--smoke`) |
|------|--------------------------------|-------------------|
| `--epochs` | `50` | `2` |
| `--batch-size` | `0` + **`--auto-batch-size`** (AdamW probe, cap by ≥20 steps/epoch) | `32` |

| `--device` | `cuda` | `cuda` |
| `--num-workers` / prefetch | `8` / `4` | same |
| `--amp` / `--amp-dtype` | on / `bfloat16` | on / `bfloat16` |
| `--compile` | on | on |
| Optimizer / LR | AdamW, `1e-4`, wd `1e-4`, cosine | same |

Checkpoint rule (D-M5): **best `val_f1_macro`** → `best.pth` (not accuracy).

Also written every run:

| File | Purpose |
|------|---------|
| `train.log` | Tee of stdout/stderr (always on volume) |
| `args.json` | CLI + **`batch_size_resolved` / `batch_size_locked`** after epoch 1 |
| `checkpoint_latest.pth` | Full trainer state each epoch (resume) |
| `checkpoint_mid.pth` | Full state at `epochs // 2` |
| `metrics.json` | Per-epoch metrics + optional `gpu_used_frac` |
| `best.pth` / `last.pth` | Best val F1 / final weights |

Shared across a seed wave:

| File | Purpose |
|------|---------|
| `/vol/runs/kvasir/locked_batch_sizes.json` | Arch → batch size (seed 43/44 reuse; no re-probe) |
| `/vol/runs/kvasir/waves/seed{S}/wave_summary.{json,txt}` | Cost + metrics table |
| `/vol/runs/kvasir/waves/seed{S}/wave.log` | Append-only wave completion log |

```bash
modal run -m modal_runner.app -- summarize-kvasir-seed --seed 42
```

---

## 2. Full expansion — Kvasir matrix (15 runs = 3 waves × 5 GPUs)

Preferred commands (one seed wave at a time):

```bash
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 43
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 44
```

Each cell invokes (batch size resolved by auto-probe unless overridden):

```text
python -m XAI_Enhancer_module.kvasir.train \
  --data-root /vol/data/kvasir-v2 \
  --arch <ARCH> --seed <SEED> --epochs 50 --batch-size 0 \
  --output-dir /vol/runs/kvasir --device cuda --num-workers 8 \
  --amp --amp-dtype bfloat16 --compile --a100 --auto-batch-size
```

| # | arch | seed | Output directory |
|--:|------|-----:|------------------|
| 1 | resnet18 | 42 | `/vol/runs/kvasir/resnet18/seed42/` |
| 2 | resnet18 | 43 | `/vol/runs/kvasir/resnet18/seed43/` |
| 3 | resnet18 | 44 | `/vol/runs/kvasir/resnet18/seed44/` |
| 4 | resnet34 | 42 | `/vol/runs/kvasir/resnet34/seed42/` |
| 5 | resnet34 | 43 | `/vol/runs/kvasir/resnet34/seed43/` |
| 6 | resnet34 | 44 | `/vol/runs/kvasir/resnet34/seed44/` |
| 7 | resnet50 | 42 | `/vol/runs/kvasir/resnet50/seed42/` |
| 8 | resnet50 | 43 | `/vol/runs/kvasir/resnet50/seed43/` |
| 9 | resnet50 | 44 | `/vol/runs/kvasir/resnet50/seed44/` |
| 10 | densenet121 | 42 | `/vol/runs/kvasir/densenet121/seed42/` |
| 11 | densenet121 | 43 | `/vol/runs/kvasir/densenet121/seed43/` |
| 12 | densenet121 | 44 | `/vol/runs/kvasir/densenet121/seed44/` |
| 13 | vgg16 | 42 | `/vol/runs/kvasir/vgg16/seed42/` |
| 14 | vgg16 | 43 | `/vol/runs/kvasir/vgg16/seed43/` |
| 15 | vgg16 | 44 | `/vol/runs/kvasir/vgg16/seed44/` |

**Data:** fixed Phase 1A 70/10/20 split under `/vol/data/kvasir-v2/splits/{train,val,test}.txt`.  
Changing `--seed` reseeds **training RNG** (init / augmentation / shuffle); it does **not** rebuild the split files.

**Per-run artifacts:**

```text
args.json          # full CLI + checkpoint_metric=val_f1_macro
metrics.json       # per-epoch train_loss, val_acc, val_f1_macro, val_auroc
best.pth           # best val_f1_macro
last.pth           # final epoch
```

---

## 3. Full expansion — IBS matrix (25 runs)

Command:

```bash
modal run -m modal_runner.app -- train-ibs-matrix --epochs 50
```

Each cell invokes:

```text
python -m XAI_Enhancer_module.ibs.train \
  --data-root /vol/data/IBS-patient-dataset \
  --arch <ARCH> --fold <FOLD> --seed 42 --epochs 50 --batch-size 128 \
  --output-dir /vol/runs/ibs --device cuda --num-workers 8 \
  --amp --amp-dtype bfloat16 --compile
```

| # | arch | fold | Split files | Output directory |
|--:|------|-----:|-------------|------------------|
| 1–5 | resnet18 | 0…4 | `splits/fold{k}/{train,val,test}.txt` | `/vol/runs/ibs/resnet18/fold{k}/` |
| 6–10 | resnet34 | 0…4 | same | `/vol/runs/ibs/resnet34/fold{k}/` |
| 11–15 | resnet50 | 0…4 | same | `/vol/runs/ibs/resnet50/fold{k}/` |
| 16–20 | densenet121 | 0…4 | same | `/vol/runs/ibs/densenet121/fold{k}/` |
| 21–25 | vgg16 | 0…4 | same | `/vol/runs/ibs/vgg16/fold{k}/` |

**Data:** Phase 1B patient-disjoint folds (`StratifiedGroupKFold`, 126 exam groups).  
`--fold k` selects which fold’s train/val/test lists to load.  
`--seed 42` only affects training RNG (not the fold membership).

Same per-run artifacts as Kvasir (`args.json`, `metrics.json`, `best.pth`, `last.pth`).

---

## 4. Call graph (Modal)

```text
modal run -m modal_runner.app -- train-kvasir-matrix ...
        │
        ▼
modal_runner/app.py :: train_kvasir_matrix()          # @app.function, GPU=A100
        │
        ▼
modal_runner/jobs/train.py :: train_kvasir_matrix()   # for arch × seed
        │
        ▼
modal_runner/jobs/train.py :: train_kvasir(...)       # one cell
        │
        ▼
python -m XAI_Enhancer_module.kvasir.train ...
        │
        ├── common/train_utils.seed_everything / run_dir / write_args_json
        ├── kvasir/data.KvasirDataset(split=train|val)
        └── checkpoint on val macro-F1 → runs/kvasir/{arch}/seed{seed}/best.pth
```

IBS is identical with `train_ibs_matrix` → `train_ibs` → `ibs/train.py` and
`IBSDataset(..., fold=k)`.

Local orchestrator (`common/train_matrix.py`) builds the same CLI strings via
`subprocess` instead of Modal `run_module`.

---

## 5. Phase 2 module map

| Path | Role |
|------|------|
| `common/train_utils.py` | `seed_everything`, `run_dir`, `write_args_json`, `collect_logits`, `classification_metrics` (acc / F1 / AUROC / ECE / per-class) |
| `common/train_matrix.py` | Local nested loops (dry-run supported) |
| `kvasir/train.py` | Single Kvasir run; `--seed`; out `…/seed{seed}/` |
| `ibs/train.py` | Single IBS run; **required** `--fold`; out `…/fold{fold}/` |
| `kvasir/eval_classification.py` | Test (or val) metrics; default `--split test` |
| `ibs/eval_classification.py` | Same + required `--fold` |
| `modal_runner/jobs/train.py` | Modal wrappers + matrix loops |
| `modal_runner/jobs/evaluate.py` | Default ckpt paths under `seed*` / `fold*` |
| `modal_runner/app.py` | CLI: `train-kvasir[-matrix]`, `train-ibs[-matrix]`, `eval-*-cls` |
| `modal_runner/config.py` | `KVASIR_ARCHS`, `IBS_ARCHS`, volume paths |

---

## 6. Classifier eval (after a cell finishes)

```bash
# Kvasir seed 42
modal run -m modal_runner.app -- eval-kvasir-cls \
  --arch resnet50 --seed 42 --split test

# IBS fold 0
modal run -m modal_runner.app -- eval-ibs-cls \
  --arch resnet50 --fold 0 --split test
```

Defaults resolve checkpoints to:

- `/vol/runs/kvasir/{arch}/seed{seed}/best.pth`
- `/vol/runs/ibs/{arch}/fold{fold}/best.pth`

Writes `cls_{split}.json` next to the checkpoint (accuracy, macro/weighted F1,
AUROC, ECE, per-class report).

---

## 7. Narrowing the matrix (overrides)

```bash
# Only resnet18 × seeds 42,43
modal run -m modal_runner.app -- train-kvasir-matrix \
  --archs resnet18 --seeds 42 43 --epochs 50

# Only densenet121 × folds 0 and 1
modal run -m modal_runner.app -- train-ibs-matrix \
  --archs densenet121 --folds 0 1 --epochs 50

# Local dry-run (print commands, no train)
python -m XAI_Enhancer_module.common.train_matrix \
  --dataset both --dry-run --archs resnet18 --seeds 42 --folds 0
```

Single-cell smoke (not part of the matrix CLI):

```bash
modal run -m modal_runner.app -- train-kvasir --arch resnet18 --seed 42 --smoke
modal run -m modal_runner.app -- train-ibs --arch resnet18 --fold 0 --smoke
```

---

## 8. Estimated Modal cost (full matrices)

**Billing basis** ([Modal pricing](https://modal.com/pricing), Sep 2026):

| Resource | Rate | Hourly equiv. |
|----------|------|---------------|
| **A100 40 GB** (what `gpu="A100"` requests) | **$0.000583 / sec** | **≈ $2.10 / GPU-hr** |
| A100 80 GB (if upgraded; same $ as 40 GB on auto-upgrade for `A100`) | $0.000694 / sec | ≈ $2.50 / GPU-hr |
| CPU + RAM on the train container | metered separately | typically **~10–20%** on top of GPU |

Our train functions use `GPU_TRAIN = "A100"` and `memory=65536` (`modal_runner/config.py` / `app.py`). Estimates below use **$2.10 / GPU-hr** and add **~15%** for CPU/RAM. They **exclude** region multipliers (≈1.15–1.75× if you pin a region) and non-preemptible (3×). Starter plan includes **$30 / mo** free credits.

### How runtime was estimated

Calibrated from Phase 2 smoke runs on Modal (resnet18, AMP bfloat16, `torch.compile`):

- **IBS** `--fold 0 --smoke` (2 epochs, batch 32): ~118 train steps/epoch; after compile warmup ~1.7 it/s → ~1–1.5 min/epoch at smoke batch.
- Full matrix uses **batch 128**, **50 epochs**. Larger batch raises images/sec but not linearly; heavier arches (resnet50 / densenet121 / vgg16) are slower than resnet18.

Per-cell GPU-hours for a **50-epoch** run (point estimate; low–high in parentheses):

| Arch | Kvasir (~5491 train) | IBS (~3.7k train / fold) |
|------|---------------------:|-------------------------:|
| resnet18 | 1.1 h (0.8–1.5) | 0.9 h (0.6–1.2) |
| resnet34 | 1.5 h (1.2–2.0) | 1.2 h (0.9–1.6) |
| resnet50 | 2.3 h (1.8–3.0) | 1.8 h (1.4–2.5) |
| densenet121 | 2.6 h (2.0–3.5) | 2.0 h (1.5–2.8) |
| vgg16 | 3.0 h (2.5–4.0) | 2.5 h (2.0–3.5) |
| **Sum / seed or fold** | **≈ 10.5 h** (8.3–14) | **≈ 8.4 h** (6.4–11.6) |

Extra overhead: each cell is a fresh `python -m …` process, so **compile warm-up is paid per cell** (~2–5 min) → roughly **+1–3 GPU-hr** across a full matrix.

### Kvasir matrix — 15 runs (`train-kvasir-seed` × 3)

| | Low | Mid (point) | High |
|--|----:|------------:|-----:|
| GPU-hours (5 arches × 3 seeds) | ~25 | **~32** | ~42 |
| GPU $ @ $2.10/hr | ~$52 | **~$66** | ~$88 |
| + ~15% CPU/RAM | ~$60 | **~$76** | ~$101 |
| Wall-clock **per seed** (5 A100s parallel) | ~3–4 h | **~4–5 h** | ~6–8 h |
| Wall-clock **all 3 seeds** (sequential waves) | ~9–12 h | **~12–15 h** | ~18–24 h |

**Recommended cost probe:** run `--seed 42` only first (~1/3 of Kvasir GPU-$ ≈ **$20–$35**), read Modal usage, then launch 43 / 44.

### IBS matrix — 25 runs (`train-ibs-matrix`)

| | Low | Mid (point) | High |
|--|----:|------------:|-----:|
| GPU-hours (5 arches × 5 folds) | ~32 | **~42** | ~58 |
| GPU $ @ $2.10/hr | ~$67 | **~$88** | ~$122 |
| + ~15% CPU/RAM | ~$77 | **~$101** | ~$140 |
| Wall-clock if **sequential** | ~1.3 days | **~1.8 days** | ~2.4 days |

### Both matrices — 40 runs

| | Low | Mid (point) | High |
|--|----:|------------:|-----:|
| GPU-hours | ~57 | **~74** | ~100 |
| **All-in $ (GPU + ~15% CPU/RAM)** | **~$140** | **~$180** | **~$240** |
| Sequential wall-clock | ~2.4 days | **~3.1 days** | ~4.2 days |

**Practical takeaway:** budget about **$150–$250** for both full 50-epoch matrices on default A100 billing, or **~$75–$100** for Kvasir alone and **~$100–$140** for IBS alone. After the first real 50-epoch cell finishes, rescale:  
`actual_cost ≈ (measured_GPU_hr_per_cell) × N_cells × $2.10 × 1.15`.

### Ways to cut cost / wall-clock

1. **Kvasir parallel seed waves** (`train-kvasir-seed`) — same GPU-$ as sequential, ~3× less wall-clock per seed; Starter plans typically allow ≥10 concurrent GPUs.
2. **Smoke / short epochs first** (`--smoke` or `--epochs 5`) to validate paths before paying for 50 epochs.
3. **Narrow arches** for a pilot (e.g. `--archs resnet18 resnet50`) then expand.
4. Watch for **region multipliers** in the Modal dashboard if you pin regions.
5. Academic / startup credit grants on Modal can offset a large fraction of this budget.
6. **Resume** failed cells with `--resume auto` instead of restarting from epoch 1.

Classifier eval (`eval-*-cls`) after training is cheap (minutes per checkpoint) — typically **&lt; $5** for all 40 evals.

---

## 9. Design notes

1. **Kvasir: parallel per seed** — `train-kvasir-seed` maps five `train_kvasir` A100 containers. Use `modal run --detach` so logout does not kill the wave. IBS matrix remains sequential for now.
2. **Kvasir seed ≠ new split** — Phase 1A splits are fixed on disk; seeds diversify training stochasticity for mean±SD tables.
3. **IBS fold = new patient partition** — each fold has a disjoint test exam set; report mean±SD over folds.
4. **Legacy paths** — eval still falls back to `/vol/runs/{ds}/{arch}/best.pth` if an old flat layout exists.
5. **Do not use** the numeric `IBS-preprocessed-dataset` for these matrices; trainers default to `IBS-patient-dataset` (D-M4).
6. **GPU packing** — auto-batch probes with **AdamW** (not SGD) and caps batch so `n_train / batch ≥ 20` steps/epoch (Kvasir ≈ batch ≤ ~275). That may leave VRAM below 82% on light arches; raising batch further would produce 5-step epochs and unstable training. Prefer effective batch via `--grad-accum-steps` if you need larger updates without fewer steps.
7. **Detach** — `train-kvasir-seed` calls a single remote orchestrator (`train_kvasir_seed_wave`) that maps the 5 GPUs. Do not map from the local entrypoint under `--detach` (Modal only keeps the last function alive).

---

## 10. Maintenance

When you change default arches, seeds, folds, or run-dir layout, update:

1. This file (sections 1–3 and **§8 cost table** if arches/epochs change),
2. `modal_runner/config.py` (`KVASIR_ARCHS` / `IBS_ARCHS`),
3. `common/train_matrix.py` constants,
4. `MODAL_EXPERIMENT_PLAN.md` Phase 2 target layout.
