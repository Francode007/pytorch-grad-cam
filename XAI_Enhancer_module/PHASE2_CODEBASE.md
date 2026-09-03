# Phase 2 codebase guide — training matrices & classifier metrics

Supplementary map of the **Phase 2** code added for the CMPB revision
(R3-1 patient/seed protocol, D-M5 val macro-F1 checkpoints).

Use this alongside:

- Operator checklist: [`MODAL_EXPERIMENT_PLAN.md`](MODAL_EXPERIMENT_PLAN.md)
- Short repro commands: [`REPRODUCE.md`](REPRODUCE.md)
- Locked decisions: [`../cmpb_revision/DECISIONS.md`](../cmpb_revision/DECISIONS.md)

---

## 1. What the matrix commands are

### Kvasir: parallel per seed

`train-kvasir-seed` fans out **one A100 per architecture** for a **single**
`--seed` (5 GPUs in parallel). Run seed 42 first, estimate cost, then 43 / 44.

```bash
# Survives Mac sleep / logout / closed terminal (spawn + --detach)
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 43
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 44
```

| Modal CLI | Behaviour | Count per invocation |
|-----------|-----------|---------------------:|
| `train-kvasir-seed --seed S` | **5 A100s in parallel** (all arches, one seed) | **5** |
| `train-kvasir` | Single arch × seed (resume / backfill / debug) | **1** |
| `train-kvasir-matrix` | Legacy **sequential** arches×seeds on one GPU | up to **15** |

### IBS: parallel per fold (same pattern)

`train-ibs-fold` fans out **one A100 per architecture** for a **single**
`--fold`. Fold 0 first (locks batch sizes), then folds 1–4.

```bash
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 0
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 1
# ... folds 2, 3, 4
```

| Modal CLI | Behaviour | Count per invocation |
|-----------|-----------|---------------------:|
| `train-ibs-fold --fold K` | **5 A100s in parallel** (all arches, one fold) | **5** |
| `train-ibs-cv` | **5×5 map** (25 jobs; extras queue at concurrency limit) | **25** |
| `train-ibs` | Single arch × fold (resume / debug) | **1** |
| `train-ibs-matrix` | Legacy sequential on one GPU | up to **25** |

Operator day-to-day: [`PHASE2_RUNBOOK.md`](PHASE2_RUNBOOK.md).

### Default axes

| Axis | Values | Defined in |
|------|--------|------------|
| Architectures (Kvasir) | `resnet18`, `resnet34`, `resnet50`, `vgg19`, `vgg16` | `config.KVASIR_ARCHS` |
| Architectures (IBS) | same as Kvasir | `config.IBS_ARCHS` |
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

IBS mirrors this under `/vol/runs/ibs/` with `fold{k}` paths and
`waves/fold{k}/` summaries; lock file is `/vol/runs/ibs/locked_batch_sizes.json`
(written after fold 0 epoch 1, reused by folds 1–4).

```bash
modal run -m modal_runner.app -- summarize-kvasir-seed --seed 42
modal run -m modal_runner.app -- summarize-ibs-fold --fold 0
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
| 10 | vgg19 | 42 | `/vol/runs/kvasir/vgg19/seed42/` |
| 11 | vgg19 | 43 | `/vol/runs/kvasir/vgg19/seed43/` |
| 12 | vgg19 | 44 | `/vol/runs/kvasir/vgg19/seed44/` |
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

Preferred (same pattern as Kvasir seeds): one fold = 5 A100s, then repeat folds 1–4.

```bash
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 0
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 1
# ... folds 2, 3, 4
```

Or all 25 cells in one map (jobs beyond your GPU concurrency queue):

```bash
modal run --detach -m modal_runner.app -- train-ibs-cv
```

Each cell invokes (batch size resolved by auto-probe / lock unless overridden):

```text
python -m XAI_Enhancer_module.ibs.train \
  --data-root /vol/data/IBS-patient-dataset \
  --arch <ARCH> --fold <FOLD> --seed 42 --epochs 50 --batch-size 0 \
  --output-dir /vol/runs/ibs --device cuda --num-workers 8 \
  --amp --amp-dtype bfloat16 --compile --a100 --auto-batch-size
```

| # | arch | fold | Split files | Output directory |
|--:|------|-----:|-------------|------------------|
| 1–5 | resnet18 | 0…4 | `splits/fold{k}/{train,val,test}.txt` | `/vol/runs/ibs/resnet18/fold{k}/` |
| 6–10 | resnet34 | 0…4 | same | `/vol/runs/ibs/resnet34/fold{k}/` |
| 11–15 | resnet50 | 0…4 | same | `/vol/runs/ibs/resnet50/fold{k}/` |
| 16–20 | vgg19 | 0…4 | same | `/vol/runs/ibs/vgg19/fold{k}/` |
| 21–25 | vgg16 | 0…4 | same | `/vol/runs/ibs/vgg16/fold{k}/` |

**Data:** Phase 1B patient-disjoint folds (`StratifiedGroupKFold`, 126 exam groups).  
`--fold k` selects which fold’s train/val/test lists to load.  
`--seed 42` only affects training RNG (not the fold membership).

Same per-run artifacts as Kvasir (`train.log`, `args.json`, `metrics.json`,
`best.pth`, `last.pth`, `checkpoint_latest.pth`, `checkpoint_mid.pth`).

Wave artifacts: `/vol/runs/ibs/waves/fold{k}/wave_summary.{json,txt}` + `wave.log`.

---

## 4. Call graph (Modal)

```text
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42
        │
        ▼  .spawn()  (detach-safe; laptop can disconnect)
modal_runner/app.py :: train_kvasir_seed_wave()       # CPU orchestrator
        │
        ├── train_kvasir.map(archs…)                  # 5× A100
        │         │
        │         ▼
        │   python -m XAI_Enhancer_module.kvasir.train ...
        │         → /vol/runs/kvasir/{arch}/seed{seed}/
        │
        └── summarize_kvasir_seed → waves/seed{seed}/ + locked_batch_sizes.json

modal run --detach -m modal_runner.app -- train-ibs-fold --fold 0
        │
        ▼  .spawn()
modal_runner/app.py :: train_ibs_fold_wave()
        ├── train_ibs.map(archs…)                     # 5× A100
        │         → /vol/runs/ibs/{arch}/fold{fold}/
        └── summarize_ibs_fold → waves/fold{fold}/

modal run --detach -m modal_runner.app -- train-ibs-cv
        │
        ▼
modal_runner/app.py :: train_ibs_cv_wave()
        └── train_ibs.starmap(arch×fold pairs)        # up to 25 A100 jobs
```

Legacy sequential: `train-*-matrix` still exists but is not recommended.

---

## 5. Phase 2 module map

| Path | Role |
|------|------|
| `common/train_utils.py` | `seed_everything`, `run_dir`, `write_args_json`, `collect_logits`, `classification_metrics` |
| `common/train_matrix.py` | Local nested loops (dry-run supported) |
| `kvasir/train.py` | Kvasir cell: auto-batch, tee log, mid/latest resume, seed dir |
| `ibs/train.py` | IBS cell: same machinery; required `--fold` |
| `kvasir/eval_classification.py` | Test metrics; default `--split test` |
| `ibs/eval_classification.py` | Same + required `--fold` |
| `modal_runner/jobs/train.py` | Modal wrappers |
| `modal_runner/jobs/summarize.py` | Wave cost/metrics tables + batch locks |
| `modal_runner/jobs/reset.py` | Wipe seed dirs / locks (`train-kvasir-seed --reset`) |
| `modal_runner/jobs/evaluate.py` | Default ckpt paths under `seed*` / `fold*` |
| `modal_runner/app.py` | CLI: `train-kvasir-seed`, `train-ibs-fold`, `train-ibs-cv`, summarize/resume |
| `modal_runner/config.py` | `KVASIR_ARCHS` / `IBS_ARCHS` (= vgg19 set), timeouts |
| `PHASE2_RUNBOOK.md` | Day-to-day launch / log / resume commands |

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
| densenet121 → **vgg19** (matrix) | ~2.8 h (2.2–3.8) | ~2.2 h (1.6–3.0) |
| vgg16 | 3.0 h (2.5–4.0) | 2.5 h (2.0–3.5) |
| **Sum / seed or fold** | **≈ 10.7 h** (8.5–14) | **≈ 8.6 h** (6.5–12) |

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

### IBS matrix — 25 runs (`train-ibs-fold` × 5, or `train-ibs-cv`)

| | Low | Mid (point) | High |
|--|----:|------------:|-----:|
| GPU-hours (5 arches × 5 folds) | ~32 | **~43** | ~60 |
| GPU $ @ $2.10/hr | ~$67 | **~$90** | ~$126 |
| + ~15% CPU/RAM | ~$77 | **~$104** | ~$145 |
| Wall-clock **per fold** (5 A100s parallel) | ~2.5–4 h | **~4–5 h** | ~6–8 h |
| Wall-clock **all 5 folds** (sequential waves) | ~12–20 h | **~20–25 h** | ~30–40 h |

**Recommended cost probe:** run `--fold 0` first (~1/5 of IBS GPU-$), then continue.

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

1. **Parallel per seed / fold** — `train-kvasir-seed` and `train-ibs-fold` each map five A100 containers. Launch with `modal run --detach` (`.spawn()` so logout does not cancel GPUs).
2. **Kvasir seed ≠ new split** — Phase 1A splits are fixed on disk; seeds diversify training stochasticity for mean±SD tables.
3. **IBS fold = new patient partition** — each fold has a disjoint test exam set; report mean±SD over folds.
4. **Arch set** — both matrices use VGG19 (not DenseNet121). DenseNet leftovers on the volume are ignored for reporting.
5. **Legacy paths** — eval still falls back to `/vol/runs/{ds}/{arch}/best.pth` if an old flat layout exists.
6. **Do not use** the numeric `IBS-preprocessed-dataset` for these matrices; trainers default to `IBS-patient-dataset` (D-M4).
7. **GPU packing** — auto-batch probes with **AdamW** and caps batch so `n_train / batch ≥ 20` steps/epoch. Prefer `--grad-accum-steps` if you need larger effective batch without fewer steps.
8. **Detach** — waves are spawned remotely; do not rely on a waiting local `.remote()` for multi-hour jobs.

---

## 10. Maintenance

When you change default arches, seeds, folds, or run-dir layout, update:

1. This file (sections 1–3 and **§8 cost table** if arches/epochs change),
2. [`PHASE2_RUNBOOK.md`](PHASE2_RUNBOOK.md),
3. `modal_runner/config.py` (`KVASIR_ARCHS` / `IBS_ARCHS`),
4. `common/train_matrix.py` constants,
5. `MODAL_EXPERIMENT_PLAN.md` Phase 2 section.
