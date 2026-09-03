# Phase 2 operator runbook — parallel A100 training

**Branch:** `modal_kvasir`  
**Status:** experiments in progress (Kvasir seeds + IBS folds)  
**Volume:** `xai-enhancer-vol` → `/vol/{data,models,runs}`

Use this while jobs are running. Deeper matrix/cost notes: [`PHASE2_CODEBASE.md`](PHASE2_CODEBASE.md). Full checklist: [`MODAL_EXPERIMENT_PLAN.md`](MODAL_EXPERIMENT_PLAN.md).

---

## Architecture set (both datasets)

| # | Arch | Notes |
|--:|------|--------|
| 1 | `resnet18` | |
| 2 | `resnet34` | |
| 3 | `resnet50` | |
| 4 | `vgg19` | Replaces `densenet121` in the revision matrix |
| 5 | `vgg16` | |

Defined in `modal_runner/config.py` as `KVASIR_ARCHS` / `IBS_ARCHS` (identical).

Leftover DenseNet121 Kvasir runs (if any) under `/vol/runs/kvasir/densenet121/` are **not** part of the matrix.

---

## Detach / logout (important)

Always launch long jobs with **`--detach`**. The entrypoint uses **`.spawn()`**, so closing the laptop does **not** cancel GPUs.

```bash
modal run --detach -m modal_runner.app -- <action> ...
```

Do **not** use plain `.remote()` waiters for multi-hour waves. Monitor in the [Modal dashboard](https://modal.com/apps).

---

## Kvasir — 5 arches × 3 seeds

### Launch (one seed wave = 5 A100s)

```bash
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 43
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 44
```

- Seed 42 auto-probes batch sizes and writes `/vol/runs/kvasir/locked_batch_sizes.json`.
- Seeds 43 / 44 **reuse** those locks (fair mean±SD). Do **not** pass `--reset` on later seeds.

### Backfill a single arch (e.g. VGG19 after a matrix swap)

```bash
modal run --detach -m modal_runner.app -- train-kvasir --arch vgg19 --seed 42
modal run --detach -m modal_runner.app -- train-kvasir --arch vgg19 --seed 43
```

### Resume a crashed arch

```bash
modal run --detach -m modal_runner.app -- \
  train-kvasir --arch vgg16 --seed 42 --resume auto
```

### Clean relaunch of one seed (wipes that seed’s run dirs + locks)

```bash
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42 --reset
```

Only use `--reset` when you intend to throw away that seed’s checkpoints.

### Layout

```
/vol/runs/kvasir/{arch}/seed{42,43,44}/
  train.log
  args.json                 # batch_size_resolved, batch_size_locked
  metrics.json
  best.pth                  # best val macro-F1
  last.pth
  checkpoint_latest.pth     # full trainer state (resume)
  checkpoint_mid.pth        # epoch epochs//2

/vol/runs/kvasir/locked_batch_sizes.json
/vol/runs/kvasir/waves/seed{S}/wave_summary.{json,txt}
/vol/runs/kvasir/waves/seed{S}/wave.log
```

### Check logs / summary

```bash
# Recompute & print table from volume metrics
modal run -m modal_runner.app -- summarize-kvasir-seed --seed 42

# Pull locally
modal volume get xai-enhancer-vol /runs/kvasir/waves/seed42 \
  ./modal_artifacts/kvasir_waves/seed42
modal volume get xai-enhancer-vol /runs/kvasir/resnet50/seed42/train.log \
  ./modal_artifacts/kvasir_r50_s42_train.log
modal volume get xai-enhancer-vol /runs/kvasir/locked_batch_sizes.json \
  ./modal_artifacts/kvasir_locked_batch_sizes.json
```

In `train.log`: `[auto-batch]`, `[batch] locked`, `Epoch N/50`, `[gpu epochN]`.

---

## IBS — 5 arches × 5 patient folds

Same trainer features as Kvasir (auto-batch, `train.log`, mid/latest ckpts, spawn detach).

### Launch (recommended: one fold at a time = 5 A100s)

```bash
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 0
# after fold 0 finishes (check cost / locks):
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 1
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 2
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 3
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 4
```

Fold 0 probes + locks batch sizes → `/vol/runs/ibs/locked_batch_sizes.json`.  
Later folds reuse those per-arch sizes.

### All 25 cells in one map (queues past GPU concurrency)

```bash
modal run --detach -m modal_runner.app -- train-ibs-cv
```

### Resume one cell

```bash
modal run --detach -m modal_runner.app -- \
  train-ibs --arch vgg19 --fold 0 --resume auto
```

### Layout

```
/vol/runs/ibs/{arch}/fold{0..4}/
  train.log  args.json  metrics.json
  best.pth  last.pth  checkpoint_latest.pth  checkpoint_mid.pth

/vol/runs/ibs/locked_batch_sizes.json
/vol/runs/ibs/waves/fold{k}/wave_summary.{json,txt}
/vol/runs/ibs/waves/fold{k}/wave.log
```

### Check logs / summary (after fold 0)

```bash
modal run -m modal_runner.app -- summarize-ibs-fold --fold 0

modal volume get xai-enhancer-vol /runs/ibs/waves/fold0 \
  ./modal_artifacts/ibs_waves/fold0
modal volume get xai-enhancer-vol /runs/ibs/resnet18/fold0/train.log \
  ./modal_artifacts/ibs_r18_f0_train.log
modal volume get xai-enhancer-vol /runs/ibs/locked_batch_sizes.json \
  ./modal_artifacts/ibs_locked_batch_sizes.json
```

---

## Auto-batch behaviour

| Step | What happens |
|------|----------------|
| Probe | AdamW forward+backward; target ~82% VRAM |
| Cap | `n_train / batch ≥ 20` steps/epoch (avoids tiny epoch counts) |
| Lock | After **epoch 1** succeeds → `args.json` + shared `locked_batch_sizes.json` |
| Reuse | Later seeds (Kvasir) / folds (IBS) load lock; skip re-probe |

Disable probe: `--no-auto-batch --batch-size N`. Force re-probe: trainer `--force-auto-batch` (via single-arch CLI if exposed).

Light CNNs (e.g. ResNet-18) may sit **below** 82% VRAM once the min-steps cap applies — that is expected.

---

## Concurrency tip

Starter plans often allow ~10 concurrent GPUs. Prefer:

1. Finish Kvasir seed wave **or**
2. Run one IBS fold wave (5 GPUs)

Avoid stacking a full Kvasir seed wave + IBS fold wave if you hit queueing / unexpected cancels.

---

## Classifier eval (after a cell has `best.pth`)

```bash
modal run -m modal_runner.app -- eval-kvasir-cls --arch resnet50 --seed 42 --split test
modal run -m modal_runner.app -- eval-ibs-cls --arch resnet50 --fold 0 --split test
```

---

## Quick troubleshooting

| Symptom | Action |
|---------|--------|
| `Function call was cancelled` after logout | Old `.remote()` path — use current branch with `--detach` + spawn |
| `_Tee` / `isatty` crash | Fixed on current `modal_kvasir`; pull latest |
| Wave says `ok` but only 2 epochs | Stale smoke `metrics.json`; wipe with `--reset` or delete that run dir |
| Wrong arch (DenseNet) in matrix | Matrix is VGG19; DenseNet dirs are leftovers |
| Need live stdout | Modal dashboard → app `xai-enhancer` → container logs |
