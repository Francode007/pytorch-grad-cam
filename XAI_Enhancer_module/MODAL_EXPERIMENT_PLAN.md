# Modal experiment plan — CMPB revision (branch `modal_kvasir`)

This is the **operator checklist** for running the revision on Modal.
Planning rationale lives in `../cmpb_revision/` (`09-coding-roadmap.md`, `experiment-plan.md`).
Account / secret setup lives in `../modal_runner/README.md`.

**Always run from the repository root on branch `modal_kvasir`.**

```bash
cd /path/to/pytorch-grad-cam
git checkout modal_kvasir
```

Volume: `xai-enhancer-vol` → `/vol/{data,models,runs}`  
App entry: `modal run -m modal_runner.app -- <action> ...`

---

## 0. Prerequisites (already done if your volume has data)

```bash
modal profile current
modal run -m modal_runner.app -- status

# ImageNet weights
modal run -m modal_runner.app -- download-models

# Kvasir (Kaggle) — or skip if already on volume
modal run -m modal_runner.app -- download-kvasir

# IBS — patient-aware tree (revision default):
modal run -m modal_runner.app -- download-ibs-patient
# Legacy numeric dump only if needed:
# modal run -m modal_runner.app -- download-ibs
# or: modal volume put … IBS-preprocessed-dataset.zip && ingest-ibs-zip
```

Pull anything back anytime:

```bash
modal volume ls xai-enhancer-vol /data
modal volume ls xai-enhancer-vol /runs
modal volume get xai-enhancer-vol /data/kvasir-v2/splits ./modal_artifacts/kvasir_splits
modal volume get xai-enhancer-vol /runs ./modal_artifacts/runs
```

---

## Phase 1 — Splits & protocol (P0)

**Goal (R3-1):** Kvasir train/val/test 70/10/20 + pHash near-dup guard; IBS patient-level folds when IDs exist; split-summary CSVs.

**Status:** 1A ✅ completed · 1B ✅ completed · 1C optional · **Phase 2 next**

### 1A. Kvasir splits — ✅ completed (2026-09-03)

Done on Modal volume `xai-enhancer-vol`: stratified **70/10/20** (seed=42) + pHash Hamming ≤ 6 near-dup reassignment; smoke summary OK.

| Split | Images (post-dedupe) |
|-------|---------------------:|
| train | 5491 |
| val   | 921 |
| test  | 1588 |
| **total** | **8000** |

Near-dups: 979 pairs, 414 images moved, 258 components (see D-M3).

Re-run only if you intentionally want to regenerate splits:

```bash
modal run -m modal_runner.app -- prepare-kvasir-splits --seed 42
modal run -m modal_runner.app -- smoke-splits
```

Fetch the dataset table locally (if not already under `modal_artifacts/`):

```bash
modal volume get xai-enhancer-vol \
  /data/kvasir-v2/splits/split_summary_kvasir.csv \
  ./modal_artifacts/split_summary_kvasir.csv

modal volume get xai-enhancer-vol \
  /data/kvasir-v2/splits/PROTOCOL.txt \
  ./modal_artifacts/kvasir_PROTOCOL.txt

modal volume get xai-enhancer-vol \
  /data/kvasir-v2/splits/near_duplicates.csv \
  ./modal_artifacts/near_duplicates.csv
```

**Acceptance (met):** `train+val+test = 8000`; summary CSV has 8 classes; PROTOCOL records seed=42 and dedupe counts.

### 1B. IBS patient folds — ✅ completed (2026-09-03)

Done on Modal volume `xai-enhancer-vol`: private Kaggle [`franchisn/ibs-dataset`](https://www.kaggle.com/datasets/franchisn/ibs-dataset)
flattened to `/vol/data/IBS-patient-dataset`; exam map from bundled
`XAI_Enhancer_module/ibs/metadata/ibs_groups.csv` → `/vol/data/ibs_groups.csv`.

| | Count |
|--|------:|
| Images | 5547 (711 IBS + 4836 Normal) |
| Exam groups | **126** (16 IBS + 110 Normal) |
| Folds | 5 (`StratifiedGroupKFold`, seed=42, inner val 15%) |
| Unresolved IDs | **0** |

Re-run only if you intentionally want to regenerate folds:

```bash
modal run -m modal_runner.app -- download-ibs-patient
modal run -m modal_runner.app -- prepare-ibs-folds --n-folds 5 --seed 42
```

Fetch the summary locally (if not already under `modal_artifacts/`):

```bash
modal volume get xai-enhancer-vol \
  /data/IBS-patient-dataset/splits/split_summary_ibs.csv \
  ./modal_artifacts/split_summary_ibs.csv
```

**Acceptance (met):** 126 exam groups; fold prep asserts no group leakage across train/val/test;
summary has `n_patients_*` columns (see D-M4).

> Do **not** use the numeric `pre-processed-ibs` dump for R3-1 — it has no
> recoverable patient/exam IDs (D-M4).

### 1C. Optional smoke train (sanity check GPU path)

```bash
# 2 epochs on A100 — verifies data loaders + volume writes
modal run -m modal_runner.app -- train-kvasir --arch resnet18 --smoke
# IBS smoke still uses deprecated image-level split until Phase 2 --fold lands:
# modal run -m modal_runner.app -- train-ibs --arch resnet18 --smoke
```

---

## Phase 2 — Training & classifier metrics (P0)

**Goal:** 5 archs × 3 seeds (Kvasir) and 5 archs × 5 folds (IBS); best ckpt by **val macro-F1**; test AUROC/F1/ECE.

**Status:** code landed on `modal_kvasir` — launch GPU matrix next.

**Codebase map (matrices, run dirs, eval):** [`PHASE2_CODEBASE.md`](PHASE2_CODEBASE.md)

### Code (landed)

| Piece | Path | Notes |
|-------|------|--------|
| Seed/fold-aware train | `kvasir/train.py`, `ibs/train.py` | Kvasir `--seed`; IBS `--fold`; val **macro-F1** ckpt; `args.json` |
| Classifier metrics | `kvasir/eval_classification.py`, `ibs/eval_classification.py` | AUROC, F1, ECE, per-class; default `--split test` |
| Matrix orchestrator | `common/train_matrix.py` | local 5×3 / 5×5 loops |
| Modal wiring | `modal_runner/jobs/train.py`, `app.py` | `train-*-matrix`, `--fold` / `--seeds` |

### Modal commands

```bash
# Smoke (2 epochs)
modal run -m modal_runner.app -- train-kvasir --arch resnet18 --seed 42 --smoke
modal run -m modal_runner.app -- train-ibs --arch resnet18 --fold 0 --smoke

# Single full run
modal run -m modal_runner.app -- train-kvasir --arch resnet50 --seed 42 --epochs 50
modal run -m modal_runner.app -- train-ibs --arch resnet50 --fold 0 --epochs 50

# Full matrices (long)
modal run -m modal_runner.app -- train-kvasir-matrix --epochs 50
modal run -m modal_runner.app -- train-ibs-matrix --epochs 50

# Classifier eval on held-out test
modal run -m modal_runner.app -- eval-kvasir-cls --arch resnet50 --seed 42 --split test
modal run -m modal_runner.app -- eval-ibs-cls --arch resnet50 --fold 0 --split test
```

**Target layout on volume:**

```
/vol/runs/kvasir/{arch}/seed{42,43,44}/best.pth
/vol/runs/ibs/{arch}/fold{0..4}/best.pth
```

**Acceptance:** JSON metrics per run; mean ± SD across seeds/folds printable.

---

## Phase 3 — Evaluation harness (P0)

**Goal:** per-image Parquet/CSV; uniform baseline; `--layer-set`; default `--split test`; `--enhanced-method standard`.

### Code still to land

| Piece | Path |
|-------|------|
| Per-image logging | `evaluator/imagenet_proper_auc_evaluator.py` |
| Uniform aggregator | `enhanced_combiner/aggregator.py` (`type: uniform`) |
| Layer sets + protocol header | `kvasir/eval_cams.py`, `ibs/eval_cams.py` |
| Modal flags | `modal_runner/jobs/evaluate.py`, `app.py` (`--layer-set`, `--enhanced-method`) |

### Modal commands (after Phase 3 code lands)

```bash
# Smoke CAM eval (50 images)
modal run -m modal_runner.app -- eval-kvasir-cams \
  --arch resnet50 --split test --max-images 50 \
  --enhanced-method standard --layer-mode all

# Full test (long — Table 1 precursor)
modal run -m modal_runner.app -- eval-kvasir-cams \
  --arch resnet50 --split test --enhanced-method standard

modal run -m modal_runner.app -- eval-ibs-cams \
  --arch resnet50 --split test --enhanced-method standard
```

Pull per-image logs:

```bash
modal volume get xai-enhancer-vol /runs/kvasir/resnet50/cam_eval ./modal_artifacts/cam_eval_kvasir_r50
```

---

## Phase 4 — Statistics (P0)

**Goal:** bootstrap 95% CI, paired Wilcoxon (Holm), Cliff's δ, win/tie/loss from per-image CSVs.

### Code still to land

| Piece | Path |
|-------|------|
| Stats script | `analysis/stats.py` (new) |
| Modal job | `modal_runner/jobs/stats.py` + `app.py` action `stats` |

### Modal / local commands (after code lands)

```bash
# Prefer pulling CSVs and running stats locally (CPU, cheap), OR:
# modal run -m modal_runner.app -- stats --input-dir /vol/runs/kvasir

modal volume get xai-enhancer-vol /runs ./modal_artifacts/runs
python -m XAI_Enhancer_module.analysis.stats \
  --input-dir ./modal_artifacts/runs \
  --output-dir ./modal_artifacts/stats
```

---

## Phase 5 — Baselines & ablations (P1)

**Goal:** Score-CAM / Group-CAM / Opti-CAM; Kvasir-SEG test-only + continuous metrics; similarity switch; latency; robustness.

### Code still to land

| Piece | Path |
|-------|------|
| Group-CAM / Opti-CAM | `baselines/group_cam.py`, `baselines/opti_cam.py` |
| Clinical proxy | `clinical_evaluation_proxy.py` |
| Similarity switch | `utils/optimized_cam_extractor.py` |
| Latency | `benchmark_xai_overhead.py` |
| Robustness | `robustness_augmentations_xai.py` |
| Modal jobs | extend `evaluate.py` / new `jobs/ablation.py` |

### Modal commands (sketch after wiring)

```bash
# Upload Kvasir-SEG once
modal volume put xai-enhancer-vol data/Kvasir-SEG /data/Kvasir-SEG

# Clinical proxy / benchmarks — to be exposed as:
# modal run -m modal_runner.app -- clinical-proxy --arch resnet50
# modal run -m modal_runner.app -- benchmark-overhead --arch resnet50
# modal run -m modal_runner.app -- robustness --arch resnet50
```

Until those actions exist, run modules inside a one-off Modal function or extend `app.py` the same way as `eval_kvasir_cams`.

---

## Experiment tiers ↔ Modal (quick map)

| Tier (`experiment-plan.md`) | Phase | Modal actions available today |
|-----------------------------|-------|-------------------------------|
| 0.4 Kvasir near-dup | 1A ✅ | `prepare-kvasir-splits` (done on volume) |
| 0.1–0.2 IBS IDs / counts | 1B ✅ | `download-ibs-patient`, `prepare-ibs-folds` (done on volume) |
| 1.1–1.3 Train + classifier table | 2 | `train-kvasir`, `train-kvasir-matrix`, `train-ibs`, `train-ibs-matrix`, `eval-*-cls` |
| 1.4 + 2.1 Table 1 / CAM | 3 | `eval-*-cams` (per-image log / uniform / layer-set pending) |
| 2.2 Stats / win-tie-loss | 4 | pull + `analysis/stats.py` (to build) |
| 2.3–2.5, 3.*, 4.* | 5 | extend Modal jobs after code |

---

## Recommended order **right now** (Phase 1B)

1A is done. Next: IBS patient-level folds, then optional GPU smoke.

```bash
cd /path/to/pytorch-grad-cam
git checkout modal_kvasir

modal run -m modal_runner.app -- status
modal run -m modal_runner.app -- download-ibs-patient
modal run -m modal_runner.app -- prepare-ibs-folds --n-folds 5 --seed 42

mkdir -p modal_artifacts
modal volume get xai-enhancer-vol \
  /data/IBS-patient-dataset/splits ./modal_artifacts/ibs_splits

# Optional GPU smoke
modal run -m modal_runner.app -- train-kvasir --arch resnet18 --smoke
```

**Admin:** email editor for extension (patient-level re-split + seeds) if not already sent.

---

## Codebase ownership

| Area | Owns remote I/O | Owns science logic |
|------|-----------------|--------------------|
| `modal_runner/` | Modal App, volumes, secrets, CLI | thin wrappers only |
| `XAI_Enhancer_module/common/` | — | splits, (soon) train_matrix, stats |
| `XAI_Enhancer_module/kvasir/` | — | data, train, eval_classification, eval_cams |
| `XAI_Enhancer_module/ibs/` | — | data, train, eval_* |
| `XAI_Enhancer_module/evaluator/` | — | ROAD / Ins / Del |
| `cmpb_revision/` | — | review plan, decisions (not executed on GPU) |

Do **not** put large datasets or checkpoints in git; keep them on `xai-enhancer-vol` and sync with `modal volume get/put`.
