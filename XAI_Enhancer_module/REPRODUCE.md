# Reproducing CMPB revision experiments

- Planning: `../cmpb_revision/` (repository root).
- **Cloud (Modal):** branch `modal_kvasir` — setup in [`../modal_runner/README.md`](../modal_runner/README.md), **phase commands** in [`MODAL_EXPERIMENT_PLAN.md`](MODAL_EXPERIMENT_PLAN.md).

## Splits (Phase 1, R3-1)

```bash
# Kvasir-v2: 70/10/20 stratified + optional pHash near-duplicate reassignment
python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data --skip-download --dedupe

# Smoke: print Kvasir split summary CSV (+ IBS filename audit)
python -m XAI_Enhancer_module.common.smoke_splits --dedupe

# IBS: build patient tree from franchisn/ibs-dataset (once), then 5-fold CV
# (bundled map: XAI_Enhancer_module/ibs/metadata/ibs_groups.csv)
python -m XAI_Enhancer_module.ibs.metadata.build_ibs_groups_csv --force
python -m XAI_Enhancer_module.ibs.download_and_prepare --skip-download \
  --patient-dataset-root data/IBS-patient-dataset
```

See [`ibs/metadata/README.md`](ibs/metadata/README.md). The numeric
`pre-processed-ibs` dump is **not** used for patient-level folds (D-M4).

## Training (Phase 2)

**Operator runbook (Modal parallel waves, logs, resume):** [`PHASE2_RUNBOOK.md`](PHASE2_RUNBOOK.md)  
**Deep dive (matrices, call graph, cost):** [`PHASE2_CODEBASE.md`](PHASE2_CODEBASE.md)

Architectures: `resnet18`, `resnet34`, `resnet50`, `vgg19`, `vgg16`.

### Modal (recommended)

```bash
# Kvasir — 5 A100s × one seed (repeat 43, 44)
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42

# IBS — 5 A100s × one fold (repeat 1–4); or train-ibs-cv for all 25
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 0

# Inspect wave summaries
modal run -m modal_runner.app -- summarize-kvasir-seed --seed 42
modal run -m modal_runner.app -- summarize-ibs-fold --fold 0
```

### Local (single cell / matrix loop)

```bash
# Kvasir — one seed (best ckpt by val macro-F1 → runs/kvasir/{arch}/seed{seed}/)
python -m XAI_Enhancer_module.kvasir.train --arch resnet50 --data-root data/kvasir-v2 --seed 42

# IBS — one patient fold
python -m XAI_Enhancer_module.ibs.train --arch resnet50 --data-root data/IBS-patient-dataset --fold 0

# Full matrices (sequential local)
python -m XAI_Enhancer_module.common.train_matrix --dataset both --epochs 50

# Test-set classifier metrics
python -m XAI_Enhancer_module.kvasir.eval_classification \
  --checkpoint runs/kvasir/resnet50/seed42/best.pth --split test
python -m XAI_Enhancer_module.ibs.eval_classification \
  --checkpoint runs/ibs/resnet50/fold0/best.pth --fold 0 --split test
```

### On Modal

```bash
modal run -m modal_runner.app -- train-kvasir --arch resnet18 --seed 42 --smoke
modal run -m modal_runner.app -- train-ibs --arch resnet18 --fold 0 --smoke
modal run -m modal_runner.app -- train-kvasir-matrix --epochs 50
modal run -m modal_runner.app -- train-ibs-matrix --epochs 50
```

Full phase checklist: [`MODAL_EXPERIMENT_PLAN.md`](MODAL_EXPERIMENT_PLAN.md).

See `cmpb_revision/09-coding-roadmap.md` for the full pipeline.
