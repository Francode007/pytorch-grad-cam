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

## Training (to be updated in Phase 2)

```bash
python -m XAI_Enhancer_module.kvasir.train --arch resnet50 --data-root data/kvasir-v2 --seed 42
```

### On Modal (Phase 1 — run now)

```bash
modal run -m modal_runner.app -- prepare-kvasir-splits --seed 42
modal run -m modal_runner.app -- download-ibs-patient
modal run -m modal_runner.app -- prepare-ibs-folds --seed 42
modal run -m modal_runner.app -- smoke-splits
modal run -m modal_runner.app -- train-kvasir --arch resnet18 --smoke
```

Full phase checklist: [`MODAL_EXPERIMENT_PLAN.md`](MODAL_EXPERIMENT_PLAN.md).

See `cmpb_revision/09-coding-roadmap.md` for the full pipeline.
