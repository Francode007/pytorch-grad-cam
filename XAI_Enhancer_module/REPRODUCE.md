# Reproducing CMPB revision experiments

- Planning: `../cmpb_revision/` (repository root).
- **Cloud (Modal):** branch `modal_kvasir` — see [`../modal_runner/README.md`](../modal_runner/README.md).

## Splits (Phase 1, R3-1)

```bash
# Kvasir-v2: 70/10/20 stratified + optional pHash near-duplicate reassignment
python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data --skip-download --dedupe

# Smoke: print Kvasir split summary CSV (+ IBS filename audit)
python -m XAI_Enhancer_module.common.smoke_splits --dedupe

# IBS: 5-fold patient-level CV — requires a patient/exam map for the Kaggle numeric dump
python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root data --skip-download \
  --groups-csv path/to/ibs_groups.csv
```

`ibs_groups.csv` format: `rel_path,group_id` (or `filename,group_id`). The Kaggle
`pre-processed-ibs` release has no recoverable IDs (`2882.jpg`, no EXIF); see D-M4.

Kvasir-v2 smoke (seed=42, then pHash Hamming ≤ 6): 8,000 images → 5,491 / 921 / 1,588
after 414 near-duplicate reassignments (979 pairs). Summaries:
`data/<dataset>/splits/split_summary_*.csv`.

## Training (to be updated in Phase 2)

```bash
python -m XAI_Enhancer_module.kvasir.train --arch resnet50 --data-root data/kvasir-v2 --seed 42
```

### On Modal (after `modal setup`)

```bash
modal run -m modal_runner.app -- download-models
modal run -m modal_runner.app -- download-kvasir
modal run -m modal_runner.app -- train-kvasir --arch resnet50 --smoke
```

See `cmpb_revision/09-coding-roadmap.md` for the full pipeline.
