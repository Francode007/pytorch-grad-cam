# Reproducing CMPB revision experiments (branch `kvasir_v2`)

Planning documents: `../cmpb_revision/` (repository root).

## Splits

```bash
# Kvasir-v2: 70/10/20 + optional dedupe
python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data --skip-download --dedupe

# IBS: 5-fold patient-level CV
python -m XAI_Enhancer_module.ibs.download_and_prepare --data-root data --skip-download
```

Split summaries are written to `data/<dataset>/splits/split_summary_*.csv`.

## Training (to be updated in Phase 2)

```bash
python -m XAI_Enhancer_module.kvasir.train --arch resnet50 --data-root data/kvasir-v2 --seed 42
```

See `cmpb_revision/09-coding-roadmap.md` for the full pipeline.
