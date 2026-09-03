# Getting started in the IDE (branch `kvasir_v2`)

## 1. Checkout

```bash
git fetch origin
git checkout kvasir_v2
# or, if creating locally from kvasir_v1:
# git checkout -b kvasir_v2 origin/kvasir_v1
```

## 2. Read before coding

1. `cmpb_revision/09-coding-roadmap.md` — especially **§0** (code vs paper).
2. `cmpb_revision/00-executive-summary-and-triage.md` — what is mandatory vs defensible.
3. `cmpb_revision/DECISIONS.md` — fill in D-M1 and D-M2 on Day 1.

## 3. Copy-paste prompt for your IDE agent

Use this as the first message in a new agent/chat session with the repo open on `kvasir_v2`:

---

**Context:** I am revising CMPB-D-26-01921 (XAI-Enhancer). All planning documents are in `cmpb_revision/` at the repo root. Implementation is in `XAI_Enhancer_module/`. Branch: `kvasir_v2`. Deadline: 13 Sep 2026.

**Blocking facts (from code audit — see `cmpb_revision/09-coding-roadmap.md` §0):**
1. The paper describes **input-image masking** (Eq. 2–3); the code does **layer activation gating** via forward hooks in `enhanced_cams/*_enhanced.py` and `utils/optimized_cam_extractor.py`. We are taking **Decision A**: rewrite Methods/Fig. 1 to match activation gating unless I say otherwise.
2. `kvasir/eval_cams.py` defaults to `--enhanced-method stagewise`; paper Eq. 5 is flat softmax (`standard`). Table 1 and Table 4 must use the **same** aggregation — default to `standard` unless I confirm stagewise from old logs.
3. Mandatory reviewer fixes: patient-level IBS splits, Kvasir train/val/**test**, classifier AUROC/F1 on test, per-image metrics with bootstrap CIs and paired Wilcoxon tests, uniform-average baseline, Score-CAM/Group-CAM baselines, limitations in Abstract/Conclusion.

**Execute in order (do not skip P0):**

**Phase 1 — Splits & protocol (P0)**  
- Finish `XAI_Enhancer_module/common/splits.py`: Kvasir 70/10/20 stratified splits; IBS `StratifiedGroupKFold` with `extract_group_id()` from image paths; `write_split_summary()` CSV.  
- Update `kvasir/data.py` and `ibs/data.py` to call common split helpers; support `test` split in datasets.  
- Near-duplicate scan for Kvasir (perceptual hash, Hamming ≤ 6 across splits).

**Phase 2 — Training & classifier metrics (P0)**  
- `kvasir/train.py` and `ibs/train.py`: `--fold`, `--seed`, best checkpoint by **val macro-F1**, save `args.json`.  
- Extend `eval_classification.py`: AUROC, per-class report, ECE, run on `--split test`.  
- Add `common/train_matrix.py` to orchestrate 5 archs × 3 seeds (Kvasir) and 5 archs × 5 folds (IBS).

**Phase 3 — Evaluation harness (P0)**  
- `evaluator/imagenet_proper_auc_evaluator.py`: per-image Parquet/CSV logging; `--layer-set {all,conv3x3,block_outputs,stage_outputs,last_5,last}`; default `--split test`.  
- `enhanced_combiner/aggregator.py`: add `type: uniform` (1/L average).  
- `eval_cams.py` (kvasir + ibs): default `--enhanced-method standard`; record protocol in run header; fix `--step-size` documentation.

**Phase 4 — Statistics (P0)**  
- New `analysis/stats.py`: bootstrap 95% CI, paired Wilcoxon (Holm), Cliff's δ, win/tie/loss table from per-image CSVs.

**Phase 5 — Baselines & ablations (P1)**  
- Score-CAM, Group-CAM, Opti-CAM in eval pipeline; Kvasir-SEG test-only + continuous metrics in `clinical_evaluation_proxy.py`; similarity switch in `optimized_cam_extractor.py`; re-benchmark `benchmark_xai_overhead.py`; extend `robustness_augmentations_xai.py`.

**Conventions:**  
- Match existing style in `XAI_Enhancer_module/`.  
- Do not modify unrelated pytorch_grad_cam upstream files.  
- After each phase: commit with a clear message, update `cmpb_revision/DECISIONS.md` if needed, add reproduction commands to `XAI_Enhancer_module/REPRODUCE.md`.  
- Ground changes in `cmpb_revision/` review docs; cite reviewer IDs in commit messages.

Start with **Phase 1**. Show me the IBS filename samples used to infer `extract_group_id`, then implement and run a smoke test that prints the split summary CSV.

---

## 4. Suggested first commands (after Phase 1 lands)

```bash
# Kvasir splits
python -m XAI_Enhancer_module.kvasir.download_and_prepare --data-root data

# Smoke: split summary only
python -c "
from XAI_Enhancer_module.common.splits import prepare_kvasir_splits, write_split_summary
from pathlib import Path
root = Path('data/kvasir-v2')
prepare_kvasir_splits(str(root), seed=42)
write_split_summary(str(root), dataset='kvasir')
"

# Train one model (smoke)
python -m XAI_Enhancer_module.kvasir.train --arch resnet18 --epochs 2 --data-root data/kvasir-v2 --seed 42
```

## 5. What not to put in this folder

- Trained checkpoints, large datasets, or wandb logs (use `runs/` and `.gitignore`).
- Changes inside `XAI_Enhancer_module/` for revision code — only planning docs live here.
