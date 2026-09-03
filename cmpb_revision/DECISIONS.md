# Locked decisions for the revision

Update this file when a choice is final. Code and manuscript must match these entries.

| ID | Topic | Decision | Date | Notes |
|---|---|---|---|---|
| D-M1 | Phase 2 masking | **Pending** — default **A** (rewrite paper: activation gating); confirm with co-authors | | See `09-coding-roadmap.md` §0.1 |
| D-M2 | Aggregation for Table 1 / 4 | **Pending** — target **`standard`** (flat softmax, Eq. 5); audit `kvasir_v1` logs | | Change eval default in Phase 3 |
| D-M3 | Kvasir split | **70/10/20** train/val/test, stratified, seed=42 | 2026-09-03 | `common/splits.py` + `kvasir/download_and_prepare.py` |
| D-M4 | IBS split | **5-fold patient-level CV**, inner val 15%, `extract_group_id` heuristic | 2026-09-03 | `common/splits.py` + `ibs/download_and_prepare.py` |
| D-M5 | Checkpoint selection | **Val macro-F1** (not accuracy) | | `train.py` to be updated |
| D-M6 | Evaluation split | **test** only for reported metrics | | `eval_cams.py` default |
