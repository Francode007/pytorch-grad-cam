# Locked decisions for the revision

Update this file when a choice is final. Code and manuscript must match these entries.

| ID | Topic | Decision | Date | Notes |
|---|---|---|---|---|
| D-M1 | Phase 2 masking | **Pending** — default **A** (rewrite paper: activation gating); confirm with co-authors | | See `09-coding-roadmap.md` §0.1 |
| D-M2 | Aggregation for Table 1 / 4 | **Pending** — target **`standard`** (flat softmax, Eq. 5); audit `kvasir_v1` logs | | Change eval default in Phase 3 |
| D-M3 | Kvasir split | **70/10/20** train/val/test, stratified, seed=42; pHash Hamming ≤ 6 near-dups reassigned | 2026-09-03 | 8000 images → 5491/921/1588 after 414 near-dup moves (979 pairs, 258 components) |
| D-M4 | IBS split | **5-fold patient-level CV**, inner val 15%; **groups_csv required** for the Kaggle dump | 2026-09-03 | Kaggle `pre-processed-ibs` filenames are integer-only (`IBS/2882.jpg`); no EXIF; class folders only. `extract_group_id` raises (does not hash the class folder). Provide Dryad/owner mapping via `--groups-csv`. |
| D-M5 | Checkpoint selection | **Val macro-F1** (not accuracy) | | `train.py` to be updated |
| D-M6 | Evaluation split | **test** only for reported metrics | | `eval_cams.py` default |
