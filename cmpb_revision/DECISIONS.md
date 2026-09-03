# Locked decisions for the revision

Update this file when a choice is final. Code and manuscript must match these entries.

| ID | Topic | Decision | Date | Notes |
|---|---|---|---|---|
| D-M1 | Phase 2 masking | **Pending** — default **A** (rewrite paper: activation gating); confirm with co-authors | | See `09-coding-roadmap.md` §0.1 |
| D-M2 | Aggregation for Table 1 / 4 | **Pending** — target **`standard`** (flat softmax, Eq. 5); audit `kvasir_v1` logs | | Change eval default in Phase 3 |
| D-M3 | Kvasir split | **70/10/20** train/val/test, stratified, seed=42; pHash Hamming ≤ 6 near-dups reassigned | 2026-09-03 | **Phase 1A ✅ complete** on Modal `xai-enhancer-vol`. 8000 → 5491/921/1588 after 414 near-dup moves (979 pairs, 258 components) |
| D-M4 | IBS split | **5-fold patient-level CV**, inner val 15%; data = flattened `franchisn/ibs-dataset` (`IBS-patient-dataset`) with bundled `ibs/metadata/ibs_groups.csv` (**126** exam groups: 16 IBS + 110 Normal) | 2026-09-03 | **Phase 1B ✅ complete** on Modal. Numeric `pre-processed-ibs` dump has no exam IDs and cannot be fully remapped (pHash/ORB). `Normal_1/2/3` are upload shards, not clinical subtypes. `extract_group_id` recognizes `Proc…_s_f.JPG`. |
| D-M5 | Checkpoint selection | **Val macro-F1** (not accuracy) | 2026-09-03 | Implemented in `kvasir/train.py` + `ibs/train.py` (Phase 2). |
| D-M6 | Evaluation split | **test** only for reported metrics | | `eval_cams.py` default |
