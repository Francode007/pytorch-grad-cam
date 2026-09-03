# Coding Roadmap and Action Plan for the Revision

Deadline: **Sat 13 Sep 2026**. Today: Tue 2 Sep. Eleven days.
Code base: `github.com/Francode007/pytorch-grad-cam`, branch `kvasir_v1`, directory `XAI_Enhancer_module/` (commit `f035373`). All paths below are relative to that directory.

This roadmap is built from reading the actual code, not only the manuscript. Section 0 lists things in the code that change what you must do; they come before anything else.

---

## 0. BLOCKING: code-vs-manuscript discrepancies found in the repository

Resolve these on Day 1, before any GPU job starts, because they decide *which method* you are evaluating.

### 0.1 Paper masks the **input image**; code masks **layer activations**

- Paper (Sec. 3.1.2, Eq. 2–3, Fig. 1): upsample M_l → min-max normalise → X̂_l = X ⊙ M̄_l → forward pass.
- Code (`enhanced_cams/*_enhanced.py::normalize_and_mask_activations` and `utils/optimized_cam_extractor.py::compute_modified_outputs_batch`): per-channel min-max of `grads * activations`, multiplied with the layer's own activations; a **forward hook replaces that layer's output** inside a mega-batch forward pass. The input image is never masked. The only `X ⊙ map` code is `ablation/diagram_artifacts.py`, which draws the Figure 1 illustration.
- Consequence: Eq. 2–4 and Fig. 1 describe an algorithm that did not produce Tables 1–4. No reviewer caught it; a re-reviewer who opens the linked repository will.

**Decision (Day 1):**

| Option | Meaning | Cost | Verdict |
|---|---|---|---|
| **A. Rewrite the paper to match the code** | Phase 2 becomes "layer-level activation gating": Â_l = norm_c(∇A_l ⊙ A_l) ⊙ A_l; forward continues from layer l. Fig. 1 redrawn. D7 novelty text adjusted (this is an activation-level perturbation, related to Ablation-CAM/Score-CAM but at the layer axis). | Low compute, high writing | **Recommended.** It is what was measured. |
| B. Implement input masking as written, re-run everything | New `mask_mode="input"` code path; all tables regenerate. | Highest compute; outcome unknown | Only if co-authors reject A. |
| C. A as main + B as ablation on one backbone | Add the switch anyway; report both. | Medium | Do only if Tiers 1–2 finish early. |

Under A, describe accurately that each channel is gated independently and that all layers downstream of l see the modified tensor.

### 0.2 Aggregation default is `stagewise`, not the flat softmax of Eq. 5

`kvasir/eval_cams.py` and `ibs/eval_cams.py` default `--enhanced-method stagewise` → `EnhancedCAMAggregator.aggregate_stagewise` (softmax over stage-mean scores × softmax within stage). Eq. 5–6 is `{"type": "standard"}`. The ablation scripts (`softmax_temperature_validation.py`, `enhancer_weight_extraction.py`) use the flat softmax.

**Action:** locate the exact command lines / logs that produced Table 1. If `stagewise`, either re-run Table 1 with `standard` (matches Eq. 5–6 and Table 4) or describe the hierarchical scheme and re-run Table 4 with it. Tables 1 and 4 must use one aggregation. Record it in the protocol table.

### 0.3 Training hyper-parameters in the paper differ from `train.py`

Paper: Adam, lr 1e-3, F1-monitored. Code defaults: **AdamW, lr 1e-4, wd 1e-4, cosine, best checkpoint by val accuracy**. You retrain anyway (Section 2); make the paper match the new runs and log the exact args.

### 0.4 Facts the code settles for the "which backbone / which CAM" questions

| Table / Fig | Script | Backbone | Base CAM | Settings to disclose |
|---|---|---|---|---|
| Table 4 | `ablation/softmax_temperature_validation.py` | resnet50 | HiResCAM | `layer_mode=all`, flat softmax, val split |
| Table 3 | `clinical_evaluation_proxy.py` | resnet50 | Grad-CAM | **Otsu** binarisation; all 1,000 Kvasir-SEG images (no test filtering) |
| Table 5 | `benchmark_xai_overhead.py` | resnet50 | — | `layer_batch_size=4`, warm-up 50, `reset_peak_memory_stats` used |
| Table 6 | `robustness_augmentations_xai.py` | resnet50 (hard-coded) | Grad-CAM | σ=0.05 (normalised units), contrast α∈±0.25, Kvasir val |
| Fig. 2 | `ablation/layerwise_road_extraction.py` | all 5 | — | last 5 `nn.Conv2d` modules |
| Fig. 3 / Table 2 | `ablation/enhancer_weight_extraction.py` | all 5 | HiResCAM | **all `nn.Conv2d`** incl. 1×1 and `downsample` convs |
| Del/Ins | `evaluator/imagenet_proper_auc_evaluator.py` | — | — | `--step-size 224` px/step (eval_cams default); Insertion baseline = Gaussian blur k=11, σ=5 |
| LayerCAM | `eval_cams.py` | — | — | two variants: single last-layer `LayerCAM`, and `layercam_fused` (stage layers, tanh γ=2, element-wise max). State which one Table 1 used |

The 389 MB LayerCAM/HR-CAM numbers in Table 5 are not produced by `benchmark_xai_overhead.py` as written (it benchmarks resnet50 methods under one protocol); they came from elsewhere. Re-measure everything in one run.

---

## 1. Calendar (11 days)

Principle: GPU runs are the bottleneck; start them on Day 2 and write while they run. Every day ends with results committed to the repo and numbers pasted into the manuscript draft.

| Day | Date | Coding | GPU | Writing / admin |
|---|---|---|---|---|
| 1 | Tue 2 Sep (tonight) + Wed 3 | §0 decisions; §2.1 splits (patient-level IBS, 3-way Kvasir); §2.2 seed/fold-aware training + AUROC eval; smoke tests on 50 images | — | Email editor for extension (state: patient-level re-split + seeds requested); email IBS data owner if patient IDs missing |
| 2 | Thu 4 | §2.3 per-image logging + uniform baseline + `--layer-set`; §2.4 stats script | **Launch T1: all training runs** (15 Kvasir + 25 IBS) | Wording pass (`08-…`); Introduction & Related Work rewrite |
| 3 | Fri 5 | §2.5 Score-CAM/Group-CAM/Opti-CAM baselines; §2.6 similarity-function switch; §2.7 Kvasir-SEG test filtering + continuous metrics | T1 continues; run `eval_classification` as checkpoints finish | Methods rewrite (Option A description, Eq. 6 fix, notation) |
| 4 | Sat 6 | §2.8 latency/VRAM re-benchmark (batched + sparse); §2.9 robustness (protocol, conditioned SSIM, TTA) | **Launch T2: Table 1 (all methods, all seeds/folds)** — the longest job | Dataset table; data-usage paragraph; classifier table |
| 5 | Sun 7 | Bug-fix from T2 logs; §2.10 S_l–ROAD correlation + layer-set ablation script | T2 continues; T4.1 benchmark (short) | Section 4.1/4.3 rewrite from regenerated Fig. 2/3 |
| 6 | Mon 8 | Stats over T2 outputs (CIs, Wilcoxon, win/tie/loss) | T2.4 Kvasir-SEG; T2.5 T-sensitivity (val→test); T3.1 similarity ablation | Results Sec. 4.2 with new numbers |
| 7 | Tue 9 | — | T3.3 layer-set/sparse ablation; T3.4 correlation; T4.3 robustness + TTA; T3.7 frequency-band ablation | Discussion + Limitations subsection; Table 5/6 text |
| 8 | Wed 10 | Figures regenerated (Fig. 2, 3 box plots, 4, 5, 6, latency-vs-ROAD) | Any re-runs | Abstract, Conclusion, title |
| 9 | Thu 11 | Freeze code; tag release; update README with reproduction commands | Optional T5 ViT only if everything else is done | Response letter (`07-…`) filled with numbers |
| 10 | Fri 12 | — | — | Co-author review; language edit; supplementary protocol table |
| 11 | Sat 13 | — | — | Submit (or submit by extended deadline if granted) |

If the extension is granted, keep the same order and use the extra time for Option C (§0.1), Opti-CAM on the full set, and the ViT experiment.

---

## 2. Code changes, by priority

Each item: files, what to change, acceptance test. Estimated size is in lines of code so you can schedule, not in hours.

### 2.1 Splits: patient-level IBS, 3-way Kvasir (P0)

**Files:** `ibs/data.py`, `kvasir/data.py`, new `common/splits.py`.

- `kvasir/data.py::prepare_splits` → produce `train.txt / val.txt / test.txt` (default 70/10/20, stratified per class, seeded). Keep the old 2-way function under a deprecated name so old logs remain interpretable.
- `ibs/data.py`: add `extract_group_id(path) -> str`. Inspect the Kaggle `pre-processed-ibs` filenames and the original Dryad zips (`IBS-*.zip`, `IBS-C-*.zip`, `IBS-D-*.zip`, `normal-*.zip`) for exam/patient prefixes, sub-folders or EXIF timestamps. Implement whichever exists; raise loudly if none.
- New `prepare_patient_folds(data_root, n_folds=5, seed)` using `sklearn.model_selection.StratifiedGroupKFold` (groups = patient/exam id, strat = N/I/C/D subtype), writing `splits/fold{k}/{train,val,test}.txt` with an inner val split carved from train by group.
- `common/splits.py::write_split_summary()` → CSV with patients and images per class per split/fold (this is the new dataset table).
- Near-duplicate guard for Kvasir: `imagehash.phash` on all 8,000 images, flag pairs with Hamming ≤ 6 across splits, move to the same split, log count.

**Acceptance:** assert no group id appears in two IBS splits; assert `len(train)+len(val)+len(test) == N`; summary CSV renders.

### 2.2 Training: seeds, folds, proper model selection, classifier metrics (P0)

**Files:** `ibs/train.py`, `kvasir/train.py`, `*/eval_classification.py`, new `common/train_matrix.py`.

- Add `--fold k` (IBS) and honour `--seed` for init + shuffling (`torch.manual_seed`, `np.random.seed`, `random.seed`, `DataLoader(generator=...)`).
- Select best checkpoint by **val macro-F1** (paper says F1; code uses accuracy). Log per-epoch loss/acc/F1 to JSON.
- Save `args.json` beside each checkpoint (this is your protocol record).
- `eval_classification.py`: add AUROC (binary; macro OvR for Kvasir via `roc_auc_score(multi_class="ovr")`), per-class precision/recall/F1/support, confusion matrix, ECE and mean max-softmax; run on `test`.
- `train_matrix.py`: loops archs × seeds (Kvasir: 5×3) and archs × folds (IBS: 5×5); writes `runs/{dataset}/{arch}/seed{s}|fold{k}/`.

**Acceptance:** `eval_classification --split test` produces a JSON per run; an aggregation script prints mean ± SD and 95% CI per arch.

### 2.3 Evaluation harness: per-image logging, uniform baseline, layer sets, test split (P0)

**Files:** `evaluator/imagenet_proper_auc_evaluator.py`, `kvasir/eval_cams.py`, `ibs/eval_cams.py`, `enhanced_combiner/aggregator.py`.

- Persist **per-image** rows (`image_id, method, ins, del, road_20..80_morf/lerf, road, pred, conf`) to a Parquet/CSV per run. Every statistic afterwards is computed from these files; nothing is re-perturbed.
- `aggregator.py`: add `{"type": "uniform"}` (w_l = 1/L). This is the T→∞ baseline.
- `--layer-set {all, conv3x3, block_outputs, stage_outputs, last_5, last}`; implement in `_get_enhanced_cam_layers` for ResNet (block = `layerX[i]` output; stage = `layerX` output) and VGG (block = each conv; stage = last conv before each pool). `block_outputs` and `stage_outputs` are the **sparse variant** for R1.
- `--split test` default everywhere; refuse to run on `val` unless `--allow-val`.
- Fix `--enhanced-method` default to whatever §0.2 decided; print it in the header of every results file.
- Set `--step-size` explicitly and record it (224 px/step is coarse: 224 steps for a 224² image is fine, but state it).
- Fix ROAD imputation seed (`torch.manual_seed` before noisy imputation) and record it.

### 2.4 Statistics script (P0)

**New:** `analysis/stats.py`.

- Input: per-image CSVs. Output: per (dataset, arch, method, metric) mean ± SD across seeds/folds, bootstrap 95% CI over images (2,000 resamples), paired Wilcoxon signed-rank vs. base CAM and vs. LayerCAM/HR-CAM/uniform (Holm within table), Cliff's δ.
- Emit Table 1 (LaTeX, mean ± SD, bold only where p < 0.05) and a win/tie/loss table (tie = not significant).

### 2.5 Baselines: Score-CAM, Group-CAM, Opti-CAM, composability (P1)

**Files:** `eval_cams.py`, new `baselines/group_cam.py`, `baselines/opti_cam.py`.

- Score-CAM is already in `pytorch_grad_cam`; enable `scorecam` in `--methods` for the terminal layer.
- Group-CAM: port `wofmanaf/Group-CAM` (groups=32, blur-blend). ~150 LOC.
- Opti-CAM: port official implementation (Adam, 100 iterations, target-logit objective); run on a 400-image test subset per backbone if full set is too slow; report N.
- Composability: `--base-cam ScoreCAM` already maps to `ScoreCAMEnhanced`; run it on one backbone per dataset.

### 2.6 Similarity-function switch (P1)

**Files:** `utils/optimized_cam_extractor.py::compute_cosine_similarities`, `enhanced_combiner/extractor_v2.py`.

- `--similarity {cosine, centered_cosine, prob_cosine, neg_kl, neg_js, target_ratio}`; centering = subtract per-sample mean over classes before cosine.
- `--standardize-scores` (per-image z-score of S_l across layers before softmax).
- Log raw S_l per layer per image (needed for §2.10 and for Fig. 3 box plots).

### 2.7 Kvasir-SEG alignment: test-only, continuous metrics, all baselines (P1)

**File:** `clinical_evaluation_proxy.py`.

- Filter Kvasir-SEG filenames to those in `splits/test.txt` of the same run; print N.
- Metrics: keep Otsu IoU/Dice (state Otsu), add energy-based pointing game (saliency mass inside mask), pointing-game hit rate, soft Dice, Dice-vs-threshold AUC over thresholds 0.1..0.9.
- Loop over all methods from §2.3/§2.5, all backbones (at least resnet50 + vgg16), and the uniform baseline.
- Bug to fix while there: `build_kvasir_model(pretrained=True)` downloads ImageNet weights before loading the checkpoint — harmless but slow; use `pretrained=False`.

### 2.8 Latency and VRAM re-benchmark (P1)

**File:** `benchmark_xai_overhead.py`.

- Parametrise `--arch` (all five), `--layer-set`, `--layer-batch-size {1, 4, 16, all}`; add Score-CAM, Group-CAM, Opti-CAM, LayerCAM(-fused), HR-CAM under the **same** warm-up/reset protocol; median over ≥100 images; also report CPU latency for resnet50 once.
- Output CSV → new Table 5 and the latency-vs-ROAD figure (join with Table 1 by method).

### 2.9 Robustness protocol, conditioned stability, TTA (P1)

**File:** `robustness_augmentations_xai.py`.

- Un-hard-code resnet50; take `--arch --checkpoint --split test`.
- σ ∈ {0.01, 0.03, 0.05}, α ∈ {±0.1, ±0.25}; fixed seed.
- Record for each image: prediction before/after, |Δconf|, SSIM/Pearson for Grad-CAM, HiResCAM, LayerCAM-fused, HR-CAM, uniform, XAI-Enhancer; ROAD of each method on the perturbed image.
- TTA variant: `--tta n` averages the enhancer map over n noisy copies (n = 4, 8); report SSIM/Pearson, ROAD, latency.
- Report stability overall and conditioned on unchanged prediction.

### 2.10 Method-validation scripts (P2)

- `ablation/weight_road_correlation.py` already exists — extend to all block outputs, all archs/datasets, Spearman ρ between mean S_l and per-layer ROAD; box plots of S_l per layer (replaces Fig. 3 mean±SD); name the module at the ResNet-50 dip.
- `ablation/layerwise_road_extraction.py`: extend from last 5 to all block/stage outputs so Fig. 2 spans real resolution changes.
- New `ablation/frequency_ablation.py`: classifier accuracy/AUROC vs Gaussian blur σ ∈ {0, 1, 2, 3, 5} px on IBS vs Kvasir test sets (R3-Q1.2 evidence).
- If §0.1 Option C: `--mask-mode {activation, input}` in the enhanced CAM classes; input mode = upsample CAM, min-max, `X ⊙ M̄` on the normalised tensor (state this), forward.

### 2.11 Repository hygiene (P2, Day 9)

- Remove/relocate `old_versions/`, `test_files/`, `analysis_results/*.pkl`, `.DS_Store`, ImageNet/Colab leftovers from the module root, or move them to `legacy/`.
- Top-level `REPRODUCE.md` with the exact commands for every table/figure, seeds, and the protocol table (`analysis/protocol_table.csv`) that lists, per table/figure: dataset, split, N, backbone, base CAM, layer set, aggregation, similarity, T, step size, perturbation params, seeds.
- Tag `v2-revision` and cite the tag in the paper's Code Availability.

---

## 3. GPU run matrix

| ID | Job | Count | Notes |
|---|---|---|---|
| T1 | Train Kvasir 5 archs × 3 seeds; IBS 5 archs × 5 folds | 40 runs × 50 epochs | Use `--a100` preset (AMP bf16, batch 128). Run sequentially per GPU; parallelise across GPUs if available. |
| T1b | `eval_classification --split test` per run | 40 | Seconds each. |
| T2 | Table 1: base {GradCAM, GradCAM++, HiResCAM} + enhanced ×3 + uniform + LayerCAM + LayerCAM-fused + HR-CAM + ScoreCAM + GroupCAM (+ OptiCAM subset), per run, on test | 40 runs × ~12 methods | Longest job. Per-image CSVs. HR-CAM head must be trained per run (`--hrcam-epochs 20`). |
| T2.4 | Kvasir-SEG alignment on test-split images | 15 Kvasir runs × methods | Short. |
| T2.5 | T ∈ {0.1, 0.5, 1, 2, 5, 10, ∞} on val (select) then test (report), per arch | 15 × 7 | Reuse cached S_l and per-layer CAMs; only the aggregation changes. |
| T3.1 | Similarity functions × {resnet18, resnet50, vgg16} × both datasets | 6 × 6 | Reuse cached logits where possible. |
| T3.3 | Layer sets {all, conv3x3, block, stage} × all archs, one seed | 20 | Also produces sparse-variant latency. |
| T3.4 | Per-layer ROAD at block outputs + S_l correlation | 10 | |
| T3.7 | Frequency-band ablation | 10 | Minutes. |
| T4.1 | Latency/VRAM benchmark | 1 script | ~1 h. |
| T4.3 | Robustness + TTA | 10 runs (one seed) | |
| T5 | ViT-B/16 on Kvasir (optional) | 1 train + eval | Only if T1–T4 are done. |

Order of launch: T1 → (T1b as checkpoints land) → T2 → T4.1 → T2.4/T2.5/T3.1 → T3.3/T3.4/T4.3/T3.7 → T5.

---

## 4. Writing tasks tied to results (owner: you; reviewer: co-authors)

| When | Task |
|---|---|
| Day 1 | Extension request to editor. Draft the §0.1 Option A method description and redraw Fig. 1. |
| Day 2–3 | Introduction (hypothesis wording, general motivation, contributions per D7), Related Work (perturbation-weighted CAM paragraph), Methods (Eq. 6 → M_l^up, S_l notation, centering if adopted, masking description, data-usage paragraph). |
| Day 4 | Dataset table (patients/images per class per split), classifier table, protocol table skeleton. |
| Day 5–6 | Sec. 4.1/4.3/5.1 from regenerated Fig. 2/3 with one consistent story; Sec. 4.2 from `stats.py` output; win/tie/loss table; discuss IBS/ResNet-18 and ResNet-34. |
| Day 7 | Sec. 4.5 (test-only, continuous metrics), 4.6 renamed to sensitivity, Sec. 5.3–5.4 with new Tables 5–6, new Limitations 5.6, 5.5 reduced to future work. |
| Day 8 | Abstract (limitations sentence, no "consistently/highly faithful"), Conclusion, title. |
| Day 9 | Response letter (`07-response-letter-skeleton.md`) with numbers; every reviewer point cross-referenced to a page/table. |
| Day 10 | Language edit; supplementary; final consistency check of every number between tables, text and response letter. |

---

## 5. Scope control: what to cut if time runs short (in this order)

1. Opti-CAM on full sets → subset of 400 images with CI (say so).
2. ViT experiment → future-work sentence only (D4 scope-only text).
3. Option C input-masking ablation → not run; Option A text only.
4. Similarity ablation → resnet50 + resnet18 on Kvasir only.
5. TTA robustness → n = 4 only, resnet50 only.
6. IBS 5 folds → 3 folds (never fewer; never image-level).

Never cut: patient-level IBS splits, 3-way Kvasir split, classifier metrics, CIs/paired tests, uniform-average baseline, §0.1/§0.2 consistency between paper and code, limitations in Abstract/Conclusion.

---

## 6. Deliverables checklist (tick before submission)

- [ ] §0.1 decision recorded; Methods and Fig. 1 describe what the code does
- [ ] §0.2 aggregation identical across Tables 1 and 4 and stated in captions
- [ ] Patient-level IBS folds; 3-way Kvasir split; near-duplicate scan logged
- [ ] Dataset table: patients and images per class per split/fold
- [ ] Classifier table on test: acc, macro-F1, AUROC, calibration, mean ± SD, CI
- [ ] Table 1 on test with mean ± SD, CIs, paired tests, uniform + Score/Group/Opti-CAM rows
- [ ] Win/tie/loss table; "consistently/universal" removed everywhere
- [ ] Table 3 on test-split Kvasir-SEG images, continuous metrics, all baselines, backbone + Otsu stated
- [ ] Table 4 as sensitivity analysis, val-selected/test-reported, per-arch in supplement
- [ ] Table 5 one protocol, all methods, per backbone, sequential vs batched, sparse variant
- [ ] Table 6 protocol stated, all baselines, prediction-conditioned, TTA row
- [ ] Fig. 2 extended; Fig. 3 box plots; dip module named; S_l–ROAD ρ reported
- [ ] Similarity-function ablation; centering decision stated
- [ ] Frequency-band ablation (or hypothesis wording if inconclusive)
- [ ] Limitations in Abstract, new 5.6, Conclusion; ViT text is future work only
- [ ] Equation refs, notation, Eq. 6, captions/text contradictions fixed
- [ ] Protocol table in supplement; `REPRODUCE.md`; tagged release cited
- [ ] Response letter: every numbered point answered with page/table reference
- [ ] Extension confirmation (or submission by 13 Sep)
