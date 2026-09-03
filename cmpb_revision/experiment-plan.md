# Experiment Plan for the Revision

Ordered so that mandatory items come first and every later experiment reuses the models and per-image logs produced earlier. Each entry states which review points it closes. Compute notes assume the A100 used in the paper.

---

## Tier 0 — Before any GPU time

| # | Task | Closes |
|---|---|---|
| 0.1 | Inspect the IBS Dryad archives for exam/patient identifiers (filenames, folder structure, EXIF, README). If absent, email the data owner for a per-image patient mapping. Decide the IBS protocol (D8). | R3-1 |
| 0.2 | Reconcile the IBS image count (5,547 vs 3,883) and produce a per-class image table. | R3-1, S8 |
| 0.3 | Re-derive the IBS/ResNet-18 block of Table 1 from raw logs; check for transcription errors. | R3-5, S11 |
| 0.4 | Perceptual-hash near-duplicate scan of Kvasir-v2 across the new splits; log pairs. | R3-1 (Kvasir part) |
| 0.5 | Build a single evaluation-config file that records, per table/figure: dataset, split, N, backbone, base CAM, layer set, T, perturbation params, seeds. This becomes the supplementary protocol table. | R2.2, S12 |
| 0.6 | Request a deadline extension from the editor, citing patient-level re-splitting and seed repetition. | — |

## Tier 1 — Mandatory retraining and classifier reporting

| # | Task | Closes |
|---|---|---|
| 1.1 | **Kvasir-v2**: stratified train/val/test (e.g., 70/10/20). Train 5 architectures × 3 seeds = 15 runs. Early stopping on val F1. | R3-1, R3-2, R3-3, S3 |
| 1.2 | **IBS**: stratified 5-fold patient-level CV (stratify by N/I/C/D), inner val split. 5 architectures × 5 folds = 25 runs. | R3-1, R3-2, R3-3 |
| 1.3 | **Classifier table**: accuracy, macro-F1, AUROC (binary / macro OvR), per-class support, mean ± SD across seeds/folds, 95% CI; calibration (ECE) and mean max-softmax. | R3-2, R3-Q3 |
| 1.4 | Per-image logging infrastructure: store per-image ROAD/Ins/Del and per-layer S_l for every method so all later statistics are recomputable without re-running perturbations. | R3-3, D5 |

Compute: 40 fine-tuning runs of 50 epochs on 4–6k images at 224 px. ResNet-50 is the slowest; with mixed precision each run is in the tens of minutes, so Tier 1 training is a matter of hours, not days. The perturbation metrics (ROAD at four thresholds × MoRF/LeRF, plus Deletion/Insertion) per method per image are the dominant cost — parallelise across GPUs if available and cache perturbed-image predictions where methods share them.

## Tier 2 — Regenerate the core results with statistics and the missing baselines

| # | Task | Closes |
|---|---|---|
| 2.1 | Table 1 regenerated on **test** splits for all seeds/folds: base CAM (Grad-CAM, Grad-CAM++, HiResCAM), enhanced, LayerCAM, HR-CAM, **+ uniform multi-layer average (T→∞)**, **+ Score-CAM, Group-CAM, Opti-CAM** (terminal layer). Report mean ± SD; bootstrap 95% CI over images; paired Wilcoxon (Holm) and Cliff's δ for enhanced vs. base and vs. each multi-layer baseline. | R3-3, R3-5, R3-6, S1, D5, D7 |
| 2.2 | **Win/tie/loss table** derived from 2.1 (tie = not significant). | R3-5 |
| 2.3 | **Composability**: XAI-Enhancer with Score-CAM as the per-layer base, on one backbone per dataset. | R3-6, D7 |
| 2.4 | **Table 3 (Kvasir-SEG)** on test-split images only; continuous metrics (energy-based pointing game, pointing-game hit rate, soft Dice, Dice-vs-threshold AUC); all baselines from 2.1; state backbone. | R2.3, S2 |
| 2.5 | **Table 4 (T sensitivity)** on test with T chosen on val; per-architecture (at least VGG-16, ResNet-18, ResNet-50); include T→∞. Rename to sensitivity analysis. | R2.5, S1, S3, D9 |

## Tier 3 — Method-validation ablations (small, high explanatory value)

| # | Task | Closes |
|---|---|---|
| 3.1 | **Similarity-function ablation** (D6): raw cosine, centered cosine, probability cosine, −KL, −JS, target-prob ratio; report faithfulness and weight SD; 2–3 backbones × 1–2 datasets. | R3-4, S1 |
| 3.2 | **Per-image standardisation of S_l** before softmax (z-score across layers) and re-sweep T; does a low-T regime now beat uniform? | S1, D9 |
| 3.3 | **Layer-set ablation**: all conv modules vs. 3×3 only vs. block outputs vs. stage outputs. Report faithfulness and latency. This produces the **sparse variant (XAI-Enhancer-S)**. | R1.1, R2.4, D1, D10 |
| 3.4 | **S_l vs per-layer ROAD correlation** (Spearman) per backbone × dataset, over block outputs (extend Figure 2 to all blocks). | S5, D10 |
| 3.5 | **Masking imputation ablation** (S10): zero / mean / blurred background / noisy-linear imputation in Eq. 3; state pre- vs post-normalisation masking. Pick the imputation with the best 3.4 correlation. | S10, S5 |
| 3.6 | **Figure 3 regenerated**: per-layer box plots; identify the module at the ResNet-50 dip; version with 3×3/block-only layer set. | R2.4, D10 |
| 3.7 | **Frequency-band ablation of the classifiers**: accuracy/AUROC vs. Gaussian blur σ (or Fourier high-frequency removal) for IBS vs Kvasir-v2 test images. Direct evidence (or not) that the IBS signal is high-frequency. | R3-Q1.2 |

## Tier 4 — Deployment and robustness

| # | Task | Closes |
|---|---|---|
| 4.1 | **Table 5 re-measured** under one protocol (same backbone, warm-up, peak-memory reset per method, median over ≥100 images): all methods incl. Score-CAM/Group-CAM/Opti-CAM; per backbone; sequential vs. **batched** masked inference; XAI-Enhancer-S. Optional CPU latency for one backbone. | R1.1, R1.2, R3-7, S9, D1, D2 |
| 4.2 | **Latency-vs-ROAD trade-off figure** from 4.1 + 2.1 + 3.3. | R1.1, R3-7 |
| 4.3 | **Table 6 (robustness) redone**: specify σ / contrast / N / backbone; all baselines incl. LayerCAM, HR-CAM; report prediction-flip rate and mean Δconfidence; SSIM/Pearson conditioned on unchanged prediction; ROAD on perturbed inputs; **TTA variant** (n = 4, 8 noisy copies) with SSIM/Pearson + ROAD + latency. | R1.3, R3-7, D3 |

## Tier 5 — Optional, only after Tiers 0–4 are done

| # | Task | Closes |
|---|---|---|
| 5.1 | ViT-B/16 or DeiT-S on Kvasir-v2 (same split protocol); per-block token-grid Grad-CAM → enhancer; compare with last-block Grad-CAM, uniform average, attention roll-out, Chefer et al. relevance. Supplementary. | R1.4, D4 |

## Tier 6 — Writing

| # | Task | Closes |
|---|---|---|
| 6.1 | Abstract: add limitations sentence (latency, stability); remove "consistently", "highly faithful", "superior"; state post-hoc nature; align mechanism claim with S1 outcome. | R3-5, R3-7, R3-Q1.1, R3-8 |
| 6.2 | Title: drop "highly faithful". | R3-8 |
| 6.3 | Introduction: reframe motivation (depth-dependent abstraction; resolution as one CNN-specific mechanism); "fail fundamentally" → hypothesis; contributions list rewritten per D7. | R2.1, R3-Q1.2, R3-6, S4 |
| 6.4 | Related Work: add perturbation-weighted CAM paragraph (Score-CAM, Ablation-CAM, Group-CAM, Opti-CAM). | R3-6 |
| 6.5 | Methods: fix Eq. 6 (M_l^up); unify S_l/α_l; describe centering if adopted; specify masking pre/post normalisation; add data-usage paragraph; add dataset table (patients/images per class per split). | R3-1, R3-2, R3-4, S7, S12 |
| 6.6 | Results: rewrite 4.1/4.3/5.1 from the regenerated figures with one consistent story; fix equation cross-references; rename 4.6; rename Table 2. | S4–S7, R2.5 |
| 6.7 | Discussion: new Limitations subsection (patient metadata for Kvasir; K=2 for IBS; latency; stability; CNN-only; single-centre IBS data). | R3-7, D4, D8 |
| 6.8 | Conclusion: remove "universal applicability", "decisively", "theoretical groundwork for evolving architectures"; add limitations. | R1.4, R3-8 |
| 6.9 | Language edit pass (`08-manuscript-wording-edits.md`). | R1/R3-Q9 |
| 6.10 | Supplementary: protocol table (0.5), long-form Table 1 with CIs, per-architecture T sensitivity, similarity ablation, layer-set ablation, ViT (if done). | R2.2 |

---

## Decision points during the plan

- **After 0.1:** if IBS patient IDs are unrecoverable and the owner cannot supply a mapping → adopt the proxy-grouping fallback or demote IBS (D8 §2.2). This changes the abstract.
- **After 2.1/2.5:** if uniform averaging ≈ weighted everywhere → reposition the contribution (D9 §5, S1). If weighting helps on specific architectures → keep the mechanism claim, scoped to those cases.
- **After 3.4:** if S_l does not correlate with per-layer ROAD → describe S_l as a diagnostic of masked-input sensitivity, not as "objective faithfulness"; tighten the language in Introduction/Sec. 3.1.2.
- **After 2.1 with Opti-CAM:** if Opti-CAM wins Insertion/Deletion → emphasise ROAD, cost, composability; do not hide the cell.
