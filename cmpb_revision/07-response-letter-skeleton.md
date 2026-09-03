# Response to Reviewers — Skeleton

Manuscript CMPB-D-26-01921. Placeholders in [brackets] are to be filled with the regenerated numbers. Keep each response in three parts: (1) restate the point, (2) what was changed (with section/table/page pointers), (3) any remaining disagreement, stated once and politely. The editor asked that every change be outlined and that "suitable rebuttals" accompany any comment not addressed.

---

## General response

We thank the Associate Editor and the three reviewers for a careful and constructive review. The revision makes the following major changes:

1. All IBS experiments were repeated under **patient-level five-fold cross-validation**; all Kvasir-v2 experiments were repeated on a **stratified train/validation/test split** with three seeds. A new table reports patients and images per class per split, and a data-usage paragraph specifies which images were used for training, model selection, temperature selection and reporting. (R3-1, R3-2)
2. **Classifier performance** (accuracy, macro-F1, AUROC, calibration) is now reported for every architecture on held-out test data. (R3-2, R3-Q3)
3. All explanation metrics are reported with **means ± SD across seeds/folds, bootstrap 95% CIs over images, paired significance tests and effect sizes**, and a win/tie/loss summary replaces categorical claims of consistent superiority. (R3-3, R3-5)
4. **New baselines**: uniform multi-layer averaging, Score-CAM, Group-CAM and Opti-CAM; Related Work now positions XAI-Enhancer precisely relative to perturbation-weighted CAM methods. (R3-6, R2.5)
5. The **similarity function** is analysed theoretically and empirically; logits are now centered before cosine similarity [if adopted]. (R3-4)
6. **Deployment**: batched inference and a sparse variant are benchmarked; Table 5 re-measured under one protocol; latency and stability limitations are stated in the Abstract, Discussion and Conclusion. (R1.1, R1.2, R3-7)
7. **Robustness**: protocol specified, prediction-conditioned stability and faithfulness under perturbation reported, TTA variant evaluated. (R1.3)
8. **Alignment evaluation** (Kvasir-SEG) restricted to test-split images, with continuous-valued metrics and all baselines. (R2.3)
9. **Wording**: promotional and causal language removed throughout, including the title; the transformer discussion is now explicitly future work; the motivation no longer rests on a CNN-specific "low-resolution bottleneck". (R1.4, R2.1, R3-8, R3-Q1)
10. Corrections to equation cross-references, notation, figure captions and internal inconsistencies identified during revision.

---

## Reviewer #1

### R1-1 (latency; sparse layer sampling)
[Use D1 §5 text. Fill: sequential → batched latency per backbone; XAI-Enhancer-S latency and ROAD; Score-CAM/Group-CAM/Opti-CAM latency. Point to Abstract, Sec. 5.3, Table 5, Fig. N, Conclusion.]

### R1-2 (VRAM)
[Use D2 §4 text. Point to Table 5 and Sec. 5.3.]

### R1-3 (robustness; TTA)
[Use D3 §5 text. Fill: σ values, flip rates, conditioned SSIM, ROAD under perturbation, TTA SSIM/ROAD/latency. Point to Table 6, Sec. 5.4, Abstract, Conclusion.]

### R1-4 (ViT experiments)
[Use D4 §4 (scope-only) or §5 (with experiment). Point to Introduction, Sec. 5.5, Conclusion, and Supplementary if applicable.]

### R1 Q8/Q9 (structure, language)
The manuscript has been language-edited and the wording changes requested by Reviewer 3 have been applied throughout.

---

## Reviewer #2

### R2-1 ("low-resolution terminal bottleneck" vs transformers)
We agree. The Introduction now motivates the method by the depth-dependence of feature abstraction and the unreliability of a priori single-layer selection, treating resolution loss as one contributing mechanism specific to CNNs. Section 5.5 no longer asserts transformer applicability. [Point to revised paragraphs.]

### R2-2 (missing ablation settings; which base CAM/backbone)
All tables and figures now state dataset, split, N, backbone and base CAM in their captions, and Supplementary Table S1 lists the full evaluation protocol for every result (layer set, hook location, T, perturbation parameters, seeds). Table 4 was computed with [backbone] and [base CAM].

### R2-3 (Table 3: binarisation, baselines)
We agree binarisation at a fixed threshold understates and destabilises agreement. Table 3 now reports [energy-based pointing game, pointing-game hit rate, soft Dice, Dice-vs-threshold AUC] for Grad-CAM, Grad-CAM++, HiResCAM, LayerCAM, HR-CAM, Score-CAM, Group-CAM, Opti-CAM, the uniform average and XAI-Enhancer on [backbone]. During this revision we also noticed that Kvasir-SEG's images are the Kvasir-v2 polyp class; the evaluation is therefore now restricted to the [N] Kvasir-SEG images in the held-out test split, and we state this in Section 4.5.

### R2-4 (Figure 3b dip)
[Use D10 §4 text. Fill: module name; correlation ρ between S_l and per-layer ROAD.]

### R2-5 (temperature)
[Use D9 §4 text. Fill: per-architecture sensitivity; uniform-average results.]

---

## Reviewer #3

### Q1-1 (explanation vs. diagnostic performance)
We have added explicit statements in the Introduction (para. N) and Conclusion that XAI-Enhancer is a post-hoc method that leaves the classifier and its predictions unchanged, and have removed the sentence in Section 5.2 that could be read as recommending architectural changes.

### Q1-2 ("fail fundamentally"; evidence for textural IBS signal)
We have reformulated the statement as a hypothesis (Introduction, para. 3) and tested it: [frequency-band ablation result — e.g., IBS classifier AUROC falls from A to B under Gaussian blur σ = s while the Kvasir-v2 classifier falls from C to D; Supplementary Fig. S-N]. [If the result is weak, say: "The evidence is suggestive but not conclusive, and we have limited the claim accordingly."]

### Q3 (classifier performance)
Table N reports accuracy, macro-F1, AUROC, per-class support and calibration for all ten models on held-out test data (mean ± SD over seeds/folds). [One sentence interpreting: e.g., IBS classifiers are highly confident, which explains the high absolute Deletion and Insertion values in Table 1.]

### Major issue 1 (patient-disjoint splits)
[Use D8 §4 text (IBS) and D8 §1 text (Kvasir-v2). Point to dataset table, Sec. 3.2.1, Sec. 3.2.2, Limitations.]

### Major issue 2 (classifier validity; data usage)
See Q3. Section 3.2.2 now contains a data-usage paragraph: [training on train; early stopping and T on val; all reported metrics on test; IBS folds as described]. Table 4 is now computed on test with T selected on validation.

### Major issue 3 (repeated runs, CIs, tests)
[Use D5 §4 text.]

### Major issue 4 (cosine similarity of logits)
[Use D6 §5 text. Fill: ablation outcome.]

### Major issue 5 (counterexamples; consistency claims)
We agree and have removed all statements of consistent or universal improvement from the Abstract, Sections 4.2 and 6. Table N (win/tie/loss with significance) shows that [e.g., applied to HiResCAM the enhancer improved ROAD significantly in X of 10 configurations and was never significantly worse; applied to Grad-CAM/Grad-CAM++ it improved ROAD in Y of 10 and was significantly worse in Z (IBS/ResNet-18)]. We discuss the IBS/ResNet-18 and IBS/ResNet-34 cases explicitly in Section 4.2. [If S11 was a transcription error: "We also identified and corrected a transcription error in the IBS/ResNet-18 rows of the original Table 1; the corrected values are …"]

### Major issue 6 (Group-CAM, Opti-CAM; novelty)
[Use D7 §6 text.]

### Major issue 7 (latency and stability in Abstract/Discussion/Conclusion)
Both limitations are now stated in the Abstract (final sentence), in a new Limitations subsection (5.6) and in the Conclusion. See also responses R1-1 and R1-3.

### Q8 (wording)
All listed terms have been removed or replaced (see `08-manuscript-wording-edits.md` for the mapping); the title has been changed to "[…]".

### Q9 (language)
The manuscript has been language-edited.
