# Self-Identified Issues (not raised by the reviewers)

These are problems found by reading the manuscript closely against its own numbers and against the public documentation of the datasets and baselines. None of them were flagged by R1–R3, but a revised manuscript "may need to be re-reviewed" (decision letter), possibly by a new reviewer. Fixing them now is cheaper than a second major revision. Ordered by severity.

---

## S1 — The manuscript's own Table 4 shows uniform averaging beats the learned weighting (HIGH)

**Where.** Table 4 / Section 4.6, Table 2, Eq. 5.

**What.** As T → ∞, w_l → 1/L (plain average of all layer CAMs). Table 4 shows ROAD, Insertion and Deletion all improve monotonically toward that limit (ROAD 0.1591 at T=0.1 → 0.1638 at T=10). Table 2 shows the T=1 weights already have SD 0.0013–0.0017 around 1/L, i.e., they are within a few percent of uniform. So the "dynamic, image-specific" logit-similarity weighting — the stated key novelty — has no demonstrated positive effect on faithfulness; the improvement over single-layer CAMs in Table 1 is attributable to multi-layer averaging.

**Why it matters.** The abstract says the method works "by dynamically compensating for these bottlenecks"; the contributions say the enhancer "dynamically aggregates layer-specific importance". A re-reviewer who reads Table 4 carefully can argue the paper's mechanism claim is contradicted by its own ablation.

**Fix.** (1) Add a **uniform-average (T→∞) baseline** row to Table 1 for every configuration. (2) Search for the regime where weighting matters (ResNet-18/34 where raw SD is 0.36–0.47; images with degenerate layers; IBS). (3) Make the similarity discriminative: centered cosine (D6) and/or per-image z-scoring of S_l across layers before softmax, then re-sweep T. (4) If weighting still does not help, reposition: the contribution is (a) training-free multi-layer aggregation that is robustly better than any single layer and than LayerCAM/HR-CAM, and (b) the per-layer masked-similarity as a *diagnostic* of information flow. Rewrite abstract and contributions accordingly. See D9.

---

## S2 — Kvasir-SEG is the Kvasir-v2 polyp class: Section 4.5 evaluates alignment on training images (HIGH)

**Where.** Section 4.5, Table 3, Figure 5.

**What.** Kvasir-SEG's 1,000 images are the polyp class of Kvasir-v2 with added masks (the dataset authors state the segmented images "have duplicates in the images folder of polyps since the images were taken from there"). The classifier was trained on 80% of Kvasir-v2 at image level, so ~800 of the 1,000 Kvasir-SEG images were training images. Localisation agreement on training images is not evidence of clinically meaningful attention.

**Fix.** Restrict the alignment evaluation to Kvasir-SEG images that are in the held-out test split (~200 under an 80/20 split; more if the split is redesigned), match by filename, and state N. Combine with R2.3 (continuous metrics, all baselines, backbone/threshold stated).

---

## S3 — Temperature and early stopping were selected on the same images used for every reported result (MEDIUM–HIGH)

**Where.** Sec. 3.2.2 (F1 on the validation split guides training), Sec. 4.3 ("all images in the validation datasets"), Sec. 4.6 ("Kvasir-v2 validation cohort (N = 1600)" = 20% of 8,000).

**What.** One 20% split serves as early-stopping set, hyperparameter-selection set and final evaluation set. The effect on T is negligible (T barely matters), but the *protocol* is improper and R3-2 explicitly asks "which data were used for training, model selection, temperature selection, and final evaluation".

**Fix.** Train/val/test for Kvasir-v2; patient-level CV with inner validation for IBS (D8). Select on val; report on test. Add a data-usage paragraph or flowchart.

---

## S4 — The "final heavy spatial compression at Index −1" explanation for VGG is factually wrong (MEDIUM)

**Where.** Sec. 4.1, Figure 2a and caption; Introduction paragraph 3.

**What.** For a 224×224 input, VGG-16's last five conv layers are conv4_2 (28×28), conv4_3 (28×28), conv5_1, conv5_2, conv5_3 (all 14×14). Max-pooling happens *after* conv5_3, not between Index −2 and Index −1. So the "sudden and precipitous degradation" between conv5_2 and conv5_3 cannot be caused by spatial compression; both are 14×14. Similarly, ResNet-50's last five conv modules all live in `layer4` at 7×7 and include 1×1 convs; differences among them are not about resolution either. The mechanism the paper invokes does not match the layers it measured.

**Why it matters.** It is the central mechanistic claim of the Introduction ("successive pooling ... lossy low-pass filter"), and it is checkable by anyone who knows the architectures. R2 already noticed the motivation is CNN-specific; a re-reviewer may notice it is also wrong for these specific layers.

**Fix.** Reframe: layers differ in *semantic abstraction, sparsity and class-specificity*, not only resolution; the final layer's features are the most class-discriminative but also the most spatially diffuse *as a CAM*, for reasons that include resolution (across stages) and feature selectivity (within a stage). Describe Figure 2 as an empirical finding about depth, not as a consequence of pooling. Extend Figure 2 beyond the last five layers so that actual resolution changes are visible.

---

## S5 — Figure 2 (per-layer ROAD) and Figure 3 (per-layer cosine similarity) disagree, and S_l is never validated as a faithfulness proxy (MEDIUM–HIGH)

**Where.** Sec. 4.1 vs Sec. 4.3.

**What.** Figure 2a says VGG's per-layer ROAD swings sharply across the last five layers; Figure 3b says VGG-16's per-layer similarity is "remarkably stable". Figure 3a says ResNet-18's late layers have "catastrophic" low similarity; Table 1 says terminal-layer Grad-CAM on IBS/ResNet-18 has the highest ROAD in the entire IBS table (0.461). If S_l measured "the objective faithfulness of every intermediate layer" (Introduction), it should track per-layer ROAD. The paper never tests this.

**Fix.** Compute, per backbone × dataset, the Spearman correlation between mean S_l and per-layer ROAD over the same layers (extend Figure 2 to all layers or at least to the block outputs). Report it. If strong: Phase 2 is validated. If weak: S_l measures something else (probably how OOD the masked image is — see S10), and the paper must say so and lean on S1's repositioning.

---

## S6 — Internal contradictions and caption/text mismatches (MEDIUM)

- Fig. 3a caption: "severe spatial bottlenecks in shallower residual networks (ResNet-18, ResNet-34)". Sec. 5.1: "ResNet-18 preserves textural features, ResNet-34 destroys them". These cannot both be true.
- Fig. 2 caption: intermediate layers "(Index −3) consistently outperform the final layer". Sec. 4.1 text: VGG peaks "optimally at Index −2"; ResNet-50 peaks at "Index −3". Caption generalises one case.
- Abstract: "exposes severe, dataset-dependent structural bottlenecks in residual networks". Sec. 4.1: sequential VGG shows "a sudden and precipitous degradation". Which family has the problem depends on which figure one reads.
- Sec. 4.2.3: "generalizes ... without dataset-specific hyperparameter tuning" — true, but T was tuned on Kvasir-v2 only (Table 4) and applied to IBS.

**Fix.** Rewrite Sections 4.1, 4.3, 5.1 from the (regenerated) figures with one consistent story.

---

## S7 — Equation and notation errors (LOW, but signals carelessness)

- Sec. 4.3: "raw, pre-softmax logit-similarity scores (α_l) calculated using Equation (3)" — Eq. 3 is the masking; the similarity is Eq. 4.
- Sec. 4.3: "our methodology applies a softmax function ... (Equation 4)" — the softmax is Eq. 5.
- Symbol for similarity is S_l in Eqs. 4–5 and α_l in Sec. 4.3 and Fig. 3.
- Eq. 6 sums w_l · M_l, but the M_l have different spatial sizes; it must be M_l^up (the upsampled maps of Sec. 3.1.1).
- Fig. 3 y-axis extends to 1.5 and −1.0 for a quantity bounded in [−1, 1]; clarify the shaded band is ±1 SD.
- Fig. 1 labels "Softmax (Regularization)"; Table 2 is titled "Regularization Proof". Softmax over L values in [−1, 1] trivially has smaller SD than the inputs; this is arithmetic, not regularisation or proof. Rename ("Weight distribution after softmax").
- Data Availability cites the IBS dataset as [5] (the 2024 XAI paper); the dataset paper is [36]. Ethics statement says "dataset" (singular) for three datasets.

---

## S8 — IBS image count and per-class counts (MEDIUM)

Manuscript: 5,547 images. Source study (Tabata et al. 2023, and the Dryad record): 2,479 + 382 + 538 + 484 = 3,883 images used, after excluding terminal-ileum, retroflexion, NBI/dye frames and poor bowel-prep frames. Either the Dryad release is larger than the used subset (state this and give counts) or the number is wrong. Per-class image counts, and the class balance of the binary task (35 vs 88 patients; image ratio unknown), are never given. Add a dataset table (patients and images per class per split — R3-1 asks for exactly this).

---

## S9 — Table 5 is internally inconsistent and under-specified (MEDIUM)

- Grad-CAM 2,527 MB vs LayerCAM/HR-CAM 389 MB on (presumably) the same backbone is not plausible under a single measurement protocol (see D2).
- One latency figure for a method whose cost scales with L ∈ {13, …, 53}; backbone unstated.
- The claimed batchability (Sec. 3.1.3) was not benchmarked (see D1).

**Fix.** Re-measure everything under one protocol; report per backbone; add sequential and batched modes; add new baselines.

---

## S10 — Phase 2 uses hard multiplicative masking, the OOD artefact the paper itself criticises (MEDIUM)

**Where.** Eq. 3 vs Sec. 2.3 and Sec. 3.2.3.

**What.** The paper argues (correctly) that Deletion/Insertion "induce severe Out-Of-Distribution (OOD) artifacts" and motivates ROAD for that reason. But its own weighting mechanism multiplies the image by a [0,1] mask, driving most pixels to zero (black, or mean-colour after normalisation — the text does not say whether X is the raw or normalised tensor). This is the same OOD perturbation. Group-CAM explicitly blends the masked region with a blurred copy of the input to avoid this. S_l may therefore partly measure how OOD each layer's mask makes the image, not how faithful the layer is (which would also explain S5).

**Fix.** (1) State whether masking is applied before or after normalisation. (2) Ablate the imputation: zero / mean / Gaussian-blurred background / ROAD-style noisy linear imputation. (3) Adopt the one that gives the best S_l–ROAD correlation (S5). This is a modest experiment with high explanatory value.

---

## S11 — Probable transcription error in the IBS/ResNet-18 rows of Table 1 (MEDIUM)

Base Grad-CAM and base HiResCAM: identical Insertion (0.973) and Deletion (0.214) but different ROAD (0.461 vs 0.442). Enhanced Grad-CAM and enhanced Grad-CAM++: identical ROAD (0.442) and Insertion (0.991) with different Deletion. Base Grad-CAM and base Grad-CAM++: identical ROAD (0.461). The reviewer's headline counterexample (R3-5) lives in exactly these cells. Re-derive the block from raw logs before writing the response; if it was a copy error, disclose and correct it. Also explain why IBS/ResNet-18 ROAD is 0.44–0.48 when every other IBS configuration is ≤ 0.30 — this is an outlier that a classifier-performance table (R3-2) will probably illuminate (e.g., a saturated, highly confident model).

---

## S12 — Under-specified evaluation protocol (MEDIUM)

Missing from the manuscript, all needed for reproducibility (R2 marked "No"):

- The layer set L per backbone (Table 2 implies *every* conv module including 1×1 bottleneck and shortcut-projection convs; confirm).
- Hook location: module output (pre-BN, pre-residual-add, pre-ReLU) vs. block output. This changes Grad-CAM vs HiResCAM equivalence at the final layer and the meaning of "terminal layer".
- LayerCAM baseline: which layers were fused and by which rule (LayerCAM's paper uses scaling + element-wise max/sum over selected layers).
- HR-CAM: which levels; how it was trained; on which data.
- Deletion/Insertion: step size (pixels per step), blur kernel for the Insertion baseline, number of steps, whether scores are probabilities.
- ROAD: imputation noise level; the four thresholds are given (20/40/60/80%) but not the seed.
- Table 3: binarisation threshold; backbone.
- Table 6: noise σ, contrast factors, N, backbone, dataset.
- Number of images evaluated per dataset for Table 1 (N=1,600 for Kvasir-v2 is implied; IBS unspecified).
- Training details: pretrained on ImageNet (implied by "pre-trained" and ImageNet normalisation) — say so; which layers fine-tuned; weight decay; LR schedule; seed.

**Fix.** A supplementary "Evaluation protocol" table with one row per table/figure.

---

## S13 — Tone/precision issues beyond R3's list (LOW)

"rigorous" ×4, "comprehensive" ×5, "remarkable/remarkably", "brilliantly expose", "seamlessly", "dramatically", "vastly outperforming", "paradigm shift", "mathematically faithful", "mathematically ensures", "democratize the feature ensemble". Also "superior explanation maps" in the abstract's Methods sentence (a methods sentence should not assert the result). Full replacement list in `08-manuscript-wording-edits.md`.
