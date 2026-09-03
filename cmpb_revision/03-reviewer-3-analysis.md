# Reviewer 3: Point-by-Point Analysis

Reviewer 3's profile: the decisive reviewer. Acknowledges relevance, the multi-architecture evaluation, and the reporting of cost/robustness, then states that "the current experimental design and analysis do not adequately support the claims of consistent superiority, high faithfulness, generalizability, or clinical trustworthiness." Marked **statistics as No**, requested classifier metrics, and gave seven numbered major issues plus three Q1 points and a wording list. Every one of R3's factual observations checks out against the manuscript. Two are non-negotiable (patient-level split; classifier metrics), two are wording, and three are partially defensible with supporting experiments.

---

## Q1.1 — Distinguish explaining a fixed classifier from improving diagnosis

> "Please distinguish clearly between improving the explanation of a fixed classifier and improving diagnostic performance. XAI-Enhancer is a post-hoc explanation method and does not itself improve the classifier's predictions."

**What the manuscript says.** Abstract: terminal-layer reliance "compromises explanation faithfulness, hindering accurate clinical decision-making"; Conclusion: "advancing the trustworthiness of AI-assisted medical diagnostics"; Section 5.2: "future diagnostic methodologies should move away from treating explainability as a post-hoc, terminal-layer afterthought, and instead adopt architecture-aware strategies that actively preserve hierarchical spatial fidelity." That last sentence in particular reads as if the method changes the model.

**Assessment.** Correct. The method never touches the classifier. **WORK (text).** Add one explicit sentence in the Introduction and one in the Conclusion: "XAI-Enhancer is a post-hoc method; it does not alter the classifier or its predictions. All classification metrics are identical with and without it." Remove or rewrite the Section 5.2 sentence.

---

## Q1.2 — "Fail fundamentally" for IBS is too strong; provide direct evidence that the IBS signal is fine-grained textural/microvascular

> "The claim that terminal-layer CAM methods 'fail fundamentally' for IBS is too strong. Please reformulate this as a limitation to be tested empirically and provide direct evidence that the relevant IBS signal consists of the proposed fine-grained textural or microvascular features."

**What the manuscript says.** Introduction: "conventional CAMs fail fundamentally when applied to complex, holistic conditions like IBS [5]. ... Conditions like IBS typically manifest as fine-grained textural anomalies and subtle microvascular deformations, which are best captured by the high-pass filters of early convolutional layers." The only support is citation [5] (Tabata et al. 2024, a Grad-CAM/XRAI study on the same dataset) and the general representation-learning reference [18].

**Assessment.**
- Correct that the claim is asserted, not shown. Nothing in the paper measures the spatial-frequency content of the IBS signal.
- Partially defensible: [5] and [36] do report that the discriminative changes are "minute endoscopic changes, which cannot typically be detected by human investigators", which supports "sub-visual" but not specifically "high-frequency texture".
- A direct test is cheap and would be a genuine contribution: **frequency-band ablation of the classifier**. Low-pass filter the test images at increasing Gaussian σ (or remove the top-k% of the Fourier spectrum) and measure accuracy/AUROC decay for the IBS classifier vs. the Kvasir-v2 classifier. If IBS accuracy collapses at mild blur while Kvasir-v2 survives, that is direct evidence that the IBS signal is high-frequency. The converse would falsify the paper's premise, which the authors should want to know.
- Also worth reporting: per-layer ROAD on IBS for the *earliest* layers (Figure 2 only covers the terminal five), since the argument is about early layers.

**Verdict: HYBRID.** Downgrade the wording to a hypothesis ("we hypothesise that ... and test this in Section X"). Add the frequency-band ablation as evidence. If the evidence is weak, say so and keep the multi-layer argument on the general grounds of unreliable single-layer selection.

---

## Q3 — Predictive performance of every classifier on a held-out test set (AUROC, F1)

> "Provide the predictive performance of every underlying classifier on a held-out test set, including at least AUROC, F1 score. XAI results cannot be interpreted adequately without establishing classifier validity."

**What the manuscript says.** Section 3.2.2: 50 epochs, batch 32, Adam 1e-3, 80:20 split, F1 monitored on the validation split. **No performance number for any model appears anywhere in the paper.**

**Assessment.** Correct and non-negotiable. Note also that several Table 1 patterns only make sense if the classifiers are examined: on IBS with VGG, Insertion ≈ 0.90 *and* Deletion ≈ 0.88–0.90 for every method, which means the predicted-class probability stays high whether pixels are inserted or deleted — a saturated, over-confident classifier on which perturbation metrics barely discriminate. Conversely IBS/ResNet-18 gives ROAD 0.44–0.48 when every other IBS configuration gives 0.01–0.30. Readers need accuracy, calibration, and confidence distributions to interpret these.

**Verdict: WORK (mandatory).** New table: per model × dataset, on the held-out **test** split: accuracy, macro-F1, AUROC (binary for IBS; macro one-vs-rest for Kvasir-v2), per-class support, mean ± SD over seeds, 95% CI. Add a sentence on calibration (ECE or mean max-softmax) because it directly affects Deletion/Insertion interpretability.

---

## Major issue 1 — Patient-disjoint splits for IBS; report patients and images per class per split

> "The IBS dataset contains 5,547 images obtained from approximately 123 participants, but the manuscript describes only an 80:20 split. Because each participant contributes multiple images, an image-level split may place images from the same patient in both training and evaluation sets. The experiments should be repeated using patient-disjoint training, validation, and test sets. The numbers of patients and images in each class and split should be reported."

**What the manuscript says.** Section 3.2.1: 5,547 images; 11+12+12+88 = 123 participants; binary Normal vs IBS. Section 3.2.2: 80:20 train/validation split at (implicitly) image level; no test set.

**Assessment.**
- Correct and the most serious point. The source study (Tabata et al., PLOS Digit Health 2023) states each patient contributed 20–40 images, ~5 per colonic segment. Images from one colonoscopy share scope, lighting, bowel-prep quality, and mucosal appearance. An image-level split lets the network identify the *patient*, not the condition. Everything downstream — classifier F1, the XAI metrics on the "validation" set, the layer-wise similarity curves in Figure 3a, and the "IBS is textural" narrative — is contaminated.
- The binary task has 35 IBS patients vs 88 normal patients. A patient-level split leaves ~7 IBS patients in a 20% test set. This is small; **patient-level k-fold cross-validation (e.g., 5-fold, stratified by group I/C/D)** is the appropriate design, and it also answers Major issue 3 (variance across folds).
- Feasibility: the Dryad deposit (doi:10.5061/dryad.9s4mw6mkp) is organised as four class-level zip archives (IBS, IBS-C, IBS-D, normal) exported from Google Drive; the public metadata does not describe patient identifiers. The authors must inspect whether filenames or EXIF encode exam/patient IDs. If they do not, see thread **D8** for the fallback (contact the data owner; if no mapping exists, cluster by exam using timestamps/colour statistics, or state the limitation explicitly and evaluate the XAI claims primarily on Kvasir-v2).
- Image count discrepancy (S8): the source publication used 2,479 + 382 + 538 + 484 = 3,883 images. The manuscript says 5,547. If the Dryad archive contains 5,547 files, say so and give per-class counts; if the manuscript's number is wrong, fix it.
- Kvasir-v2 has no patient metadata, and its authors state that near-duplicate frames of the same finding may exist across the set. This is a limitation that can only be stated, not fixed (D8).

**Verdict: WORK (mandatory).** Patient-level stratified CV for IBS; a proper train/val/test split for Kvasir-v2; new dataset table with patients and images per class per split; limitation statement for Kvasir-v2.

---

## Major issue 2 — Predictive validity of classifiers; which data were used for training, model selection, temperature selection, final evaluation

Same as Q3 plus a data-usage question. **What the manuscript says:** training on 80%, "model convergence ... guided by the F1 score evaluated on the designated validation split"; Table 4 (T selection) on "the Kvasir-v2 validation cohort (N = 1600)" = 20% of 8,000, i.e., the same split; Section 4.3 weight analysis "for all images in the validation datasets". So the same 20% was used for early stopping, for choosing T, and for every reported XAI number. That is a data-leakage chain, mild in effect (T barely matters) but formally improper.

**Verdict: WORK (mandatory).** Adopt train/val/test (Kvasir-v2 e.g. 70/10/20 stratified; IBS patient-level folds with an inner validation split). Select epoch and T on val; report on test. Add a "data usage" paragraph or a flowchart figure that states exactly which images were used for what.

---

## Major issue 3 — Single values; no repeated runs, CIs or significance tests

> "Please repeat the experiments across multiple random seeds or patient-level folds and report means, standard deviations or 95% confidence intervals between explanation methods."

**What the manuscript says.** Every number in Tables 1, 3, 4, 6 is a single point estimate. Section 4.6 calls Δ ≤ 0.001 "statistically marginal" without any statistic.

**Assessment.**
- Correct. Partially defensible in *one* respect (thread **D5**): the enhancer is deterministic given a trained model, so the only sources of variance are (i) classifier training randomness and (ii) sampling of evaluation images. Source (ii) can be quantified **without any retraining**: each metric is a per-image quantity averaged over N images, so bootstrap CIs and paired tests (Wilcoxon signed-rank on per-image base-vs-enhanced differences) are available immediately. Source (i) requires seeds/folds, which Major issue 1 forces anyway.
- Note a subtlety: many Insertion differences in Table 1 are ≤ 0.007. With N ≈ 1,600 paired images these may or may not be significant; the paired test will tell, and the manuscript must then label non-significant cells as ties rather than wins.

**Verdict: HYBRID.** Report mean ± SD over seeds/folds *and* bootstrap 95% CIs over images; paired Wilcoxon (Holm-corrected) for base vs. enhanced and for enhanced vs. LayerCAM/HR-CAM; effect sizes (Cliff's δ or paired Cohen's d). Add a win/tie/loss summary.

---

## Major issue 4 — Cosine similarity of raw logits does not measure preservation of a probability distribution or confidence

> "Raw logits are not probabilities, and cosine similarity may change under transformations that preserve the softmax distribution. Please justify this choice theoretically."

**What the manuscript says.** Eq. 4; Section 3.1.2: cosine over "the entire logit vector ... ensures the highlighted spatial features preserve the model's complete diagnostic distribution" and "penalises masked representations that might artificially maintain target class probability while inadvertently spiking the logits of incorrect pathologies."

**Assessment.**
- The reviewer's technical point is exactly right: softmax is invariant to adding a constant to every logit, cosine is not; cosine is invariant to positive scaling, softmax is not. So cosine similarity of raw logits is neither a function of the softmax distribution nor of confidence. The sentence "preserve the model's complete diagnostic distribution" is therefore inaccurate as written.
- There is nevertheless a coherent defence (thread **D6**): (a) masking darkens most of the image and reduces the overall activation magnitude, so *scale*-invariance is desirable — the method wants to know whether the *pattern* of class evidence survives, not whether confidence survives; (b) the raw logit vector carries information about all classes, which target-probability-only measures (Score-CAM's CIC, Opti-CAM's logit objective) discard; (c) the shift-sensitivity is trivially fixed by **centering the logits** (subtracting the mean over classes) before cosine, which makes the score invariant to exactly the shifts that preserve softmax while keeping scale-invariance. Centering also removes the shared component that inflates all cosine values toward 1 and compresses the weight dynamic range (S1/R2.5).
- The honest response is an **ablation over similarity functions**: raw-logit cosine (current), centered-logit cosine, cosine on softmax probabilities, negative KL/JS divergence between softmax distributions, target-class probability ratio. Report Table 1-style metrics for each. If they all agree within noise, that itself answers the reviewer ("the choice is not consequential"); if centered cosine or KL is better, adopt it.
- For the binary IBS task the logit vector is 2-D and cosine reduces to the angle between two 2-vectors — a weak signal. Say this.

**Verdict: HYBRID.** Correct the text, give the theoretical argument (invariances, multi-class sensitivity), and add the similarity-function ablation.

---

## Major issue 5 — Table 1 counterexamples; avoid "consistently improves every metric or architecture"

> "For IBS/ResNet-18, the ROAD scores of Grad-CAM and Grad-CAM++ decrease from 0.461 to 0.442 after enhancement. Several Insertion scores also decrease."

**What the manuscript says.** Section 4.2.1: "resulted in consistent enhancements across all three evaluation metrics"; Section 4.2.2: "consistently outperformed LayerCAM and HR-CAM across both datasets"; Abstract: "consistently outperforms existing single-layer CAM methods and multi-layer aggregation techniques"; Conclusion: "consistent improvements in Deletion, Insertion, and ROAD scores".

**Full audit of Table 1 (base → enhanced, 90 comparisons = 5 archs × 2 datasets × 3 base CAMs × 3 metrics):**

| Metric | Wins | Losses | Loss cases (base → enhanced) |
|---|---|---|---|
| ROAD | 28 | 2 | IBS/ResNet-18 Grad-CAM 0.461→0.442; Grad-CAM++ 0.461→0.442 |
| Insertion | 22 | 8 | IBS/VGG-16 GC++ 0.900→0.898; IBS/VGG-19 GC 0.912→0.910, GC++ 0.906→0.904; Kvasir/ResNet-34 GC 0.825→0.824, HiRes 0.827→0.825; Kvasir/ResNet-50 GC 0.832→0.829, GC++ 0.825→0.818, HiRes 0.833→0.830 |
| Deletion | 29 | 1 | IBS/ResNet-18 GC++ 0.192→0.203 |
| **Total** | **79** | **11** | |

All Insertion losses are ≤ 0.007 and almost certainly within noise (which is why CIs are needed). The ROAD losses on IBS/ResNet-18 are −0.019.

**Audit against the multi-layer baselines (enhanced variant vs. best of LayerCAM/HR-CAM, ROAD):**

| Enhanced variant | Wins / 10 settings | Losses |
|---|---|---|
| Enhanced HiResCAM | 9 | IBS/ResNet-34 (0.081 vs LayerCAM 0.162) |
| Enhanced Grad-CAM | 6 (+1 tie) | IBS/ResNet-34 (0.123 vs 0.162); Kvasir/VGG-16 (0.207 vs 0.228); Kvasir/VGG-19 (0.206 vs 0.228) |
| Enhanced Grad-CAM++ | 4 | IBS/VGG-16, IBS/VGG-19, IBS/ResNet-34, Kvasir/VGG-16, Kvasir/VGG-19, Kvasir/ResNet-34 |

Further cells where a multi-layer baseline beats the enhancer: HR-CAM Insertion is the best in Kvasir/ResNet-18 (0.781), ResNet-34 (0.841) and ResNet-50 (0.840); LayerCAM/HR-CAM Deletion beats enhanced Grad-CAM and Grad-CAM++ on IBS/ResNet-18 and enhanced HiResCAM on IBS/ResNet-50.

**So the reviewer is right and, if anything, understated it.** The defensible statement is: "Applied to HiResCAM, the enhancer improved ROAD in 10/10 configurations and matched or exceeded both multi-layer baselines in 9/10; applied to Grad-CAM and Grad-CAM++, improvements were frequent but not universal, and on IBS/ResNet-34 LayerCAM was the best method overall."

**Additional observation (S11):** in the IBS/ResNet-18 block, base Grad-CAM and base HiResCAM have identical Insertion (0.973) and Deletion (0.214) but different ROAD (0.461 vs 0.442), and enhanced Grad-CAM/Grad-CAM++ both equal 0.442 — the same as base HiResCAM. Identical Ins/Del with different ROAD is possible (ROAD uses stochastic noisy imputation) but unusual; re-verify these rows from the raw logs before responding, because the reviewer's headline counterexample may be a transcription error. If it is, say so transparently.

**Verdict: WORK.** Rewrite every "consistently/all" claim; add a win/tie/loss table with significance; discuss IBS/ResNet-18 and IBS/ResNet-34 explicitly (the former is also an outlier in absolute ROAD, 0.44–0.48 vs ≤0.30 elsewhere on IBS, which needs an explanation grounded in the classifier's behaviour).

---

## Major issue 6 — Novelty vs. Group-CAM, Opti-CAM; explain precisely how XAI-Enhancer differs

> "The novelty of the proposed aggregation strategy should be established through direct comparison with relevant multi-layer and optimization-based explanation methods, including Group-CAM and Opti-CAM where applicable."

**What the manuscript says.** Related work covers LayerCAM, HR-CAM, DE-SSG-CAM, and cites Ablation-CAM [27] in passing. Score-CAM, Group-CAM and Opti-CAM are not cited. The paper's stated novelty (R2's summary) is "deriving the weights through a training-free approach using counterfactual [masked] inputs".

**Assessment.**
- The reviewer is right that the closest prior work is missing. Score-CAM (Wang et al., 2020), Group-CAM (Zhang et al., 2021) and Opti-CAM (Zhang et al., CVIU 2024) all (i) upsample and normalise an activation-derived map, (ii) multiply it with the input, (iii) run a forward pass, and (iv) use the resulting class score to weight the map. That is precisely Phase 2 of XAI-Enhancer, applied to *channels* (Score-CAM), *channel groups* (Group-CAM), or *optimised channel weights* (Opti-CAM) within a single layer, rather than to *layers*. Ablation-CAM is the deletion-based cousin.
- The genuine difference (thread **D7**): XAI-Enhancer operates on the *layer* axis and is orthogonal to (and composable with) channel-level methods; it uses the full logit vector rather than the target-class score; it needs L ≤ 53 forward passes rather than C = 512–2048 (Score-CAM) or iterative optimisation (Opti-CAM). This is a real but incremental distinction and must be stated as such, with citations, in Related Work and in the contributions list.
- Direct comparison is feasible: Score-CAM, Ablation-CAM, EigenCAM and LayerCAM are already in the `pytorch-grad-cam` code base the authors forked; Group-CAM and Opti-CAM have public PyTorch implementations. Add at minimum Score-CAM and Group-CAM as baselines in Table 1 (and Table 3), and Opti-CAM if compute permits (it is slow). Also show the enhancer *wrapping* Score-CAM to demonstrate orthogonality.

**Verdict: HYBRID.** Add the citations and a precise differentiation paragraph; add Score-CAM/Group-CAM (and ideally Opti-CAM) to the baselines; report their latency in Table 5, which incidentally helps the latency defence.

---

## Major issue 7 — Latency (20×) and lower perturbation stability belong in the Abstract, Discussion and Conclusion

**What the manuscript says.** Both limitations are in Sections 5.3–5.4 but absent from the Abstract and Conclusion; the Abstract's last sentence is "produces highly faithful explanations, thereby advancing the trustworthiness of AI-assisted medical diagnostics."

**Assessment.** Correct. The framing defence (D1, D3) is about *interpretation*, not about *where* the limitation is reported. **WORK (text).** Add one sentence to the Abstract ("At the cost of an L-fold increase in inference passes (~0.35 s per image, unsuited to frame-rate video) and reduced heat-map stability under input noise") and a limitations paragraph to the Conclusion.

---

## Q8 — Promotional and causal wording

Listed terms: "fails fundamentally", "catastrophic", "profound", "decisively proves", "universal applicability", "highly faithful". All appear in the manuscript (Introduction; Sections 4.1, 4.3, 5.1; Conclusion; title). "Highly faithful" is in the **title**. Recommend changing the title to "... for faithful model explanations ..." or "... improving the faithfulness of model explanations ...". Full list in `08-manuscript-wording-edits.md`.

**Verdict: WORK.**

---

## Q9 — Language editing

Marked "Yes" by R1 and R3. The manuscript already declares Gemini was used for language. A human or professional edit pass focused on removing adjectival stacking ("rigorous", "comprehensive", "remarkable", "brilliantly", "seamlessly", "dramatically") will address both reviewers.
