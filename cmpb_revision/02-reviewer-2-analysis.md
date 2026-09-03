# Reviewer 2: Point-by-Point Analysis

Reviewer 2's profile: the most favourable reviewer. Calls the method "novel", the CAM+masking synergy "very well-motivated", and the comparison "comprehensive". Marked reproducibility as **No** because of missing settings. The four weaknesses are precise, low-cost, and largely correct. Each should be accepted, and in one case (temperature) the reviewer is handing the authors a better framing than the one in the paper.

---

## R2.1 — "Low-resolution terminal bottleneck" wording is CNN-specific and conflicts with the transformer claim

> "The authors state that the same framework is applicable to transformer-based backbones. Attention maps across multiple transformer layers can indeed be aggregated as proposed. However, the Introduction section may contain wording stating that the work addresses a 'low-resolution terminal bottleneck,' which is not an inherent concern for transformer-based backbones."

**What the manuscript says.** Introduction, paragraph 3: "forcing explanations through a low-resolution terminal bottleneck causes severe spatial diffusion". The whole motivation (Introduction and Section 2.2) is built on pooling-induced resolution loss. Section 5.5 then claims the principle is architecture-agnostic.

**Critical assessment.**
- Correct. ViTs keep a constant token grid (e.g., 14×14 for ViT-B/16 at 224 px) across all blocks, so there is no progressive spatial compression. The paper's motivating story does not transfer.
- More importantly, this exposes a deeper issue with the paper's *own CNN story* (see `05-self-identified-issues.md`, S4): in VGG-16 the last three conv layers (conv5_1, conv5_2, conv5_3) all operate at 14×14, and in ResNet-50 the last five conv modules all operate at 7×7. The per-layer differences the paper reports in Figure 2 among the "terminal five" layers therefore cannot be caused by resolution loss. The honest, general motivation is that *different depths encode different levels of abstraction and class-specificity*, and that no single depth is a priori the most faithful; resolution is one factor among several. That motivation applies to transformers too.

**Verdict: WORK (text).** Reframe the motivation around "depth-dependent abstraction and unreliable single-layer selection" with resolution loss as one contributing mechanism in CNNs. Remove "low-resolution terminal bottleneck" as the sole cause. This also pre-empts R1.4 and R3-Q1.2.

---

## R2.2 — Reproducibility: ablation hyperparameters missing; which CAM (Grad-CAM / Grad-CAM++ / HiResCAM) and which backbone underlie Table 4

> "Some hyperparameters and settings in the ablation studies are missing. It is not clear whether they come from GradCAM, GradCAM++ or HiResCAM." and "it is unclear which backbone and base XAI method Table 4 is derived from."

**What the manuscript says.** Section 4.6 / Table 4: "across the Kvasir-v2 validation cohort (N = 1600)". No backbone, no base CAM. The same omission applies to **Table 3** (Kvasir-SEG alignment: backbone unspecified), **Table 5** (latency: backbone unspecified, although L differs 4× between backbones), **Table 6** (robustness: backbone, dataset, noise parameters unspecified), and **Figure 4/5** (backbone unspecified). Also missing across the paper: hook location (module output vs. post-ReLU), the exact layer set L, LayerCAM baseline layer selection and fusion rule, Deletion/Insertion step size and blur parameters, ROAD imputation parameters, binarisation threshold for Table 3, number of images per evaluation.

**Critical assessment.** Entirely correct and cheap to fix. This is the reason R2 marked reproducibility "No"; flipping it is a matter of a dedicated "Implementation details for evaluation" subsection or a supplementary table.

**Verdict: WORK.** Add an evaluation-settings table listing, for every table/figure: dataset and split, N images, backbone, base CAM, layer set, T, perturbation parameters, random seed. State the base CAM in the caption of every table.

---

## R2.3 — Table 3 values are low; binarised maps deflate Dice/IoU; add other baselines

> "The values in Table 3 are quite low. Using binarized heatmaps may have deflated these values. Consider computing Dice and IoU using continuous-valued heatmaps instead. The table could also include Dice and IoU results for the other baselines."

**What the manuscript says.** Section 4.5 / Table 3: Grad-CAM IoU 0.1495 / Dice 0.2350 vs XAI-Enhancer 0.1668 / 0.2702, on "binarized classification heatmaps" against Kvasir-SEG masks. Threshold not given. Only one baseline. Backbone not given.

**Critical assessment.**
- Correct on all three counts. IoU ≈ 0.15–0.17 is low even for weakly-supervised localisation; with an unspecified threshold the comparison is fragile (a different threshold can reverse the ranking).
- Better metrics for continuous saliency vs. binary ground truth: energy-based pointing game (Wang et al., Score-CAM, CVPRW 2020: fraction of saliency mass inside the mask), pointing game hit-rate (max-saliency inside mask), soft Dice, and threshold-sweep AUC (Dice as a function of threshold, report the area or the max). Report at least two.
- Section 4.5 has a bigger problem the reviewer did not notice (S2 in `05-self-identified-issues.md`): Kvasir-SEG's 1,000 images are the Kvasir-v2 polyp class. With an 80:20 image-level split, ~800 of those 1,000 images were in the classifier's training set. The alignment evaluation must be restricted to Kvasir-SEG images that fall in the held-out test split, and the text must say so. Fixing this simultaneously with R2.3 is efficient.
- Add HiResCAM, Grad-CAM++, LayerCAM, HR-CAM, and (if included per R3-6) Score-CAM/Group-CAM/Opti-CAM to the table. Also add the uniform-average variant (see S1).

**Verdict: WORK.** Recompute Table 3 with continuous metrics, all baselines, specified backbone/threshold, and test-split-only images.

---

## R2.4 — Figure 3b dip looks like an artifact (large SD); "VGG more stable" not justified

> "The dip in Figure 3b looks more like an artifact, as evidenced by the large standard deviation. Therefore, the conclusion that VGG is more stable may not be fully justified. Further investigation is encouraged."

**What the manuscript says.** Figure 3b: raw cosine similarity vs. normalised depth for VGG-16 and ResNet-50 on Kvasir-v2; ResNet-50 shows a dip at normalised depth ≈ 0.1 with a wide shaded band. Section 4.3: "ResNet-50 exhibits an early structural bottleneck at shallow layers (normalized depth ≈ 0.1) before recovering via skip connections"; VGG-16 "maintains remarkably stable similarity".

**Critical assessment.**
- Likely correct. At normalised depth ≈ 0.1 in a 53-layer enumeration (index ≈ 5), the module is inside `layer1`, which in torchvision's ResNet-50 contains 1×1 convs and a 1×1 shortcut projection (`layer1.0.downsample.0`). A CAM computed on a 1×1 bottleneck or projection conv at 56×56 is not a meaningful spatial explanation; masking with it yields near-random masked logits, hence a low mean and a large SD across images. That is an artifact of including every conv module in L, not a "structural bottleneck".
- The paper's own Table 2 supports the artifact reading: ResNet-50's average raw SD is 0.0963 — the *lowest* of the ResNets — so the ResNet-50 curve is dominated by one or two anomalous layers rather than broad volatility.
- "Stability" of VGG here means the cosine similarity barely changes across layers (Table 2: raw SD 0.017 for VGG-16). That is equally consistent with the similarity measure being *uninformative* on VGG (all layers look alike to it) as with VGG "preserving spatial information". The paper cannot distinguish these without relating S_l to an independent faithfulness measure per layer — and Figure 2a (per-layer ROAD) shows VGG is *not* stable across its last five layers. See S5.

**Verdict: HYBRID.** Investigate: identify the exact module at the dip; plot per-layer boxplots (not just mean ± SD); rerun Figure 3 with L restricted to 3×3 convs / block outputs to show whether the dip disappears. Rewrite the VGG-vs-ResNet narrative to describe the data ("a small number of low-similarity layers, concentrated in 1×1 projection convolutions") rather than to assert architectural "stability". Thread **D10** covers what can still be defended (the existence of layer-wise heterogeneity that a fixed layer choice ignores).

---

## R2.5 — T = 1 not justified; Table 4 is monotonic; reframe as sensitivity; 0.005 ROAD range is an advantage

> "The choice of Softmax temperature T=1 is not justified. Table 4 shows values monotonically increasing or decreasing, depending on the column. Using the motivation of an inflection point and diminishing returns is subjective. The authors could frame this not as an ablation but as a demonstration of parameter sensitivity. The fact that the ROAD score range is within 0.005 could instead be presented as an advantage."

**What the manuscript says.** Section 4.6 / Table 4 / Figure 6: ROAD 0.1591 (T=0.1) → 0.1638 (T=10), strictly increasing; Insertion strictly increasing; Deletion strictly decreasing; weight SD strictly decreasing. Text: T=1 is a "critical inflection point of diminishing returns" and "optimal operational threshold"; T=10 "collapses the framework into a static naive average (1/L) that loses dynamic, image-specific adaptability".

**Critical assessment.**
- The reviewer is right that there is no inflection point in the data: every column is monotonic, and T=10 is the best on all three faithfulness metrics. Calling T=1 "optimal" is not supported.
- The reviewer's suggested reframing (parameter insensitivity is a strength) is generous and should be accepted. But the authors must also confront what the monotonicity implies (S1): the *uniform average* of all layer CAMs is at least as faithful as the logit-similarity-weighted average. The paper's claimed novelty is the weighting, so the paper must either (a) show a regime where the weighting demonstrably helps (e.g., on specific architectures like ResNet-18 where Table 2 shows the weights actually vary; on images where a layer's CAM is degenerate; or after centering/standardising S_l so that the softmax has something to discriminate), or (b) reposition the contribution: the similarity scores are a *diagnostic* of layer-wise information flow (Section 4.3), and the aggregation is deliberately near-uniform because uniform multi-layer averaging is robustly good. Option (b) is honest and still publishable, but it changes the abstract.
- Why the weights are near-uniform: S_l ∈ [−1, 1], so with T=1 the ratio between the largest and smallest weight is at most e² ≈ 7.4, and in practice most S_l cluster near 0.7–1.0 (Figure 3), so weights are within a few percent of 1/L (Table 2: SD 0.0013–0.0017 around a mean of 1/53–1/13). Cosine similarity between raw logit vectors is inflated by a shared mean component, which compresses the dynamic range further (see thread **D6** for the fix: center logits before cosine, or standardise S_l across layers before softmax).
- Also: T was tuned on the same N=1600 validation cohort used for final evaluation and for early stopping (S3). With a proper train/val/test split, T should be chosen on val and Table 4 rerun on test.

**Verdict: WORK (accept R2's reframing) + confront S1.** Rename Section 4.6 "Sensitivity to the Softmax temperature". State that faithfulness varies by <0.005 ROAD across two orders of magnitude of T and that T=1 is a default, not an optimum. Add a uniform-average row to Table 1. Add the similarity-function/standardisation ablation (D6). Rewrite the abstract's "dynamically compensating for these bottlenecks" if the data do not support it.

---

## R2 strengths (use them)

R2 lists three strengths: comprehensive baseline comparison, well-motivated CAM+masking synergy, and fine-grained maps in Figure 4. These are the sentences to lean on in the response letter's opening and in the revised abstract. Do not over-claim beyond them.
