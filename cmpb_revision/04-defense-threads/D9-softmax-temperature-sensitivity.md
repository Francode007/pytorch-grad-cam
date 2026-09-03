# Defense Thread D9 — Softmax temperature: turn "unjustified T=1" into "robust to T"

**Raised by:** R2.5 (T=1 not justified; monotonic columns; inflection-point language is subjective; present the 0.005 range as an advantage; state backbone/base CAM).
**Defensibility:** The reviewer is offering the defence. Accept it — but only after confronting what the monotonicity means for the novelty claim (S1). A defence that ignores S1 will not survive a second look.

---

## 1. What the data actually show (Table 4)

| T | Ins ↑ | Del ↓ | ROAD ↑ | Weight SD |
|---|---|---|---|---|
| 0.1 | 0.7731 | 0.4346 | 0.1591 | 0.0042 |
| 0.5 | 0.7787 | 0.4318 | 0.1620 | 0.0021 |
| 1.0 | 0.7797 | 0.4315 | 0.1628 | 0.0014 |
| 2.0 | 0.7802 | 0.4312 | 0.1632 | 0.0009 |
| 5.0 | 0.7808 | 0.4308 | 0.1636 | 0.0004 |
| 10.0 | 0.7810 | 0.4307 | 0.1638 | 0.0002 |

Every column is strictly monotonic. Faithfulness improves as the weights approach uniform. There is no inflection point. The manuscript's "T = 1.0 represents a critical inflection point of diminishing returns" is not a description of these numbers.

Two readings, both of which the paper must address:

- **R2's reading (the defence):** ROAD varies by 0.0047 (3%) across two orders of magnitude of T. The method is insensitive to its only hyperparameter. T=1 is a reasonable default. This is a genuine practical advantage over methods with tuned hyperparameters (Group-CAM's group count and threshold, Opti-CAM's iterations/learning rate, DE-SSG-CAM's evolutionary search).
- **The uncomfortable reading (S1):** the best configuration is the one where the weighting does nothing (T→∞ ⇒ w_l = 1/L). Therefore, on this dataset/backbone, the logit-similarity weighting — presented as the key novelty — contributes nothing measurable to faithfulness; all of the gain in Table 1 comes from averaging CAMs across layers. If a re-reviewer says this, the paper needs an answer prepared in advance.

## 2. Why the weights are near-uniform at T=1 (mechanics)

S_l ∈ [−1, 1], so exp(S_l / 1) ∈ [0.37, 2.72] and the max/min weight ratio is at most e² ≈ 7.4. In practice most S_l lie in ~[0.6, 1.0] (Figure 3), giving weight ratios ≈ 1.5 at most. Table 2 confirms: post-softmax SD is 0.0013–0.0017 around a mean of 1/L (0.019–0.077). On VGG-16 the *raw* similarity SD is already only 0.017 — the similarity measure barely distinguishes layers before softmax. Two causes: (a) cosine of raw logits is inflated by the shared mean component (fix: centering, see D6); (b) the "temperature" is applied to a quantity with dynamic range ~0.3, so T=1 is effectively a *high* temperature. Standardising S_l across layers per image (z-score) before softmax would give T a meaningful scale.

## 3. What to do

1. **Rename Section 4.6** "Sensitivity to the softmax temperature" and rewrite: T=1 is a default; faithfulness is insensitive to T; a small residual gain from flatter weighting is observed and discussed.
2. **Add a "Uniform average (T→∞)" row to Table 1** for every configuration. This is the single most important new baseline in the revision, because it isolates the contribution of the weighting from the contribution of multi-layer averaging.
3. **Look for the regime where weighting helps.** Candidates: ResNet-18/34 (Table 2: raw SD 0.36–0.47, so weights *do* vary); images where at least one layer's CAM is degenerate (the 1×1/projection layers of R2.4); the IBS dataset. Report per-architecture T-sensitivity, not just one backbone. If weighting helps on ResNets with volatile layers and is neutral on VGG, that is a coherent, honest story: "the weighting acts as a safeguard against degenerate layers; where all layers are informative it reduces to averaging".
4. **Try standardised similarities** (z-score across layers before softmax) and centered cosine (D6) so that T operates on a quantity with a meaningful scale; report whether a low-T regime then beats uniform.
5. **Select T on the validation split, evaluate on test** (S3). Table 4 is currently computed on the same N=1,600 images used for reporting and for early stopping.
6. **State backbone and base CAM in the caption** (R2.2).
7. **Remove "statistically marginal"** unless a test is reported (D5).

## 4. Draft rebuttal text

> We agree with the reviewer's reading of Table 4: all metrics vary monotonically with T and the total variation in ROAD across T ∈ [0.1, 10] is below 0.005. We have therefore reframed Section 4.6 as a sensitivity analysis rather than an optimisation, removed the "inflection point" language, and now describe T = 1 as a default to which the method is insensitive, which we consider a practical advantage relative to methods with tuned hyperparameters. We also make explicit the implication the reviewer's observation raises: as T → ∞ the weights approach a uniform average, so the small residual gain at large T indicates that, for this backbone and dataset, most of the improvement over single-layer CAMs stems from multi-layer aggregation itself. To quantify this we have added a uniform-average baseline to Table 1 for every configuration and report per-architecture sensitivity in the supplement. [Summarise: where weighting helps (e.g., ResNets with degenerate layers) and where it is neutral.] Table 4 is now computed on the held-out test split with T selected on the validation split, and the caption states the backbone (…) and base CAM (…).

## 5. Residual risk

Medium. The outcome of step 3 determines whether the abstract can keep "dynamically compensating for these bottlenecks". If uniform averaging matches the weighted version everywhere, the abstract and contributions must be rewritten around (i) multi-layer aggregation as a robust, training-free improvement and (ii) the per-layer similarity as an analysis tool. That is still a publishable paper in CMPB; an abstract that claims a mechanism the tables contradict is not.
