# Defense Thread D5 — "No repeated runs" partially misreads where the variance lives

**Raised by:** R3-3 (repeat across seeds or patient-level folds; report means, SD or 95% CI; significance tests between explanation methods).
**Defensibility:** Partial. The *explanation method* is deterministic; the variance R3 wants comes from two distinct sources, only one of which requires retraining. Concede the request, but structure the response so that the cheap half is done exhaustively and the expensive half is done as part of the mandatory re-split.

---

## 1. The argument

1. **XAI-Enhancer has no stochastic component.** Given a trained network and an image, Phases 1–3 are deterministic (Grad-CAM is deterministic; masking, forward pass, cosine, softmax are deterministic). Repeating the *explanation* with different seeds returns the same numbers. So "repeat across multiple random seeds" cannot mean re-running the explainer; it can only mean (a) retraining the classifier or (b) treating the evaluation images as a sample.
2. **Sampling variance over images can be quantified immediately, with no retraining.** Every metric in Table 1 is a mean over N per-image scores (N = 1,600 for Kvasir-v2; ~1,100 for IBS). Bootstrap CIs over images and *paired* tests (Wilcoxon signed-rank on per-image differences between base and enhanced maps, or between enhanced and LayerCAM/HR-CAM) are the standard way to compare explanation methods on a fixed model, and they are more powerful than unpaired comparisons across seeds because the model and image are held fixed.
3. **Training variance across seeds/folds is a property of the classifier, not the explainer,** but it does matter for the *generality* of the conclusion (does the enhancer help on any trained model of this architecture, or only on this one?). Since R3-1 forces patient-level folds for IBS anyway and a new split for Kvasir-v2, the retraining will happen; the explainer must then be evaluated on each fold/seed and the results aggregated.
4. **ROAD itself has a stochastic component** (noisy linear imputation). Fix and report its seed, or average over a few imputation seeds; this is cheap.

## 2. Evidence in the manuscript

- Sec. 3.1 – 3.1.3: no random element in the method.
- Sec. 4.6: "the performance delta is statistically marginal (Δ ≤ 0.001)" — asserted without a test, which is exactly what invites R3's comment. Remove the word "statistically" unless a test is reported.

## 3. What to add

| Source of variance | How to quantify | Cost |
|---|---|---|
| Evaluation image sampling | Bootstrap 95% CI (e.g., 2,000 resamples) per cell of Tables 1, 3, 4, 6; paired Wilcoxon signed-rank base vs enhanced, enhanced vs each multi-layer baseline; Holm correction within each table; effect size (Cliff's δ or paired Cohen's d) | Re-run metrics with per-image logging; no retraining |
| Classifier training | ≥3 seeds per architecture on Kvasir-v2 (train/val/test); 5 patient-level folds on IBS; report mean ± SD of each metric across seeds/folds; report how often the enhanced method wins per seed/fold | Retraining (mandatory anyway under R3-1/R3-2) |
| ROAD imputation noise | Fix seed; or 3 seeds averaged | Trivial |

Present Table 1 as mean ± SD across seeds/folds with CI in a supplementary long-form table; add a compact **win/tie/loss** table (tie = not significant) — this simultaneously resolves R3-5.

## 4. Draft rebuttal text

> We agree that point estimates without uncertainty are insufficient. We note that XAI-Enhancer itself is deterministic given a trained model; the relevant sources of variability are (i) the sample of evaluation images and (ii) classifier training. For (i), all metrics are now reported with bootstrap 95% confidence intervals over images, and comparisons between explanation methods on the same model use paired Wilcoxon signed-rank tests with Holm correction and Cliff's δ as effect size; cells where the difference is not significant are labelled as ties in the new win/tie/loss summary (Table N). For (ii), all Kvasir-v2 models were retrained with three random seeds on a stratified train/validation/test split, and all IBS models were retrained under five-fold patient-level cross-validation (see response to Major issue 1); Table 1 now reports means ± SD across seeds/folds. The ROAD imputation seed is fixed and stated.

## 5. Residual risk

Low. The one thing that could go wrong is that some previously reported "wins" (especially the ≤0.007 Insertion differences and the 0.001–0.005 T sensitivity) become ties. That is a *good* outcome for credibility as long as the claims are rewritten accordingly (R3-5, R2.5).
