# Defense Thread D3 — Lower SSIM/Pearson under perturbation is not (only) a flaw

**Raised by:** R1 (implement TTA or gradient denoising); R3-7 (lower perturbation stability must be in Abstract/Conclusion).
**Defensibility:** Moderate–strong on interpretation; the presentation in Table 6 is too thin to carry the argument alone. Keep the trade-off framing, strengthen it, and add a small TTA experiment.

---

## 1. The argument

1. **Stability and faithfulness are in known tension.** A heat-map that is identical for every input has perfect SSIM under any perturbation and zero faithfulness. Yeh et al. (NeurIPS 2019, "On the (In)fidelity and Sensitivity of Explanations") formalise this: reducing sensitivity (e.g., by smoothing) trades against infidelity, and low-resolution explanations are intrinsically smoother. The coarse 7×7 / 14×14 maps of terminal-layer Grad-CAM are low-pass by construction and therefore *cannot* respond to high-frequency input noise; the enhancer's early-layer components can and do.
2. **If the model's output changes, a faithful explanation should change.** Gaussian sensor noise and contrast shifts change the classifier's logits. Explanation stability should be measured *conditional on the prediction being unchanged*, or alongside the change in the model's own output. Table 6 does neither; it treats any change in the map as a defect.
3. **This is a property of multi-layer methods, not of the enhancer.** LayerCAM and HR-CAM also inject early-layer detail. If they show the same SSIM drop (very likely), the trade-off is generic. Table 6 only compares against Grad-CAM, which makes the enhancer look uniquely fragile.
4. **The paper already discloses it.** Section 5.4 is a dedicated limitations subsection. R3's demand is about *placement* (Abstract/Conclusion), which should simply be granted.

## 2. Evidence in the manuscript

- Sec. 5.4: "a fundamental trade-off between spatial faithfulness and noise robustness. The terminal-layer compression that produces coarse heatmaps in standard CAMs inherently shields them from high-frequency pixel noise."
- Table 6: Grad-CAM SSIM 0.8128 / Pearson 0.8268; XAI-Enhancer 0.6867 / 0.7039.

## 3. Weaknesses to repair

- Table 6 does not state backbone, dataset, N, noise σ, contrast factor, or whether the perturbed prediction changed. Unreproducible as written (R2's reproducibility complaint applies).
- Only one baseline.
- No faithfulness metric on the perturbed images: the reader cannot see whether the enhancer's map *changed to track the model* or just became noisy.

## 4. What to add

1. **Protocol.** Specify σ values (e.g., σ ∈ {0.01, 0.03, 0.05} in normalised units), contrast factors, N images, backbone, base CAM.
2. **Prediction-conditioned stability.** Report SSIM/Pearson separately for images whose predicted class is unchanged and report the prediction-flip rate and mean |Δ confidence| under each perturbation. Also report ROAD of each method *on the perturbed image* — if the enhancer's map changes but stays faithful to the perturbed prediction, that is the correct behaviour.
3. **All baselines,** especially LayerCAM and HR-CAM.
4. **TTA variant (what R1 asked for).** Average the enhancer output over n ∈ {4, 8} noisy copies of the input (SmoothGrad-style; because the final map is linear in the per-layer maps, averaging at the map level is exact). Report SSIM/Pearson *and* ROAD/Ins/Del for the TTA variant, plus its latency (n × 345 ms sequential, less when batched). Expect SSIM to rise and ROAD to stay about the same or drop slightly; report whatever happens.
5. **Placement.** Sentence in Abstract; paragraph in Conclusion.

## 5. Draft rebuttal text

> We agree that reduced heat-map stability under input perturbation is an important limitation and now state it in the Abstract and Conclusion in addition to Section 5.4. We would add context: explanation stability and explanation faithfulness are known to be in tension (Yeh et al., 2019); a low-resolution terminal-layer map is smooth by construction and therefore insensitive to pixel-level noise, whereas an explanation that incorporates high-resolution layers necessarily responds to it. To separate "tracking the model" from "adding noise", we now (i) report the prediction-flip rate and confidence change of the classifier under each perturbation, (ii) report stability conditioned on an unchanged prediction, (iii) report the faithfulness (ROAD) of each method on the perturbed inputs, and (iv) include LayerCAM and HR-CAM, which exhibit [similar/different] behaviour. Following Reviewer 1's suggestion we also evaluate a test-time-augmentation variant that averages the enhanced map over n noisy copies; it raises SSIM from A to B at a ROAD change of C and an n-fold latency increase (new Table 6). The full perturbation protocol is now specified.

## 6. Residual risk

Low if the prediction-conditioned analysis shows the enhancer is stable when the model is stable. If it shows the enhancer is unstable even when the prediction is unchanged, then the TTA variant becomes the recommended default under noisy acquisition, and the paper should say so.
