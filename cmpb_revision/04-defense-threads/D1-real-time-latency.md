# Defense Thread D1 — Latency and the "real-time endoscopy" objection

**Raised by:** R1 (must implement sparse layer sampling to reach 33 ms); R3-7 (20× slower; "unsuitable for real-time endoscopy in its current form"; move to Abstract/Conclusion).
**Defensibility:** Strong on framing, weak on presentation. Defend the scope; concede the placement; add two cheap experiments that make the 33 ms argument irrelevant.

---

## 1. The argument

1. **The paper never claims real-time operation.** Section 5.3 already states the method "exceeds the latency threshold for real-time video processing" and positions it for "near-real-time CDSS analysing static endoscopic captures or targeted keyframes". The reviewers are holding the paper to a requirement it did not set.
2. **Post-hoc explanation is, in clinical practice, a review-time activity.** Endoscopists capture still frames of findings during the procedure; explanations are consulted when a finding is examined, documented, or audited — not at 30 fps overlaid on live video. 0.35 s per captured frame is imperceptible in that workflow. (Frame it this way, without over-claiming clinical validation the paper does not have.)
3. **345 ms is at the cheap end of the masking-based family.** Any method that estimates weights by re-querying the model pays this cost: Score-CAM needs one masked pass per channel (512 for VGG conv5, 2048 for ResNet-50 layer4), RISE needs thousands of random masks, Opti-CAM runs ~100 iterations of forward+backward, LIME/SHAP are slower still. XAI-Enhancer needs L ≤ 53 passes. If R3's Major issue 6 forces Score-CAM/Group-CAM/Opti-CAM into the baseline set, their latencies should go into Table 5 — they will make 345 ms look modest.
4. **The cost is not architectural; it is the current implementation.** Section 3.1.3 says the L masked passes are independent and batchable. Table 5's 345 ms (≈ 6.5–26 ms per pass depending on L) is consistent with *sequential* single-image passes. Batching the L masked images into one tensor typically gives several-fold speed-ups on an A100 for these backbones. This was not benchmarked, and it should be.

## 2. Evidence in the manuscript

- Sec. 3.1.3: "O(L) ... the independent nature of the masked passes allows for parallelized batch execution on modern GPUs."
- Sec. 5.3: "exceeds the latency threshold for real-time video processing, it remains ... viable ... for near-real-time CDSS analysing static endoscopic captures or targeted keyframes. Future optimizations will focus on sparse layer sampling."
- Table 5: 344.91 ms vs 16.91–17.03 ms.

## 3. Weaknesses the reviewers can still exploit

- Table 5 gives one latency figure for a method whose cost is O(L) with L ∈ {13, 16, 20, 36, 53}; the backbone is not stated. A reader cannot tell whether 345 ms is VGG-16 or ResNet-50.
- The batching claim is unsubstantiated.
- "Sparse layer sampling" is named as future work, which invites "then do it".

## 4. What to add (small, high-yield)

1. **Batched implementation benchmark.** Same backbone(s), same GPU, warm-up excluded, median of ≥100 images. Report sequential vs. batched latency and peak VRAM for each backbone.
2. **XAI-Enhancer-S (sparse) variant.** Restrict L to stage/block outputs: ResNet `layer1..layer4` outputs (4 layers, or 8 by adding the last block of each stage), VGG the last conv of each of the 5 stages. Report ROAD/Ins/Del and latency. This also (a) removes the 1×1/projection convs that likely cause the Figure 3b artifact (R2.4), (b) is a natural ablation of "how many layers do you actually need", and (c) is what R1 literally asked for.
3. **Trade-off figure.** Latency (x) vs ROAD (y) for base CAM, Enhancer-S with k layers, full Enhancer, Score-CAM, Group-CAM, Opti-CAM. One figure answers R1.1, R3-6 and R3-7 together.
4. **Placement.** One sentence in the Abstract, one paragraph in the Conclusion (as R3 demands). Do not fight this.

## 5. Draft rebuttal text

> We thank the reviewers for raising deployment latency. We agree that at ~0.35 s per image the method in its submitted form is not suited to frame-rate video analysis, and we now state this explicitly in the Abstract, Section 5.3 and the Conclusion. We would however clarify the intended use: XAI-Enhancer is a post-hoc explanation method for captured frames and keyframes reviewed in a decision-support workflow, where sub-second latency is acceptable; it was not designed for live-video overlay. Following the reviewers' suggestion we have (i) implemented batched masked inference, which reduces latency from X ms to Y ms on [backbone]; (ii) added a sparse variant (XAI-Enhancer-S) that aggregates only the L=k stage outputs, achieving Z ms with a ROAD change of W (new Table 5 and Figure N); and (iii) reported per-backbone latency and the latency of the masking-based baselines Score-CAM, Group-CAM and Opti-CAM, which require respectively C, G and ~100 model queries per image, for context. We note that any weighting scheme that queries the model on perturbed inputs incurs a multiple of single-pass cost; XAI-Enhancer's L ≤ 53 queries place it at the efficient end of that family.

## 6. Residual risk

Low once the above is added. If the sparse variant loses noticeable ROAD, report it honestly as the price of speed; the trade-off curve is still a contribution.
