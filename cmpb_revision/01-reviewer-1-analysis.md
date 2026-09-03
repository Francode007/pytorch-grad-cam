# Reviewer 1: Point-by-Point Analysis

Reviewer 1's profile: answered "Yes" to objectives, reproducibility, statistics, and conclusions; flagged "Yes" to structure/flow and language editing without specifics. All substantive comments are in Q1 and concern **deployment** (latency, VRAM), **robustness**, and **ViT generalisation**. This reviewer is reading the paper as an engineer asking "can I ship this in an endoscopy suite", which is not the paper's stated scope. Several of these points are therefore defensible on scope, but each one also has a cheap experiment that removes the objection entirely, and the tone of the comments ("the authors need to", "must") suggests the reviewer expects action, not only argument.

---

## R1.1 — Latency: 344.91 ms vs the 33 ms real-time threshold; "implement sparse layer sampling"

> "The requirement of independent masked forward passes for each convolutional layer scales inference latency linearly. The XAI-Enhancer requires 344.91 milliseconds per image, which exceeds the 33-millisecond threshold required for 30 frames per second real-time video endoscopy. The authors need to implement sparse layer sampling to reduce this latency and meet real-time processing constraints."

**What the manuscript says.** Section 3.1.3 states complexity is O(L) and that "the independent nature of the masked passes allows for parallelized batch execution on modern GPUs". Section 5.3 and Table 5 report 344.91 ms vs ~17 ms for baselines, concede the method "exceeds the latency threshold for real-time video processing", and position it for "near-real-time CDSS analysing static endoscopic captures or targeted keyframes". Sparse layer sampling is already named as future work.

**Critical assessment.**
- The reviewer is factually correct about the numbers and about real-time endoscopy being out of reach.
- However, the reviewer's premise (that real-time video is the target) is *not* what the paper claims. The paper never claims real-time; it explicitly scopes to static captures. So the argument is defensible on scope (see thread **D1**).
- The paper's own claim that masked passes "allow parallelized batch execution" is undermined by Table 5: 345 ms for L=13–53 layers is ~6.5–26 ms per masked pass, which is what *sequential* single-image forward passes cost on an A100. The batched implementation the paper says is possible was evidently not benchmarked. This is a self-inflicted weakness: benchmarking the batched version (all L masked images in one tensor) would likely cut latency several-fold at zero methodological cost.
- Table 5 reports a single latency for "XAI-Enhancer" although L ranges from 13 (VGG-16) to 53 (ResNet-50); latency must differ ~4× across backbones. The table does not say which backbone was measured.
- Sparse layer sampling is a genuinely reasonable ask *and* it doubles as an ablation the paper needs for other reasons: the layer set L currently includes every conv module, including 1×1 bottleneck and shortcut-projection convolutions in ResNets (Table 2: 20/36/53 layers). Restricting to stage/block outputs (e.g., 4–5 layers for ResNet, 5 for VGG) is a natural "sparse" variant.

**Verdict: HYBRID.** Defend the scope (post-hoc explanation of stored frames/keyframes, not frame-by-frame video). Do the two cheap things that make the objection moot: (i) benchmark a batched implementation, (ii) add a sparse variant ("XAI-Enhancer-S": block outputs only) with a latency vs. ROAD trade-off plot. Report latency per backbone. Add the latency limitation to the Abstract and Conclusion as R3-7 also demands.

**Response direction.** "We agree the current implementation is not suitable for frame-rate video analysis and now state this in the Abstract, Section 5.3 and the Conclusion. We note the method is intended for post-hoc explanation of captured frames in a CDSS. Following the reviewer's suggestion we (a) implemented batched masked inference, reducing latency from X to Y ms, and (b) evaluated a sparse variant using only stage outputs (L = k), which achieves Z ms at a ROAD cost of W. Results are in new Table 5 / Figure N." Also contextualise: Score-CAM requires one masked pass *per channel* (512–2048 passes), RISE thousands, Opti-CAM ~100 iterative forward+backward passes; a per-layer scheme with L ≤ 53 passes is the cheap end of the masking-based family.

---

## R1.2 — Peak VRAM 2724.24 MB: deployment viability on clinical workstations

> "The methodology increases peak VRAM allocation to 2724.24 MB per image pass. The authors must detail the deployment viability of this memory requirement on standard clinical workstations."

**What the manuscript says.** Table 5: Base Grad-CAM 2527.30 MB, LayerCAM 389.02 MB, HR-CAM 389.02 MB, XAI-Enhancer 2724.24 MB.

**Critical assessment.**
- By the paper's own Table 5, the enhancer adds only ~197 MB (+7.8%) over base Grad-CAM. 2.7 GB fits on every discrete GPU sold in the last decade and on integrated GPUs with shared memory; it is not a deployment barrier. This is straightforwardly defensible (thread **D2**).
- The real problem in Table 5 is one the reviewer did not name but which makes the table look unreliable: Grad-CAM at 2527 MB vs LayerCAM/HR-CAM at 389 MB on presumably the same backbone is not physically plausible if measured the same way. Likely causes: Grad-CAM was measured first and includes CUDA context/cuDNN workspace warm-up; or Grad-CAM/Enhancer measurements include the autograd graph while LayerCAM/HR-CAM were measured without gradients; or different backbones. This must be re-measured consistently (same backbone, same warm-up, `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`), and the table must say which backbone.
- Batched masked inference (R1.1) will *increase* peak VRAM (L images in one batch). Report the trade-off honestly: sequential = low memory/high latency; batched = higher memory/low latency. Both are well within a 4–8 GB card.

**Verdict: DEFEND, but fix Table 5 first.** Add a sentence to Section 5.3 on deployability (fits consumer GPUs; CPU fallback latency if measured), and report memory for sequential and batched modes.

---

## R1.3 — Robustness: SSIM 0.6867 vs 0.8128; "implement TTA or gradient denoising"

> "The enhanced spatial localization causes higher sensitivity to clinical perturbations and domain shifts. The structural similarity index measure score under simulated Gaussian sensor noise drops to 0.6867 compared to 0.8128 for the baseline Grad-CAM. The authors should implement and evaluate test-time augmentation or gradient denoising to improve robustness."

**What the manuscript says.** Section 5.4 and Table 6 report the numbers, frame it as "a fundamental trade-off between spatial faithfulness and noise robustness", and name TTA/gradient denoising as future work.

**Critical assessment.**
- The reviewer is repeating back the paper's own limitation and asking that the named future work be done now. This is a common pattern and is best handled by doing a minimal version of it.
- The framing *is* defensible on theory (thread **D3**): a constant heat-map has perfect SSIM under any perturbation; explanation stability and explanation faithfulness are in known tension (Yeh et al., NeurIPS 2019, "On the (In)fidelity and Sensitivity of Explanations"). Moreover, if the classifier's prediction or confidence changes under noise, a *faithful* explanation should change too; Table 6 does not report whether predictions changed. The fair comparison is stability conditioned on unchanged prediction.
- Table 6 is under-specified: no backbone, no dataset, no noise σ, no contrast range, no N. R2 raised the same reproducibility issue for Table 4; it applies here too.
- Table 6 compares only against Grad-CAM. LayerCAM and HR-CAM also use early layers; if they show similar SSIM drops, the trade-off is a property of multi-layer methods in general, not a flaw of the enhancer. Add them.
- A SmoothGrad-style variant (average the enhancer output over n noisy copies of the input, n = 4–8) is a few lines of code. Because the enhancer output is a weighted sum of maps, TTA can be applied at the map level. Report SSIM/Pearson *and* ROAD for the TTA variant so that the robustness gain is not bought with a hidden faithfulness loss.

**Verdict: HYBRID.** Keep the trade-off framing and strengthen it with the stability-vs-faithfulness literature and a prediction-conditioned stability analysis. Add the TTA variant as a small experiment (n noisy copies), report both robustness and faithfulness, and specify the protocol. Move the limitation into the Abstract/Conclusion (R3-7).

---

## R1.4 — No ViT experiments despite applicability claims

> "The study lacks empirical evaluation on Vision Transformers despite theoretical claims regarding the applicability of the logit-similarity weighting mechanism to attention heads. The authors need to provide quantitative experiments on transformer architectures to validate these theoretical claims."

**What the manuscript says.** Section 5.5 ("Applicability to Attention Mechanisms and Vision Transformers") says the principle is "architecture-agnostic" and "could be adapted ... an important direction for future work". The Conclusion says the work is "laying the theoretical groundwork for interpretability in evolving architectures". The Introduction mentions ViTs in the first sentence.

**Critical assessment.**
- The reviewer is right that a claim was made without evidence, but wrong that the paper made a "theoretical claim" that requires validation: Section 5.5 is explicitly labelled future work. This is defensible on scope (thread **D4**), *provided* the manuscript stops implying applicability elsewhere (Conclusion's "laying the theoretical groundwork", and R2's point about "low-resolution terminal bottleneck" wording, which is CNN-specific and undermines the ViT extension argument).
- Practically, a minimal ViT experiment is feasible with the authors' own code base (the repository is a fork of `pytorch-grad-cam`, which supports ViT via `reshape_transform`): fine-tune one ViT-B/16 or DeiT-S on Kvasir-v2, hook the 12 block outputs, reshape tokens to 14×14, apply Grad-CAM per block, and run the enhancer unchanged. If it works, it strengthens the paper considerably; if it does not, the authors learn that before a re-reviewer does.
- Risk of doing it: it adds a new architecture family to a paper whose mandatory revisions (patient-level splits, seeds, classifier metrics) are already compute-heavy. Do it only after mandatory items are done.

**Verdict: DEFEND on scope + soften text.** Rewrite Section 5.5 and the Conclusion so that no applicability is *asserted*; state that extension to transformers is untested. If compute allows after mandatory items, add a single-architecture ViT experiment on Kvasir-v2 as supplementary material and say so. Either path is acceptable to a reasonable reviewer; asserting applicability without evidence is not.

---

## R1 Q8/Q9 — Structure/flow and language editing: "Yes" with no specifics

R3 gives the specifics (promotional wording). Treat R1's "Yes" as covered by the wording pass in `08-manuscript-wording-edits.md` and by a professional language edit. Mention in the response that the manuscript was language-edited.
