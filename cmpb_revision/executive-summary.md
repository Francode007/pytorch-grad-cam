# Executive Summary and Triage

**Manuscript:** CMPB-D-26-01921, *XAI-Enhancer: Hierarchical feature aggregation for highly faithful model explanations in gastrointestinal image classification*
**Decision:** Major revision (Associate Editor: Oliver Faust). Resubmission due **13 Sep 2026**.
**Reviewers:** three. R1 is deployment/robustness-oriented; R2 is broadly positive and asks for clarifications; R3 is the methodologically strict reviewer whose seven "major issues" decide the outcome.

Everything in this analysis is grounded in the two attached documents. Section and table numbers refer to the submitted manuscript (`uploads/main__1__3922.pdf`); reviewer quotes are from `uploads/Review_letter_CPMB_04c1.pdf`.

---

## 1. Bottom line

The reviewers are not disputing the core idea (all three call it well-motivated and R2 calls the CAM+masking synergy "very well-motivated"). They are disputing **the strength of the claims relative to the evidence**, and R3 has identified two things that are genuinely not defensible:

1. **Image-level 80:20 split on a multi-image-per-patient dataset (IBS)** with no held-out test set. Patient leakage inflates classifier validity and, downstream, every XAI metric computed on the "validation" set.
2. **No classifier performance is reported anywhere in the manuscript.** Not a single accuracy, F1, or AUROC value for any of the ten trained models (5 architectures × 2 datasets). Every faithfulness number in Table 1 is uninterpretable without this.

A third issue is decisive for the *narrative*: the manuscript's own Table 4 shows that as T → ∞ (weights collapse to a uniform 1/L average) ROAD, Insertion and Deletion all get monotonically *better*. That means the "dynamic logit-similarity weighting" (the stated key novelty) is not shown to contribute anything measurable to faithfulness beyond plain multi-layer averaging. R2 noticed the monotonicity; R3 did not connect it to novelty. A re-reviewer might. This must be confronted head-on rather than hidden.

Beyond that, most of the remaining points are either (a) legitimate requests for reporting detail that cost little, or (b) partially defensible with a rebuttal *plus* a modest supporting experiment.

---

## 2. Triage matrix

Legend: **WORK** = concede and do the work; **DEFEND** = rebut with argument (a defense thread exists in `04-defense-threads/`); **HYBRID** = rebut the framing but add a small experiment/edit to close the point.

| # | Point | Raised by | Verdict | Effort | Thread |
|---|-------|-----------|---------|--------|--------|
| A1 | Patient-disjoint train/val/test splits for IBS; report patients & images per class per split | R3-1 | **WORK** (mandatory) | High (retrain 5 IBS models) | D8 (Kvasir part only) |
| A2 | Report classifier AUROC/F1/etc. on held-out test for all 10 models; state which data were used for training, model selection, T selection, final evaluation | R3-2, R3-Q3 | **WORK** (mandatory) | Low–Medium | — |
| A3 | Repeated runs / CIs / significance tests | R3-3 | **HYBRID** | Medium | D5 |
| A4 | Cosine similarity on raw logits: justify theoretically | R3-4 | **HYBRID** | Low–Medium | D6 |
| A5 | Table 1 counterexamples; remove "consistently improves every metric" language | R3-5 | **WORK** | Low (text) + re-verify IBS/ResNet-18 rows | — |
| A6 | Compare against Group-CAM, Opti-CAM; state how the method differs | R3-6 | **HYBRID** | Medium | D7 |
| A7 | Latency (20×) and lower perturbation stability must appear in Abstract/Discussion/Conclusion | R3-7, R1 | **WORK** (text) + **DEFEND** (framing) | Low | D1, D3 |
| A8 | Implement sparse layer sampling to hit 33 ms | R1 | **HYBRID** | Medium | D1 |
| A9 | Deployment viability of 2.7 GB VRAM | R1 | **DEFEND** + fix Table 5 inconsistency | Low | D2 |
| A10 | Implement TTA / gradient denoising for robustness | R1 | **HYBRID** | Low–Medium | D3 |
| A11 | Quantitative ViT experiments | R1 | **DEFEND** (scope) + soften text; optional small experiment | Low (text) / Medium (exp) | D4 |
| A12 | "Low-resolution terminal bottleneck" wording is CNN-specific, conflicts with ViT claim | R2 | **WORK** (text) | Trivial | D4 |
| A13 | Ablation hyperparameters missing; which base CAM/backbone for Table 4 (also Tables 3, 5, 6) | R2 | **WORK** | Trivial | — |
| A14 | Table 3: binarised Dice/IoU deflated; use continuous metrics; add other baselines | R2 | **WORK** | Low–Medium | — |
| A15 | Figure 3b dip is an artifact; VGG-stability conclusion not justified | R2 | **HYBRID** | Low | D10 |
| A16 | T=1 not justified; reframe Table 4 as sensitivity analysis; present 0.005 ROAD range as an advantage | R2 | **WORK** (reframe, largely accept R2's suggestion) | Low | D9 |
| A17 | Distinguish explanation quality from diagnostic performance | R3-Q1.1 | **WORK** (text) | Trivial | — |
| A18 | "Fail fundamentally" for IBS too strong; give direct evidence IBS signal is fine-grained/textural | R3-Q1.2 | **HYBRID** | Low (text) + Low–Medium (frequency ablation) | — |
| A19 | Promotional/causal wording; language editing | R3-8, R1/R3-9 | **WORK** | Low | see `08-manuscript-wording-edits.md` |

### Code-level findings (from reading `XAI_Enhancer_module` on branch `kvasir_v1`) — see `09-coding-roadmap.md` §0

| # | Finding | Severity |
|---|---------|----------|
| C1 | The code masks **layer activations** via a forward hook (`normalize_and_mask_activations` + `compute_modified_outputs_batch`); the paper's Eq. 2–3 and Fig. 1 describe masking the **input image**. The only `X ⊙ M` code draws the Fig. 1 illustration. Paper and code must be reconciled before any experiment is re-run. | **Blocking** |
| C2 | `eval_cams.py` defaults to `--enhanced-method stagewise` (hierarchical stage/layer softmax); the paper's Eq. 5 and the T-ablation script use the flat softmax. Verify which produced Table 1. | **Blocking** |
| C3 | `train.py` defaults (AdamW, lr 1e-4, wd 1e-4, cosine, best-by-accuracy) differ from the paper (Adam, lr 1e-3, F1-monitored). | High |
| C4 | No test split exists in `prepare_splits`; Table 3 uses all 1,000 Kvasir-SEG images with Otsu binarisation; Table 6 hard-codes resnet50 with σ=0.05; Table 4 = resnet50 + HiResCAM; Fig. 3 uses every `nn.Conv2d` including 1×1/downsample convs. These answer R2's "which backbone/settings" questions and must go into a protocol table. | Medium |

### Self-identified issues (not raised by reviewers, but a re-reviewer could raise them)

See `05-self-identified-issues.md`. The most important ones:

| # | Issue | Severity |
|---|-------|----------|
| S1 | Table 4 monotonicity ⇒ uniform averaging beats the learned weights; the key novelty has no demonstrated effect on faithfulness | **High** |
| S2 | Kvasir-SEG images *are* the Kvasir-v2 polyp class ⇒ Section 4.5 alignment evaluation includes ~80% training images | **High** |
| S3 | T was selected on the same N=1600 Kvasir-v2 "validation cohort" used for final evaluation and for early stopping | Medium–High |
| S4 | The "final heavy spatial compression at Index −1" explanation for VGG (Sec. 4.1) is factually wrong: conv5_1/5_2/5_3 all have 14×14 resolution | Medium |
| S5 | Figure 2 (per-layer ROAD) and Figure 3 (per-layer cosine similarity) tell opposite stories for VGG and are never reconciled; suggests S_l does not track ROAD faithfulness | Medium–High |
| S6 | Internal contradictions: Sec 5.1 "ResNet-18 preserves textural features" vs Fig 3a caption "severe bottlenecks in ResNet-18"; Fig 2 caption "Index −3" vs text "Index −2" | Medium |
| S7 | Equation cross-references wrong in Sec 4.3 (cites Eq. 3 and 4 for similarity and softmax; should be Eq. 4 and 5); α_l vs S_l notation; Eq. 6 uses M_l not M_l^up | Low (but signals carelessness) |
| S8 | IBS image count 5,547 vs 3,883 in the source publication; per-class image counts never given | Medium |
| S9 | Table 5 VRAM inconsistency (Grad-CAM 2527 MB vs LayerCAM/HR-CAM 389 MB) and single latency number despite L ranging 13–53 | Medium |
| S10 | The masking step (Eq. 3) is hard multiplicative masking, the very OOD artefact the paper criticises in Sec 2.3 | Medium |
| S11 | Likely transcription error in IBS/ResNet-18 rows (identical Ins/Del for Grad-CAM and HiResCAM but different ROAD) | Medium |
| S12 | No specification of the layer set L (includes 1×1 and shortcut-projection convs), of hook location, of Deletion/Insertion step size, of binarisation threshold, of perturbation parameters | Medium |

---

## 3. What to do first (priority order)

1. **Re-split IBS by patient** (A1). Check whether the Dryad filenames encode exam/patient IDs (see D8). If they do, build patient-disjoint train/val/test. If they do not, contact the dataset owner (H. Mihara) for a mapping *and* prepare the fallback in D8.
2. **Retrain all models with the new split protocol and multiple seeds** (A1, A3). Use a 3-way split for Kvasir-v2 as well, so that T selection and model selection happen on validation, and everything reported happens on test (fixes S3).
3. **Report classifier performance** (A2): accuracy, macro-F1, AUROC (one-vs-rest for Kvasir), per-class support, with CIs.
4. **Re-run Table 1 with bootstrap CIs and paired tests over images** (A3). Add a **uniform-average (T→∞) row** and a **Score-CAM / Group-CAM / Opti-CAM** comparison (A6, S1).
5. **Fix Section 4.5** (S2, A14): evaluate only on Kvasir-SEG images in the test split; continuous-valued metrics; all baselines; specify backbone and threshold.
6. **Similarity-function ablation** (A4, D6): cosine on raw logits vs. centered logits vs. softmax-KL vs. target-probability preservation.
7. **Efficiency and robustness additions** (A8, A10): a "sparse layer" variant (block outputs only) with latency/ROAD trade-off curve; a batched implementation; a TTA/SmoothGrad-style variant with SSIM + ROAD.
8. **Rewrite claims and wording** (A5, A7, A12, A16–A19, S4–S7). Add limitations to Abstract, Discussion, Conclusion.
9. Optional if time permits: small ViT experiment (A11). Otherwise, drop applicability claims to a single "future work" sentence.

The mandatory items (1–4) involve retraining ten classifiers several times and re-running ROAD on every configuration. Given the 13 Sep deadline, request an extension from the editor early (Editorial Manager allows this and editors routinely grant it for major revisions that require retraining). State in the request that patient-level re-splitting and seed repetition were requested.

---

## 4. How to read the rest of this analysis

- `01-reviewer-1-analysis.md`, `02-reviewer-2-analysis.md`, `03-reviewer-3-analysis.md`: each comment quoted, what the manuscript actually says (with section/table pointers), whether the reviewer is right, the verdict, and the concrete response.
- `04-defense-threads/`: one file per defensible point, structured as argument → evidence in the manuscript → supporting literature → draft rebuttal text → residual risk and the small addition that neutralises it.
- `05-self-identified-issues.md`: things the reviewers did not catch.
- `06-experiment-plan.md`: the exact experiments, what each one answers, and the order to run them.
- `07-response-letter-skeleton.md`: response-to-reviewers scaffold, numbered to match the review.
- `08-manuscript-wording-edits.md`: line-level wording changes.
