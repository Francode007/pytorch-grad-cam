# Defense Thread D4 — Vision Transformer experiments are out of scope (but stop implying otherwise)

**Raised by:** R1 (provide quantitative ViT experiments to validate "theoretical claims"); R2 ("low-resolution terminal bottleneck" wording contradicts ViT applicability).
**Defensibility:** Strong on scope, *conditional* on removing every sentence that asserts or implies transformer applicability. Optional small experiment if compute permits after mandatory revisions.

---

## 1. The argument

1. **The paper is about convolutional hierarchies.** Title, abstract, methods (Eq. 1: "spatial activation maps A_l ∈ R^{C×H×W}"), and all experiments are CNN-specific. Section 5.5 is explicitly headed as an *applicability* discussion and ends with "an important direction for future work". Future-work statements are not claims requiring validation.
2. **Adding a new architecture family in a major revision is not what the other reviewers asked for**, and it competes for compute with the mandatory items (patient-level re-splitting, seeds, classifier metrics, new baselines). Reviewers and editors accept "beyond scope; stated as future work" when the manuscript is otherwise honest about it.
3. **R2 actually agrees the extension is plausible** ("attention maps across multiple transformer layers can indeed be aggregated as proposed") but objects to the CNN-specific *motivation*. That is a wording fix, and it converges with a wording fix the paper needs for its own CNN story (see S4 in `05-self-identified-issues.md`).

## 2. What must change in the text for the defence to hold

- Introduction: replace "forcing explanations through a low-resolution terminal bottleneck" with a motivation about depth-dependent abstraction and the unreliability of single-layer selection; mention resolution loss as one CNN-specific mechanism.
- Conclusion: delete "laying the theoretical groundwork for interpretability in evolving architectures" (this is the sentence R1 read as a claim).
- Section 5.5: rewrite to one short paragraph: the weighting scheme is defined on any set of intermediate maps and a forward function; transformers expose such maps (token grids per block, attention roll-out); adapting the scheme is untested and left for future work. Do not speculate about attention heads unless you test it.
- Keyword/first sentence of Introduction mention ViTs; fine as context, but do not let the paper appear to promise them.

## 3. Optional experiment (only after mandatory items are done)

A minimal, low-risk design that would turn the defence into evidence:

- Fine-tune **one** ViT-B/16 (or DeiT-S for speed) on Kvasir-v2 with the same split protocol.
- Hook the output of each of the 12 transformer blocks; drop the CLS token; reshape 196 tokens → 14×14 (the `reshape_transform` mechanism already exists in the `pytorch-grad-cam` code base the authors forked).
- Apply Grad-CAM per block, then the enhancer unchanged (mask, forward, cosine, softmax, sum).
- Report ROAD/Ins/Del for last-block Grad-CAM vs enhanced vs uniform average, plus attention roll-out (Abnar & Zuidema 2020) and Chefer et al. (2021) relevance as transformer-native baselines — both already cited by the paper as [42], [43].

If it helps: one supplementary table and a sentence in 5.5. If it does not help: report it as a negative result in the supplement or drop it and keep the future-work sentence; either way the authors know before a re-reviewer does.

## 4. Draft rebuttal text (scope-only version)

> We thank the reviewer for the suggestion. The present study is scoped to convolutional architectures, and Section 5.5 was intended as a discussion of a possible extension rather than a claim. We agree that our wording ("laying the theoretical groundwork...") could be read as a claim and have removed it. We have also revised the Introduction so that the motivation no longer rests on the CNN-specific loss of spatial resolution (as Reviewer 2 also noted), but on the more general observation that no single depth is a priori the most faithful explanation layer. Section 5.5 now states plainly that transformer applicability is untested and is left for future work.

## 5. Draft rebuttal text (with experiment)

> ... In addition, to provide an initial empirical indication, we fine-tuned a ViT-B/16 on Kvasir-v2 and applied the enhancer to per-block token-grid Grad-CAM maps (Supplementary Table S-N). [Summarise result honestly.] We regard this as preliminary and keep the main claims of the paper restricted to CNNs.

## 6. Residual risk

Low with the scope-only version *if* every implied claim is removed. Medium if any applicability language survives, because R1 will read it again.
