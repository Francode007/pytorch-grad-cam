# Defense Thread D6 — Why cosine similarity of logit vectors (and how to make the defence airtight)

**Raised by:** R3-4 ("Raw logits are not probabilities, and cosine similarity may change under transformations that preserve the softmax distribution. Please justify this choice theoretically.")
**Defensibility:** Moderate. The reviewer's technical observation is correct; the paper's *justifying sentence* is wrong; but there is a sound rationale for a scale-invariant, all-class similarity, and a one-line modification (centering) removes the reviewer's objection exactly. Back it with an ablation.

---

## 1. What the reviewer is right about

Let Y ∈ R^K be the logit vector. Softmax(Y) is invariant to Y → Y + c·1 for any scalar c (shift by a constant across classes). Cosine similarity cos(Y, Ŷ) is **not** shift-invariant: e.g., Y = (2, 0), Ŷ = (2 + c, 0 + c) have identical softmax for all c but cos → 1/√2·... changes with c. Conversely cosine **is** invariant to Ŷ → a·Ŷ (a > 0), while softmax is not (temperature scaling changes confidence). Therefore "cosine similarity of raw logits" is neither a function of the softmax distribution nor of predictive confidence, and the manuscript's sentence "ensures the highlighted spatial features preserve the model's complete diagnostic distribution" (Sec. 3.1.2) is inaccurate as written.

For the binary IBS task, Y ∈ R², and cos(Y, Ŷ) is just the cosine of the angle between two 2-vectors; this is a low-information signal, and the paper should acknowledge it.

## 2. The rationale that *can* be defended

1. **Scale-invariance is desirable here.** The masked input X ⊙ M̄ zeroes most of the image, which lowers overall activation magnitude and, typically, the norm of the logit vector. A confidence-based criterion (target probability, or Score-CAM's increase-in-confidence) would then penalise every layer for the masking operation itself, not for *which* regions it kept. Cosine asks a different and arguably more appropriate question: does the masked image produce the *same pattern of class evidence*, regardless of how much of it survives? This is the intended semantics; state it.
2. **All-class sensitivity.** Target-class-only criteria (Score-CAM, Group-CAM, Opti-CAM's y_c objective) ignore what happens to the other K−1 logits. A masked region that keeps the target logit high but also raises a confusable class (e.g., esophagitis vs. normal Z-line in Kvasir-v2) is a worse explanation. Cosine over the full vector penalises this; the paper says so already (Sec. 3.1.2, last sentence) and this part of the argument is sound.
3. **Shift-sensitivity is removable by centering.** Define Ỹ = Y − mean(Y)·1 (subtract the mean over classes). Then cos(Ỹ, Ŷ̃) is invariant to any constant shift of either vector *and* to positive scaling. Since the softmax-preserving transformations are exactly the constant shifts, centered cosine is invariant to every softmax-preserving transformation (plus scaling, by design). This answers the reviewer's objection precisely. Centering also removes the shared "mean logit" component that pushes all raw cosines toward 1, which is one reason Table 2 shows raw-similarity SDs as low as 0.017 (VGG-16) and why the softmax weights are near-uniform (see S1).
4. **Relation to divergence measures.** One could instead use −KL(softmax(Y) ‖ softmax(Ŷ)) or −JS. These are shift-invariant but *not* scale-invariant, so they conflate "kept the pattern" with "kept the confidence". Both are legitimate; the choice should be made empirically.

## 3. The ablation that settles it

Run the full pipeline with S_l replaced by each of:

| Variant | Formula | Invariances |
|---|---|---|
| Raw cosine (current) | cos(Y, Ŷ) | scale |
| Centered cosine | cos(Y − Ȳ, Ŷ − Ŷ̄) | scale + shift |
| Probability cosine | cos(softmax Y, softmax Ŷ) | shift |
| Negative KL | −KL(softmax Y ‖ softmax Ŷ) | shift |
| Negative JS | −JS(softmax Y, softmax Ŷ) | shift |
| Target-prob preservation | softmax(Ŷ)_c / softmax(Y)_c | shift |
| Pearson of logits | corr(Y, Ŷ) | scale + shift (equals centered cosine) |

Report ROAD/Ins/Del on one dataset × 2–3 backbones (including ResNet-18, where weights actually vary), with CIs. Also report the resulting weight SD (as in Table 2) to show whether centering widens the dynamic range of the weights. Include the uniform-average (no weighting) row as the reference.

Three possible outcomes, all publishable:

- All variants within noise of each other and of uniform averaging → the choice is inconsequential; say so; the contribution is multi-layer averaging plus a diagnostic signal (see S1).
- Centered cosine (or KL) is better → adopt it; the reviewer improved the method; thank them.
- Raw cosine is best → keep it, but with the corrected theoretical description.

## 4. Text corrections regardless of outcome

- Sec. 3.1.2: replace "preserve the model's complete diagnostic distribution" with "preserve the direction of the model's logit vector, i.e., the relative pattern of evidence across all classes, independently of its magnitude".
- State explicitly that cosine is scale-invariant and (if adopted) that centering makes it shift-invariant, and why scale-invariance is wanted under masking.
- Note the K=2 limitation for IBS.
- Fix notation: the paper uses S_l in Eqs. 4–5 and α_l in Sec. 4.3/Fig. 3; use one symbol.

## 5. Draft rebuttal text

> The reviewer is correct that cosine similarity of raw logit vectors is not a function of the softmax distribution: it is invariant to positive scaling but not to constant shifts, whereas softmax has the opposite invariances. Our description ("preserve the model's complete diagnostic distribution") was inaccurate and has been corrected. Our intent is different from preserving confidence: because the masking operation removes most of the image, the norm of the logit vector generally decreases for every layer; we therefore deliberately use a scale-invariant measure that asks whether the *relative pattern of evidence across all classes* is preserved, penalising masks that keep the target logit high while raising a confusable class. To remove the shift-sensitivity the reviewer identifies, we now center the logits before computing cosine similarity, which makes the score invariant to every softmax-preserving transformation while retaining scale-invariance. We further report an ablation over similarity functions (raw cosine, centered cosine, cosine of probabilities, KL and JS divergence of the softmax distributions, target-probability ratio; new Table N). [Summarise result.] We also note that for the binary IBS task the logit vector is two-dimensional and the similarity signal is correspondingly weak; this is now stated as a limitation.

## 6. Residual risk

Low. The most likely outcome (all variants ≈ uniform) is uncomfortable for the novelty story but is exactly the kind of result R2 says should be "presented as an advantage" (insensitivity to design choices). Pair this thread with the handling of S1.
