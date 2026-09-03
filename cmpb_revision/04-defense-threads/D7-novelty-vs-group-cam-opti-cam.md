# Defense Thread D7 — Novelty relative to Score-CAM, Group-CAM, Opti-CAM (and Ablation-CAM)

**Raised by:** R3-6 ("establish novelty through direct comparison with relevant multi-layer and optimization-based explanation methods, including Group-CAM and Opti-CAM where applicable ... explain precisely how XAI-Enhancer differs").
**Defensibility:** Moderate. There *is* a clean, honest distinction — the enhancer weights **layers**, the cited methods weight **channels within one layer** — and the two are orthogonal and composable. But the manuscript currently omits the closest prior work entirely, which reads as either unawareness or evasion. Fix the related work, state the distinction as incremental-but-real, and run the comparison.

---

## 1. What the closest prior methods do (verified against their papers)

| Method | Unit weighted | How the weight is obtained | Model queries per image | Layers |
|---|---|---|---|---|
| **Score-CAM** (Wang et al., CVPRW 2020) | each channel k of one layer | upsample & normalise A_k, mask input, forward pass, weight = increase in target-class score vs. baseline | C (512–2048) | 1 |
| **Group-CAM** (Zhang, Rao, Yang, arXiv 2021) | groups of channels of one layer | sum channels in group → initial mask → de-noise → blend with blurred input → forward pass → weight = target confidence | G (~32) | 1 |
| **Opti-CAM** (Zhang, Torres, Sicre, Avrithis, Ayache, CVIU 2024) | channels of one layer, weights optimised | mask input with current combination, maximise target logit of masked image by gradient descent (Adam) | ~100 fwd+bwd | 1 |
| **Ablation-CAM** (Ramaswamy, WACV 2020; cited as [27]) | each channel | zero the channel, measure drop in target score | C | 1 |
| **LayerCAM** (Jiang et al., TIP 2021; cited) | pixel-wise gradient weighting; multi-layer by fixed fusion | gradients | 1 backward | several, fixed rule |
| **HR-CAM** (Shinde et al., MICCAI 2019; cited) | multi-level features | learned aggregation (requires training) | 1 | several, learned |
| **DE-SSG-CAM** ([28]; cited) | layer selection | evolutionary optimisation offline | offline | several, precomputed |
| **XAI-Enhancer** | each layer l | upsample & normalise M_l, mask input, forward pass, weight = softmax(cos(Y, Ŷ_l)) | L (13–53) | all |

Phase 2 of XAI-Enhancer is structurally identical to the Score-CAM/Group-CAM weight step (mask → forward → score), with three differences: the *unit* is a layer's CAM rather than a channel/group; the *score* is a full-vector similarity rather than a target-class confidence; and there is a softmax over units. Opti-CAM differs further by optimising rather than scoring.

## 2. The honest novelty statement

> XAI-Enhancer transfers the masked-forward-pass weighting principle of Score-CAM/Group-CAM from the channel axis of a single layer to the layer axis of the whole network. It is therefore orthogonal to those methods: any single-layer CAM (including Score-CAM or Opti-CAM) can serve as the base explainer in Phase 1, and the enhancer aggregates the resulting per-layer maps. Compared to existing multi-layer approaches it requires no training (unlike HR-CAM), no offline optimisation (unlike DE-SSG-CAM), and no fixed layer-selection rule (unlike LayerCAM's fusion). Compared to channel-level masking methods it requires L rather than C model queries.

That is a real contribution. It is *not* "a novel training-free weighting via counterfactual inputs" in general — that idea belongs to Score-CAM (2020). The contributions bullet in the Introduction ("A Novel Hierarchical Explanation Methodology") should be rewritten along the lines above.

## 3. Direct comparison: what to run

- **Baselines to add to Table 1 (and Table 3):** Score-CAM, Group-CAM, and Opti-CAM at the terminal layer (the standard usage) on all backbones/datasets. Score-CAM is already in the forked `pytorch-grad-cam` code base; Group-CAM and Opti-CAM have public PyTorch implementations. Opti-CAM is slow (~100 iterations); if compute is tight, run it on a subset with CIs and say so.
- **Composability demonstration:** XAI-Enhancer wrapping Score-CAM (Phase 1 base = Score-CAM per layer). Even on one backbone, this shows orthogonality concretely.
- **Latency for each** in Table 5 — this supports D1.
- **Uniform-average multi-layer baseline** (T→∞) — required by S1 regardless.

## 4. What if a channel-level method wins on some metric?

Report it. Opti-CAM in particular is optimised directly for masked-logit maximisation and tends to do very well on Insertion/Deletion-style metrics (its own paper reports this and also argues that localisation and classifier-faithfulness are not aligned). The enhancer's differentiators are then: (i) faithfulness across *multiple* metrics including ROAD, (ii) L vs. ~100 queries, (iii) ability to wrap Opti-CAM itself, (iv) the diagnostic per-layer signal (Section 4.3). A paper can lose one cell to a strong baseline and still be accepted if it is honest and its positioning is correct.

## 5. Text to add to Related Work (Sec. 2.2)

A short paragraph on **perturbation-weighted CAMs**: Score-CAM, Ablation-CAM, Group-CAM, Opti-CAM — what they weight, how, and that they operate within a single layer. Then position XAI-Enhancer as the layer-axis analogue. Cite all four.

## 6. Draft rebuttal text

> We thank the reviewer for pointing to Group-CAM and Opti-CAM; together with Score-CAM and Ablation-CAM they constitute the closest prior work and should have been discussed. Related Work now includes a paragraph on perturbation-weighted CAM methods. The precise distinction is as follows: these methods derive weights for *channels (or channel groups) within a single layer* by masking the input with each channel's map and re-querying the model (Score-CAM, Group-CAM) or by optimising the channel weights to maximise the masked target logit (Opti-CAM). XAI-Enhancer applies the masked-query principle along the *layer* axis, weighting the CAM of each layer of the network, and is therefore orthogonal to and composable with them: any of these methods can serve as the per-layer base explainer. We now report Score-CAM, Group-CAM and Opti-CAM as terminal-layer baselines in Tables 1, 3 and 5, and demonstrate composability by reporting XAI-Enhancer applied to Score-CAM on [backbone]. [Summarise results, including any cells where a baseline is best.] We have rewritten the contributions list accordingly.

## 7. Residual risk

Medium until the baselines are run; low afterwards. The main risk is compute time for Opti-CAM; mitigate with a subset and CIs.
