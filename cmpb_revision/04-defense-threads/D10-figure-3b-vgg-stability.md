# Defense Thread D10 — Figure 3b and the "VGG is more stable" conclusion

**Raised by:** R2.4 ("The dip in Figure 3b looks more like an artifact, as evidenced by the large standard deviation ... the conclusion that VGG is more stable may not be fully justified.")
**Defensibility:** Weak for the specific "stability" conclusion; strong for the underlying point that per-layer faithfulness is heterogeneous and a fixed layer choice is unreliable. Concede the artifact, investigate it, and re-anchor the narrative on what the data robustly show.

---

## 1. Why the reviewer is probably right

- The dip is at normalised depth ≈ 0.1 of 53 enumerated conv modules in ResNet-50, i.e., around module index 5, inside `layer1`. torchvision's `layer1` contains three bottleneck blocks (1×1 → 3×3 → 1×1) and a 1×1 shortcut projection (`layer1.0.downsample.0`). A CAM computed on a 1×1 bottleneck or projection conv at 56×56 resolution is a poor spatial explanation; masking the input with it produces near-random masked logits, so the mean similarity is low and the across-image SD is large. That is the signature of a *degenerate layer*, not of an "early structural bottleneck ... before recovering via skip connections" (Sec. 4.3).
- The paper's own Table 2 shows ResNet-50's average raw similarity SD (0.0963) is the lowest among the ResNets. A single-layer outlier, not broad volatility, is driving the visual impression in Figure 3b.
- On the VGG side, "remarkably stable similarity across depth" (raw SD 0.017, Table 2) is equally consistent with the similarity measure being *uninformative* for VGG. Figure 2a shows VGG's per-layer **ROAD** varies substantially across the last five layers, so "stable" S_l does not mean "stably faithful" layers. The two figures contradict each other on VGG and the paper never reconciles them (see S5).

## 2. What is still defensible

The broader point of Section 4.3 survives: which layers give faithful explanations differs across architectures and datasets, and there is no a priori best layer. Figure 2 (per-layer ROAD) shows this more credibly than Figure 3 (per-layer cosine similarity), because ROAD is the paper's own faithfulness criterion.

## 3. What to do

1. **Identify the module at the dip** and report it by name. If it is a 1×1 or projection conv, say so.
2. **Show distributions, not mean ± SD.** Per-layer boxplots or violin plots of S_l across images for ResNet-50; the artifact will be visible as a bimodal or wide distribution at one or two layers.
3. **Rerun Figure 3 with L restricted to 3×3 convs or to block outputs.** If the dip disappears, the "bottleneck" was an enumeration artifact and the text must change. This is the same layer restriction that produces the sparse variant in D1 and the ablation in D9, so it costs nothing extra.
4. **Correlate S_l with per-layer ROAD** (Figure 2 data vs Figure 3 data, same layers). If the weighting criterion is meant to track faithfulness, the correlation should be positive and reasonably strong. Report it whatever it is. A weak correlation is an important finding that reshapes the paper (S5); a strong one is a powerful validation of Phase 2.
5. **Rewrite the narrative** in Sec. 4.3 and Fig. 3 caption: replace "remarkable stability", "early structural bottleneck", "catastrophic structural bottlenecks", "profound discrepancy empirically proves" with descriptive statements ("a small number of layers — concentrated in 1×1 projection convolutions — yield low and highly variable similarity"; "VGG-16 similarities vary little across depth"). Reconcile with Figure 2.
6. **Fix the internal contradiction** between Fig. 3a caption ("severe spatial bottlenecks in ResNet-18, ResNet-34") and Sec. 5.1 ("ResNet-18 preserves textural features, ResNet-34 destroys them").

## 4. Draft rebuttal text

> We agree. On inspection, the low-similarity point at normalised depth ≈ 0.1 corresponds to [module name], a 1×1 [bottleneck/projection] convolution whose activation map is not a meaningful spatial explanation; the masked forward pass through such a map yields near-random logits, hence the low mean and high variance. When the layer set is restricted to 3×3 convolutions / block outputs (Figure 3, revised), the dip is absent. We have removed the description of this point as a "structural bottleneck" and no longer claim that VGG-16 is "more stable"; we now describe the observed distributions directly (per-layer box plots, Figure 3) and report the correlation between per-layer similarity and per-layer ROAD (ρ = …), which [supports / qualifies] the use of similarity as a proxy for layer faithfulness. We also corrected an inconsistency between the caption of Figure 3a and Section 5.1 regarding ResNet-18.

## 5. Residual risk

Low for R2. The correlation analysis in step 4 carries its own risk (S5) but is the honest thing to do and it is much better to run it now than to have a re-reviewer ask for it.
