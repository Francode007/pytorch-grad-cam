# Manuscript Wording Edits

Location → current text → replacement or action. Covers R3-Q8's explicit list, R3-Q1, R2.1, R1.4, and the tone issues in S13. Every replacement is chosen so that the sentence remains true after the revision even if the mechanism claim (S1) has to be softened.

---

## Title

| Current | Replace with |
|---|---|
| "…for highly faithful model explanations in gastrointestinal image classification" | "…for improving the faithfulness of model explanations in gastrointestinal image classification" (or "…for faithful multi-layer explanations…") |

## Abstract

| Current | Replace with |
|---|---|
| "This reliance compromises explanation faithfulness, hindering accurate clinical decision-making." | "This reliance can reduce explanation faithfulness." (do not link to diagnostic accuracy — R3-Q1.1) |
| "XAI-Enhancer generates superior explanation maps." | "XAI-Enhancer produces a single aggregated explanation map." (methods sentence should not assert results) |
| "Rigorous experimentation … demonstrates the robustness and generalizability of the proposed method." | "We evaluate the method on five architectures … and two datasets …" |
| "XAI-Enhancer consistently outperforms existing single-layer CAM methods and multi-layer aggregation techniques" | "XAI-Enhancer improved ROAD in [X of 30] base-CAM configurations and matched or exceeded LayerCAM and HR-CAM in [Y of 10] settings when applied to HiResCAM; improvements were smaller or absent for [cases]." |
| "exposes severe, dataset-dependent structural bottlenecks in residual networks, demonstrating that spatial degradation is erratic and highly unpredictable" | "shows that the layer yielding the most faithful explanation varies with architecture and dataset" |
| "By dynamically compensating for these bottlenecks and effectively balancing…" | Depends on S1 outcome. If weighting helps in some regime: "By weighting layers according to a masked-input similarity criterion…". If not: "By aggregating explanations across all layers…" |
| "produces highly faithful explanations, thereby advancing the trustworthiness of AI-assisted medical diagnostics." | "produces more faithful explanations than terminal-layer CAMs at the cost of an L-fold increase in inference passes (~0.35 s per image, unsuited to frame-rate video) and reduced heat-map stability under input noise." |

## Introduction

| Current | Replace with |
|---|---|
| "conventional CAMs fail fundamentally when applied to complex, holistic conditions like IBS [5]" | "conventional terminal-layer CAMs may be poorly suited to diffuse conditions such as IBS [5]; we treat this as a hypothesis and test it in Section X" |
| "A critical architectural flaw is their exclusive reliance on the terminal convolutional layer" | "A limitation is their reliance on a single, usually terminal, convolutional layer" |
| "Conditions like IBS typically manifest as fine-grained textural anomalies and subtle microvascular deformations, which are best captured by the high-pass filters of early convolutional layers." | "Prior work suggests that the discriminative changes in IBS are sub-visual and diffuse [5, 36]; we hypothesise that such signal is better represented at intermediate depths and test this in Section X." |
| "forcing explanations through a low-resolution terminal bottleneck causes severe spatial diffusion, resulting in coarse, bloated heatmaps" | "explanations derived from a single deep layer inherit that layer's spatial resolution and feature selectivity; in CNNs the terminal layer is both the coarsest and the most class-selective, which can yield diffuse heat-maps" |
| "calculates the objective faithfulness of every intermediate layer" | "estimates the relative informativeness of each intermediate layer's explanation" (S_l is not validated as faithfulness — S5) |
| Contribution 1: "A Novel Hierarchical Explanation Methodology … optimally fuse" | "A training-free, layer-level extension of masked-input weighting (Score-CAM, Group-CAM) that aggregates the explanations of all convolutional layers" (see D7 §2) |
| Contribution 2: "Rigorous Quantitative Evaluation … demonstrating superiority" | "Quantitative evaluation … with confidence intervals and significance tests against single-layer, multi-layer and perturbation-weighted baselines" |
| Contribution 3: "…revealing that feature preservation is fundamentally dependent on…" | "…showing that the most faithful explanation layer depends on both architecture and dataset" |

## Related Work

- Add a paragraph on perturbation-weighted CAMs (Score-CAM, Ablation-CAM, Group-CAM, Opti-CAM) — see D7 §5.
- "Early methodologies like LayerCAM [17] utilize fixed selection rules and naive linear summation, which can paradoxically diminish…" → "LayerCAM [17] fuses selected layers with a fixed rule, which can introduce background noise when shallow layers are included" (drop "naive", "paradoxically").
- Add to Sec. 2.3 a sentence acknowledging that masking-based weighting itself perturbs the input distribution (S10).

## Methods

| Current | Replace with |
|---|---|
| "this formulation ensures the highlighted spatial features preserve the model's complete diagnostic distribution" | "this formulation measures whether the masked image preserves the direction of the logit vector, i.e., the relative pattern of evidence across all classes, independently of its magnitude" (+ centering sentence if adopted; D6) |
| "acts as a powerful spatial regularizer, dramatically reducing layer-wise variance" | "normalises the similarities into a weight distribution over layers" |
| "The optimal parameter of T = 1.0 was selected" | "T = 1 is used as a default; Section 4.6 shows results are insensitive to T over two orders of magnitude" |
| "provides a comprehensive, mathematically faithful visual explanation" | "provides an aggregated visual explanation" |
| Eq. 6: M_l | M_l^up |
| "democratize the feature ensemble" | delete |
| Fig. 1 label "Softmax (Regularization)" | "Softmax normalisation" |

## Results

| Current | Replace with |
|---|---|
| 4.1 "fundamentally challenging the pervasive assumption" | "indicating that the final layer is not always the most faithful" |
| 4.1 "sudden and precipitous degradation … as spatial resolution undergoes its final, heavy compression at the terminal layer" | Remove the resolution explanation (S4). "a marked decrease at the terminal layer (conv5_3), which shares the 14×14 resolution of conv5_1–conv5_2; the decrease is therefore not attributable to resolution loss" |
| 4.1 "profound disparity", "profound unreliability of manual target-layer selection" | "a clear difference"; "the difficulty of selecting a single target layer a priori" |
| 4.1 "This empirically corroborates our hypothesis … far more effective" | "This is consistent with our hypothesis … more faithful on these datasets" |
| 4.2 "reveals substantial improvements", "systematically yields" | "shows improvements in most configurations" |
| 4.2.1 "resulted in consistent enhancements across all three evaluation metrics" | "improved ROAD in 28/30, Deletion in 29/30 and Insertion in 22/30 base-CAM configurations (Table N; significance in Table M)" (update with regenerated counts) |
| 4.2.1 "nearly three-fold improvement" | keep only if it survives the re-split; otherwise report the new number |
| 4.2.2 "demonstrates clear superiority over existing multi-layer aggregation techniques"; "consistently outperformed LayerCAM and HR-CAM" | "compared favourably with LayerCAM and HR-CAM when applied to HiResCAM (9/10 settings on ROAD); when applied to Grad-CAM++ the multi-layer baselines were often better; on IBS/ResNet-34 LayerCAM achieved the highest ROAD of all methods" |
| 4.2.3 "adapts seamlessly", "universal" | "was applicable without modification to" |
| 4.3 "critical diagnostic lens", "catastrophic structural bottlenecks", "plummeting sharply", "severely destruct" | "an analysis tool"; "low similarity at late layers"; "decreases"; "reduce" |
| 4.3 "remarkably stable similarity", "profound discrepancy empirically proves" | "similarity varies little across depth"; "this difference indicates" |
| 4.3 "brilliantly expose", "powerful spatial regularizer", "mathematically ensures", "democratized, highly stable hierarchical ensemble" | delete adjectives; describe the arithmetic |
| 4.3 "Equation (3)" → "Equation (4)"; "Equation 4" → "Equation (5)" | fix cross-references (S7) |
| Table 2 title "Regularization Proof" | "Dispersion of raw similarities and of post-softmax weights across layers" |
| 4.4 "highly localized, clinically faithful explanations" | "more localised explanations" (clinical faithfulness is not established by Figure 4) |
| 4.5 "serving as a robust clinical ground truth" | "serving as a localisation reference" |
| 4.6 title "Ablation Study" | "Sensitivity to the Softmax Temperature" |
| 4.6 "critical inflection point of diminishing returns", "optimal operational threshold", "statistically marginal" | delete; describe monotonic trend and <0.005 range; report the paired test |

## Discussion / Conclusion

| Current | Replace with |
|---|---|
| 5.1 "decisively proves that architectural depth alone is not a reliable proxy" | "indicates that architectural depth alone is not a reliable proxy" |
| 5.1 "ResNet-18 preserves textural features, ResNet-34 destroys them" | reconcile with Fig. 3a caption (S6); describe what the regenerated data show |
| 5.2 "compromises clinical trustworthiness" | "may reduce the usefulness of explanations to clinicians" |
| 5.2 "This paradigm shift suggests that future diagnostic methodologies should move away from treating explainability as a post-hoc, terminal-layer afterthought, and instead adopt architecture-aware strategies that actively preserve hierarchical spatial fidelity." | delete or: "These results suggest that post-hoc explanations should not rely on a single pre-selected layer." (R3-Q1.1) |
| 5.3 "remains highly viable and efficient" | "remains applicable" |
| 5.5 whole subsection | one paragraph; "untested; future work" (D4) |
| 6 "produces highly faithful model explanations" | "improves the faithfulness of model explanations" |
| 6 "Rigorous cross-dataset validation … confirms the universal applicability of the approach, with consistent improvements" | "Evaluation on two datasets and five architectures showed improvements in most configurations (Table N), with exceptions discussed in Section 4.2" |
| 6 "demonstrated clear superiority over existing multi-layer aggregation techniques" | "compared favourably with LayerCAM and HR-CAM in most settings" |
| 6 "revealed that spatial degradation in deep residual networks is highly erratic and fundamentally dataset-dependent" | "showed that the most faithful explanation layer varies with architecture and dataset" |
| 6 "bridges the gap between complex deep learning models and clinical interpretability … paves the way … laying the theoretical groundwork for interpretability in evolving architectures" | delete; add a limitations sentence (latency; stability; CNN-only; no patient metadata for Kvasir; single-centre IBS data) |

## Global find-and-replace candidates

"rigorous(ly)" → delete or "quantitative"; "comprehensive" → delete; "remarkable/remarkably" → delete; "profound(ly)" → delete; "dramatically" → delete; "seamlessly" → delete; "brilliantly" → delete; "vastly" → delete; "severe(ly)" → "marked(ly)" or delete; "catastrophic" → "large"; "decisively proves/empirically proves" → "indicates"; "superior/superiority" → "higher/better" with the number; "optimal(ly)" → only where an optimisation was performed; "highly faithful" → "faithful" or "more faithful"; "universal" → delete; "paradigm shift" → delete; "mathematically" → delete.
