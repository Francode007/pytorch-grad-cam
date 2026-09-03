# Defense Thread D2 — Peak VRAM of 2,724 MB and "standard clinical workstations"

**Raised by:** R1 ("must detail the deployment viability of this memory requirement").
**Defensibility:** Strong. The number is not a problem; the table that contains it is.

---

## 1. The argument

1. **The enhancer's marginal memory cost is small.** From the manuscript's own Table 5, Base Grad-CAM peaks at 2,527 MB and XAI-Enhancer at 2,724 MB: an increase of ~197 MB (+7.8%). Almost all the footprint is the backbone, its activations and the autograd graph needed by *any* gradient-based CAM.
2. **2.7 GB is not a deployment barrier.** Entry-level discrete GPUs have carried 4–8 GB for many years; integrated GPUs on clinical PCs share system RAM. The A100 used in the paper is irrelevant to the requirement; the requirement is 2.7 GB.
3. **Memory and latency trade against each other and both are within budget.** Sequential masked passes (as benchmarked) minimise memory; batching L masked images (D1) raises peak memory by roughly L × per-image activation memory but stays well under typical card sizes for these backbones at 224 px. Report both modes.

## 2. The real weakness: Table 5 is internally inconsistent

Grad-CAM at 2,527 MB and LayerCAM/HR-CAM at 389 MB cannot be the same backbone measured the same way: LayerCAM is a gradient-based method that needs the same graph as Grad-CAM. Plausible explanations, all fixable:

- Grad-CAM (and Enhancer) were measured first and their figure includes CUDA context / cuDNN workspace allocation; LayerCAM/HR-CAM were measured afterwards on a warm process.
- Grad-CAM/Enhancer were measured with `max_memory_allocated()` including model weights and gradients; LayerCAM/HR-CAM with `torch.no_grad()` or on a different backbone.
- Different backbones (VGG-16 weights alone are ~528 MB fp32; ResNet-18 ~45 MB).

If a reviewer or editor notices this, it undermines confidence in the whole table. Re-measure all four (plus new baselines) with an identical protocol: same backbone, warm-up iterations, `torch.cuda.reset_peak_memory_stats()` before each method, `max_memory_allocated()` after, report median over ≥100 images, state backbone and batch mode.

## 3. What to add

- One sentence in Section 5.3: "Peak memory (Table 5) is dominated by the backbone and autograd graph shared with all gradient-based CAMs; the enhancer adds ~X MB in sequential mode and ~Y MB in batched mode, both within the capacity of consumer-grade GPUs."
- Optionally, a CPU-only latency figure for one backbone to show the method runs without a GPU at all (it will be slow, but it demonstrates no hard dependency).

## 4. Draft rebuttal text

> The 2,724 MB peak allocation is dominated by the backbone and the autograd graph, which are required by every gradient-based CAM method; the enhancer's marginal cost is ~X MB in sequential mode. We have re-measured all methods under an identical protocol (same backbone, warm-up excluded, peak statistics reset per method), corrected an inconsistency in the previously reported LayerCAM/HR-CAM figures, and now report both sequential and batched modes. All configurations fit within the 4–8 GB of memory available on entry-level discrete GPUs, and we now state this in Section 5.3.

## 5. Residual risk

Negligible after the re-measurement.
