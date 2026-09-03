# Defense Thread D8 — Patient-level splitting: what is defensible (Kvasir-v2) and what is not (IBS)

**Raised by:** R3-1 (repeat with patient-disjoint train/val/test; report patients and images per class per split).
**Defensibility:** **None for IBS** — the request must be met or, if the public data make it impossible, the impossibility must be demonstrated and its consequences stated. **Defensible for Kvasir-v2**, where patient identifiers do not exist in the public release, and for Kvasir-SEG for the same reason. This thread covers the defensible part and the fallback path for the non-defensible part.

---

## 1. Kvasir-v2: the defensible position

- Kvasir-v2 (Pogorelov et al., MMSys 2017) is distributed as eight class folders of 1,000 JPEGs each, with no patient, examination or sequence metadata. Patient-disjoint splitting is therefore not possible for any user of the public dataset, and the overwhelming majority of the published literature on Kvasir-v2 uses image-level splits for that reason.
- What *can* be done: (i) state the limitation explicitly; (ii) use a stratified image-level **train/val/test** split (not train/val only), with test held out from every stage including T selection and early stopping; (iii) optionally, run a perceptual-hash near-duplicate check (e.g., pHash Hamming distance ≤ k) across splits and remove/merge near-duplicates, reporting how many were found. This last step is cheap and directly addresses the leakage mechanism the reviewer is worried about, even without patient IDs.
- Kvasir-SEG (used in Sec. 4.5) is the Kvasir-v2 polyp class with masks; its 1,000 images have "duplicates in the polyp folder" by the dataset authors' own description. So alignment evaluation must use only the Kvasir-SEG images that fall in the held-out test split (see S2 in `05-self-identified-issues.md`).

### Draft text for the response and the Limitations paragraph

> Kvasir-v2 and Kvasir-SEG are released without patient or examination identifiers, so patient-disjoint partitioning is not possible for these datasets; this is a limitation shared by all studies using them. We mitigate the risk of near-duplicate leakage by (i) holding out a test split that is never used for model selection, temperature selection or reporting, and (ii) screening for near-duplicate frames across splits with perceptual hashing (N pairs found and reassigned to the same split). We state this limitation in Section 3.2.1 and in the Conclusion.

## 2. IBS: not defensible — the path to compliance

### 2.1 Establish whether patient IDs are recoverable

The Dryad deposit (doi:10.5061/dryad.9s4mw6mkp, published 21 Sep 2022) contains four Google-Drive-exported archives (`IBS-…zip`, `IBS-C-…zip`, `IBS-D-…zip`, `normal-…zip`) and a 599-byte README. The public metadata does not describe patient identifiers. The authors must check:

1. **Filenames** inside each archive: endoscopy reporting systems typically export names containing an exam/accession number and a frame index (e.g., `123456_01.jpg`). If a common prefix groups ~20–40 images, that is the exam ID.
2. **Sub-folder structure** inside the archives (Google Drive exports preserve folder hierarchy; there may be one folder per patient).
3. **EXIF/JPEG metadata** (capture timestamps, device fields). Frames from one exam cluster tightly in time.
4. **The README.**

If any of these yields exam-level grouping, build **stratified 5-fold patient-level cross-validation** (stratify by I/C/D/N so that each fold contains IBS patients of each subtype), with an inner validation split for early stopping. Report per fold: patients and images per class. With 35 IBS patients this leaves ~7 IBS patients per test fold, which is why CV rather than a single split is the right design.

### 2.2 If no identifier exists in the public release

- **Contact the data owner** (H. Mihara, University of Toyama, is listed as owner of the images). A per-image patient mapping without any other PHI is likely obtainable for a re-analysis and would be a defensible, documented step.
- **Proxy grouping** as a fallback only: cluster images by capture timestamp (if EXIF present) or by scope/lighting/colour statistics into pseudo-exams, split by cluster, and *state clearly that this approximates patient-level splitting*. Reviewers accept a documented approximation far more readily than an image-level split.
- **Worst case:** if neither is possible, the IBS results cannot support the paper's IBS-specific claims (Introduction, Sections 4.1, 4.3, 5.1). The honest options are (a) demote IBS to a secondary, explicitly-caveated dataset and carry the main claims on Kvasir-v2, or (b) add a second dataset with patient metadata (e.g., HyperKvasir has no patient IDs either; SUN-SEG/PolypGen have case-level structure for polyps but not for IBS). Option (a) is more realistic within the revision window.

### 2.3 Fix the image count

Tabata et al. (PLOS Digit Health 2023) used 2,479 + 382 + 538 + 484 = 3,883 images. The manuscript reports 5,547. If the Dryad archives contain 5,547 files, say "the full public release comprises 5,547 images, of which the original study used 3,883"; give per-class image counts; and describe any filtering (the source study excluded terminal ileum, retroflexion, NBI and dye images and used BBPS 2–3 only — if the extra ~1,660 images include such frames, that is worth knowing).

## 3. Why this matters more for IBS than for a polyp dataset

Each colonoscopy contributes 20–40 frames captured with the same scope, light source, bowel-prep quality and mucosal colour. For a task defined as "subtle textural differences", those exam-level covariates are exactly what a CNN will latch onto if the same exam appears in training and evaluation. The consequences propagate: classifier F1 (unreported), the XAI faithfulness scores on the "validation" set, the layer-wise similarity curves of Figure 3a, and the "IBS signal is textural" narrative all inherit the leakage. This is why R3 calls it the first major issue and why no rebuttal short of re-splitting will satisfy a re-review.

## 4. Draft rebuttal text (IBS, compliant path)

> We agree that an image-level split of a multi-image-per-patient dataset risks patient leakage. We have re-partitioned the IBS dataset at the patient level using [exam identifiers recovered from filenames / a patient mapping kindly provided by the dataset owner], and repeated all IBS experiments under stratified five-fold patient-level cross-validation with an inner validation split for model selection. Table N reports the number of patients and images per class in every fold. All IBS results in Tables 1–6 and Figures 2–4 have been regenerated under this protocol; [summarise how results changed]. We also corrected the description of the dataset size (5,547 images in the public release; per-class counts now reported).
