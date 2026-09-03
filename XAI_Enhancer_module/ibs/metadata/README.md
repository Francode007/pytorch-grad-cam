# IBS patient / exam metadata

## `ibs_groups.csv`

Maps each image in the **patient-structured** release
[`franchisn/ibs-dataset`](https://www.kaggle.com/datasets/franchisn/ibs-dataset)
(after flattening to `IBS/` + `Normal/`) to an exam ID.

| Column | Meaning |
|--------|---------|
| `rel_path` | `IBS/Proc….JPG` or `Normal/Proc….JPG` |
| `group_id` | Exam id (`ProcYYYYMMDD#####` or `CNVP…`) — use for `StratifiedGroupKFold` |
| `label` | `IBS` or `Normal` |
| `raw_rel` | Path inside the original Kaggle tree |
| `shard` | Upload shard (`IBS_1`, `Normal_1`…`Normal_3`) — **not** clinical subtype |
| `patient_folder` | Per-shard patient folder (fullwidth digits normalized to ASCII) |

**Counts:** 5547 images, **126** exam groups (16 IBS + 110 Normal).

Filenames alone are enough for `extract_group_id` after the `Proc…_session_frame.JPG`
heuristic was added; the CSV is the canonical audit trail and Modal default.

## Rebuild

```bash
# private dataset — needs Kaggle auth
kaggle datasets download -d franchisn/ibs-dataset --unzip \
  -p data/ibs-raw-unnormalized

python -m XAI_Enhancer_module.ibs.metadata.build_ibs_groups_csv --force
```

## Note on `pre-processed-ibs`

The flat numeric dump (`IBS/2882.jpg`) has the same class counts but **does not**
share recoverable content IDs with this tree (pHash/ORB remapping recovers only a
minority of pairs). Revision patient-level CV uses this patient-aware set, not the
numeric dump.
