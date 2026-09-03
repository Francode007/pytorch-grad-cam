# Modal runner — XAI-Enhancer (branch `modal_kvasir`)

Run the full CMPB revision pipeline on [Modal](https://modal.com) instead of Lambda:
download data/models → prepare splits → train → evaluate.

```
modal_runner/
├── app.py           # Modal App + CLI dispatcher (entry point)
├── config.py        # volume paths, GPU/timeouts, secret names
├── image.py         # container image
├── runtime.py       # TORCH_HOME, Kaggle creds, subprocess helpers
└── jobs/
    ├── download.py  # Kvasir / IBS / torchvision weights
    ├── splits.py    # 70/10/20 + IBS folds + smoke
    ├── train.py     # classifier training
    ├── summarize.py # wave cost/metrics tables + batch locks
    ├── reset.py     # wipe seed dirs / locks
    └── evaluate.py  # classification + CAM metrics
```

**Phase 2 operator runbook** (launch / logs / resume while experiments run):
[`../XAI_Enhancer_module/PHASE2_RUNBOOK.md`](../XAI_Enhancer_module/PHASE2_RUNBOOK.md)

Persistent storage is a single Modal Volume (`xai-enhancer-vol`) mounted at `/vol`:

| Path | Contents |
|------|----------|
| `/vol/data/kvasir-v2` | Kvasir-v2 images + splits |
| `/vol/data/IBS-patient-dataset` | Patient-aware IBS (revision default) + folds |
| `/vol/data/ibs_groups.csv` | Exam-id map (from bundled metadata) |
| `/vol/data/IBS-preprocessed-dataset` | Legacy numeric dump (optional) |
| `/vol/models` | `TORCH_HOME` (ImageNet pretrained) |
| `/vol/runs/kvasir` | Kvasir checkpoints + eval outputs |
| `/vol/runs/ibs` | IBS checkpoints + eval outputs |

---

## 1. One-time Modal account setup

Do this on your laptop (repo root: `pytorch-grad-cam`).

### 1.1 Create a Modal account

1. Open [https://modal.com](https://modal.com) and sign up (GitHub login is fine).
2. Confirm you can open the dashboard: [https://modal.com/apps](https://modal.com/apps).

### 1.2 Install the Modal CLI locally

```bash
cd /path/to/pytorch-grad-cam
git checkout modal_kvasir

python3 -m pip install -U modal
# or: python3 -m pip install -r modal_runner/requirements.txt
```

### 1.3 Authenticate (connect this machine to your Modal account)

```bash
python3 -m modal setup
```

This opens a browser, creates a token, and writes `~/.modal.toml`.

Verify:

```bash
modal profile current
modal volume list
```

### 1.4 Create the Kaggle secret (required for dataset downloads)

1. Kaggle → Account → **Create New Token** → downloads `kaggle.json`.
2. Create a Modal secret from that file:

```bash
# kaggle.json looks like: {"username":"...", "key":"..."}
export KAGGLE_USERNAME="$(python3 -c 'import json;print(json.load(open("kaggle.json"))["username"])')"
export KAGGLE_KEY="$(python3 -c 'import json;print(json.load(open("kaggle.json"))["key"])')"

modal secret create kaggle-credentials \
  KAGGLE_USERNAME="$KAGGLE_USERNAME" \
  KAGGLE_KEY="$KAGGLE_KEY"
```

Or pass the values directly:

```bash
modal secret create kaggle-credentials \
  KAGGLE_USERNAME=your_kaggle_username \
  KAGGLE_KEY=your_kaggle_api_key
```

List secrets:

```bash
modal secret list
```

### 1.5 (Optional) Create the volume early

The app creates `xai-enhancer-vol` automatically on first run. To create it yourself:

```bash
modal volume create xai-enhancer-vol
```

---

## 2. After setup — download models and datasets

Always run from the **repository root** on `modal_kvasir`.

### 2.1 Show CLI help

```bash
modal run -m modal_runner.app -- --help
```

### 2.2 Download ImageNet pretrained backbones (CPU)

Writes under `/vol/models` (`TORCH_HOME`):

```bash
modal run -m modal_runner.app -- download-models
```

### 2.3 Download Kvasir-v2 (CPU + Kaggle secret)

```bash
modal run -m modal_runner.app -- download-kvasir
```

This also prepares stratified **70/10/20** splits. To re-run splits with pHash near-dup merging:

```bash
modal run -m modal_runner.app -- prepare-kvasir-splits
# or without dedupe:
modal run -m modal_runner.app -- prepare-kvasir-splits --no-dedupe
```

### 2.4 Download IBS patient dataset (CPU + Kaggle secret)

Uses private [`franchisn/ibs-dataset`](https://www.kaggle.com/datasets/franchisn/ibs-dataset)
(unnormalized, patient folders) → `/vol/data/IBS-patient-dataset` and installs
bundled `ibs_groups.csv` (126 exam groups).

```bash
modal run -m modal_runner.app -- download-ibs-patient
modal run -m modal_runner.app -- prepare-ibs-folds --seed 42
```

Legacy numeric dump (`pre-processed-ibs`) has **no** exam IDs — only use if you
need the old flat tree:

```bash
modal run -m modal_runner.app -- download-ibs   # or ingest-ibs-zip after volume put
```

If `download-ibs-patient` returns **403**, open the dataset in a browser with the
same Kaggle account as the Modal secret and accept terms, then retry.
### 2.5 Inspect the volume

```bash
modal run -m modal_runner.app -- status
modal volume ls xai-enhancer-vol /
modal volume ls xai-enhancer-vol /data
modal volume ls xai-enhancer-vol /models/hub/checkpoints
```

---

## 3. Train and evaluate (A100)

### Smoke train (2 epochs)

```bash
modal run -m modal_runner.app -- train-kvasir --arch resnet18 --smoke
```

### Full single-arch train

```bash
modal run --detach -m modal_runner.app -- train-kvasir --arch resnet50 --seed 42 --epochs 50
```

### Kvasir revision matrix (5 A100s × one seed; repeat for 43, 44)

```bash
# Logout-safe: continues after Mac sleep / closed terminal
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42

# After seed 42 finishes (check Modal dashboard / volume metrics):
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 43
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 44
```

# After each wave, Modal writes a cost/metrics table and locks batch sizes:

```
/vol/runs/kvasir/waves/seed{seed}/wave_summary.json
/vol/runs/kvasir/waves/seed{seed}/wave_summary.txt
/vol/runs/kvasir/waves/seed{seed}/wave.log
/vol/runs/kvasir/locked_batch_sizes.json
```

Clean relaunch after a failed wave (clears bad locks + seed run dirs):

```bash
modal run --detach -m modal_runner.app -- train-kvasir-seed --seed 42 --reset
```

Re-print / re-save a finished wave anytime:

```bash
modal run -m modal_runner.app -- summarize-kvasir-seed --seed 42
```

Pull summaries locally:

```bash
modal volume get xai-enhancer-vol /runs/kvasir/waves ./modal_artifacts/kvasir_waves
modal volume get xai-enhancer-vol /runs/kvasir/locked_batch_sizes.json \
  ./modal_artifacts/locked_batch_sizes.json
```

Resume one arch if a container dies:

```bash
modal run --detach -m modal_runner.app -- \
  train-kvasir --arch vgg16 --seed 42 --resume auto
```

Logs + checkpoints live on the volume under `/vol/runs/kvasir/{arch}/seed{seed}/`
(`train.log`, `args.json` with locked `batch_size`, `checkpoint_latest.pth`,
`checkpoint_mid.pth`, `best.pth`).

### Classification metrics (test split)

```bash
modal run -m modal_runner.app -- eval-kvasir-cls --arch resnet50 --split test
```

### CAM faithfulness (start with a small image cap)

```bash
modal run -m modal_runner.app -- eval-kvasir-cams --arch resnet50 --max-images 50
modal run -m modal_runner.app -- eval-kvasir-cams --arch resnet50   # full test set
```

### IBS revision matrix (5 A100s × one fold; repeat 1–4)

Architectures match Kvasir: `resnet18`, `resnet34`, `resnet50`, `vgg19`, `vgg16`.

```bash
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 0
# After fold 0 (locks batch sizes):
modal run --detach -m modal_runner.app -- train-ibs-fold --fold 1
# ... folds 2, 3, 4

# Or all 25 cells in one map (queues at GPU concurrency limit)
modal run --detach -m modal_runner.app -- train-ibs-cv
```

Wave summaries + locks:

```
/vol/runs/ibs/waves/fold{k}/wave_summary.{json,txt}
/vol/runs/ibs/locked_batch_sizes.json
```

```bash
modal run -m modal_runner.app -- summarize-ibs-fold --fold 0
modal volume get xai-enhancer-vol /runs/ibs/waves/fold0 ./modal_artifacts/ibs_waves/fold0
```

Resume one cell:

```bash
modal run --detach -m modal_runner.app -- \
  train-ibs --arch vgg19 --fold 0 --resume auto
```

Eval (after `best.pth` exists):

```bash
modal run -m modal_runner.app -- eval-ibs-cls --arch resnet50 --fold 0 --split test
modal run -m modal_runner.app -- eval-ibs-cams --arch resnet50 --fold 0 --max-images 50
```

---

## 4. Pull results back to your laptop

```bash
# Kvasir / IBS runs (logs, ckpts, wave summaries)
modal volume get xai-enhancer-vol /runs/kvasir ./modal_artifacts/runs/kvasir
modal volume get xai-enhancer-vol /runs/ibs ./modal_artifacts/runs/ibs

# Split summary CSV
modal volume get xai-enhancer-vol /data/kvasir-v2/splits/split_summary_kvasir.csv \
  ./modal_artifacts/split_summary_kvasir.csv
```

---

## 5. Design notes

- **Jobs are pure Python** under `jobs/`; `app.py` only wraps them with Modal resources (image, volume, GPU, secrets). You can unit-test job logic without Modal.
- **Training/eval call existing modules** via `python -m XAI_Enhancer_module...` so CLI behaviour stays identical to local/Lambda runs.
- **Code sync**: `image.py` uses `add_local_dir` so edits on your laptop are picked up on the next `modal run` (large `data/` / `runs/` / `.pth` are ignored).
- **GPU default**: A100 for train/eval (`config.GPU_TRAIN`). Change there if you want L40S/H100.
- **Secrets**: never commit `kaggle.json`; only the Modal secret `kaggle-credentials`.

---

## 6. Troubleshooting

| Symptom | Fix |
|--------|-----|
| `Secret 'kaggle-credentials' not found` | Run §1.4 |
| `Kaggle credentials missing` | Secret keys must be exactly `KAGGLE_USERNAME` and `KAGGLE_KEY` |
| `Cannot infer patient/exam id` on IBS folds | Provide `--groups-csv` (Phase 1 finding) |
| Out of memory on CAM eval | Lower `--max-images`, or edit `jobs/evaluate.py` batch sizes |
| Stale code in container | Re-run from repo root; confirm you are on `modal_kvasir` |
| Wave cancelled after Mac logout | Use `--detach` on current branch (`.spawn()`, not waiting `.remote()`) |
| Need training stdout | Volume `train.log` or Modal dashboard container logs |

Dashboard: [https://modal.com/apps](https://modal.com/apps) → app `xai-enhancer`.  
More Phase 2 ops: [`PHASE2_RUNBOOK.md`](../XAI_Enhancer_module/PHASE2_RUNBOOK.md).
