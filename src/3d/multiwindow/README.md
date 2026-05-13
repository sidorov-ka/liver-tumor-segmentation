# Multi-window 3D refinement (second stage)

Second-stage **3D full-volume** training with `nnUNetPlans_3d_midres125` / `3d_fullres`. The network sees **four** input channels:

1. **Coarse tumour probability** (softmax tumour class from a stage-1 3D model, e.g. BoundaryOverseg size-gated checkpoint), voxel-aligned with nnU-Net preprocessed spacing.
2. **HU window [-1000, 1000]** — wide “all values” view (Lim et al., *Diagnostics*, 2025).
3. **HU window [0, 1000]** — soft-tissue / liver-friendly contrast.
4. **HU window [400, 1000]** — dense / contrast-rich structures.

Approximate HU is recovered from nnU-Net–normalized intensities using **`dataset_fingerprint.json`** percentiles and global mean/std (same convention as nnU-Net CT preprocessing).

## Loss

Standard nnU-Net **Dice + CE** (with deep supervision) plus a **Tversky** term on the tumour class with **β > α** to penalise false negatives (under-segmentation of large tumours). Weights: `NNUNET_MW_TVERSKY_*` env vars (see `multiwindow/config.py`).

**Lighter default schedule (4ch batches):** `NNUNET_MW_ITER_PER_EPOCH` (default 100) and `NNUNET_MW_VAL_ITER` (default 15). **Epochs:** default **150** (`NNUNET_MW_EPOCHS`; the shell script exports `NNUNET_MW_EPOCHS=150` unless already set).

## End-of-training full-volume validation

`perform_actual_validation` builds the same **4-channel** tensor as training (prob cache + Lim windows). Ensure every validation case has `<case_id>.npz` under `NNUNET_MW_PROB_DIR` (same as training).

If GPU runs out of memory during `fold_*/validation` export, set **`NNUNET_MW_VAL_PERFORM_ON_DEVICE=0`** so sliding-window accumulation stays on CPU (slower, less VRAM).

Multi-window stacks 4 channels per patch; host RAM spikes mainly from **many DA workers** and a **deep `num_cached` queue** in `NonDetMultiThreadedAugmenter`. Defaults are tuned down in code and `train_3d_multiwindow_refinement.sh`:

- `nnUNet_n_proc_DA` defaults to **2** in the shell script if unset (nnU-Net otherwise often uses 12).
- **`NNUNET_MW_MAX_BATCH_SIZE`** caps plan `batch_size` (default **1**; set **`0`** to disable cap and use full plan batch size).
- **`NNUNET_MW_DA_NUM_CACHED_TRAIN`** / **`NNUNET_MW_DA_NUM_CACHED_VAL`** (default **2**) — augmenter queue depth.
- **`NNUNET_MW_PIN_MEMORY`** (default **false**) — avoids extra pinned host pages for CUDA H2D.

For minimum RAM: `nnUNet_n_proc_DA=0` (single-threaded DA, slowest).

## Target spacing and caches

`scripts/3d/train_3d_multiwindow_refinement.sh` defaults to **`1.25 1.0 1.0`** — the same **target spacing** as `configurations.3d_fullres.spacing` in `nnUNet_preprocessed/Dataset001_LiverTumor/nnUNetPlans_3d_midres125.json`, and the same as `scripts/3d/train_nnunet_3d.sh` / `train_3d_boundary_shape.sh`. That matches the usual `data_identifier` folder `nnUNetPlans_3d_midres125_3d_fullres` in this repo.

To experiment with **native** median spacing (`original_median_spacing_after_transp` in the same `plans.json`), re-run **plan_and_preprocess** with that spacing, then **re-cache prob maps**, then:

```bash
NNUNET_MW_TARGET_SPACING="1.0 0.76953125 0.76953125" bash scripts/3d/train_3d_multiwindow_refinement.sh --skip-preprocess
```

**Important:** tumour `prob` `.npz` must match the **same** preprocessed grid as training (`load_case`). **You do not need to re-build prob** after this script default fix if your cache was already produced under the same `1.25 1.0 1.0` preprocess as boundary / `train_nnunet_3d.sh`. Re-cache only if you change target spacing or replace `nnUNet_preprocessed` for `3d_fullres`.

## Run folders

By default each run is written under:

`results_3d_multiwindow_runs/<YYYYMMDD_HHMMSS>_multiwindow_3d_fullres/`

(same timestamp pattern as `results_3d_boundary_shape_runs/`). Override with `RUN_NAME=...` or `RESULTS_ROOT=...`.

## Steps

1. **Preprocess** (once per spacing): run `bash scripts/3d/train_3d_multiwindow_refinement.sh` **without** `--skip-preprocess`, or call `nnUNetv2_plan_and_preprocess` with the same `-overwrite_target_spacing` as `NNUNET_MW_TARGET_SPACING`.

2. **Cache tumour probabilities** (one `.npz` per case, key `prob`, shape `(1, Z, Y, X)`):

```bash
bash scripts/3d/cache_tumor_prob_maps.sh
```

Same as calling `scripts/3d/cache_tumor_prob_for_multiwindow.py` directly:

```bash
.venv/bin/python scripts/3d/cache_tumor_prob_for_multiwindow.py \
  --model-dir results_3d_boundary_shape_runs/20260509_160927_boundary_size_gated/Dataset001_LiverTumor/nnUNetTrainer_150_BoundaryOverseg_50epochs__nnUNetPlans_3d_midres125__3d_fullres \
  --output-dir results_3d_multiwindow_runs/_prob_cache_fold0_size_gated \
  --fold 0 --split all
```

3. **Train** (default: `results_3d_multiwindow_runs/<timestamp>_multiwindow_3d_fullres/`):

```bash
export NNUNET_MW_PROB_DIR="${PWD}/results_3d_multiwindow_runs/_prob_cache_fold0_size_gated"
export NNUNET_PRETRAINED_PARTIAL=1   # partial load from 1-channel 3D baseline
bash scripts/3d/train_3d_multiwindow_refinement.sh --skip-preprocess
```

Optional: `bash scripts/3d/train_3d_multiwindow_refinement.sh --cache-prob --skip-preprocess` to populate `NNUNET_MW_PROB_DIR` from `PROB_MODEL_DIR` / default size-gated run.

4. **Inference** (sliding window + Gaussian + mirroring, same style as nnU-Net 3D):

```bash
.venv/bin/python scripts/3d/infer_multiwindow_refinement_3d.py \
  --model-dir results_3d_multiwindow_runs/<run>/Dataset001_LiverTumor/nnUNetTrainer_150_MultiWindowRefine_50epochs__nnUNetPlans_3d_midres125__3d_fullres \
  --prob-dir "${NNUNET_MW_PROB_DIR}" \
  -i nnUNet_raw/Dataset001_LiverTumor/imagesTr
```

By default predictions go to ``<model-dir>/fold_0/validation/`` (override with ``-o``).

Trainer class: `nnUNetTrainer_150_MultiWindowRefine_50epochs` (registered via `scripts/3d/run_nnunet_with_local_3d_trainers.py`).
