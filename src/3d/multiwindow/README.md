# Multi-window 3D refinement (second stage)

Second-stage **3D full-volume** training on top of the same preprocessed data as other `nnUNetPlans_3d_midres125` / `3d_fullres` runs. The network sees **four** input channels:

1. **Coarse tumour probability** (softmax tumour class from a stage-1 3D model, e.g. BoundaryOverseg size-gated checkpoint), voxel-aligned with nnU-Net preprocessed spacing.
2. **HU window [-1000, 1000]** — wide “all values” view (Lim et al., *Diagnostics*, 2025).
3. **HU window [0, 1000]** — soft-tissue / liver-friendly contrast.
4. **HU window [400, 1000]** — dense / contrast-rich structures.

Approximate HU is recovered from nnU-Net–normalized intensities using **`dataset_fingerprint.json`** percentiles and global mean/std (same convention as nnU-Net CT preprocessing).

## Loss

Standard nnU-Net **Dice + CE** (with deep supervision) plus a **Tversky** term on the tumour class with **β > α** to penalise false negatives (under-segmentation of large tumours). Weights: `NNUNET_MW_TVERSKY_*` env vars (see `multiwindow/config.py`).

## Steps

1. **Preprocess** (once per machine, same as boundary scripts):  
   `nnUNetv2_plan_and_preprocess ...` or `bash scripts/train_3d_multiwindow_refinement.sh` without `--skip-preprocess`.

2. **Cache tumour probabilities** (one `.npz` per case, key `prob`, shape `(1, Z, Y, X)`):

```bash
bash scripts/cache_tumor_prob_maps.sh
```

Same as calling `scripts/cache_tumor_prob_for_multiwindow.py` directly:

```bash
.venv/bin/python scripts/cache_tumor_prob_for_multiwindow.py \
  --model-dir results_3d_boundary_shape_runs/20260509_160927_boundary_size_gated/Dataset001_LiverTumor/nnUNetTrainer_150_BoundaryOverseg_50epochs__nnUNetPlans_3d_midres125__3d_fullres \
  --output-dir results_3d_multiwindow_runs/_prob_cache_fold0_size_gated \
  --fold 0 --split all
```

3. **Train** (writes under `results_3d_multiwindow_runs/` by default):

```bash
export NNUNET_MW_PROB_DIR="${PWD}/results_3d_multiwindow_runs/_prob_cache_fold0_size_gated"
export NNUNET_PRETRAINED_PARTIAL=1   # partial load from 1-channel 3D baseline
bash scripts/train_3d_multiwindow_refinement.sh --skip-preprocess
```

Optional: `bash scripts/train_3d_multiwindow_refinement.sh --cache-prob --skip-preprocess` to populate `NNUNET_MW_PROB_DIR` from `PROB_MODEL_DIR` / default size-gated run.

4. **Inference** (sliding window + Gaussian + mirroring, same style as nnU-Net 3D):

```bash
.venv/bin/python scripts/infer_multiwindow_refinement_3d.py \
  --model-dir results_3d_multiwindow_runs/<run>/Dataset001_LiverTumor/nnUNetTrainer_150_MultiWindowRefine_50epochs__nnUNetPlans_3d_midres125__3d_fullres \
  --prob-dir "${NNUNET_MW_PROB_DIR}" \
  -i nnUNet_raw/Dataset001_LiverTumor/imagesTr \
  -o /path/to/out_preds
```

Trainer class: `nnUNetTrainer_150_MultiWindowRefine_50epochs` (registered via `scripts/run_nnunet_with_local_3d_trainers.py`).
