# Liver tumor segmentation (nnU-Net v2)

Multiclass liver and tumor segmentation on CT with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), dataset id `Dataset001_LiverTumor` (nnU-Net folder layout).

**Second stages (2D, separate checkpoints from stage 1):**

| Module | Role |
|--------|------|
| `coarse_to_fine` | 2D refiner on exported slices (probability + ROI). `src/2d/coarse_to_fine/`, `scripts/2d/train_coarse_to_fine.py`, `infer_coarse_to_fine.py` |
| `multiview` | Three HU windows + tumor probability. `src/2d/multiview/`, `train_multiview.py`, `infer_multiview.py` |
| `uncertainty` | Three HU windows + probability + entropy. `src/2d/uncertainty/`, `train_uncertainty.py`, `infer_uncertainty.py` |
| `boundary_aware_coarse_to_fine` | Five-channel tiny U-Net; refinement in a boundary ring. `src/2d/boundary_aware_coarse_to_fine/`, `train_boundary_aware_coarse_to_fine.py`, `infer_boundary_aware_coarse_to_fine.py` |

**3D (local trainers under `src/3d/`, registered via `scripts/3d/run_nnunet_with_local_3d_trainers.py`):** default Dice+CE fine-tune, boundary/shape losses, multi-window refinement. Shell entrypoints live in `scripts/3d/` (see table below).

## Requirements

- Python 3.10+
- CUDA GPU recommended for training and full-volume inference
- Dependencies: `requirements.txt` (includes `flake8` for style checks)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## nnU-Net environment variables

If unset, scripts default to directories **at the repository root**:

| Variable | Purpose |
|----------|---------|
| `nnUNet_raw` | Raw dataset (`<repo>/nnUNet_raw`) |
| `nnUNet_preprocessed` | Preprocessed data |
| `nnUNet_results` | nnU-Net training outputs |

## Data layout

```
nnUNet_raw/Dataset001_LiverTumor/
├── dataset.json
├── imagesTr/
└── labelsTr/
```

After changing the training case list, update `numTraining` in `dataset.json`.

## Scripts (what ships in this repo)

### `scripts/2d/`

| Script | Purpose |
|--------|---------|
| `train_nnunet_2d.sh` | Stage-1 nnU-Net 2d (plan, preprocess, fold 0) |
| `export.py` | Slice export (`train/*.npz`, `val/*.npz`) for second stages |
| `train_coarse_to_fine.sh` / `train_coarse_to_fine.py` | coarse_to_fine training |
| `train_multiview.sh` / `train_multiview.py` | multiview training |
| `train_uncertainty.sh` / `train_uncertainty.py` | uncertainty training |
| `train_boundary_aware_coarse_to_fine.sh` / `.py` | boundary-aware training |
| `infer_coarse_to_fine.py` | Full-volume nnU-Net ± coarse_to_fine |
| `infer_multiview.py`, `infer_uncertainty.py`, `infer_boundary_aware_coarse_to_fine.py` | Full-volume inference for those models |
| `run_nnunet2d_validation_export.py` | Optional fold `validation/` export + `--evaluate` |

### `scripts/3d/`

| Script | Purpose |
|--------|---------|
| `train_nnunet_3d.sh` | Stage-1 nnU-Net `3d_fullres` |
| `train_3d_default_finetune.sh` | 3D baseline fine-tune (default loss) |
| `train_3d_boundary_shape.sh` | 3D boundary/shape fine-tune |
| `train_3d_multiwindow_refinement.sh` | 3D multi-window refinement |
| `run_nnunet_with_local_3d_trainers.py` | Launches nnU-Net with local trainer classes |
| `cache_tumor_prob_maps.sh` / `cache_tumor_prob_for_multiwindow.py` | Tumor probability cache for multi-window |
| `infer_multiwindow_refinement_3d.py` | Full-volume inference for multi-window 3D model |
| `infer_fuse_softmax_blend.py` | Blend two saved softmax folders into one segmentation |
| `train_voxel_gating_blender.py` | Train per-voxel linear blender on two pred folders |
| `infer_voxel_gating_blender.py` | Apply `blender.pth` to fuse two pred folders |
| `revalidate_3d_boundary_shape_runs.sh` | Re-run `--val` for boundary trainer runs |

### `scripts/visualization/`

Matplotlib helpers (outputs typically under `visualizations/`): `visualize_tumor_slice.py`, `visualize_case_multislice_contact.py`, `plot_multiview_delta_vs_volume.py`, `plot_val_delta_vs_gt_volume_from_preds.py`, `compare_two_preds_val_slice.py`.

### Root `scripts/`

`evaluate_segmentations.py` — pooled / per-case Dice and IoU vs reference labels.

## Default output locations (all gitignored except `.gitkeep` where noted)

Training and large artifacts stay **out of git** (see `.gitignore`):

- `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`
- `results_coarse_to_fine/`, `results_multiview/`, `results_uncertainty/`, `results_boundary_aware_coarse_to_fine/`
- `results_3d_default_finetune/`, `results_3d_boundary_shape_runs/`, `results_3d_multiwindow_runs/` (placeholder: `results_3d_multiwindow_runs/.gitkeep`)
- `refinement_export/`, `inference_comparison/`, `coarse_to_fine_export/`, etc.

Point `--model-dir`, `--export-dir`, and result paths at **your** local directories after training.

## Typical pipelines

1. **nnU-Net 2d stage 1:** `bash scripts/2d/train_nnunet_2d.sh` (optional `--skip-preprocess`).
2. **Export slices** for second stages: `python3 scripts/2d/export.py --output-dir refinement_export/fold0` (paths configurable).
3. **Train a second stage** with the matching `train_*.sh` / `train_*.py` and `--export-dir`.
4. **Full-volume inference** with the matching `infer_*.py` (tile step defaults to **0.75**, same idea as `export.py`).
5. **nnU-Net 3d:** `bash scripts/3d/train_nnunet_3d.sh`; optional fine-tune scripts above; multi-window needs cached probs then `train_3d_multiwindow_refinement.sh`.
6. **Metrics:** `python3 scripts/evaluate_segmentations.py --pred-dir … --gt-dir … --output-json metrics.json`.

Boundary loss options and presets: `src/3d/boundary_shape/README.md`, `src/3d/boundary_shape/presets/*.env`. Multi-window details: `src/3d/multiwindow/README.md`.

## Code style

```bash
.venv/bin/flake8 src scripts
```

Configuration: `.flake8` (max line length 120, `E203` ignored for slice spacing).

## Repository tree (source and scripts only)

```
liver-tumor-segmentation/
├── README.md
├── requirements.txt
├── LICENSE
├── .flake8
├── src/
│   ├── 2d/
│   │   ├── coarse_to_fine/
│   │   ├── multiview/
│   │   ├── uncertainty/
│   │   └── boundary_aware_coarse_to_fine/
│   └── 3d/
│       ├── default_finetune/
│       ├── boundary_shape/
│       ├── multiwindow/
│       └── nnunetv2/training/nnUNetTrainer/   # trainer shims
├── scripts/
│   ├── 2d/
│   ├── 3d/
│   ├── visualization/
│   └── evaluate_segmentations.py
└── nnUNet_* / results_*     # local; see .gitignore
```

## License

See `LICENSE`.
