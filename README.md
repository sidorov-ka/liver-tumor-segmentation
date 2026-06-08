# Liver tumor segmentation (nnU-Net v2)

Multiclass liver and tumor segmentation on CT with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), dataset id `Dataset001_LiverTumor` (nnU-Net folder layout).

**3D experiments** (local trainers under `src/3d/`, registered via `scripts/3d/run_nnunet_with_local_3d_trainers.py`): baseline `3d_fullres` training, default Dice+CE fine-tune, and boundary/shape loss fine-tuning.

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

### `scripts/3d/`

| Script | Purpose |
|--------|---------|
| `train_nnunet_3d.sh` | Stage-1 nnU-Net `3d_fullres` |
| `train_3d_default_finetune.sh` | 3D baseline fine-tune (default loss) |
| `train_3d_boundary_shape.sh` | 3D boundary/shape fine-tune |
| `run_nnunet_with_local_3d_trainers.py` | Launches nnU-Net with local trainer classes |
| `infer_fuse_softmax_blend.py` | Blend two saved softmax folders into one segmentation |
| `train_voxel_gating_blender.py` | Train per-voxel linear blender on two pred folders |
| `infer_voxel_gating_blender.py` | Apply `blender.pth` to fuse two pred folders |
| `revalidate_3d_boundary_shape_runs.sh` | Re-run `--val` for boundary trainer runs |

### `scripts/visualization/`

Matplotlib helpers (outputs typically under `visualizations/`): `visualize_tumor_slice.py`, `visualize_case_multislice_contact.py`, `visualize_case_three_planes.py`, `plot_val_delta_vs_gt_volume_from_preds.py`, `compare_two_preds_val_slice.py`.

### Root `scripts/`

`evaluate_segmentations.py` — pooled / per-case Dice and IoU vs reference labels.

## Default output locations (all gitignored except `.gitkeep` where noted)

Training and large artifacts stay **out of git** (see `.gitignore`):

- `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`
- `results_3d_default_finetune/`, `results_3d_boundary_shape_runs/`

Point `--model-dir` and result paths at **your** local directories after training.

## Typical pipeline

1. **nnU-Net 3d stage 1:** `bash scripts/3d/train_nnunet_3d.sh` (optional `--skip-preprocess`).
2. **Fine-tune:** `bash scripts/3d/train_3d_default_finetune.sh` (control) and/or `bash scripts/3d/train_3d_boundary_shape.sh` (custom loss).
3. **Metrics:** `python3 scripts/evaluate_segmentations.py --pred-dir … --gt-dir … --output-json metrics.json`.

Boundary loss options and presets: `src/3d/boundary_shape/README.md`, `src/3d/boundary_shape/presets/*.env`.

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
│   └── 3d/
│       ├── default_finetune/
│       ├── boundary_shape/
│       └── nnunetv2/training/nnUNetTrainer/   # trainer shims
├── scripts/
│   ├── 3d/
│   ├── visualization/
│   └── evaluate_segmentations.py
└── nnUNet_* / results_*     # local; see .gitignore
```

## Git push (Cursor / WSL)

If `git push` fails with `vscode-git-…sock` / `ECONNREFUSED`, the repo sets **`git.terminalAuthentication: false`** in `.vscode/settings.json` so the integrated terminal does not use the broken VS Code credential socket. Prefer SSH remote (`git@github.com:…`) after adding your key to GitHub.

## License

See `LICENSE`.
