# Liver tumor segmentation (nnU-Net v2)

Multiclass liver and tumor segmentation on CT with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).
Training dataset: `Dataset001_LiverTumor` (LiTS train, nnU-Net layout).
Held-out test (later): **3D-IRCADb-01**.

## Experiment design

Four matched **from-scratch** arms (500 epochs, no fine-tune, no loss curriculum).
Start with **fold 0**; full protocol uses **5 folds**; final test on IRCAD.

| Arm | Trainer | Loss idea | Output |
|-----|---------|-----------|--------|
| 1 Baseline | `nnUNetTrainer_500_Baseline` | default Dice+CE | `results_3d_baseline/` |
| 2 Anatomical | `nnUNetTrainer_500_Anatomical` | + boundary ring, liver FP hard-neg, Tversky | `results_3d_anatomical/` |
| 3 Boundary / HD-soft | `nnUNetTrainer_500_BoundaryHD` | + soft Hausdorff proxy | `results_3d_boundary_hd/` |
| 4 Topology | `nnUNetTrainer_500_Topology` | + soft clDice | `results_3d_topology/` |

Local trainers live under `src/3d/` and are registered via
`scripts/3d/run_nnunet_with_local_3d_trainers.py`.

## Requirements

- Python 3.10+
- CUDA GPU recommended
- `requirements.txt` (includes `flake8`)

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
| `nnUNet_results` | Set per arm by the train scripts |

## Data layout

```
nnUNet_raw/Dataset001_LiverTumor/
├── dataset.json
├── imagesTr/          # LiTS train (131)
├── labelsTr/
├── imagesTs/          # 3D-IRCADb-01 test (20): ircad_XX_0000.nii.gz
└── labelsTs/          # GT for metrics: ircad_XX.nii.gz
```

Labels: background `0`, liver `1`, tumor `2`.

Convert IRCAD from the official zip:

```bash
.venv/bin/python scripts/data/convert_3dircadb1_to_nnunet.py \
  --zip /mnt/c/Users/kasid/Downloads/3Dircadb1.zip
```

## Train

```bash
# once
bash scripts/3d/preprocess_nnunet_3d.sh

# fold 0 (default)
bash scripts/3d/train_3d_baseline.sh --skip-preprocess
bash scripts/3d/train_3d_anatomical.sh --skip-preprocess
bash scripts/3d/train_3d_boundary_hd.sh --skip-preprocess
bash scripts/3d/train_3d_topology.sh --skip-preprocess

# later: other folds
FOLD=1 bash scripts/3d/train_3d_baseline.sh --skip-preprocess
```

Post-training fold validation uses `checkpoint_best.pth` by default
(`NNUNET_VALIDATION_WITH_BEST=0` to use the final checkpoint).

## Scripts

### `scripts/3d/`

| Script | Purpose |
|--------|---------|
| `preprocess_nnunet_3d.sh` | plan + preprocess |
| `train_3d_baseline.sh` | arm 1 |
| `train_3d_anatomical.sh` | arm 2 |
| `train_3d_boundary_hd.sh` | arm 3 |
| `train_3d_topology.sh` | arm 4 |
| `train_3d_arm.sh` | shared launcher |
| `run_nnunet_with_local_3d_trainers.py` | registers local trainers |

### `scripts/visualization/`

Matplotlib helpers under `visualizations/`.

### Root `scripts/`

`evaluate_segmentations.py` — Dice / IoU vs reference labels.

## Code style

```bash
.venv/bin/flake8 src scripts
```

## Repository tree

```
liver-tumor-segmentation/
├── README.md
├── requirements.txt
├── src/3d/
│   ├── baseline/
│   ├── anatomical/
│   ├── boundary_hd/
│   ├── topology/
│   ├── common/
│   └── nnunetv2/training/nnUNetTrainer/
├── scripts/3d/
└── nnUNet_* / results_3d_*   # local; gitignored
```

## License

See `LICENSE`.
