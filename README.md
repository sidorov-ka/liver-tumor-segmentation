# Liver tumor segmentation (nnU-Net v2)

Multiclass CT segmentation of liver and tumor using [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) on the LiTS-style dataset (`Dataset001_LiverTumor`).

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) dependencies (installed via `requirements.txt`)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Training data must follow the [nnU-Net raw dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md):

```
nnUNet_raw/Dataset001_LiverTumor/
├── dataset.json
├── imagesTr/
└── labelsTr/
```

Update `dataset.json` `numTraining` when adding or removing cases.

## Training

From the repository root (with `.venv` activated):

```bash
bash scripts/train.sh
```

The script:

1. Exports `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` to paths under the repo root if unset.
2. Runs planning and preprocessing for configuration `2d` with `--clean`.
3. Trains fold `0` with `nnUNetTrainer_100epochs` (100 epochs, 2D U-Net).

## Environment variables

| Variable | Description |
|----------|-------------|
| `nnUNet_raw` | Raw datasets (default: `<repo>/nnUNet_raw`) |
| `nnUNet_preprocessed` | Preprocessed caches (default: `<repo>/nnUNet_preprocessed`) |
| `nnUNet_results` | Checkpoints and logs (default: `<repo>/nnUNet_results`) |

## Training configuration (current defaults)

| Setting | Value |
|---------|--------|
| Dataset ID | `1` |
| Configuration | `2d` |
| Fold | `0` |
| Trainer | `nnUNetTrainer_100epochs` |
| Preprocess workers | `-np 1 -npfp 1` (low RAM) |

Resume training from the latest checkpoint:

```bash
export nnUNet_raw=$PWD/nnUNet_raw nnUNet_preprocessed=$PWD/nnUNet_preprocessed nnUNet_results=$PWD/nnUNet_results
nnUNetv2_train 1 2d 0 -tr nnUNetTrainer_100epochs --c
```

## Repository layout

```
liver-tumor-segmentation/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
├── nnUNet_raw/
├── nnUNet_preprocessed/
├── nnUNet_results/
└── scripts/
    └── train.sh
```

## License

See `LICENSE`.
