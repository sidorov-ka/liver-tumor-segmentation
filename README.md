# Liver tumor segmentation (nnU-Net v2)

Multiclass liver and tumor segmentation on CT with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).
Training dataset: `Dataset001_LiverTumor` (LiTS train). Held-out test: **3D-IRCADb-01**.

## Experiment design

Four matched **from-scratch** arms (500 epochs, standard `nnUNetPlans`):

| Arm | Trainer | Loss | Output |
|-----|---------|------|--------|
| 1 Baseline | `nnUNetTrainer_500_Baseline` | Dice+CE | `results_3d_baseline/` |
| 2 Anatomical | `nnUNetTrainer_500_Anatomical` | + boundary, liver FP, Tversky | `results_3d_anatomical/` |
| 3 Boundary-HD | `nnUNetTrainer_500_BoundaryHD` | + soft Hausdorff | `results_3d_boundary_hd/` |
| 4 Topology | `nnUNetTrainer_500_Topology` | + soft clDice | `results_3d_topology/` |

Custom trainers live in `src/3d/` and are registered via `scripts/3d/run_nnunet_with_local_3d_trainers.py`.

## Setup (DataSphere or local)

**Option A — project venv** (once):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Option B — DataSphere Jupyter kernel** (if deps already installed in kernel):

```python
import os, subprocess, sys
env = os.environ.copy()
env["PYTHON_BIN"] = sys.executable
subprocess.run(["bash", "scripts/3d/train.sh", "preflight"], env=env, check=True)
```

`train.sh` auto-detects Python in order: `PYTHON_BIN` → `.venv` → active `VIRTUAL_ENV` → `python3` on PATH.
If `.venv` already exists with requirements installed, you do **not** need to recreate it.

Upload `nnUNet_raw/` separately (not in git). Set paths if needed:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
```

## Train

Single entrypoint — `scripts/3d/train.sh`:

```bash
bash scripts/3d/train.sh preflight
bash scripts/3d/train.sh plan          # once
bash scripts/3d/train.sh baseline
bash scripts/3d/train.sh anatomical
bash scripts/3d/train.sh boundary-hd
bash scripts/3d/train.sh topology
```

DataSphere Python Console:

```python
import subprocess
for step in ("preflight", "plan", "baseline", "anatomical", "boundary-hd", "topology"):
    subprocess.run(["bash", "scripts/3d/train.sh", step], check=True)
```

Other folds: `FOLD=1 bash scripts/3d/train.sh baseline`

## Data layout

```
nnUNet_raw/Dataset001_LiverTumor/
├── dataset.json
├── imagesTr/    # 131 cases
├── labelsTr/
├── imagesTs/    # IRCAD test
└── labelsTs/
```

Labels: `0` background, `1` liver, `2` tumor.

## What is in git vs not

| In git (push) | Not in git (upload / generate on DataSphere) |
|---------------|-----------------------------------------------|
| `src/3d/` custom trainers & losses | `.venv/` — recreate with `pip install` |
| `scripts/3d/train.sh`, `train_3d_arm.sh`, `resolve_python.sh`, `run_nnunet_with_local_3d_trainers.py` | `nnUNet_raw/` (~18 GB) — upload separately |
| `requirements.txt` | `nnUNet_preprocessed/` — run `train.sh plan` |
| | `nnUNet_results/`, `results_3d_*` — training outputs |
| | `*.pth` checkpoints |

`nnunetv2` itself installs from PyPI into `.venv`; only our **wrappers** (`src/3d/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_500_*.py` + loss code) are in the repo.

## License

See `LICENSE`.
