#!/usr/bin/env bash
# Full-volume inference: nnU-Net + UncertaintyUNet2d refinement (see scripts/infer_uncertainty.py).
#
# Example:
#   bash scripts/infer_uncertainty.sh -i nnUNet_raw/Dataset001_LiverTumor/imagesTr \
#     --uncertainty-dir results_uncertainty/Dataset001_LiverTumor/fold_0/uncertainty/run_<stamp> \
#     -o inference_comparison/uncertainty
#
# infer_uncertainty.py: omit --update-mode → replace in ROI (training-aligned); pass --update-mode blend for mixing.
# By default skips case_0004 and case_0018 (heavy); use --no-skip-heavy-val to run them.
# Low host RAM: default logits float16 + in-place renorm in Python; try larger --tile-step-size (e.g. 0.9),
# --no-mirroring. Low GPU VRAM: --nnunet-low-vram or --device cpu.
# Interpreter: PYTHON=python3 (default) or PYTHON=python
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python3}"
exec "${PYTHON}" "${REPO_ROOT}/scripts/infer_uncertainty.py" "$@"
