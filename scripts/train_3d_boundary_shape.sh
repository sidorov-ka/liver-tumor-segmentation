#!/usr/bin/env bash
# Fine-tune the 150-epoch 3D nnU-Net baseline with boundary/over-segmentation loss.
#
# Optional overrides:
#   NNUNET_BOUNDARY_OVERSEG_EPOCHS=50
#   NNUNET_BOUNDARY_OVERSEG_LR=1e-3
#   NNUNET_BOUNDARY_OVERSEG_BOUNDARY_WEIGHT=0.10
#   NNUNET_BOUNDARY_OVERSEG_OVERSEG_WEIGHT=0.05
#   NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_FP_WEIGHT=4.0
#   NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_FP_WEIGHT=0.5
#   NNUNET_BOUNDARY_OVERSEG_BOUNDARY_RADIUS=2
#   NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_IGNORE_RADIUS=2
#   NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_IGNORE_RADIUS=4
#   NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_TOPK_FRACTION=0.01
#   NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_TOPK_FRACTION=0.002
#   NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_VOLUME_GUARD_THRESHOLD=0.02
#   NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_VOLUME_GUARD_MIN_SCALE=0.10
#   NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_WEIGHT=0.05
#   NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_ALPHA=0.30
#   NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_BETA=0.70
#   NNUNET_BOUNDARY_OVERSEG_BOUNDARY_START_EPOCH=5
#   NNUNET_BOUNDARY_OVERSEG_FP_START_EPOCH=10
#   NNUNET_BOUNDARY_OVERSEG_RAMP_EPOCHS=10
# Skip heavy preprocessing if cache is already valid:
#   bash scripts/train_3d_boundary_shape.sh --skip-preprocess
#   SKIP_NNUNET_PREPROCESS=1 bash scripts/train_3d_boundary_shape.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly BASE_NNUNET_RESULTS="${BASE_NNUNET_RESULTS:-${REPO_ROOT}/nnUNet_results}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results_3d_boundary_shape}"
export nnUNet_results="${RESULTS_ROOT}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD=0
readonly TRAINER="nnUNetTrainer_150_BoundaryOverseg_50epochs"
readonly PLANS="nnUNetPlans_3d_midres125"
readonly TARGET_SPACING=(1.25 1.0 1.0)
readonly PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${BASE_NNUNET_RESULTS}/Dataset001_LiverTumor/nnUNetTrainer_150__${PLANS}__${CONFIGURATION}/fold_0/checkpoint_final.pth}"

SKIP_PREPROCESS=0
if [[ "${1:-}" == "--skip-preprocess" ]]; then
  SKIP_PREPROCESS=1
elif [[ "$#" -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  exit 2
fi
if [[ "${SKIP_NNUNET_PREPROCESS:-0}" == "1" ]]; then
  SKIP_PREPROCESS=1
fi

if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "Missing pretrained weights: ${PRETRAINED_WEIGHTS}" >&2
  exit 1
fi

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" \
    -overwrite_target_spacing "${TARGET_SPACING[@]}" \
    -overwrite_plans_name "${PLANS}" \
    --clean
else
  echo "Skipping nnUNetv2_plan_and_preprocess (preprocessed data assumed valid; use after a full preprocess run)."
fi

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/run_nnunet_with_local_3d_trainers.py" \
  "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  -pretrained_weights "${PRETRAINED_WEIGHTS}"
