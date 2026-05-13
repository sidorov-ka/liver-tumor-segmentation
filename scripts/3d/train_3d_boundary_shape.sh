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
#   NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_LARGE_TUMOR_THRESHOLD=0.02
#   NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_LARGE_TUMOR_MAX_THRESHOLD=0.10
#   NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_MIN_SCALE=1.0
#   NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_SCHEDULE_START_EPOCH=-1
#   NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_SCHEDULE_RAMP_EPOCHS=0
#   NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_IGNORE_EXTRA_RADIUS=0
#   NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_WEIGHT=0.0
#   NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_THRESHOLD=0.05
#   NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_FRACTION=0.85
#   NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_INVERSE_GATE=0
#   NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_THRESHOLD=0.04
#   NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_TEMPERATURE=0.015
#   NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_MIN_SCALE=0.0
#   NNUNET_BOUNDARY_OVERSEG_BOUNDARY_START_EPOCH=5
#   NNUNET_BOUNDARY_OVERSEG_FP_START_EPOCH=10
#   NNUNET_BOUNDARY_OVERSEG_RAMP_EPOCHS=10
# Reproduce the previous 2026-05-04 Tversky run:
#   source src/3d/boundary_shape/presets/tversky_guard_2026_05_04.env
# Reproduce the failed adaptive large-tumor run:
#   source src/3d/boundary_shape/presets/adaptive_large_tumor_2026_05_09.env
# Run placement:
#   RUN_NAME=my_experiment bash scripts/3d/train_3d_boundary_shape.sh --skip-preprocess
#   RESULTS_ROOT=/abs/path/to/results_root bash scripts/3d/train_3d_boundary_shape.sh --skip-preprocess
# Все раны (включая 20260504_083549_saved_good_boundary) лежат под results_3d_boundary_shape_runs/.
# Массовая перевалидация: bash scripts/3d/revalidate_3d_boundary_shape_runs.sh
# Skip heavy preprocessing if cache is already valid:
#   bash scripts/3d/train_3d_boundary_shape.sh --skip-preprocess
#   SKIP_NNUNET_PREPROCESS=1 bash scripts/3d/train_3d_boundary_shape.sh
#
# Full-volume fold_*/validation uses checkpoint_best.pth (nnU-Net --val_best) by default.
# To write validation from the final epoch instead (old behaviour):
#   NNUNET_VALIDATION_WITH_BEST=0 bash scripts/3d/train_3d_boundary_shape.sh --skip-preprocess
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly BASE_NNUNET_RESULTS="${BASE_NNUNET_RESULTS:-${REPO_ROOT}/nnUNet_results}"
readonly RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_boundary_size_gated}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results_3d_boundary_shape_runs/${RUN_NAME}}"
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

mkdir -p "${RESULTS_ROOT}"
echo "Writing BoundaryOverseg run to: ${RESULTS_ROOT}"

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" \
    -overwrite_target_spacing "${TARGET_SPACING[@]}" \
    -overwrite_plans_name "${PLANS}" \
    --clean
else
  echo "Skipping nnUNetv2_plan_and_preprocess (preprocessed data assumed valid; use after a full preprocess run)."
fi

VAL_BEST_ARGS=()
if [[ "${NNUNET_VALIDATION_WITH_BEST:-1}" != "0" ]]; then
  VAL_BEST_ARGS=(--val_best)
  echo "nnU-Net post-training validation will use checkpoint_best.pth (fold_${FOLD}/validation/)."
fi

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/3d/run_nnunet_with_local_3d_trainers.py" \
  "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  -pretrained_weights "${PRETRAINED_WEIGHTS}" \
  "${VAL_BEST_ARGS[@]}"
