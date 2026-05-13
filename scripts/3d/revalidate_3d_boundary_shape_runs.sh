#!/usr/bin/env bash
# Re-run full-volume nnU-Net validation for every BoundaryOverseg experiment under
# results_3d_boundary_shape_runs/ (same trainer/plans, only --val).
#
# Default: --val_best (checkpoint_best.pth). For final-epoch weights:
#   VALIDATE_WITH_BEST=0 bash scripts/3d/revalidate_3d_boundary_shape_runs.sh
#
# Optional:
#   RUNS_ROOT=/abs/path/to/results_3d_boundary_shape_runs
#   RUN_FILTER=substring (matched against run folder name); empty = all runs
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/results_3d_boundary_shape_runs}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD=0
readonly TRAINER="nnUNetTrainer_150_BoundaryOverseg_50epochs"
readonly PLANS="nnUNetPlans_3d_midres125"

VAL_ARGS=(--val)
if [[ "${VALIDATE_WITH_BEST:-1}" != "0" ]]; then
  VAL_ARGS+=(--val_best)
fi

if [[ ! -d "${RUNS_ROOT}" ]]; then
  echo "RUNS_ROOT not found: ${RUNS_ROOT}" >&2
  exit 1
fi

shopt -s nullglob
RUN_DIRS=("${RUNS_ROOT}"/*/)
shopt -u nullglob

for run_path in "${RUN_DIRS[@]}"; do
  name="$(basename "${run_path}")"
  if [[ ! -d "${run_path}Dataset001_LiverTumor" ]]; then
    continue
  fi
  if [[ -n "${RUN_FILTER:-}" ]] && [[ "${name}" != *"${RUN_FILTER}"* ]]; then
    continue
  fi
  export nnUNet_results="${run_path%/}"
  echo "================ ${name} (nnUNet_results=${nnUNet_results}) ================"
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/3d/run_nnunet_with_local_3d_trainers.py" \
    "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
    -tr "${TRAINER}" \
    -p "${PLANS}" \
    "${VAL_ARGS[@]}"
done

echo "Done. Validation outputs are under each run's fold_${FOLD}/validation/"
