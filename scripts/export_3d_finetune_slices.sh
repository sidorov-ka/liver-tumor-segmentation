#!/usr/bin/env bash
# Export train/val .npz slices for multiview / coarse_to_fine from a 3D nnU-Net
# fine-tune folder (BoundaryOverseg fullres), using nnUNetPredictor + checkpoint_best.
#
# Usage:
#   bash scripts/export_3d_finetune_slices.sh size_gated
#   bash scripts/export_3d_finetune_slices.sh adaptive_large
#   bash scripts/export_3d_finetune_slices.sh all
# For any other 3D nnU-Net folder, call export.py directly with --model-dir / --output-dir.
#
# Outputs (defaults, under repo):
#   refinement_export/fold0_3d_size_gated/
#   refinement_export/fold0_3d_adaptive_large/
#
# Requires: nnUNet_raw, nnUNet_preprocessed (or pass through env / export.py flags).
# Extra args after preset name are forwarded to scripts/export.py (e.g. --device cpu).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer repo venv so numpy/nnU-Net match training; override with PYTHON=...
if [[ -z "${PYTHON:-}" ]] && [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
BO="nnUNetTrainer_150_BoundaryOverseg_50epochs__nnUNetPlans_3d_midres125__3d_fullres"
RUNS="${REPO_ROOT}/results_3d_boundary_shape_runs"

run_one() {
  local model_dir="$1"
  local out_dir="$2"
  shift 2
  if [[ ! -d "${model_dir}" ]]; then
    echo "ERROR: model-dir not found: ${model_dir}" >&2
    exit 1
  fi
  mkdir -p "${out_dir}"
  echo "=== export.py ===" >&2
  echo "  model-dir: ${model_dir}" >&2
  echo "  output-dir: ${out_dir}" >&2
  "${PYTHON}" "${REPO_ROOT}/scripts/export.py" \
    --output-dir "${out_dir}" \
    --model-dir "${model_dir}" \
    --dataset-folder Dataset001_LiverTumor \
    --fold 0 \
    --checkpoint checkpoint_best.pth \
    "$@"
}

preset="${1:?usage: $0 size_gated|adaptive_large|all [-- export.py extra args]}"
shift || true

case "${preset}" in
  size_gated)
    run_one "${RUNS}/20260509_160927_boundary_size_gated/Dataset001_LiverTumor/${BO}" \
      "${REPO_ROOT}/refinement_export/fold0_3d_size_gated" "$@"
    ;;
  adaptive_large)
    run_one "${RUNS}/20260509_131406_boundary_adaptive_large_tumor/Dataset001_LiverTumor/${BO}" \
      "${REPO_ROOT}/refinement_export/fold0_3d_adaptive_large" "$@"
    ;;
  all)
    "$0" size_gated "$@"
    "$0" adaptive_large "$@"
    ;;
  *)
    echo "usage: $0 size_gated|adaptive_large|all [-- export.py args]" >&2
    exit 1
    ;;
esac
