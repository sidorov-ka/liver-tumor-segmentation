#!/usr/bin/env bash
# Full-volume nnU-Net inference on validation cases only (same IDs as in splits_final.json for the chosen fold).
# Tuned for lower host RAM: larger sliding-window step, no TTA, sequential preprocessing/export.
#
# Usage (from repo root, venv active):
#   bash scripts/predict_val_fold_lowmem.sh
#   bash scripts/predict_val_fold_lowmem.sh --continue   # skip cases already present in OUT_DIR
#
# Environment:
#   FOLD        fold index (default: 0), must match training
#   STEP_SIZE   sliding-window step (default: 0.75; higher = less RAM)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/nnUNet_results}"

readonly DATASET_FOLDER="Dataset001_LiverTumor"
readonly FOLD="${FOLD:-0}"
readonly STEP_SIZE="${STEP_SIZE:-0.75}"
# Experiment layout: inference_comparison/baseline/ — nnU-Net val-only preds; coarse_to_fine/ for two-stage infer.py.
readonly STAGING="${REPO_ROOT}/inference_comparison/baseline/_staging_val_fold${FOLD}"
readonly OUT_DIR="${REPO_ROOT}/inference_comparison/baseline/nnunet_val_fold${FOLD}_preds"

CONTINUE=()
if [[ "${1:-}" == "--continue" ]] || [[ "${1:-}" == "-c" ]]; then
  CONTINUE=(--continue_prediction)
fi

mkdir -p "${STAGING}" "${OUT_DIR}"

python3 << PY
import json
import os
import sys
from pathlib import Path

repo = Path("${REPO_ROOT}")
pre = repo / "nnUNet_preprocessed" / "${DATASET_FOLDER}" / "splits_final.json"
images_tr = repo / "nnUNet_raw" / "${DATASET_FOLDER}" / "imagesTr"
staging = Path("${STAGING}")

if not pre.is_file():
    print(f"Missing {pre}", file=sys.stderr)
    sys.exit(1)
if not images_tr.is_dir():
    print(f"Missing {images_tr}", file=sys.stderr)
    sys.exit(1)

splits = json.loads(pre.read_text(encoding="utf-8"))
fold = int("${FOLD}")
if fold < 0 or fold >= len(splits):
    print(f"fold {fold} out of range (len={len(splits)})", file=sys.stderr)
    sys.exit(1)
val_ids = splits[fold]["val"]
n = 0
for cid in val_ids:
    for src in sorted(images_tr.glob(f"{cid}_*.nii.gz")):
        dst = staging / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        n += 1
if n == 0:
    print(f"No files linked under {images_tr} for val cases.", file=sys.stderr)
    sys.exit(1)
print(f"Linked {n} image files for {len(val_ids)} val cases -> {staging}", flush=True)
PY

echo "Writing predictions to ${OUT_DIR}"
echo "step_size=${STEP_SIZE} --disable_tta -npp 0 -nps 0"

# Do not append empty array — "${CONTINUE[@]:-}" can pass a stray "" and break argparse.
PRED_ARGS=(
  -i "${STAGING}"
  -o "${OUT_DIR}"
  -d 1
  -c 2d
  -f "${FOLD}"
  -tr nnUNetTrainer_100epochs
  -chk checkpoint_best.pth
  -step_size "${STEP_SIZE}"
  --disable_tta
  -npp 0
  -nps 0
)
if [[ ${#CONTINUE[@]} -gt 0 ]]; then
  PRED_ARGS+=("${CONTINUE[@]}")
fi
nnUNetv2_predict "${PRED_ARGS[@]}"
