#!/usr/bin/env bash
# Fair comparison: same scripts/infer.py path for nnU-Net-only vs nnU-Net + refiner; outputs under
# inference_comparison/coarse_to_fine/compare_<timestamp>/ (both preds + val_metrics JSON).
#
# By default runs on the validation split only (same case IDs as splits_final.json for FOLD), not full imagesTr.
#
# Usage (from repo root, .venv active):
#   bash scripts/compare_nnunet_vs_refined.sh
#   bash scripts/compare_nnunet_vs_refined.sh refinement_results/.../run_2026_03_28_12_00_00
#   bash scripts/compare_nnunet_vs_refined.sh nnUNet_raw/Dataset001_LiverTumor/imagesTr   # all training cases (not recommended for metrics)
#   bash scripts/compare_nnunet_vs_refined.sh nnUNet_raw/.../imagesTr refinement_results/.../run_*
#
# If refinement dir is omitted, uses the latest coarse_to_fine/run_* under refinement_results.
#
# Optional env: FOLD=0 CHECKPOINT=checkpoint_best.pth TILE_STEP= (unset = infer default 0.5) NO_MIRROR=1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/nnUNet_results}"

readonly DATASET_FOLDER="${DATASET_FOLDER:-Dataset001_LiverTumor}"
readonly LABELS_DIR="${LABELS_DIR:-${REPO_ROOT}/nnUNet_raw/${DATASET_FOLDER}/labelsTr}"

FOLD="${FOLD:-0}"
readonly STAGING_DEFAULT="${REPO_ROOT}/inference_comparison/baseline/_staging_val_fold${FOLD}"

# Args: [input_images_dir] [refinement_run_dir]. One arg: refinement run if it contains checkpoint_best.pth, else input dir.
INPUT=""
REF=""
if [[ $# -eq 0 ]]; then
  INPUT="${STAGING_DEFAULT}"
elif [[ $# -eq 1 ]]; then
  if [[ -d "$1" ]] && [[ -f "$1/checkpoint_best.pth" ]]; then
    REF="$1"
    INPUT="${STAGING_DEFAULT}"
  else
    INPUT="$1"
  fi
else
  INPUT="$1"
  REF="$2"
fi

if [[ -z "${INPUT}" ]]; then
  INPUT="${STAGING_DEFAULT}"
fi
if [[ -z "${REF}" ]]; then
  REF=$(ls -td "${REPO_ROOT}/refinement_results/${DATASET_FOLDER}/fold_${FOLD}/coarse_to_fine/run_"* 2>/dev/null | head -1 || true)
fi
if [[ -z "${REF}" ]] || [[ ! -d "${REF}" ]]; then
  echo "Need a refinement run dir (checkpoint_best.pth + meta.json). Pass as arg 2 or train train_refiner.py first." >&2
  exit 1
fi

CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"

mkdir -p "${STAGING_DEFAULT}"
if [[ "${INPUT}" == "${STAGING_DEFAULT}" ]] || [[ "$(realpath -m "${INPUT}")" == "${STAGING_DEFAULT}" ]]; then
  echo "Val-only input: linking ${STAGING_DEFAULT} from splits_final.json fold ${FOLD}"
  python3 << PY
import json
import os
import sys
from pathlib import Path

repo = Path("${REPO_ROOT}")
pre = repo / "nnUNet_preprocessed" / "${DATASET_FOLDER}" / "splits_final.json"
images_tr = repo / "nnUNet_raw" / "${DATASET_FOLDER}" / "imagesTr"
staging = Path("${STAGING_DEFAULT}")

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
  INPUT="${STAGING_DEFAULT}"
fi

STAMP="$(date +%Y_%m_%d_%H_%M_%S)"
COMPARE="${REPO_ROOT}/inference_comparison/coarse_to_fine/compare_${STAMP}"
OUT_NN="${COMPARE}/nnunet_stage1"
OUT_2S="${COMPARE}/two_stage"
mkdir -p "${COMPARE}"

COMMON=(--fold "${FOLD}" --checkpoint "${CHECKPOINT}")
if [[ -n "${TILE_STEP:-}" ]]; then
  COMMON+=(--tile-step-size "${TILE_STEP}")
fi
if [[ "${NO_MIRROR:-0}" == "1" ]]; then
  COMMON+=(--no-mirroring)
fi

echo "Compare root: ${COMPARE}"
echo "Input: ${INPUT}"
echo "Refiner: ${REF}"
echo "Running nnU-Net stage-1 only -> ${OUT_NN}"
python scripts/infer.py --stage1-only -i "${INPUT}" -o "${OUT_NN}" "${COMMON[@]}"

echo "Running nnU-Net + refiner -> ${OUT_2S}"
python scripts/infer.py --refinement-dir "${REF}" -i "${INPUT}" -o "${OUT_2S}" "${COMMON[@]}"

METRICS_JSON="${COMPARE}/val_metrics_fold${FOLD}.json"
echo "Evaluating -> ${METRICS_JSON}"
python scripts/evaluate_segmentations.py \
  --labels-dir "${LABELS_DIR}" \
  --fold "${FOLD}" \
  --pred nnunet_stage1="${OUT_NN}" \
  --pred two_stage="${OUT_2S}" \
  --output-json "${METRICS_JSON}"

echo "Done. Predictions and metrics under: ${COMPARE}"
