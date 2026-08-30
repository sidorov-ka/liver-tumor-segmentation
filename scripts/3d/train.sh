#!/usr/bin/env bash
# Single entrypoint: preflight, plan, and train all four loss arms.
#
#   bash scripts/3d/train.sh preflight
#   bash scripts/3d/train.sh plan
#   bash scripts/3d/train.sh baseline
#   bash scripts/3d/train.sh anatomical
#   bash scripts/3d/train.sh boundary-hd
#   bash scripts/3d/train.sh topology
#
# DataSphere Python Console:
#   import subprocess
#   subprocess.run(["bash", "scripts/3d/train.sh", "plan"], check=True)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=resolve_python.sh
source "${REPO_ROOT}/scripts/3d/resolve_python.sh"
PYTHON_BIN_SET="${PYTHON_BIN:-}"
require_nnunet_python "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD="${FOLD:-0}"
readonly PLANS="${PLANS:-nnUNetPlans}"
readonly PLANS_JSON="${nnUNet_preprocessed}/Dataset001_LiverTumor/${PLANS}.json"

usage() {
  cat <<'EOF'
Usage: bash scripts/3d/train.sh <mode>

Modes:
  preflight     Validate raw dataset (read-only)
  plan          Standard nnUNetv2_plan_and_preprocess (once)
  baseline      nnUNetTrainer_500_Baseline
  anatomical    nnUNetTrainer_500_Anatomical
  boundary-hd   nnUNetTrainer_500_BoundaryHD
  topology      nnUNetTrainer_500_Topology

Env: PYTHON_BIN, FOLD, PLANS, nnUNet_raw, nnUNet_preprocessed, NNUNET_* process counts.

Python resolution (DataSphere-friendly):
  1. PYTHON_BIN if set (e.g. Jupyter kernel: export PYTHON_BIN="$(which python)")
  2. .venv/bin/python in repo root
  3. active $VIRTUAL_ENV
  4. python3 from PATH
EOF
}

check_runtime() {
  TRAIN_MODE="${TRAIN_MODE:-}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
from importlib.metadata import PackageNotFoundError, version

print(f"python={sys.executable}")

import torch

try:
    nnunet_version = version("nnunetv2")
except PackageNotFoundError as exc:
    raise SystemExit("nnunetv2 is not installed") from exc

print(f"torch={torch.__version__}")
print(f"nnunetv2={nnunet_version}")
if nnunet_version != "2.6.4":
    raise SystemExit(f"Expected nnunetv2==2.6.4, got {nnunet_version}")

mode = os.environ.get("TRAIN_MODE", "")
allow_cpu = os.environ.get("ALLOW_CPU", "").strip().lower() in ("1", "true", "yes", "y")
if mode not in ("preflight",) and not allow_cpu and not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required. Set ALLOW_CPU=1 only for preflight.")
if mode not in ("preflight",) and torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY
}

run_preflight() {
  export TRAIN_MODE="preflight"
  check_runtime
  "${PYTHON_BIN}" - <<'PY'
import gzip
import json
import os
import shutil
from pathlib import Path

raw = Path(os.environ["nnUNet_raw"]) / "Dataset001_LiverTumor"
for path in (raw / "dataset.json", raw / "imagesTr", raw / "labelsTr"):
    if not path.exists():
        raise SystemExit(f"Missing required path: {path}")

meta = json.loads((raw / "dataset.json").read_text())
if meta.get("numTraining") != 131:
    raise SystemExit(f"dataset.json numTraining must be 131, got {meta.get('numTraining')}")

images = sorted((raw / "imagesTr").glob("case_*_0000.nii.gz"))
labels = sorted((raw / "labelsTr").glob("case_*.nii.gz"))
if len(images) != 131 or len(labels) != 131:
    raise SystemExit(f"Expected 131 image/label pairs, got {len(images)}/{len(labels)}")

ids_img = {p.name.replace("_0000.nii.gz", "") for p in images}
ids_lbl = {p.name.replace(".nii.gz", "") for p in labels}
if ids_img != ids_lbl:
    raise SystemExit("Image/label ID mismatch")

import nibabel as nib
import numpy as np

allowed = {0, 1, 2}
for label_path in labels:
    with gzip.open(label_path, "rb") as handle:
        while handle.read(1024 * 1024):
            pass
    bad = set(np.unique(np.asanyarray(nib.load(label_path).dataobj)).astype(int)) - allowed
    if bad:
        raise SystemExit(f"{label_path.name}: unexpected labels {sorted(bad)}")

usage = shutil.disk_usage(raw)
print(f"preflight OK: 131 cases, disk_free_gb={usage.free / (1024 ** 3):.2f}")
PY
}

run_plan() {
  export TRAIN_MODE="plan"
  run_preflight
  check_runtime
  PLAN_ARGS=()
  if [[ "${NNUNET_CLEAN_PREPROCESS:-0}" == "1" ]]; then
    PLAN_ARGS+=(--clean)
  fi
  "${PLAN_BIN}" \
    -d "${DATASET_ID}" \
    --verify_dataset_integrity \
    -c "${CONFIGURATION}" \
    -npfp "${NNUNET_FINGERPRINT_PROCESSES:-4}" \
    -np "${NNUNET_PREPROCESS_PROCESSES:-4}" \
    "${PLAN_ARGS[@]}"
  if [[ ! -f "${PLANS_JSON}" ]]; then
    echo "Missing ${PLANS_JSON}" >&2
    exit 1
  fi
  echo "plan OK: ${PLANS_JSON}"
}

require_plans() {
  if [[ ! -f "${PLANS_JSON}" ]]; then
    echo "Missing ${PLANS_JSON}. Run: bash scripts/3d/train.sh plan" >&2
    exit 1
  fi
}

run_arm_mode() {
  local mode="$1"
  local results_subdir trainer_class
  case "${mode}" in
    baseline) results_subdir="results_3d_baseline"; trainer_class="nnUNetTrainer_500_Baseline" ;;
    anatomical) results_subdir="results_3d_anatomical"; trainer_class="nnUNetTrainer_500_Anatomical" ;;
    boundary-hd) results_subdir="results_3d_boundary_hd"; trainer_class="nnUNetTrainer_500_BoundaryHD" ;;
    topology) results_subdir="results_3d_topology"; trainer_class="nnUNetTrainer_500_Topology" ;;
    *) echo "Unknown arm: ${mode}" >&2; exit 2 ;;
  esac
  export TRAIN_MODE="${mode}"
  require_plans
  check_runtime
  export SKIP_NNUNET_PREPROCESS=1
  bash "${REPO_ROOT}/scripts/3d/train_3d_arm.sh" "${results_subdir}" "${trainer_class}" --skip-preprocess
}

MODE="${1:-}"
case "${MODE}" in
  preflight) run_preflight ;;
  plan) run_plan ;;
  baseline|anatomical|boundary-hd|topology) run_arm_mode "${MODE}" ;;
  -h|--help|help|"") usage; [[ -n "${MODE}" ]] || exit 2 ;;
  *) echo "Unknown mode: ${MODE}" >&2; usage; exit 2 ;;
esac
