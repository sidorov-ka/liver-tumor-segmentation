#!/usr/bin/env bash
# Resolve Python for nnU-Net scripts (local .venv, active venv, or DataSphere kernel).
#
# Priority:
#   1. PYTHON_BIN (explicit)
#   2. REPO_ROOT/.venv/bin/python
#   3. $VIRTUAL_ENV/bin/python
#   4. python3 from PATH
#
# Sets PYTHON_BIN and PLAN_BIN. Sources check that nnunetv2 is importable.
set -euo pipefail

_resolve_python_candidate() {
  local candidate="$1"
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi
  return 1
}

resolve_python_bin() {
  local repo_root="${1:?repo root required}"
  local candidates=()

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=(
    "${repo_root}/.venv/bin/python"
    "${VIRTUAL_ENV:-}/bin/python"
    "$(command -v python3 2>/dev/null || true)"
  )

  local candidate resolved
  for candidate in "${candidates[@]}"; do
    if resolved="$(_resolve_python_candidate "${candidate}" 2>/dev/null)"; then
      PYTHON_BIN="${resolved}"
      return 0
    fi
  done

  cat >&2 <<'EOF'
No usable Python found.

Options:
  1. Create project venv (once):
       python3 -m venv .venv
       .venv/bin/pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
       .venv/bin/pip install -r requirements.txt
  2. Use an existing DataSphere kernel:
       export PYTHON_BIN=/path/to/kernel/python
EOF
  return 1
}

resolve_plan_bin() {
  local python_bin="${PYTHON_BIN:?PYTHON_BIN must be set}"
  local prefix bin_dir plan_bin

  prefix="$("${python_bin}" -c 'import sys; print(sys.prefix)')"
  bin_dir="$(dirname "${python_bin}")"
  plan_bin="${bin_dir}/nnUNetv2_plan_and_preprocess"

  if [[ -x "${plan_bin}" ]]; then
    PLAN_BIN="${plan_bin}"
    return 0
  fi

  plan_bin="$("${python_bin}" -c 'import shutil; print(shutil.which("nnUNetv2_plan_and_preprocess") or "")')"
  if [[ -n "${plan_bin}" && -x "${plan_bin}" ]]; then
    PLAN_BIN="${plan_bin}"
    return 0
  fi

  echo "nnUNetv2_plan_and_preprocess not found for python=${python_bin} (prefix=${prefix})." >&2
  echo "Install deps: pip install -r requirements.txt" >&2
  return 1
}

require_nnunet_python() {
  local repo_root="${1:?repo root required}"

  resolve_python_bin "${repo_root}"
  resolve_plan_bin

  if ! "${PYTHON_BIN}" -c 'import nnunetv2' >/dev/null 2>&1; then
    echo "nnunetv2 is not installed for python=${PYTHON_BIN}" >&2
    echo "Run: ${PYTHON_BIN} -m pip install -r requirements.txt" >&2
    return 1
  fi

  local source="PATH"
  if [[ -n "${PYTHON_BIN_SET:-}" ]]; then
    source="PYTHON_BIN"
  elif [[ "${PYTHON_BIN}" == "${repo_root}/.venv/bin/python" ]]; then
    source="project .venv"
  elif [[ -n "${VIRTUAL_ENV:-}" && "${PYTHON_BIN}" == "${VIRTUAL_ENV}/bin/python" ]]; then
    source="active venv"
  fi

  echo "Using python=${PYTHON_BIN} (${source})"
}
