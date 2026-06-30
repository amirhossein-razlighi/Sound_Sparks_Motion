#!/bin/bash
# =============================================================================
# Test-time runtime breakdown for the reviewer's cost question.
#
# Runs editing/profile_runtime.py through the SAME config -> CLI plumbing as
# run.sh (parse_config.py), so it profiles exactly what a real run does:
# backbone inference, VLM scoring, per-iteration optimization cost, hardware.
#
# Usage (inside an salloc GPU shell; this is a real GPU job, NOT a login node):
#   bash editing/scripts/rebuttal/profile_runtime.sh editing/configs/rebuttal/man_pets_dog.yaml
#
# Knobs (env):
#   PROFILE_WARMUP=2   untimed warmup iterations
#   PROFILE_ITERS=5    timed iterations to average
#   OUTPUT_DIR=...     where runtime_breakdown.json lands (default: a profile dir)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PARSE_SCRIPT="${REPO_ROOT}/editing/scripts/utils/parse_config.py"
PROFILE_SCRIPT="${REPO_ROOT}/editing/profile_runtime.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Model roots (CKPT_ROOT / QWEN_ROOT / GEMMA_ROOT).
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"

CONFIG_FILE="${1:?Usage: bash $0 <config.yaml>}"
[[ "${CONFIG_FILE}" != /* ]] && CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"
[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: config not found: ${CONFIG_FILE}" >&2; exit 1; }

# Parse config -> null-delimited args (fail loudly if parse_config errors).
TMP_ARGS="$(mktemp)"
trap 'rm -f "${TMP_ARGS}"' EXIT
if ! "${PYTHON_BIN}" "${PARSE_SCRIPT}" "${CONFIG_FILE}" "${REPO_ROOT}" > "${TMP_ARGS}"; then
    echo "ERROR: parse_config.py failed for ${CONFIG_FILE}" >&2
    exit 1
fi
ARGS=()
while IFS= read -r -d '' tok; do ARGS+=("${tok}"); done < "${TMP_ARGS}"

# Always redirect to a dedicated, config-named profile dir so we NEVER touch a
# real run's outputs. Default: results/rebuttal/_profile/<config-basename>/ ;
# override by exporting OUTPUT_DIR.
CONFIG_BASENAME="$(basename "${CONFIG_FILE}" .yaml)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/rebuttal/_profile/${CONFIG_BASENAME}}"
FOUND_OUT=0
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--output-dir" ]]; then ARGS[$((i + 1))]="${OUTPUT_DIR}"; FOUND_OUT=1; fi
done
if [[ "${FOUND_OUT}" == "0" ]]; then ARGS+=("--output-dir" "${OUTPUT_DIR}"); fi
echo "Profile output dir: ${OUTPUT_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no visible GPU. Run inside an salloc GPU shell." >&2
    exit 1
fi

VENV="${REPO_ROOT}/.venv"
[[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export WANDB_MODE=disabled

cd "${REPO_ROOT}"
PROFILE_WARMUP="${PROFILE_WARMUP:-2}" PROFILE_ITERS="${PROFILE_ITERS:-5}" \
    "${PYTHON_BIN}" "${PROFILE_SCRIPT}" "${ARGS[@]}"
