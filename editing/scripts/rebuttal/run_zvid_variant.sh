#!/bin/bash
# =============================================================================
# Video-latent residual ablation runner — one scenario per call.
#
# Optimizes a residual delta_z added to the source video VAE latent (z_vid),
# updated by the SAME Qwen motion critic as ours, with text+audio FROZEN. Probes
# tuning the video latent directly vs the audio-conditioning pathway. Diffusion
# seed stays 42 so the baseline render matches the other ablation variants.
#
# Usage:
#   bash editing/scripts/rebuttal/run_zvid_variant.sh <config.yaml>
#     <config.yaml> : an editing/configs/rebuttal/<scenario>.yaml
#
# Output:  results/rebuttal/<scenario>/zvid/
#   ├── baseline_video.mp4
#   ├── mode_zvid/best_optimized_video_zvid.mp4  (+ best_delta_z.pt, csv, params)
#   ├── run_config.json
#   └── metrics.json
#
# Env knobs: RESULTS_ROOT, ZVID_REG (L2 on delta_z; default 0.0), RUN_EVAL=0,
#   FORCE=1 (redo), RAFT_WEIGHTS, EVAL_DEVICE, DRY_RUN=1, plus CKPT/QWEN/GEMMA roots.
# Resumable: drops a .completed marker on success; re-running skips done cells.
# =============================================================================
set -euo pipefail

CONFIG_FILE="${1:?Usage: run_zvid_variant.sh <config.yaml>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"

PARSE_SCRIPT="${REPO_ROOT}/editing/scripts/utils/parse_config.py"
MAIN_SCRIPT="${REPO_ROOT}/editing/optimize_zvid_residual.py"
EVAL_SCRIPT="${REPO_ROOT}/editing/scripts/eval_metrics.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE="${FORCE:-0}"
ZVID_REG="${ZVID_REG:-0.0}"

if [[ "${CONFIG_FILE}" != /* ]]; then CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"; fi
[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: config not found: ${CONFIG_FILE}" >&2; exit 1; }

SCENARIO="$(basename "${CONFIG_FILE}" .yaml)"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/rebuttal}"
OUTPUT_DIR="${RESULTS_ROOT}/${SCENARIO}/zvid"
DONE_MARKER="${OUTPUT_DIR}/.completed"
FINAL_VIDEO="${OUTPUT_DIR}/mode_zvid/best_optimized_video_zvid.mp4"

ZVID_ARGS=(--zvid-reg-weight "${ZVID_REG}")
EXTRA=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then read -ra EXTRA <<< "${EXTRA_ARGS}"; fi
ZVID_ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

export OUTPUT_DIR
for v in CKPT_ROOT QWEN_ROOT GEMMA_ROOT; do
    [[ -z "${!v:-}" ]] && { echo "ERROR: ${v} not set. Export it or edit ${SCRIPT_DIR}/env.sh." >&2; exit 1; }
done

PARSE_OUT="$(mktemp)"; trap 'rm -f "${PARSE_OUT}"' EXIT
if ! "${PYTHON_BIN}" "${PARSE_SCRIPT}" "${CONFIG_FILE}" "${REPO_ROOT}" > "${PARSE_OUT}"; then
    echo "ERROR: parse_config.py failed for ${CONFIG_FILE}." >&2; exit 1
fi
BASE_ARGS=(); while IFS= read -r -d '' tok; do BASE_ARGS+=("${tok}"); done < "${PARSE_OUT}"

echo "========================================================"
echo "  Rebuttal video-latent residual (z_vid + delta_z)"
echo "  Scenario : ${SCENARIO}"
echo "  zvid_reg : ${ZVID_REG}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  DRY_RUN  : ${DRY_RUN}"
echo "========================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Command that would run:"
    printf '  %q ' "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${BASE_ARGS[@]}" "${ZVID_ARGS[@]}"; printf '\n'
    exit 0
fi

if [[ "${FORCE}" != "1" && -f "${DONE_MARKER}" ]]; then
    echo "  [skip] already complete: ${DONE_MARKER}  (set FORCE=1 to redo)"; exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no visible GPU. Run inside an salloc GPU session or via sbatch." >&2; exit 1
fi

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda tensorboard 2>/dev/null || true; fi
VENV="${REPO_ROOT}/.venv"; [[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
export WANDB_MODE="${WANDB_MODE:-offline}"; export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"

mkdir -p "${OUTPUT_DIR}"; cd "${REPO_ROOT}"

if [[ "${FORCE}" != "1" && -f "${FINAL_VIDEO}" ]]; then
    echo "---- optimized video already present; skipping optimization (FORCE=1 to redo) ----"
else
    "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${BASE_ARGS[@]}" "${ZVID_ARGS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${FORCE}" != "1" && -f "${OUTPUT_DIR}/metrics.json" ]]; then
        echo "---- metrics.json present; skipping eval ----"
    else
        EVAL_ARGS=(--output-dir "${OUTPUT_DIR}" --mode zvid)
        [[ -n "${RAFT_WEIGHTS:-}" ]] && EVAL_ARGS+=(--raft-weights "${RAFT_WEIGHTS}")
        [[ -n "${EVAL_DEVICE:-}"  ]] && EVAL_ARGS+=(--device "${EVAL_DEVICE}")
        "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}" || \
            echo "WARN: eval_metrics.py failed for ${OUTPUT_DIR} (optimization output still saved)."
    fi
fi

if [[ -f "${FINAL_VIDEO}" ]] && { [[ "${RUN_EVAL}" != "1" ]] || [[ -f "${OUTPUT_DIR}/metrics.json" ]]; }; then
    touch "${DONE_MARKER}"; echo "  marked complete: ${DONE_MARKER}"
else
    echo "  NOT marked complete (missing video or metrics.json) — re-run will resume." >&2
fi
echo "  Done: ${OUTPUT_DIR}"
