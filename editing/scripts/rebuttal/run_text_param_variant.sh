#!/bin/bash
# =============================================================================
# Text-parameterization ablation runner — residual (OURS) vs direct.
#
# Answers the reviewer's question: "why parameterize text as a residual but
# audio directly?" Runs the SAME method (same config, same audio pathway, same
# Qwen critic, losses, LR, seed) and changes ONLY how the text embedding is
# parameterized:
#
#   VARIANT=residual     OURS: optimize a delta added to the base prompt
#                        embedding; reg = ||delta||^2 (anchors toward the prompt).
#   VARIANT=direct       optimize the FULL text embedding directly (init = base
#                        prompt embedding), anchored to base with L2. Provably
#                        equivalent to residual — a sanity/equivalence control.
#   VARIANT=direct_free  optimize the FULL embedding with NO anchor (text reg
#                        dropped) — the genuinely different alternative; shows
#                        whether un-anchored direct optimization drifts into
#                        adversarial embedding regions.
#
# Everything else (opt_mode, audio init, enhance_prompt, iterations, …) is taken
# verbatim from the config, so only the text parameterization differs.
#
# Usage (inside an salloc GPU shell or via sbatch):
#   VARIANT=direct bash editing/scripts/rebuttal/run_text_param_variant.sh <config.yaml>
#   # scenario config, e.g. editing/configs/rebuttal/man_pets_dog.yaml
#
# Output:  results/rebuttal/<scenario>/textparam_<variant>/
#   ├── baseline_video.mp4
#   ├── mode_<opt_mode>/best_optimized_video_<opt_mode>.mp4  (+ latents, csv)
#   ├── run_config.json   (records text_param / text_reg_anchor)
#   └── metrics.json
#
# Env knobs: VARIANT (residual|direct|direct_free; default direct), RESULTS_ROOT,
#   RUN_EVAL=0, FORCE=1, RAFT_WEIGHTS, EVAL_DEVICE, DRY_RUN=1, CKPT/QWEN/GEMMA roots.
# Resumable: drops a .completed marker on success; re-running skips done cells.
# =============================================================================
set -euo pipefail

CONFIG_FILE="${1:?Usage: [VARIANT=direct] run_text_param_variant.sh <config.yaml>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"

PARSE_SCRIPT="${REPO_ROOT}/editing/scripts/utils/parse_config.py"
MAIN_SCRIPT="${REPO_ROOT}/editing/optimize_qwen_vl.py"
EVAL_SCRIPT="${REPO_ROOT}/editing/scripts/eval_metrics.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE="${FORCE:-0}"
VARIANT="${VARIANT:-direct}"

# Map variant -> text-parameterization flags.
case "${VARIANT}" in
    residual)    TEXT_ARGS=(--text-param residual) ;;
    direct)      TEXT_ARGS=(--text-param direct --text-reg-anchor base) ;;
    direct_free) TEXT_ARGS=(--text-param direct --text-reg-anchor none) ;;
    *) echo "ERROR: unknown VARIANT '${VARIANT}'. Use residual|direct|direct_free." >&2; exit 1 ;;
esac

if [[ "${CONFIG_FILE}" != /* ]]; then CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"; fi
[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: config not found: ${CONFIG_FILE}" >&2; exit 1; }

SCENARIO="$(basename "${CONFIG_FILE}" .yaml)"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/rebuttal}"
OUTPUT_DIR="${RESULTS_ROOT}/${SCENARIO}/textparam_${VARIANT}"
DONE_MARKER="${OUTPUT_DIR}/.completed"

for v in CKPT_ROOT QWEN_ROOT GEMMA_ROOT; do
    [[ -z "${!v:-}" ]] && { echo "ERROR: ${v} not set. Export it or edit ${SCRIPT_DIR}/env.sh." >&2; exit 1; }
done

PARSE_OUT="$(mktemp)"; trap 'rm -f "${PARSE_OUT}"' EXIT
if ! "${PYTHON_BIN}" "${PARSE_SCRIPT}" "${CONFIG_FILE}" "${REPO_ROOT}" > "${PARSE_OUT}"; then
    echo "ERROR: parse_config.py failed for ${CONFIG_FILE}." >&2; exit 1
fi
BASE_ARGS=(); while IFS= read -r -d '' tok; do BASE_ARGS+=("${tok}"); done < "${PARSE_OUT}"

# Determine opt_mode (for the final-video path) from the parsed args; default both.
MODE="both"
for ((i = 0; i < ${#BASE_ARGS[@]}; i++)); do
    if [[ "${BASE_ARGS[$i]}" == "--opt-mode" ]]; then MODE="${BASE_ARGS[$((i + 1))]}"; fi
done
MODE="${MODE%%,*}"  # first mode if a list
FINAL_VIDEO="${OUTPUT_DIR}/mode_${MODE}/best_optimized_video_${MODE}.mp4"

# Override output dir (argparse takes the last --output-dir).
RUN_ARGS=("${BASE_ARGS[@]}" "${TEXT_ARGS[@]}" --output-dir "${OUTPUT_DIR}")

echo "========================================================"
echo "  Rebuttal text-parameterization ablation"
echo "  Scenario : ${SCENARIO}"
echo "  Variant  : ${VARIANT}   (${TEXT_ARGS[*]})"
echo "  opt_mode : ${MODE}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  DRY_RUN  : ${DRY_RUN}"
echo "========================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Command that would run:"
    printf '  %q ' "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${RUN_ARGS[@]}"; printf '\n'
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
    "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${RUN_ARGS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${FORCE}" != "1" && -f "${OUTPUT_DIR}/metrics.json" ]]; then
        echo "---- metrics.json present; skipping eval ----"
    else
        EVAL_ARGS=(--output-dir "${OUTPUT_DIR}" --mode "${MODE}")
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
