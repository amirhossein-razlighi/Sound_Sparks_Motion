#!/bin/bash
# =============================================================================
# Apply optimized conditioning latents from one video to a new target video.
#
# Two usage modes:
#
#   1. Config file (recommended):
#      bash editing/scripts/transfer.sh editing/configs/transfer/yours.yaml
#
#   2. Environment variables (legacy):
#      TARGET_VIDEO=/path/to/target.mp4 \
#      OPT_DIR=/path/to/results/QwenVL/my_exp/mode_both \
#      bash editing/scripts/transfer.sh
#
# Required env vars (both modes):
#   CKPT_ROOT   — directory containing ltx-2.3-22b-dev.safetensors
#   QWEN_ROOT   — directory containing Qwen2.5-VL-7B-Instruct weights
#   GEMMA_ROOT  — directory containing Gemma-3-12b text encoder weights
#
# Optional:
#   DRY_RUN=1   — print command without running (no GPU needed)
#   OUTPUT_DIR  — override the auto-generated output path
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAIN_SCRIPT="${REPO_ROOT}/editing/transfer_optimized.py"
PARSE_SCRIPT="${SCRIPT_DIR}/utils/parse_transfer_config.py"

# ---------------------------------------------------------------------------
# Detect mode: YAML config vs legacy env vars
# ---------------------------------------------------------------------------
CONFIG_FILE="${1:-}"

if [[ -n "${CONFIG_FILE}" && "${CONFIG_FILE}" != -* ]]; then
    # ── YAML CONFIG MODE ──────────────────────────────────────────────────────
    if [[ "${CONFIG_FILE}" != /* ]]; then
        CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"
    fi
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
        exit 1
    fi

    ARGS=()
    while IFS= read -r -d '' tok; do
        ARGS+=("${tok}")
    done < <("${PYTHON_BIN}" "${PARSE_SCRIPT}" "${CONFIG_FILE}" "${REPO_ROOT}")

    # Extract output dir for display / W&B dir
    OUTPUT_DIR_SELECTED=""
    for ((i = 0; i < ${#ARGS[@]}; i++)); do
        if [[ "${ARGS[$i]}" == "--output-dir" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
            OUTPUT_DIR_SELECTED="${ARGS[$((i + 1))]}"
            break
        fi
    done

    echo "========================================================"
    echo "  Transfer optimized latents"
    echo "  Config     : ${CONFIG_FILE}"
    echo "  Output dir : ${OUTPUT_DIR_SELECTED}"
    echo "  DRY_RUN    : ${DRY_RUN}"
    echo "========================================================"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo ""
        echo "Command that would run:"
        printf '  %q ' "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${ARGS[@]}"
        printf '\n'
        exit 0
    fi

else
    # ── LEGACY ENV-VAR MODE ───────────────────────────────────────────────────
    TARGET_VIDEO="${TARGET_VIDEO:-}"
    OPT_DIR="${OPT_DIR:-}"
    TRANSFER_MODE="${TRANSFER_MODE:-both}"

    EDIT_PROMPT="${EDIT_PROMPT:-}"
    STATIC_PROMPT="${STATIC_PROMPT:-}"
    NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, artifacts, distorted}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-transfer}"

    CKPT_ROOT="${CKPT_ROOT:-}"
    QWEN_ROOT="${QWEN_ROOT:-}"
    GEMMA_ROOT="${GEMMA_ROOT:-}"

    SEED="${SEED:-42}"
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
    RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-5}"
    ENHANCE_PROMPT="${ENHANCE_PROMPT:-1}"

    HEIGHT="${HEIGHT:-320}"
    WIDTH="${WIDTH:-512}"
    NUM_FRAMES="${NUM_FRAMES:-95}"
    FRAME_RATE="${FRAME_RATE:-}"
    QUANTIZATION="${QUANTIZATION:-fp8-cast}"

    CFG_SCALE="${CFG_SCALE:-}"
    AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
    A2V_SCALE="${A2V_SCALE:-}"

    QWEN_EVAL_FRAMES="${QWEN_EVAL_FRAMES:-16}"

    CLIP_DIAG="${CLIP_DIAG:-1}"
    CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
    CLIP_MAX_FRAMES="${CLIP_MAX_FRAMES:-0}"

    if [[ -z "${TARGET_VIDEO}" ]]; then
        echo "ERROR: provide a YAML config or set TARGET_VIDEO." >&2
        echo "Usage: bash $0 editing/configs/transfer/yours.yaml" >&2
        echo "       TARGET_VIDEO=... OPT_DIR=... bash $0" >&2
        exit 1
    fi
    if [[ "${TARGET_VIDEO}" != /* ]]; then TARGET_VIDEO="${REPO_ROOT}/${TARGET_VIDEO}"; fi
    if [[ ! -f "${TARGET_VIDEO}" ]]; then
        echo "ERROR: TARGET_VIDEO not found: ${TARGET_VIDEO}" >&2; exit 1
    fi
    if [[ -z "${OPT_DIR}" ]]; then
        echo "ERROR: OPT_DIR is not set." >&2; exit 1
    fi
    if [[ ! -d "${OPT_DIR}" ]]; then
        echo "ERROR: OPT_DIR not found: ${OPT_DIR}" >&2; exit 1
    fi
    if [[ -z "${CKPT_ROOT}" ]]; then echo "ERROR: CKPT_ROOT is not set." >&2; exit 1; fi
    if [[ -z "${QWEN_ROOT}" ]];  then echo "ERROR: QWEN_ROOT is not set."  >&2; exit 1; fi
    if [[ -z "${GEMMA_ROOT}" ]]; then echo "ERROR: GEMMA_ROOT is not set." >&2; exit 1; fi

    PROMPT_SLUG=$(echo "${EDIT_PROMPT}" | tr '[:upper:]' '[:lower:]' | tr -s ' ' | cut -d' ' -f1-5 | tr ' ' '_')
    OUTPUT_DIR_SELECTED="${OUTPUT_DIR:-${REPO_ROOT}/results/transfer/${PROMPT_SLUG}/${EXPERIMENT_NAME}}"

    echo "========================================================"
    echo "  Transfer optimized latents (env-var mode)"
    echo "  Target video : ${TARGET_VIDEO}"
    echo "  Opt dir      : ${OPT_DIR}"
    echo "  Mode         : ${TRANSFER_MODE}"
    echo "  Edit prompt  : ${EDIT_PROMPT}"
    echo "  Output dir   : ${OUTPUT_DIR_SELECTED}"
    echo "  DRY_RUN      : ${DRY_RUN}"
    echo "========================================================"

    ARGS=(
        --target-video "${TARGET_VIDEO}"
        --opt-dir "${OPT_DIR}"
        --mode "${TRANSFER_MODE}"
        --negative-prompt "${NEGATIVE_PROMPT}"
        --output-dir "${OUTPUT_DIR_SELECTED}"
        --checkpoint-path "${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
        --gemma-root "${GEMMA_ROOT}"
        --qwen-model "${QWEN_ROOT}"
        --qwen-eval-frames "${QWEN_EVAL_FRAMES}"
        --clip-model "${CLIP_MODEL}"
        --clip-max-frames "${CLIP_MAX_FRAMES}"
        --seed "${SEED}"
        --num-inference-steps "${NUM_INFERENCE_STEPS}"
        --retake-start-frames "${RETAKE_START_FRAMES}"
        --height "${HEIGHT}"
        --width "${WIDTH}"
        --num-frames "${NUM_FRAMES}"
        --quantization "${QUANTIZATION}"
        --no-gradient-checkpointing
        --no-low-memory-guidance
    )

    [[ -n "${FRAME_RATE}" ]]      && ARGS+=( --frame-rate "${FRAME_RATE}" )
    [[ -n "${CFG_SCALE}" ]]       && ARGS+=( --cfg-scale "${CFG_SCALE}" )
    [[ -n "${AUDIO_CFG_SCALE}" ]] && ARGS+=( --audio-cfg-scale "${AUDIO_CFG_SCALE}" )
    [[ -n "${A2V_SCALE}" ]]       && ARGS+=( --a2v-scale "${A2V_SCALE}" )
    [[ -n "${EDIT_PROMPT}" ]]     && ARGS+=( --edit-prompt "${EDIT_PROMPT}" )
    [[ -n "${STATIC_PROMPT}" ]]   && ARGS+=( --static-prompt "${STATIC_PROMPT}" )
    [[ "${ENHANCE_PROMPT}" == "1" ]] && ARGS+=( --enhance-prompt )
    if [[ "${CLIP_DIAG}" == "1" ]]; then
        ARGS+=( --clip-diag )
    else
        ARGS+=( --no-clip-diag )
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'Command: %q ' "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${ARGS[@]}"
        printf '\n'
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# GPU check + environment
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no visible GPU detected." >&2
    exit 1
fi

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
    module load opencv cuda 2>/dev/null || true
fi

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

OUTPUT_DIR_SELECTED="${OUTPUT_DIR_SELECTED:-}"
if [[ -n "${OUTPUT_DIR_SELECTED}" ]]; then
    mkdir -p "${OUTPUT_DIR_SELECTED}"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${MAIN_SCRIPT}" "${ARGS[@]}"

echo ""
echo "Transfer complete. Outputs: ${OUTPUT_DIR_SELECTED}"
