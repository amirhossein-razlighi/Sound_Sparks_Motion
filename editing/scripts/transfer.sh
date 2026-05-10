#!/bin/bash
# =============================================================================
# Transfer optimized conditioning latents from one video to a new target video.
#
# Points at a mode_audio / mode_text / mode_both directory produced by a
# previous run of optimize_qwen_vl.py and applies those optimized latents
# to a different target video.
#
# Usage:
#   TARGET_VIDEO=/path/to/target.mp4 \
#   OPT_DIR=/path/to/results/QwenVL/my_prompt/my_exp/mode_both \
#   TRANSFER_MODE=both \
#   EDIT_PROMPT="A cat yawning" \
#   STATIC_PROMPT="A cat sitting still" \
#   bash editing/scripts/transfer.sh
#
# Required environment variables:
#   TARGET_VIDEO   — path to the target video
#   OPT_DIR        — path to a mode_audio / mode_text / mode_both directory
#   CKPT_ROOT      — directory containing ltx-2.3-22b-dev.safetensors
#   QWEN_ROOT      — directory containing Qwen2.5-VL-7B-Instruct weights
#   GEMMA_ROOT     — directory containing Gemma-3-12b text encoder weights
#
# Optional overrides:
#   TRANSFER_MODE  — text | audio | both (default: both)
#   EDIT_PROMPT    — description of the desired edit for the target video
#   STATIC_PROMPT  — before-state description for CLIP diagnostics
#   EXPERIMENT_NAME — output subdirectory label
#   OUTPUT_DIR     — override the auto-generated output path
#   DRY_RUN=1      — print command without running
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TARGET_VIDEO="${TARGET_VIDEO:-${1:-}}"
OPT_DIR="${OPT_DIR:-${2:-}}"
TRANSFER_MODE="${TRANSFER_MODE:-both}"          # text | audio | both

EDIT_PROMPT="${EDIT_PROMPT:-}"
STATIC_PROMPT="${STATIC_PROMPT:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, artifacts, distorted}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-transfer}"

CKPT_ROOT="${CKPT_ROOT:-}"
QWEN_ROOT="${QWEN_ROOT:-}"
GEMMA_ROOT="${GEMMA_ROOT:-}"

PROMPT_SLUG=$(echo "${EDIT_PROMPT}" | tr '[:upper:]' '[:lower:]' | tr -s ' ' | cut -d' ' -f1-5 | tr ' ' '_')
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/transfer/${PROMPT_SLUG}/${EXPERIMENT_NAME}}"

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

QWEN_EVAL_FRAMES="${QWEN_EVAL_FRAMES:-16}"      # frames for Qwen scoring (no grad)

CLIP_DIAG="${CLIP_DIAG:-1}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
CLIP_MAX_FRAMES="${CLIP_MAX_FRAMES:-0}"          # 0 = all frames

WANDB_PROJECT="${WANDB_PROJECT:-sound-sparks-motion}"
WANDB_MODE="${WANDB_MODE:-offline}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAIN_SCRIPT="${REPO_ROOT}/editing/transfer_optimized.py"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${TARGET_VIDEO}" ]]; then
    echo "ERROR: TARGET_VIDEO is not set." >&2
    echo "Usage: TARGET_VIDEO=/path/to/video.mp4 OPT_DIR=/path/to/mode_both bash $0" >&2
    exit 1
fi
if [[ "${TARGET_VIDEO}" != /* ]]; then
    TARGET_VIDEO="${REPO_ROOT}/${TARGET_VIDEO}"
fi
if [[ ! -f "${TARGET_VIDEO}" ]]; then
    echo "ERROR: TARGET_VIDEO not found: ${TARGET_VIDEO}" >&2
    exit 1
fi

if [[ -z "${OPT_DIR}" ]]; then
    echo "ERROR: OPT_DIR is not set (path to mode_audio / mode_text / mode_both dir)." >&2
    exit 1
fi
if [[ ! -d "${OPT_DIR}" ]]; then
    echo "ERROR: OPT_DIR not found: ${OPT_DIR}" >&2
    exit 1
fi

if [[ -z "${CKPT_ROOT}" ]]; then echo "ERROR: CKPT_ROOT is not set." >&2; exit 1; fi
if [[ -z "${QWEN_ROOT}" ]];  then echo "ERROR: QWEN_ROOT is not set."  >&2; exit 1; fi
if [[ -z "${GEMMA_ROOT}" ]]; then echo "ERROR: GEMMA_ROOT is not set." >&2; exit 1; fi

echo "========================================================"
echo "  Transfer optimized latents"
echo "  Target video     : ${TARGET_VIDEO}"
echo "  Opt dir          : ${OPT_DIR}"
echo "  Transfer mode    : ${TRANSFER_MODE}"
echo "  Edit prompt      : ${EDIT_PROMPT}"
echo "  Qwen eval frames : ${QWEN_EVAL_FRAMES}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "  DRY_RUN          : ${DRY_RUN}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARGS=(
    --target-video "${TARGET_VIDEO}"
    --opt-dir "${OPT_DIR}"
    --mode "${TRANSFER_MODE}"
    --negative-prompt "${NEGATIVE_PROMPT}"
    --output-dir "${OUTPUT_DIR}"
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
export WANDB_PROJECT WANDB_MODE
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
printf '%s\n' "${EDIT_PROMPT}" > "${OUTPUT_DIR}/prompt.txt"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${MAIN_SCRIPT}" "${ARGS[@]}"

echo ""
echo "Transfer complete. Outputs: ${OUTPUT_DIR}"
