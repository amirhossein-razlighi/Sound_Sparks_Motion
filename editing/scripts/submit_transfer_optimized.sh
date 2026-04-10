#!/bin/bash
# =============================================================================
# SLURM job: transfer optimized latents from one video to a new target video.
#
# Points at a mode_audio / mode_text / mode_both directory produced by
# optimize_qwen_vl.py and applies those latents to a different target video.
#
# Usage:
#   TARGET_VIDEO=/path/to/cat.mp4 \
#   OPT_DIR=/path/to/results/QwenVL/a_dog_yawning/both/mode_both \
#   TRANSFER_MODE=both \
#   EDIT_PROMPT="A cat yawning" \
#   STATIC_PROMPT="A cat sitting still" \
#   sbatch editing/scripts/submit_transfer_optimized.sh
# =============================================================================

#SBATCH --job-name=ltx_transfer
#SBATCH --account=def-amahdavi
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

nvidia-smi

# ---------------------------------------------------------------------------
# Settings — override with environment variables before sbatch
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/home/amirrz/my_codes/LTX-2}"
CKPT_ROOT="${CKPT_ROOT:-/project/def-amahdavi/amirrz/LTX-2/checkpoints}"
GEMMA_ROOT="${GEMMA_ROOT:-/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized}"
QWEN_ROOT="${QWEN_ROOT:-/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct}"

TARGET_VIDEO="${TARGET_VIDEO:-${1:-}}"
OPT_DIR="${OPT_DIR:-${2:-}}"
TRANSFER_MODE="${TRANSFER_MODE:-both}"        # text | audio | both
EDIT_PROMPT="${EDIT_PROMPT:-A man shouting.}"
STATIC_PROMPT="${STATIC_PROMPT:-A man is standing.}"            # before-state description for CLIP diagnostics
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, artifacts, distorted}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-normal_w_lpips/}"

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

# Qwen evaluation of transfer result
QWEN_EVAL_FRAMES="${QWEN_EVAL_FRAMES:-16}"    # frames for inference-only Qwen scoring (no grad)

# CLIP diagnostics
CLIP_DIAG="${CLIP_DIAG:-1}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
CLIP_MAX_FRAMES="${CLIP_MAX_FRAMES:-0}"        # 0 = all frames

# W&B
WANDB_PROJECT="${WANDB_PROJECT:-ltx-transfer}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${TARGET_VIDEO}" ]]; then
    echo "ERROR: TARGET_VIDEO is not set." >&2
    echo "Usage: TARGET_VIDEO=/path/to/video.mp4 OPT_DIR=/path/to/mode_both sbatch $0" >&2
    exit 1
fi
if [[ ! -f "${TARGET_VIDEO}" ]]; then
    echo "ERROR: TARGET_VIDEO not found: ${TARGET_VIDEO}" >&2
    exit 1
fi
if [[ -z "${OPT_DIR}" ]]; then
    echo "ERROR: OPT_DIR is not set (path to the mode_audio/mode_text/mode_both dir)." >&2
    exit 1
fi
if [[ ! -d "${OPT_DIR}" ]]; then
    echo "ERROR: OPT_DIR not found: ${OPT_DIR}" >&2
    exit 1
fi

echo "========================================================"
echo "  Job ID           : ${SLURM_JOB_ID:-local}"
echo "  Target video     : ${TARGET_VIDEO}"
echo "  Opt dir          : ${OPT_DIR}"
echo "  Transfer mode    : ${TRANSFER_MODE}"
echo "  Edit prompt      : ${EDIT_PROMPT}"
echo "  Static prompt    : ${STATIC_PROMPT}"
echo "  Qwen eval frames : ${QWEN_EVAL_FRAMES}"
echo "  CLIP diag        : ${CLIP_DIAG}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9
source "${REPO_ROOT}/.venv/bin/activate"

export TORCH_HOME="${TORCH_HOME:-/home/amirrz/.cache/torch}"
export HF_HOME="${HF_HOME:-/home/amirrz/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_MODE
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/home/${USER}/.config/wandb}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
printf '%s\n' "${EDIT_PROMPT}" > "${OUTPUT_DIR}/prompt.txt"

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARGS=(
    --target-video "${TARGET_VIDEO}"
    --opt-dir "${OPT_DIR}"
    --mode "${TRANSFER_MODE}"
    --edit-prompt "${EDIT_PROMPT}"
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

[[ -n "${FRAME_RATE}" ]]       && ARGS+=( --frame-rate "${FRAME_RATE}" )
[[ -n "${CFG_SCALE}" ]]        && ARGS+=( --cfg-scale "${CFG_SCALE}" )
[[ -n "${AUDIO_CFG_SCALE}" ]]  && ARGS+=( --audio-cfg-scale "${AUDIO_CFG_SCALE}" )
[[ -n "${A2V_SCALE}" ]]        && ARGS+=( --a2v-scale "${A2V_SCALE}" )
[[ -n "${STATIC_PROMPT}" ]]    && ARGS+=( --static-prompt "${STATIC_PROMPT}" )

if [[ "${ENHANCE_PROMPT}" == "1" ]]; then
    ARGS+=( --enhance-prompt )
fi

if [[ "${CLIP_DIAG}" == "1" ]]; then
    ARGS+=( --clip-diag )
else
    ARGS+=( --no-clip-diag )
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "${REPO_ROOT}/editing/transfer_optimized.py" "${ARGS[@]}"

echo "Finished with exit code $?"
