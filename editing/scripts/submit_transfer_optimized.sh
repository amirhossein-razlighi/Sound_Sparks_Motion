#!/bin/bash
# =============================================================================
# SLURM job: transfer optimized latents from one video to a new target video.
#
# Usage:
#   TARGET_VIDEO=/path/to/cat.mp4 \
#   OPT_DIR=/path/to/results/Xclip/a_dog_yawns/mode_audio \
#   TRANSFER_MODE=audio \
#   EDIT_PROMPT="A cat yawning" \
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
# Settings
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/home/amirrz/my_codes/LTX-2}"
CKPT_ROOT="${CKPT_ROOT:-/project/def-amahdavi/amirrz/LTX-2/checkpoints}"
GEMMA_ROOT="${GEMMA_ROOT:-/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized}"

TARGET_VIDEO="${TARGET_VIDEO:-${1:-}}"
OPT_DIR="${OPT_DIR:-${2:-}}"
TRANSFER_MODE="${TRANSFER_MODE:-audio}"         # text | audio | both
EDIT_PROMPT="${EDIT_PROMPT:-A cat yawning}"

PROMPT_SLUG=$(echo "${EDIT_PROMPT}" | tr '[:upper:]' '[:lower:]' | tr -s ' ' | cut -d' ' -f1-5 | tr ' ' '_')
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/transfer/${PROMPT_SLUG}/${TRANSFER_MODE}}"

SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"

HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-95}"
FRAME_RATE="${FRAME_RATE:-}"

QUANTIZATION="${QUANTIZATION:-fp8-cast}"
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${TARGET_VIDEO}" ]]; then
    echo "ERROR: TARGET_VIDEO is not set." >&2
    echo "Usage: TARGET_VIDEO=/path/to/video.mp4 OPT_DIR=/path/to/opt_dir sbatch $0" >&2
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

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${EDIT_PROMPT}" > "${OUTPUT_DIR}/prompt.txt"

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARGS=(
    --target-video "${TARGET_VIDEO}"
    --opt-dir "${OPT_DIR}"
    --mode "${TRANSFER_MODE}"
    --edit-prompt "${EDIT_PROMPT}"
    --output-dir "${OUTPUT_DIR}"
    --checkpoint-path "${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
    --gemma-root "${GEMMA_ROOT}"
    --seed "${SEED}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
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

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "${REPO_ROOT}/editing/transfer_optimized.py" "${ARGS[@]}"

echo "Finished with exit code $?"
