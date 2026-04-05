#!/bin/bash
# =============================================================================
# SLURM job: multi-modal optimization (text / audio / both) with CLIP loss.
#
# Runs one optimization mode per job. To compare all three, submit three jobs:
#
#   OPT_MODE=text  sbatch editing/scripts/submit_optimize_multimodal.sh /path/to/video.mp4
#   OPT_MODE=audio sbatch editing/scripts/submit_optimize_multimodal.sh /path/to/video.mp4
#   OPT_MODE=both  sbatch editing/scripts/submit_optimize_multimodal.sh /path/to/video.mp4
#
# Or run all three sequentially in one job:
#   OPT_MODE=text,audio,both sbatch editing/scripts/submit_optimize_multimodal.sh /path/to/video.mp4
#
# Required env vars / positional arg:
#   SRC_VIDEO  (or first positional arg)
#   EDIT_PROMPT
# =============================================================================

#SBATCH --job-name=ltx_multimodal_opt
#SBATCH --account=def-amahdavi
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

SRC_VIDEO="${SRC_VIDEO:-${1:-}}"
OPT_MODE="${OPT_MODE:-audio}"                   # text | audio | both | text,audio,both

# EDIT_PROMPT="${EDIT_PROMPT:-A red cars door opens}"
# EDIT_PROMPT="${EDIT_PROMPT:-A bottle of wine drops on the table and shatters into pieces}"
# EDIT_PROMPT="${EDIT_PROMPT:-A dog yawns}"
# EDIT_PROMPT="${EDIT_PROMPT:-A balloon gets loose from the string and flies away into the sky}"
# EDIT_PROMPT="${EDIT_PROMPT:-A dog jumping up and down inplace}"
# EDIT_PROMPT="${EDIT_PROMPT:-A balloon pops}"
EDIT_PROMPT="${EDIT_PROMPT:-All wine bottles drop on the table}"
# EDIT_PROMPT="${EDIT_PROMPT:-A man opens the cars door}"

# First 5 words, lowercased, underscore-joined — keeps dir names short and readable
PROMPT_SLUG=$(echo "${EDIT_PROMPT}" | tr '[:upper:]' '[:lower:]' | tr -s ' ' | cut -d' ' -f1-5 | tr ' ' '_')
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/Xclip/${PROMPT_SLUG}/$(echo "${OPT_MODE}" | tr ',' '_')}"

# CLIP model (needs internet or pre-cached in HF_HOME)
CLIP_MODEL="${CLIP_MODEL:-microsoft/xclip-base-patch32}"
CLIP_MAX_FRAMES="${CLIP_MAX_FRAMES:-30}"

SEED="${SEED:-42}"
# Use fewer steps than the flow-loss pipeline — CLIP loss is cheaper to compute
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
RETAKE_NUM_INFERENCE_STEPS="${RETAKE_NUM_INFERENCE_STEPS:-30}"
FINAL_RETAKE_NUM_INFERENCE_STEPS="${FINAL_RETAKE_NUM_INFERENCE_STEPS:-30}"
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-5}"

ITERATIONS="${ITERATIONS:-50}"
LR="${LR:-0.001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
AUD_OPT_LAST_STEPS="${AUD_OPT_LAST_STEPS:-8}"
EARLY_STOPPING="${EARLY_STOPPING:-10}"

MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-95}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"

LATENT_REG_WEIGHT="${LATENT_REG_WEIGHT:-0.01}"
TEXT_REG_WEIGHT="${TEXT_REG_WEIGHT:-0.001}"

# Shape — conservative for gradient-through-transformer on 80GB
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-95}"
FRAME_RATE="${FRAME_RATE:-}"

# Guidance — low-memory mode avoids duplicating the batch for CFG
LOW_MEMORY_GUIDANCE="${LOW_MEMORY_GUIDANCE:-1}"
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"

# fp8-cast is mandatory on a single 80GB GPU (LTX 22B = ~44GB bf16 → ~22GB fp8)
QUANTIZATION="${QUANTIZATION:-fp8-cast}"

SAVE_FINAL_VIDEOS="${SAVE_FINAL_VIDEOS:-1}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO is not set." >&2
    echo "Usage: sbatch $0 /path/to/source.mp4" >&2
    exit 1
fi

if [[ ! -f "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO not found: ${SRC_VIDEO}" >&2
    exit 1
fi

echo "========================================================"
echo "  Job ID           : ${SLURM_JOB_ID:-local}"
echo "  Source video     : ${SRC_VIDEO}"
echo "  Opt mode         : ${OPT_MODE}"
echo "  Edit prompt      : ${EDIT_PROMPT}"
echo "  CLIP model       : ${CLIP_MODEL}"
echo "  Iterations       : ${ITERATIONS}  LR: ${LR}"
echo "  Resolution       : ${WIDTH}x${HEIGHT}  frames: ${NUM_FRAMES}"
echo "  Quantization     : ${QUANTIZATION}"
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

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARGS=(
    --src-video "${SRC_VIDEO}"
    --edit-prompt "${EDIT_PROMPT}"
    --output-dir "${OUTPUT_DIR}"
    --opt-mode "${OPT_MODE}"
    --clip-model "${CLIP_MODEL}"
    --clip-max-frames "${CLIP_MAX_FRAMES}"
    --checkpoint-path "${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
    --gemma-root "${GEMMA_ROOT}"
    --seed "${SEED}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --retake-num-inference-steps "${RETAKE_NUM_INFERENCE_STEPS}"
    --final-retake-num-inference-steps "${FINAL_RETAKE_NUM_INFERENCE_STEPS}"
    --retake-start-frames "${RETAKE_START_FRAMES}"
    --iterations "${ITERATIONS}"
    --lr "${LR}"
    --grad-clip "${GRAD_CLIP}"
    --audio-opt-last-steps "${AUD_OPT_LAST_STEPS}"
    --early-stopping "${EARLY_STOPPING}"
    --max-eval-frames "${MAX_EVAL_FRAMES}"
    --frame-stride "${FRAME_STRIDE}"
    --latent-reg-weight "${LATENT_REG_WEIGHT}"
    --text-reg-weight "${TEXT_REG_WEIGHT}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num-frames "${NUM_FRAMES}"
    --quantization "${QUANTIZATION}"
    --gradient-checkpointing
)

[[ -n "${FRAME_RATE}" ]]       && ARGS+=( --frame-rate "${FRAME_RATE}" )
[[ -n "${CFG_SCALE}" ]]        && ARGS+=( --cfg-scale "${CFG_SCALE}" )
[[ -n "${AUDIO_CFG_SCALE}" ]]  && ARGS+=( --audio-cfg-scale "${AUDIO_CFG_SCALE}" )
[[ -n "${A2V_SCALE}" ]]        && ARGS+=( --a2v-scale "${A2V_SCALE}" )

if [[ "${LOW_MEMORY_GUIDANCE}" == "1" ]]; then
    ARGS+=( --low-memory-guidance )
else
    ARGS+=( --no-low-memory-guidance )
fi

if [[ "${SAVE_FINAL_VIDEOS}" != "1" ]]; then
    ARGS+=( --no-save-final-videos )
fi

if [[ "${RESUME:-0}" == "1" ]]; then
    ARGS+=( --resume )
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${EDIT_PROMPT}" > "${OUTPUT_DIR}/prompt.txt"

python "${REPO_ROOT}/editing/optimize_multimodal.py" "${ARGS[@]}"

echo "Finished with exit code $?"
