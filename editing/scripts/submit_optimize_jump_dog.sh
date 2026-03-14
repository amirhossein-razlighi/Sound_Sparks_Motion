#!/bin/bash
# =============================================================================
# SLURM job: optimize audio latent for the jumping-dog experiment.
#
# Default profile is intentionally conservative for 2xH100 allocations because
# the current 2-rank FSDP init path can OOM during parameter flatten/sharding
# before optimization begins.
#
# Usage:
#   sbatch editing/scripts/submit_optimize_jump_dog.sh /path/to/source.mp4
#
# Optional overrides via environment variables before sbatch:
#   export SRC_VIDEO=/path/to/source.mp4
#   export OUTPUT_DIR=/path/to/output_dir
#   export EDIT_PROMPT="A dog in the scene"
#   export TARGET_PROMPT="The dog jumps energetically"
#   export RAFT_WEIGHTS_PATH=/path/to/raft_large_C_T_SKHT_V2-ff5fadd5.pth
# =============================================================================

#SBATCH --job-name=ltx_opt_jump_dog
#SBATCH --account=def-amahdavi
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

SRC_VIDEO="${SRC_VIDEO:-${1:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/editing_results_jump_optimize}"

EDIT_PROMPT="${EDIT_PROMPT:-A dog in the scene}"
TARGET_PROMPT="${TARGET_PROMPT:-The dog jumps up and down}"

SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-16}"
TI2V_NUM_INFERENCE_STEPS="${TI2V_NUM_INFERENCE_STEPS:-16}"
RETAKE_NUM_INFERENCE_STEPS="${RETAKE_NUM_INFERENCE_STEPS:-20}"
FINAL_RETAKE_NUM_INFERENCE_STEPS="${FINAL_RETAKE_NUM_INFERENCE_STEPS:-20}"
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-5}"

ITERATIONS="${ITERATIONS:-6}"
LR="${LR:-0.03}"
AUD_OPT_LAST_STEPS="${AUD_OPT_LAST_STEPS:-6}"
FINAL_AUD_OPT_LAST_STEPS="${FINAL_AUD_OPT_LAST_STEPS:-0}"
FLOW_WEIGHT="${FLOW_WEIGHT:-1.0}"
MAG_CURVE_WEIGHT="${MAG_CURVE_WEIGHT:-0.25}"
LATENT_REG_WEIGHT="${LATENT_REG_WEIGHT:-0.02}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-9}"
FRAME_STRIDE="${FRAME_STRIDE:-3}"
FLOW_WIDTH="${FLOW_WIDTH:-224}"
FLOW_HEIGHT="${FLOW_HEIGHT:-128}"

# Keep this file available on shared storage or $HOME so compute nodes can read it.
RAFT_MODEL="${RAFT_MODEL:-raft_small}"
RAFT_WEIGHTS_PATH="${RAFT_WEIGHTS_PATH:-}"

# Optional shape overrides (leave empty to auto-detect from source video)
HEIGHT="${HEIGHT:-224}"
WIDTH="${WIDTH:-224}"
NUM_FRAMES="${NUM_FRAMES:-129}"
FRAME_RATE="${FRAME_RATE:-}"

# Optional guidance overrides (leave empty for auto-detect)
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"

# Optional: fp8-cast | fp8-scaled-mm | (empty = none)
# Keep TI2V unquantized by default to avoid fp8 load-time OOM spikes.
QUANTIZATION="${QUANTIZATION:-}"
TI2V_QUANTIZATION="${TI2V_QUANTIZATION:-}"
RETAKE_QUANTIZATION="${RETAKE_QUANTIZATION:-}"

# Multi-GPU gradient mode (FSDP sharded transformer)
# Default disabled for 2xH100 because current FSDP init OOMs before training.
MULTI_GPU="${MULTI_GPU:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

# Final render is enabled by default so optimized/baseline videos are saved.
SAVE_FINAL_VIDEOS="${SAVE_FINAL_VIDEOS:-1}"

# Infer visible GPUs from SLURM/CUDA mask when possible.
VISIBLE_GPU_COUNT=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    VISIBLE_GPU_COUNT="${#_VISIBLE_GPU_ARRAY[@]}"
else
    VISIBLE_GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

if [[ -z "${NPROC_PER_NODE}" ]]; then
    NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
fi

# FSDP init can transiently peak above steady-state memory; default to fp8
# Retake weights in multi-GPU mode unless explicitly overridden.
if [[ "${MULTI_GPU}" == "1" ]] && [[ -z "${RETAKE_QUANTIZATION}" ]]; then
    RETAKE_QUANTIZATION="fp8-cast"
fi

if [[ "${MULTI_GPU}" == "1" ]] && [[ "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
    echo "ERROR: NPROC_PER_NODE=${NPROC_PER_NODE} but only ${VISIBLE_GPU_COUNT} GPUs are visible." >&2
    echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    echo "       Set NPROC_PER_NODE<=${VISIBLE_GPU_COUNT} or request more GPUs in Slurm." >&2
    exit 1
fi

# Optional target video (if set, TARGET_PROMPT is ignored)
TARGET_VIDEO="${TARGET_VIDEO:-/home/amirrz/my_codes/LTX-2/results/editing_results_jump_optimize/target_motion_video.mp4}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${SRC_VIDEO}" ]]; then
    echo "ERROR: Source video not set." >&2
    echo "Usage: sbatch editing/scripts/submit_optimize_jump_dog.sh /path/to/source.mp4" >&2
    exit 1
fi

if [[ -z "${RAFT_WEIGHTS_PATH}" ]]; then
    if [[ "${RAFT_MODEL}" == "raft_small" ]]; then
        RAFT_WEIGHTS_PATH="/home/amirrz/.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth"
    else
        RAFT_WEIGHTS_PATH="/home/amirrz/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
    fi
fi

if [[ ! -f "${RAFT_WEIGHTS_PATH}" ]]; then
    echo "ERROR: RAFT checkpoint not found: ${RAFT_WEIGHTS_PATH}" >&2
    echo "Set RAFT_WEIGHTS_PATH to a local .pth file available on compute nodes." >&2
    exit 1
fi

if [[ -z "${TARGET_VIDEO}" ]]; then
    echo "WARNING: TARGET_VIDEO is not set; job will generate target video via TI2V,"
    echo "         which can increase peak VRAM and trigger OOM on 80GB GPUs." 
    echo "         Prefer passing TARGET_VIDEO=... to skip TI2V generation." 
fi

echo "========================================================"
echo "  Job ID              : ${SLURM_JOB_ID}"
echo "  Source video        : ${SRC_VIDEO}"
echo "  Edit prompt         : ${EDIT_PROMPT}"
if [[ -n "${TARGET_VIDEO}" ]]; then
    echo "  Target video        : ${TARGET_VIDEO}"
else
    echo "  Target prompt       : ${TARGET_PROMPT}"
fi
echo "  RAFT model          : ${RAFT_MODEL}"
echo "  RAFT weights path   : ${RAFT_WEIGHTS_PATH}"
echo "  Iterations          : ${ITERATIONS}"
echo "  LR                  : ${LR}"
echo "  Audio opt last steps: ${AUD_OPT_LAST_STEPS}"
if [[ -n "${FINAL_AUD_OPT_LAST_STEPS}" ]]; then
    echo "  Final audio last st.: ${FINAL_AUD_OPT_LAST_STEPS}"
fi
echo "  TI2V steps          : ${TI2V_NUM_INFERENCE_STEPS}"
echo "  Retake steps        : ${RETAKE_NUM_INFERENCE_STEPS}"
echo "  Final retake steps  : ${FINAL_RETAKE_NUM_INFERENCE_STEPS}"
echo "  Output dir          : ${OUTPUT_DIR}"
echo "  Multi GPU           : ${MULTI_GPU}"
echo "  Save final videos   : ${SAVE_FINAL_VIDEOS}"
echo "  NPROC per node      : ${NPROC_PER_NODE}"
echo "  Visible GPUs        : ${VISIBLE_GPU_COUNT}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9
source "${REPO_ROOT}/.venv/bin/activate"

# Keep torch cache explicit for reproducible offline loading.
export TORCH_HOME="${TORCH_HOME:-/home/amirrz/.cache/torch}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

# ---------------------------------------------------------------------------
# Build optional arguments
# ---------------------------------------------------------------------------
SHAPE_ARGS=""
[[ -n "${HEIGHT}" ]]     && SHAPE_ARGS+=" --height ${HEIGHT}"
[[ -n "${WIDTH}" ]]      && SHAPE_ARGS+=" --width ${WIDTH}"
[[ -n "${NUM_FRAMES}" ]] && SHAPE_ARGS+=" --num-frames ${NUM_FRAMES}"
[[ -n "${FRAME_RATE}" ]] && SHAPE_ARGS+=" --frame-rate ${FRAME_RATE}"

GUIDE_ARGS=""
[[ -n "${CFG_SCALE}" ]]       && GUIDE_ARGS+=" --cfg-scale ${CFG_SCALE}"
[[ -n "${AUDIO_CFG_SCALE}" ]] && GUIDE_ARGS+=" --audio-cfg-scale ${AUDIO_CFG_SCALE}"
[[ -n "${A2V_SCALE}" ]]       && GUIDE_ARGS+=" --a2v-scale ${A2V_SCALE}"

QUANT_ARG=""
[[ -n "${QUANTIZATION}" ]] && QUANT_ARG="--quantization ${QUANTIZATION}"

STAGE_QUANT_ARGS=""
[[ -n "${TI2V_QUANTIZATION}" ]] && STAGE_QUANT_ARGS+=" --ti2v-quantization ${TI2V_QUANTIZATION}"
[[ -n "${RETAKE_QUANTIZATION}" ]] && STAGE_QUANT_ARGS+=" --retake-quantization ${RETAKE_QUANTIZATION}"

TARGET_ARG_NAME=""
TARGET_ARG_VALUE=""
if [[ -n "${TARGET_VIDEO}" ]]; then
    TARGET_ARG_NAME="--target-video"
    TARGET_ARG_VALUE="${TARGET_VIDEO}"
else
    TARGET_ARG_NAME="--target-prompt"
    TARGET_ARG_VALUE="${TARGET_PROMPT}"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --src-video "${SRC_VIDEO}"
    --edit-prompt "${EDIT_PROMPT}"
    "${TARGET_ARG_NAME}" "${TARGET_ARG_VALUE}"
    --output-dir "${OUTPUT_DIR}"
    --checkpoint-path "${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
    --gemma-root "${GEMMA_ROOT}"
    --seed "${SEED}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --ti2v-num-inference-steps "${TI2V_NUM_INFERENCE_STEPS}"
    --retake-num-inference-steps "${RETAKE_NUM_INFERENCE_STEPS}"
    --final-retake-num-inference-steps "${FINAL_RETAKE_NUM_INFERENCE_STEPS}"
    --retake-start-frames "${RETAKE_START_FRAMES}"
    --iterations "${ITERATIONS}"
    --lr "${LR}"
    --audio-opt-last-steps "${AUD_OPT_LAST_STEPS}"
    --flow-weight "${FLOW_WEIGHT}"
    --mag-curve-weight "${MAG_CURVE_WEIGHT}"
    --latent-reg-weight "${LATENT_REG_WEIGHT}"
    --max-eval-frames "${MAX_EVAL_FRAMES}"
    --frame-stride "${FRAME_STRIDE}"
    --flow-width "${FLOW_WIDTH}"
    --flow-height "${FLOW_HEIGHT}"
    --raft-model "${RAFT_MODEL}"
    --raft-weights-path "${RAFT_WEIGHTS_PATH}"
)

if [[ -n "${FINAL_AUD_OPT_LAST_STEPS}" ]]; then
    COMMON_ARGS+=( --final-audio-opt-last-steps "${FINAL_AUD_OPT_LAST_STEPS}" )
fi

if [[ -n "${SHAPE_ARGS}" ]]; then
    # shellcheck disable=SC2206
    COMMON_ARGS+=( ${SHAPE_ARGS} )
fi
if [[ -n "${GUIDE_ARGS}" ]]; then
    # shellcheck disable=SC2206
    COMMON_ARGS+=( ${GUIDE_ARGS} )
fi
if [[ -n "${STAGE_QUANT_ARGS}" ]]; then
    # shellcheck disable=SC2206
    COMMON_ARGS+=( ${STAGE_QUANT_ARGS} )
fi
if [[ -n "${QUANT_ARG}" ]]; then
    # shellcheck disable=SC2206
    COMMON_ARGS+=( ${QUANT_ARG} )
fi

if [[ "${SAVE_FINAL_VIDEOS}" != "1" ]]; then
    COMMON_ARGS+=( --no-save-final-videos )
fi

if [[ "${MULTI_GPU}" == "1" ]]; then
    torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" \
        "${REPO_ROOT}/editing/optimize_audio_embedding.py" \
        --distributed-shard-transformer \
        "${COMMON_ARGS[@]}"
else
    python "${REPO_ROOT}/editing/optimize_audio_embedding.py" "${COMMON_ARGS[@]}"
fi

echo "Finished with exit code $?"
