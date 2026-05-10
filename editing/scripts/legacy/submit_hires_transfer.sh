#!/bin/bash
# =============================================================================
# SLURM job: high-resolution transfer
#
# Re-runs LTX retake with the optimized latents from a prior
# optimize_qwen_vl.py run, but at the source video's NATIVE resolution
# instead of the downscaled optimization resolution.
#
# Saves under <OPT_DIR>/hires_transfer/ (or OUTPUT_DIR if set):
#   baseline_video.mp4   — retake at native res, unmodified latents
#   transfer_both.mp4    — retake at native res WITH optimized latents
#
# Usage:
#   OPT_DIR=/path/to/results/.../rsf1_accum1_modelinspace_... \
#   SRC_VIDEO=/path/to/source.mp4 \
#   sbatch editing/scripts/submit_hires_transfer.sh
# =============================================================================

#SBATCH --job-name=ltx_hires_transfer
#SBATCH --account=your-hpc-account
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
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT}}"

OPT_DIR="${OPT_DIR:-}"          # top-level exp dir (contains prompt.txt + mode_both/)
SRC_VIDEO="${SRC_VIDEO:-}"      # source video at native resolution (no resize)
TRANSFER_MODE="${TRANSFER_MODE:-both}"
OUTPUT_DIR="${OUTPUT_DIR:-}"    # default: <OPT_DIR>/hires_transfer/
MAX_SIDE="${MAX_SIDE:-}"        # cap longer dim, e.g. MAX_SIDE=1280. empty = full native
NUM_FRAMES="${NUM_FRAMES:-}"    # trim to N frames, e.g. NUM_FRAMES=95. empty = all frames
VAE_TILE_SIZE="${VAE_TILE_SIZE:-256}"              # spatial VAE tile px (default 256; 512 OOMs at 1080p+)
VAE_TEMPORAL_TILE_FRAMES="${VAE_TEMPORAL_TILE_FRAMES:-16}"  # temporal VAE tile frames (default 16; 64 OOMs due to ~30 GB/tile)

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"          # empty => inherit from <OPT_DIR>/run_config.json
RETAKE_NUM_INFERENCE_STEPS="${RETAKE_NUM_INFERENCE_STEPS:-}"
FINAL_RETAKE_NUM_INFERENCE_STEPS="${FINAL_RETAKE_NUM_INFERENCE_STEPS:-}"
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-}"
SEED="${SEED:-}"
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"
QUANTIZATION="${QUANTIZATION:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
GEMMA_ROOT="${GEMMA_ROOT:-}"
ENHANCE_PROMPT="${ENHANCE_PROMPT:-}"                  # 1/true or 0/false
LOW_MEMORY_GUIDANCE="${LOW_MEMORY_GUIDANCE:-}"        # 1/true or 0/false
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-}"  # 1/true or 0/false
CLIP_DIAG="${CLIP_DIAG:-}"                            # 1/true or 0/false

WANDB_PROJECT="${WANDB_PROJECT:-ltx-hires-transfer}"
WANDB_MODE="${WANDB_MODE:-offline}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${OPT_DIR}" ]]; then
    echo "ERROR: OPT_DIR is not set." >&2
    echo "  OPT_DIR=/path/to/results/.../my_exp  SRC_VIDEO=/path/to/video.mp4  sbatch $0" >&2
    exit 1
fi

if [[ -z "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO is not set." >&2
    exit 1
fi

if [[ ! -d "${OPT_DIR}" ]]; then
    echo "ERROR: OPT_DIR not found: ${OPT_DIR}" >&2
    exit 1
fi

if [[ ! -f "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO not found: ${SRC_VIDEO}" >&2
    exit 1
fi

_WANDB_DIR="${OUTPUT_DIR:-${OPT_DIR}/hires_transfer}"

echo "========================================================"
echo "  Job ID       : ${SLURM_JOB_ID:-local}"
echo "  Opt dir      : ${OPT_DIR}"
echo "  Src video    : ${SRC_VIDEO}"
echo "  Transfer mode: ${TRANSFER_MODE}"
echo "  Resolution   : ${MAX_SIDE:+capped to max-side=${MAX_SIDE}}${MAX_SIDE:-native (read from src video)}"
echo "  Output dir   : ${_WANDB_DIR}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9 tensorboard
source "${REPO_ROOT}/.venv/bin/activate"

export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export WANDB_PROJECT
export WANDB_MODE
export WANDB_DIR="${_WANDB_DIR}/wandb"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/home/${USER}/.config/wandb}"
export WANDB_DISABLE_GIT=true

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARGS=(
    --opt-dir              "${OPT_DIR}"
    --src-video            "${SRC_VIDEO}"
    --mode                 "${TRANSFER_MODE}"
)

[[ -n "${OUTPUT_DIR}" ]]     && ARGS+=( --output-dir    "${OUTPUT_DIR}" )
[[ -n "${MAX_SIDE}" ]]       && ARGS+=( --max-side      "${MAX_SIDE}" )
[[ -n "${NUM_FRAMES}" ]]     && ARGS+=( --num-frames    "${NUM_FRAMES}" )
[[ -n "${NUM_INFERENCE_STEPS}" ]]        && ARGS+=( --num-inference-steps "${NUM_INFERENCE_STEPS}" )
[[ -n "${RETAKE_NUM_INFERENCE_STEPS}" ]] && ARGS+=( --retake-num-inference-steps "${RETAKE_NUM_INFERENCE_STEPS}" )
[[ -n "${FINAL_RETAKE_NUM_INFERENCE_STEPS}" ]] && ARGS+=( --final-retake-num-inference-steps "${FINAL_RETAKE_NUM_INFERENCE_STEPS}" )
[[ -n "${RETAKE_START_FRAMES}" ]]        && ARGS+=( --retake-start-frames "${RETAKE_START_FRAMES}" )
[[ -n "${SEED}" ]]                       && ARGS+=( --seed "${SEED}" )
[[ -n "${CFG_SCALE}" ]]                  && ARGS+=( --cfg-scale "${CFG_SCALE}" )
[[ -n "${AUDIO_CFG_SCALE}" ]]            && ARGS+=( --audio-cfg-scale "${AUDIO_CFG_SCALE}" )
[[ -n "${A2V_SCALE}" ]]                  && ARGS+=( --a2v-scale "${A2V_SCALE}" )
[[ -n "${QUANTIZATION}" ]]               && ARGS+=( --quantization "${QUANTIZATION}" )
[[ -n "${CHECKPOINT_PATH}" ]]            && ARGS+=( --checkpoint-path "${CHECKPOINT_PATH}" )
[[ -n "${GEMMA_ROOT}" ]]                 && ARGS+=( --gemma-root "${GEMMA_ROOT}" )
ARGS+=( --vae-tile-size "${VAE_TILE_SIZE}" )
ARGS+=( --vae-temporal-tile-frames "${VAE_TEMPORAL_TILE_FRAMES}" )
[[ -n "${QWEN_MODEL:-}" ]]   && ARGS+=( --qwen-model    "${QWEN_MODEL}" )

case "${ENHANCE_PROMPT,,}" in
    1|true|yes) ARGS+=( --enhance-prompt ) ;;
    0|false|no) ARGS+=( --no-enhance-prompt ) ;;
esac

case "${LOW_MEMORY_GUIDANCE,,}" in
    1|true|yes) ARGS+=( --low-memory-guidance ) ;;
    0|false|no) ARGS+=( --no-low-memory-guidance ) ;;
esac

case "${GRADIENT_CHECKPOINTING,,}" in
    1|true|yes) ARGS+=( --gradient-checkpointing ) ;;
    0|false|no) ARGS+=( --no-gradient-checkpointing ) ;;
esac

case "${CLIP_DIAG,,}" in
    1|true|yes) ARGS+=( --clip-diag ) ;;
    0|false|no) ARGS+=( --no-clip-diag ) ;;
esac

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "${REPO_ROOT}/editing/hires_transfer.py" "${ARGS[@]}"

echo "Finished with exit code $?"
