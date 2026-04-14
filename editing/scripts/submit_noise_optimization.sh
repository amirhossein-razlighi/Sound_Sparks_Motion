#!/bin/bash
# =============================================================================
# SLURM job: noise + latent optimisation with Qwen2.5-VL alignment loss.
#
# Extends standard text/audio latent optimisation with an additional
# **video_delta** parameter — a learned perturbation in the source video's
# VAE latent space that steers the denoising trajectory.
#
# Usage:
#   SRC_VIDEO=/path/to/video.mp4 \
#   EDIT_PROMPT="The rabbit raises its paw" \
#   sbatch editing/scripts/submit_noise_optimization.sh
# =============================================================================

#SBATCH --job-name=ltx_noise_opt
#SBATCH --account=def-amahdavi
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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

SRC_VIDEO="${SRC_VIDEO:-}"
EDIT_PROMPT="${EDIT_PROMPT:-}"
STATIC_PROMPT="${STATIC_PROMPT:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Optimisation mode: both_vd = text + audio + video_delta (recommended)
# Other options: vd, audio_vd, text_vd, all
OPT_MODE="${OPT_MODE:-both_vd}"

# --- Hyperparameters ---
ITERATIONS="${ITERATIONS:-30}"
LR="${LR:-0.005}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
AUDIO_OPT_LAST_STEPS="${AUDIO_OPT_LAST_STEPS:-6}"
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
QWEN_MAX_FRAMES="${QWEN_MAX_FRAMES:-8}"
QWEN_SAMPLE_MODE="${QWEN_SAMPLE_MODE:-linspace}"
QWEN_GRAD_ACCUM_STEPS="${QWEN_GRAD_ACCUM_STEPS:-1}"

# --- Video delta specific ---
VIDEO_DELTA_REG_WEIGHT="${VIDEO_DELTA_REG_WEIGHT:-0.1}"
VD_LOW_FREQ_WEIGHT="${VD_LOW_FREQ_WEIGHT:-3.0}"

# --- Perceptual quality ---
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.0}"
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.0}"

# --- Regularisation ---
LATENT_REG_WEIGHT="${LATENT_REG_WEIGHT:-0.01}"
TEXT_REG_WEIGHT="${TEXT_REG_WEIGHT:-0.001}"

QUANTIZATION="${QUANTIZATION:-fp8-cast}"
SEED="${SEED:-42}"

WANDB_PROJECT="${WANDB_PROJECT:-ltx-noise-opt}"
WANDB_MODE="${WANDB_MODE:-offline}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO is not set." >&2
    exit 1
fi
if [[ -z "${EDIT_PROMPT}" ]]; then
    echo "ERROR: EDIT_PROMPT is not set." >&2
    exit 1
fi

# Auto output dir
if [[ -z "${OUTPUT_DIR}" ]]; then
    _slug=$(echo "${EDIT_PROMPT}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/__*/_/g' | head -c 60)
    OUTPUT_DIR="${REPO_ROOT}/results/NoiseOpt/${_slug}/rsf${RETAKE_START_FRAMES}_lr${LR}_vdreg${VIDEO_DELTA_REG_WEIGHT}_mode_${OPT_MODE}"
fi

echo "========================================================"
echo "  Job ID         : ${SLURM_JOB_ID:-local}"
echo "  Src video      : ${SRC_VIDEO}"
echo "  Edit prompt    : ${EDIT_PROMPT}"
echo "  Opt mode       : ${OPT_MODE}"
echo "  Iterations     : ${ITERATIONS}"
echo "  LR             : ${LR}"
echo "  VD reg weight  : ${VIDEO_DELTA_REG_WEIGHT}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9 tensorboard
source "${REPO_ROOT}/.venv/bin/activate"

export TORCH_HOME="${TORCH_HOME:-/home/amirrz/.cache/torch}"
export HF_HOME="${HF_HOME:-/home/amirrz/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export WANDB_PROJECT
export WANDB_MODE
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/home/${USER}/.config/wandb}"
export WANDB_DISABLE_GIT=true

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

# ---------------------------------------------------------------------------
# Save prompts
# ---------------------------------------------------------------------------
echo "${EDIT_PROMPT}" > "${OUTPUT_DIR}/prompt.txt"
[[ -n "${STATIC_PROMPT}" ]] && echo "${STATIC_PROMPT}" > "${OUTPUT_DIR}/static_prompt.txt"
[[ -n "${NEGATIVE_PROMPT}" ]] && echo "${NEGATIVE_PROMPT}" > "${OUTPUT_DIR}/negative_prompt.txt"

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARGS=(
    --src-video             "${SRC_VIDEO}"
    --edit-prompt           "${EDIT_PROMPT}"
    --output-dir            "${OUTPUT_DIR}"
    --opt-mode              "${OPT_MODE}"

    --iterations            "${ITERATIONS}"
    --lr                    "${LR}"
    --grad-clip             "${GRAD_CLIP}"
    --audio-opt-last-steps  "${AUDIO_OPT_LAST_STEPS}"
    --retake-start-frames   "${RETAKE_START_FRAMES}"
    --num-inference-steps   "${NUM_INFERENCE_STEPS}"
    --seed                  "${SEED}"

    --qwen-model            "${QWEN_ROOT}"
    --qwen-max-frames       "${QWEN_MAX_FRAMES}"
    --qwen-sample-mode      "${QWEN_SAMPLE_MODE}"
    --qwen-grad-accum-steps "${QWEN_GRAD_ACCUM_STEPS}"

    --video-delta-reg-weight "${VIDEO_DELTA_REG_WEIGHT}"
    --vd-low-freq-weight    "${VD_LOW_FREQ_WEIGHT}"

    --latent-reg-weight     "${LATENT_REG_WEIGHT}"
    --text-reg-weight       "${TEXT_REG_WEIGHT}"
    --lpips-weight          "${LPIPS_WEIGHT}"
    --temporal-weight       "${TEMPORAL_WEIGHT}"

    --quantization          "${QUANTIZATION}"
    --gradient-checkpointing
    --checkpoint-path       "${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
    --gemma-root            "${GEMMA_ROOT}"
)

[[ -n "${STATIC_PROMPT}" ]]   && ARGS+=( --static-prompt   "${STATIC_PROMPT}" )
[[ -n "${NEGATIVE_PROMPT}" ]] && ARGS+=( --negative-prompt  "${NEGATIVE_PROMPT}" )

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "${REPO_ROOT}/editing/optimize_noise_qwen_vl.py" "${ARGS[@]}"

echo "Finished with exit code $?"
