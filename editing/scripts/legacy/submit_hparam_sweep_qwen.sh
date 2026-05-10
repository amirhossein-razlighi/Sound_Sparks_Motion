#!/bin/bash
# =============================================================================
# SLURM array job: hyperparameter sweep for Qwen-loss optimization.
#
# Sweeps the three most impactful hyperparameters on a fixed reference
# experiment ("a_dog_yawning") which achieved our best yes_prob (0.9987).
#
# Grid:
#   LR                  in {0.001, 0.003, 0.005, 0.01}          (4 values)
#   LATENT_REG_WEIGHT   in {0.001, 0.01, 0.1}                   (3 values)
#   AUD_OPT_LAST_STEPS  in {4, 8, 12, 16}                       (4 values)
#   Total: 4 × 3 × 4 = 48 array jobs
#
# Usage:
#   SRC_VIDEO=/path/to/dog_yawning.mp4 sbatch \
#       --array=0-47 \
#       editing/scripts/submit_hparam_sweep_qwen.sh
#
# Or limit concurrency to avoid saturating the cluster:
#   SRC_VIDEO=... sbatch --array=0-47%8 editing/scripts/submit_hparam_sweep_qwen.sh
#
# Results are logged to W&B project ltx-qwen-opt with tag "hparam-sweep".
# Use the W&B Parallel Coordinates panel to find the best config.
# =============================================================================

#SBATCH --job-name=qwen_sweep
#SBATCH --account=your-hpc-account
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=qwen_sweep_%A_%a.out
#SBATCH --error=qwen_sweep_%A_%a.err

set -euo pipefail

nvidia-smi

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
LR_VALUES=(0.001 0.003 0.005 0.01)
REG_VALUES=(0.001 0.01 0.1)
AUD_STEPS_VALUES=(4 8 12 16)

N_LR=${#LR_VALUES[@]}         # 4
N_REG=${#REG_VALUES[@]}       # 3
N_AUD=${#AUD_STEPS_VALUES[@]} # 4

# Decode array task ID → (lr_idx, reg_idx, aud_idx)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
lr_idx=$(( TASK_ID / (N_REG * N_AUD) ))
rem=$(( TASK_ID % (N_REG * N_AUD) ))
reg_idx=$(( rem / N_AUD ))
aud_idx=$(( rem % N_AUD ))

LR="${LR_VALUES[$lr_idx]}"
LATENT_REG_WEIGHT="${REG_VALUES[$reg_idx]}"
AUD_OPT_LAST_STEPS="${AUD_STEPS_VALUES[$aud_idx]}"

# ---------------------------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT}}"
CKPT_ROOT="${CKPT_ROOT:-${CKPT_ROOT}}"
GEMMA_ROOT="${GEMMA_ROOT:-${GEMMA_ROOT}}"
QWEN_ROOT="${QWEN_ROOT:-${QWEN_ROOT}}"

SRC_VIDEO="${SRC_VIDEO:-}"
if [[ -z "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO is not set." >&2
    echo "Usage: SRC_VIDEO=/path/to/video.mp4 sbatch --array=0-47 $0" >&2
    exit 1
fi
if [[ ! -f "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO not found: ${SRC_VIDEO}" >&2
    exit 1
fi

EDIT_PROMPT="${EDIT_PROMPT:-A dog yawning}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, artifacts, distorted}"
OPT_MODE="${OPT_MODE:-both}"     # run both audio+text so we capture all signal

# Build a slug for the sweep run
SWEEP_SLUG="lr${LR}_reg${LATENT_REG_WEIGHT}_aud${AUD_OPT_LAST_STEPS}"
OUTPUT_DIR="${REPO_ROOT}/results/hparam_sweep/${SWEEP_SLUG}"

# Fixed settings — do not sweep these (use memory-safe defaults)
QWEN_MAX_FRAMES="${QWEN_MAX_FRAMES:-8}"
QWEN_IMG_SIZE=224
QWEN_GRADIENT_RUBRIC="${QWEN_GRADIENT_RUBRIC:-motion}"
SEED=42
NUM_INFERENCE_STEPS=30
RETAKE_NUM_INFERENCE_STEPS=30
FINAL_RETAKE_NUM_INFERENCE_STEPS=30
RETAKE_START_FRAMES=5
ITERATIONS=30
GRAD_CLIP=1.0
TEXT_REG_WEIGHT=0.001
EARLY_STOPPING=20
VISUALIZE_EVERY_ITERS=10
MAX_EVAL_FRAMES=95
FRAME_STRIDE=1
HEIGHT=320
WIDTH=512
NUM_FRAMES=95
QUANTIZATION="fp8-cast"
LOW_MEMORY_GUIDANCE=1

echo "========================================================"
echo "  Array task       : ${TASK_ID} / 47"
echo "  Sweep slug       : ${SWEEP_SLUG}"
echo "  LR               : ${LR}"
echo "  LATENT_REG_WEIGHT: ${LATENT_REG_WEIGHT}"
echo "  AUD_OPT_LAST_STEPS: ${AUD_OPT_LAST_STEPS}"
echo "  Edit prompt      : ${EDIT_PROMPT}"
echo "  Opt mode         : ${OPT_MODE}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9 tensorboard
source "${REPO_ROOT}/.venv/bin/activate"

export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

export WANDB_PROJECT="${WANDB_PROJECT:-ltx-qwen-opt}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_TAGS="${WANDB_TAGS:-hparam-sweep,qwen-loss}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/home/${USER}/.config/wandb}"
export WANDB_DISABLE_GIT=true

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
printf '%s\n' "${EDIT_PROMPT}" > "${OUTPUT_DIR}/prompt.txt"
printf '%s\n' "${NEGATIVE_PROMPT}" > "${OUTPUT_DIR}/negative_prompt.txt"

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARGS=(
    --src-video "${SRC_VIDEO}"
    --edit-prompt "${EDIT_PROMPT}"
    --negative-prompt "${NEGATIVE_PROMPT}"
    --output-dir "${OUTPUT_DIR}"
    --opt-mode "${OPT_MODE}"
    --qwen-model "${QWEN_ROOT}"
    --qwen-max-frames "${QWEN_MAX_FRAMES}"
    --qwen-img-size "${QWEN_IMG_SIZE}"
    --qwen-gradient-rubric "${QWEN_GRADIENT_RUBRIC}"
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
    --visualize-every-iters "${VISUALIZE_EVERY_ITERS}"
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
    --low-memory-guidance
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "${REPO_ROOT}/editing/optimize_qwen_vl.py" "${ARGS[@]}"

echo "Done. Exit code: $?"
