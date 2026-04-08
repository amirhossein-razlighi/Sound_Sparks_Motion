#!/bin/bash
# =============================================================================
# SLURM job: multi-modal optimization (text / audio / both) with Qwen2.5-VL loss.
#
# Same experiment structure as submit_optimize_multimodal.sh but replaces
# X-CLIP with Qwen2.5-VL (7B) as the video-text alignment scorer.
#
# Resource note (single H100/H200 80GB):
#   LTX-22B fp8  : ~22 GB
#   Qwen2.5-VL-7B bf16 : ~14 GB
#   Activations + optimizer states: ~30 GB
#   Total estimate: ~65 GB — fits in 80 GB with gradient checkpointing.
#
# If you hit OOM, keep QWEN_GRADIENT_RUBRIC=motion, reduce QWEN_MAX_FRAMES,
# or switch to Qwen2.5-VL-3B-Instruct (~6 GB bf16).
#
# Usage (three modes, three separate jobs):
#   OPT_MODE=text  sbatch editing/scripts/submit_optimize_qwen_vl.sh /path/to/video.mp4
#   OPT_MODE=audio sbatch editing/scripts/submit_optimize_qwen_vl.sh /path/to/video.mp4
#   OPT_MODE=both  sbatch editing/scripts/submit_optimize_qwen_vl.sh /path/to/video.mp4
#
# Or all three in one job:
#   OPT_MODE=text,audio,both sbatch editing/scripts/submit_optimize_qwen_vl.sh /path/to/video.mp4
# =============================================================================

#SBATCH --job-name=ltx_qwen_opt
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
QWEN_ROOT="${QWEN_ROOT:-/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct}"

SRC_VIDEO="${SRC_VIDEO:-${1:-}}"
OPT_MODE="${OPT_MODE:-both}"

EDIT_PROMPT="${EDIT_PROMPT:-A red rose blooming.}"
STATIC_PROMPT="${STATIC_PROMPT:-A red rose bud in a green grass field.}"

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, artifacts, distorted}"
ENHANCE_PROMPT="${ENHANCE_PROMPT:-1}"
PROMPT_SLUG=$(echo "${EDIT_PROMPT}" | tr '[:upper:]' '[:lower:]' | tr -s ' ' | cut -d' ' -f1-5 | tr ' ' '_')
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/QwenVL/${PROMPT_SLUG}/$(echo "${OPT_MODE}" | tr ',' '_')}"

# Qwen2.5-VL settings
QWEN_MAX_FRAMES="${QWEN_MAX_FRAMES:-8}"
# Must be divisible by 28. 224 → 64 spatial tokens/chunk. 252 → 81 tokens/chunk.
QWEN_IMG_SIZE="${QWEN_IMG_SIZE:-224}"
QWEN_SAMPLE_MODE="${QWEN_SAMPLE_MODE:-linspace}" # linspace | contiguous | contiguous_random
QWEN_CONTIGUOUS_START_FRAME="${QWEN_CONTIGUOUS_START_FRAME:-4}"
QWEN_GRADIENT_RUBRIC="${QWEN_GRADIENT_RUBRIC:-motion}"
QWEN_MOTION_QUESTION="${QWEN_MOTION_QUESTION:-Does this video clearly show the action or state change described by the edit prompt: \"${EDIT_PROMPT}\"? Answer only 'yes' or 'no'.}"

SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
RETAKE_NUM_INFERENCE_STEPS="${RETAKE_NUM_INFERENCE_STEPS:-30}"
FINAL_RETAKE_NUM_INFERENCE_STEPS="${FINAL_RETAKE_NUM_INFERENCE_STEPS:-30}"
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-10}"

ITERATIONS="${ITERATIONS:-30}"
LR="${LR:-0.005}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
BEST_MIN_LOSS_DELTA="${BEST_MIN_LOSS_DELTA:-0.0}"
CLIP_SIMILARITY_DIAG_MODEL="${CLIP_SIMILARITY_DIAG_MODEL:-openai/clip-vit-base-patch32}"
CLIP_SIMILARITY_DIAG="${CLIP_SIMILARITY_DIAG:-1}"
CLIP_SIMILARITY_DIAG_MAX_FRAMES="${CLIP_SIMILARITY_DIAG_MAX_FRAMES:-0}"
CLIP_SIMILARITY_DIAG_BATCH_SIZE="${CLIP_SIMILARITY_DIAG_BATCH_SIZE:-8}"
AUD_OPT_LAST_STEPS="${AUD_OPT_LAST_STEPS:-8}"
EARLY_STOPPING="${EARLY_STOPPING:-15}"
VISUALIZE_EVERY_ITERS="${VISUALIZE_EVERY_ITERS:-5}"

MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-95}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"

LATENT_REG_WEIGHT="${LATENT_REG_WEIGHT:-0.01}"
TEXT_REG_WEIGHT="${TEXT_REG_WEIGHT:-0.001}"
REG_SCHEDULE="${REG_SCHEDULE:-cosine_increase}"   # constant | linear_warmup | cosine_increase

# Perceptual quality preservation (anti-adversarial)
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.1}"               # 0.0 = disabled; 0.1-0.5 recommended for hard edits
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.05}"         # 0.0 = disabled; 0.05-0.2 for temporal smoothness
LPIPS_BACKBONE="${LPIPS_BACKBONE:-alex}"           # alex (~30MB) | vgg (~60MB)
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"              # constant | cosine

HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-95}"
FRAME_RATE="${FRAME_RATE:-}"

LOW_MEMORY_GUIDANCE="${LOW_MEMORY_GUIDANCE:-1}"
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"

QUANTIZATION="${QUANTIZATION:-fp8-cast}"
SAVE_FINAL_VIDEOS="${SAVE_FINAL_VIDEOS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ltx-qwen-opt}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-qwen-loss,cluster-offline}"
WANDB_MODE="${WANDB_MODE:-offline}"

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

if [[ ! -d "${QWEN_ROOT}" ]]; then
    echo "ERROR: Qwen2.5-VL model not found at ${QWEN_ROOT}" >&2
    echo "Run the download script on the login node first:" >&2
    echo "  bash ${REPO_ROOT}/editing/scripts/download_qwen_vl.sh" >&2
    exit 1
fi

echo "========================================================"
echo "  Job ID           : ${SLURM_JOB_ID:-local}"
echo "  Source video     : ${SRC_VIDEO}"
echo "  Opt mode         : ${OPT_MODE}"
echo "  Edit prompt      : ${EDIT_PROMPT}"
echo "  Static prompt    : ${STATIC_PROMPT}"
echo "  Enhance prompt   : ${ENHANCE_PROMPT}"
echo "  Qwen model       : ${QWEN_ROOT}"
echo "  Qwen frames      : ${QWEN_MAX_FRAMES}  img_size: ${QWEN_IMG_SIZE}  sample: ${QWEN_SAMPLE_MODE}@${QWEN_CONTIGUOUS_START_FRAME}  rubric: ${QWEN_GRADIENT_RUBRIC}"
echo "  Qwen motion Q    : ${QWEN_MOTION_QUESTION}"
echo "  Iterations       : ${ITERATIONS}  LR: ${LR}  LR_sched: ${LR_SCHEDULE}"
echo "  Perceptual       : lpips=${LPIPS_WEIGHT}  temporal=${TEMPORAL_WEIGHT}  backbone=${LPIPS_BACKBONE}  reg_sched=${REG_SCHEDULE}"
echo "  Best min delta   : ${BEST_MIN_LOSS_DELTA}"
echo "  CLIP diag        : ${CLIP_SIMILARITY_DIAG}  model: ${CLIP_SIMILARITY_DIAG_MODEL}  max_frames: ${CLIP_SIMILARITY_DIAG_MAX_FRAMES}  batch: ${CLIP_SIMILARITY_DIAG_BATCH_SIZE}"
echo "  Visualize every  : ${VISUALIZE_EVERY_ITERS}"
echo "  W&B project      : ${WANDB_PROJECT}"
echo "  W&B mode         : ${WANDB_MODE}"
echo "  Resolution       : ${WIDTH}x${HEIGHT}  frames: ${NUM_FRAMES}"
echo "  Quantization     : ${QUANTIZATION}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9 tensorboard
source "${REPO_ROOT}/.venv/bin/activate"

export TORCH_HOME="${TORCH_HOME:-/home/amirrz/.cache/torch}"
export HF_HOME="${HF_HOME:-/home/amirrz/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_TAGS
export WANDB_MODE
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/home/${USER}/.config/wandb}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

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
    --qwen-sample-mode "${QWEN_SAMPLE_MODE}"
    --qwen-contiguous-start-frame "${QWEN_CONTIGUOUS_START_FRAME}"
    --qwen-gradient-rubric "${QWEN_GRADIENT_RUBRIC}"
    --qwen-motion-question "${QWEN_MOTION_QUESTION}"
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
    --best-min-loss-delta "${BEST_MIN_LOSS_DELTA}"
    --static-prompt "${STATIC_PROMPT}"
    --clip-similarity-diag-model "${CLIP_SIMILARITY_DIAG_MODEL}"
    --clip-similarity-diag-max-frames "${CLIP_SIMILARITY_DIAG_MAX_FRAMES}"
    --clip-similarity-diag-batch-size "${CLIP_SIMILARITY_DIAG_BATCH_SIZE}"
    --audio-opt-last-steps "${AUD_OPT_LAST_STEPS}"
    --visualize-every-iters "${VISUALIZE_EVERY_ITERS}"
    --early-stopping "${EARLY_STOPPING}"
    --max-eval-frames "${MAX_EVAL_FRAMES}"
    --frame-stride "${FRAME_STRIDE}"
    --latent-reg-weight "${LATENT_REG_WEIGHT}"
    --text-reg-weight "${TEXT_REG_WEIGHT}"
    --reg-schedule "${REG_SCHEDULE}"
    --lpips-weight "${LPIPS_WEIGHT}"
    --temporal-weight "${TEMPORAL_WEIGHT}"
    --lpips-backbone "${LPIPS_BACKBONE}"
    --lr-schedule "${LR_SCHEDULE}"
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
if [[ "${ENHANCE_PROMPT}" == "1" ]]; then
    ARGS+=( --enhance-prompt )
fi

if [[ "${LOW_MEMORY_GUIDANCE}" == "1" ]]; then
    ARGS+=( --low-memory-guidance )
else
    ARGS+=( --no-low-memory-guidance )
fi

if [[ "${SAVE_FINAL_VIDEOS}" != "1" ]]; then
    ARGS+=( --no-save-final-videos )
fi

if [[ "${CLIP_SIMILARITY_DIAG}" == "1" ]]; then
    ARGS+=( --clip-similarity-diag )
else
    ARGS+=( --no-clip-similarity-diag )
fi

if [[ "${RESUME:-0}" == "1" ]]; then
    ARGS+=( --resume )
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${EDIT_PROMPT}" > "${OUTPUT_DIR}/prompt.txt"
printf '%s\n' "${STATIC_PROMPT}" > "${OUTPUT_DIR}/static_prompt.txt"
printf '%s\n' "${NEGATIVE_PROMPT}" > "${OUTPUT_DIR}/negative_prompt.txt"
printf '%s\n' "${QWEN_MOTION_QUESTION}" > "${OUTPUT_DIR}/qwen_motion_question.txt"

python "${REPO_ROOT}/editing/optimize_qwen_vl.py" "${ARGS[@]}"

echo "Finished with exit code $?"
