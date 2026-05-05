#!/bin/bash
# =============================================================================
# Interactive salloc timing benchmark for Qwen-VL optimization.
#
# Runs optimize_qwen_vl_benchmark.py — same model logic as the normal
# salloc runner but strips all W&B / TensorBoard / preview-render / CLIP
# diagnostic / attention-map paths so wall-clock time reflects only the
# model forward/backward passes and optimizer step.
#
# Reports per-iteration time (warm avg/min/max) and total elapsed time.
#
# Usage:
#   bash editing/scripts/run_benchmark_salloc.sh
#   bash editing/scripts/run_benchmark_salloc.sh /path/to/video.mp4 "Edit prompt" "Static prompt" audio
# =============================================================================

set -euo pipefail

# ============================================================
# PROMPTS & VIDEO  — edit these
# ============================================================
SRC_VIDEO="input_videos/a_turtle_resting_on_a_rock.mp4"
EDIT_PROMPT="Turtle extending its neck."
STATIC_PROMPT="A turtle resting on a rock."
NAME_OF_THIS_EXP="benchmark"

# ============================================================
# MODE TO BENCHMARK
# ============================================================
OPT_MODE="both"   # text | audio | both

# ============================================================
# EXPERIMENT SETTINGS  (match your normal run for a fair comparison)
# ============================================================
RETAKE_START_FRAMES=3
QWEN_GRAD_ACCUM_STEPS=1
QWEN_SAMPLE_MODE="linspace"
LPIPS_WEIGHT=0.1
TEMPORAL_WEIGHT=0.05
LR=0.005
QWEN_MAX_FRAMES=8

# ============================================================
# FIXED SETTINGS
# ============================================================
ITERATIONS=15           # keep short for a timing run; increase for full benchmark
EARLY_STOPPING=0        # disable — we want all iters timed
SEED=42
NUM_INFERENCE_STEPS=30
RETAKE_NUM_INFERENCE_STEPS=30
FINAL_RETAKE_NUM_INFERENCE_STEPS=30
GRAD_CLIP=0.0
BEST_MIN_LOSS_DELTA=0.0
AUD_OPT_LAST_STEPS=8
LR_SCHEDULE="cosine"
LATENT_REG_WEIGHT=0.01
TEXT_REG_WEIGHT=0.001
REG_SCHEDULE="cosine_increase"
HEIGHT=320
WIDTH=512
NUM_FRAMES=95
FRAME_RATE=""
QUANTIZATION="fp8-cast"
NEGATIVE_PROMPT="blurry, low quality, artifacts, distorted"
QWEN_IMG_SIZE=224
QWEN_CONTIGUOUS_START_FRAME=4
QWEN_GRADIENT_RUBRIC="motion"
LPIPS_BACKBONE="alex"
MAX_EVAL_FRAMES=95
FRAME_STRIDE=1
LOW_MEMORY_GUIDANCE=1
CFG_SCALE=""
AUDIO_CFG_SCALE=""
A2V_SCALE=""

# ============================================================
# Internal paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-/project/def-amahdavi/amirrz/LTX-2/checkpoints}"
GEMMA_ROOT="${GEMMA_ROOT:-/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized}"
QWEN_ROOT="${QWEN_ROOT:-/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct}"

# Optional positional overrides
[[ $# -ge 1 ]] && SRC_VIDEO="$1"
[[ $# -ge 2 ]] && EDIT_PROMPT="$2"
[[ $# -ge 3 ]] && STATIC_PROMPT="$3"
[[ $# -ge 4 ]] && OPT_MODE="$4"

if [[ "${SRC_VIDEO}" != /* ]]; then
    SRC_VIDEO="${REPO_ROOT}/${SRC_VIDEO}"
fi

if [[ "${OPT_MODE}" == "text" || "${OPT_MODE}" == "both" ]]; then
    ENHANCE_PROMPT=1
else
    ENHANCE_PROMPT=0
fi

QWEN_MOTION_QUESTION="Does this video clearly show the action or state change described by the edit prompt: \"${EDIT_PROMPT}\"? Answer only 'yes' or 'no'."
PROMPT_SLUG=$(echo "${EDIT_PROMPT}" | tr '[:upper:]' '[:lower:]' | tr -s ' ' | cut -d' ' -f1-5 | tr ' ' '_')
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/benchmark/${PROMPT_SLUG}/${NAME_OF_THIS_EXP}}"

echo "========================================================"
echo "  BENCHMARK RUN"
echo "  Source video  : ${SRC_VIDEO}"
echo "  Opt mode      : ${OPT_MODE}"
echo "  Edit prompt   : ${EDIT_PROMPT}"
echo "  Iterations    : ${ITERATIONS}  LR: ${LR}"
echo "  Qwen frames   : ${QWEN_MAX_FRAMES}  img_size: ${QWEN_IMG_SIZE}  sample: ${QWEN_SAMPLE_MODE}"
echo "  Quantization  : ${QUANTIZATION}"
echo "  Output dir    : ${OUTPUT_DIR}"
echo "========================================================"

# ============================================================
# Validation
# ============================================================
if [[ ! -f "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO not found: ${SRC_VIDEO}" >&2
    exit 1
fi

if [[ ! -d "${QWEN_ROOT}" ]]; then
    echo "ERROR: Qwen2.5-VL model not found at ${QWEN_ROOT}" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Run inside an salloc GPU session." >&2
    exit 1
fi

nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: no visible GPU. Run inside an salloc GPU session." >&2
    exit 1
}

# ============================================================
# Environment
# ============================================================
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
    module load opencv cuda/12.9 tensorboard || true
fi

source "${REPO_ROOT}/.venv/bin/activate"

export TORCH_HOME="${TORCH_HOME:-/home/amirrz/.cache/torch}"
export HF_HOME="${HF_HOME:-/home/amirrz/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

# Disable all experiment tracking
export WANDB_DISABLED=1
export WANDB_MODE=disabled

mkdir -p "${OUTPUT_DIR}"

# ============================================================
# Build arguments
# ============================================================
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
    --qwen-grad-accum-steps "${QWEN_GRAD_ACCUM_STEPS}"
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
    --audio-opt-last-steps "${AUD_OPT_LAST_STEPS}"
    --visualize-every-iters 0
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
    --save-final-videos
    --no-clip-similarity-diag
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

# ============================================================
# Run
# ============================================================
python "${REPO_ROOT}/editing/optimize_qwen_vl_benchmark.py" "${ARGS[@]}"

echo "Benchmark finished."
