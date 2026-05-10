#!/bin/bash
# =============================================================================
# Run a single multimodal ablation experiment interactively.
#
# Equivalent functionality to run.sh but with all parameters spelled out
# directly (no YAML config) for fine-grained control over ablation conditions.
#
# Optimization mode choices:
#   text          — optimize text conditioning only
#   audio         — optimize audio conditioning only
#   both          — optimize both jointly (main method)
#   text,audio,both — run all three modes sequentially
#
# Usage:
#   bash editing/scripts/ablations/run_ablation.sh
#
# Or with positional overrides:
#   bash editing/scripts/ablations/run_ablation.sh \
#     /path/to/video.mp4 "Edit prompt" "Static prompt" both
#
# Environment overrides:
#   CKPT_ROOT, QWEN_ROOT, GEMMA_ROOT — model paths (required)
#   OUTPUT_DIR                        — override output directory
#   Any of the settings variables below
# =============================================================================

set -euo pipefail

# ============================================================
# PROMPTS & VIDEO — edit these
# ============================================================
SRC_VIDEO="${SRC_VIDEO:-input_videos/a_turtle_resting_on_a_rock.mp4}"
EDIT_PROMPT="${EDIT_PROMPT:-Turtle extending its neck.}"
STATIC_PROMPT="${STATIC_PROMPT:-A turtle resting on a rock.}"
NAME_OF_THIS_EXP="${NAME_OF_THIS_EXP:-ablation}"

# ============================================================
# MODE
# ============================================================
OPT_MODE="${OPT_MODE:-both}"   # text | audio | both | text,audio,both

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-3}"
QWEN_GRAD_ACCUM_STEPS="${QWEN_GRAD_ACCUM_STEPS:-1}"
QWEN_SAMPLE_MODE="${QWEN_SAMPLE_MODE:-linspace}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.1}"
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-0.05}"
LR="${LR:-0.005}"
QWEN_MAX_FRAMES="${QWEN_MAX_FRAMES:-8}"

# ============================================================
# FIXED SETTINGS
# ============================================================
ITERATIONS="${ITERATIONS:-30}"
EARLY_STOPPING="${EARLY_STOPPING:-15}"
VISUALIZE_EVERY_ITERS="${VISUALIZE_EVERY_ITERS:-5}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
RETAKE_NUM_INFERENCE_STEPS="${RETAKE_NUM_INFERENCE_STEPS:-30}"
FINAL_RETAKE_NUM_INFERENCE_STEPS="${FINAL_RETAKE_NUM_INFERENCE_STEPS:-30}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
BEST_MIN_LOSS_DELTA="${BEST_MIN_LOSS_DELTA:-0.0}"
AUD_OPT_LAST_STEPS="${AUD_OPT_LAST_STEPS:-8}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LATENT_REG_WEIGHT="${LATENT_REG_WEIGHT:-0.01}"
TEXT_REG_WEIGHT="${TEXT_REG_WEIGHT:-0.001}"
REG_SCHEDULE="${REG_SCHEDULE:-cosine_increase}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-95}"
FRAME_RATE="${FRAME_RATE:-}"
QUANTIZATION="${QUANTIZATION:-fp8-cast}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, artifacts, distorted}"
QWEN_IMG_SIZE="${QWEN_IMG_SIZE:-224}"
QWEN_CONTIGUOUS_START_FRAME="${QWEN_CONTIGUOUS_START_FRAME:-4}"
QWEN_GRADIENT_RUBRIC="${QWEN_GRADIENT_RUBRIC:-motion}"
LPIPS_BACKBONE="${LPIPS_BACKBONE:-alex}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-95}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
LOW_MEMORY_GUIDANCE="${LOW_MEMORY_GUIDANCE:-1}"
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"
SAVE_FINAL_VIDEOS="${SAVE_FINAL_VIDEOS:-1}"
RESUME="${RESUME:-0}"
CLIP_SIMILARITY_DIAG_MODEL="${CLIP_SIMILARITY_DIAG_MODEL:-openai/clip-vit-base-patch32}"
CLIP_SIMILARITY_DIAG="${CLIP_SIMILARITY_DIAG:-1}"
CLIP_SIMILARITY_DIAG_MAX_FRAMES="${CLIP_SIMILARITY_DIAG_MAX_FRAMES:-0}"
CLIP_SIMILARITY_DIAG_BATCH_SIZE="${CLIP_SIMILARITY_DIAG_BATCH_SIZE:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-sound-sparks-motion}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-qwen-loss}"
WANDB_MODE="${WANDB_MODE:-offline}"

# ============================================================
# Internal paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-}"
GEMMA_ROOT="${GEMMA_ROOT:-}"
QWEN_ROOT="${QWEN_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/QwenVL/${PROMPT_SLUG}/${NAME_OF_THIS_EXP}}"

# ============================================================
# Validation
# ============================================================
if [[ -z "${CKPT_ROOT}" ]]; then echo "ERROR: CKPT_ROOT not set." >&2; exit 1; fi
if [[ -z "${QWEN_ROOT}"  ]]; then echo "ERROR: QWEN_ROOT not set."  >&2; exit 1; fi
if [[ -z "${GEMMA_ROOT}" ]]; then echo "ERROR: GEMMA_ROOT not set." >&2; exit 1; fi

if [[ ! -f "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO not found: ${SRC_VIDEO}" >&2; exit 1
fi
if [[ ! -d "${QWEN_ROOT}" ]]; then
    echo "ERROR: Qwen model not found at ${QWEN_ROOT}" >&2
    echo "Run: bash editing/scripts/setup/download_qwen.sh" >&2; exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no visible GPU detected." >&2; exit 1
fi

echo "========================================================"
echo "  Source video     : ${SRC_VIDEO}"
echo "  Opt mode         : ${OPT_MODE}"
echo "  Edit prompt      : ${EDIT_PROMPT}"
echo "  Qwen frames      : ${QWEN_MAX_FRAMES}  sample: ${QWEN_SAMPLE_MODE}  accum: ${QWEN_GRAD_ACCUM_STEPS}"
echo "  Iterations       : ${ITERATIONS}  LR: ${LR}"
echo "  Perceptual       : lpips=${LPIPS_WEIGHT}  temporal=${TEMPORAL_WEIGHT}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "========================================================"

# ============================================================
# Environment
# ============================================================
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
    module load opencv cuda tensorboard 2>/dev/null || true
fi

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export WANDB_PROJECT WANDB_ENTITY WANDB_TAGS WANDB_MODE
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_DISABLE_GIT=true

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

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
[[ -n "${WANDB_ENTITY}" ]]     && ARGS+=( --wandb-entity "${WANDB_ENTITY}" )

[[ "${ENHANCE_PROMPT}"       == "1" ]] && ARGS+=( --enhance-prompt )
[[ "${SAVE_FINAL_VIDEOS}"    != "1" ]] && ARGS+=( --no-save-final-videos )
[[ "${RESUME}"               == "1" ]] && ARGS+=( --resume )

if [[ "${LOW_MEMORY_GUIDANCE}" == "1" ]]; then
    ARGS+=( --low-memory-guidance )
else
    ARGS+=( --no-low-memory-guidance )
fi

if [[ "${CLIP_SIMILARITY_DIAG}" == "1" ]]; then
    ARGS+=( --clip-similarity-diag )
else
    ARGS+=( --no-clip-similarity-diag )
fi

printf '%s\n' "${EDIT_PROMPT}"         > "${OUTPUT_DIR}/prompt.txt"
printf '%s\n' "${STATIC_PROMPT}"       > "${OUTPUT_DIR}/static_prompt.txt"
printf '%s\n' "${NEGATIVE_PROMPT}"     > "${OUTPUT_DIR}/negative_prompt.txt"
printf '%s\n' "${QWEN_MOTION_QUESTION}" > "${OUTPUT_DIR}/qwen_motion_question.txt"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${REPO_ROOT}/editing/optimize_qwen_vl.py" "${ARGS[@]}"

echo "Finished."
