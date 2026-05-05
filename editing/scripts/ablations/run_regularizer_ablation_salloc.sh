#!/bin/bash
# =============================================================================
# Interactive salloc runner for regularizer ablations under Qwen optimization.
#
# This script is intentionally NOT an sbatch script. Run it from an already
# allocated GPU shell.
#
# Cases:
#   no_reg        : no LPIPS, no temporal, no latent L2, no text L2
#   lpips_only    : LPIPS only
#   temporal_only : temporal consistency only
#   latent_only   : latent L2 only
#   all_reg       : LPIPS + temporal + latent L2
#
# Text L2 is set to 0 in all cases by default to avoid mixing in a fourth
# regularizer when studying the three named in the paper ablation. You can set
# TEXT_REG_FOR_CASES to a nonzero value if you intentionally want text L2 too.
#
# Usage:
#   bash editing/scripts/ablations/run_regularizer_ablation_salloc.sh
#
# Optional positional overrides:
#   bash editing/scripts/ablations/run_regularizer_ablation_salloc.sh \
#     /path/to/video.mp4 \
#     "Edit prompt" \
#     "Static prompt" \
#     "both"
#
# Useful env overrides:
#   REG_CASES="no_reg,lpips_only,temporal_only,latent_only,all_reg" bash ...
# =============================================================================

set -euo pipefail

# ============================================================
# Scenario config: edit these between paper cases
# ============================================================
SRC_VIDEO="${SRC_VIDEO:-/home/amirrz/my_codes/LTX-2/input_videos/a_red_rose_bud_in_a_green.mp4}"
EDIT_PROMPT="${EDIT_PROMPT:-A red rose blooming}"
STATIC_PROMPT="${STATIC_PROMPT:-A red rose bud in a green scene.}"
NAME_OF_THIS_EXP="${NAME_OF_THIS_EXP:-rose_blooming}"
OPT_MODE="${OPT_MODE:-both}"

# Comma-separated subset/order of regularizer cases to run.
REG_CASES="${REG_CASES:-no_reg,lpips_only,temporal_only,latent_only,all_reg}"

# ============================================================
# Shared optimizer/generation settings
# ============================================================
ITERATIONS="${ITERATIONS:-30}"
EARLY_STOPPING="${EARLY_STOPPING:-15}"
VISUALIZE_EVERY_ITERS="${VISUALIZE_EVERY_ITERS:-5}"
SEED="${SEED:-42}"
LR="${LR:-0.005}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
REG_SCHEDULE="${REG_SCHEDULE:-constant}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
BEST_MIN_LOSS_DELTA="${BEST_MIN_LOSS_DELTA:-0.0}"
AUD_OPT_LAST_STEPS="${AUD_OPT_LAST_STEPS:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
RETAKE_NUM_INFERENCE_STEPS="${RETAKE_NUM_INFERENCE_STEPS:-30}"
FINAL_RETAKE_NUM_INFERENCE_STEPS="${FINAL_RETAKE_NUM_INFERENCE_STEPS:-30}"
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-5}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-95}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-95}"
FRAME_RATE="${FRAME_RATE:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, artifacts, distorted}"
QUANTIZATION="${QUANTIZATION:-fp8-cast}"
LOW_MEMORY_GUIDANCE="${LOW_MEMORY_GUIDANCE:-1}"
SAVE_FINAL_VIDEOS="${SAVE_FINAL_VIDEOS:-1}"
RESUME="${RESUME:-0}"
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"

# Weights used in the named cases.
LPIPS_ONLY_WEIGHT="${LPIPS_ONLY_WEIGHT:-0.1}"
TEMPORAL_ONLY_WEIGHT="${TEMPORAL_ONLY_WEIGHT:-0.05}"
LATENT_ONLY_WEIGHT="${LATENT_ONLY_WEIGHT:-0.01}"
TEXT_REG_FOR_CASES="${TEXT_REG_FOR_CASES:-0.0}"
LPIPS_BACKBONE="${LPIPS_BACKBONE:-alex}"

# Qwen scorer settings.
QWEN_MAX_FRAMES="${QWEN_MAX_FRAMES:-8}"
QWEN_IMG_SIZE="${QWEN_IMG_SIZE:-224}"
QWEN_SAMPLE_MODE="${QWEN_SAMPLE_MODE:-linspace}"
QWEN_CONTIGUOUS_START_FRAME="${QWEN_CONTIGUOUS_START_FRAME:-4}"
QWEN_GRADIENT_RUBRIC="${QWEN_GRADIENT_RUBRIC:-motion}"
QWEN_GRAD_ACCUM_STEPS="${QWEN_GRAD_ACCUM_STEPS:-1}"
CLIP_SIMILARITY_DIAG="${CLIP_SIMILARITY_DIAG:-1}"
CLIP_SIMILARITY_DIAG_MODEL="${CLIP_SIMILARITY_DIAG_MODEL:-openai/clip-vit-base-patch32}"
CLIP_SIMILARITY_DIAG_MAX_FRAMES="${CLIP_SIMILARITY_DIAG_MAX_FRAMES:-0}"
CLIP_SIMILARITY_DIAG_BATCH_SIZE="${CLIP_SIMILARITY_DIAG_BATCH_SIZE:-8}"

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-/project/def-amahdavi/amirrz/LTX-2/checkpoints}"
GEMMA_ROOT="${GEMMA_ROOT:-/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized}"
QWEN_ROOT="${QWEN_ROOT:-/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/ablations/regularizers/${NAME_OF_THIS_EXP}}"

[[ $# -ge 1 ]] && SRC_VIDEO="$1"
[[ $# -ge 2 ]] && EDIT_PROMPT="$2"
[[ $# -ge 3 ]] && STATIC_PROMPT="$3"
[[ $# -ge 4 ]] && OPT_MODE="$4"

if [[ "${SRC_VIDEO}" != /* ]]; then
    SRC_VIDEO="${REPO_ROOT}/${SRC_VIDEO}"
fi

slugify() {
    echo "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | tr -cs '[:alnum:]' '_' \
        | sed 's/^_*//; s/_*$//; s/__*/_/g'
}

PROMPT_SLUG="$(slugify "${EDIT_PROMPT}")"
MODE_SLUG="$(slugify "${OPT_MODE}")"
OUTPUT_ROOT="${OUTPUT_ROOT}/${PROMPT_SLUG}/mode_${MODE_SLUG}"
QWEN_MOTION_QUESTION="Does this video clearly show the action or state change described by the edit prompt: \"${EDIT_PROMPT}\"? Answer only 'yes' or 'no'."

if [[ "${OPT_MODE}" == *"text"* || "${OPT_MODE}" == *"both"* ]]; then
    ENHANCE_PROMPT="${ENHANCE_PROMPT:-1}"
else
    ENHANCE_PROMPT="${ENHANCE_PROMPT:-0}"
fi

echo "========================================================"
echo "  Regularizer ablation under Qwen loss"
echo "  Source video   : ${SRC_VIDEO}"
echo "  Edit prompt    : ${EDIT_PROMPT}"
echo "  Static prompt  : ${STATIC_PROMPT}"
echo "  Opt mode       : ${OPT_MODE}"
echo "  Cases          : ${REG_CASES}"
echo "  Output root    : ${OUTPUT_ROOT}"
echo "========================================================"

if [[ ! -f "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO not found: ${SRC_VIDEO}" >&2
    exit 1
fi

if [[ ! -d "${QWEN_ROOT}" ]]; then
    echo "ERROR: Qwen model not found at ${QWEN_ROOT}" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Run this inside an salloc GPU session." >&2
    exit 1
fi

nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: no visible GPU detected. Run this inside an salloc GPU session." >&2
    exit 1
}

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi

if type module >/dev/null 2>&1; then
    module load opencv cuda/12.9 tensorboard || true
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

export TORCH_HOME="${TORCH_HOME:-/home/amirrz/.cache/torch}"
export HF_HOME="${HF_HOME:-/home/amirrz/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export WANDB_PROJECT="${WANDB_PROJECT:-ltx-qwen-opt}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_TAGS="${WANDB_TAGS:-regularizer-ablation}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"

mkdir -p "${OUTPUT_ROOT}"

case_weights() {
    local case_name="$1"
    case "${case_name}" in
        no_reg)
            CASE_LPIPS_WEIGHT="0.0"
            CASE_TEMPORAL_WEIGHT="0.0"
            CASE_LATENT_REG_WEIGHT="0.0"
            ;;
        lpips_only)
            CASE_LPIPS_WEIGHT="${LPIPS_ONLY_WEIGHT}"
            CASE_TEMPORAL_WEIGHT="0.0"
            CASE_LATENT_REG_WEIGHT="0.0"
            ;;
        temporal_only)
            CASE_LPIPS_WEIGHT="0.0"
            CASE_TEMPORAL_WEIGHT="${TEMPORAL_ONLY_WEIGHT}"
            CASE_LATENT_REG_WEIGHT="0.0"
            ;;
        latent_only)
            CASE_LPIPS_WEIGHT="0.0"
            CASE_TEMPORAL_WEIGHT="0.0"
            CASE_LATENT_REG_WEIGHT="${LATENT_ONLY_WEIGHT}"
            ;;
        all_reg)
            CASE_LPIPS_WEIGHT="${LPIPS_ONLY_WEIGHT}"
            CASE_TEMPORAL_WEIGHT="${TEMPORAL_ONLY_WEIGHT}"
            CASE_LATENT_REG_WEIGHT="${LATENT_ONLY_WEIGHT}"
            ;;
        *)
            echo "ERROR: unknown regularizer case '${case_name}'." >&2
            echo "Use: no_reg, lpips_only, temporal_only, latent_only, all_reg" >&2
            exit 1
            ;;
    esac
}

run_regularizer_case() {
    local case_name="$1"
    case_weights "${case_name}"

    local out_dir="${OUTPUT_ROOT}/${case_name}"
    local wandb_dir="${out_dir}/wandb"
    mkdir -p "${out_dir}" "${wandb_dir}" "/tmp/${USER}/wandb_cache"

    printf '%s\n' "${EDIT_PROMPT}" > "${out_dir}/prompt.txt"
    printf '%s\n' "${STATIC_PROMPT}" > "${out_dir}/static_prompt.txt"
    printf '%s\n' "${NEGATIVE_PROMPT}" > "${out_dir}/negative_prompt.txt"
    printf '%s\n' "${QWEN_MOTION_QUESTION}" > "${out_dir}/qwen_motion_question.txt"
    {
        printf 'case=%s\n' "${case_name}"
        printf 'lpips_weight=%s\n' "${CASE_LPIPS_WEIGHT}"
        printf 'temporal_weight=%s\n' "${CASE_TEMPORAL_WEIGHT}"
        printf 'latent_reg_weight=%s\n' "${CASE_LATENT_REG_WEIGHT}"
        printf 'text_reg_weight=%s\n' "${TEXT_REG_FOR_CASES}"
    } > "${out_dir}/regularizer_case.txt"

    export WANDB_DIR="${wandb_dir}"
    export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
    export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/home/${USER}/.config/wandb}"

    local args=(
        --src-video "${SRC_VIDEO}"
        --edit-prompt "${EDIT_PROMPT}"
        --negative-prompt "${NEGATIVE_PROMPT}"
        --output-dir "${out_dir}"
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
        --latent-reg-weight "${CASE_LATENT_REG_WEIGHT}"
        --text-reg-weight "${TEXT_REG_FOR_CASES}"
        --reg-schedule "${REG_SCHEDULE}"
        --lpips-weight "${CASE_LPIPS_WEIGHT}"
        --temporal-weight "${CASE_TEMPORAL_WEIGHT}"
        --lpips-backbone "${LPIPS_BACKBONE}"
        --lr-schedule "${LR_SCHEDULE}"
        --height "${HEIGHT}"
        --width "${WIDTH}"
        --num-frames "${NUM_FRAMES}"
        --quantization "${QUANTIZATION}"
        --gradient-checkpointing
    )

    [[ -n "${FRAME_RATE}" ]] && args+=( --frame-rate "${FRAME_RATE}" )
    [[ -n "${CFG_SCALE}" ]] && args+=( --cfg-scale "${CFG_SCALE}" )
    [[ -n "${AUDIO_CFG_SCALE}" ]] && args+=( --audio-cfg-scale "${AUDIO_CFG_SCALE}" )
    [[ -n "${A2V_SCALE}" ]] && args+=( --a2v-scale "${A2V_SCALE}" )

    if [[ "${ENHANCE_PROMPT}" == "1" ]]; then
        args+=( --enhance-prompt )
    fi

    if [[ "${LOW_MEMORY_GUIDANCE}" == "1" ]]; then
        args+=( --low-memory-guidance )
    else
        args+=( --no-low-memory-guidance )
    fi

    if [[ "${SAVE_FINAL_VIDEOS}" != "1" ]]; then
        args+=( --no-save-final-videos )
    fi

    if [[ "${CLIP_SIMILARITY_DIAG}" == "1" ]]; then
        args+=( --clip-similarity-diag )
    else
        args+=( --no-clip-similarity-diag )
    fi

    if [[ -n "${WANDB_ENTITY}" ]]; then
        args+=( --wandb-entity "${WANDB_ENTITY}" )
    fi

    if [[ "${RESUME}" == "1" ]]; then
        args+=( --resume )
    fi

    echo
    echo ">>> Running regularizer case: ${case_name}"
    echo "    lpips=${CASE_LPIPS_WEIGHT} temporal=${CASE_TEMPORAL_WEIGHT} latent=${CASE_LATENT_REG_WEIGHT} text=${TEXT_REG_FOR_CASES}"
    echo "    out: ${out_dir}"
    python "${REPO_ROOT}/editing/optimize_qwen_vl.py" "${args[@]}"
}

IFS=',' read -ra CASE_LIST <<< "${REG_CASES}"
for case_name in "${CASE_LIST[@]}"; do
    case_name="$(echo "${case_name}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    run_regularizer_case "${case_name}"
done

echo
echo "Finished regularizer ablation. Results under:"
echo "  ${OUTPUT_ROOT}"
