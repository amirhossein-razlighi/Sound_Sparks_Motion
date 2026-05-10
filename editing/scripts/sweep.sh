#!/bin/bash
# =============================================================================
# Sweep launcher for multimodal motion editing experiments.
#
# Builds a Cartesian product of sweep parameters and runs each combination
# sequentially on the current machine/GPU session (no job scheduler needed).
#
# Edit the CONFIG and SWEEP PARAMETERS sections below, then run:
#   bash editing/scripts/sweep.sh
#
# Each sweep parameter can be a single value or a bash array.
# A Cartesian product of all combinations is formed and run one after another.
#
# Dry-run (print commands without running):
#   DRY_RUN=1 bash editing/scripts/sweep.sh
#
# Example:
#   RETAKE_START_FRAMES=(3 5 25)  → 3 values
#   LR=(0.005 0.05)               → 2 values
#   → 6 sequential runs
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# REQUIRED: video and prompts — always edit these
# ============================================================
SRC_VIDEO="${SRC_VIDEO:-input_videos/hummingbird_hovering_near_a_flower_during_a.mp4}"
EDIT_PROMPT="${EDIT_PROMPT:-Hummingbird extending its beak.}"
STATIC_PROMPT="${STATIC_PROMPT:-Hummingbird hovering near a flower during a light drizzle.}"
BASE_EXP_NAME="${BASE_EXP_NAME:-sweep}"   # prefix; sweep tags are appended automatically

# ============================================================
# MODEL PATHS — override via env vars or set here
# ============================================================
CKPT_ROOT="${CKPT_ROOT:-}"
QWEN_ROOT="${QWEN_ROOT:-}"
GEMMA_ROOT="${GEMMA_ROOT:-}"

# ============================================================
# SWEEP PARAMETERS
# Single value  → one-element array, e.g. (0.005)
# Multiple      → space-separated,   e.g. (0.005 0.01 0.05)
# ============================================================
RETAKE_START_FRAMES=(3 5 25)
QWEN_GRAD_ACCUM_STEPS=(1 3)
QWEN_SAMPLE_MODE=(linspace normal)
QWEN_MAX_FRAMES=(8 24)
LR=(0.005)

# LPIPS_ENABLED=0  → LPIPS_WEIGHT=0.0, TEMPORAL_WEIGHT=0.0  (one combo)
# LPIPS_ENABLED=1  → Cartesian product of the non-zero arrays below
LPIPS_ENABLED=(0 1)
LPIPS_WEIGHT_VALUES=(0.1)
TEMPORAL_WEIGHT_VALUES=(0.05)

# ============================================================
# FIXED SETTINGS — applied identically to every sweep run
# ============================================================
OPT_MODE="${OPT_MODE:-both}"
ITERATIONS="${ITERATIONS:-30}"
EARLY_STOPPING="${EARLY_STOPPING:-15}"
VISUALIZE_EVERY_ITERS="${VISUALIZE_EVERY_ITERS:-5}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
RETAKE_NUM_INFERENCE_STEPS="${RETAKE_NUM_INFERENCE_STEPS:-30}"
FINAL_RETAKE_NUM_INFERENCE_STEPS="${FINAL_RETAKE_NUM_INFERENCE_STEPS:-30}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LATENT_REG_WEIGHT="${LATENT_REG_WEIGHT:-0.01}"
TEXT_REG_WEIGHT="${TEXT_REG_WEIGHT:-0.001}"
REG_SCHEDULE="${REG_SCHEDULE:-cosine_increase}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-95}"
FRAME_RATE="${FRAME_RATE:-}"
QUANTIZATION="${QUANTIZATION:-fp8-cast}"
ENHANCE_PROMPT="${ENHANCE_PROMPT:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sound-sparks-motion}"
WANDB_TAGS="${WANDB_TAGS:-qwen-loss}"
WANDB_MODE="${WANDB_MODE:-offline}"
QWEN_IMG_SIZE="${QWEN_IMG_SIZE:-224}"
QWEN_CONTIGUOUS_START_FRAME="${QWEN_CONTIGUOUS_START_FRAME:-4}"
QWEN_GRADIENT_RUBRIC="${QWEN_GRADIENT_RUBRIC:-motion}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, artifacts, distorted}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAIN_SCRIPT="${REPO_ROOT}/editing/optimize_qwen_vl.py"

# ============================================================
# Cartesian product builder
# ============================================================
combos=("")

_expand() {
    local param="$1"; shift
    local values=("$@")
    local new_combos=()
    for combo in "${combos[@]}"; do
        for val in "${values[@]}"; do
            if [[ -z "$combo" ]]; then
                new_combos+=("${param}=${val}")
            else
                new_combos+=("${combo}|${param}=${val}")
            fi
        done
    done
    combos=("${new_combos[@]}")
}

_expand RETAKE_START_FRAMES  "${RETAKE_START_FRAMES[@]}"
_expand QWEN_GRAD_ACCUM_STEPS "${QWEN_GRAD_ACCUM_STEPS[@]}"
_expand QWEN_SAMPLE_MODE     "${QWEN_SAMPLE_MODE[@]}"
_expand QWEN_MAX_FRAMES      "${QWEN_MAX_FRAMES[@]}"
_expand LR                   "${LR[@]}"

# Build LPIPS pairs
_lpips_pairs=()
for _en in "${LPIPS_ENABLED[@]}"; do
    if [[ "$_en" == "0" ]]; then
        _lpips_pairs+=("0.0,0.0")
    else
        for _lw in "${LPIPS_WEIGHT_VALUES[@]}"; do
            for _tw in "${TEMPORAL_WEIGHT_VALUES[@]}"; do
                _lpips_pairs+=("${_lw},${_tw}")
            done
        done
    fi
done
_expand LPIPS_PAIR "${_lpips_pairs[@]}"

total="${#combos[@]}"
echo "=================================================="
echo "  Sweep — ${total} run(s) to execute sequentially"
echo "  SRC_VIDEO    : ${SRC_VIDEO}"
echo "  EDIT_PROMPT  : ${EDIT_PROMPT}"
echo "  DRY_RUN      : ${DRY_RUN}"
echo "=================================================="

# Resolve src video path
if [[ "${SRC_VIDEO}" != /* ]]; then
    SRC_VIDEO="${REPO_ROOT}/${SRC_VIDEO}"
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO not found: ${SRC_VIDEO}" >&2
    exit 1
fi

# GPU check (skipped in dry-run)
if [[ "${DRY_RUN}" != "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: no visible GPU detected." >&2
        exit 1
    fi

    # Load modules / activate venv once for all runs
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
fi

# ============================================================
# Generate Qwen motion question
# ============================================================
QWEN_MOTION_QUESTION="Does this video clearly show the action or state change described by the edit prompt: \"${EDIT_PROMPT}\"? Answer only 'yes' or 'no'."

PROMPT_SLUG=$(echo "${EDIT_PROMPT}" | tr '[:upper:]' '[:lower:]' | tr -s ' ' | cut -d' ' -f1-5 | tr ' ' '_')

# ============================================================
# Run one combination
# ============================================================
idx=0
for combo in "${combos[@]}"; do
    idx=$(( idx + 1 ))

    IFS='|' read -ra pairs <<< "$combo"

    sweep_env=()
    sweep_tag=""
    lpips_w="0.0"
    temp_w="0.0"

    for pair in "${pairs[@]}"; do
        key="${pair%%=*}"
        v="${pair#*=}"
        if [[ "$key" == "LPIPS_PAIR" ]]; then
            lpips_w="${v%%,*}"
            temp_w="${v##*,}"
            if [[ "$lpips_w" == "0.0" ]]; then
                sweep_tag+="lpipsOFF_"
            else
                sweep_tag+="lpips${lpips_w}_temp${temp_w}_"
            fi
        else
            sweep_env+=("${pair}")
            case "$key" in
                RETAKE_START_FRAMES)   sweep_tag+="rsf${v}_"    ;;
                LR)                    sweep_tag+="lr${v}_"     ;;
                QWEN_GRAD_ACCUM_STEPS) sweep_tag+="accum${v}_"  ;;
                QWEN_SAMPLE_MODE)      sweep_tag+="mode${v}_"   ;;
                QWEN_MAX_FRAMES)       sweep_tag+="frames${v}_" ;;
            esac
        fi
    done
    sweep_tag="${sweep_tag%_}"

    OUTPUT_DIR="${REPO_ROOT}/results/QwenVL/${PROMPT_SLUG}/${BASE_EXP_NAME}/${sweep_tag}"

    echo ""
    echo "[${idx}/${total}]  ${combo}"
    echo "          Output: ${OUTPUT_DIR}"

    # Parse sweep_env into local vars
    declare -A _senv=()
    for pair in "${sweep_env[@]}"; do
        _senv["${pair%%=*}"]="${pair#*=}"
    done

    _rsf="${_senv[RETAKE_START_FRAMES]:-${RETAKE_START_FRAMES[0]}}"
    _accum="${_senv[QWEN_GRAD_ACCUM_STEPS]:-${QWEN_GRAD_ACCUM_STEPS[0]}}"
    _mode="${_senv[QWEN_SAMPLE_MODE]:-${QWEN_SAMPLE_MODE[0]}}"
    _frames="${_senv[QWEN_MAX_FRAMES]:-${QWEN_MAX_FRAMES[0]}}"
    _lr="${_senv[LR]:-${LR[0]}}"
    unset _senv

    ARGS=(
        --src-video "${SRC_VIDEO}"
        --edit-prompt "${EDIT_PROMPT}"
        --static-prompt "${STATIC_PROMPT}"
        --negative-prompt "${NEGATIVE_PROMPT}"
        --output-dir "${OUTPUT_DIR}"
        --opt-mode "${OPT_MODE}"
        --checkpoint-path "${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
        --qwen-model "${QWEN_ROOT}"
        --gemma-root "${GEMMA_ROOT}"
        --seed "${SEED}"
        --height "${HEIGHT}" --width "${WIDTH}" --num-frames "${NUM_FRAMES}"
        --iterations "${ITERATIONS}"
        --early-stopping "${EARLY_STOPPING}"
        --lr "${_lr}"
        --lr-schedule "${LR_SCHEDULE}"
        --grad-clip "${GRAD_CLIP}"
        --retake-start-frames "${_rsf}"
        --num-inference-steps "${NUM_INFERENCE_STEPS}"
        --retake-num-inference-steps "${RETAKE_NUM_INFERENCE_STEPS}"
        --final-retake-num-inference-steps "${FINAL_RETAKE_NUM_INFERENCE_STEPS}"
        --qwen-max-frames "${_frames}"
        --qwen-img-size "${QWEN_IMG_SIZE}"
        --qwen-sample-mode "${_mode}"
        --qwen-contiguous-start-frame "${QWEN_CONTIGUOUS_START_FRAME}"
        --qwen-gradient-rubric "${QWEN_GRADIENT_RUBRIC}"
        --qwen-grad-accum-steps "${_accum}"
        --qwen-motion-question "${QWEN_MOTION_QUESTION}"
        --lpips-weight "${lpips_w}"
        --temporal-weight "${temp_w}"
        --lpips-backbone "alex"
        --latent-reg-weight "${LATENT_REG_WEIGHT}"
        --text-reg-weight "${TEXT_REG_WEIGHT}"
        --reg-schedule "${REG_SCHEDULE}"
        --visualize-every-iters "${VISUALIZE_EVERY_ITERS}"
        --quantization "${QUANTIZATION}"
        --clip-similarity-diag
        --gradient-checkpointing
        --save-final-videos
        --low-memory-guidance
    )

    [[ -n "${FRAME_RATE}" ]] && ARGS+=( --frame-rate "${FRAME_RATE}" )
    [[ "${ENHANCE_PROMPT}" == "1" ]] && ARGS+=( --enhance-prompt )

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  [DRY_RUN] %q ' "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${ARGS[@]}"
        printf '\n'
    else
        mkdir -p "${OUTPUT_DIR}"
        export WANDB_PROJECT WANDB_TAGS WANDB_MODE
        export WANDB_DIR="${OUTPUT_DIR}/wandb"
        export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
        export WANDB_DISABLE_GIT=true
        mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

        "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${ARGS[@]}"
        echo "  [${idx}/${total}] Finished: ${OUTPUT_DIR}"
    fi
done

echo ""
echo "=================================================="
echo "  Sweep complete. ${total} run(s) executed."
echo "=================================================="
