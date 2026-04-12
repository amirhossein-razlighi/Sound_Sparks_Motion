#!/bin/bash
# =============================================================================
# Sweep launcher for submit_optimize_qwen_vl.sh
#
# Edit the CONFIG section, then run:
#   bash editing/scripts/sweep_qwen_vl.sh
#
# Each sweep parameter can be a single value OR a bash array.
# A cartesian product of all combinations is submitted as separate sbatch jobs.
#
# Example:
#   RETAKE_START_FRAMES=(1 5 10 15)   → 4 values
#   LR=(0.005 0.05)                   → 2 values
#   → 4 × 2 = 8 jobs submitted
#
# Dry-run (print commands, do not submit):
#   DRY_RUN=1 bash editing/scripts/sweep_qwen_vl.sh
# =============================================================================

# ============================================================
# PROMPTS & VIDEO  — always edit these
# ============================================================
SRC_VIDEO="/path/to/your/video.mp4"
EDIT_PROMPT="A red car door opens."
STATIC_PROMPT="A red car in the middle of a garage."
NAME_OF_THIS_EXP="my_sweep"   # base name; sweep tags are appended automatically

# ============================================================
# SWEEP PARAMETERS
# Single value  →  one element array, e.g. (0.0)
# Multiple values → space-separated, e.g. (0.0 0.1 0.5)
# ============================================================
RETAKE_START_FRAMES=(5)
QWEN_GRAD_ACCUM_STEPS=(3)
QWEN_SAMPLE_MODE=(linspace)
LPIPS_WEIGHT=(0.0)
TEMPORAL_WEIGHT=(0.0)
LR=(0.005)

# ============================================================
# FIXED SETTINGS  (not swept, passed as-is to every job)
# Leave a variable empty ("") to use the sbatch script default.
# ============================================================
ITERATIONS=30
QWEN_MAX_FRAMES=8
EARLY_STOPPING=15
VISUALIZE_EVERY_ITERS=5
SEED=42
NUM_INFERENCE_STEPS=30
RETAKE_NUM_INFERENCE_STEPS=30
FINAL_RETAKE_NUM_INFERENCE_STEPS=30
GRAD_CLIP=0.0
LR_SCHEDULE=cosine
LATENT_REG_WEIGHT=0.01
TEXT_REG_WEIGHT=0.001
REG_SCHEDULE=cosine_increase
HEIGHT=320
WIDTH=512
NUM_FRAMES=95
FRAME_RATE=""
QUANTIZATION=fp8-cast
ENHANCE_PROMPT=1
OPT_MODE=both
WANDB_PROJECT=ltx-qwen-opt
WANDB_TAGS="qwen-loss,cluster-offline"

# ============================================================
# SLURM settings for each submitted job
# ============================================================
SBATCH_TIME="02:00:00"
SBATCH_MEM="64G"
SBATCH_GPU="h100:1"
SBATCH_ACCOUNT="def-amahdavi"

# ============================================================
# Internal: paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/submit_optimize_qwen_vl.sh"
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Cartesian product
# ============================================================
# Each entry in `combos` is a |-separated string of KEY=val pairs.
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
_expand LPIPS_WEIGHT         "${LPIPS_WEIGHT[@]}"
_expand TEMPORAL_WEIGHT      "${TEMPORAL_WEIGHT[@]}"
_expand LR                   "${LR[@]}"

total="${#combos[@]}"
echo "=================================================="
echo "  Sweep launcher — ${total} job(s) to submit"
echo "  SRC_VIDEO    : ${SRC_VIDEO}"
echo "  EDIT_PROMPT  : ${EDIT_PROMPT}"
echo "  STATIC_PROMPT: ${STATIC_PROMPT}"
echo "  DRY_RUN      : ${DRY_RUN}"
echo "=================================================="

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "ERROR: sbatch script not found: ${SBATCH_SCRIPT}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${SRC_VIDEO}" ]]; then
    echo "ERROR: SRC_VIDEO not found: ${SRC_VIDEO}" >&2
    exit 1
fi

# ============================================================
# Submit one job per combination
# ============================================================
idx=0
for combo in "${combos[@]}"; do
    idx=$(( idx + 1 ))

    # Parse combo string into an array of KEY=val pairs
    IFS='|' read -ra pairs <<< "$combo"

    sweep_env=()
    sweep_tag=""
    for pair in "${pairs[@]}"; do
        sweep_env+=("${pair}")
        key="${pair%%=*}"
        val="${pair#*=}"
        # Build a short tag for the output directory
        case "$key" in
            RETAKE_START_FRAMES)   sweep_tag+="rsf${val}_"   ;;
            LR)                    sweep_tag+="lr${val}_"    ;;
            LPIPS_WEIGHT)          sweep_tag+="lpips${val}_" ;;
            TEMPORAL_WEIGHT)       sweep_tag+="temp${val}_"  ;;
            QWEN_GRAD_ACCUM_STEPS) sweep_tag+="accum${val}_" ;;
            QWEN_SAMPLE_MODE)      sweep_tag+="mode${val}_"  ;;
        esac
    done
    sweep_tag="${sweep_tag%_}"   # strip trailing underscore

    job_name="${NAME_OF_THIS_EXP}/${sweep_tag}"

    echo ""
    echo "[${idx}/${total}]  ${combo}"
    echo "          output subdir: ${job_name}"

    # Build the fixed env vars to pass alongside the sweep overrides
    fixed_env=(
        SRC_VIDEO="${SRC_VIDEO}"
        EDIT_PROMPT="${EDIT_PROMPT}"
        STATIC_PROMPT="${STATIC_PROMPT}"
        NAME_OF_THIS_EXP="${job_name}/"
        ITERATIONS="${ITERATIONS}"
        QWEN_MAX_FRAMES="${QWEN_MAX_FRAMES}"
        EARLY_STOPPING="${EARLY_STOPPING}"
        VISUALIZE_EVERY_ITERS="${VISUALIZE_EVERY_ITERS}"
        SEED="${SEED}"
        NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}"
        RETAKE_NUM_INFERENCE_STEPS="${RETAKE_NUM_INFERENCE_STEPS}"
        FINAL_RETAKE_NUM_INFERENCE_STEPS="${FINAL_RETAKE_NUM_INFERENCE_STEPS}"
        GRAD_CLIP="${GRAD_CLIP}"
        LR_SCHEDULE="${LR_SCHEDULE}"
        LATENT_REG_WEIGHT="${LATENT_REG_WEIGHT}"
        TEXT_REG_WEIGHT="${TEXT_REG_WEIGHT}"
        REG_SCHEDULE="${REG_SCHEDULE}"
        HEIGHT="${HEIGHT}"
        WIDTH="${WIDTH}"
        NUM_FRAMES="${NUM_FRAMES}"
        QUANTIZATION="${QUANTIZATION}"
        ENHANCE_PROMPT="${ENHANCE_PROMPT}"
        OPT_MODE="${OPT_MODE}"
        WANDB_PROJECT="${WANDB_PROJECT}"
        WANDB_TAGS="${WANDB_TAGS}"
    )

    # Optional fixed vars — only pass if non-empty
    [[ -n "${FRAME_RATE}" ]] && fixed_env+=( FRAME_RATE="${FRAME_RATE}" )

    # Sweep overrides come last so they win over any fixed defaults
    all_env=("${fixed_env[@]}" "${sweep_env[@]}")

    sbatch_cmd=(
        sbatch
        --account="${SBATCH_ACCOUNT}"
        --gpus-per-node="${SBATCH_GPU}"
        --mem="${SBATCH_MEM}"
        --time="${SBATCH_TIME}"
        --job-name="qwen_${sweep_tag}"
        "${SBATCH_SCRIPT}"
        "${SRC_VIDEO}"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "    [DRY_RUN] env ${all_env[*]} ${sbatch_cmd[*]}"
    else
        env "${all_env[@]}" "${sbatch_cmd[@]}"
    fi
done

echo ""
echo "=================================================="
echo "  Done. ${total} job(s) submitted."
echo "=================================================="
