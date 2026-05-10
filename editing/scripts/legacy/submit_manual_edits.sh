#!/bin/bash
# =============================================================================
# SLURM single job: run ALL (or a selected subset of) perturbations in one job.
#
# Uses RetakePipeline — the full source video is encoded as context for every
# perturbation.  The model loads once and all perturbations run sequentially.
# Increase --time if running many perturbations (each ~10–20 min on H100).
#
# Quick start:
#   sbatch editing/scripts/submit_all.sh \
#       /path/to/video.mp4 "A person playing guitar on stage"
#
# Or set via environment variables before submitting:
#   export SRC_VIDEO=/path/to/video.mp4
#   export PROMPT="A person playing guitar on stage"
#   sbatch editing/scripts/submit_all.sh
# =============================================================================

#SBATCH --job-name=ltx_audio_edit_all
#SBATCH --account=your-hpc-account
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT}}"
CKPT_ROOT="${CKPT_ROOT:-${CKPT_ROOT}}"
GEMMA_ROOT="${GEMMA_ROOT:-${GEMMA_ROOT}}"

# Positional args: $1 = src video path, $2 = prompt
SRC_VIDEO="${SRC_VIDEO:-${1:-}}"
PROMPT="${PROMPT:-${2:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/editing_results}"

SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
FRAME_RATE="${FRAME_RATE:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
# CFG/guidance scales: leave empty to auto-detect from checkpoint metadata
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"

# Space-separated subset, or empty = all
PERTURBATIONS_SUBSET="${PERTURBATIONS_SUBSET:-}"

# Number of frames at the start to keep unchanged (context anchor for the retake).
# 1 = preserve just the first frame (default); increase for stronger scene anchoring.
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-10}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${SRC_VIDEO}" ]]; then
    echo "ERROR: Source video not set." >&2
    echo "Usage: sbatch submit_all.sh /path/to/video.mp4 'your prompt'" >&2
    echo "   or: export SRC_VIDEO=... PROMPT=... && sbatch submit_all.sh" >&2
    exit 1
fi
if [[ -z "${PROMPT}" ]]; then
    echo "ERROR: Prompt not set." >&2
    exit 1
fi

echo "========================================================"
echo "  Job ID              : ${SLURM_JOB_ID}"
echo "  Source video        : ${SRC_VIDEO}"
echo "  Prompt              : ${PROMPT}"
echo "  Retake start frames : ${RETAKE_START_FRAMES}"
echo "  Output dir          : ${OUTPUT_DIR}"
echo "  Perturbations       : ${PERTURBATIONS_SUBSET:-ALL}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9
source "${REPO_ROOT}/.venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Build optional arguments
# ---------------------------------------------------------------------------
SHAPE_ARGS=""
[[ -n "${HEIGHT}" ]]             && SHAPE_ARGS+=" --height ${HEIGHT}"
[[ -n "${WIDTH}" ]]              && SHAPE_ARGS+=" --width ${WIDTH}"
[[ -n "${NUM_FRAMES}" ]]         && SHAPE_ARGS+=" --num-frames ${NUM_FRAMES}"
[[ -n "${FRAME_RATE}" ]]         && SHAPE_ARGS+=" --frame-rate ${FRAME_RATE}"
# CFG/guidance scales only passed if explicitly set; otherwise auto-detected from checkpoint
[[ -n "${CFG_SCALE}" ]]       && SHAPE_ARGS+=" --cfg-scale ${CFG_SCALE}"
[[ -n "${AUDIO_CFG_SCALE}" ]] && SHAPE_ARGS+=" --audio-cfg-scale ${AUDIO_CFG_SCALE}"
[[ -n "${A2V_SCALE}" ]]       && SHAPE_ARGS+=" --a2v-scale ${A2V_SCALE}"

PERT_ARGS=""
if [[ -n "${PERTURBATIONS_SUBSET}" ]]; then
    PERT_ARGS="--perturbations ${PERTURBATIONS_SUBSET}"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "${REPO_ROOT}/editing/experiment.py" \
    --src-video              "${SRC_VIDEO}" \
    --prompt                 "${PROMPT}" \
    --output-dir             "${OUTPUT_DIR}" \
    --checkpoint-path        "${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors" \
    --gemma-root             "${GEMMA_ROOT}" \
    --seed                   "${SEED}" \
    --num-inference-steps    "${NUM_INFERENCE_STEPS}" \
    --retake-start-frames    "${RETAKE_START_FRAMES}" \
    ${SHAPE_ARGS} \
    ${PERT_ARGS}

echo "All perturbations finished with exit code $?"
