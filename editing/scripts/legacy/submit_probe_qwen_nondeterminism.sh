#!/bin/bash
# =============================================================================
# SLURM job: probe Qwen2.5-VL forward/backward non-determinism.
#
# Runs probe_qwen_nondeterminism.py under three conditions back-to-back:
#   1. baseline      — torch defaults, cudnn.deterministic=False
#   2. deterministic — cudnn.deterministic=True, cudnn.benchmark=False
#   3. seeded        — deterministic + re-seed before every Qwen call
#
# Each mode runs N_REPEATS identical forward+backward passes on the same
# fixed pixel values and reports:
#   - yes_prob  stdev  → forward non-determinism
#   - grad cos_sim     → backward non-determinism (direction stability)
#   - grad norm stdev  → backward non-determinism (magnitude stability)
#
# Usage:
#   sbatch editing/scripts/submit_probe_qwen_nondeterminism.sh
# =============================================================================

#SBATCH --job-name=qwen_probe
#SBATCH --account=your-hpc-account
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

nvidia-smi

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT}}"
QWEN_ROOT="${QWEN_ROOT:-${QWEN_ROOT}}"
VIDEO="${VIDEO:-${REPO_ROOT}/input_videos/a_red_ferrari_standing_still_in_the.mp4}"
EDIT_PROMPT="${EDIT_PROMPT:-A red car door opens.}"
N_FRAMES="${N_FRAMES:-8}"
N_REPEATS="${N_REPEATS:-20}"

echo "========================================================"
echo "  Job ID      : ${SLURM_JOB_ID:-local}"
echo "  Qwen model  : ${QWEN_ROOT}"
echo "  Video       : ${VIDEO}"
echo "  Edit prompt : ${EDIT_PROMPT}"
echo "  N frames    : ${N_FRAMES}"
echo "  N repeats   : ${N_REPEATS}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9
source "${REPO_ROOT}/.venv/bin/activate"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT="${REPO_ROOT}/editing/scripts/probe_qwen_nondeterminism.py"

COMMON_ARGS=(
    --qwen-model  "${QWEN_ROOT}"
    --video       "${VIDEO}"
    --edit-prompt "${EDIT_PROMPT}"
    --n-frames    "${N_FRAMES}"
    --n-repeats   "${N_REPEATS}"
)

# ---------------------------------------------------------------------------
# Run all three modes sequentially (model is re-loaded each time to ensure
# the determinism flag is applied before any CUDA kernel is JIT-compiled)
# ---------------------------------------------------------------------------
for MODE in baseline deterministic seeded; do
    echo ""
    echo "################################################################"
    echo "##  MODE: ${MODE}"
    echo "################################################################"
    python "${SCRIPT}" "${COMMON_ARGS[@]}" --mode "${MODE}"
done

echo ""
echo "All three modes finished. Exit code: $?"
