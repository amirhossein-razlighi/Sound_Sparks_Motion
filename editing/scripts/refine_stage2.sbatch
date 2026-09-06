#!/bin/bash
# Stage-2 distilled refinement of demo videos -> highres_refined.mp4 (1152x720).
#   sbatch editing/scripts/refine_stage2.sbatch          # all dirs
#   DIRS="results/videos/man_raises_hand/ results/videos/man_pets_dog/" \
#     SIGMAS="0.421875,0.0" OUT_NAME=highres_refined_gentle.mp4 \
#     sbatch editing/scripts/refine_stage2.sbatch        # variants/subsets
#SBATCH --job-name=ltx-refine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/ltx_refine_%j.out
#SBATCH --error=slurm_logs/ltx_refine_%j.err
#SBATCH --account=def-amahdavi
set -euo pipefail

REPO_ROOT="/home/amirrz/my_codes/Sound_Sparks_Motion"
cd "${REPO_ROOT}"
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda 2>/dev/null || true; fi
source .venv/bin/activate
source editing/scripts/rebuttal/env.sh   # CKPT_ROOT, GEMMA_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python3 editing/scripts/refine_stage2.py
echo "REFINE JOB DONE"
