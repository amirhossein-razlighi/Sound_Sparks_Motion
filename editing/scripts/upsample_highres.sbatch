#!/bin/bash
# Upscale paper-demo videos (results/videos/*/best_optimized_*.mp4) with the
# LTX x2 latent upsampler -> highres_2x.mp4 (1024x640) + highres.mp4 (1728x1080).
#   sbatch editing/scripts/upsample_highres.sbatch
#SBATCH --job-name=ltx-highres
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/ltx_highres_%j.out
#SBATCH --error=slurm_logs/ltx_highres_%j.err
#SBATCH --account=def-amahdavi
set -euo pipefail

REPO_ROOT="/home/amirrz/my_codes/Sound_Sparks_Motion"
cd "${REPO_ROOT}"
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda 2>/dev/null || true; fi
source .venv/bin/activate
source editing/scripts/rebuttal/env.sh   # CKPT_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python3 editing/scripts/upsample_highres.py --videos-root "${VIDEOS_ROOT:-results/videos}" --pattern "${PATTERN:-best_optimized_*.mp4}"
echo "HIGHRES DONE"
