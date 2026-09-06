#!/bin/bash
# MiniMax-H3 Phase-1 smoke: one Ref2VA generation with a source-clip reference.
#   sbatch h3_probe/smoke.sbatch
#SBATCH --job-name=h3-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/h3_smoke_%j.out
#SBATCH --error=slurm_logs/h3_smoke_%j.err
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg31701,rg13401,rg21803
set -euo pipefail

REPO_ROOT="/home/amirrz/my_codes/Sound_Sparks_Motion"
cd "${REPO_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

/scratch/amirrz/H3_exp/venv/bin/python h3_probe/smoke_ref2va.py
echo "H3 SMOKE DONE"
