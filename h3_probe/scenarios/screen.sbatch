#!/bin/bash
# Baseline screening of user-study candidates (1 GPU).  sbatch --export=ALL,GROUP=existing h3_probe/scenarios/screen.sbatch
#SBATCH --job-name=h3-screen
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --time=02:30:00
#SBATCH --output=slurm_logs/h3_screen_%j.out
#SBATCH --error=slurm_logs/h3_screen_%j.err
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg31701,rg13401,rg21803,rg21802,rg31502,rg32202
set -euo pipefail
cd /home/amirrz/my_codes/Sound_Sparks_Motion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/scratch/amirrz/.cache/torch PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CANDS=${CANDS:-h3_probe/scenarios/candidates_r1.json} OUT_DIR=${OUT_DIR:-/scratch/amirrz/H3_exp/outputs/screen_r1}
/scratch/amirrz/H3_exp/venv/bin/python h3_probe/scenarios/screen.py
echo "SCREEN JOB DONE"
