#!/bin/bash
# Critic-guided optimization of the pinned audio latent on H3.
#   sbatch --export=ALL,SLUG=cat_yawns,BANK="a;b;c",OUT_DIR=... h3_probe/optpin.sbatch
#SBATCH --job-name=h3-optpin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/h3_optpin_%j.out
#SBATCH --error=slurm_logs/h3_optpin_%j.err
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg31701,rg13401,rg21803
set -euo pipefail
cd /home/amirrz/my_codes/Sound_Sparks_Motion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
/scratch/amirrz/H3_exp/venv/bin/python h3_probe/optimize_pinned.py
echo "OPTPIN JOB DONE"
