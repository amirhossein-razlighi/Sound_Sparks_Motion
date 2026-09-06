#!/bin/bash
# Baseline-failure hunt: render + score H3 text-editing baselines for all
# untested paper scenarios + counter-prior edits.   sbatch h3_probe/sweep.sbatch
#SBATCH --job-name=h3-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/h3_sweep_%j.out
#SBATCH --error=slurm_logs/h3_sweep_%j.err
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg31701,rg13401,rg21803
set -euo pipefail
cd /home/amirrz/my_codes/Sound_Sparks_Motion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
/scratch/amirrz/H3_exp/venv/bin/python h3_probe/baseline_sweep.py
echo "SWEEP JOB DONE"
