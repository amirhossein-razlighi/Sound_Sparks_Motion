#!/bin/bash
# SPSA optimization of the H3 audio-reference latent (Qwen critic) in the
# grammar-aligned [video editing] regime.   sbatch h3_probe/opt.sbatch
#SBATCH --job-name=h3-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/h3_eval_%j.out
#SBATCH --error=slurm_logs/h3_eval_%j.err
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg31701,rg13401,rg21803
set -euo pipefail
cd /home/amirrz/my_codes/Sound_Sparks_Motion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
/scratch/amirrz/H3_exp/venv/bin/python h3_probe/eval_arms.py
echo "H3 OPT DONE"
