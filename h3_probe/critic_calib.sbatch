#!/bin/bash
# Critic sensitivity calibration (Qwen only, 1 GPU).   sbatch h3_probe/critic_calib.sbatch
#SBATCH --job-name=h3-calib
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/h3_calib_%j.out
#SBATCH --error=slurm_logs/h3_calib_%j.err
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg31701,rg13401,rg21803,rg21802,rg31502
set -euo pipefail
cd /home/amirrz/my_codes/Sound_Sparks_Motion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/scratch/amirrz/.cache/torch
G=/scratch/amirrz/H3_exp/outputs/fullmethod_goldfish_both
V="$G/baseline.mp4;$G/iter_02.mp4;$G/iter_05.mp4"
:
V="$V;/scratch/amirrz/H3_exp/outputs/t2va_sounds/goldfish__t2va.mp4;/home/amirrz/my_codes/Sound_Sparks_Motion/results/videos/goldfish/best_optimized_video_both.mp4;/home/amirrz/my_codes/Sound_Sparks_Motion/results/videos/goldfish/retake_input_prepared.mp4"
export VIDEOS="$V" OUT=/scratch/amirrz/H3_exp/outputs/critic_calib_goldfish_r3_${FRAMES:-24}.json QUESTIONS_JSON=/scratch/amirrz/H3_exp/inputs/bank/questions_r3.json
/scratch/amirrz/H3_exp/venv/bin/python h3_probe/critic_calib.py
echo "CALIB JOB DONE"
