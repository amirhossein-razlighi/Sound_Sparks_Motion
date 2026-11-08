#!/bin/bash
# Noise-optimisation smoke test: 3 iterations of dog_yawning; measures s/iter and peak memory of full-chain backprop.
#SBATCH --job-name=ssm-noise-smoke
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --account=def-amahdavi
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/noiseopt/logs/smoke_%j.out
set -x
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
python noiseopt/run_noiseopt.py --scenario dog_yawning --output-dir /scratch/amirrz/iclr_rebuttal/noiseopt/smoke_dog_$SLURM_JOB_ID \
  --iter-every 1 --set iterations=3
echo "SMOKE rc=$?"
