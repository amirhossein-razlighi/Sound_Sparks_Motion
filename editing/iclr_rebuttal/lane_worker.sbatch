#!/bin/bash
# One-GPU batch job running a lane worker (drains lane/queue.txt). Short walltime -> fast backfill start.
#SBATCH --job-name=ssm-lane
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=02:59:00
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg32102,rg31801,rg21702
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/laneb_%j.out
bash /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/worker.sh b$SLURM_JOB_ID
