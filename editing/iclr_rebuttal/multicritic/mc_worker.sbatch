#!/bin/bash
# 2-GPU batch job running one multi-critic queue worker. Short walltime -> backfill.
#SBATCH --job-name=ssm-mcb
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=02:59:00
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg32102,rg31801,rg21702
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/multicritic/logs/mcb_%j.out
IDLE_MAX=20 bash /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/multicritic/mc_worker.sh b$SLURM_JOB_ID
