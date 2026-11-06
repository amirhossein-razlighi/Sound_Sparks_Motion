#!/bin/bash
#SBATCH --job-name=ive-prep
#SBATCH --account=def-amahdavi
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/ive_prep_%j.out
cd /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/ivebench
/home/amirrz/my_codes/Sound_Sparks_Motion/.venv/bin/python prepare_inputs.py
