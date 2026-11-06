#!/bin/bash
# TRILLIUM: pairwise VLM judge with source reference (trillium_pairwise_src.py). Resumable.
#   cd /scratch/amirrz/iclr_rebuttal && sbatch <repo>/editing/iclr_rebuttal/trillium_pairwise_src.sbatch <pairs.csv> <out_dir>
#SBATCH --job-name=ssm-pwsrc
#SBATCH --account=def-amahdavi
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/pwsrc_%j.out
set -uo pipefail
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env_trillium.sh
python trillium_pairwise_src.py --pairs "${1:?pairs}" --out "${2:?out}" --models qwen3vl,internvl3 --n-frames 24; echo "pairwise rc=$?"
echo "end=$(date -Is)"
