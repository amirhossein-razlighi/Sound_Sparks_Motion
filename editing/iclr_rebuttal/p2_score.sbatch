#!/bin/bash
# P2: loop-critic rescoring + held-out VLM scoring of manifests (inference only).
# Usage: sbatch --export=ALL,MANIFESTS="comparisons controls",MODELS=qwen25,qwen3vl,internvl3 p2_score.sbatch
#SBATCH --job-name=ssm-p2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg32102,rg31801
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/p2_%j.out
set -uo pipefail
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
MD=/scratch/amirrz/iclr_rebuttal/manifests
OUT=/scratch/amirrz/iclr_rebuttal/heldout
MANIFESTS=${MANIFESTS:-comparisons controls}
LOOP_ARGS=(); for m in $MANIFESTS; do LOOP_ARGS+=(--manifest "$MD/$m.csv"); done
python loop_rescore.py "${LOOP_ARGS[@]}" --out "$OUT" || echo "loop_rescore rc=$?"
cat $(for m in $MANIFESTS; do echo "$MD/$m.csv"; done) | awk 'NR==1 || !/^id,/' > "$OUT/_manifest_${SLURM_JOB_ID}.csv"
python heldout_score.py --manifest "$OUT/_manifest_${SLURM_JOB_ID}.csv" --out "$OUT" --models "${MODELS:-qwen25,qwen3vl,internvl3}"
echo "end=$(date -Is)"
