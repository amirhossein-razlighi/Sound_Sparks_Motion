#!/bin/bash
# CPU-only: manifest + DINO/motion metrics + report + curves + package (held-out scores for runs not yet scored = blank).
#SBATCH --job-name=ssm-mcr
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --account=def-amahdavi
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/multicritic/logs/mcr_%j.out
ROOT=/scratch/amirrz/iclr_rebuttal/multicritic
H=/home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/multicritic
source /home/amirrz/my_codes/Sound_Sparks_Motion/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
python $H/mc_manifest.py
[[ -n $SKIP_METRICS ]] || python $H/mc_metrics.py > $ROOT/logs/metrics_$SLURM_JOB_ID.log 2>&1
python $H/mc_report.py && /scratch/amirrz/iclr_rebuttal/venv_vbench/bin/python $H/mc_curves.py && \
  /scratch/amirrz/iclr_rebuttal/venv_vbench/bin/python -c "
import zipfile; z=zipfile.ZipFile('$ROOT/multicritic_package.zip','a'); z.write('$ROOT/report/curves.png','multicritic/report/curves.png'); z.close()"
echo "MC_REPORT_CPU rc=$? job=$SLURM_JOB_ID" >> $ROOT/events.txt
