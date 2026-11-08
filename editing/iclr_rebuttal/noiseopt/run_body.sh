#!/bin/bash
# One noise-optimisation run on one GPU: bash run_body.sh <scenario> <lr> [tag]. Used via salloc ... srun (shared storage).
S=$1; LR=$2; TAG=${3:-lr$LR}; O=/scratch/amirrz/iclr_rebuttal/noiseopt/$S/$TAG
[[ -e $O/timing.json ]] && { echo "already done: $O"; exit 0; }
[[ -e $O ]] && mv $O $O.old_$(date +%m%d%H%M%S)
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
python noiseopt/run_noiseopt.py --scenario $S --output-dir $O --noise-lr $LR --iter-every 1
echo "NOISE_RUN $S $TAG rc=$?"
