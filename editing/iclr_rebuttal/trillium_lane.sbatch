#!/bin/bash
# TRILLIUM: one H100, drains queue $SSM_LANE_DIR/queue.txt with worker.sh. Submit from $SCRATCH:
#   cd /scratch/amirrz/iclr_rebuttal && SSM_LANE_DIR=<lane> sbatch --time=HH:MM:SS <repo>/editing/iclr_rebuttal/trillium_lane.sbatch
# End the queue with a line "CMD touch <lane>/STOP" so the worker exits instead of idling.
#SBATCH --job-name=ssm-trl
#SBATCH --account=def-amahdavi
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/trl_%j.out
set -uo pipefail
export SSM_JOB_ENV=/home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env_trillium.sh
export SSM_OUT_ROOT=${SSM_OUT_ROOT:-/scratch/amirrz/iclr_rebuttal/out}
export SSM_LANE_DIR=${SSM_LANE_DIR:?set SSM_LANE_DIR}
mkdir -p "$SSM_OUT_ROOT/logs" "$SSM_LANE_DIR"; touch "$SSM_LANE_DIR/.lock"
bash /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/worker.sh "${SLURM_JOB_ID}"
