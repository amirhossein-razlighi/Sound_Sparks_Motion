#!/bin/bash
# P3/P4 optimisation runs. Task i = line i (0-based) of $TASKS (default opt_tasks.txt):
#   <exp> <scenario> <tag> [key=value ...]   -> results/iclr_rebuttal/<exp>/<scenario>/<tag>/
#SBATCH --job-name=ssm-opt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg32102,rg31801
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/opt_%A_%a.out
set -uo pipefail
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
TASKS=${TASKS:-/home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/opt_tasks.txt}
LINE=$(grep -v '^#' "$TASKS" | grep -v '^\s*$' | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read -r EXP SCEN TAG REST <<< "$LINE"
OUT=/scratch/amirrz/iclr_rebuttal/$EXP/$SCEN/$TAG
SETS=(); for kv in $REST; do SETS+=(--set "$kv"); done
echo "TASK: $LINE -> $OUT"
if [[ -f "$OUT/mode_both/best_optimized_video_both.mp4" && -f "$OUT/timing.json" ]]; then echo "already done"; exit 0; fi
python run_opt.py --scenario "$SCEN" --output-dir "$OUT" "${SETS[@]}" ${EXTRA_SETS:-}
RC=$?
echo "run_opt rc=$RC"
if [[ $RC -eq 0 ]]; then
  python ../scripts/eval_metrics.py --output-dir "$OUT" --mode both --device cuda --raft-disable-cudnn || echo "eval_metrics failed rc=$?"
fi
echo "end=$(date -Is)"
exit $RC
