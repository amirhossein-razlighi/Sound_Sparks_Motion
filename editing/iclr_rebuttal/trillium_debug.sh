#!/bin/bash
#SBATCH --job-name=ssm-dbg
#SBATCH --account=def-amahdavi
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --output=/scratch/amirrz/iclr_rebuttal/logs/dbg_%j.out
# TRILLIUM debug-partition worker (additive; does not change worker.sh).
# Trillium `debug` = 1 h wall (1 node), 1 job/user. Each job pops tasks from the SAME shared lane queue as the compute lanes,
# one at a time, and runs each through worker.sh (on a private one-task lane, so the run logic is identical).
#  * A task only starts if the time left > EST_S (estimated full-task seconds).
#  * If Slurm is about to kill the job (sbatch --signal=B:USR1@240), the running task is stopped, its partial
#    output dir is RENAMED (never deleted) to <tag>.preempted_<jobid>, and its line is put back at the TOP of the
#    shared queue, so the next job (debug or compute) restarts it from scratch.
# Usage (from $SCRATCH):  sbatch <repo>/editing/iclr_rebuttal/trillium_debug.sh <lane_dir> <EST_S>
LANE=${1:?lane dir}; EST_S=${2:-5400}
REPO=/home/amirrz/my_codes/Sound_Sparks_Motion
export SSM_JOB_ENV=$REPO/editing/iclr_rebuttal/job_env_trillium.sh
export SSM_OUT_ROOT=${SSM_OUT_ROOT:-/scratch/amirrz/iclr_rebuttal/out}
END=$(( $(date +%s) + $(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>2)s+=3600*$(n-2); if(n>3)s+=86400*$(n-3); print s}') ))
CUR=""; WPID=""
log() { echo "$(date -Is) dbg$SLURM_JOB_ID $*" | tee -a "$LANE/events.txt"; }
requeue() {
  [[ -z "$CUR" ]] && exit 0
  log "PREEMPT (time limit) -> requeue: $CUR"
  [[ -n "$WPID" ]] && { kill -TERM -- -"$WPID"; sleep 20; kill -KILL -- -"$WPID"; } 2>/dev/null
  read -r T E S G _ <<< "$CUR"
  if [[ $T == OPT && -d $SSM_OUT_ROOT/$E/$S/$G ]]; then mv "$SSM_OUT_ROOT/$E/$S/$G" "$SSM_OUT_ROOT/$E/$S/$G.preempted_$SLURM_JOB_ID"; fi
  NAME=$(echo "${E}_${S}_${G}" | tr -c 'A-Za-z0-9_.\n-' '_' | cut -c1-120); LOGF=$SSM_OUT_ROOT/logs/lane_${T}_${NAME}.log
  [[ -f $LOGF ]] && mv "$LOGF" "${LOGF%.log}.preempted_$SLURM_JOB_ID.log"
  flock "$LANE/.lock" bash -c "{ printf '%s\n' \"\$1\"; cat '$LANE/queue.txt'; } > '$LANE/.q.tmp' && mv '$LANE/.q.tmp' '$LANE/queue.txt'" _ "$CUR"
  echo "PREEMPTED|$CUR" >> "$LANE/done.txt"
  exit 0
}
trap requeue USR1 TERM
while true; do
  [[ -f $LANE/STOP ]] && { log "STOP file"; exit 0; }
  LEFT=$(( END - $(date +%s) ))
  (( LEFT < EST_S )) && { log "only ${LEFT}s left (< EST ${EST_S}s), exiting"; exit 0; }
  CUR=$(flock "$LANE/.lock" bash -c "l=\$(grep -v '^\s*\(#\|$\)' $LANE/queue.txt | head -1); [[ -n \"\$l\" ]] && { grep -vxF \"\$l\" $LANE/queue.txt > $LANE/.q.tmp; mv $LANE/.q.tmp $LANE/queue.txt; echo \"\$l\"; }")
  [[ -z "$CUR" ]] && { log "queue empty"; exit 0; }
  if [[ $CUR == CMD* ]]; then log "run $CUR"; bash -c "${CUR#CMD }"; CUR=""; continue; fi
  PRIV=$LANE/dbg_$SLURM_JOB_ID; mkdir -p "$PRIV"; touch "$PRIV/.lock"
  printf '%s\nCMD touch %s/STOP\n' "$CUR" "$PRIV" > "$PRIV/queue.txt"; rm -f "$PRIV/STOP"
  log "START $CUR (left ${LEFT}s)"
  SSM_LANE_DIR=$PRIV setsid bash "$REPO/editing/iclr_rebuttal/worker.sh" "dbg$SLURM_JOB_ID" & WPID=$!
  wait "$WPID"
  head -1 "$PRIV/done.txt" >> "$LANE/done.txt"; cat "$PRIV/events.txt" >> "$LANE/events.txt"; rm -f "$PRIV/done.txt" "$PRIV/events.txt"
  log "END $CUR"; CUR=""; WPID=""
done
