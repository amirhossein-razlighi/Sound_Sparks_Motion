#!/bin/bash
# Multi-critic queue worker for ONE 2-GPU slot (a Slurm step / batch job with exactly 2 visible GPUs).
#   queues (one task per line, popped under flock):
#     $ROOT/q2.txt : "<scenario> B"          -> needs both GPUs; only popped once $ROOT/calibration/PASS exists
#     $ROOT/q1.txt : "calib" | "<scenario> A" -> one GPU each; two run side by side when q2 is empty/blocked
#   $ROOT/STOP makes the worker exit after its current task; it also exits after IDLE_MAX x 30 s with nothing to do.
ROOT=/scratch/amirrz/iclr_rebuttal/multicritic
H=/home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/multicritic
W=${1:-w$$}
ev() { echo "$(date '+%m-%d %H:%M:%S') $W $*" >> $ROOT/events.txt; }
pop() {  # pop first line of $1
  ( flock 9; l=$(head -1 "$1" 2>/dev/null); [[ -n $l ]] && sed -i '1d' "$1"; echo "$l" ) 9>$ROOT/.lock
}
run1() {  # task, gpu index
  local t=$1 g=$2
  ev "START [$t] gpu$g"
  if [[ $t == calib ]]; then
    CUDA_VISIBLE_DEVICES=$g bash $H/mc_run.sh calib && touch $ROOT/calibration/PASS
  else
    CUDA_VISIBLE_DEVICES=$g bash $H/mc_run.sh $t
  fi
  ev "END [$t] rc=$?"
}
if [[ -n $CUDA_VISIBLE_DEVICES ]]; then G=(${CUDA_VISIBLE_DEVICES//,/ }); else G=($(nvidia-smi --query-gpu=index --format=csv,noheader)); fi
ev "up host=$(hostname) job=${SLURM_JOB_ID:-} step=${SLURM_STEP_ID:-} gpus=${G[*]}"
idle=0
while [[ ! -e $ROOT/STOP ]]; do
  t=""
  [[ -e $ROOT/calibration/PASS ]] && t=$(pop $ROOT/q2.txt)
  if [[ -n $t ]]; then
    idle=0; ev "START [$t] gpus ${G[*]}"; bash $H/mc_run.sh $t; ev "END [$t] rc=$?"; continue
  fi
  a=$(pop $ROOT/q1.txt)
  if [[ -n $a ]]; then
    idle=0
    b=""; [[ ${#G[@]} -ge 2 ]] && b=$(pop $ROOT/q1.txt)
    run1 "$a" ${G[0]} &
    [[ -n $b ]] && { run1 "$b" ${G[1]} & }
    wait
    continue
  fi
  idle=$((idle+1)); [[ $idle -gt ${IDLE_MAX:-240} ]] && break
  sleep 30
done
ev "exit"
