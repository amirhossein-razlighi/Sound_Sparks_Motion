#!/bin/bash
# TRILLIUM login-node feeder: debug QoS allows 1 job/user (running OR pending), so a job cannot submit its successor.
# This loop keeps exactly one trillium_debug.sh job alive while the shared lane queue has work.
#   nohup bash trillium_debug_feeder.sh <lane_dir> <EST_S> > /scratch/amirrz/iclr_rebuttal/logs/feeder.log 2>&1 &
LANE=${1:?lane dir}; EST_S=${2:-5400}
REPO=/home/amirrz/my_codes/Sound_Sparks_Motion
cd /scratch/amirrz/iclr_rebuttal || exit 1
while true; do
  [[ -f $LANE/STOP || -f $LANE/FEEDER_STOP ]] && { echo "$(date -Is) stop file, feeder exits"; exit 0; }
  if ! grep -v '^\s*\(#\|$\)' "$LANE/queue.txt" | grep -qv '^CMD touch .*STOP$'; then echo "$(date -Is) no tasks left, feeder exits"; exit 0; fi
  if [[ -z "$(squeue -h -u "$USER" -p debug -o %i)" ]]; then
    echo "$(date -Is) submit: $(sbatch "$REPO/editing/iclr_rebuttal/trillium_debug.sh" "$LANE" "$EST_S" 2>&1 2>&1 | grep -E "Submitted|ERROR" | head -1)"
  fi
  sleep 60
done
