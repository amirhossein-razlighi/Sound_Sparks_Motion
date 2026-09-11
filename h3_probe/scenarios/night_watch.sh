#!/bin/bash
# Wake the coordinator when the overnight runner logs something that needs judgement or attention.
L=${LOG_FILE:-h3_probe/results/overnight/night.log}; ST=$(dirname $L)/.watch_seen_$(basename $L .log); [ -f $ST ] || wc -l < $L > $ST; P=${1:-120}
PAT='SCREENED|OPT DONE|OPT FAILED|source sheet|salloc failed|allocation .* RUNNING|releasing|bank feed|unknown task|rc=[1-9]'
while :; do
  seen=$(cat $ST 2>/dev/null); seen=${seen:-0}; total=$(wc -l < $L)
  if [ "$total" -gt "$seen" ]; then new=$(tail -n +$((seen+1)) $L | grep -E "$PAT"); echo $total > $ST
    if [ -n "$new" ]; then echo "$new"; exit 0; fi; fi
  if [ "${RUNNER_CHECK:-1}" = 1 ]; then pgrep -f 'night_run[23]?.sh' > /dev/null || { echo "RUNNER NOT RUNNING"; exit 2; }
  else [ $(( $(date +%s) - $(stat -c %Y $L) )) -gt 5400 ] && { echo "LOG STALE 90 min - runner may be dead"; exit 2; }; fi  # RUNNER_CHECK=0 when the runner is on another login node (ssh needs MFA now)
  sleep $P
done
