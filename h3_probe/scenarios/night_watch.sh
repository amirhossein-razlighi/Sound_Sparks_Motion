#!/bin/bash
# Wake the coordinator when the overnight runner logs something that needs judgement or attention.
L=h3_probe/results/overnight/night.log; ST=h3_probe/results/overnight/.watch_seen; touch $ST; P=${1:-120}
PAT='SCREENED|OPT DONE|OPT FAILED|source sheet|salloc failed|allocation .* RUNNING|releasing|bank feed|unknown task|rc=[1-9]'
while :; do
  seen=$(cat $ST 2>/dev/null); seen=${seen:-0}; total=$(wc -l < $L)
  if [ "$total" -gt "$seen" ]; then new=$(tail -n +$((seen+1)) $L | grep -E "$PAT"); echo $total > $ST
    if [ -n "$new" ]; then echo "$new"; exit 0; fi; fi
  pgrep -f night_run.sh > /dev/null || { echo "RUNNER NOT RUNNING"; exit 2; }
  sleep $P
done
