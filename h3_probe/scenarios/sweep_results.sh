#!/bin/bash
# Poll until a NEW results.json appears under outputs/us_* (or no h3-full* jobs remain), print its summary, exit.
# State file lists results.json paths already reported.  Usage: sweep_results.sh [poll_seconds]
O=/scratch/amirrz/H3_exp/outputs; ST=/scratch/amirrz/H3_exp/outputs/.sweep_seen; touch $ST; P=${1:-300}
while :; do
  new=0
  for r in $O/us_*/results.json; do
    grep -qxF "$r" $ST && continue
    echo "$r" >> $ST; new=1; d=$(dirname $r); s=$(basename $d)
    echo "=== NEW RESULT: $s ==="
    awk -F, 'NR>1{printf "it%s y=%.3f p=%.2f b=%s; ",$1,$3,$9,$18}' $d/opt_log.csv; echo
    python3 -c "
import json; r=json.load(open('$r')); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items() if k in ('baseline_yes','final_yes','best_iter','frame_diff_vs_baseline','select_by')})"
  done
  [ $new = 1 ] && exit 0
  # failed jobs (OOM etc.) leave no results.json: report h3-full logs with a Traceback newer than the state file
  for f in $(find slurm_logs -name 'h3_full*.err' -newer $ST 2>/dev/null); do
    if grep -q "Traceback" $f && ! grep -q "JOB DONE" ${f%.err}.out 2>/dev/null; then
      j=$(basename $f .err); st=$(sacct -j ${j##*_} -n -X -o State 2>/dev/null | head -1 | tr -d ' ')
      case "$st" in FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL) grep -qxF "$f" $ST || { echo "$f" >> $ST; echo "=== JOB FAILED: $j ($st) ==="; grep -m1 -E "Error|error" $f | cut -c1-200; new=1; };; esac
    fi
  done
  [ $new = 1 ] && exit 0
  n=$(squeue -u amirrz -h -o "%j" | grep -c "h3-full"); [ "$n" = 0 ] && { echo "=== NO h3-full JOBS LEFT ==="; exit 0; }
  sleep $P
done
