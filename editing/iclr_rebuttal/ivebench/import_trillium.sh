#!/bin/bash
# Unpack every results_trillium/ivebench/<name>__<cfg>.tgz.b64 into $IVE_ROOT/runs/<name>/<cfg>/ (skip ones already there).
R=${IVE_ROOT:-/scratch/amirrz/iclr_rebuttal/ivebench}; G=/home/amirrz/my_codes/Sound_Sparks_Motion/results_trillium/ivebench
for f in $G/*.tgz.b64; do b=$(basename $f .tgz.b64); n=${b%%__*}; c=${b#*__}; D=$R/runs/$n/$c
  [[ -f $D/best_optimized_video_both.mp4 ]] && continue
  mkdir -p $D && base64 -d $f | tar -xzf - -C $D && echo "trillium" > $D/origin.txt && echo "imported $n $c"
done
