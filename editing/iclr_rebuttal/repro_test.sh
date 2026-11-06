#!/bin/bash
# Reproduce the paper's dog_yawning baseline (seed 42) with different code versions; baseline only (--iterations 0).
#   bash repro_test.sh april|main|head|head_runsh
# april     : commit aacd7fd (paper-run era) with the paper run's exact argv (from its run_config.json)
# main      : branch main (submitted version) via its own parse_config on editing/configs/dog_yawning.yaml
# head      : current HEAD via parse_config on editing/configs/dog_yawning.yaml (the normal run.sh path)
V=$1
# Variants ending in _xf put the April-era xformers 0.0.33 (uv.lock extra "xformers") on PYTHONPATH.
if [[ $V == *_xf ]]; then export PYTHONPATH=/home/amirrz/my_codes/xformers_0033/site${PYTHONPATH:+:$PYTHONPATH}; fi
BASEV=${V%_xf}
REPO=/home/amirrz/my_codes/Sound_Sparks_Motion
WT=/home/amirrz/my_codes/ssm_wt
OUT=/scratch/amirrz/iclr_rebuttal/repro/$V
PAPER=/project/def-amahdavi/amirrz/results/a_dog_yawning/linspace_w_lpips/both
mkdir -p $OUT
python -c "import ltx_core.model.transformer.attention as A; print('XFORMERS_ACTIVE', A.memory_efficient_attention is not None)"
case $BASEV in
  april)
    mapfile -t A < <(python3 -c "
import json; a=json.load(open('$PAPER/run_config.json'))['argv']
a[0]='$WT/april/editing/optimize_qwen_vl.py'
i=a.index('--src-video'); a[i+1]='$REPO/input_videos/dog_sitting_on_chair.mp4'
i=a.index('--output-dir'); a[i+1]='$OUT'
print('\n'.join(a))")
    cd $WT/april && python "${A[@]}" --iterations 0 ;;
  main|head)
    ROOT=$([ $BASEV = main ] && echo $WT/main || echo $REPO)
    A=(); while IFS= read -r -d '' t; do A+=("$t"); done < <(python3 $ROOT/editing/scripts/utils/parse_config.py $ROOT/editing/configs/dog_yawning.yaml $REPO)
    cd $ROOT && python $ROOT/editing/optimize_qwen_vl.py "${A[@]}" --iterations 0 --output-dir $OUT ;;
esac
python - <<EOF
import numpy as np, av
def load(p):
    c=av.open(p); f=np.stack([x.to_ndarray(format="rgb24") for x in c.decode(video=0)]).astype(np.float32)/255; c.close(); return f
a=load("$OUT/baseline_video.mp4"); b=load("$PAPER/baseline_video.mp4"); n=min(len(a),len(b))
m=float(np.abs(a[:n]-b[:n]).mean()); print("REPRO $V MAD vs paper baseline:", m, "frames", len(a), len(b))
open("$OUT/mad.txt","w").write(f"{m}\n")
EOF
