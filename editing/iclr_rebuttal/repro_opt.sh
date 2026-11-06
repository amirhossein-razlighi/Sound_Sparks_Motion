#!/bin/bash
# Full re-run of a PAPER optimisation with its exact recorded argv (run_config.json "argv"), then compare the final
# optimised video with the paper's ours video and the baseline with the paper baseline.
#   bash repro_opt.sh <scenario> april|head
# april: commit aacd7fd worktree (paper era); head: current repo script. Both with xformers (job_env.sh).
S=$1; CODE=$2
REPO=/home/amirrz/my_codes/Sound_Sparks_Motion; WT=/home/amirrz/my_codes/ssm_wt
cd $REPO/editing/iclr_rebuttal
PAPER=$(python3 -c "import sys; sys.path.insert(0,'.'); from common import paper_run_dir; print(paper_run_dir('$S'))" 2>/dev/null)
OUT=/scratch/amirrz/iclr_rebuttal/repro_opt/$S/$CODE
mkdir -p $OUT
case $CODE in april) SCRIPT=$WT/april/editing/optimize_qwen_vl.py ;; head) SCRIPT=$REPO/editing/optimize_qwen_vl.py ;; *) SCRIPT=$WT/$CODE/editing/optimize_qwen_vl.py ;; esac
mapfile -t A < <(python3 -c "
import json, os; a=json.load(open('$PAPER/run_config.json'))['argv']
a[0]='$SCRIPT'
i=a.index('--src-video'); a[i+1]='$REPO/input_videos/'+os.path.basename(a[i+1])
i=a.index('--output-dir'); a[i+1]='$OUT'
print('\n'.join(a))")
echo "ARGV: ${A[*]}"
cd $(dirname $(dirname $SCRIPT)) && python "${A[@]}"
python - <<EOF
import numpy as np, av
def load(p):
    c=av.open(p); f=np.stack([x.to_ndarray(format="rgb24") for x in c.decode(video=0)]).astype(np.float32)/255; c.close(); return f
for name, a, b in (("baseline", "$OUT/baseline_video.mp4", "$PAPER/baseline_video.mp4"),
                   ("optimized", "$OUT/mode_both/best_optimized_video_both.mp4", "$PAPER/mode_both/best_optimized_video_both.mp4")):
    x, y = load(a), load(b); n = min(len(x), len(y))
    m = float(np.abs(x[:n] - y[:n]).mean()); print(f"REPRO_OPT $S $CODE {name} MAD vs paper: {m:.4f}")
    open("$OUT/mad_" + name + ".txt", "w").write(f"{m}\n")
EOF
