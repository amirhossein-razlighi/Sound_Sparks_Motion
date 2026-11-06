#!/bin/bash
# One IVEBench item x one config. Runs inside a lane worker (env sourced, cwd editing/iclr_rebuttal).
#   bash ivebench/run_item.sh <name e.g. short_0009> <cfg A|B|C>    or    bash ivebench/run_item.sh short_0009:A
# Template = the paper's man_climbs_rocks run (K5, 24 linspace frames, 3 critic passes, lpips 0.1, temp 0.05, 30 it,
# lr 0.005, audio+text) with these per-config overrides:
#   A: K=5, critic 8 frames linspace, 3 passes   (paper red_car / robot setting)
#   B: K=5, critic 24 frames linspace, 3 passes  (paper man_climbs setting)
#   C: K=1, critic 24 frames normal, 1 pass      (paper goldfish setting; frees the clip from frame 1)
#   D: = A with Gemma prompt enhancement OFF       (IVEBench prompts are already long; enhancement may add shot changes)
#   E: = D with K=25 preserved frames              (paper monkey / turtle prefix length)
# LTX prompt and critic question use IVEBench's target_prompt (the paper uses one string for both).
set -o pipefail
N=$1; CFG=$2
[[ $N == *:* ]] && { CFG=${N#*:}; N=${N%%:*}; }
R=${IVE_ROOT:-/scratch/amirrz/iclr_rebuttal/ivebench}
DST=$R/runs/$N/$CFG
[[ -f $DST/best_optimized_video_both.mp4 ]] && { echo "already done $N $CFG"; exit 0; }
LOCAL=${IVE_LOCAL:-${SLURM_TMPDIR:-/localscratch/amirrz.${SLURM_JOB_ID}.0}}/ive/${N}__$CFG
rm -rf "$LOCAL"; mkdir -p "$LOCAL" "$DST"
case $CFG in
  A) SETS=(retake_start_frames=5 qwen_max_frames=8 qwen_sample_mode=linspace qwen_grad_accum_steps=3) ;;
  B) SETS=(retake_start_frames=5 qwen_max_frames=24 qwen_sample_mode=linspace qwen_grad_accum_steps=3) ;;
  C) SETS=(retake_start_frames=1 qwen_max_frames=24 qwen_sample_mode=normal qwen_grad_accum_steps=1) ;;
  D) SETS=(retake_start_frames=5 qwen_max_frames=8 qwen_sample_mode=linspace qwen_grad_accum_steps=3 enhance_prompt=false) ;;
  E) SETS=(retake_start_frames=25 qwen_max_frames=8 qwen_sample_mode=linspace qwen_grad_accum_steps=3 enhance_prompt=false) ;;
  *) echo "bad cfg $CFG"; exit 2 ;;
esac
mapfile -t P < <(python - "$R" "$N" <<'PY'
import json, sys
R, N = sys.argv[1], sys.argv[2]
it = next(x for x in json.load(open(f"{R}/subset_subject_motion_short.json")) if x["src_video_name"] == N + ".mp4")
meta = json.load(open(f"{R}/inputs/{N}.json"))
tp = it["target_prompt"].strip()
print(tp); print(it["source_prompt"].strip())
print(f'Does this video clearly show the action or state change described by the edit prompt: "{tp}"? Answer only \'yes\' or \'no\'.')
print(meta["n"])
PY
)
[[ ${#P[@]} -eq 4 ]] || { echo "prompt lookup failed"; exit 2; }
ARGS=(--scenario man_climbs_rocks --output-dir "$LOCAL"
      --set "src_video=$R/inputs/$N.mp4" --set "edit_prompt=${P[0]}" --set "static_prompt=${P[1]}"
      --set "qwen_motion_question=${P[2]}" --set "num_frames=${P[3]}" --set height=288 --set width=512 --set max_eval_frames=${P[3]})
for kv in "${SETS[@]}"; do ARGS+=(--set "$kv"); done
python run_opt.py "${ARGS[@]}"; rc=$?
# keep everything (one tar) + the few files needed downstream (inode budget on /scratch)
tar -C "$(dirname "$LOCAL")" -cf "$DST/run_full.tar" "$(basename "$LOCAL")"
for f in baseline_video.mp4 retake_input_prepared.mp4 run_config.json timing.json iclr_overrides.json OOM.json mode_both/best_optimized_video_both.mp4 mode_both/optimization_log_qwen_both.csv; do
  [[ -f $LOCAL/$f ]] && cp "$LOCAL/$f" "$DST/"
done
echo "rc=$rc" > "$DST/rc.txt"; rm -rf "$LOCAL"
exit $rc
