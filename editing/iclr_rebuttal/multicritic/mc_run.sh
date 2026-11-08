#!/bin/bash
# One multi-critic proof-of-concept run inside a Slurm step.
#   bash mc_run.sh <scenario> B   -> Qwen2.5-VL + InternVL3 in the loop (needs 2 GPUs: cuda:0 LTX+Qwen, cuda:1 InternVL3)
#   bash mc_run.sh <scenario> A   -> control: Qwen2.5-VL only, same wrapper/logging, same session (1 GPU)
#   bash mc_run.sh calib          -> InternVL3 differentiable-path calibration gate (1 GPU)
#   optional 3rd arg REP (2,3,...) -> repeat run of the same arm, dir suffix _r<REP> (runs are not bit-reproducible: GPU backward nondeterminism)
#   <scenario> may be ive:<item>:<cfg> (e.g. ive:short_0251:E) -> the IVEBench item with exactly the overrides of
#   ivebench/run_item.sh for that config (template man_climbs_rocks run_config); output dir ive_<item>_<cfg>
# Outputs: /scratch/amirrz/iclr_rebuttal/multicritic/<scenario>/<arm>/ (an existing finished run is renamed, never deleted)
S=$1; ARM=$2; REP=${3:-}
ROOT=/scratch/amirrz/iclr_rebuttal/multicritic
source /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh
cd /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/multicritic
if [[ $S == calib ]]; then
  python calibrate_internvl.py --out $ROOT/calibration 2>&1 | tee $ROOT/calibration.log; exit ${PIPESTATUS[0]}
fi
case $ARM in
  B) NAME=B_qwen25+internvl3; CR=qwen25,internvl3 ;;
  A) NAME=A_qwen25_only; CR=qwen25 ;;
  *) echo "arm A|B"; exit 2 ;;
esac
[[ -n $REP ]] && NAME=${NAME}_r$REP
EXTRA=()
if [[ $S == ive:* ]]; then
  IFS=: read -r _ N CFG <<< "$S"
  IR=/scratch/amirrz/iclr_rebuttal/ivebench
  case $CFG in
    A) SETS=(retake_start_frames=5 qwen_max_frames=8 qwen_sample_mode=linspace qwen_grad_accum_steps=3) ;;
    B) SETS=(retake_start_frames=5 qwen_max_frames=24 qwen_sample_mode=linspace qwen_grad_accum_steps=3) ;;
    C) SETS=(retake_start_frames=1 qwen_max_frames=24 qwen_sample_mode=normal qwen_grad_accum_steps=1) ;;
    E) SETS=(retake_start_frames=25 qwen_max_frames=8 qwen_sample_mode=linspace qwen_grad_accum_steps=3 enhance_prompt=false) ;;
    F) SETS=(retake_start_frames=25 qwen_max_frames=8 qwen_sample_mode=linspace qwen_grad_accum_steps=3 enhance_prompt=false lpips_weight=0.3 temporal_weight=0.2) ;;
    G) SETS=(retake_start_frames=25 qwen_max_frames=8 qwen_sample_mode=linspace qwen_grad_accum_steps=3 enhance_prompt=false); QPHRASE=1 ;;
    *) echo "cfg $CFG not supported here"; exit 2 ;;
  esac
  mapfile -t P < <(QPHRASE=${QPHRASE:-0} python - "$IR" "$N" <<'PY'
import json, os, sys
R, N = sys.argv[1], sys.argv[2]
it = next(x for x in json.load(open(f"{R}/subset_subject_motion_short.json")) if x["src_video_name"] == N + ".mp4")
meta = json.load(open(f"{R}/inputs/{N}.json"))
tp = it["target_prompt"].strip()
print(tp); print(it["source_prompt"].strip())
qt = it["target_phrase"].strip() if os.environ.get("QPHRASE") == "1" else tp
print(f'Does this video clearly show the action or state change described by the edit prompt: "{qt}"? Answer only \'yes\' or \'no\'.')
print(meta["n"])
PY
)
  [[ ${#P[@]} -eq 4 ]] || { echo "prompt lookup failed"; exit 2; }
  EXTRA=(--set "src_video=$IR/inputs/$N.mp4" --set "edit_prompt=${P[0]}" --set "static_prompt=${P[1]}"
         --set "qwen_motion_question=${P[2]}" --set "num_frames=${P[3]}" --set height=288 --set width=512 --set max_eval_frames=${P[3]})
  for kv in "${SETS[@]}"; do EXTRA+=(--set "$kv"); done
  SCEN=man_climbs_rocks; S=ive_${N}_${CFG}
else
  SCEN=$S
fi
OUT=$ROOT/$S/$NAME
[[ -e $OUT ]] && mv "$OUT" "$OUT.old_$(date +%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "start $(date -Is) host=$(hostname) gpus=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' ' ')" > "$OUT/run.log"
python run_multicritic.py --scenario "$SCEN" --output-dir "$OUT" --critics "$CR" "${EXTRA[@]}" >> "$OUT/run.log" 2>&1
rc=$?
echo "end $(date -Is) rc=$rc" >> "$OUT/run.log"
echo "MC_DONE $S $NAME rc=$rc" >> $ROOT/events.txt
exit $rc
