#!/bin/bash
# Lane worker: runs inside an interactive allocation step with ONE GPU and drains the shared queue.
#   srun --jobid=$ALLOC --gres=gpu:h100:1 --cpus-per-task=6 --mem=60G --exact -n1 bash worker.sh <wid>
# Queue line formats (lane/queue.txt; first line popped under flock):
#   BON  <scenario> [seeds] [out_root]
#   OPT  <exp> <scenario> <tag> [key=value ...]
#   P2   <manifest names (comma sep)> <models (comma sep)>
#   XFER <config.yaml> <out_dir>
#   CMD  <arbitrary shell>
WID=${1:-0}
# Defaults = Rorqual. Trillium: export SSM_JOB_ENV, SSM_OUT_ROOT (and SSM_LANE_DIR) before launching.
source "${SSM_JOB_ENV:-/home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal/job_env.sh}"
R=${SSM_OUT_ROOT:-/scratch/amirrz/iclr_rebuttal}
L=${SSM_LANE_DIR:-$R/lane}
LOC=/localscratch/amirrz.${SLURM_JOB_ID}.0/models
if [[ -f "$LOC/.staged" ]]; then
  export SSM_CKPT_PATH=$LOC/ltx-2.3-22b-dev.safetensors SSM_GEMMA_ROOT=$LOC/gemma-3-12b-it-qat-q4_0-unquantized SSM_QWEN_ROOT=$LOC/Qwen2.5-VL-7B-Instruct
  export HELDOUT_MODEL_ROOT=$LOC CKPT_ROOT=$LOC GEMMA_ROOT=$LOC/gemma-3-12b-it-qat-q4_0-unquantized QWEN_ROOT=$LOC/Qwen2.5-VL-7B-Instruct
fi
idle=0
while true; do
  [[ -f $L/STOP ]] && { echo "[w$WID] STOP file"; exit 0; }
  LINE=$(flock $L/.lock bash -c "l=\$(grep -v '^\s*\(#\|$\)' $L/queue.txt | head -1); [[ -n \"\$l\" ]] && { grep -vxF \"\$l\" $L/queue.txt > $L/.q.tmp; mv $L/.q.tmp $L/queue.txt; echo \"\$l\"; }")
  if [[ -z "$LINE" ]]; then idle=$((idle+1)); [[ $idle -gt ${IDLE_MAX:-480} ]] && { echo "[w$WID] idle exit"; exit 0; }; sleep 30; continue; fi
  idle=0
  read -r TYPE A1 A2 A3 REST <<< "$LINE"
  NAME=$(echo "${A1}_${A2}_${A3}" | tr -c 'A-Za-z0-9_.\n-' '_' | cut -c1-120)
  LOG=$R/logs/lane_${TYPE}_${NAME}.log
  T0=$(date +%s); echo "$(date -Is) w$WID START $LINE" >> $L/events.txt
  (
    case $TYPE in
      BON)  python best_of_n.py --scenario "$A1" ${A2:+--seeds "$A2"} ${A3:+--out-root "$A3"} ;;
      OPT)  OUT=$R/$A1/$A2/$A3; SETS=(); MODE=both; for kv in $REST; do SETS+=(--set "$kv"); [[ $kv == opt_mode=* ]] && MODE=${kv#opt_mode=}; done
            if [[ -f "$OUT/timing.json" && -f "$OUT/mode_$MODE/best_optimized_video_$MODE.mp4" ]]; then echo already done; exit 0; fi
            python run_opt.py --scenario "$A2" --output-dir "$OUT" "${SETS[@]}"; rc=$?
            [[ $rc -eq 0 ]] && { python ../scripts/eval_metrics.py --output-dir "$OUT" --mode $MODE --device cuda --raft-disable-cudnn || echo "eval_metrics rc=$?"; }
            exit $rc ;;
      P2)   MD=$R/manifests; OUT=$R/heldout; M=(); for m in ${A1//,/ }; do M+=(--manifest $MD/$m.csv); done
            python loop_rescore.py "${M[@]}" --out $OUT || echo "loop_rescore rc=$?"
            for m in ${A1//,/ }; do python heldout_score.py --manifest $MD/$m.csv --out $OUT --models "${A2:-qwen25,qwen3vl,internvl3}" || exit $?; done ;;
      XFER) [[ -f "$A2/transfer_both.mp4" ]] && { echo already done; exit 0; }; OUTPUT_DIR="$A2" bash ../scripts/transfer.sh "$A1" ;;
      CMD)  bash -c "${LINE#CMD }" ;;
      *)    echo "unknown task $TYPE"; exit 2 ;;
    esac
  ) > "$LOG" 2>&1
  RC=$?
  echo "$(date -Is) w$WID END rc=$RC $(( $(date +%s)-T0 ))s $LINE" >> $L/events.txt
  echo "$RC|$LINE" >> $L/done.txt
done
