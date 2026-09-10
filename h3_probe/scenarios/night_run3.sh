#!/bin/bash
# Overnight runner (login node, run under setsid nohup).  Works through results/overnight/queue.txt one GPU step at a
# time inside gpubase_interac allocations (2xH100, 3 h), re-acquiring an allocation whenever the current one is gone or
# too short for the next step.  Task lines (first non-done line is executed, then moved to done.txt with its status):
#   WAITLOG|<logfile>|<pattern>            wait until pattern appears in logfile (used to let a running step finish)
#   GEN|<prompts.json>|<slug;slug>|<candidates.json>   t2va sources (1 GPU) -> trim to 89 f -> source sheets
#   SCREEN|<candidates.json>|<slug;slug>   baseline render + critic (1 GPU) -> outputs/overnight/screen/
#   OPT|<slug>|<any|lin>[|<candidates.json>]   our method, both mode (2 GPUs) -> outputs/overnight/<slug>/opt -> package
# After SCREEN, slugs whose baseline yes_any < AUTO_ROOM are auto-appended as OPT tasks (unless results/overnight/no_auto exists).
# Shared-allocation mode (interac QOS allows ONE running job per user, so parallelism = one 4-GPU allocation, two runners):
#   master (default): asks for NGPU_ALLOC=4 x H100 / ALLOC_TIME=8h (falls back to 2 GPUs / 3 h if no free node within 600 s),
#     runs its own queue on 2 GPUs per step, publishes the allocation id in ALLOC_FILE, waits for the follower's step to end
#     before releasing an allocation, and keeps the allocation while FOLLOWER_QUEUE still has tasks.
#   follower (FOLLOW=1 RUNNER_TAG=B QUEUE_FILE=... LOG_FILE=... DONE_FILE=...): never allocates; runs its queue on the other
#     2 GPUs of the master's allocation once the master's log says it is staged. OPT tasks only (SCREEN shares one results file).
set -uo pipefail
cd /home/amirrz/my_codes/Sound_Sparks_Motion
R=/home/amirrz/my_codes/Sound_Sparks_Motion/h3_probe/scenarios; ON=h3_probe/results/overnight
Q=${QUEUE_FILE:-$ON/queue.txt}; D=${DONE_FILE:-$ON/done.txt}; L=${LOG_FILE:-$ON/night.log}; AF=${ALLOC_FILE:-$ON/alloc_id}
ML=${MASTER_LOG:-$ON/night.log}; FOLLOW=${FOLLOW:-0}; TAGR=${RUNNER_TAG:-A}; FQ=${FOLLOWER_QUEUE:-}
NG_WANT=${NGPU_ALLOC:-4}; T_WANT=${ALLOC_TIME:-8:00:00}; STEP_MEM=""
trap 'rm -f $ON/step_$TAGR.lock' EXIT
OUTB=/scratch/amirrz/H3_exp/outputs/overnight; mkdir -p $OUTB $OUTB/screen; touch $Q $D
PY=/scratch/amirrz/H3_exp/venv/bin/python
EXCL=rg31701,rg13401,rg21803,rg21802,rg31502,rg32202,rg21702,rg32102
AUTO_ROOM=${AUTO_ROOM:-0.5}
ALLOC=${ALLOC_ID:-}
[ -n "$ALLOC" ] && NEED_STAGE=1 || NEED_STAGE=0   # an inherited allocation must be staged on first use
ALLCANDS="$R/candidates_r1.json:$R/candidates_r2.json:$R/candidates_r3.json:$R/candidates_r4.json:$R/candidates_r5.json:$R/candidates_r6.json:$R/candidates_bank.json"
BASEENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/scratch/amirrz/.cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
secs_left(){ local t; t=$(squeue -j "$1" -h -o %L 2>/dev/null | head -1); [ -z "$t" ] && { echo 0; return; }
  local d=0 h=0 m=0 s=0; if [[ $t == *-* ]]; then d=${t%%-*}; t=${t#*-}; fi; IFS=: read -r a b c <<< "$t"
  if [ -n "${c:-}" ]; then h=$a; m=$b; s=$c; elif [ -n "${b:-}" ]; then m=$a; s=$b; else s=$a; fi; echo $((10#$d*86400+10#$h*3600+10#$m*60+10#$s)); }
LOCAL=""; STEPENV=""
stage_models(){ # copy H3 ckpt + Qwen to the node-local NVMe once per allocation: Lustre mmap loading stalled 15-25 min per process
  LOCAL=/localscratch/amirrz.$ALLOC.0/models; log "staging models to $LOCAL ..."
  timeout 2400 srun --jobid=$ALLOC --ntasks=1 --cpus-per-task=8 --export=ALL bash -c "mkdir -p $LOCAL && rsync -a --exclude docs --exclude scripts /scratch/amirrz/H3_exp/ckpt/ $LOCAL/ckpt/ && rsync -a /project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct/ $LOCAL/qwen/ && sed -i \"s#/scratch/amirrz/H3_exp/ckpt#$LOCAL/ckpt#g\" $LOCAL/ckpt/modular_model_index.json $LOCAL/ckpt/model_index.json && grep -c $LOCAL $LOCAL/ckpt/modular_model_index.json && du -sh $LOCAL/ckpt $LOCAL/qwen" > $ON/logs/stage_$ALLOC.log 2>&1
  if [ $? -eq 0 ]; then STEPENV="H3_CKPT=$LOCAL/ckpt H3_QWEN=$LOCAL/qwen"; log "staged: $(tail -2 $ON/logs/stage_$ALLOC.log | tr '\n' ' ')"; else STEPENV=""; log "staging FAILED (see $ON/logs/stage_$ALLOC.log) - steps will load from Lustre"; fi; }
alloc_ok(){ [ -n "$ALLOC" ] && [ "$(squeue -j $ALLOC -h -o %T 2>/dev/null)" = "RUNNING" ]; }
alloc_gpus(){ squeue -j "$ALLOC" -h -o %b 2>/dev/null | grep -oE ':[0-9]+$' | tr -d ':'; }
set_step_mem(){ if [ "$(alloc_gpus)" = "4" ]; then STEP_MEM=200G; else STEP_MEM=""; fi; }   # two 2-GPU steps must share a 4-GPU job's memory
other_busy(){ ls $ON/step_*.lock 2>/dev/null | grep -v "step_$TAGR.lock" | grep -q .; }
follow_alloc(){ # follower: wait until the master's allocation is running, staged, 4-GPU, long enough and not being released
  local need=$1 n=0
  while :; do ALLOC=$(cat $AF 2>/dev/null)
    if alloc_ok && [ ! -f $ON/release_pending ] && [ "$(secs_left $ALLOC)" -ge "$need" ] && [ "$(alloc_gpus)" = "4" ] && grep -q "staged:.*amirrz.$ALLOC.0" $ML 2>/dev/null; then
      LOCAL=/localscratch/amirrz.$ALLOC.0/models; STEPENV="H3_CKPT=$LOCAL/ckpt H3_QWEN=$LOCAL/qwen"; set_step_mem; return 0; fi
    [ $((n % 10)) -eq 0 ] && log "follower waiting for a shared 4-GPU allocation (alloc=${ALLOC:-none} running=$(alloc_ok && echo y || echo n) gpus=$(alloc_gpus) left=$(secs_left "${ALLOC:-0}")s release_pending=$([ -f $ON/release_pending ] && echo y || echo n))"
    n=$((n+1)); sleep 60; done; }
ensure_alloc(){ # $1 = seconds needed
  local need=$1
  if [ "$FOLLOW" = 1 ]; then follow_alloc $need; return 0; fi
  if alloc_ok; then local left; left=$(secs_left $ALLOC); if [ "$left" -ge "$need" ]; then [ "$NEED_STAGE" = 1 ] && { stage_models; NEED_STAGE=0; }; set_step_mem; return 0; fi
    log "allocation $ALLOC has only ${left}s left (< ${need}s) -> releasing"; touch $ON/release_pending
    while other_busy; do log "waiting for the follower's step to finish before releasing $ALLOC"; sleep 60; done
    scancel $ALLOC; ALLOC=""; fi
  touch $ON/release_pending
  while :; do local out=""; ALLOC=""
    if [ "$NG_WANT" -gt 2 ]; then log "requesting allocation (${NG_WANT}xH100, $T_WANT, up to 600s) ..."
      out=$(salloc -p gpubase_interac --account=def-amahdavi --gres=gpu:h100:$NG_WANT --cpus-per-task=$((4*NG_WANT)) --mem=$((110*NG_WANT))G --time=$T_WANT --exclude=$EXCL --job-name=h3-night --no-shell --immediate=600 2>&1)
      ALLOC=$(echo "$out" | grep -o "Granted job allocation [0-9]*" | awk '{print $4}')
      [ -z "$ALLOC" ] && log "no ${NG_WANT}-GPU node within 600s ($(echo "$out" | tail -1 | cut -c1-120)) -> falling back to 2xH100, 3h"; fi
    if [ -z "$ALLOC" ]; then log "requesting allocation (2xH100, 3h) ..."
      out=$(salloc -p gpubase_interac --account=def-amahdavi --gres=gpu:h100:2 --cpus-per-task=8 --mem=220G --time=3:00:00 --exclude=$EXCL --job-name=h3-night --no-shell 2>&1)
      ALLOC=$(echo "$out" | grep -o "Granted job allocation [0-9]*" | awk '{print $4}'); fi
    if [ -n "$ALLOC" ]; then for i in $(seq 1 60); do alloc_ok && break; sleep 10; done
      if alloc_ok; then log "allocation $ALLOC RUNNING on $(squeue -j $ALLOC -h -o %N), $(secs_left $ALLOC)s, gpus=$(alloc_gpus)"; echo $ALLOC > $AF; stage_models; set_step_mem; rm -f $ON/release_pending; return 0; fi; fi
    log "salloc failed/timed out: $(echo "$out" | tail -1 | cut -c1-120)"; sleep 120; done; }
run_step(){ # $1 ngpu $2 timeout-seconds $3 logfile $4 command   (sequential; NO --overlap: with it every step sees one GPU)
  local ng=$1 to=$2 lg=$3 cmd=$4; log "step -> $lg (gpus=$ng, timeout=${to}s, alloc=$ALLOC${STEP_MEM:+, mem=$STEP_MEM})"; touch $ON/step_$TAGR.lock
  timeout $to srun --jobid=$ALLOC --ntasks=1 --cpus-per-task=8 --gres=gpu:h100:$ng ${STEP_MEM:+--mem=$STEP_MEM} --export=ALL bash -c "cd /home/amirrz/my_codes/Sound_Sparks_Motion; export $BASEENV $STEPENV; $cmd" > $lg 2>&1; local rc=$?
  rm -f $ON/step_$TAGR.lock; log "step <- $lg rc=$rc"; return $rc; }
bank_feed(){ # append GEN+SCREEN for the next 4 unused bank scenarios
  local used; used=$(cat $Q $D 2>/dev/null); local pick=()
  for s in $($PY -c "import json;print(' '.join(k for k in json.load(open('$R/candidates_bank.json')) if not k.startswith('_')))"); do
    echo "$used" | grep -q "$s" || pick+=("$s"); [ ${#pick[@]} -ge 4 ] && break; done
  [ ${#pick[@]} -eq 0 ] && return 1
  local sl; sl=$(IFS=';'; echo "${pick[*]}"); echo "GEN|$R/gen_inputs_bank.json|$sl|$R/candidates_bank.json" >> $Q; echo "SCREEN|$R/candidates_bank.json|$sl" >> $Q
  log "bank feed: $sl"; return 0; }
count_wins(){ $PY - <<'PY'
import json,glob
n=0
for f in glob.glob("h3_probe/results/user_study_candidates/*/verdict.json"):
    v=json.load(open(f))
    if v.get("auto_verdict")=="strong_win": n+=1
print(n)
PY
}
idle_since=0
while :; do
  task=$(grep -v '^\s*$' $Q | head -1)
  if [ -z "$task" ]; then
    if [ "$FOLLOW" != 1 ]; then wins=$(count_wins); if [ "$wins" -lt 5 ] && [ ! -f $ON/no_bank ]; then bank_feed && continue; fi; fi
    if [ "$FOLLOW" != 1 ] && alloc_ok && ! other_busy && ! { [ -n "$FQ" ] && grep -q '[^[:space:]]' $FQ 2>/dev/null; }; then
      idle_since=$((idle_since+60)); [ $idle_since -ge 900 ] && { log "idle 15 min with empty queues -> releasing $ALLOC"; scancel $ALLOC; ALLOC=""; idle_since=0; }; fi
    sleep 60; continue; fi
  idle_since=0; IFS='|' read -r T A1 A2 A3 A4 <<< "$task"
  status=ok
  case "$T" in
    WAITLOG) log "waiting for '$A2' in $A1"; while ! grep -q "$A2" "$A1" 2>/dev/null; do sleep 60; done; log "found '$A2' in $A1";;
    GEN) ensure_alloc 2400; sl=$(echo "$A1" | sed 's/.*//'); run_step 1 2100 $ON/logs/gen_$(date +%H%M%S).log "export PROMPTS_JSON=$A1 ONLY=$(echo $A2 | tr ';' ',') OUT_DIR=/scratch/amirrz/H3_exp/inputs/gen GEN_H=320 GEN_W=512 GEN_FRAMES=124 GEN_SEED=${GEN_SEED:-7}; $PY h3_probe/t2va_sounds.py" || status=fail
         OPENBLAS_NUM_THREADS=1 $PY $R/trim_sources.py $A3 >> $L 2>&1
         for s in $(echo $A2 | tr ';' ' '); do f=/scratch/amirrz/H3_exp/inputs/gen/${s}__t2va.mp4; [ -f $f ] && OPENBLAS_NUM_THREADS=1 W=220 N=12 bash $R/ab_sheet.sh /scratch/amirrz/H3_exp/inputs/gen/${s}__src_sheet.jpg $f >/dev/null 2>&1 && log "source sheet: /scratch/amirrz/H3_exp/inputs/gen/${s}__src_sheet.jpg"; done;;
    SCREEN) n=$(echo "$A2" | tr ';' '\n' | grep -c .); ensure_alloc $((1200+1000*n)); run_step 1 $((900+1000*n)) $ON/logs/screen_$(date +%H%M%S).log "export CANDS=$A1 ONLY='$A2' OUT_DIR=$OUTB/screen; $PY h3_probe/scenarios/screen.py" || status=fail
         for s in $(echo $A2 | tr ';' ' '); do r=$($PY -c "import json;r=json.load(open('$OUTB/screen/screen_results.json')).get('$s',{});q=r.get('q_scenario',{});print('%.3f %.3f'%(q.get('yes_lin',-1),q.get('yes_any',-1)) if q else 'err')" 2>/dev/null)
           log "SCREENED $s baseline lin/any = $r  sheet: $OUTB/screen/${s}_sheet.png"
           any=$(echo $r | awk '{print $2}'); if [ "$r" != "err" ] && [ -n "$any" ] && [ ! -f $ON/no_auto ] && awk "BEGIN{exit !($any < $AUTO_ROOM)}"; then echo "OPT|$s|any|$A1" >> $Q; log "auto: room for $s (any=$any) -> OPT queued"; else log "auto: no room / no auto for $s (any=$any)"; fi; done;;
    OPT) ensure_alloc 5700; s=$A1; obj=${A2:-any}; cj=${A3:-}; extra=${A4:-}; tag=$(echo "$extra" | grep -oE 'TAG=[A-Za-z0-9_]+' | cut -d= -f2); od=$OUTB/$s/opt${tag:+_$tag}; pk=${s}${tag:+_$tag}; mkdir -p $od
         # 5th field = extra env overriding the defaults, e.g. "TEMPORAL_W=0.5 LPIPS_W=1.5 TAG=anchor" (TAG -> separate out dir + package name)
         run_step 2 5400 $ON/logs/opt_${s}.log "export SCEN_JSON=$ALLCANDS QWEN_IMG=224 FP32_HEAD=1 QWEN_ACCUM=1 GRAD_STEPS=2 ITERS=16 EARLY=${EARLY_STOP:-14} LPIPS_W=1.0 TEMPORAL_W=0.3 AUDIO_REG=0 TEXT_REG=0 SAVE_PREVIEWS=1 DECODE_GRAD_FRAC=0.5 OPTIM=ngd NGD_ETA=0.03 NGD_MOM=0.3 ATTN_VIS=0 GRAD_CHECK=0 OPT_MODE=both PERC_MAX=0.25 CRITIC_OBJ=$obj SELECT_BY=$obj SLUG=$s OUT_DIR=$od $extra; nvidia-smi -L; ([ -f $od/capture.pt ] || { [ -f $OUTB/$s/opt/capture.pt ] && ln -s $OUTB/$s/opt/capture.pt $od/capture.pt; } || $PY h3_probe/h3_full_method.py --phase a) && $PY h3_probe/h3_full_method.py --phase b; echo PHASES_EXIT \$?" || status=fail
         if [ -f $od/results.json ]; then OPENBLAS_NUM_THREADS=1 $PY $R/package_candidate.py $pk $od >> $L 2>&1; log "OPT DONE $pk: $($PY -c "import json;r=json.load(open('$od/results.json'));print('baseline any=%.3f -> best iter %s, final any=%.3f, fd=%.3f'%(r.get('baseline_yes_any',-1),r.get('best_iter'),r.get('final_yes_any',-1),r.get('frame_diff_vs_baseline',-1)))")"; else log "OPT FAILED $s (no results.json) - see $ON/logs/opt_${s}.log"; status=fail; fi;;
    ABL) ensure_alloc 5700; s=$A1; mode=${A2:-audio}; env=$(ONLY=$s MODES=$mode PRINT_ENV=1 $PY $R/ablate.py 2>/dev/null | grep "OPT_MODE=$mode" | head -1)
         if [ -z "$env" ]; then log "ABL: no env for $s/$mode"; status=fail; else od=$(echo "$env" | grep -oE "OUT_DIR=[^ ]+" | cut -d= -f2)
           run_step 2 5400 $ON/logs/abl_${s}_${mode}.log "export $env; nvidia-smi -L; $PY h3_probe/h3_full_method.py --phase b; echo PHASES_EXIT \$?" || status=fail
           [ -f $od/results.json ] && { OPENBLAS_NUM_THREADS=1 $PY $R/package_ablation.py >> $L 2>&1; log "ABL DONE $s/$mode"; } || { log "ABL FAILED $s/$mode"; status=fail; }; fi;;
    *) log "unknown task: $task"; status=skip;;
  esac
  # pop the task
  $PY - "$task" "$status" "$Q" "$D" <<'PY'
import sys
t,st,q,d=sys.argv[1:5]; lines=open(q).read().split("\n")
for i,l in enumerate(lines):
    if l==t: del lines[i]; break
open(q,"w").write("\n".join(lines)); open(d,"a").write(f"{st}|{t}\n")
PY
  log "task done ($status): $task"
done
