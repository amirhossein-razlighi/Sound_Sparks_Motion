#!/bin/bash
# One multi-critic proof-of-concept run inside a Slurm step.
#   bash mc_run.sh <scenario> B   -> Qwen2.5-VL + InternVL3 in the loop (needs 2 GPUs: cuda:0 LTX+Qwen, cuda:1 InternVL3)
#   bash mc_run.sh <scenario> A   -> control: Qwen2.5-VL only, same wrapper/logging, same session (1 GPU)
#   bash mc_run.sh calib          -> InternVL3 differentiable-path calibration gate (1 GPU)
# Outputs: /scratch/amirrz/iclr_rebuttal/multicritic/<scenario>/<arm>/ (an existing finished run is renamed, never deleted)
S=$1; ARM=$2
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
OUT=$ROOT/$S/$NAME
[[ -e $OUT ]] && mv "$OUT" "$OUT.old_$(date +%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "start $(date -Is) host=$(hostname) gpus=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' ' ')" > "$OUT/run.log"
python run_multicritic.py --scenario "$S" --output-dir "$OUT" --critics "$CR" >> "$OUT/run.log" 2>&1
rc=$?
echo "end $(date -Is) rc=$rc" >> "$OUT/run.log"
echo "MC_DONE $S $NAME rc=$rc" >> $ROOT/events.txt
exit $rc
