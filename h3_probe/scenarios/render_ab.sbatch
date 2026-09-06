#!/bin/bash
# Re-render A (baseline) / B (ours, best latents) pairs of finished user-study runs at 16 and 32 steps.
#   sbatch --export=ALL,SLUGS="turtle_extends_neck;cat_yawns" h3_probe/scenarios/render_ab.sbatch
# Each entry: <slug> (OUT=/scratch/amirrz/H3_exp/outputs/us_<slug>) or <slug>=<out_dir>; OUT must hold capture.pt + latents.pt.
#SBATCH --job-name=h3-render
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=02:30:00
#SBATCH --output=slurm_logs/h3_render_%j.out
#SBATCH --error=slurm_logs/h3_render_%j.err
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg31701,rg13401,rg21803,rg21802,rg31502,rg32202
set -uo pipefail
cd /home/amirrz/my_codes/Sound_Sparks_Motion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/scratch/amirrz/.cache/torch PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export SCEN_JSON=${SCEN_JSON:-/home/amirrz/my_codes/Sound_Sparks_Motion/h3_probe/scenarios/candidates_r1.json}
export QWEN_IMG=224 FP32_HEAD=1 CRITIC_OBJ=any SELECT_BY=any OPT_MODE=both ATTN_VIS=${ATTN_VIS:-0} GRAD_CHECK=0 SAVE_PREVIEWS=0
export RENDER_STEPS=${RENDER_STEPS:-"16;32"}
PY=/scratch/amirrz/H3_exp/venv/bin/python
for e in $(echo "${SLUGS:?}" | tr ';,' '  '); do
  s=${e%%=*}; OUT=/scratch/amirrz/H3_exp/outputs/us_$s; [ "$e" != "$s" ] && OUT=${e#*=}   # slug or slug=/explicit/out_dir
  if [ ! -f "$OUT/latents.pt" ]; then echo "[skip] $s: no latents.pt"; continue; fi
  echo "=== render $s ==="
  SLUG=$s OUT_DIR=$OUT INIT_LATENTS=$OUT/latents.pt $PY h3_probe/h3_full_method.py --phase render || echo "[render failed] $s"
done
echo "RENDER JOB DONE"
