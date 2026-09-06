#!/bin/bash
# Our full gradient-based method on MiniMax-H3 (audio latent + text residual, Qwen critic),
# transformer sharded over 2 GPUs.   sbatch --export=ALL,SLUG=boy_crouches h3_probe/full_method.sbatch
#SBATCH --job-name=h3-full
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=02:59:00
#SBATCH --output=slurm_logs/h3_full_%j.out
#SBATCH --error=slurm_logs/h3_full_%j.err
#SBATCH --account=def-amahdavi
#SBATCH --exclude=rg31701,rg13401,rg21803,rg21802,rg31502
set -euo pipefail
cd /home/amirrz/my_codes/Sound_Sparks_Motion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/scratch/amirrz/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
PY=/scratch/amirrz/H3_exp/venv/bin/python
# Phase A (stock pipeline, auto-offload) in its own process so its ~180 GB of offloaded weights are gone before Phase B.
OUT=${OUT_DIR:-/scratch/amirrz/H3_exp/outputs/fullmethod_${SLUG:-boy_crouches}_${OPT_MODE:-both}}
if [ -f "$OUT/capture.pt" ] && [ "${REUSE_CAPTURE:-1}" = "1" ]; then
  echo "reusing $OUT/capture.pt (Phase A skipped)"
else
  $PY h3_probe/h3_full_method.py --phase a
fi
$PY h3_probe/h3_full_method.py --phase b
echo "H3 FULL METHOD JOB DONE"
