#!/bin/bash
# VBench parts of the metric bundle (consistency / smoothness / flicker, then quality + Dynamic Degree) for one manifest.
# Env: MANIFEST, OUT. Uses the offline cached weights and the scipy-stack module (pandas for pyiqa); no job_env.sh.
set -x
cd /home/amirrz/my_codes/Sound_Sparks_Motion/editing/iclr_rebuttal
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack
R=/scratch/amirrz/iclr_rebuttal
export VBENCH_CACHE_DIR=$R/vbench_cache TORCH_HOME=$R/vbench_cache/torch_home HF_HUB_OFFLINE=1
PYV=$R/venv_vbench/bin/python
ts=$(date +%H%M%S)
for d in vbench vbench_quality; do [ -d "$OUT/$d" ] && mv "$OUT/$d" "$OUT/$d.old_$ts"; done
$PYV p7_vbench.py --manifest "$MANIFEST" --out "$OUT"; echo "VB rc=$?"
$PYV p7_vbench.py --manifest "$MANIFEST" --out "$OUT" --dims imaging_quality,aesthetic_quality,dynamic_degree --tag quality; echo "VQ rc=$?"
