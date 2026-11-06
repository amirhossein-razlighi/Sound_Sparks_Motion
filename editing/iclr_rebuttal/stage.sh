#!/bin/bash
# One step per allocation: copy model weights to node-local disk (Lustre mmap stalls on interactive nodes).
LOC=/localscratch/amirrz.${SLURM_JOB_ID}.0/models; mkdir -p $LOC
H=/project/def-amahdavi/amirrz/HF/models
rsync -a /project/def-amahdavi/amirrz/LTX-2/checkpoints/ltx-2.3-22b-dev.safetensors $LOC/ &
for m in gemma-3-12b-it-qat-q4_0-unquantized Qwen2.5-VL-7B-Instruct Qwen3-VL-8B-Instruct InternVL3-8B-hf; do rsync -a --exclude .cache $H/$m $LOC/ & done
wait; touch $LOC/.staged; df -h $LOC | tail -1; du -sh $LOC
