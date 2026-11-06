# Sourced by the iclr_rebuttal sbatch scripts. Same environment as editing/scripts/rebuttal/run_variant.sh.
REPO_ROOT=/home/amirrz/my_codes/Sound_Sparks_Motion
source "${REPO_ROOT}/editing/scripts/rebuttal/env.sh"
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
type module >/dev/null 2>&1 && { module load opencv cuda tensorboard 2>/dev/null || true; }
source "${REPO_ROOT}/.venv/bin/activate"
# ROOT-CAUSE FIX (2026-11-06): the paper runs (April) had xformers 0.0.33+5d4b92a5 installed (uv.lock extra "xformers"),
# so LTX attention used xformers; the current .venv was synced without that extra -> PyTorch SDPA -> different samples.
# Put the exact April xformers build on the path (side folder; .venv itself unchanged). Verified: dog_yawning seed-42
# baseline MAD vs paper 0.108 (SDPA) -> 0.038 (xformers), same video.
export PYTHONPATH=/home/amirrz/my_codes/xformers_0033/site${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=1 WANDB_MODE=disabled
export TORCH_HOME=/scratch/amirrz/.cache/torch
cd "${REPO_ROOT}/editing/iclr_rebuttal"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} start=$(date -Is)"
# Guard: refuse to run if LTX would not use xformers (see ROOT-CAUSE FIX above).
python -c "import ltx_core.model.transformer.attention as A, sys; sys.exit(0 if A.memory_efficient_attention is not None else 1)" \
  || { echo "FATAL: xformers not active for LTX attention - refusing to run"; exit 1; }
echo "xformers active for LTX attention"
