# Sourced by the iclr_rebuttal sbatch scripts. Same environment as editing/scripts/rebuttal/run_variant.sh.
REPO_ROOT=/home/amirrz/my_codes/Sound_Sparks_Motion
source "${REPO_ROOT}/editing/scripts/rebuttal/env.sh"
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
type module >/dev/null 2>&1 && { module load opencv cuda tensorboard 2>/dev/null || true; }
source "${REPO_ROOT}/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=1 WANDB_MODE=disabled
export TORCH_HOME=/scratch/amirrz/.cache/torch
cd "${REPO_ROOT}/editing/iclr_rebuttal"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} start=$(date -Is)"
