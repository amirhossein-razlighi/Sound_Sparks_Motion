# TRILLIUM job env (pattern: job_env.sh). Sourced by worker.sh via SSM_JOB_ENV.
# /home is read-only on Trillium compute nodes: code/venv are read from /home, all writes go to $SCRATCH.
REPO_ROOT=/home/amirrz/my_codes/Sound_Sparks_Motion
source "${REPO_ROOT}/editing/scripts/rebuttal/env.sh"   # CKPT_ROOT / QWEN_ROOT / GEMMA_ROOT (/project paths)
source "${REPO_ROOT}/.venv/bin/activate"
export SSM_CKPT_PATH=/project/def-amahdavi/amirrz/LTX-2/checkpoints/ltx-2.3-22b-dev.safetensors
export SSM_GEMMA_ROOT=/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized
export SSM_QWEN_ROOT=/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct
export SSM_PAPER_RESULTS=/scratch/amirrz/iclr_rebuttal/ssm_bundle/paper_results
export SSM_COMPARISONS=/scratch/amirrz/iclr_rebuttal/ssm_bundle/comparisons
export SSM_OUT_ROOT=${SSM_OUT_ROOT:-/scratch/amirrz/iclr_rebuttal/out}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=1 WANDB_MODE=disabled
export TORCH_HOME=/scratch/amirrz/torch            # raft_large_C_T_SKHT_V2 cached here
export IVE_ROOT=/scratch/amirrz/iclr_rebuttal/ivebench IVE_LOCAL=${SLURM_TMPDIR:-/dev/shm/amirrz_${SLURM_JOB_ID:-login}}
export MPLCONFIGDIR=/scratch/amirrz/matplotlib PYTHONDONTWRITEBYTECODE=1
# ROOT-CAUSE FIX (RORQUAL 12:42): the paper runs used xformers 0.0.33 (uv.lock extra), LTX attention silently falls back
# to SDPA without it -> different samples. Exact wheel installed outside the shared venv; refuse to run without it.
export PYTHONPATH=/project/def-amahdavi/amirrz/xformers_0033/site${PYTHONPATH:+:$PYTHONPATH}
cd "${REPO_ROOT}/editing/iclr_rebuttal"
if ! python -c "import ltx_core.model.transformer.attention as A, sys; sys.exit(0 if A.memory_efficient_attention is not None else 1)"; then
  echo "FATAL: xformers not active in LTX attention - refusing to run"; exit 9
fi
echo "XFORMERS_ACTIVE True ($(python -c 'import xformers; print(xformers.__version__)'))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -Is)"
