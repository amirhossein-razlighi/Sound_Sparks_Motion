#!/bin/bash
# =============================================================================
# Ovi Phase-1 smoke test: one stock T2V+audio generation on 1 GPU.
# Validates weights + engine end-to-end before building the SDEdit harness.
#
#   sbatch ovi_probe/smoke.sbatch
# =============================================================================
#SBATCH --job-name=ovi-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/ovi_smoke_%j.out
#SBATCH --error=slurm_logs/ovi_smoke_%j.err
#SBATCH --account=def-amahdavi
set -euo pipefail

REPO_ROOT="/home/amirrz/my_codes/Sound_Sparks_Motion"
cd "${REPO_ROOT}"

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda 2>/dev/null || true; fi
source .venv/bin/activate

# Compute nodes are offline; everything must come from local paths.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

nvidia-smi -L

# Ovi's configs are cwd-relative -> run from the Ovi repo dir.
cd "${REPO_ROOT}/ovi_probe/Ovi"
python3 ../smoke_t2v.py

echo "SMOKE DONE"
