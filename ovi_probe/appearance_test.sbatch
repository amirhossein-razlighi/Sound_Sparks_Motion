#!/bin/bash
# Diagnostic: can Ovi+SDEdit edit at all (appearance edits + prompt-leverage
# control)?   sbatch ovi_probe/appearance_test.sbatch
#SBATCH --job-name=ovi-appear
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/ovi_appear_%j.out
#SBATCH --error=slurm_logs/ovi_appear_%j.err
#SBATCH --account=def-amahdavi
set -euo pipefail

REPO_ROOT="/home/amirrz/my_codes/Sound_Sparks_Motion"
cd "${REPO_ROOT}"
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda 2>/dev/null || true; fi
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${REPO_ROOT}/ovi_probe/Ovi"
python3 ../appearance_test.py
echo "APPEAR DONE"
