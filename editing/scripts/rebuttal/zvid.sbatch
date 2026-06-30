#!/bin/bash
# =============================================================================
# SLURM array: video-latent residual ablation (rebuttal).
#
# Optimizes a residual delta_z on the source video VAE latent (z_vid) with the
# SAME Qwen motion critic as ours, text+audio frozen. 8 scenarios -> array 0-7.
# Set ZVID_REG (env) to add an L2 anchor on delta_z (default 0; e.g. 0.01 to
# match the audio method's latent_reg_weight for an anchored comparison).
#
# EDIT #SBATCH account/partition, then:  sbatch editing/scripts/rebuttal/zvid.sbatch
# Subsets:  sbatch --array=0-0 ...   (first scenario / smoke)
#           sbatch --array=0-7%4 ... (<=4 GPUs at once)
# =============================================================================
#SBATCH --job-name=ssm-zvid
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/ssm_zvid_%A_%a.out
#SBATCH --error=slurm_logs/ssm_zvid_%A_%a.err
#SBATCH --account=def-amahdavi

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
[[ -f "${REPO_ROOT}/editing/scripts/rebuttal/env.sh" ]] && source "${REPO_ROOT}/editing/scripts/rebuttal/env.sh"

SCENARIOS=(
    bugatti_lights_flash
    dog_jumping
    dog_yawning
    falcon_bird_opening_wings
    groom_raising_hand
    man_pets_dog
    red_car_door_opens
    red_rose_blooming
)

IDX="${SLURM_ARRAY_TASK_ID:?Run as a SLURM array job (sbatch --array=...).}"
SCENARIO="${SCENARIOS[$IDX]}"

echo "[array ${IDX}] scenario=${SCENARIO} host=$(hostname)"

bash editing/scripts/rebuttal/run_zvid_variant.sh \
    "editing/configs/rebuttal/${SCENARIO}.yaml"
