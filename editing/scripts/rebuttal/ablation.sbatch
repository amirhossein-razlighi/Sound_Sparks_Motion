#!/bin/bash
# =============================================================================
# SLURM array job for the audio-init ablation (rebuttal Priority 1).
#
# 8 scenarios x 5 variants = 40 tasks (array 0-39). One GPU task per (scenario,
# variant). Each task writes results/rebuttal/<scenario>/<variant>/ with the
# rendered videos, logs, latents, and metrics.json.
#
# EDIT the #SBATCH lines (account, partition, time, gpu type) for your cluster,
# then submit from the repo root:
#
#   sbatch editing/scripts/rebuttal/ablation.sbatch
#
# Subset examples:
#   sbatch --array=0-39           ...   # everything (default below)
#   sbatch --array=5-39           ...   # skip the 'source' variant (idx%5==0)
#   sbatch --array=0-4            ...   # just the first scenario, all variants
#   sbatch --array=2,7,12         ...   # specific (scenario,variant) cells
#
# Make sure CKPT_ROOT / QWEN_ROOT / GEMMA_ROOT are exported (or set in the
# config), and --export them or load them in this script below.
# =============================================================================
#SBATCH --job-name=ssm-audioinit
#SBATCH --array=0-39%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/ssm_audioinit_%A_%a.out
#SBATCH --error=slurm_logs/ssm_audioinit_%A_%a.err
# #SBATCH --account=def-amahdavi          # <-- set me
# #SBATCH --partition=gpu                 # <-- set me if needed

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p slurm_logs

# Model roots (CKPT_ROOT / QWEN_ROOT / GEMMA_ROOT). Override by exporting before sbatch.
[[ -f "${REPO_ROOT}/editing/scripts/rebuttal/env.sh" ]] && source "${REPO_ROOT}/editing/scripts/rebuttal/env.sh"

# --- the experiment grid (order matters: index = scenario*5 + variant) -------
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
VARIANTS=(source zero random_s42 random_s1 random_s2)
N_VARIANTS=${#VARIANTS[@]}

IDX="${SLURM_ARRAY_TASK_ID:?This script must be run as a SLURM array job (sbatch --array=...).}"
S_IDX=$(( IDX / N_VARIANTS ))
V_IDX=$(( IDX % N_VARIANTS ))
SCENARIO="${SCENARIOS[$S_IDX]}"
VARIANT="${VARIANTS[$V_IDX]}"

echo "[array ${IDX}] scenario=${SCENARIO} variant=${VARIANT} host=$(hostname)"

# Model roots come from env.sh (sourced above). Offline / RAFT options also there.

bash editing/scripts/rebuttal/run_variant.sh \
    "editing/configs/rebuttal/${SCENARIO}.yaml" "${VARIANT}"
