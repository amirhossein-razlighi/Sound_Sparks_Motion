#!/bin/bash
# =============================================================================
# SLURM array job for the audio-init ablation (rebuttal Priority 1).
#
# 8 scenarios x 4 control variants = 32 tasks (array 0-31). One GPU task per
# (scenario, variant). Each writes results/rebuttal/<scenario>/<variant>/ with
# the rendered videos, logs, latents, and metrics.json.
#
# NOTE: 'source' (= ours: source-audio init, reg on, seed 42) is NOT run here —
# it is identical to your existing main-experiment runs, so nothing to recompute.
# Only the zero/random controls run. For the 'ours' objective-metric row, run
# eval_metrics.py on the existing 'ours' output dirs (no re-optimization needed).
#
# EDIT the #SBATCH lines (account, partition, time, gpu type) for your cluster,
# then submit from the repo root:
#
#   sbatch editing/scripts/rebuttal/ablation.sbatch
#
# Subset examples (4 variants/scenario -> index = scenario*4 + variant):
#   sbatch --array=0-31           ...   # everything (default below)
#   sbatch --array=0-3            ...   # just the first scenario, all variants
#   sbatch --array=2,7,12         ...   # specific (scenario,variant) cells
#
# Make sure CKPT_ROOT / QWEN_ROOT / GEMMA_ROOT are exported (or set in the
# config), and --export them or load them in this script below.
# =============================================================================
#SBATCH --job-name=ssm-audioinit
#SBATCH --array=0-31%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/ssm_audioinit_%A_%a.out
#SBATCH --error=slurm_logs/ssm_audioinit_%A_%a.err
#SBATCH --account=def-amahdavi

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p slurm_logs

# Model roots (CKPT_ROOT / QWEN_ROOT / GEMMA_ROOT). Override by exporting before sbatch.
[[ -f "${REPO_ROOT}/editing/scripts/rebuttal/env.sh" ]] && source "${REPO_ROOT}/editing/scripts/rebuttal/env.sh"

# --- the experiment grid (order matters: index = scenario*N + variant) -------
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
# 'source' (ours) omitted on purpose — already covered by the main runs. To add
# it back: prepend 'source' here AND widen --array to 0-$((8*N-1)).
VARIANTS=(zero random_s42 random_s1 random_s2)
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
