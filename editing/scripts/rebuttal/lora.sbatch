#!/bin/bash
# =============================================================================
# SLURM array: LoRA capacity-control ablation (rebuttal).
#
# Trains a LoRA on the frozen LTX DiT with the SAME Qwen critic/losses as ours,
# NO learnable text/audio latents. Shows free params can't match the audio path.
# Grid = 8 scenarios x PRESETS x RANKS. Default PRESETS=(audio), RANKS=(64) -> 8
# cells (array 0-7). The 'audio' preset is the fair locus-matched control (LoRA on
# the audio self/cross + audio->video attention). Add 'all' (max capacity) via
# PRESETS=(audio all) and widen --array to 0-$((8*2-1)).
#
# EDIT #SBATCH account/partition, then:  sbatch editing/scripts/rebuttal/lora.sbatch
# Subsets:  sbatch --array=0-0 ...   (first cell only / smoke)
#           sbatch --array=0-7%4 ... (<=4 GPUs at once)
# =============================================================================
#SBATCH --job-name=ssm-lora
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/ssm_lora_%A_%a.out
#SBATCH --error=slurm_logs/ssm_lora_%A_%a.err
#SBATCH --account=def-amahdavi

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
[[ -f "${REPO_ROOT}/editing/scripts/rebuttal/env.sh" ]] && source "${REPO_ROOT}/editing/scripts/rebuttal/env.sh"

# --- grid (index = scenario*N_RANKS + rank_idx) ------------------------------
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
# (scenario, preset, rank) grid. Default: the fair locus-matched 'audio' control
# at rank 64 -> 8 cells (array 0-7). To also run the max-capacity upper bound, use
# PRESETS=(audio all) and widen --array to 0-$((8*2-1)). 'a2v' is the tightest scope.
PRESETS=(audio)            # any of: audio (fair control) | all (max capacity) | video (video-only) | a2v
RANKS=(64)
N_RANKS=${#RANKS[@]}
N_PRESETS=${#PRESETS[@]}
N_PER_SCEN=$(( N_PRESETS * N_RANKS ))

IDX="${SLURM_ARRAY_TASK_ID:?Run as a SLURM array job (sbatch --array=...).}"
S_IDX=$(( IDX / N_PER_SCEN ))
REM=$(( IDX % N_PER_SCEN ))
P_IDX=$(( REM / N_RANKS ))
R_IDX=$(( REM % N_RANKS ))
SCENARIO="${SCENARIOS[$S_IDX]}"
PRESET="${PRESETS[$P_IDX]}"
RANK="${RANKS[$R_IDX]}"

echo "[array ${IDX}] scenario=${SCENARIO} preset=${PRESET} lora_rank=${RANK} host=$(hostname)"

LORA_PRESET="${PRESET}" bash editing/scripts/rebuttal/run_lora_variant.sh \
    "editing/configs/rebuttal/${SCENARIO}.yaml" "${RANK}"
