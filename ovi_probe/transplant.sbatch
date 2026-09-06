#!/bin/bash
# =============================================================================
# Ovi audio-latent transplant probe (no optimization): harvest an edit-matched
# audio latent from a strength-1.0 generation, inject it (alpha sweep) into the
# SDEdit of the source video, Qwen-score each variant.
#
#   sbatch ovi_probe/transplant.sbatch
#   SCENARIO=dog_yawning sbatch ovi_probe/transplant.sbatch
# =============================================================================
#SBATCH --job-name=ovi-transplant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/ovi_transplant_%j.out
#SBATCH --error=slurm_logs/ovi_transplant_%j.err
#SBATCH --account=def-amahdavi
set -euo pipefail

REPO_ROOT="/home/amirrz/my_codes/Sound_Sparks_Motion"
OUT_ROOT="/scratch/amirrz/Ovi_exp/outputs/transplant"
cd "${REPO_ROOT}"

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda 2>/dev/null || true; fi
source .venv/bin/activate
source editing/scripts/rebuttal/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -A SRC PROMPT EDIT
SRC[man_pets_dog]="input_videos/cook_spinach_main.mp4"
PROMPT[man_pets_dog]="The man reaches out his hand and pets the dog next to him. <AUDCAP>Soft rustling of fur as a hand strokes a dog, gentle petting sounds, calm indoor ambience.<ENDAUDCAP>"
EDIT[man_pets_dog]="The man pets the dog."
SRC[dog_yawning]="input_videos/dog_sitting_on_chair_main.mp4"
PROMPT[dog_yawning]="A dog opens its mouth wide in a big yawn. <AUDCAP>A soft dog yawn whine, quiet room ambience.<ENDAUDCAP>"
EDIT[dog_yawning]="A dog yawning"
SRC[red_rose_blooming]="input_videos/a_red_rose_bud_in_a_green_main.mp4"
PROMPT[red_rose_blooming]="A red rose bud slowly opens its petals and blooms. <AUDCAP>Gentle breeze, soft outdoor nature ambience, birds faint in the distance.<ENDAUDCAP>"
EDIT[red_rose_blooming]="A red rose blooming."

SCENARIO="${SCENARIO:-man_pets_dog}"
STRENGTH="${STRENGTH:-0.7}"
ALPHAS="${ALPHAS:-0.0,0.5,1.0}"
PIN="${PIN:-0}"
PIN_FLAG=""; [[ "${PIN}" == "1" ]] && PIN_FLAG="--pin-audio"

cd "${REPO_ROOT}/ovi_probe/Ovi"
python3 ../audio_transplant.py \
    --src-video "${REPO_ROOT}/${SRC[$SCENARIO]}" \
    --prompt "${PROMPT[$SCENARIO]}" \
    --edit-prompt "${EDIT[$SCENARIO]}" \
    --out-dir "${OUT_ROOT}/${SCENARIO}" \
    --strength "${STRENGTH}" \
    --alphas "${ALPHAS}" \
    --seed 42 \
    --sample-steps 30 ${PIN_FLAG}
echo "TRANSPLANT DONE ${SCENARIO}"
