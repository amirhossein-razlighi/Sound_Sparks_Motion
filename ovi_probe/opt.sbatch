#!/bin/bash
# =============================================================================
# Ovi Phase-3: optimize the audio-stream latent with the Qwen motion critic.
# One scenario per job. Gate on man_pets_dog first, then run the rest.
#
#   sbatch ovi_probe/opt.sbatch                                  # man_pets_dog
#   SCENARIO=dog_yawning sbatch ovi_probe/opt.sbatch
#   SCENARIO=red_rose_blooming sbatch ovi_probe/opt.sbatch
# =============================================================================
#SBATCH --job-name=ovi-opt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/ovi_opt_%j.out
#SBATCH --error=slurm_logs/ovi_opt_%j.err
#SBATCH --account=def-amahdavi
set -euo pipefail

REPO_ROOT="/home/amirrz/my_codes/Sound_Sparks_Motion"
OUT_ROOT="/scratch/amirrz/Ovi_exp/outputs/opt"
cd "${REPO_ROOT}"

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda 2>/dev/null || true; fi
source .venv/bin/activate
source editing/scripts/rebuttal/env.sh   # QWEN_ROOT etc.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -A SRC PROMPT EDIT
SRC[man_pets_dog]="input_videos/cook_spinach_main.mp4"
PROMPT[man_pets_dog]="The man pets the dog. <AUDCAP>Soft rustling of fur as a hand strokes a dog, calm indoor ambience.<ENDAUDCAP>"
EDIT[man_pets_dog]="The man pets the dog."
SRC[dog_yawning]="input_videos/dog_sitting_on_chair_main.mp4"
PROMPT[dog_yawning]="A dog yawning. <AUDCAP>A soft dog yawn whine, quiet room ambience.<ENDAUDCAP>"
EDIT[dog_yawning]="A dog yawning"
SRC[red_rose_blooming]="input_videos/a_red_rose_bud_in_a_green_main.mp4"
PROMPT[red_rose_blooming]="A red rose blooming. <AUDCAP>Gentle breeze, soft outdoor nature ambience, birds faint in the distance.<ENDAUDCAP>"
EDIT[red_rose_blooming]="A red rose blooming."

SCENARIO="${SCENARIO:-man_pets_dog}"
STRENGTH="${STRENGTH:-0.7}"
ITERS="${ITERS:-12}"
OPT_METHOD="${OPT_METHOD:-grad}"
DECODE_SEG="${DECODE_SEG:-1}"
DECODE_CTX="${DECODE_CTX:-1}"
# spsa needs 1 GPU; grad needs 2. Out-dir gets a method suffix so runs don't collide.

cd "${REPO_ROOT}/ovi_probe/Ovi"
python3 ../optimize_audio_latent.py \
    --src-video "${REPO_ROOT}/${SRC[$SCENARIO]}" \
    --prompt "${PROMPT[$SCENARIO]}" \
    --edit-prompt "${EDIT[$SCENARIO]}" \
    --out-dir "${OUT_ROOT}/${SCENARIO}_${OPT_METHOD}" \
    --strength "${STRENGTH}" \
    --seed 42 \
    --sample-steps 30 \
    --iterations "${ITERS}" \
    --opt-method "${OPT_METHOD}" \
    --decode-seg "${DECODE_SEG}" \
    --decode-ctx "${DECODE_CTX}" \
    --latent-reg-weight 0.01 \
    --qwen-max-frames 24 \
    --qwen-img-size 224
echo "OPT DONE ${SCENARIO}"
