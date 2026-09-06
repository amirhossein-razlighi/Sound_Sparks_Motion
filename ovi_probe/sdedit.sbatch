#!/bin/bash
# =============================================================================
# Ovi Phase-2: SDEdit A/V editing baseline on 3 scenarios x 2 strengths.
# No optimization yet — establishes (a) source preservation vs edit-freedom
# tradeoff and (b) the no-opt baseline the Phase-3 optimization must beat.
#
#   sbatch ovi_probe/sdedit.sbatch            # all 3 scenarios (sequential, 1 GPU)
#   SCENARIOS="man_pets_dog" sbatch ovi_probe/sdedit.sbatch
# =============================================================================
#SBATCH --job-name=ovi-sdedit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/ovi_sdedit_%j.out
#SBATCH --error=slurm_logs/ovi_sdedit_%j.err
#SBATCH --account=def-amahdavi
set -euo pipefail

REPO_ROOT="/home/amirrz/my_codes/Sound_Sparks_Motion"
OUT_ROOT="/scratch/amirrz/Ovi_exp/outputs/sdedit"
cd "${REPO_ROOT}"

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda 2>/dev/null || true; fi
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# scenario -> src video + Ovi-style prompt (edit sentence + audio caption)
declare -A SRC PROMPT
SRC[man_pets_dog]="input_videos/cook_spinach_main.mp4"
PROMPT[man_pets_dog]="The man pets the dog. <AUDCAP>Soft rustling of fur as a hand strokes a dog, calm indoor ambience.<ENDAUDCAP>"
SRC[dog_yawning]="input_videos/dog_sitting_on_chair_main.mp4"
PROMPT[dog_yawning]="A dog yawning. <AUDCAP>A soft dog yawn whine, quiet room ambience.<ENDAUDCAP>"
SRC[red_rose_blooming]="input_videos/a_red_rose_bud_in_a_green_main.mp4"
PROMPT[red_rose_blooming]="A red rose blooming. <AUDCAP>Gentle breeze, soft outdoor nature ambience, birds faint in the distance.<ENDAUDCAP>"

SCENARIOS="${SCENARIOS:-man_pets_dog dog_yawning red_rose_blooming}"
STRENGTHS="${STRENGTHS:-0.5 0.7}"

cd "${REPO_ROOT}/ovi_probe/Ovi"
for sc in ${SCENARIOS}; do
    for st in ${STRENGTHS}; do
        echo "######## ${sc} @ strength ${st} ########"
        python3 ../sdedit_edit.py \
            --src-video "${REPO_ROOT}/${SRC[$sc]}" \
            --prompt "${PROMPT[$sc]}" \
            --strength "${st}" \
            --seed 42 \
            --sample-steps 50 \
            --out-dir "${OUT_ROOT}/${sc}" \
            --tag "${sc}"
    done
done
echo "SDEDIT PHASE-2 DONE"
