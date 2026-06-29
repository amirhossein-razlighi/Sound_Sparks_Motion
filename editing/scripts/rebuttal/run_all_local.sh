#!/bin/bash
# =============================================================================
# Sequential driver for the audio-init ablation — for an interactive salloc
# GPU shell (one GPU, runs everything back-to-back). For cluster parallelism
# use ablation.sbatch instead.
#
# Usage (inside salloc with a GPU):
#   bash editing/scripts/rebuttal/run_all_local.sh
#
# RESUMABLE: if interrupted, just re-run — each cell drops a .completed marker
# when its video + metrics.json are present, so finished cells are skipped, a
# cell with a video but no metrics only re-runs eval, and an interrupted cell
# redoes its optimization. Nothing is recomputed needlessly. FORCE=1 redoes all.
#
# Env knobs:
#   SCENARIOS="dog_yawning red_rose_blooming"   # subset of scenarios
#   VARIANTS="zero random_s42"                  # subset of variants
#   FORCE=1                                      # ignore .completed, redo everything
#   DRY_RUN=1                                    # print commands only
#   plus all the env vars run_variant.sh honors (CKPT_ROOT, RAFT_WEIGHTS, ...)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Load model roots (CKPT_ROOT / QWEN_ROOT / GEMMA_ROOT); override by exporting first.
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"

SCENARIOS="${SCENARIOS:-bugatti_lights_flash dog_jumping dog_yawning falcon_bird_opening_wings groom_raising_hand man_pets_dog red_car_door_opens red_rose_blooming}"
# 'source' (ours) omitted by default — identical to the existing main runs, no
# need to recompute. Add it back via VARIANTS="source zero ..." if you want a
# fresh apples-to-apples source row.
VARIANTS="${VARIANTS:-zero random_s42 random_s1 random_s2}"

echo "Scenarios: ${SCENARIOS}"
echo "Variants : ${VARIANTS}"

for scenario in ${SCENARIOS}; do
    cfg="${REPO_ROOT}/editing/configs/rebuttal/${scenario}.yaml"
    if [[ ! -f "${cfg}" ]]; then
        echo "WARN: missing config ${cfg}; skipping." >&2
        continue
    fi
    for variant in ${VARIANTS}; do
        echo ""
        echo "######## ${scenario} / ${variant} ########"
        bash "${SCRIPT_DIR}/run_variant.sh" "${cfg}" "${variant}" || \
            echo "ERROR: ${scenario}/${variant} failed; continuing." >&2
    done
done

echo ""
echo "All requested runs attempted. Aggregate with:"
echo "  python3 editing/scripts/rebuttal/aggregate_metrics.py --results-root results/rebuttal"
