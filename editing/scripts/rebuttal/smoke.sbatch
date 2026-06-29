#!/bin/bash
# =============================================================================
# SMOKE TEST for the audio-init ablation — run this BEFORE the full sweep.
#
# Exercises the two NEW code paths (zero-init and random-init) end-to-end on a
# single scenario with a deliberately shrunk run (few iterations, few inference
# steps, no previews) so it finishes in minutes. It verifies:
#   * the --audio-init zero / random branches build the latent without error,
#   * --audio-reg-anchor none drops the audio reg cleanly,
#   * the Retake render + optimization loop + final render all run,
#   * eval_metrics.py writes metrics.json.
# It does NOT produce publishable numbers — it only proves the plumbing works.
#
# EDIT the #SBATCH account/partition for your cluster, then from the repo root:
#   sbatch editing/scripts/rebuttal/smoke.sbatch
#
# Knobs (env):
#   SMOKE_SCENARIO=dog_yawning           which config in configs/rebuttal/
#   SMOKE_VARIANTS="zero random_s42"     space list mapped to array indices
#   SMOKE_EXTRA="--iterations 2 ..."     override the shrink args
# =============================================================================
#SBATCH --job-name=ssm-smoke
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/ssm_smoke_%A_%a.out
#SBATCH --error=slurm_logs/ssm_smoke_%A_%a.err
# #SBATCH --account=def-amahdavi          # <-- set me
# #SBATCH --partition=gpu                 # <-- set me if needed

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p slurm_logs

# Model roots (CKPT_ROOT / QWEN_ROOT / GEMMA_ROOT). Override by exporting before sbatch.
[[ -f "${REPO_ROOT}/editing/scripts/rebuttal/env.sh" ]] && source "${REPO_ROOT}/editing/scripts/rebuttal/env.sh"

SMOKE_SCENARIO="${SMOKE_SCENARIO:-dog_yawning}"
read -ra SMOKE_VARS <<< "${SMOKE_VARIANTS:-zero random_s42}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#SMOKE_VARS[@]} )); then
    echo "[smoke ${IDX}] no variant for this index; nothing to do."; exit 0
fi
VARIANT="${SMOKE_VARS[$IDX]}"

# Shrink the run so it finishes fast. Faithful tensor shapes (frames/size kept);
# only step counts / iterations / previews reduced. Results land in a separate
# tree so they never get confused with real ablation outputs.
export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/rebuttal_smoke}"
export EXTRA_ARGS="${SMOKE_EXTRA:---iterations 2 --early-stopping 0 --visualize-every-iters 0 --num-inference-steps 8 --retake-num-inference-steps 8 --final-retake-num-inference-steps 8 --qwen-max-frames 8 --qwen-grad-accum-steps 1 --max-eval-frames 33}"

echo "[smoke ${IDX}] scenario=${SMOKE_SCENARIO} variant=${VARIANT} host=$(hostname)"
echo "[smoke ${IDX}] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[smoke ${IDX}] EXTRA_ARGS=${EXTRA_ARGS}"

bash editing/scripts/rebuttal/run_variant.sh \
    "editing/configs/rebuttal/${SMOKE_SCENARIO}.yaml" "${VARIANT}"

echo "[smoke ${IDX}] OK — check ${RESULTS_ROOT}/${SMOKE_SCENARIO}/${VARIANT}/{mode_both/best_optimized_video_both.mp4,metrics.json}"
