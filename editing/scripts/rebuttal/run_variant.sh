#!/bin/bash
# =============================================================================
# Rebuttal Priority-1 ablation runner — one (scenario, variant) per call.
#
# Runs the audio-init ablation: can ZERO / RANDOM audio init still perform the
# edit, vs the SOURCE-init method (ours)?  Per the rebuttal plan, the audio L2
# reg is DROPPED for zero/random (audio_reg_anchor=none) so the control is clean.
# The diffusion seed stays 42 for every variant, so the baseline render is
# identical and variants are directly comparable.
#
# Usage:
#   bash editing/scripts/rebuttal/run_variant.sh <config.yaml> <variant>
#
#   <config.yaml> : an editing/configs/rebuttal/<scenario>.yaml
#   <variant>     : source | zero | random_s42 | random_s1 | random_s2
#
# Output:
#   results/rebuttal/<scenario>/<variant>/
#     ├── baseline_video.mp4
#     ├── mode_both/best_optimized_video_both.mp4   (+ logs, latents, params)
#     ├── run_config.json
#     └── metrics.json          (critic-independent objective metrics)
#
# Required env (or put paths in the config): CKPT_ROOT, QWEN_ROOT, GEMMA_ROOT.
# Optional env:
#   RESULTS_ROOT  (default: <repo>/results/rebuttal)
#   RUN_EVAL=0    skip the eval_metrics.py step
#   RAFT_WEIGHTS  local RAFT .pth for offline nodes (else torchvision download)
#   EVAL_DEVICE   cuda / cpu  (default: auto)
#   DRY_RUN=1     print the command and exit (no GPU needed)
# Meant to run inside an salloc GPU shell or from the .sbatch array.
# =============================================================================
set -euo pipefail

CONFIG_FILE="${1:?Usage: run_variant.sh <config.yaml> <variant>}"
VARIANT="${2:?Usage: run_variant.sh <config.yaml> <variant>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PARSE_SCRIPT="${REPO_ROOT}/editing/scripts/utils/parse_config.py"
MAIN_SCRIPT="${REPO_ROOT}/editing/optimize_qwen_vl.py"
EVAL_SCRIPT="${REPO_ROOT}/editing/scripts/eval_metrics.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ "${CONFIG_FILE}" != /* ]]; then CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"; fi
[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: config not found: ${CONFIG_FILE}" >&2; exit 1; }

SCENARIO="$(basename "${CONFIG_FILE}" .yaml)"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/rebuttal}"
OUTPUT_DIR="${RESULTS_ROOT}/${SCENARIO}/${VARIANT}"

# ---------------------------------------------------------------------------
# Variant -> override flags. opt-mode is BOTH for all (text+audio jointly);
# only the audio init + reg anchor + init seed change.
# ---------------------------------------------------------------------------
VARIANT_ARGS=(--opt-mode both)
case "${VARIANT}" in
    source)      VARIANT_ARGS+=(--audio-init source --audio-reg-anchor source --audio-init-seed 42) ;;
    zero)        VARIANT_ARGS+=(--audio-init zero   --audio-reg-anchor none) ;;
    random_s42)  VARIANT_ARGS+=(--audio-init random --audio-reg-anchor none --audio-init-seed 42) ;;
    random_s1)   VARIANT_ARGS+=(--audio-init random --audio-reg-anchor none --audio-init-seed 1) ;;
    random_s2)   VARIANT_ARGS+=(--audio-init random --audio-reg-anchor none --audio-init-seed 2) ;;
    *) echo "ERROR: unknown variant '${VARIANT}'. Use source|zero|random_s42|random_s1|random_s2" >&2; exit 1 ;;
esac

# Optional free-form overrides appended LAST (argparse last value wins). Space
# split — do not put spaces inside a single value. Used by smoke.sbatch to
# shrink runs (e.g. EXTRA_ARGS="--iterations 2 --num-inference-steps 8").
EXTRA=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then read -ra EXTRA <<< "${EXTRA_ARGS}"; fi
VARIANT_ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

# ---------------------------------------------------------------------------
# Build base args from the YAML, then append variant overrides (argparse last
# value wins, so these cleanly override the config's audio_* keys).
# OUTPUT_DIR is exported so parse_config bakes it into --output-dir.
# ---------------------------------------------------------------------------
export OUTPUT_DIR
BASE_ARGS=()
while IFS= read -r -d '' tok; do BASE_ARGS+=("${tok}"); done \
    < <("${PYTHON_BIN}" "${PARSE_SCRIPT}" "${CONFIG_FILE}" "${REPO_ROOT}")

echo "========================================================"
echo "  Rebuttal ablation — audio-init control"
echo "  Scenario : ${SCENARIO}"
echo "  Variant  : ${VARIANT}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Overrides: ${VARIANT_ARGS[*]}"
echo "  DRY_RUN  : ${DRY_RUN}"
echo "========================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Command that would run:"
    printf '  %q ' "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${BASE_ARGS[@]}" "${VARIANT_ARGS[@]}"; printf '\n'
    exit 0
fi

# ---------------------------------------------------------------------------
# GPU + environment (mirrors editing/scripts/run.sh).
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no visible GPU. Run inside an salloc GPU session or via sbatch." >&2
    exit 1
fi

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
    module load opencv cuda tensorboard 2>/dev/null || true
fi

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1) Optimize
# ---------------------------------------------------------------------------
"${PYTHON_BIN}" "${MAIN_SCRIPT}" "${BASE_ARGS[@]}" "${VARIANT_ARGS[@]}"

# ---------------------------------------------------------------------------
# 2) Critic-independent objective metrics -> <output>/metrics.json
# ---------------------------------------------------------------------------
if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "---- computing objective metrics ----"
    EVAL_ARGS=(--output-dir "${OUTPUT_DIR}" --mode both)
    [[ -n "${RAFT_WEIGHTS:-}" ]] && EVAL_ARGS+=(--raft-weights "${RAFT_WEIGHTS}")
    [[ -n "${EVAL_DEVICE:-}"  ]] && EVAL_ARGS+=(--device "${EVAL_DEVICE}")
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}" || \
        echo "WARN: eval_metrics.py failed for ${OUTPUT_DIR} (optimization output is still saved)."
fi

echo "========================================================"
echo "  Done: ${OUTPUT_DIR}"
echo "========================================================"
