#!/bin/bash
# =============================================================================
# Run a multimodal motion editing experiment from a YAML config.
#
# Usage:
#   bash editing/scripts/run.sh editing/configs/example_hummingbird.yaml
#   bash editing/scripts/run.sh --benchmark editing/configs/example_hummingbird.yaml
#
# Required environment variables (set before calling, or put paths in the config):
#   CKPT_ROOT   — directory containing ltx-2.3-22b-dev.safetensors
#   QWEN_ROOT   — directory containing Qwen2.5-VL-7B-Instruct weights
#   GEMMA_ROOT  — directory containing Gemma-3-12b text encoder weights
#
# Optional flags / environment overrides:
#   --benchmark — benchmark mode: uses optimize_qwen_vl_benchmark.py, disables W&B
#                 and intermediate visualization, forces one final best-video render
#   --dry-run   — same as DRY_RUN=1
#   REPO_ROOT   — repo root (auto-detected from script location if unset)
#   OUTPUT_DIR  — override the output directory generated from the config
#   DRY_RUN=1   — print the final command without running it (no GPU needed)
#   BENCHMARK=1 — same as --benchmark
#
# Works on any machine with a GPU. On HPC clusters with a module system,
# relevant modules are loaded automatically if the 'module' command is available.
# =============================================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
BENCHMARK="${BENCHMARK:-0}"

# Parse flags before the config positional arg
CONFIG_FILE=""
for arg in "$@"; do
    case "$arg" in
        --benchmark) BENCHMARK=1 ;;
        --dry-run)   DRY_RUN=1   ;;
        *)           CONFIG_FILE="$arg" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PARSE_SCRIPT="${SCRIPT_DIR}/utils/parse_config.py"

if [[ "${BENCHMARK}" == "1" ]]; then
    MAIN_SCRIPT="${REPO_ROOT}/editing/optimize_qwen_vl_benchmark.py"
else
    MAIN_SCRIPT="${REPO_ROOT}/editing/optimize_qwen_vl.py"
fi

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${CONFIG_FILE}" ]]; then
    echo "ERROR: provide a YAML config file." >&2
    echo "Usage: bash $0 editing/configs/my_experiment.yaml" >&2
    echo ""
    echo "Available configs:"
    ls "${REPO_ROOT}/editing/configs/"*.yaml 2>/dev/null | xargs -I{} basename {} || true
    exit 1
fi

if [[ "${CONFIG_FILE}" != /* ]]; then
    CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse config → CLI args
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
ARGS=()
while IFS= read -r -d '' tok; do
    ARGS+=("${tok}")
done < <("${PYTHON_BIN}" "${PARSE_SCRIPT}" "${CONFIG_FILE}" "${REPO_ROOT}")

# Extract output dir for display
OUTPUT_DIR_SELECTED=""
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--output-dir" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
        OUTPUT_DIR_SELECTED="${ARGS[$((i + 1))]}"
        break
    fi
done

echo "========================================================"
echo "  Sound Sparks Motion — optimization run"
echo "  Config     : ${CONFIG_FILE}"
echo "  Output dir : ${OUTPUT_DIR_SELECTED}"
echo "  Benchmark  : ${BENCHMARK}"
echo "  DRY_RUN    : ${DRY_RUN}"
echo "========================================================"

# In benchmark mode, force settings for clean timing and minimal I/O
if [[ "${BENCHMARK}" == "1" ]]; then
    ARGS+=(
        --visualize-every-iters 0
        --no-clip-similarity-diag
        --save-final-videos
    )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo ""
    echo "Command that would run:"
    printf '  %q ' "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${ARGS[@]}"
    printf '\n'
    exit 0
fi

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no visible GPU. Run on a machine with a GPU or inside an salloc session." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Load HPC modules if the module system is present
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
    module load opencv cuda tensorboard 2>/dev/null || true
fi

# Activate virtual environment
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# W&B settings — extracted from args for env export
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR_SELECTED}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/${USER}/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${HOME}/.config/wandb}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"

if [[ "${BENCHMARK}" == "1" ]]; then
    export WANDB_DISABLED=true
    export WANDB_MODE=disabled
fi

mkdir -p "${OUTPUT_DIR_SELECTED}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${MAIN_SCRIPT}" "${ARGS[@]}"

echo ""
echo "========================================================"
echo "  Run complete. Outputs: ${OUTPUT_DIR_SELECTED}"
echo "========================================================"
