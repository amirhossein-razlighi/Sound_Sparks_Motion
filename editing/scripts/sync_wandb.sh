#!/bin/bash
# =============================================================================
# Sync all offline W&B runs under a results directory to the online W&B.
#
# Usage:
#   bash editing/scripts/sync_wandb.sh <base_dir>
#
# Example:
#   bash editing/scripts/sync_wandb.sh results/QwenVL/a_goldfish_jumps_out_of
#
# The script walks all subdirectories looking for paths that match:
#   <base_dir>/**/wandb/wandb/
# and calls `wandb sync` on each offline run dir found inside.
# =============================================================================

set -euo pipefail

BASE_DIR="${1:-}"
if [[ -z "${BASE_DIR}" ]]; then
    echo "Usage: bash $0 <base_dir>" >&2
    exit 1
fi

if [[ ! -d "${BASE_DIR}" ]]; then
    echo "ERROR: directory not found: ${BASE_DIR}" >&2
    exit 1
fi

# Activate venv if not already inside one
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Activating venv: ${REPO_ROOT}/.venv"
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# -----------------------------------------------------------------------
# Find all  .../wandb/wandb/  directories (two nested wandb folders).
# Each such directory contains the individual offline-run-* subdirs.
# -----------------------------------------------------------------------
mapfile -t WANDB_DIRS < <(
    find "${BASE_DIR}" -type d -name "wandb" \
        | while IFS= read -r d; do
            [[ "$(basename "$(dirname "$d")")" == "wandb" ]] && echo "$d"
          done \
        | sort
)

if [[ "${#WANDB_DIRS[@]}" -eq 0 ]]; then
    echo "No wandb/wandb/ directories found under: ${BASE_DIR}"
    exit 0
fi

echo "=================================================="
echo "  Found ${#WANDB_DIRS[@]} wandb dir(s) to sync"
echo "  Base : ${BASE_DIR}"
echo "=================================================="

synced=0
failed=0

for wdir in "${WANDB_DIRS[@]}"; do
    # Relative path for readable output
    rel="${wdir#"${BASE_DIR}"/}"
    echo ""
    echo "── ${rel}"

    # Find offline run dirs inside (offline-run-* or run-*)
    mapfile -t run_dirs < <(
        find "${wdir}" -maxdepth 1 -type d \( -name "offline-run-*" -o -name "run-*" \) | sort
    )

    if [[ "${#run_dirs[@]}" -eq 0 ]]; then
        echo "   (no offline runs found, skipping)"
        continue
    fi

    for run_dir in "${run_dirs[@]}"; do
        run_name="$(basename "$run_dir")"
        echo -n "   syncing ${run_name} ... "
        if wandb sync "${run_dir}" 2>&1 | tail -1; then
            synced=$(( synced + 1 ))
        else
            echo "   FAILED: ${run_dir}" >&2
            failed=$(( failed + 1 ))
        fi
    done
done

echo ""
echo "=================================================="
echo "  Sync complete — ${synced} succeeded, ${failed} failed"
echo "=================================================="
