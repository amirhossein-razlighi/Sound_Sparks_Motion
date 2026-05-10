#!/bin/bash
# =============================================================================
# Download Qwen2.5-VL model weights from HuggingFace.
#
# Run this once on a machine with internet access.
#
# Usage:
#   bash editing/scripts/setup/download_qwen.sh
#
# Optional env vars:
#   QWEN_MODEL  — HF model ID  (default: Qwen/Qwen2.5-VL-7B-Instruct)
#   QWEN_DEST   — local path   (default: ~/models/Qwen2.5-VL-7B-Instruct)
#   HF_TOKEN    — HuggingFace token if the repo requires authentication
#
# Memory footprint (bf16):
#   7B model : ~14 GB — default, fits alongside LTX-22B fp8 in 80 GB
#   3B model : ~ 6 GB — lower memory; use QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
QWEN_DEST="${QWEN_DEST:-${HOME}/models/Qwen2.5-VL-7B-Instruct}"

echo "========================================================"
echo "  Model : ${QWEN_MODEL}"
echo "  Dest  : ${QWEN_DEST}"
echo "========================================================"

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

python3 - <<EOF
from huggingface_hub import snapshot_download
import os
from pathlib import Path

model_id = "${QWEN_MODEL}"
dest = "${QWEN_DEST}"

Path(dest).mkdir(parents=True, exist_ok=True)
print(f"Downloading {model_id} -> {dest} ...")

snapshot_download(
    repo_id=model_id,
    local_dir=dest,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
print(f"Done. Model saved to: {dest}")
EOF

echo ""
echo "========================================================"
echo "Download complete: ${QWEN_DEST}"
echo "Set QWEN_ROOT=${QWEN_DEST} before running experiments."
echo "========================================================"
