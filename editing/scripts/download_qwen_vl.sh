#!/bin/bash
# =============================================================================
# Download Qwen2.5-VL model weights from HuggingFace.
# Run this on the LOGIN NODE where internet access is available.
#
# Usage:
#   bash editing/scripts/download_qwen_vl.sh
#
# Optional env vars:
#   QWEN_MODEL    — HF model ID      (default: Qwen/Qwen2.5-VL-7B-Instruct)
#   QWEN_DEST     — local save path  (default: /project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct)
#   HF_TOKEN      — HuggingFace token if the repo requires authentication
#
# Memory footprint (bf16):
#   7B model : ~14 GB  — default, fits in H200 80 GB alongside LTX-22B fp8
#   3B model : ~ 6 GB  — use if you're tight on memory
#   To use 3B instead: QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct bash ...
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/amirrz/my_codes/LTX-2}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
QWEN_DEST="${QWEN_DEST:-/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct}"

echo "========================================================"
echo "  Model : ${QWEN_MODEL}"
echo "  Dest  : ${QWEN_DEST}"
echo "========================================================"

# Activate venv to use huggingface_hub / transformers
source "${REPO_ROOT}/.venv/bin/activate"

# Optional: set HF token for gated models
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export HF_HOME="${HF_HOME:-/home/amirrz/.cache/huggingface}"

python - <<EOF
from huggingface_hub import snapshot_download
import os

model_id = "${QWEN_MODEL}"
dest = "${QWEN_DEST}"

os.makedirs(dest, exist_ok=True)
print(f"Downloading {model_id} → {dest} ...")

snapshot_download(
    repo_id=model_id,
    local_dir=dest,
    local_dir_use_symlinks=False,   # copy files, not symlinks (safer on Lustre)
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)

print(f"Done. Model saved to: {dest}")
EOF

echo ""
echo "========================================================"
echo "Download complete: ${QWEN_DEST}"
echo "Set QWEN_ROOT=${QWEN_DEST} (or leave default) when submitting jobs."
echo "========================================================"
