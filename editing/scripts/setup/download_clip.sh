#!/bin/bash
# =============================================================================
# Download CLIP / X-CLIP model weights into the HuggingFace hub cache.
#
# These models are loaded by name at runtime so only the cache needs to be
# populated; no explicit local_dir is required.
#
# Usage:
#   bash editing/scripts/setup/download_clip.sh
#
# Optional env vars:
#   CLIP_MODELS — space-separated list of HF model IDs to download
#                 (default: CLIP-B/32 + X-CLIP-B/32)
#   HF_HOME     — HuggingFace cache root (default: ~/.cache/huggingface)
#   HF_TOKEN    — HuggingFace token (if needed)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

CLIP_MODELS="${CLIP_MODELS:-openai/clip-vit-base-patch32 microsoft/xclip-base-patch32}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

echo "========================================================"
echo "  Models  : ${CLIP_MODELS}"
echo "  HF cache: ${HF_HUB_CACHE}"
echo "========================================================"

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export HF_HOME
export HF_HUB_CACHE

mkdir -p "${HF_HUB_CACHE}"

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

models = os.environ["CLIP_MODELS"].split()
hub_cache = os.environ["HF_HUB_CACHE"]

for model_id in models:
    print(f"\nCaching {model_id} -> {hub_cache} ...")
    snapshot_download(
        repo_id=model_id,
        cache_dir=hub_cache,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"Done: {model_id}")

print("\nAll CLIP models cached.")
PY

echo ""
echo "Download complete."
