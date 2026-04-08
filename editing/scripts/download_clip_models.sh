#!/bin/bash
# Download CLIP-family models on a login node so offline cluster jobs can use
# the Hugging Face cache.
#
# Default:
#   bash editing/scripts/download_clip_models.sh
#
# Custom:
#   bash editing/scripts/download_clip_models.sh openai/clip-vit-large-patch14

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python"
fi

HF_HOME="${HF_HOME:-/home/${USER}/.cache/huggingface}"
export HF_HOME

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("openai/clip-vit-base-patch32")
fi

"${PYTHON}" - "${MODELS[@]}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

models = sys.argv[1:]
hf_home = os.environ["HF_HOME"]
hub_cache = os.path.join(hf_home, "hub")

print(f"HF_HOME={hf_home}")
print(f"Hub cache={hub_cache}")
for model in models:
    print(f"Downloading {model} ...", flush=True)
    snapshot_download(
        repo_id=model,
        cache_dir=hub_cache,
        resume_download=True,
    )
    print(f"Done: {model}", flush=True)
PY
