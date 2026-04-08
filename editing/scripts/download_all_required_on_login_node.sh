#!/bin/bash
# =============================================================================
# Download all model assets needed by the LTX-2 + Qwen optimization scripts.
#
# Run this on a LOGIN NODE with internet access. Compute jobs run with
# HF_HUB_OFFLINE=1, so this script prepares both:
#   1) local model directories used by submit_optimize_qwen_vl.sh
#   2) the Hugging Face hub cache used by CLIP/X-CLIP model names
#
# Defaults match editing/scripts/submit_optimize_qwen_vl.sh.
#
# Common usage:
#   bash editing/scripts/download_all_required_on_login_node.sh
#
# Useful overrides:
#   HF_TOKEN=... bash editing/scripts/download_all_required_on_login_node.sh
#   DOWNLOAD_LTX_AUX=0 bash editing/scripts/download_all_required_on_login_node.sh
#   CLIP_MODELS="openai/clip-vit-base-patch32 openai/clip-vit-large-patch14" bash ...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python"
fi

HF_HOME="${HF_HOME:-/home/${USER}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

HF_MODELS_ROOT="${HF_MODELS_ROOT:-/project/def-amahdavi/amirrz/HF/models}"
CKPT_ROOT="${CKPT_ROOT:-/project/def-amahdavi/amirrz/LTX-2/checkpoints}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
QWEN_DEST="${QWEN_DEST:-${HF_MODELS_ROOT}/Qwen2.5-VL-7B-Instruct}"

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-3-12b-it-qat-q4_0-unquantized}"
GEMMA_DEST="${GEMMA_DEST:-${HF_MODELS_ROOT}/gemma-3-12b-it-qat-q4_0-unquantized}"

LTX_REPO="${LTX_REPO:-Lightricks/LTX-2.3}"
LTX_MAIN_FILES="${LTX_MAIN_FILES:-ltx-2.3-22b-dev.safetensors}"
DOWNLOAD_LTX_AUX="${DOWNLOAD_LTX_AUX:-1}"
LTX_AUX_FILES="${LTX_AUX_FILES:-ltx-2.3-spatial-upscaler-x2-1.0.safetensors ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors ltx-2.3-temporal-upscaler-x2-1.0.safetensors ltx-2.3-22b-distilled-lora-384.safetensors}"

# CLIP base is the default random-frame regularizer. X-CLIP is kept because
# older/optional losses use it. Add CLIP-L here if you want the heavier model.
CLIP_MODELS="${CLIP_MODELS:-openai/clip-vit-base-patch32 microsoft/xclip-base-patch32}"

export HF_HOME
export HF_HUB_CACHE
if [[ -n "${HF_TOKEN}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

mkdir -p "${HF_HUB_CACHE}" "${HF_MODELS_ROOT}" "${CKPT_ROOT}"

echo "========================================================"
echo "  Repo root       : ${REPO_ROOT}"
echo "  Python          : ${PYTHON}"
echo "  HF_HOME         : ${HF_HOME}"
echo "  HF hub cache    : ${HF_HUB_CACHE}"
echo "  HF models root  : ${HF_MODELS_ROOT}"
echo "  Checkpoint root : ${CKPT_ROOT}"
echo "  Qwen            : ${QWEN_MODEL} -> ${QWEN_DEST}"
echo "  Gemma           : ${GEMMA_MODEL} -> ${GEMMA_DEST}"
echo "  LTX repo        : ${LTX_REPO}"
echo "  LTX main files  : ${LTX_MAIN_FILES}"
echo "  LTX aux files   : ${DOWNLOAD_LTX_AUX} (${LTX_AUX_FILES})"
echo "  CLIP models     : ${CLIP_MODELS}"
echo "========================================================"

"${PYTHON}" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _split_words(value: str) -> list[str]:
    return [item for item in value.split() if item.strip()]


def download_snapshot(repo_id: str, local_dir: str, *, label: str) -> None:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n[{label}] snapshot {repo_id} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=_token(),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[{label}] done", flush=True)


def cache_snapshot(repo_id: str, *, label: str) -> None:
    hub_cache = os.environ["HF_HUB_CACHE"]
    print(f"\n[{label}] cache snapshot {repo_id} -> {hub_cache}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        cache_dir=hub_cache,
        token=_token(),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[{label}] done", flush=True)


def download_repo_files(repo_id: str, filenames: list[str], local_dir: str, *, label: str) -> None:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        print(f"\n[{label}] file {repo_id}/{filename} -> {local_dir}", flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            token=_token(),
        )
        print(f"[{label}] done: {filename}", flush=True)


download_snapshot(os.environ["QWEN_MODEL"], os.environ["QWEN_DEST"], label="Qwen")
download_snapshot(os.environ["GEMMA_MODEL"], os.environ["GEMMA_DEST"], label="Gemma")

ltx_files = _split_words(os.environ["LTX_MAIN_FILES"])
if os.environ.get("DOWNLOAD_LTX_AUX", "1") == "1":
    ltx_files.extend(_split_words(os.environ["LTX_AUX_FILES"]))
download_repo_files(os.environ["LTX_REPO"], ltx_files, os.environ["CKPT_ROOT"], label="LTX")

for model in _split_words(os.environ["CLIP_MODELS"]):
    cache_snapshot(model, label="CLIP")

print("\nAll requested assets are downloaded.", flush=True)
PY

echo ""
echo "========================================================"
echo "Download complete."
echo ""
echo "Use these defaults in compute jobs:"
echo "  export HF_HOME=${HF_HOME}"
echo "  export QWEN_ROOT=${QWEN_DEST}"
echo "  export GEMMA_ROOT=${GEMMA_DEST}"
echo "  export CKPT_ROOT=${CKPT_ROOT}"
echo "========================================================"
