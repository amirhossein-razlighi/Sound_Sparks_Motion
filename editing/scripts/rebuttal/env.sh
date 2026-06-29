# Model roots for the rebuttal ablation. Sourced by run_all_local.sh and
# ablation.sbatch. Override any of these by exporting before you launch.
#
#   source editing/scripts/rebuttal/env.sh      # to use in an salloc shell
#
# parse_config.py expects CKPT_ROOT to contain ltx-2.3-22b-dev.safetensors.
export CKPT_ROOT="${CKPT_ROOT:-/project/def-amahdavi/amirrz/LTX-2/checkpoints}"
export QWEN_ROOT="${QWEN_ROOT:-/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct}"
export GEMMA_ROOT="${GEMMA_ROOT:-/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized}"

# Offline compute nodes: uncomment to forbid HF hub network calls (weights must
# already be cached, e.g. the CLIP diagnostic model openai/clip-vit-base-patch32).
# export HF_HUB_OFFLINE=1

# Optional: local RAFT weights for the objective motion metric (else torchvision
# tries to download). Leave unset to fall back to the weight-free motion proxy.
# export RAFT_WEIGHTS=/path/to/raft_large_C_T_SKHT_V2.pth
