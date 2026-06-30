#!/bin/bash
# =============================================================================
# LoRA capacity-control ablation runner — one (scenario, rank) per call.
#
# Trains a LoRA on the frozen LTX DiT with the SAME Qwen critic + losses as ours,
# but NO learnable text/audio latents. Shows that raw extra free parameters
# cannot match the audio-conditioning pathway. The diffusion seed stays 42 so the
# baseline render is identical to the other ablation variants.
#
# Usage:
#   LORA_PRESET=audio bash editing/scripts/rebuttal/run_lora_variant.sh <config.yaml> [rank]
#     <config.yaml> : an editing/configs/rebuttal/<scenario>.yaml
#     [rank]        : LoRA rank (default 64). alpha defaults to rank.
#     LORA_PRESET   : all (every attention; max capacity, default) | audio (audio
#                     self/cross + audio->video; the fair locus-matched control) |
#                     a2v (only audio->video attention). Output dir is
#                     lora_<preset>_r<rank> so presets don't collide.
#
# Output:  results/rebuttal/<scenario>/lora_r<rank>/
#   ├── baseline_video.mp4
#   ├── mode_lora/best_optimized_video_lora.mp4  (+ best_lora.pt, csv, params)
#   ├── run_config.json
#   └── metrics.json
#
# Env knobs: RESULTS_ROOT, LORA_ALPHA, LORA_TARGETS, LORA_DROPOUT, RUN_EVAL=0,
#   FORCE=1 (redo), RAFT_WEIGHTS, EVAL_DEVICE, DRY_RUN=1, plus CKPT_ROOT/QWEN_ROOT/GEMMA_ROOT.
# Resumable: a .completed marker is dropped on success; re-running skips done cells.
# =============================================================================
set -euo pipefail

CONFIG_FILE="${1:?Usage: run_lora_variant.sh <config.yaml> [rank]}"
RANK="${2:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"

PARSE_SCRIPT="${REPO_ROOT}/editing/scripts/utils/parse_config.py"
MAIN_SCRIPT="${REPO_ROOT}/editing/optimize_lora_critic.py"
EVAL_SCRIPT="${REPO_ROOT}/editing/scripts/eval_metrics.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE="${FORCE:-0}"

ALPHA="${LORA_ALPHA:-${RANK}}"
DROPOUT="${LORA_DROPOUT:-0.0}"

# LoRA target preset (LORA_PRESET): which modules get adapters.
#   all   = every attention (video+audio+a2v+v2a) in every block — max capacity
#   audio = audio self/cross + audio->video attention — the audio pathway (fair control)
#   a2v   = only audio->video attention — the tightest analogue to the audio latent
# LORA_TARGETS, if set, overrides the preset (and labels the run 'custom').
PRESET="${LORA_PRESET:-all}"
case "${PRESET}" in
    all)   PRESET_TARGETS="to_q,to_k,to_v,to_out.0" ;;
    a2v)   PRESET_TARGETS="audio_to_video_attn.to_q,audio_to_video_attn.to_k,audio_to_video_attn.to_v,audio_to_video_attn.to_out.0" ;;
    audio) PRESET_TARGETS="audio_attn1.to_q,audio_attn1.to_k,audio_attn1.to_v,audio_attn1.to_out.0,audio_attn2.to_q,audio_attn2.to_k,audio_attn2.to_v,audio_attn2.to_out.0,audio_to_video_attn.to_q,audio_to_video_attn.to_k,audio_to_video_attn.to_v,audio_to_video_attn.to_out.0" ;;
    *) echo "ERROR: unknown LORA_PRESET '${PRESET}'. Use all|audio|a2v." >&2; exit 1 ;;
esac
if [[ -n "${LORA_TARGETS:-}" ]]; then
    TARGETS="${LORA_TARGETS}"; LABEL="custom"
else
    TARGETS="${PRESET_TARGETS}"; LABEL="${PRESET}"
fi

if [[ "${CONFIG_FILE}" != /* ]]; then CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"; fi
[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: config not found: ${CONFIG_FILE}" >&2; exit 1; }

SCENARIO="$(basename "${CONFIG_FILE}" .yaml)"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/rebuttal}"
OUTPUT_DIR="${RESULTS_ROOT}/${SCENARIO}/lora_${LABEL}_r${RANK}"
DONE_MARKER="${OUTPUT_DIR}/.completed"
FINAL_VIDEO="${OUTPUT_DIR}/mode_lora/best_optimized_video_lora.mp4"

LORA_ARGS=(--lora-rank "${RANK}" --lora-alpha "${ALPHA}" --lora-targets "${TARGETS}" --lora-dropout "${DROPOUT}")

# Optional free-form overrides appended LAST (e.g. EXTRA_ARGS for smoke shrink).
EXTRA=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then read -ra EXTRA <<< "${EXTRA_ARGS}"; fi
LORA_ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

export OUTPUT_DIR
for v in CKPT_ROOT QWEN_ROOT GEMMA_ROOT; do
    [[ -z "${!v:-}" ]] && { echo "ERROR: ${v} not set. Export it or edit ${SCRIPT_DIR}/env.sh." >&2; exit 1; }
done

PARSE_OUT="$(mktemp)"; trap 'rm -f "${PARSE_OUT}"' EXIT
if ! "${PYTHON_BIN}" "${PARSE_SCRIPT}" "${CONFIG_FILE}" "${REPO_ROOT}" > "${PARSE_OUT}"; then
    echo "ERROR: parse_config.py failed for ${CONFIG_FILE}." >&2; exit 1
fi
BASE_ARGS=(); while IFS= read -r -d '' tok; do BASE_ARGS+=("${tok}"); done < "${PARSE_OUT}"

echo "========================================================"
echo "  Rebuttal LoRA capacity control"
echo "  Scenario : ${SCENARIO}"
echo "  Preset   : ${LABEL}  (rank=${RANK}, alpha=${ALPHA})"
echo "  Targets  : ${TARGETS}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  DRY_RUN  : ${DRY_RUN}"
echo "========================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Command that would run:"
    printf '  %q ' "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${BASE_ARGS[@]}" "${LORA_ARGS[@]}"; printf '\n'
    exit 0
fi

if [[ "${FORCE}" != "1" && -f "${DONE_MARKER}" ]]; then
    echo "  [skip] already complete: ${DONE_MARKER}  (set FORCE=1 to redo)"; exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no visible GPU. Run inside an salloc GPU session or via sbatch." >&2; exit 1
fi

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then module load opencv cuda tensorboard 2>/dev/null || true; fi
VENV="${REPO_ROOT}/.venv"; [[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
export WANDB_MODE="${WANDB_MODE:-offline}"; export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"

mkdir -p "${OUTPUT_DIR}"; cd "${REPO_ROOT}"

# 1) Optimize LoRA (skip if final video already present and not forced)
if [[ "${FORCE}" != "1" && -f "${FINAL_VIDEO}" ]]; then
    echo "---- optimized video already present; skipping optimization (FORCE=1 to redo) ----"
else
    "${PYTHON_BIN}" "${MAIN_SCRIPT}" "${BASE_ARGS[@]}" "${LORA_ARGS[@]}"
fi

# 2) Objective metrics -> metrics.json (skip if present and not forced)
if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${FORCE}" != "1" && -f "${OUTPUT_DIR}/metrics.json" ]]; then
        echo "---- metrics.json present; skipping eval ----"
    else
        EVAL_ARGS=(--output-dir "${OUTPUT_DIR}" --mode lora)
        [[ -n "${RAFT_WEIGHTS:-}" ]] && EVAL_ARGS+=(--raft-weights "${RAFT_WEIGHTS}")
        [[ -n "${EVAL_DEVICE:-}"  ]] && EVAL_ARGS+=(--device "${EVAL_DEVICE}")
        "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}" || \
            echo "WARN: eval_metrics.py failed for ${OUTPUT_DIR} (optimization output still saved)."
    fi
fi

if [[ -f "${FINAL_VIDEO}" ]] && { [[ "${RUN_EVAL}" != "1" ]] || [[ -f "${OUTPUT_DIR}/metrics.json" ]]; }; then
    touch "${DONE_MARKER}"; echo "  marked complete: ${DONE_MARKER}"
else
    echo "  NOT marked complete (missing video or metrics.json) — re-run will resume." >&2
fi
echo "  Done: ${OUTPUT_DIR}"
