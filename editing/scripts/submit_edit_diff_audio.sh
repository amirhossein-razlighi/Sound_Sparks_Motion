#!/bin/bash
# =============================================================================
# SLURM job: audio-diff video editing (edit_with_audio_diff.py)
#
# 3-step pipeline:
#   Step 1 — TI2VidOneStagePipeline(first_frame + edit_prompt) → video + audio
#   Step 2 — audio diff: gen_audio <diff_mode> src_audio → diff_audio
#   Step 3 — RetakePipeline(src_video + diff_audio) → edited video
#
# Quick start:
#   sbatch editing/scripts/submit_edit_diff.sh \
#       /path/to/video.mp4 "The guitarist drops her guitar and leaves the scene."
#
# Or via environment variables:
#   export SRC_VIDEO=/path/to/video.mp4
#   export EDIT_PROMPT="The guitarist drops her guitar and leaves the scene."
#   sbatch editing/scripts/submit_edit_diff.sh
#
# Runtime estimate: ~2 × 10-20 min (two full diffusion passes) on a single H100.
# =============================================================================

#SBATCH --job-name=ltx_edit_diff
#SBATCH --account=def-amahdavi
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=48G
#SBATCH --time=02:45:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ---------------------------------------------------------------------------
# Settings — override via environment variables before sbatch
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/home/amirrz/my_codes/LTX-2}"
CKPT_ROOT="${CKPT_ROOT:-/project/def-amahdavi/amirrz/LTX-2/checkpoints}"
GEMMA_ROOT="${GEMMA_ROOT:-/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized}"

# Positional args: $1 = src video path, $2 = edit prompt
SRC_VIDEO="${SRC_VIDEO:-${1:-}}"
EDIT_PROMPT="${EDIT_PROMPT:-${2:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/edit_results_X}"

# Diff modes to run (space-separated subset). Default: all four.
DIFF_MODES="${DIFF_MODES:-direct time freq_mag freq_complex}"

SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
FRAME_RATE="${FRAME_RATE:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
# CFG/guidance scales: leave empty to auto-detect from checkpoint metadata
CFG_SCALE="${CFG_SCALE:-}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-}"
A2V_SCALE="${A2V_SCALE:-}"

# Optional: fp8-cast | fp8-scaled-mm | (empty = none)
QUANTIZATION="${QUANTIZATION:-}"

# Number of frames at the start to keep unchanged (context anchor for the retake).
# 1 = preserve just the first frame (default); increase for stronger scene anchoring.
RETAKE_START_FRAMES="${RETAKE_START_FRAMES:-10}"

# Amplitude multiplier for the diff waveform before the retake conditioning.
# Diff signals are often near-silent; scaling them up (e.g. 5-20) pushes them
# into the audio encoder's normal operating range and makes edits more pronounced.
DIFF_GAIN="${DIFF_GAIN:-1}"

# Optional: space-separated --lora flags.
# Default: IC-LoRA Union Control (Canny+Depth+Pose). Override or clear via env:
#   LORA_ARGS="--lora /other/lora.safetensors 0.8"  sbatch ...

# _DEFAULT_LORA="${CKPT_ROOT}/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
# LORA_ARGS="${LORA_ARGS:---lora ${_DEFAULT_LORA} 1.0}"
LORA_ARGS=""
echo "Using LoRA args: ${LORA_ARGS}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${SRC_VIDEO}" ]]; then
    echo "ERROR: Source video not set." >&2
    echo "Usage: sbatch submit_edit_diff.sh /path/to/video.mp4 'edit prompt'" >&2
    exit 1
fi
if [[ -z "${EDIT_PROMPT}" ]]; then
    echo "ERROR: Edit prompt not set." >&2
    echo "Usage: sbatch submit_edit_diff.sh /path/to/video.mp4 'edit prompt'" >&2
    exit 1
fi

echo "======================================================="
echo "  Job ID              : ${SLURM_JOB_ID}"
echo "  Source video        : ${SRC_VIDEO}"
echo "  Edit prompt         : ${EDIT_PROMPT}"
echo "  Diff modes          : ${DIFF_MODES}"
echo "  Retake start frames : ${RETAKE_START_FRAMES}"
echo "  Diff gain           : ${DIFF_GAIN}"
echo "  Output dir          : ${OUTPUT_DIR}"
[[ -n "${LORA_ARGS}" ]] && echo "  LoRA args           : ${LORA_ARGS}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load opencv cuda/12.9
source "${REPO_ROOT}/.venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Build optional arguments
# ---------------------------------------------------------------------------
SHAPE_ARGS=""
[[ -n "${HEIGHT}" ]]     && SHAPE_ARGS+=" --height ${HEIGHT}"
[[ -n "${WIDTH}" ]]      && SHAPE_ARGS+=" --width ${WIDTH}"
[[ -n "${NUM_FRAMES}" ]] && SHAPE_ARGS+=" --num-frames ${NUM_FRAMES}"
[[ -n "${FRAME_RATE}" ]] && SHAPE_ARGS+=" --frame-rate ${FRAME_RATE}"
# CFG/guidance scales only passed if explicitly set; otherwise auto-detected from checkpoint
[[ -n "${CFG_SCALE}" ]]       && SHAPE_ARGS+=" --cfg-scale ${CFG_SCALE}"
[[ -n "${AUDIO_CFG_SCALE}" ]] && SHAPE_ARGS+=" --audio-cfg-scale ${AUDIO_CFG_SCALE}"
[[ -n "${A2V_SCALE}" ]]       && SHAPE_ARGS+=" --a2v-scale ${A2V_SCALE}"

QUANT_ARG=""
[[ -n "${QUANTIZATION}" ]] && QUANT_ARG="--quantization ${QUANTIZATION}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "Running edit_with_audio_diff for modes: ${DIFF_MODES}..."
python "${REPO_ROOT}/editing/edit_with_audio_diff.py" \
    --src-video              "${SRC_VIDEO}" \
    --edit-prompt            "${EDIT_PROMPT}" \
    --output-dir             "${OUTPUT_DIR}" \
    --diff-modes             ${DIFF_MODES} \
    --checkpoint-path        "${CKPT_ROOT}/ltx-2.3-22b-dev.safetensors" \
    --gemma-root             "${GEMMA_ROOT}" \
    --seed                   "${SEED}" \
    --num-inference-steps    "${NUM_INFERENCE_STEPS}" \
    --retake-start-frames    "${RETAKE_START_FRAMES}" \
    --diff-gain              "${DIFF_GAIN}" \
    ${SHAPE_ARGS} \
    ${QUANT_ARG} \
    ${LORA_ARGS}

echo "edit_with_audio_diff finished with exit code $?"