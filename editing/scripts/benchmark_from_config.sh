#!/bin/bash
# =============================================================================
# Replay a Qwen optimization run in benchmark (timing) mode.
#
# Reads a run_config.json saved by a previous optimize_qwen_vl.py run,
# reuses its exact argv, swaps only --output-dir, and runs
# optimize_qwen_vl_benchmark.py — which strips W&B / preview / CLIP paths
# so wall-clock time reflects only model forward/backward passes.
#
# Usage:
#   bash editing/scripts/benchmark_from_config.sh /path/to/run_config.json
#
# Dry-run (no GPU required):
#   DRY_RUN=1 bash editing/scripts/benchmark_from_config.sh /path/to/run_config.json
#
# Useful overrides:
#   OUTPUT_ROOT=/path/to/benchmarks bash ...
#   OUTPUT_DIR=/exact/output/dir bash ...
#   RUN_NAME=my_replay_name bash ...
# =============================================================================

set -euo pipefail

CONFIG_JSON="${1:-${CONFIG_JSON:-}}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BENCHMARK_SCRIPT="${REPO_ROOT}/editing/optimize_qwen_vl_benchmark.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/benchmark_from_config}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${CONFIG_JSON}" ]]; then
    echo "ERROR: provide a run_config.json path." >&2
    echo "Usage: bash $0 /path/to/run_config.json" >&2
    exit 1
fi

if [[ "${CONFIG_JSON}" != /* ]]; then
    CONFIG_JSON="${REPO_ROOT}/${CONFIG_JSON}"
fi

if [[ ! -f "${CONFIG_JSON}" ]]; then
    echo "ERROR: run_config.json not found: ${CONFIG_JSON}" >&2
    exit 1
fi

if [[ ! -f "${BENCHMARK_SCRIPT}" ]]; then
    echo "ERROR: benchmark script not found: ${BENCHMARK_SCRIPT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse saved config → benchmark args (Python inline script)
# ---------------------------------------------------------------------------
BENCH_ARGS=()
while IFS= read -r -d '' token; do
    BENCH_ARGS+=("${token}")
done < <("${PYTHON_BIN}" - "${CONFIG_JSON}" "${OUTPUT_ROOT}" <<'PY'
import json
import os
import re
import sys
import time
from pathlib import Path

config_path = Path(sys.argv[1]).expanduser().resolve()
output_root = Path(sys.argv[2]).expanduser()

with config_path.open() as f:
    config = json.load(f)

argv = config.get("argv") or []
if len(argv) < 2:
    raise SystemExit(
        "run_config.json does not contain a usable argv list. "
        "This replay script expects configs produced by optimize_qwen_vl.py."
    )

entrypoint = Path(str(argv[0])).name
if entrypoint != "optimize_qwen_vl.py":
    raise SystemExit(
        f"Unsupported config entrypoint {entrypoint!r}. "
        "Use a normal Qwen config produced by editing/optimize_qwen_vl.py."
    )

args_obj = config.get("args") or {}
args = list(argv[1:])


def slugify(value: object, default: str = "run") -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(value or "").lower())
    slug = "_".join(words).strip("_")
    return slug or default


def set_flag_value(tokens: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(tokens):
        if tokens[i] == flag:
            out.extend([flag, value])
            replaced = True
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    if not replaced:
        out.extend([flag, value])
    return out


def remove_boolean_flag(tokens: list[str], flag: str) -> list[str]:
    return [tok for tok in tokens if tok != flag]


old_output = args_obj.get("output_dir") or (config.get("derived") or {}).get("output_dir") or ""
old_parts = [p for p in Path(str(old_output)).parts if p not in {"", "/"}]
source_tag = "__".join(old_parts[-3:]) if old_parts else "config_replay"
source_tag = slugify(source_tag, "config_replay")

prompt_slug  = slugify(args_obj.get("edit_prompt"), "prompt")
mode_slug    = slugify(args_obj.get("opt_mode"), "mode")
seed         = args_obj.get("seed", "seed")
qwen_frames  = args_obj.get("qwen_max_frames", "qwen")
sample_mode  = slugify(args_obj.get("qwen_sample_mode"), "sample")

if os.environ.get("OUTPUT_DIR"):
    new_output = Path(os.environ["OUTPUT_DIR"]).expanduser()
else:
    run_name = os.environ.get("RUN_NAME")
    if not run_name:
        run_name = time.strftime("bench_%Y%m%d_%H%M%S")
    run_name = slugify(run_name, "benchmark")
    new_output = (
        output_root
        / prompt_slug
        / f"mode_{mode_slug}"
        / f"qwen_frames_{qwen_frames}_{sample_mode}_seed_{seed}"
        / f"{source_tag}__{run_name}"
    )

args = set_flag_value(args, "--output-dir", str(new_output))

# Ensure the final video is rendered even in benchmark mode
args = remove_boolean_flag(args, "--no-save-final-videos")
if "--save-final-videos" not in args:
    args.append("--save-final-videos")

for token in args:
    sys.stdout.buffer.write(str(token).encode("utf-8") + b"\0")
PY
)

OUTPUT_DIR_SELECTED=""
for ((i = 0; i < ${#BENCH_ARGS[@]}; i++)); do
    if [[ "${BENCH_ARGS[$i]}" == "--output-dir" && $((i + 1)) -lt ${#BENCH_ARGS[@]} ]]; then
        OUTPUT_DIR_SELECTED="${BENCH_ARGS[$((i + 1))]}"
        break
    fi
done

echo "========================================================"
echo "  Qwen benchmark replay from run_config.json"
echo "  Config     : ${CONFIG_JSON}"
echo "  Output dir : ${OUTPUT_DIR_SELECTED}"
echo "  DRY_RUN    : ${DRY_RUN}"
echo "========================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Command: %q ' "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}" "${BENCH_ARGS[@]}"
    printf '\n'
    exit 0
fi

# ---------------------------------------------------------------------------
# GPU check + environment
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no visible GPU detected." >&2
    exit 1
fi

if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi
if type module >/dev/null 2>&1; then
    module load opencv cuda tensorboard 2>/dev/null || true
fi

VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export WANDB_DISABLED=1
export WANDB_MODE=disabled

mkdir -p "${OUTPUT_DIR_SELECTED}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" "${BENCHMARK_SCRIPT}" "${BENCH_ARGS[@]}"

echo ""
echo "Benchmark complete. Outputs: ${OUTPUT_DIR_SELECTED}"
