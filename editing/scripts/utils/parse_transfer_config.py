#!/usr/bin/env python3
"""Parse a YAML transfer config and emit null-delimited CLI args for transfer_optimized.py.

Called by transfer.sh — not intended for direct use.

Usage (in bash):
    ARGS=()
    while IFS= read -r -d '' tok; do ARGS+=("$tok"); done \
        < <(python3 editing/scripts/utils/parse_transfer_config.py config.yaml /repo/root)
    python3 editing/transfer_optimized.py "${ARGS[@]}"
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        sys.exit(
            "ERROR: pyyaml is required to parse YAML configs.\n"
            "Install it with:  pip install pyyaml\n"
            "or:               uv pip install pyyaml"
        )


def _emit(token: str) -> None:
    sys.stdout.buffer.write(token.encode("utf-8") + b"\0")


def _slugify(text: str, words: int = 5) -> str:
    return "_".join(re.findall(r"[A-Za-z0-9]+", text.lower())[:words])


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: parse_transfer_config.py <config.yaml> [repo_root]")

    config_path = Path(sys.argv[1]).expanduser().resolve()
    if not config_path.is_file():
        sys.exit(f"ERROR: config not found: {config_path}")

    repo_root = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) > 2
        else config_path.parent.parent.parent  # configs/transfer/ → repo root
    )

    cfg = _load_yaml(config_path)

    def val(key: str, default=None):
        return cfg.get(key, default)

    def e(flag: str, key: str, default=None, *, skip_none: bool = True) -> None:
        v = val(key, default)
        if skip_none and (v is None or str(v).strip() == ""):
            return
        _emit(flag)
        _emit(str(v))

    # ------------------------------------------------------------------ required

    target_video = val("target_video", "")
    if not target_video:
        sys.exit("Config must define 'target_video'.")
    if not Path(target_video).is_absolute():
        target_video = str(repo_root / target_video)
    _emit("--target-video")
    _emit(target_video)

    opt_dir = val("opt_dir", "")
    if not opt_dir:
        sys.exit("Config must define 'opt_dir' (path to mode_both / mode_text / mode_audio dir).")
    if not Path(opt_dir).is_absolute():
        opt_dir = str(repo_root / opt_dir)
    _emit("--opt-dir")
    _emit(opt_dir)

    # ------------------------------------------------------------------ model paths

    ckpt_root  = os.environ.get("CKPT_ROOT")  or val("ckpt_root",  "")
    qwen_root  = os.environ.get("QWEN_ROOT")  or val("qwen_root",  "")
    gemma_root = os.environ.get("GEMMA_ROOT") or val("gemma_root", "")

    if not ckpt_root:
        sys.exit("Set CKPT_ROOT env var or 'ckpt_root' in the config.")
    if not qwen_root:
        sys.exit("Set QWEN_ROOT env var or 'qwen_root' in the config.")
    if not gemma_root:
        sys.exit("Set GEMMA_ROOT env var or 'gemma_root' in the config.")

    _emit("--checkpoint-path")
    _emit(str(Path(ckpt_root) / "ltx-2.3-22b-dev.safetensors"))
    _emit("--qwen-model")
    _emit(str(qwen_root))
    _emit("--gemma-root")
    _emit(str(gemma_root))

    # ------------------------------------------------------------------ output dir

    output_dir = os.environ.get("OUTPUT_DIR") or val("output_dir")
    if not output_dir:
        edit_prompt = val("edit_prompt", "") or ""
        prompt_slug = _slugify(edit_prompt) if edit_prompt else "transfer"
        exp_name = val("experiment_name", "transfer")
        output_dir = str(repo_root / "results" / "transfer" / prompt_slug / exp_name)
    _emit("--output-dir")
    _emit(str(output_dir))

    # ------------------------------------------------------------------ transfer mode

    e("--mode", "mode", "both")

    # ------------------------------------------------------------------ prompts

    e("--edit-prompt",     "edit_prompt")
    e("--static-prompt",   "static_prompt")
    e("--negative-prompt", "negative_prompt")

    # ------------------------------------------------------------------ inference

    e("--seed",                "seed",                42)
    e("--num-inference-steps", "num_inference_steps", 30)
    e("--retake-start-frames", "retake_start_frames", 5)
    e("--qwen-eval-frames",    "qwen_eval_frames",    16)

    # ------------------------------------------------------------------ video dims

    e("--height",     "height",     320)
    e("--width",      "width",      512)
    e("--num-frames", "num_frames", 95)
    e("--frame-rate", "frame_rate")  # skipped if null

    # ------------------------------------------------------------------ runtime

    e("--quantization", "quantization", "fp8-cast")

    if val("enhance_prompt", True):
        _emit("--enhance-prompt")

    # ------------------------------------------------------------------ CLIP diagnostics

    clip_model = val("clip_model", "openai/clip-vit-base-patch32")
    _emit("--clip-model")
    _emit(str(clip_model))

    e("--clip-max-frames", "clip_max_frames", 0)

    if val("clip_diag", True):
        _emit("--clip-diag")
    else:
        _emit("--no-clip-diag")

    # ------------------------------------------------------------------ optional scale overrides

    e("--cfg-scale",       "cfg_scale")
    e("--audio-cfg-scale", "audio_cfg_scale")
    e("--a2v-scale",       "a2v_scale")


if __name__ == "__main__":
    main()
