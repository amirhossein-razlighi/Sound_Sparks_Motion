#!/usr/bin/env python3
"""Parse a YAML experiment config and emit null-delimited CLI args for optimize_qwen_vl.py.

Called by run.sh — not intended for direct use.

Usage (in bash):
    ARGS=()
    while IFS= read -r -d '' tok; do ARGS+=("$tok"); done \
        < <(python3 editing/scripts/utils/parse_config.py config.yaml /repo/root)
    python3 editing/optimize_qwen_vl.py "${ARGS[@]}"
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # pyyaml
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
        sys.exit("Usage: parse_config.py <config.yaml> [repo_root]")

    config_path = Path(sys.argv[1]).expanduser().resolve()
    if not config_path.is_file():
        sys.exit(f"ERROR: config not found: {config_path}")

    repo_root = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) > 2
        else config_path.parent.parent
    )

    cfg = _load_yaml(config_path)

    # ------------------------------------------------------------------ helpers

    def val(key: str, default=None):
        return cfg.get(key, default)

    def e(flag: str, key: str, default=None, *, skip_none: bool = True) -> None:
        v = val(key, default)
        if skip_none and (v is None or str(v).strip() == ""):
            return
        _emit(flag)
        _emit(str(v))

    # ------------------------------------------------------------------ required

    src_video = val("src_video", "")
    if not src_video:
        sys.exit("Config must define 'src_video'.")
    if not Path(src_video).is_absolute():
        src_video = str(repo_root / src_video)
    _emit("--src-video")
    _emit(src_video)

    edit_prompt = val("edit_prompt", "")
    if not edit_prompt:
        sys.exit("Config must define 'edit_prompt'.")
    _emit("--edit-prompt")
    _emit(edit_prompt)

    # ------------------------------------------------------------------ output dir

    output_dir = (
        os.environ.get("OUTPUT_DIR")
        or val("output_dir")
    )
    if not output_dir:
        prompt_slug = _slugify(edit_prompt)
        exp_name = val("experiment_name", "exp")
        output_dir = str(repo_root / "results" / "QwenVL" / prompt_slug / exp_name)
    _emit("--output-dir")
    _emit(str(output_dir))

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

    # ------------------------------------------------------------------ prompts

    e("--static-prompt",   "static_prompt")
    e("--negative-prompt", "negative_prompt")

    # ------------------------------------------------------------------ experiment

    e("--opt-mode", "opt_mode", "both")
    e("--seed",     "seed",     42)

    # ------------------------------------------------------------------ video dims

    e("--height",     "height",     320)
    e("--width",      "width",      512)
    e("--num-frames", "num_frames", 95)
    e("--frame-rate", "frame_rate")  # skipped if null

    # ------------------------------------------------------------------ optimisation

    e("--iterations",                          "iterations",                          30)
    e("--early-stopping",                      "early_stopping",                      15)
    e("--lr",                                  "lr",                                  0.005)
    e("--lr-schedule",                         "lr_schedule",                         "cosine")
    e("--grad-clip",                           "grad_clip",                           0.0)
    e("--best-min-loss-delta",                 "best_min_loss_delta",                 0.0)
    e("--retake-start-frames",                 "retake_start_frames",                 5)
    e("--num-inference-steps",                 "num_inference_steps",                 30)
    e("--retake-num-inference-steps",          "retake_num_inference_steps",          30)
    e("--final-retake-num-inference-steps",    "final_retake_num_inference_steps",    30)
    e("--audio-opt-last-steps",                "audio_opt_last_steps",                8)

    # ------------------------------------------------------------------ qwen

    e("--qwen-max-frames",             "qwen_max_frames",             8)
    e("--qwen-img-size",               "qwen_img_size",               224)
    e("--qwen-sample-mode",            "qwen_sample_mode",            "linspace")
    e("--qwen-contiguous-start-frame", "qwen_contiguous_start_frame", 4)
    e("--qwen-gradient-rubric",        "qwen_gradient_rubric",        "motion")
    e("--qwen-grad-accum-steps",       "qwen_grad_accum_steps",       3)

    motion_q = val("qwen_motion_question")
    if not motion_q:
        motion_q = (
            f'Does this video clearly show the action or state change described by '
            f'the edit prompt: "{edit_prompt}"? Answer only \'yes\' or \'no\'.'
        )
    _emit("--qwen-motion-question")
    _emit(motion_q)

    # ------------------------------------------------------------------ perceptual reg

    e("--lpips-weight",      "lpips_weight",      0.0)
    e("--temporal-weight",   "temporal_weight",   0.0)
    e("--lpips-backbone",    "lpips_backbone",    "alex")
    e("--latent-reg-weight", "latent_reg_weight", 0.01)
    e("--text-reg-weight",   "text_reg_weight",   0.001)
    e("--reg-schedule",      "reg_schedule",      "cosine_increase")

    # ------------------------------------------------------------------ diagnostics

    e("--clip-similarity-diag-model",       "clip_model",      "openai/clip-vit-base-patch32")
    e("--clip-similarity-diag-max-frames",  "clip_max_frames", 0)
    e("--clip-similarity-diag-batch-size",  "clip_batch_size", 8)
    e("--max-eval-frames",                  "max_eval_frames", 95)
    e("--frame-stride",                     "frame_stride",    1)
    e("--visualize-every-iters",            "visualize_every_iters", 5)

    # ------------------------------------------------------------------ runtime

    e("--quantization", "quantization", "fp8-cast")

    if val("enhance_prompt", True):
        _emit("--enhance-prompt")

    if val("low_memory_guidance", True):
        _emit("--low-memory-guidance")
    else:
        _emit("--no-low-memory-guidance")

    if val("save_final_videos", True):
        _emit("--save-final-videos")
    else:
        _emit("--no-save-final-videos")

    if val("clip_similarity_diag", True):
        _emit("--clip-similarity-diag")
    else:
        _emit("--no-clip-similarity-diag")

    if val("gradient_checkpointing", True):
        _emit("--gradient-checkpointing")

    if val("resume", False):
        _emit("--resume")

    if val("render_without_quantization", False):
        _emit("--render-without-quantization")

    # optional scale overrides
    e("--cfg-scale",       "cfg_scale")
    e("--audio-cfg-scale", "audio_cfg_scale")
    e("--a2v-scale",       "a2v_scale")

    # wandb — project and tags are CLI args; mode is exported as WANDB_MODE by run.sh
    e("--wandb-project", "wandb_project", "sound-sparks-motion")
    e("--wandb-tags",    "wandb_tags")


if __name__ == "__main__":
    main()
