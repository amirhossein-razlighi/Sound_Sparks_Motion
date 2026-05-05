#!/usr/bin/env python3
"""Replay the best checkpoint from an experiment and save per-frame LTX attention PNGs.

This is an ablation-only script. It does not resume optimization and it does
not edit the optimization code path. Given a finished experiment directory
containing ``run_config.json`` and one or more ``mode_*`` subdirectories, it:

1. selects the best saved mode by ``qwen_score`` / ``clip_score``;
2. rebuilds the same retake inputs;
3. injects the saved best audio latent and/or text delta;
4. renders once while hooking LTX cross-attention;
5. saves A2V and T2V overlay PNGs every K frames.

Examples
--------
    python editing/visualize_best_cross_attention_per_frame.py \\
        --experiment-dir results/QwenVL/a_dog_yawning/ablation \\
        --every-k 5

    python editing/visualize_best_cross_attention_per_frame.py \\
        --opt-dir results/QwenVL/a_dog_yawning/ablation/mode_audio \\
        --mode audio \\
        --every-k 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

_CKPT_ROOT = "/project/def-amahdavi/amirrz/LTX-2/checkpoints"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized/"

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "ltx-pipelines" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "ltx-core" / "src"))

log = logging.getLogger(__name__)

_retake_module = None


def _load_heavy_imports() -> dict[str, Any]:
    """Import LTX modules lazily so ``--help`` works in light environments."""
    import ltx_pipelines.retake as retake_module
    from ltx_pipelines.utils.constants import detect_params

    from audio_latent_opt.core import (
        _parse_loras,
        build_cached_source_latents,
        build_guiders_for_mode,
        compute_target_shape,
        flatten_video_chunks,
    )
    from audio_latent_opt.models import build_retake_pipeline, resolve_quantization_policy
    from audio_latent_opt.multimodal_loop import pre_encode_base_contexts
    from audio_latent_opt.per_frame_attn_vis import (
        PerFrameLTXAttentionCapture,
        frames_to_uint8_np,
        save_per_frame_attention_pngs,
    )
    from audio_latent_opt.runtime import build_retake_kwargs, prepare_retake_input_video
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput

    return {
        "_retake_module": retake_module,
        "detect_params": detect_params,
        "_parse_loras": _parse_loras,
        "build_cached_source_latents": build_cached_source_latents,
        "build_guiders_for_mode": build_guiders_for_mode,
        "compute_target_shape": compute_target_shape,
        "flatten_video_chunks": flatten_video_chunks,
        "build_retake_pipeline": build_retake_pipeline,
        "resolve_quantization_policy": resolve_quantization_policy,
        "pre_encode_base_contexts": pre_encode_base_contexts,
        "PerFrameLTXAttentionCapture": PerFrameLTXAttentionCapture,
        "frames_to_uint8_np": frames_to_uint8_np,
        "save_per_frame_attention_pngs": save_per_frame_attention_pngs,
        "build_retake_kwargs": build_retake_kwargs,
        "prepare_retake_input_video": prepare_retake_input_video,
        "EmbeddingsProcessorOutput": EmbeddingsProcessorOutput,
    }


def _load_run_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _mode_from_best_params(path: Path, payload: dict[str, Any]) -> str:
    mode = payload.get("mode")
    if mode in {"text", "audio", "both"}:
        return str(mode)
    stem = path.stem
    for candidate in ("text", "audio", "both"):
        if stem.endswith(candidate):
            return candidate
    raise ValueError(f"Could not infer mode from {path}")


def _score_payload(payload: dict[str, Any]) -> tuple[float, str]:
    if "qwen_score" in payload:
        return float(payload["qwen_score"]), "qwen_score"
    if "clip_score" in payload:
        return float(payload["clip_score"]), "clip_score"
    if "qwen_loss" in payload:
        return -float(payload["qwen_loss"]), "negative_qwen_loss"
    if "clip_loss" in payload:
        return -float(payload["clip_loss"]), "negative_clip_loss"
    return float("-inf"), "unknown"


def _discover_best_mode(
    *,
    experiment_dir: Path | None,
    opt_dir: Path | None,
    requested_mode: str,
) -> tuple[Path, str, Path, dict[str, Any], str, float]:
    search_root = opt_dir if opt_dir is not None else experiment_dir
    if search_root is None:
        raise ValueError("Provide --experiment-dir or --opt-dir.")

    if opt_dir is not None:
        param_paths = sorted(opt_dir.glob("best_params_*.pt"))
    else:
        param_paths = sorted(experiment_dir.glob("mode*/best_params_*.pt"))

    if not param_paths:
        raise FileNotFoundError(f"No best_params_*.pt files found under {search_root}")

    candidates = []
    for params_path in param_paths:
        payload = torch.load(params_path, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        mode = _mode_from_best_params(params_path, payload)
        if requested_mode != "auto" and mode != requested_mode:
            continue
        score, score_name = _score_payload(payload)
        candidates.append((score, score_name, params_path.parent, mode, params_path, payload))

    if not candidates:
        raise FileNotFoundError(
            f"No best_params_*.pt matched mode={requested_mode!r} under {search_root}"
        )

    score, score_name, mode_dir, mode, params_path, payload = max(candidates, key=lambda item: item[0])
    return mode_dir.resolve(), mode, params_path.resolve(), payload, score_name, float(score)


def _find_tensor(mode_dir: Path, pattern: str, required: bool) -> Path | None:
    matches = sorted(mode_dir.glob(pattern))
    if not matches:
        if required:
            raise FileNotFoundError(f"Required tensor not found: {mode_dir / pattern}")
        return None
    return matches[0]


def _interpolate_audio_latent(latent: torch.Tensor, target_t: int) -> torch.Tensor:
    if latent.shape[2] == target_t:
        return latent
    x = latent.permute(0, 1, 3, 2)
    x = F.interpolate(x, size=(x.shape[2], target_t), mode="bilinear", align_corners=False)
    return x.permute(0, 1, 3, 2)


def _interpolate_text_delta(delta: torch.Tensor, target_seq: int) -> torch.Tensor:
    if delta.shape[1] == target_seq:
        return delta
    x = delta.float().permute(0, 2, 1)
    x = F.interpolate(x, size=target_seq, mode="linear", align_corners=False)
    return x.permute(0, 2, 1)


def _get_config_value(cli_value, config_args: dict[str, Any], key: str, default=None):
    return cli_value if cli_value is not None else config_args.get(key, default)


def _build_replay_args(
    cli_args: argparse.Namespace,
    config_args: dict[str, Any],
    output_dir: Path,
) -> SimpleNamespace:
    retake_steps_override = (
        cli_args.retake_num_inference_steps
        if cli_args.retake_num_inference_steps is not None
        else cli_args.num_inference_steps
    )
    values = dict(config_args)
    values.update(
        src_video=_get_config_value(cli_args.src_video, config_args, "src_video"),
        edit_prompt=_get_config_value(cli_args.edit_prompt, config_args, "edit_prompt"),
        negative_prompt=_get_config_value(cli_args.negative_prompt, config_args, "negative_prompt", ""),
        checkpoint_path=_get_config_value(cli_args.checkpoint_path, config_args, "checkpoint_path", DEFAULT_CHECKPOINT),
        gemma_root=_get_config_value(cli_args.gemma_root, config_args, "gemma_root", DEFAULT_GEMMA_ROOT),
        height=_get_config_value(cli_args.height, config_args, "height"),
        width=_get_config_value(cli_args.width, config_args, "width"),
        num_frames=_get_config_value(cli_args.num_frames, config_args, "num_frames"),
        frame_rate=_get_config_value(cli_args.frame_rate, config_args, "frame_rate"),
        audio_sr=_get_config_value(cli_args.audio_sr, config_args, "audio_sr", 44100),
        seed=_get_config_value(cli_args.seed, config_args, "seed", 42),
        enhance_prompt=bool(_get_config_value(cli_args.enhance_prompt, config_args, "enhance_prompt", False)),
        retake_start_frames=_get_config_value(cli_args.retake_start_frames, config_args, "retake_start_frames", 1),
        retake_num_inference_steps=_get_config_value(
            retake_steps_override,
            config_args,
            "final_retake_num_inference_steps",
            config_args.get("retake_num_inference_steps", config_args.get("num_inference_steps", 30)),
        ),
        ti2v_num_inference_steps=_get_config_value(cli_args.num_inference_steps, config_args, "num_inference_steps", 30),
        cfg_scale=_get_config_value(cli_args.cfg_scale, config_args, "cfg_scale"),
        audio_cfg_scale=_get_config_value(cli_args.audio_cfg_scale, config_args, "audio_cfg_scale"),
        a2v_scale=_get_config_value(cli_args.a2v_scale, config_args, "a2v_scale"),
        low_memory_guidance=False,
        quantization=_get_config_value(cli_args.quantization, config_args, "quantization"),
        retake_quantization=_get_config_value(cli_args.retake_quantization, config_args, "retake_quantization"),
        gradient_checkpointing=bool(
            _get_config_value(cli_args.gradient_checkpointing, config_args, "gradient_checkpointing", False)
        ),
        loras=config_args.get("loras", []),
        output_dir=str(output_dir),
    )

    missing = [key for key in ("src_video", "edit_prompt") if not values.get(key)]
    if missing:
        raise ValueError(
            "Missing required replay fields: "
            + ", ".join(missing)
            + ". Pass them explicitly or use an experiment directory with run_config.json."
        )

    return SimpleNamespace(**values)


def _render_best_frames_with_attention(
    *,
    pipeline,
    src_video: str,
    cached_video_latent: torch.Tensor,
    audio_for_render: torch.Tensor,
    pos_context,
    neg_context,
    retake_kwargs: dict,
    max_frames: int | None,
    frame_stride: int,
    capture,
    flatten_video_chunks,
) -> torch.Tensor:
    orig_encode_video = _retake_module._encode_video_for_retake
    orig_encode_audio = _retake_module._encode_audio_for_retake
    orig_encode_prompts = _retake_module.encode_prompts
    orig_video_encoder_getter = pipeline.model_ledger.video_encoder
    orig_audio_encoder_getter = pipeline.model_ledger.audio_encoder

    def _cached_video(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    def _injected_audio(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return audio_for_render

    def _patched_encode_prompts(prompts, model_ledger, **kwargs):  # noqa: ARG001
        return [pos_context, neg_context]

    _retake_module._encode_video_for_retake = _cached_video
    _retake_module._encode_audio_for_retake = _injected_audio
    _retake_module.encode_prompts = _patched_encode_prompts
    pipeline.model_ledger.video_encoder = lambda: None
    pipeline.model_ledger.audio_encoder = lambda: None

    try:
        with torch.no_grad(), capture:
            video_iter, _ = pipeline(video_path=src_video, **retake_kwargs)
        return flatten_video_chunks(
            video_iter=video_iter,
            max_frames=max_frames,
            frame_stride=frame_stride,
            resize_to=None,
            sample_start=0,
        )
    finally:
        _retake_module._encode_video_for_retake = orig_encode_video
        _retake_module._encode_audio_for_retake = orig_encode_audio
        _retake_module.encode_prompts = orig_encode_prompts
        pipeline.model_ledger.video_encoder = orig_video_encoder_getter
        pipeline.model_ledger.audio_encoder = orig_audio_encoder_getter


def _write_methodology(output_dir: Path, metadata: dict[str, Any]) -> None:
    text = f"""# Per-frame Cross-attention Ablation

## Previous Visualization

The optimization-time attention visualization captured one lightweight LTX
cross-attention vector for audio-to-video and one for text-to-video, reshaped it
to the latent video grid, and averaged over latent time before overlaying it on
frames. That produced a clip-level spatial summary: "where is this modality
important on average over the clip?"

## This Visualization

This replay loads the saved best checkpoint only; no optimizer step is run. It
reconstructs the RetakePipeline inputs, injects the saved audio latent and/or
text delta for the selected mode, renders once, and hooks LTX cross-attention
during that render.

For every hooked cross-attention call, video tokens are queries. For
``audio_to_video_attn``, keys/values are audio latent tokens. For ``attn2``,
keys/values are Gemma text-conditioning tokens. The hook reduces each video
query token to one importance value using ``{metadata["importance"]}``. Values
are averaged over heads, selected transformer blocks, and denoising calls, but
the latent time axis is kept. The final grids are ``[latent_t, latent_h,
latent_w]``.

Those grids are trilinearly upsampled to the rendered video resolution and PNGs
are saved every ``K={metadata["every_k"]}`` frames, separately for A2V and T2V.

## Replay Metadata

```json
{json.dumps(metadata, indent=2, sort_keys=True)}
```
"""
    (output_dir / "methodology.md").write_text(text)


def run(cli_args: argparse.Namespace) -> None:
    global _retake_module

    imports = _load_heavy_imports()
    _retake_module = imports["_retake_module"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    experiment_dir = Path(cli_args.experiment_dir).expanduser().resolve() if cli_args.experiment_dir else None
    opt_dir = Path(cli_args.opt_dir).expanduser().resolve() if cli_args.opt_dir else None
    if experiment_dir is None and opt_dir is not None:
        experiment_dir = opt_dir.parent
    if experiment_dir is None:
        raise ValueError("Provide --experiment-dir or --opt-dir.")

    config = _load_run_config(experiment_dir)
    config_args = dict(config.get("args", {}))

    mode_dir, mode, params_path, params_payload, score_name, score = _discover_best_mode(
        experiment_dir=experiment_dir,
        opt_dir=opt_dir,
        requested_mode=cli_args.mode,
    )
    log.info("Selected best saved mode: %s (%s=%.6f) from %s", mode, score_name, score, mode_dir)

    output_dir = (
        Path(cli_args.output_dir).expanduser().resolve()
        if cli_args.output_dir
        else experiment_dir / "attention_per_frame" / f"best_{mode_dir.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_args = _build_replay_args(cli_args, config_args, output_dir)
    height, width, num_frames, frame_rate = imports["compute_target_shape"](
        replay_args.src_video,
        replay_args.height,
        replay_args.width,
        replay_args.num_frames,
        replay_args.frame_rate,
    )
    duration = num_frames / frame_rate
    log.info("Replay shape: %dx%d, %d frames @ %.3f fps", width, height, num_frames, frame_rate)

    existing_retake = experiment_dir / "retake_input_prepared.mp4"
    if existing_retake.exists() and not cli_args.force_prepare_input:
        retake_input_video = existing_retake
        log.info("Using existing prepared retake input: %s", retake_input_video)
    else:
        retake_input_video = imports["prepare_retake_input_video"](
            args=replay_args,
            is_main=True,
            output_dir=output_dir,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retake_quant = imports["resolve_quantization_policy"](
        replay_args.retake_quantization if replay_args.retake_quantization is not None else replay_args.quantization
    )

    params = imports["detect_params"](replay_args.checkpoint_path)
    final_vg, final_ag = imports["build_guiders_for_mode"](
        args=replay_args,
        params=params,
        use_low_memory_guidance=False,
    )

    log.info("Loading RetakePipeline for replay...")
    pipeline = imports["build_retake_pipeline"](
        checkpoint_path=replay_args.checkpoint_path,
        gemma_root=replay_args.gemma_root,
        loras=imports["_parse_loras"](replay_args.loras),
        device=device,
        quant_policy=retake_quant,
        gradient_checkpointing=replay_args.gradient_checkpointing,
    )

    cached_video_latent, base_audio_latent, _waveform_sr = imports["build_cached_source_latents"](
        pipeline=pipeline,
        src_video=str(retake_input_video),
        height=height,
        width=width,
        num_frames=num_frames,
        audio_sr=replay_args.audio_sr,
        device=device,
    )
    log.info("Cached video latent: %s", tuple(cached_video_latent.shape))
    log.info("Base audio latent: %s", tuple(base_audio_latent.shape))

    base_pos_context, base_neg_context = imports["pre_encode_base_contexts"](
        pipeline=pipeline,
        pos_prompt=replay_args.edit_prompt,
        neg_prompt=replay_args.negative_prompt,
        device=device,
    )

    audio_for_render = base_audio_latent
    if mode in {"audio", "both"}:
        audio_path = _find_tensor(mode_dir, f"best_audio_latent_{mode}.pt", required=False)
        if audio_path is None:
            audio_path = _find_tensor(mode_dir, "best_audio_latent_*.pt", required=True)
        saved_audio = torch.load(audio_path, map_location="cpu")
        saved_audio = _interpolate_audio_latent(saved_audio, base_audio_latent.shape[2])
        audio_for_render = saved_audio.to(device=device, dtype=base_audio_latent.dtype)
        log.info("Injected best audio latent from %s", audio_path)

    pos_context = base_pos_context
    if mode in {"text", "both"}:
        delta_path = _find_tensor(mode_dir, f"best_text_delta_{mode}.pt", required=False)
        if delta_path is None:
            delta_path = _find_tensor(mode_dir, "best_text_delta_*.pt", required=True)
        saved_delta = torch.load(delta_path, map_location="cpu")
        saved_delta = _interpolate_text_delta(saved_delta, base_pos_context.video_encoding.shape[1])
        pos_context = imports["EmbeddingsProcessorOutput"](
            video_encoding=base_pos_context.video_encoding
            + saved_delta.to(device=device, dtype=base_pos_context.video_encoding.dtype),
            audio_encoding=base_pos_context.audio_encoding,
            attention_mask=base_pos_context.attention_mask,
        )
        log.info("Injected best text delta from %s", delta_path)

    retake_kwargs = imports["build_retake_kwargs"](
        args=replay_args,
        frame_rate=frame_rate,
        duration=duration,
        video_guider_params=final_vg,
        audio_guider_params=final_ag,
    )
    retake_kwargs["num_inference_steps"] = replay_args.retake_num_inference_steps

    block_fractions = tuple(float(x.strip()) for x in cli_args.block_fractions.split(",") if x.strip())
    capture = imports["PerFrameLTXAttentionCapture"](
        pipeline,
        block_fractions=block_fractions,
        importance=cli_args.importance,
        query_chunk_size=cli_args.query_chunk_size,
    )

    log.info("Rendering selected best checkpoint once and capturing per-frame cross-attention...")
    frames = _render_best_frames_with_attention(
        pipeline=pipeline,
        src_video=str(retake_input_video),
        cached_video_latent=cached_video_latent,
        audio_for_render=audio_for_render,
        pos_context=pos_context,
        neg_context=base_neg_context,
        retake_kwargs=retake_kwargs,
        max_frames=cli_args.max_frames,
        frame_stride=cli_args.frame_stride,
        capture=capture,
        flatten_video_chunks=imports["flatten_video_chunks"],
    )
    if frames.shape[0] == 0:
        raise RuntimeError("Replay produced no frames.")

    frames_np = imports["frames_to_uint8_np"](frames)
    metadata = {
        "experiment_dir": str(experiment_dir),
        "mode_dir": str(mode_dir),
        "mode": mode,
        "best_params_path": str(params_path),
        "best_params": params_payload,
        "score_name": score_name,
        "score": score,
        "edit_prompt": replay_args.edit_prompt,
        "src_video": replay_args.src_video,
        "retake_input_video": str(retake_input_video),
        "rendered_frames": int(frames_np.shape[0]),
        "rendered_height": int(frames_np.shape[1]),
        "rendered_width": int(frames_np.shape[2]),
        "latent_shape": list(cached_video_latent.shape),
        "block_fractions": list(block_fractions),
        "hooked_blocks": capture.hooked_blocks,
        "capture_counts": capture.counts(),
        "importance": cli_args.importance,
        "every_k": int(cli_args.every_k),
        "alpha": float(cli_args.alpha),
    }

    saved_count = 0
    for key, dirname in [
        ("audio_to_video", "audio_to_video"),
        ("text_to_video", "text_to_video"),
    ]:
        grid = capture.get_grid(key, cached_video_latent)
        if grid is None:
            log.warning("No %s attention grid captured.", key)
            continue
        modality_metadata = dict(metadata)
        modality_metadata["modality"] = key
        modality_metadata["attention_grid_shape"] = list(grid.shape)
        saved = imports["save_per_frame_attention_pngs"](
            frames_np=frames_np,
            heatmap_grid=grid,
            output_dir=output_dir,
            modality=dirname,
            every_k=cli_args.every_k,
            alpha=cli_args.alpha,
            metadata=modality_metadata,
        )
        saved_count += len(saved)
        log.info("Saved %d PNGs for %s under %s/%s", len(saved), key, output_dir, dirname)

    (output_dir / "replay_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    _write_methodology(output_dir, metadata)

    if cli_args.save_render_video:
        try:
            import imageio.v2 as imageio

            video_path = output_dir / f"best_replay_{mode}.mp4"
            imageio.mimsave(video_path, list(frames_np), fps=int(round(frame_rate)), macro_block_size=1)
            log.info("Saved replay video without audio: %s", video_path)
        except Exception:
            log.warning("Could not save replay mp4.", exc_info=True)

    log.info("Done. Saved %d attention PNGs under %s", saved_count, output_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--experiment-dir", help="Run root containing run_config.json and mode_* subdirs.")
    src.add_argument("--opt-dir", help="Specific mode directory containing best_params_*.pt.")

    p.add_argument("--mode", default="auto", choices=["auto", "text", "audio", "both"])
    p.add_argument("--output-dir", default="")
    p.add_argument("--every-k", type=int, default=5, help="Save PNGs every K rendered frames.")
    p.add_argument("--alpha", type=float, default=0.55, help="Overlay opacity.")
    p.add_argument(
        "--block-fractions",
        default="0.5",
        help="Comma-separated transformer block fractions to hook, e.g. 0.25,0.5,0.75.",
    )
    p.add_argument("--importance", default="output_norm", choices=["output_norm", "max_prob", "neg_entropy", "max_logit"])
    p.add_argument("--query-chunk-size", type=int, default=1024)
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on decoded replay frames. Default: all frames.")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--save-render-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-prepare-input", action="store_true", help="Rebuild retake_input_prepared.mp4.")

    # Optional replay overrides. Defaults are read from run_config.json.
    p.add_argument("--src-video", default=None)
    p.add_argument("--edit-prompt", default=None)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--frame-rate", type=float, default=None)
    p.add_argument("--audio-sr", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--retake-start-frames", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--retake-num-inference-steps", type=int, default=None)
    p.add_argument("--enhance-prompt", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--audio-cfg-scale", type=float, default=None)
    p.add_argument("--a2v-scale", type=float, default=None)
    p.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--retake-quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--gemma-root", default=None)

    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
