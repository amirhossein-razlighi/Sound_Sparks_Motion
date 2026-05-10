#!/usr/bin/env python3
"""
High-resolution transfer: re-run LTX retake with optimized latents at the
source video's NATIVE resolution instead of the downscaled optimization resolution.

Hypothesis: artifacts visible at 320×512 may disappear at native resolution
because the model has more pixels / spatial tokens to work with.

The optimized text delta and/or audio latent are spatially interpolated to
match the new resolution's sequence length before inference — no manual
reshaping needed.

Saves under <opt-dir>/hires_transfer/  (or --output-dir):
  baseline_video.mp4   — retake at native res, unmodified latents  (for comparison)
  transfer_both.mp4    — retake at native res WITH optimized latents

Memory note
-----------
At native 1080p+ resolutions the LTX-22B transformer (~22 GB fp8) leaves very
little VRAM for the VAE decoder after denoising finishes.  This script
automatically offloads the transformer to CPU before each VAE decode pass and
restores it to CUDA afterwards, freeing ~22 GB for the decode step.

Use --max-side to cap the longer video dimension (e.g. --max-side 1280) if you
still hit OOM.  Aspect ratio and 32-pixel alignment are preserved.

Usage
-----
    python editing/hires_transfer.py \\
        --opt-dir  results/QwenVL/rabbit/.../rsf1_accum1_... \\
        --src-video input_videos/rabbit.mp4
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import transfer_optimized  # noqa: E402

log = logging.getLogger(__name__)


def _install_memory_hooks(vae_tile_size: int = 256, vae_temporal_tile_frames: int = 16) -> None:
    """Three monkey-patches to stay within 80 GB VRAM during VAE decode:

    1. build_retake_kwargs → injects a custom TilingConfig so the VAE decoder
       uses smaller tiles.  The key lever is temporal: default 64 frames gives
       intermediate feature maps of ~4 GB/layer (64 frames × 256 px × 256 px ×
       64 channels × 2 bytes); 16 frames drops that to ~1 GB/layer, bringing
       peak per-tile memory from ~30 GB down to ~8 GB.

    2. encode_video → calls gc.collect() + empty_cache() BEFORE the generator
       is first consumed so the LTX-22B transformer cache is evicted first,
       then consumes the lazy decode generator under torch.inference_mode()
       so the VAE decode does not build autograd graphs.

    3. VideoDecoder._accumulate_temporal_group_into_buffer → calls
       gc.collect() + empty_cache() before every temporal group so the PyTorch
       allocator pool is flushed between groups and doesn't accumulate.
    """
    from motion_opt.runtime import build_retake_kwargs as _real_build_kwargs
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig
    from ltx_core.model.video_vae.video_vae import VideoDecoder
    from ltx_pipelines.utils.media_io import encode_video as _real_encode_video

    # Temporal tile must be ≥ 16 and divisible by 8.
    # Spatial overlap kept at 64 px; temporal overlap = tile_size // 4, clamped to [8, 24].
    t_overlap = min(24, max(8, (vae_temporal_tile_frames // 4) // 8 * 8))
    _tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=vae_tile_size,
            tile_overlap_in_pixels=64,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=vae_temporal_tile_frames,
            tile_overlap_in_frames=t_overlap,
        ),
    )
    log.info(
        "VAE tiling: spatial=%d px  temporal=%d frames (overlap=%d)",
        vae_tile_size, vae_temporal_tile_frames, t_overlap,
    )

    # Hook 1: inject tiling_config into retake kwargs
    def _kwargs_with_tiling(*args, **kwargs):
        result = _real_build_kwargs(*args, **kwargs)
        result["tiling_config"] = _tiling
        log.info("build_retake_kwargs: injected tiling_config=%r", _tiling)
        return result

    transfer_optimized.build_retake_kwargs = _kwargs_with_tiling

    # Hook 2: flush transformer cache before VAE decode (first next() call)
    def _mem_managed_encode_video(video, **kwargs):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1e9
            free  = (torch.cuda.get_device_properties(0).total_memory
                     - torch.cuda.memory_reserved(0)) / 1e9
            log.info(
                "Pre-decode VRAM: allocated=%.2f GB  cuda_free=%.2f GB", alloc, free,
            )

        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            alloc2 = torch.cuda.memory_allocated(0) / 1e9
            free2  = (torch.cuda.get_device_properties(0).total_memory
                      - torch.cuda.memory_reserved(0)) / 1e9
            log.info(
                "Post-flush VRAM: allocated=%.2f GB  cuda_free=%.2f GB  (freed %.2f GB)",
                alloc2, free2, alloc - alloc2,
            )

        with torch.inference_mode():
            return _real_encode_video(video, **kwargs)

    transfer_optimized.encode_video = _mem_managed_encode_video

    # Hook 3: flush allocator pool before each temporal tile group so per-group
    # peak activations (~8 GB at 16 frames / 256 px) don't accumulate across groups.
    _orig_accumulate = VideoDecoder._accumulate_temporal_group_into_buffer

    def _accumulate_with_cache_flush(self, group_tiles, buffer, latent, timestep, generator):
        gc.collect()
        torch.cuda.empty_cache()
        return _orig_accumulate(self, group_tiles, buffer, latent, timestep, generator)

    VideoDecoder._accumulate_temporal_group_into_buffer = _accumulate_with_cache_flush
    log.info("Installed VideoDecoder cache-flush hook between temporal tile groups.")


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _cap_resolution(native_h: int, native_w: int, max_side: int) -> tuple[int, int]:
    """Scale down so the longer side equals max_side, preserving aspect ratio.
    Both dimensions are rounded to the nearest multiple of 32 (LTX requirement).
    Returns (height, width) or (None, None) if already within max_side.
    """
    if max(native_h, native_w) <= max_side:
        return None, None  # no cap needed, use native resolution
    scale = max_side / max(native_h, native_w)
    h = max(32, int(native_h * scale) // 32 * 32)
    w = max(32, int(native_w * scale) // 32 * 32)
    return h, w


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Required ----
    p.add_argument(
        "--opt-dir", required=True,
        help="Top-level optimization output dir — the one containing prompt.txt "
             "and mode_both/ subdirectory.",
    )
    p.add_argument(
        "--src-video", required=True,
        help="Source video at its NATIVE resolution. "
             "Height/width are read from this file — no resizing applied.",
    )

    # ---- Optional overrides ----
    p.add_argument("--mode", default="both", choices=["text", "audio", "both"])
    p.add_argument("--output-dir", default=None,
                   help="Where to write the videos "
                        "(default: <opt-dir>/hires_transfer/).")
    p.add_argument("--edit-prompt", default=None)
    p.add_argument("--static-prompt", default=None)
    p.add_argument("--negative-prompt", default=None)

    # ---- Resolution ----
    p.add_argument(
        "--num-frames", type=int, default=None,
        help="Trim the video to this many frames before inference. "
             "Default: use all frames from the source video. "
             "Must satisfy (N-1) %% 8 == 0, e.g. 25, 33, 41, ..., 95, 121, 193.",
    )
    p.add_argument(
        "--max-side", type=int, default=None,
        help="Cap the longer video dimension to this value (aspect-ratio preserving, "
             "32-pixel-aligned).  E.g. --max-side 1280 on a 1920×1056 video gives "
             "1280×704.  Default: no cap (full native resolution).",
    )

    # ---- Pipeline ----
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--retake-num-inference-steps", type=int, default=None)
    p.add_argument("--final-retake-num-inference-steps", type=int, default=None)
    p.add_argument("--retake-start-frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--audio-cfg-scale", type=float, default=None)
    p.add_argument("--a2v-scale", type=float, default=None)
    p.add_argument("--enhance-prompt", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--low-memory-guidance", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--quantization", default=None,
                   choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--checkpoint-path",
                   default=None)
    p.add_argument("--gemma-root",
                   default=None)

    # ---- Evaluation ----
    p.add_argument("--qwen-model", default=None,
                   help="Qwen2.5-VL model path for scoring. Empty = skip.")
    p.add_argument("--clip-diag", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument(
        "--vae-tile-size", type=int, default=256,
        help="Spatial tile size (px) for the VAE decoder (default: 256). "
             "Default 512 OOMs at 1080p+; 256 reduces peak VRAM ~4×. "
             "Must be ≥64 and divisible by 32.",
    )
    p.add_argument(
        "--vae-temporal-tile-frames", type=int, default=16,
        help="Temporal tile size (frames) for the VAE decoder (default: 16). "
             "Default 64 causes ~30 GB intermediate activations per tile; "
             "16 frames reduces that to ~8 GB. Must be ≥16 and divisible by 8.",
    )

    return p


def _read_txt(opt_dir: Path, name: str, override: str | None) -> str:
    if override is not None:
        return override
    p = opt_dir / name
    return p.read_text().strip() if p.exists() else ""


def _load_saved_run_args(opt_dir: Path) -> dict:
    cfg_path = opt_dir / "run_config.json"
    if not cfg_path.exists():
        log.info("No run_config.json found in %s; using CLI/default hires settings.", opt_dir)
        return {}

    try:
        payload = json.loads(cfg_path.read_text())
    except Exception:
        log.warning("Failed to parse %s", cfg_path, exc_info=True)
        return {}

    saved_args = payload.get("args")
    if not isinstance(saved_args, dict):
        log.warning("run_config.json at %s does not contain an args object.", cfg_path)
        return {}

    log.info("Loaded saved run config from %s", cfg_path)
    return saved_args


def _resolve_override(cli_value, saved_args: dict, *keys: str, default=None):
    if cli_value is not None:
        return cli_value
    for key in keys:
        if key in saved_args and saved_args[key] is not None:
            return saved_args[key]
    return default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = build_parser().parse_args()

    opt_dir = Path(args.opt_dir).expanduser().resolve()
    if not opt_dir.is_dir():
        log.error("--opt-dir not found: %s", opt_dir)
        sys.exit(1)

    src_video = Path(args.src_video).expanduser().resolve()
    if not src_video.is_file():
        log.error("--src-video not found: %s", src_video)
        sys.exit(1)

    # ---- Prompts ----
    edit_prompt = _read_txt(opt_dir, "prompt.txt", args.edit_prompt)
    static_prompt = _read_txt(opt_dir, "static_prompt.txt", args.static_prompt)
    negative_prompt = _read_txt(opt_dir, "negative_prompt.txt", args.negative_prompt)
    if not edit_prompt:
        log.error(
            "Edit prompt is empty — pass --edit-prompt or ensure prompt.txt "
            "exists in --opt-dir."
        )
        sys.exit(1)

    # ---- Mode subdir ----
    mode_dir = opt_dir / f"mode_{args.mode}"
    if not mode_dir.is_dir():
        log.error("Mode subdir not found: %s", mode_dir)
        sys.exit(1)

    saved_args = _load_saved_run_args(opt_dir)

    num_frames = _resolve_override(args.num_frames, saved_args, "num_frames", default=None)
    num_inference_steps = _resolve_override(args.num_inference_steps, saved_args, "num_inference_steps", default=30)
    retake_num_inference_steps = _resolve_override(
        args.retake_num_inference_steps,
        saved_args,
        "retake_num_inference_steps",
        "num_inference_steps",
        default=num_inference_steps,
    )
    final_retake_num_inference_steps = _resolve_override(
        args.final_retake_num_inference_steps,
        saved_args,
        "final_retake_num_inference_steps",
        "retake_num_inference_steps",
        "num_inference_steps",
        default=num_inference_steps,
    )
    retake_start_frames = _resolve_override(args.retake_start_frames, saved_args, "retake_start_frames", default=15)
    seed = _resolve_override(args.seed, saved_args, "seed", default=42)
    cfg_scale = _resolve_override(args.cfg_scale, saved_args, "cfg_scale", default=None)
    audio_cfg_scale = _resolve_override(args.audio_cfg_scale, saved_args, "audio_cfg_scale", default=None)
    a2v_scale = _resolve_override(args.a2v_scale, saved_args, "a2v_scale", default=None)
    enhance_prompt = _resolve_override(args.enhance_prompt, saved_args, "enhance_prompt", default=False)
    # Keep hires transfer on the standard guidance path unless explicitly overridden.
    # The optimization run may have used low-memory guidance only as a training-time
    # concession, and inheriting it here can unintentionally change inference behavior.
    low_memory_guidance = args.low_memory_guidance if args.low_memory_guidance is not None else False
    gradient_checkpointing = _resolve_override(
        args.gradient_checkpointing, saved_args, "gradient_checkpointing", default=False,
    )
    quantization = _resolve_override(
        args.quantization, saved_args, "retake_quantization", "quantization", default="fp8-cast",
    )
    checkpoint_path = _resolve_override(
        args.checkpoint_path, saved_args, "checkpoint_path", default=transfer_optimized.DEFAULT_CHECKPOINT,
    )
    gemma_root = _resolve_override(
        args.gemma_root, saved_args, "gemma_root", default=transfer_optimized.DEFAULT_GEMMA_ROOT,
    )
    qwen_model = _resolve_override(args.qwen_model, saved_args, "qwen_model", default="")
    clip_diag = _resolve_override(args.clip_diag, saved_args, "clip_diag", "clip_similarity_diag", default=True)
    loras = saved_args.get("loras", [])

    # ---- Output dir ----
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else opt_dir / "hires_transfer"
    )

    # ---- Resolution ----
    # Default: height=None, width=None → compute_target_shape reads native dims.
    # If --max-side is set, cap proportionally.
    height: int | None = None
    width: int | None = None

    if args.max_side:
        from ltx_pipelines.utils.media_io import get_videostream_metadata
        _, _, native_w, native_h = get_videostream_metadata(str(src_video))
        height, width = _cap_resolution(native_h, native_w, args.max_side)
        if height is not None:
            log.info(
                "Resolution capped: %dx%d → %dx%d (--max-side %d)",
                native_w, native_h, width, height, args.max_side,
            )
        else:
            log.info(
                "Native resolution %dx%d is within --max-side %d, no cap applied.",
                native_w, native_h, args.max_side,
            )

    log.info("=" * 60)
    log.info("  High-resolution transfer")
    log.info("  Opt dir    : %s", opt_dir)
    log.info("  Mode dir   : %s", mode_dir)
    log.info("  Src video  : %s", src_video)
    log.info("  Output     : %s", output_dir)
    log.info("  Edit prompt: %s", edit_prompt)
    log.info(
        "  Resolution : %s",
        f"{width}x{height}" if height else "native (read from video)",
    )
    log.info(
        "  Inherited   : num_steps=%s retake_steps=%s final_steps=%s retake_start=%s "
        "seed=%s quant=%s enhance=%s low_mem_guidance=%s grad_ckpt=%s",
        num_inference_steps,
        retake_num_inference_steps,
        final_retake_num_inference_steps,
        retake_start_frames,
        seed,
        quantization,
        enhance_prompt,
        low_memory_guidance,
        gradient_checkpointing,
    )
    log.info("=" * 60)

    # ---- Install memory hooks BEFORE calling run() ----
    _install_memory_hooks(
        vae_tile_size=args.vae_tile_size,
        vae_temporal_tile_frames=args.vae_temporal_tile_frames,
    )

    # ---- Build args namespace for transfer_optimized.run() ----
    transfer_args = argparse.Namespace(
        # Inputs
        target_video=str(src_video),
        opt_dir=str(mode_dir),
        mode=args.mode,
        output_dir=str(output_dir),

        # Prompts
        edit_prompt=edit_prompt,
        static_prompt=static_prompt,
        negative_prompt=negative_prompt,

        # Resolution (None = native)
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=None,
        audio_sr=44100,

        # Pipeline
        enhance_prompt=enhance_prompt,
        num_inference_steps=num_inference_steps,
        retake_num_inference_steps=retake_num_inference_steps,
        ti2v_num_inference_steps=num_inference_steps,
        final_retake_num_inference_steps=final_retake_num_inference_steps,
        retake_start_frames=retake_start_frames,
        seed=seed,
        cfg_scale=cfg_scale,
        audio_cfg_scale=audio_cfg_scale,
        a2v_scale=a2v_scale,
        low_memory_guidance=low_memory_guidance,
        quantization=quantization,
        gradient_checkpointing=gradient_checkpointing,
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        loras=loras,

        # Evaluation
        qwen_model=qwen_model,
        qwen_eval_frames=16,
        clip_diag=clip_diag,
        clip_model="openai/clip-vit-base-patch32",
        clip_max_frames=0,
    )

    transfer_optimized.run(transfer_args)


if __name__ == "__main__":
    main()
