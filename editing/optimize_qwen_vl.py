#!/usr/bin/env python3
"""Gradient-based optimization using Qwen2.5-VL as the alignment loss.

Identical experiment structure to optimize_multimodal.py (text / audio / both
modes) but replaces X-CLIP with Qwen2.5-VL (Qwen/Qwen2.5-VL-7B-Instruct by
default).

Loss: 1 - P("yes" | video, "Does this video show: {edit_prompt}?")

Gradient flows through Qwen2.5-VL's visual encoder back through the
differentiable video decoder to whichever parameter is being optimised.

Usage
-----
    python editing/optimize_qwen_vl.py \\
        --src-video /path/to/dog.mp4 \\
        --edit-prompt "A dog jumping energetically" \\
        --output-dir /path/to/output \\
        --opt-mode audio

Single H200 80GB: add --quantization fp8-cast --gradient-checkpointing
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch

_CKPT_ROOT = "/project/def-amahdavi/amirrz/LTX-2/checkpoints"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized/"
DEFAULT_QWEN_ROOT = "/project/def-amahdavi/amirrz/HF/models/Qwen2.5-VL-7B-Instruct"

sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio_latent_opt.core import (
    _parse_loras,
    build_cached_source_latents,
    build_guiders_for_mode,
    compute_target_shape,
)
from audio_latent_opt.models import build_retake_pipeline, resolve_quantization_policy
from audio_latent_opt.multimodal_loop_qwen import (
    gradient_optimize_multimodal_qwen,
    pre_encode_base_contexts,
    render_final_video,
)
from audio_latent_opt.qwen_loss import QWEN_IMG_SIZE, build_qwen_inputs, build_qwen_model
from audio_latent_opt.runtime import build_retake_kwargs, prepare_retake_input_video

from ltx_pipelines.utils.constants import detect_params

log = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve shape ----
    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video,
        args.height,
        args.width,
        args.num_frames,
        args.frame_rate,
    )
    duration = num_frames / frame_rate
    log.info("Video shape: %dx%d, %d frames @ %.1f fps (%.2fs)", width, height, num_frames, frame_rate, duration)

    # ---- Quantization ----
    retake_quant = resolve_quantization_policy(
        args.retake_quantization if args.retake_quantization is not None else args.quantization
    )

    # ---- Prepare input video ----
    retake_input_video = prepare_retake_input_video(
        args=args,
        is_main=True,
        output_dir=output_dir,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
    )

    # ---- Build guiders ----
    params = detect_params(args.checkpoint_path)
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args,
        params=params,
        use_low_memory_guidance=args.low_memory_guidance,
    )

    # ---- Load LTX Retake pipeline ----
    log.info("Loading RetakePipeline (checkpoint: %s)...", args.checkpoint_path)
    loras = _parse_loras(args.loras)
    pipeline = build_retake_pipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=loras,
        device=device,
        quant_policy=retake_quant,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # ---- Encode source video/audio ----
    cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
        pipeline=pipeline,
        src_video=str(retake_input_video),
        height=height,
        width=width,
        num_frames=num_frames,
        audio_sr=args.audio_sr,
        device=device,
    )
    base_audio_latent_fp32 = base_audio_latent.float().detach()

    # ---- Pre-encode text contexts (Gemma, one-time) ----
    base_pos_context, base_neg_context = pre_encode_base_contexts(
        pipeline=pipeline,
        pos_prompt=args.edit_prompt,
        neg_prompt=args.negative_prompt,
        device=device,
    )

    # ---- Load Qwen2.5-VL ----
    log.info("Loading Qwen2.5-VL model (%s)...", args.qwen_model)
    qwen_model, qwen_processor = build_qwen_model(
        args.qwen_model,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Ensure num_frames for Qwen is even
    qwen_num_frames = args.qwen_max_frames
    if qwen_num_frames % 2 != 0:
        qwen_num_frames += 1
    log.info("Building Qwen2.5-VL cached inputs (%d frames, %dpx)...", qwen_num_frames, args.qwen_img_size)
    cached_qwen_inputs, yes_token_id, no_token_id = build_qwen_inputs(
        processor=qwen_processor,
        edit_prompt=args.edit_prompt,
        num_frames=qwen_num_frames,
        img_size=args.qwen_img_size,
        device=device,
    )
    log.info("yes_token_id=%d  no_token_id=%d", yes_token_id, no_token_id)

    # ---- Build retake kwargs ----
    retake_kwargs = build_retake_kwargs(
        args=args,
        frame_rate=frame_rate,
        duration=duration,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
    )

    eval_sample_start = 0

    # ---- Run optimization ----
    modes = [m.strip() for m in args.opt_mode.split(",")]
    all_results: dict[str, dict] = {}

    for mode in modes:
        log.info("=" * 60)
        log.info("Starting optimization mode: %s", mode.upper())
        log.info("=" * 60)

        mode_dir = output_dir / f"mode_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)

        best = gradient_optimize_multimodal_qwen(
            mode=mode,
            args=args,
            is_main=True,
            output_dir=mode_dir,
            base_pos_context=base_pos_context,
            base_neg_context=base_neg_context,
            base_audio_latent=base_audio_latent,
            base_audio_latent_fp32=base_audio_latent_fp32,
            cached_video_latent=cached_video_latent,
            retake_input_video=str(retake_input_video),
            pipeline=pipeline,
            retake_kwargs=retake_kwargs,
            qwen_model=qwen_model,
            cached_qwen_inputs=cached_qwen_inputs,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            eval_sample_start=eval_sample_start,
        )
        all_results[mode] = best

        # Save best parameters
        if best.get("audio_latent") is not None:
            torch.save(best["audio_latent"].cpu(), mode_dir / f"best_audio_latent_{mode}.pt")
        if best.get("delta_v") is not None:
            torch.save(best["delta_v"].cpu(), mode_dir / f"best_text_delta_{mode}.pt")
        torch.save(
            {"mode": mode, "qwen_loss": best["qwen_loss"], "qwen_score": best["qwen_score"]},
            mode_dir / f"best_params_{mode}.pt",
        )

        log.info(
            "[%s] Optimization done — best Qwen score: %.4f (loss: %.4f)",
            mode, best["qwen_score"], best["qwen_loss"],
        )

        # ---- Render final videos ----
        if args.save_final_videos:
            log.info("[%s] Rendering final videos...", mode)

            final_retake_kwargs = dict(retake_kwargs)
            if args.final_retake_num_inference_steps is not None:
                final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps

            final_vg, final_ag = build_guiders_for_mode(
                args=args, params=params, use_low_memory_guidance=False
            )
            final_retake_kwargs["video_guider_params"] = final_vg
            final_retake_kwargs["audio_guider_params"] = final_ag

            render_final_video(
                mode=mode,
                best=best,
                pipeline=pipeline,
                src_video=str(retake_input_video),
                cached_video_latent=cached_video_latent,
                base_audio_latent=base_audio_latent,
                base_pos_context=base_pos_context,
                base_neg_context=base_neg_context,
                retake_kwargs=final_retake_kwargs,
                output_dir=mode_dir,
                num_frames=num_frames,
                frame_rate=frame_rate,
                audio_sr=waveform_sr,
                audio_opt_last_steps=args.audio_opt_last_steps,
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Print comparison summary ----
    log.info("")
    log.info("=" * 60)
    log.info("COMPARISON SUMMARY (Qwen2.5-VL loss)")
    log.info("=" * 60)
    log.info("%-10s  %-12s  %-12s", "mode", "qwen_score", "qwen_loss")
    log.info("-" * 40)
    for mode, result in sorted(all_results.items(), key=lambda x: -x[1]["qwen_score"]):
        log.info("%-10s  %-12.4f  %-12.4f", mode, result["qwen_score"], result["qwen_loss"])

    log.info("")
    log.info("Outputs saved to: %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--src-video", required=True)
    p.add_argument("--edit-prompt", required=True)
    p.add_argument("--output-dir", required=True)

    # Experiment mode
    p.add_argument(
        "--opt-mode",
        default="audio",
        help="Comma-separated list of modes: text, audio, both.",
    )

    # Qwen2.5-VL
    p.add_argument("--qwen-model", default=DEFAULT_QWEN_ROOT,
                   help="HF model ID or local path for Qwen2.5-VL.")
    p.add_argument("--qwen-max-frames", type=int, default=8,
                   help="Frames to pass to Qwen2.5-VL per iteration (must be even).")
    p.add_argument("--qwen-img-size", type=int, default=QWEN_IMG_SIZE,
                   help="Spatial size for Qwen input frames (must be divisible by 28).")

    # Regularization
    p.add_argument("--latent-reg-weight", type=float, default=0.01)
    p.add_argument("--text-reg-weight", type=float, default=0.001)

    # Optimization
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--audio-opt-last-steps", type=int, default=6)
    p.add_argument("--resume", action="store_true")

    # Eval
    p.add_argument("--max-eval-frames", type=int, default=33)
    p.add_argument("--frame-stride", type=int, default=1)

    # Video shape
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--frame-rate", type=float, default=None)
    p.add_argument("--audio-sr", type=int, default=44100)

    # Retake pipeline settings
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--enhance-prompt", action="store_true")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--retake-num-inference-steps", type=int, default=None)
    p.add_argument("--final-retake-num-inference-steps", type=int, default=None)
    p.add_argument("--retake-start-frames", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-final-videos", action=argparse.BooleanOptionalAction, default=True)

    # Guidance
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--audio-cfg-scale", type=float, default=None)
    p.add_argument("--a2v-scale", type=float, default=None)
    p.add_argument("--low-memory-guidance", action=argparse.BooleanOptionalAction, default=True)

    # Quantization + checkpointing
    p.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--retake-quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    # Paths
    p.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    p.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    p.add_argument(
        "--lora",
        dest="loras",
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        action="append",
        default=[],
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    # Enforce even qwen_max_frames
    if args.qwen_max_frames % 2 != 0:
        args.qwen_max_frames += 1
        log.warning("--qwen-max-frames rounded up to %d (must be even).", args.qwen_max_frames)

    # Enforce qwen_img_size divisible by 28 (patch_size * merge_size)
    if args.qwen_img_size % 28 != 0:
        args.qwen_img_size = (args.qwen_img_size // 28 + 1) * 28
        log.warning("--qwen-img-size rounded up to %d (must be divisible by 28).", args.qwen_img_size)

    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    args.ti2v_num_inference_steps = args.num_inference_steps

    # Provide a dummy clip_max_frames so any shared code that reads it doesn't break
    args.clip_max_frames = args.qwen_max_frames

    run(args)


if __name__ == "__main__":
    main()
