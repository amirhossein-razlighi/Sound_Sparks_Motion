#!/usr/bin/env python3
"""Gradient-based optimization of text token and/or audio latent for video motion editing.

Three experiment modes (--opt-mode):
  text   — Optimize a soft delta on the Gemma text embedding (video_encoding).
            Tests whether the motion edit can be achieved purely via text-space
            perturbation, without touching the audio conditioning at all.

  audio  — Optimize the audio VAE latent.
            Same hypothesis as the existing flow-loss pipeline, but using a
            CLIP alignment loss instead of RAFT optical flow.

  both   — Jointly optimize text delta + audio latent.
            Upper-bound experiment: can combining both modalities help?

Loss: CLIP alignment — 1 - cosine_similarity(CLIP(generated_frames), CLIP(edit_prompt))

Run all three and compare CLIP scores to determine whether audio carries
unique information for motion editing beyond what text alone can provide.

Usage
-----
    python editing/optimize_multimodal.py \\
        --src-video /path/to/dog.mp4 \\
        --edit-prompt "A dog jumping energetically" \\
        --output-dir /path/to/output \\
        --opt-mode audio   # text | audio | both

Single H200 80GB: add --quantization fp8-cast --gradient-checkpointing
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch

_CKPT_ROOT = "${CKPT_ROOT}"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "${GEMMA_ROOT}/"

# Add src/ to path so relative imports work when run as a script
sys.path.insert(0, str(Path(__file__).parent / "src"))

from motion_opt.clip_loss import build_clip_model, encode_text_for_clip
from motion_opt.core import (
    _parse_loras,
    build_cached_source_latents,
    build_guiders_for_mode,
    compute_target_shape,
)
from motion_opt.models import build_retake_pipeline, resolve_quantization_policy
from motion_opt.multimodal_loop import (
    gradient_optimize_multimodal,
    pre_encode_base_contexts,
    render_final_video,
)
from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video

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

    # ---- Prepare input video (resize + mux source audio) ----
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

    # ---- Load pipeline (LTX Retake) ----
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

    # ---- Load CLIP model ----
    log.info("Loading CLIP model (%s)...", args.clip_model)
    clip_model, clip_tokenizer = build_clip_model(args.clip_model, device)
    text_embedding = encode_text_for_clip(args.edit_prompt, clip_model, clip_tokenizer, device)
    log.info("CLIP text embedding shape: %s", tuple(text_embedding.shape))

    # ---- Build retake kwargs ----
    retake_kwargs = build_retake_kwargs(
        args=args,
        frame_rate=frame_rate,
        duration=duration,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
    )

    eval_sample_start = 0  # start from frame 0 for CLIP evaluation

    # ---- Run optimization ----
    modes = [m.strip() for m in args.opt_mode.split(",")]
    all_results: dict[str, dict] = {}

    for mode in modes:
        log.info("=" * 60)
        log.info("Starting optimization mode: %s", mode.upper())
        log.info("=" * 60)

        mode_dir = output_dir / f"mode_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)

        best = gradient_optimize_multimodal(
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
            clip_model=clip_model,
            text_embedding=text_embedding,
            eval_sample_start=eval_sample_start,
        )
        all_results[mode] = best

        # Save best parameters
        if best.get("audio_latent") is not None:
            torch.save(best["audio_latent"].cpu(), mode_dir / f"best_audio_latent_{mode}.pt")
        if best.get("delta_v") is not None:
            torch.save(best["delta_v"].cpu(), mode_dir / f"best_text_delta_{mode}.pt")
        torch.save(
            {"mode": mode, "clip_loss": best["clip_loss"], "clip_score": best["clip_score"]},
            mode_dir / f"best_params_{mode}.pt",
        )

        log.info(
            "[%s] Optimization done — best CLIP score: %.4f (loss: %.4f)",
            mode, best["clip_score"], best["clip_loss"],
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
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)
    log.info("%-10s  %-12s  %-12s", "mode", "clip_score", "clip_loss")
    log.info("-" * 40)
    for mode, result in sorted(all_results.items(), key=lambda x: -x[1]["clip_score"]):
        log.info("%-10s  %-12.4f  %-12.4f", mode, result["clip_score"], result["clip_loss"])

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
        help=(
            "Comma-separated list of modes to run: text, audio, both. "
            "E.g. --opt-mode text,audio,both runs all three sequentially."
        ),
    )

    # CLIP
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--clip-max-frames", type=int, default=16,
                   help="Max frames to pass to CLIP per iteration (subsampled uniformly).")

    # Regularization
    p.add_argument("--latent-reg-weight", type=float, default=0.01,
                   help="L2 reg weight for audio latent (mode: audio, both).")
    p.add_argument("--text-reg-weight", type=float, default=0.001,
                   help="L2 reg weight for text delta (mode: text, both).")

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
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    # Needed by build_retake_kwargs / get_guiders
    args.ti2v_num_inference_steps = args.num_inference_steps
    run(args)


if __name__ == "__main__":
    main()
