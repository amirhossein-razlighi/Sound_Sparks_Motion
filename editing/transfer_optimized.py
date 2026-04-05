#!/usr/bin/env python3
"""Transfer optimized latents from one video to a new target video.

Takes the best_audio_latent_{mode}.pt and/or best_text_delta_{mode}.pt saved
by optimize_multimodal.py and applies them to a completely different video.

Hypothesis: if the optimization found a genuine motion direction in latent
space (e.g., "jumping" or "yawning"), it should transfer across subjects.

Example
-------
    # 1. Optimize on a dog-yawning video:
    python editing/optimize_multimodal.py --src-video dog.mp4 --edit-prompt "A dog yawning" ...

    # 2. Transfer the result to a cat video:
    python editing/transfer_optimized.py \\
        --target-video cat.mp4 \\
        --opt-dir results/Xclip/a_dog_yawning/mode_audio \\
        --mode audio \\
        --edit-prompt "A cat yawning" \\
        --output-dir results/transfer/dog_to_cat_yawn
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_CKPT_ROOT = "/project/def-amahdavi/amirrz/LTX-2/checkpoints"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "/project/def-amahdavi/amirrz/HF/models/gemma-3-12b-it-qat-q4_0-unquantized/"

sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio_latent_opt.core import (
    _parse_loras,
    build_cached_source_latents,
    build_guiders_for_mode,
    compute_target_shape,
)
from audio_latent_opt.models import build_retake_pipeline, resolve_quantization_policy
from audio_latent_opt.multimodal_loop import pre_encode_base_contexts
from audio_latent_opt.runtime import build_retake_kwargs, prepare_retake_input_video

import ltx_pipelines.retake as _retake_module
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from audio_latent_opt.core import align_waveform_length

import torchaudio

log = logging.getLogger(__name__)


def _interpolate_audio_latent(
    latent: torch.Tensor,
    target_t: int,
) -> torch.Tensor:
    """Interpolate audio latent along the time axis to match target_t.

    Args:
        latent: [1, C, T_src, F] audio latent from the source optimization.
        target_t: desired time dimension for the target video.

    Returns:
        [1, C, target_t, F] interpolated latent.
    """
    t_src = latent.shape[2]
    if t_src == target_t:
        return latent
    # Treat (C, T, F) as (C, H, W) for 2-D interpolation: resize T only.
    # Permute to [1, C, F, T], interpolate along last dim, permute back.
    x = latent.permute(0, 1, 3, 2)  # [1, C, F, T]
    x = F.interpolate(x, size=(x.shape[2], target_t), mode="bilinear", align_corners=False)
    return x.permute(0, 1, 3, 2)  # [1, C, target_t, F]


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = Path(args.opt_dir).expanduser().resolve()
    mode = args.mode

    # ---- Load saved optimized tensors ----
    audio_latent_path = opt_dir / f"best_audio_latent_{mode}.pt"
    text_delta_path = opt_dir / f"best_text_delta_{mode}.pt"

    saved_audio_latent = None
    saved_text_delta = None

    if mode in ("audio", "both"):
        if not audio_latent_path.exists():
            raise FileNotFoundError(f"Audio latent not found: {audio_latent_path}")
        saved_audio_latent = torch.load(audio_latent_path, map_location="cpu")
        log.info("Loaded audio latent: %s  shape=%s", audio_latent_path.name, tuple(saved_audio_latent.shape))

    if mode in ("text", "both"):
        if not text_delta_path.exists():
            raise FileNotFoundError(f"Text delta not found: {text_delta_path}")
        saved_text_delta = torch.load(text_delta_path, map_location="cpu")
        log.info("Loaded text delta: %s  shape=%s", text_delta_path.name, tuple(saved_text_delta.shape))

    # ---- Resolve shape from target video ----
    height, width, num_frames, frame_rate = compute_target_shape(
        args.target_video,
        args.height,
        args.width,
        args.num_frames,
        args.frame_rate,
    )
    duration = num_frames / frame_rate
    log.info("Target video: %dx%d, %d frames @ %.1f fps (%.2fs)", width, height, num_frames, frame_rate, duration)

    retake_quant = resolve_quantization_policy(args.quantization)

    # ---- Prepare target video (resize + mux audio) ----
    # Reuse the same helper — it reads args.src_video, so point it at target.
    args.src_video = args.target_video
    retake_input_video = prepare_retake_input_video(
        args=args,
        is_main=True,
        output_dir=output_dir,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
    )

    # ---- Load pipeline ----
    params = detect_params(args.checkpoint_path)
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args,
        params=params,
        use_low_memory_guidance=False,
    )

    log.info("Loading RetakePipeline (%s)...", args.checkpoint_path)
    loras = _parse_loras(args.loras)
    pipeline = build_retake_pipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=loras,
        device=device,
        quant_policy=retake_quant,
        gradient_checkpointing=False,
    )

    # ---- Encode target video/audio latents ----
    cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
        pipeline=pipeline,
        src_video=str(retake_input_video),
        height=height,
        width=width,
        num_frames=num_frames,
        audio_sr=args.audio_sr,
        device=device,
    )
    log.info("Target audio latent shape: %s", tuple(base_audio_latent.shape))

    # ---- Encode text context for target prompt ----
    base_pos_context, base_neg_context = pre_encode_base_contexts(
        pipeline=pipeline,
        pos_prompt=args.edit_prompt,
        neg_prompt=args.negative_prompt,
        device=device,
    )
    log.info("Text encoding shape: %s", tuple(base_pos_context.video_encoding.shape))

    # ---- Adapt saved audio latent to target video's time dimension ----
    if saved_audio_latent is not None:
        target_t = base_audio_latent.shape[2]
        src_t = saved_audio_latent.shape[2]
        if src_t != target_t:
            log.info(
                "Audio latent T mismatch: source=%d target=%d — interpolating...", src_t, target_t
            )
            saved_audio_latent = _interpolate_audio_latent(saved_audio_latent, target_t)
        audio_for_render = saved_audio_latent.to(device=device, dtype=base_audio_latent.dtype)
    else:
        audio_for_render = base_audio_latent

    # ---- Build positive context: apply text delta if present ----
    if saved_text_delta is not None:
        target_seq = base_pos_context.video_encoding.shape[1]
        src_seq = saved_text_delta.shape[1]
        if src_seq != target_seq:
            # Interpolate delta along the sequence dimension if lengths differ
            log.info(
                "Text delta seq mismatch: source=%d target=%d — interpolating...", src_seq, target_seq
            )
            delta = saved_text_delta.float().permute(0, 2, 1)  # [1, D, seq_src]
            delta = F.interpolate(delta, size=target_seq, mode="linear", align_corners=False)
            delta = delta.permute(0, 2, 1)  # [1, seq_target, D]
        else:
            delta = saved_text_delta.float()

        pos_context = EmbeddingsProcessorOutput(
            video_encoding=base_pos_context.video_encoding + delta.to(
                device=device, dtype=base_pos_context.video_encoding.dtype
            ),
            audio_encoding=base_pos_context.audio_encoding,
            attention_mask=base_pos_context.attention_mask,
        )
        log.info("Applied text delta (shape %s) to target context.", tuple(delta.shape))
    else:
        pos_context = base_pos_context

    # ---- Build retake kwargs ----
    retake_kwargs = build_retake_kwargs(
        args=args,
        frame_rate=frame_rate,
        duration=duration,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
    )
    retake_kwargs["num_inference_steps"] = args.num_inference_steps

    # ---- Decode and align target audio for output muxing ----
    src_audio = decode_audio_from_file(str(retake_input_video), pipeline.device, max_duration=duration)
    out_audio = None
    if src_audio is not None:
        wave = src_audio.waveform.squeeze(0).float()
        if src_audio.sampling_rate != waveform_sr:
            wave = torchaudio.functional.resample(wave, orig_freq=src_audio.sampling_rate, new_freq=waveform_sr)
        wave = align_waveform_length(wave, int(duration * waveform_sr))
        if wave.shape[0] == 1:
            wave = wave.expand(2, -1).contiguous()
        elif wave.shape[0] > 2:
            wave = wave[:2].contiguous()
        out_audio = Audio(waveform=wave.cpu(), sampling_rate=waveform_sr)

    # ---- Monkey-patch pipeline internals and run inference ----
    def _cached_video(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    def _injected_audio(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return audio_for_render

    def _patched_encode_prompts(prompts, model_ledger, **kwargs):  # noqa: ARG001
        return [pos_context, base_neg_context]

    orig_video = _retake_module._encode_video_for_retake
    orig_audio = _retake_module._encode_audio_for_retake
    orig_prompts = _retake_module.encode_prompts

    _retake_module._encode_video_for_retake = _cached_video
    _retake_module._encode_audio_for_retake = _injected_audio
    _retake_module.encode_prompts = _patched_encode_prompts
    try:
        log.info("Running inference with transferred latents (mode=%s)...", mode)
        with torch.no_grad():
            video_iter, _ = pipeline(video_path=str(retake_input_video), **retake_kwargs)

        out_path = output_dir / f"transfer_{mode}.mp4"
        encode_video(
            video=video_iter,
            output_path=str(out_path),
            fps=frame_rate,
            audio=out_audio,
            tiling_config=TilingConfig.default(),
            num_frames=num_frames,
        )
        log.info("Saved: %s", out_path)
    finally:
        _retake_module._encode_video_for_retake = orig_video
        _retake_module._encode_audio_for_retake = orig_audio
        _retake_module.encode_prompts = orig_prompts

    # Save a record of what was transferred
    (output_dir / "transfer_info.txt").write_text(
        f"mode: {mode}\n"
        f"opt_dir: {opt_dir}\n"
        f"target_video: {args.target_video}\n"
        f"edit_prompt: {args.edit_prompt}\n"
    )
    log.info("Done.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--target-video", required=True, help="New video to apply the optimized latents to.")
    p.add_argument("--opt-dir", required=True,
                   help="Directory containing best_audio_latent_{mode}.pt / best_text_delta_{mode}.pt "
                        "(typically the mode_audio / mode_text subdirectory from an optimize_multimodal run).")
    p.add_argument("--mode", required=True, choices=["text", "audio", "both"],
                   help="Which saved latents to transfer.")
    p.add_argument("--edit-prompt", required=True,
                   help="Prompt describing the desired motion on the TARGET video.")
    p.add_argument("--output-dir", required=True)

    # Video shape
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--frame-rate", type=float, default=None)
    p.add_argument("--audio-sr", type=int, default=44100)

    # Pipeline
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--enhance-prompt", action="store_true")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--retake-num-inference-steps", type=int, default=None)
    p.add_argument("--final-retake-num-inference-steps", type=int, default=None)
    p.add_argument("--retake-start-frames", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--audio-cfg-scale", type=float, default=None)
    p.add_argument("--a2v-scale", type=float, default=None)
    p.add_argument("--low-memory-guidance", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
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
    args.ti2v_num_inference_steps = args.num_inference_steps
    run(args)


if __name__ == "__main__":
    main()
