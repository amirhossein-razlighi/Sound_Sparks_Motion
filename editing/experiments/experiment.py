#!/usr/bin/env python3
"""
Audio-Edit → Video Experiment
==============================
Research script to probe how audio modifications affect LTX-2's audio-conditioned
video generation (A2VidPipelineTwoStage).

Core idea
---------
LTX-2 supports bidirectional audio-video sync:
  * video changes → audio adapts
  * audio changes → video adapts

This script exploits the second pathway: take a source video, extract its audio
track, apply a battery of perturbations (noise, silence, phase scrambling, pitch
shifts, time-reversal, …), then re-generate the video with each modified audio.
Comparing outputs reveals which audio features (rhythm, spectral content, phase
coherence, …) most strongly drive video content.

Usage
-----
# Run all perturbations on a single video:
    python experiment.py \\
        --src-video /path/to/source.mp4 \\
        --prompt "A person playing guitar on stage" \\
        --output-dir ./results

# Run only specific perturbations:
    python experiment.py \\
        --src-video /path/to/source.mp4 \\
        --prompt "..." \\
        --output-dir ./results \\
        --perturbations identity silence noise_snr10 phase_randomize time_reverse

# List available perturbations:
    python experiment.py --list-perturbations

See audio_perturbations.py for the full registry of perturbation functions.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Add repository root to sys.path so we can import audio_perturbations.py
# regardless of how the script is invoked.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from audio_perturbations import (  # noqa: E402
    get_all_perturbations,
    get_perturbation_by_name,
    list_perturbation_names,
)

# LTX-2 imports
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
import ltx_pipelines.retake as _retake_module
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import (
    _prepare_audio_stream,
    _write_audio,
    decode_audio_from_file,
    encode_video,
    get_videostream_metadata,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default checkpoint paths (matching the cluster setup).
# Override via CLI flags or SLURM environment variables.
# ---------------------------------------------------------------------------
_CKPT_ROOT = "${CKPT_ROOT}"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "${GEMMA_ROOT}/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_valid_frames(n: int) -> int:
    """Return the nearest value ≥ n that satisfies num_frames = 8k + 1."""
    if n <= 1:
        return 1
    k = max(1, (n - 1 + 7) // 8)
    return 8 * k + 1


def write_temp_video_with_audio(
    src_video_path: str,
    target_frames: int,
    target_height: int,
    target_width: int,
    fps: float,
    waveform: torch.Tensor,
    sr: int,
    output_path: str,
) -> None:
    """Write a temp MP4 with exactly target_frames frames (resized) + the given audio.

    The video is decoded from ``src_video_path`` and re-encoded at the requested
    resolution so RetakePipeline sees exactly ``(target_height, target_width,
    target_frames)`` — all values that satisfy the model's constraints.
    The audio waveform (C, T) is normalised to stereo and muxed in using AAC.
    """
    src = av.open(src_video_path)
    dst = av.open(output_path, "w")

    vs_in = src.streams.video[0]
    vs_out = dst.add_stream("libx264", rate=int(round(fps)))
    vs_out.width = target_width
    vs_out.height = target_height
    vs_out.pix_fmt = "yuv420p"
    vs_out.options = {"crf": "18", "preset": "veryfast"}

    # Normalise audio to stereo BEFORE adding the stream.
    # All streams must be declared before the first mux call.
    w = waveform.cpu().float()
    if w.shape[0] == 1:
        w = w.expand(2, -1).contiguous()  # mono → stereo
    elif w.shape[0] > 2:
        w = w[:2].contiguous()
    as_out = _prepare_audio_stream(dst, sr)

    from fractions import Fraction as _Fraction
    time_base = _Fraction(1, int(round(fps)))

    frame_idx = 0
    last_frame = None
    for av_frame in src.decode(vs_in):
        if frame_idx >= target_frames:
            break
        out = av_frame.reformat(width=target_width, height=target_height, format="yuv420p")
        out.pts = frame_idx
        out.time_base = time_base
        for pkt in vs_out.encode(out):
            dst.mux(pkt)
        last_frame = out
        frame_idx += 1

    # Pad with the last frame if the source is shorter than target_frames.
    while last_frame is not None and frame_idx < target_frames:
        pad = av.VideoFrame(width=target_width, height=target_height, format="yuv420p")
        pad.pts = frame_idx
        pad.time_base = time_base
        # Copy planes from last frame
        for i in range(len(last_frame.planes)):
            np.copyto(
                np.frombuffer(pad.planes[i], dtype=np.uint8).reshape(last_frame.planes[i].shape),
                np.frombuffer(last_frame.planes[i], dtype=np.uint8).reshape(last_frame.planes[i].shape),
            )
        for pkt in vs_out.encode(pad):
            dst.mux(pkt)
        frame_idx += 1

    for pkt in vs_out.encode():
        dst.mux(pkt)

    src.close()

    _write_audio(dst, as_out, Audio(waveform=w, sampling_rate=sr))

    dst.close()
    log.info("Temp video written → %s  (%d frames, %dx%d)", output_path, frame_idx, target_width, target_height)


def save_audio_wav(waveform: torch.Tensor, sr: int, path: str) -> None:
    """Save a (C, T) float32 waveform tensor to a WAV file."""
    torchaudio.save(path, waveform.cpu().float(), sr, backend="soundfile")


def align_waveform_length(waveform: torch.Tensor, n_target_samples: int) -> torch.Tensor:
    """Trim or zero-pad waveform to exactly n_target_samples."""
    n = waveform.shape[-1]
    if n >= n_target_samples:
        return waveform[..., :n_target_samples]
    pad = torch.zeros(*waveform.shape[:-1], n_target_samples - n, device=waveform.device, dtype=waveform.dtype)
    return torch.cat([waveform, pad], dim=-1)


def compute_target_shape(
    video_path: str,
    height_override: int | None,
    width_override: int | None,
    num_frames_override: int | None,
    frame_rate_override: float | None,
) -> tuple[int, int, int, float]:
    """Determine (height, width, num_frames, fps) from the source video or overrides.

    RetakePipeline (single-stage) requires H and W to be multiples of 32.
    num_frames must satisfy 8k + 1.
    """
    fps_src, n_frames_src, w_src, h_src = get_videostream_metadata(video_path)

    fps    = frame_rate_override if frame_rate_override is not None else fps_src
    n_fr   = num_frames_override if num_frames_override is not None else n_frames_src
    height = height_override      if height_override is not None      else h_src
    width  = width_override       if width_override is not None       else w_src

    # Snap H/W to multiples of 32 (single-stage requirement).
    height = max(32, (height // 32) * 32)
    width  = max(32, (width  // 32) * 32)

    # Snap num_frames to the nearest 8k+1 that does not exceed the source.
    # We must not exceed source frame count (cannot hallucinate new frames).
    if n_fr > n_frames_src:
        n_fr = n_frames_src
    # Snap downwards so we don't exceed what the source has.
    k = max(0, (n_fr - 1) // 8)
    n_fr = max(1, 8 * k + 1)

    return height, width, n_fr, float(fps)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_experiment(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Determine target shape
    # ------------------------------------------------------------------
    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video,
        args.height,
        args.width,
        args.num_frames,
        args.frame_rate,
    )
    n_audio_samples = int(num_frames / frame_rate * args.audio_sr)

    log.info(
        "Target shape: %dx%d, %d frames @ %.1f fps  (audio SR=%d, ~%.1f s)",
        height, width, num_frames, frame_rate,
        args.audio_sr, num_frames / frame_rate,
    )

    # ------------------------------------------------------------------
    # 2. Extract source audio from the video
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_audio = decode_audio_from_file(
        args.src_video,
        device=device,
        start_time=args.audio_start_time,
        max_duration=num_frames / frame_rate,
    )
    if src_audio is None:
        raise RuntimeError(
            f"No audio stream found in '{args.src_video}'. "
            "Please provide a video file that contains an audio track."
        )

    # Resample to desired sample rate if needed
    if src_audio.sampling_rate != args.audio_sr:
        log.info("Resampling audio from %d Hz → %d Hz", src_audio.sampling_rate, args.audio_sr)
        waveform = torchaudio.functional.resample(
            src_audio.waveform.squeeze(0).float(),
            orig_freq=src_audio.sampling_rate,
            new_freq=args.audio_sr,
        )
    else:
        waveform = src_audio.waveform.squeeze(0).float()  # (C, T)

    waveform = align_waveform_length(waveform, n_audio_samples)
    log.info("Source waveform: shape=%s, SR=%d Hz", tuple(waveform.shape), args.audio_sr)

    # ------------------------------------------------------------------
    # 3. Build the pipeline (loaded once, reused for all perturbations)
    # ------------------------------------------------------------------
    log.info("Loading RetakePipeline …")
    if args.quantization == "fp8-cast":
        quant_policy = QuantizationPolicy.fp8_cast()
    elif args.quantization == "fp8-scaled-mm":
        quant_policy = QuantizationPolicy.fp8_scaled_mm()
    else:
        quant_policy = None
    pipeline = RetakePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=(),
        device=device,
        quantization=quant_policy,
    )

    # Use model-specific defaults (STG scale/blocks, CFG scales) from checkpoint
    _params = detect_params(args.checkpoint_path)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else _params.video_guider_params.cfg_scale
    audio_cfg_scale = args.audio_cfg_scale if args.audio_cfg_scale is not None else _params.audio_guider_params.cfg_scale
    a2v_scale = args.a2v_scale if args.a2v_scale is not None else _params.video_guider_params.modality_scale

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=cfg_scale,
        stg_scale=_params.video_guider_params.stg_scale,
        stg_blocks=_params.video_guider_params.stg_blocks,
        rescale_scale=_params.video_guider_params.rescale_scale,
        modality_scale=a2v_scale,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=audio_cfg_scale,
        stg_scale=_params.audio_guider_params.stg_scale,
        stg_blocks=_params.audio_guider_params.stg_blocks,
        rescale_scale=_params.audio_guider_params.rescale_scale,
    )
    tiling_cfg = TilingConfig.default()
    duration = num_frames / frame_rate

    # ------------------------------------------------------------------
    # Pre-encode the source video once using tiled_encode.
    #
    # RetakePipeline._encode_video_for_retake calls video_encoder(full_tensor)
    # (no tiling), which OOMs at 1024×1536×121 on a single H100.
    # We encode once here with tiling and monkey-patch the retake module so
    # every pipeline call returns the cached latent instead of re-encoding.
    # This also saves time: encoding happens once regardless of how many
    # perturbations we run.
    # ------------------------------------------------------------------
    log.info("Pre-encoding source video with tiled VAE encoder …")
    from ltx_pipelines.utils.media_io import load_video_conditioning

    video_encoder_for_preenc = pipeline.model_ledger.video_encoder()
    pixel_video = load_video_conditioning(
        video_path=args.src_video,
        height=height,
        width=width,
        frame_cap=num_frames,
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.inference_mode():
        cached_video_latent = video_encoder_for_preenc.tiled_encode(pixel_video, tiling_cfg).to(device)
    del video_encoder_for_preenc, pixel_video
    torch.cuda.empty_cache()
    log.info("Source video encoded → latent shape %s", tuple(cached_video_latent.shape))

    # Monkey-patch: return the cached latent, skip re-encoding inside pipeline.
    _original_encode_video_for_retake = _retake_module._encode_video_for_retake

    def _cached_encode_video_for_retake(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    _retake_module._encode_video_for_retake = _cached_encode_video_for_retake

    # ------------------------------------------------------------------
    # 4. Select perturbations
    # ------------------------------------------------------------------
    if args.perturbations:
        perturbations = [get_perturbation_by_name(n) for n in args.perturbations]
    else:
        perturbations = get_all_perturbations()

    log.info("Running %d perturbation(s): %s", len(perturbations), [p.name for p in perturbations])

    # ------------------------------------------------------------------
    # 5. Run each perturbation
    # ------------------------------------------------------------------
    results_log: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="ltx_audio_edit_") as tmp_dir:
        for i, pert in enumerate(perturbations):
            log.info(
                "[%d/%d] Perturbation: %s — %s",
                i + 1, len(perturbations), pert.name, pert.description,
            )

            # Apply perturbation
            torch.manual_seed(args.seed)  # deterministic perturbations (e.g. random_mute)
            modified_waveform = pert.apply(waveform.to(device), args.audio_sr)
            modified_waveform = align_waveform_length(modified_waveform, n_audio_samples)

            # Save modified audio as a standalone WAV for reference.
            perturbed_audio_out = str(output_dir / f"{pert.name}_audio.wav")
            save_audio_wav(modified_waveform, args.audio_sr, perturbed_audio_out)

            # Build a temp video file = original video frames + modified audio.
            # RetakePipeline reads audio directly from the video file, so we must
            # mux the modified audio into a proper MP4 before calling it.
            tmp_video_path = os.path.join(tmp_dir, f"{pert.name}.mp4")
            write_temp_video_with_audio(
                src_video_path=args.src_video,
                target_frames=num_frames,
                target_height=height,
                target_width=width,
                fps=frame_rate,
                waveform=modified_waveform,
                sr=args.audio_sr,
                output_path=tmp_video_path,
            )

            # Run RetakePipeline over the full video duration.
            # The entire source video (encoded as initial_latent) serves as
            # conditioning — the model attends to all frames, not just the first.
            output_video_path = str(output_dir / f"{pert.name}.mp4")
            try:
                video_iter, regen_audio = pipeline(
                    video_path=tmp_video_path,
                    prompt=args.prompt,
                    start_time=args.retake_start_frames / frame_rate,
                    end_time=duration,
                    seed=args.seed,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    video_guider_params=video_guider_params,
                    audio_guider_params=audio_guider_params,
                    regenerate_video=True,
                    regenerate_audio=args.regenerate_audio,
                    enhance_prompt=args.enhance_prompt,
                    tiling_config=tiling_cfg,
                )

                # Mux the modified INPUT audio into the output video so it's
                # immediately obvious what audio drove the generation.
                # Normalise to stereo for the encoder.
                w_out = modified_waveform.cpu().float()
                if w_out.shape[0] == 1:
                    w_out = w_out.expand(2, -1).contiguous()
                elif w_out.shape[0] > 2:
                    w_out = w_out[:2].contiguous()
                output_audio = Audio(waveform=w_out, sampling_rate=args.audio_sr)

                n_chunks = get_video_chunks_number(num_frames, tiling_cfg)
                encode_video(
                    video=video_iter,
                    fps=int(round(frame_rate)),
                    audio=output_audio,
                    output_path=output_video_path,
                    video_chunks_number=n_chunks,
                )

                # Also save the model's regenerated audio separately.
                regen_audio_path = str(output_dir / f"{pert.name}_regen_audio.wav")
                save_audio_wav(regen_audio.waveform, regen_audio.sampling_rate, regen_audio_path)

                log.info("  → saved: %s", output_video_path)
                results_log.append({"perturbation": pert.name, "status": "ok", "output": output_video_path})

            except Exception as exc:  # noqa: BLE001
                log.error("  ✗ FAILED (%s): %s", pert.name, exc, exc_info=True)
                results_log.append({"perturbation": pert.name, "status": "failed", "error": str(exc)})
                torch.cuda.empty_cache()

    # Restore original function.
    _retake_module._encode_video_for_retake = _original_encode_video_for_retake

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Experiment complete. Results saved to: %s", output_dir)
    log.info("%-25s %s", "PERTURBATION", "STATUS")
    log.info("-" * 45)
    for r in results_log:
        status = r["status"]
        log.info("%-25s %s", r["perturbation"], status.upper())
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Utility --
    p.add_argument(
        "--list-perturbations",
        action="store_true",
        help="Print all available perturbation names and exit.",
    )

    # -- Input / output --
    p.add_argument("--src-video", type=str, default=None,
                   help="Path to the source video (must have an audio track).")
    p.add_argument("--output-dir", type=str, default="./editing_results",
                   help="Directory where output videos and audio files are written.")

    # -- Prompt --
    p.add_argument("--prompt", type=str, default="",
                   help="Text prompt describing the video content.")
    p.add_argument("--negative-prompt", type=str, default="",
                   help="Negative text prompt.")
    p.add_argument("--enhance-prompt", action="store_true",
                   help="Enable Gemma prompt enhancement.")

    # -- Perturbations --
    p.add_argument(
        "--perturbations", nargs="+", default=None,
        metavar="NAME",
        help=(
            "Subset of perturbations to run (space-separated names). "
            "Defaults to ALL perturbations. Use --list-perturbations to see names."
        ),
    )

    # -- Video shape overrides (auto-detected from source if omitted) --
    p.add_argument("--height", type=int, default=None,
                   help="Output height in pixels (must be multiple of 32). "
                        "Auto-detected from source video if omitted.")
    p.add_argument("--width", type=int, default=None,
                   help="Output width in pixels (must be multiple of 32). "
                        "Auto-detected from source video if omitted.")
    p.add_argument("--num-frames", type=int, default=None,
                   help="Number of frames (must be 8k+1, capped at source length). "
                        "Auto-detected from source video if omitted.")
    p.add_argument("--frame-rate", type=float, default=None,
                   help="Frame rate. Auto-detected from source video if omitted.")

    # -- Audio --
    p.add_argument("--audio-sr", type=int, default=44100,
                   help="Target audio sample rate in Hz (default: 44100).")
    p.add_argument("--audio-start-time", type=float, default=0.0,
                   help="Start time in seconds for audio extraction from source video.")

    # -- Regeneration control --
    # IMPORTANT: regenerate_audio=False is required for this experiment to work.
    # When False, the modified audio latent is kept clean (not noised) throughout
    # the denoising loop, so the transformer attends to the actual modified audio
    # tokens at every step — this is what drives audio→video conditioning.
    # When True, audio starts from pure noise (same seed as video), making the
    # modified waveform completely invisible to the model (zero effect on output).
    p.add_argument("--regenerate-audio", dest="regenerate_audio", action="store_true",
                   help="Let the diffusion model regenerate audio from scratch (ignores "
                        "the modified input audio waveform; generally wrong for this "
                        "experiment). Default: off — modified audio is used as a clean "
                        "conditioning input that directly drives video generation.")
    p.set_defaults(regenerate_audio=False)

    # -- Model checkpoints --
    p.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--gemma-root", type=str, default=DEFAULT_GEMMA_ROOT)
    p.add_argument("--quantization", type=str, default=None,
                   choices=["fp8-cast", "fp8-scaled-mm"],
                   help="Optional quantization mode.")

    # -- Diffusion / sampling --
    p.add_argument("--num-inference-steps", type=int, default=40,
                   help="Number of Euler denoising steps (default: 40).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retake-start-frames", type=int, default=1,
                   help=(
                       "Number of frames at the start of the video to keep unchanged "
                       "(not regenerated). Converted to seconds via frame_rate. "
                       "Default: 1 (anchor the first frame as context)."
                   ))
    p.add_argument("--cfg-scale", type=float, default=None,
                   help="CFG scale for video guidance (default: auto-detected from checkpoint).")
    p.add_argument("--audio-cfg-scale", type=float, default=None,
                   help="CFG scale for audio guidance (default: auto-detected from checkpoint).")
    p.add_argument("--a2v-scale", type=float, default=None,
                   help="Audio-to-video modality guidance scale (default: auto-detected from checkpoint).")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_perturbations:
        print("Available perturbations:")
        for name in list_perturbation_names():
            p = get_perturbation_by_name(name)
            print(f"  {p.name:<25}  {p.description}")
        return

    if args.src_video is None:
        parser.error("--src-video is required (unless using --list-perturbations).")

    run_experiment(args)


if __name__ == "__main__":
    main()
