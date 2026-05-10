#!/usr/bin/env python3
"""
Audio-Diff Video Editing
========================
A research script for **video-only editing** via a 3-step diffusion pipeline:

  Step 1 – Audio-guided generation (edit imagination):
      TI2VidOneStagePipeline(first_frame, edit_prompt) → gen_video, gen_audio
      The model imagines what the edited scene looks, *and sounds*, like.

  Step 2 – Audio diff (edit signal extraction):
      diff_audio = compute_diff(gen_audio, src_audio, mode)
      Isolates the *acoustic delta* caused by the edit (what changed in sound).

  Step 3 – Audio-conditioned retake (video realization):
      RetakePipeline(src_video, diff_audio, regenerate_audio=False)
      The full source video is used as spatial / temporal conditioning;
      the diff audio drives the video transformer to produce an edited video.

Diff modes  (--diff-mode):
  direct       Use gen_audio directly (no subtraction) — strongest edit signal.
  time         time-domain: diff = gen - src  (waveform subtraction)
  freq_mag     Spectral magnitude diff with gen phase:
                  diff = iSTFT( (|STFT(gen)| - |STFT(src)|) * exp(i·∠STFT(gen)) )
  freq_complex Complex STFT diff → iSTFT (captures both magnitude and phase shifts).

Usage
-----
    python edit_with_audio_diff.py \\
        --src-video /path/to/source.mp4 \\
        --edit-prompt "The man drops his guitar and leaves the scene." \\
        --output-dir ./edit_results

    # Choose diff mode (default: freq_mag):
    python edit_with_audio_diff.py \\
        --src-video source.mp4 \\
        --edit-prompt "..." \\
        --diff-mode time \\
        --output-dir ./edit_results_time
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
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure editing/ is on sys.path (for running without install)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# LTX-2 imports
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
import ltx_pipelines.retake as _retake_module
import ltx_pipelines.ti2vid_one_stage as _ti2vid_module
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import (
    _prepare_audio_stream,
    _write_audio,
    decode_audio_from_file,
    encode_video,
    get_videostream_metadata,
    load_video_conditioning,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default checkpoint paths (matching the cluster setup)
# ---------------------------------------------------------------------------
_CKPT_ROOT = "${CKPT_ROOT}"
DEFAULT_CHECKPOINT = f"{_CKPT_ROOT}/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_ROOT = "${GEMMA_ROOT}/"


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from experiment.py to keep files self-contained)
# ---------------------------------------------------------------------------

def nearest_valid_frames(n: int) -> int:
    """Nearest value ≥ n satisfying num_frames = 8k + 1."""
    k = max(0, (n - 1 + 7) // 8)
    return 8 * k + 1


def compute_target_shape(
    video_path: str,
    height_override: int | None,
    width_override: int | None,
    num_frames_override: int | None,
    frame_rate_override: float | None,
) -> tuple[int, int, int, float]:
    """Determine (height, width, num_frames, fps) from source video or overrides.

    Single-stage pipeline requires H/W multiples of 32. num_frames = 8k+1, snapped
    *down* to avoid exceeding the source length.
    """
    fps_src, n_frames_src, w_src, h_src = get_videostream_metadata(video_path)

    fps    = frame_rate_override if frame_rate_override is not None else fps_src
    n_fr   = num_frames_override if num_frames_override is not None else n_frames_src
    height = height_override      if height_override is not None      else h_src
    width  = width_override       if width_override is not None       else w_src

    height = max(32, (height // 32) * 32)
    width  = max(32, (width  // 32) * 32)

    # Snap num_frames downward to largest 8k+1 ≤ n_fr
    if (n_fr - 1) % 8 != 0:
        n_fr = ((n_fr - 1) // 8) * 8 + 1
    n_fr = max(9, n_fr)

    return height, width, n_fr, fps


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
    """Write a temp MP4 whose video frames come from *src_video_path* and whose
    audio is the provided waveform (stereo, float32, shape (C, T)).

    Audio stream must be declared *before* any mux call (PyAV constraint).
    """
    src = av.open(src_video_path)
    dst = av.open(output_path, mode="w")

    vs_in  = next(s for s in src.streams if s.type == "video")
    vs_out = dst.add_stream("libx264", rate=int(round(fps)))
    vs_out.width  = target_width
    vs_out.height = target_height
    vs_out.pix_fmt = "yuv420p"
    vs_out.options = {"crf": "18", "preset": "veryfast"}

    w = waveform.cpu().float()
    if w.shape[0] == 1:
        w = w.expand(2, -1).contiguous()
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

    while last_frame is not None and frame_idx < target_frames:
        pad = av.VideoFrame(width=target_width, height=target_height, format="yuv420p")
        pad.pts = frame_idx
        pad.time_base = time_base
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
    torchaudio.save(path, waveform.cpu().float(), sr, backend="soundfile")


def align_waveform_length(waveform: torch.Tensor, n_target: int) -> torch.Tensor:
    n = waveform.shape[-1]
    if n >= n_target:
        return waveform[..., :n_target]
    pad = torch.zeros(*waveform.shape[:-1], n_target - n, device=waveform.device, dtype=waveform.dtype)
    return torch.cat([waveform, pad], dim=-1)


def extract_first_frame_png(video_path: str, out_path: str) -> None:
    """Decode the first video frame and save it as a PNG image."""
    src = av.open(video_path)
    vs = next(s for s in src.streams if s.type == "video")
    for frame in src.decode(vs):
        img = frame.to_image()  # PIL Image
        img.save(out_path, format="PNG")
        break
    src.close()


# ---------------------------------------------------------------------------
# Audio diff computation
# ---------------------------------------------------------------------------

def compute_audio_diff(
    gen_waveform: torch.Tensor,
    src_waveform: torch.Tensor,
    mode: str,
    sr: int,
) -> torch.Tensor:
    """Return the 'audio edit signal' from (gen_waveform, src_waveform).

    Parameters
    ----------
    gen_waveform, src_waveform : (C, T) float32 tensors, same length.
    mode : one of  {'direct', 'time', 'freq_mag', 'freq_complex'}
    sr   : sample rate (used for STFT window sizing).

    Returns
    -------
    diff : (C, T) float32 tensor, same shape as inputs.
    """
    # Ensure same device / dtype
    gen = gen_waveform.float()
    src = src_waveform.float().to(gen.device)
    C, T = gen.shape

    if mode == "direct":
        # Use generated audio as-is — strongest signal, no subtraction.
        return gen

    if mode == "time":
        # Simple time-domain residual: what was acoustically added by the edit.
        return gen - src

    # ---- STFT-based modes ------------------------------------------------
    n_fft      = 1024
    hop_length = 256
    win_length = 1024
    window = torch.hann_window(win_length, device=gen.device)

    def _stft(x: torch.Tensor) -> torch.Tensor:
        # x: (C, T) → list of (F, frames) complex tensors → stack (C, F, frames)
        return torch.stack([
            torch.stft(x[c], n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window,
                       return_complex=True)
            for c in range(C)
        ])  # (C, F, frames)

    def _istft(X: torch.Tensor, length: int) -> torch.Tensor:
        return torch.stack([
            torch.istft(X[c], n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=window,
                        length=length)
            for c in range(C)
        ])  # (C, T)

    S_gen = _stft(gen)
    S_src = _stft(src)

    if mode == "freq_mag":
        # Magnitude diff; keep the generated audio's phase.
        # Represents "spectral delta" of the edit with coherent phase.
        mag_diff = S_gen.abs() - S_src.abs()
        # ReLU keeps only energy that was *added* by the edit.  Remove to keep
        # both additions and removals (set to plain mag_diff for that).
        diff_complex = mag_diff * torch.exp(1j * S_gen.angle().to(torch.float32)).to(S_gen.dtype)
        return _istft(diff_complex, T)

    if mode == "freq_complex":
        # Full complex STFT diff → captures both magnitude and phase shift.
        diff_complex = S_gen - S_src
        return _istft(diff_complex, T)

    raise ValueError(f"Unknown diff mode: {mode!r}. Choose from: direct, time, freq_mag, freq_complex")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_edit(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    # Parse --lora entries (list-of-lists from append+nargs) into LoraPathStrengthAndSDOps
    lora_list: list[LoraPathStrengthAndSDOps] = []
    for entry in args.loras:   # each entry is [path] or [path, strength]
        lora_path = str(Path(entry[0]).expanduser().resolve())
        strength = float(entry[1]) if len(entry) > 1 else 1.0
        lora_list.append(LoraPathStrengthAndSDOps(lora_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    args.loras = lora_list
    if args.loras:
        log.info("LoRAs: %s", [(l.path, l.strength) for l in args.loras])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use model-specific defaults (STG scale/blocks, CFG scales) detected from checkpoint
    _params = detect_params(args.checkpoint_path)
    if args.cfg_scale is None:
        args.cfg_scale = _params.video_guider_params.cfg_scale
    if args.audio_cfg_scale is None:
        args.audio_cfg_scale = _params.audio_guider_params.cfg_scale
    if args.a2v_scale is None:
        args.a2v_scale = _params.video_guider_params.modality_scale

    # ------------------------------------------------------------------ #
    # 1. Determine output shape from source video                         #
    # ------------------------------------------------------------------ #
    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video,
        args.height, args.width, args.num_frames, args.frame_rate,
    )
    duration      = num_frames / frame_rate
    n_audio_samples = int(duration * args.audio_sr)

    log.info("Target shape: %dx%d, %d frames @ %.1f fps  (~%.1f s)", height, width, num_frames, frame_rate, duration)
    log.info("Diff modes: %s", args.diff_modes)

    # ------------------------------------------------------------------ #
    # 2. Load source audio                                                #
    # ------------------------------------------------------------------ #
    src_audio_raw = decode_audio_from_file(args.src_video, device, max_duration=duration)
    if src_audio_raw is None:
        raise RuntimeError(f"No audio stream found in '{args.src_video}'.")

    if src_audio_raw.sampling_rate != args.audio_sr:
        log.info("Resampling source audio %d → %d Hz", src_audio_raw.sampling_rate, args.audio_sr)
        src_waveform = torchaudio.functional.resample(
            src_audio_raw.waveform.squeeze(0).float(),
            orig_freq=src_audio_raw.sampling_rate,
            new_freq=args.audio_sr,
        )
    else:
        src_waveform = src_audio_raw.waveform.squeeze(0).float()

    src_waveform = align_waveform_length(src_waveform.to(device), n_audio_samples)
    log.info("Source audio: shape=%s, SR=%d Hz", tuple(src_waveform.shape), args.audio_sr)

    if args.quantization == "fp8-cast":
        quant_policy = QuantizationPolicy.fp8_cast()
    elif args.quantization == "fp8-scaled-mm":
        quant_policy = QuantizationPolicy.fp8_scaled_mm()
    else:
        quant_policy = None

    tiling_cfg = TilingConfig.default()

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.cfg_scale,
        stg_scale=_params.video_guider_params.stg_scale,
        stg_blocks=_params.video_guider_params.stg_blocks,
        rescale_scale=_params.video_guider_params.rescale_scale,
        modality_scale=args.a2v_scale,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_scale,
        stg_scale=_params.audio_guider_params.stg_scale,
        stg_blocks=_params.audio_guider_params.stg_blocks,
        rescale_scale=_params.audio_guider_params.rescale_scale,
    )

    with tempfile.TemporaryDirectory(prefix="ltx_edit_diff_") as tmp_dir:

        # ------------------------------------------------------------------ #
        # Step 1 (once): Generate video + audio from first frame + prompt    #
        # ------------------------------------------------------------------ #
        log.info("=== Step 1: TI2VidOneStagePipeline (edit imagination) ===")

        first_frame_png = os.path.join(tmp_dir, "first_frame.png")
        extract_first_frame_png(args.src_video, first_frame_png)
        log.info("First frame saved → %s", first_frame_png)

        images: list[ImageConditioningInput] = [
            ImageConditioningInput(path=first_frame_png, frame_idx=0, strength=1.0)
        ]

        gen_pipeline = TI2VidOneStagePipeline(
            checkpoint_path=args.checkpoint_path,
            gemma_root=args.gemma_root,
            loras=tuple(args.loras),
            device=device,
            quantization=quant_policy,
        )

        # Monkey-patch vae_decode_video so VAE decode uses tiled decoding.
        # Without this the bare video_decoder(full_latent) OOMs at 1024×1536×121.
        _orig_vae_decode = _ti2vid_module.vae_decode_video
        _tiling_cfg_for_patch = tiling_cfg  # capture in closure

        def _tiled_vae_decode_video(latent, decoder, tiling_config=None, generator=None):
            return _orig_vae_decode(latent, decoder, _tiling_cfg_for_patch, generator)

        _ti2vid_module.vae_decode_video = _tiled_vae_decode_video
        try:
            gen_video_iter, gen_audio = gen_pipeline(
                prompt=args.edit_prompt,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=args.num_inference_steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=images,
                enhance_prompt=args.enhance_prompt,
            )
        finally:
            _ti2vid_module.vae_decode_video = _orig_vae_decode

        n_chunks_gen = get_video_chunks_number(num_frames, tiling_cfg)
        encode_video(
            video=gen_video_iter,
            fps=int(round(frame_rate)),
            audio=gen_audio,
            output_path=str(output_dir / "step1_generated.mp4"),
            video_chunks_number=n_chunks_gen,
        )
        log.info("Step 1 output saved → %s", output_dir / "step1_generated.mp4")

        gen_waveform = gen_audio.waveform.squeeze(0).float().to(device)
        if gen_audio.sampling_rate != args.audio_sr:
            log.info("Resampling gen audio %d → %d Hz", gen_audio.sampling_rate, args.audio_sr)
            gen_waveform = torchaudio.functional.resample(
                gen_waveform, orig_freq=gen_audio.sampling_rate, new_freq=args.audio_sr
            )

        gen_waveform = align_waveform_length(gen_waveform, n_audio_samples)
        save_audio_wav(gen_waveform, args.audio_sr, str(output_dir / "step1_gen_audio.wav"))

        del gen_pipeline
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------ #
        # Step 3 setup (once): build RetakePipeline, pre-encode video        #
        # ------------------------------------------------------------------ #
        log.info("=== Step 3 setup: building RetakePipeline ===")

        retake_pipeline = RetakePipeline(
            checkpoint_path=args.checkpoint_path,
            gemma_root=args.gemma_root,
            loras=tuple(args.loras),
            device=device,
            quantization=quant_policy,
        )

        log.info("Pre-encoding source video with tiled VAE …")
        video_encoder_for_preenc = retake_pipeline.model_ledger.video_encoder()
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

        _original_encode = _retake_module._encode_video_for_retake

        def _cached_encode(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
            return cached_video_latent

        _retake_module._encode_video_for_retake = _cached_encode

        src_audio_stereo = src_waveform.cpu().float()
        if src_audio_stereo.shape[0] == 1:
            src_audio_stereo = src_audio_stereo.expand(2, -1).contiguous()
        elif src_audio_stereo.shape[0] > 2:
            src_audio_stereo = src_audio_stereo[:2].contiguous()
        output_audio = Audio(waveform=src_audio_stereo, sampling_rate=args.audio_sr)

        n_chunks = get_video_chunks_number(num_frames, tiling_cfg)
        retake_kwargs = dict(
            prompt=args.edit_prompt,
            start_time=args.retake_start_frames / frame_rate,
            end_time=duration,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            regenerate_video=True,
            regenerate_audio=False,
            enhance_prompt=args.enhance_prompt,
            tiling_config=tiling_cfg,
        )

        try:
            # -------------------------------------------------------------- #
            # Control (once): prompt + original source audio                 #
            # -------------------------------------------------------------- #
            log.info("=== Control: RetakePipeline (prompt + source audio) ===")
            tmp_video_ctrl = os.path.join(tmp_dir, "src_with_src_audio.mp4")
            write_temp_video_with_audio(
                src_video_path=args.src_video,
                target_frames=num_frames,
                target_height=height,
                target_width=width,
                fps=frame_rate,
                waveform=src_waveform,
                sr=args.audio_sr,
                output_path=tmp_video_ctrl,
            )
            video_iter_ctrl, _ = retake_pipeline(video_path=tmp_video_ctrl, **retake_kwargs)
            encode_video(
                video=video_iter_ctrl,
                fps=int(round(frame_rate)),
                audio=output_audio,
                output_path=str(output_dir / "control_prompt_only.mp4"),
                video_chunks_number=n_chunks,
            )
            log.info("Control video saved → %s", output_dir / "control_prompt_only.mp4")

            # -------------------------------------------------------------- #
            # Per-mode loop: Step 2 (diff) + Step 3 (retake)                 #
            # -------------------------------------------------------------- #
            for mode in args.diff_modes:
                log.info("=== Mode: %s — Step 2: computing audio diff ===", mode)
                mode_dir = output_dir / mode
                mode_dir.mkdir(exist_ok=True)

                diff_waveform = compute_audio_diff(gen_waveform, src_waveform, mode, args.audio_sr)
                diff_waveform = align_waveform_length(diff_waveform, n_audio_samples)
                if args.diff_gain != 1.0:
                    diff_waveform = diff_waveform * args.diff_gain
                save_audio_wav(diff_waveform, args.audio_sr, str(mode_dir / "step2_diff_audio.wav"))
                log.info(
                    "Diff audio (%s): shape=%s, range=[%.3f, %.3f]",
                    mode, tuple(diff_waveform.shape),
                    diff_waveform.min().item(), diff_waveform.max().item(),
                )

                log.info("=== Mode: %s — Step 3: RetakePipeline (prompt + diff audio) ===", mode)
                tmp_video = os.path.join(tmp_dir, f"src_with_diff_audio_{mode}.mp4")
                write_temp_video_with_audio(
                    src_video_path=args.src_video,
                    target_frames=num_frames,
                    target_height=height,
                    target_width=width,
                    fps=frame_rate,
                    waveform=diff_waveform,
                    sr=args.audio_sr,
                    output_path=tmp_video,
                )
                video_iter, _ = retake_pipeline(video_path=tmp_video, **retake_kwargs)
                encode_video(
                    video=video_iter,
                    fps=int(round(frame_rate)),
                    audio=output_audio,
                    output_path=str(mode_dir / "edited_video.mp4"),
                    video_chunks_number=n_chunks,
                )
                log.info("Edited video (%s) saved → %s", mode, mode_dir / "edited_video.mp4")

        except torch.OutOfMemoryError as exc:
            log.error("CUDA OOM: %s", exc)
            torch.cuda.empty_cache()
            raise
        finally:
            _retake_module._encode_video_for_retake = _original_encode

    log.info("Done. Outputs in: %s", output_dir)
    log.info("  step1_generated.mp4         — edit imagination (shared across all modes)")
    log.info("  step1_gen_audio.wav         — audio generated by the edit prompt")
    log.info("  control_prompt_only.mp4     — prompt + src audio (baseline, shared)")
    for mode in args.diff_modes:
        log.info("  %s/step2_diff_audio.wav", mode)
        log.info("  %s/edited_video.mp4", mode)
    log.info("")
    log.info("  Compare each mode/edited_video.mp4 vs control_prompt_only.mp4:")
    log.info("    identical → edit driven by prompt, not audio diff")
    log.info("    different → audio diff is contributing to the edit")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- I/O --
    p.add_argument("--src-video", required=True,
                   help="Path to the source video (must have an audio track).")
    p.add_argument("--edit-prompt", required=True,
                   help='Text prompt describing the desired edit, e.g. "The man drops his guitar."')
    p.add_argument("--output-dir", default="./edit_results",
                   help="Directory for all output files (default: ./edit_results).")

    # -- Diff modes --
    p.add_argument(
        "--diff-modes",
        nargs="+",
        choices=["direct", "time", "freq_mag", "freq_complex"],
        default=["direct", "time", "freq_mag", "freq_complex"],
        help=(
            "Which diff modes to run (all run in a single job; step 1 and control are "
            "shared). Default: all four modes. "
            "  direct      — use gen_audio as-is (strongest signal). "
            "  time        — waveform subtraction: gen - src. "
            "  freq_mag    — spectral magnitude diff, gen phase. "
            "  freq_complex— complex STFT diff (magnitude + phase shift). "
        ),
    )
    p.add_argument(
        "--diff-gain",
        type=float,
        default=1.0,
        help=(
            "Scalar multiplier applied to the diff waveform before the retake. "
            "Diff signals (especially 'time' and 'freq_mag') are often far smaller in "
            "amplitude than normal audio, so the audio encoder sees near-silence. "
            "Try values in 5–20 to push the diff into the model's normal operating "
            "range and make edits more pronounced. Default: 1.0 (no scaling)."
        ),
    )

    # -- Prompt --
    p.add_argument("--negative-prompt", default="",
                   help="Negative text prompt for both generation stages.")
    p.add_argument("--enhance-prompt", action="store_true",
                   help="Enable Gemma prompt enhancement (Step 1 only).")

    # -- Video shape overrides --
    p.add_argument("--height", type=int, default=None,
                   help="Output height (multiple of 32). Auto-detected if omitted.")
    p.add_argument("--width", type=int, default=None,
                   help="Output width (multiple of 32). Auto-detected if omitted.")
    p.add_argument("--num-frames", type=int, default=None,
                   help="Number of frames (8k+1). Auto-detected if omitted.")
    p.add_argument("--frame-rate", type=float, default=None,
                   help="Frame rate. Auto-detected if omitted.")

    # -- Audio --
    p.add_argument("--audio-sr", type=int, default=44100,
                   help="Target audio sample rate (default: 44100 Hz).")

    # -- Model --
    p.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    p.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    p.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument(
        "--lora",
        dest="loras",
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        action="append",
        default=[],
        help=(
            "LoRA weight file and optional strength (default 1.0). "
            "Can be specified multiple times. "
            "Example: --lora path/to/lora.safetensors 0.8"
        ),
    )

    # -- Diffusion --
    p.add_argument("--num-inference-steps", type=int, default=40,
                   help="Denoising steps for both stages (default: 40).")
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
                   help="Audio-to-video modality scale (default: auto-detected from checkpoint).")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_edit(args)


if __name__ == "__main__":
    main()
