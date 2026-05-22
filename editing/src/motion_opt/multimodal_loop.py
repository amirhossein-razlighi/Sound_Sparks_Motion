"""Shared rendering and encoding utilities for the motion editing pipeline.

This module provides the core differentiable rendering infrastructure used by
the Qwen-supervised optimization loop (multimodal_loop_qwen.py):

  - pre_encode_base_contexts  : run Gemma once to cache text embeddings
  - render_with_injected_latents : differentiable Retake forward pass with
                                   gradient flowing through the last N denoising
                                   steps to audio and/or text parameters
  - render_baseline_video     : render the source conditioning without any
                                 learned perturbation (for comparison)
  - render_final_video        : render the best-found conditioning to disk
"""
from __future__ import annotations

import gc
import logging
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
import ltx_pipelines.retake as _retake_module
import ltx_pipelines.utils.samplers as _samplers_module

from .core import flatten_video_chunks

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text context pre-encoding
# ---------------------------------------------------------------------------

def pre_encode_base_contexts(
    pipeline,
    pos_prompt: str,
    neg_prompt: str,
    device: torch.device,
) -> tuple[EmbeddingsProcessorOutput, EmbeddingsProcessorOutput]:
    """Run Gemma once to get baseline text contexts.

    These are stored as plain tensors (no grad). During optimization,
    a learnable delta is added to pos_context.video_encoding.

    Returns:
        (pos_context, neg_context) — both on *device*, no grad.
    """
    from ltx_pipelines.utils.helpers import encode_prompts

    log.info("Pre-encoding text prompts with Gemma (one-time cost)...")
    contexts = encode_prompts([pos_prompt, neg_prompt], pipeline.model_ledger)

    def _to_device(ctx: EmbeddingsProcessorOutput) -> EmbeddingsProcessorOutput:
        return EmbeddingsProcessorOutput(
            video_encoding=ctx.video_encoding.to(device).detach(),
            audio_encoding=ctx.audio_encoding.to(device).detach() if ctx.audio_encoding is not None else None,
            attention_mask=ctx.attention_mask.to(device).detach(),
        )

    pos_ctx = _to_device(contexts[0])
    neg_ctx = _to_device(contexts[1])

    # Free Gemma weights after encoding
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(
        "Text contexts ready. video_encoding shape: %s  dtype: %s",
        tuple(pos_ctx.video_encoding.shape),
        pos_ctx.video_encoding.dtype,
    )
    return pos_ctx, neg_ctx


# ---------------------------------------------------------------------------
# Differentiable render (supports text + audio injection)
# ---------------------------------------------------------------------------

def render_with_injected_latents(
    *,
    pipeline,
    src_video: str,
    injected_audio_latent: torch.Tensor,
    cached_video_latent: torch.Tensor,
    retake_kwargs: dict,
    max_frames: int,
    frame_stride: int,
    resize_to: tuple[int, int] | None,
    audio_opt_last_steps: int,
    eval_sample_start: int,
    # Text injection (None = use real Gemma encoding from retake_kwargs["prompt"])
    pos_context: EmbeddingsProcessorOutput | None = None,
    neg_context: EmbeddingsProcessorOutput | None = None,
    inject_text_context: bool = False,
) -> torch.Tensor:
    """Run a differentiable Retake forward pass with injected audio and/or text.

    Gradient flows back through the last *audio_opt_last_steps* denoising steps
    to whichever parameters are being optimized (audio latent, text delta, both).

    Returns:
        Frames as [N, 3, H, W] float [0, 1], on the same device as the pipeline.
    """
    orig_encode_video = _retake_module._encode_video_for_retake
    orig_encode_audio = _retake_module._encode_audio_for_retake
    orig_decode_video = _retake_module.vae_decode_video
    orig_decode_audio = _retake_module.vae_decode_audio
    orig_euler = _retake_module.euler_denoising_loop
    orig_video_encoder_getter = pipeline.model_ledger.video_encoder
    orig_audio_encoder_getter = pipeline.model_ledger.audio_encoder
    orig_encode_prompts = _retake_module.encode_prompts if inject_text_context else None

    # --- Euler loop: run first (N - audio_opt_last_steps) steps under no_grad ---
    def _late_grad_euler(sigmas, video_state, audio_state, stepper, denoise_fn):
        total_steps = max(int(sigmas.shape[0]) - 1, 0)
        if audio_opt_last_steps <= 0 or audio_opt_last_steps >= total_steps:
            return orig_euler(sigmas, video_state, audio_state, stepper, denoise_fn)

        grad_start = total_steps - audio_opt_last_steps
        for step_idx in range(total_steps):
            if step_idx < grad_start:
                with torch.no_grad():
                    dv, da = denoise_fn(video_state, audio_state, sigmas, step_idx)
                    dv = _samplers_module.post_process_latent(dv, video_state.denoise_mask, video_state.clean_latent)
                    da = _samplers_module.post_process_latent(da, audio_state.denoise_mask, audio_state.clean_latent)
                    next_v = stepper.step(video_state.latent, dv, sigmas, step_idx).detach()
                    next_a = stepper.step(audio_state.latent, da, sigmas, step_idx).detach()
            else:
                dv, da = denoise_fn(video_state, audio_state, sigmas, step_idx)
                dv = _samplers_module.post_process_latent(dv, video_state.denoise_mask, video_state.clean_latent)
                da = _samplers_module.post_process_latent(da, audio_state.denoise_mask, audio_state.clean_latent)
                next_v = stepper.step(video_state.latent, dv, sigmas, step_idx)
                next_a = stepper.step(audio_state.latent, da, sigmas, step_idx)

            video_state = replace(video_state, latent=next_v)
            audio_state = replace(audio_state, latent=next_a)

        return video_state, audio_state

    def _cached_video_encode(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    def _injected_audio_encode(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return injected_audio_latent

    def _float_decode_video(latent, video_decoder, tiling_config=None, generator=None):
        """Decode only the frames we need; return differentiable float tensors."""
        def _to_frames(bcfhw: torch.Tensor) -> torch.Tensor:
            return ((bcfhw[0] + 1.0) / 2.0).clamp(0.0, 1.0).permute(1, 0, 2, 3).contiguous()

        needed = min(latent.shape[2], (max_frames // 8) + 2)
        lat = latent[:, :, :needed]
        if tiling_config is not None:
            for frames in video_decoder.tiled_decode(lat, tiling_config, generator=generator):
                yield _to_frames(frames)
        else:
            yield _to_frames(video_decoder(lat, generator=generator))

    def _detached_audio_decode(latent, audio_decoder, vocoder):
        with torch.no_grad():
            return orig_decode_audio(latent.detach(), audio_decoder, vocoder)

    # Text injection: bypass Gemma and return our (possibly delta-perturbed) contexts
    def _patched_encode_prompts(prompts, model_ledger, **kwargs):  # noqa: ARG001
        return [pos_context, neg_context]

    # Apply patches
    _retake_module._encode_video_for_retake = _cached_video_encode
    _retake_module._encode_audio_for_retake = _injected_audio_encode
    _retake_module.vae_decode_video = _float_decode_video
    _retake_module.vae_decode_audio = _detached_audio_decode
    _retake_module.euler_denoising_loop = _late_grad_euler
    pipeline.model_ledger.video_encoder = lambda: None
    pipeline.model_ledger.audio_encoder = lambda: None
    if inject_text_context:
        _retake_module.encode_prompts = _patched_encode_prompts

    try:
        video_iter, _ = pipeline(video_path=src_video, **retake_kwargs)
        return flatten_video_chunks(
            video_iter=video_iter,
            max_frames=max_frames,
            frame_stride=frame_stride,
            resize_to=resize_to,
            sample_start=eval_sample_start,
        )
    finally:
        _retake_module._encode_video_for_retake = orig_encode_video
        _retake_module._encode_audio_for_retake = orig_encode_audio
        _retake_module.vae_decode_video = orig_decode_video
        _retake_module.vae_decode_audio = orig_decode_audio
        _retake_module.euler_denoising_loop = orig_euler
        pipeline.model_ledger.video_encoder = orig_video_encoder_getter
        pipeline.model_ledger.audio_encoder = orig_audio_encoder_getter
        if inject_text_context:
            _retake_module.encode_prompts = orig_encode_prompts


# ---------------------------------------------------------------------------
# Baseline video rendering (before optimisation)
# ---------------------------------------------------------------------------

def render_baseline_video(
    *,
    pipeline,
    src_video: str,
    cached_video_latent: torch.Tensor,
    base_audio_latent: torch.Tensor,
    base_pos_context: EmbeddingsProcessorOutput,
    base_neg_context: EmbeddingsProcessorOutput,
    retake_kwargs: dict,
    output_path: Path,
    num_frames: int,
    frame_rate: float,
    audio_sr: int,
) -> None:
    """Render a baseline video (unoptimised latents) to *output_path*."""
    import torchaudio
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.types import Audio
    from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
    from .core import align_waveform_length

    duration = num_frames / frame_rate
    src_audio = decode_audio_from_file(src_video, pipeline.device, max_duration=duration)
    out_audio = None
    if src_audio is not None:
        wave = src_audio.waveform.squeeze(0).float()
        if src_audio.sampling_rate != audio_sr:
            wave = torchaudio.functional.resample(wave, orig_freq=src_audio.sampling_rate, new_freq=audio_sr)
        wave = align_waveform_length(wave, int(duration * audio_sr))
        if wave.shape[0] == 1:
            wave = wave.expand(2, -1).contiguous()
        elif wave.shape[0] > 2:
            wave = wave[:2].contiguous()
        out_audio = Audio(waveform=wave.cpu(), sampling_rate=audio_sr)

    orig_encode_video = _retake_module._encode_video_for_retake
    orig_encode_audio = _retake_module._encode_audio_for_retake
    orig_encode_prompts = _retake_module.encode_prompts

    def _cached_video(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    def _base_audio(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return base_audio_latent

    def _base_text(prompts, model_ledger, **kwargs):  # noqa: ARG001
        return [base_pos_context, base_neg_context]

    _retake_module._encode_video_for_retake = _cached_video
    _retake_module._encode_audio_for_retake = _base_audio
    _retake_module.encode_prompts = _base_text
    try:
        with torch.no_grad():
            video_iter, _ = pipeline(video_path=src_video, **retake_kwargs)
        encode_video(
            video=video_iter,
            fps=int(round(frame_rate)),
            audio=out_audio,
            output_path=str(output_path),
            video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
        )
        log.info("Saved baseline video to %s", output_path)
    finally:
        _retake_module._encode_video_for_retake = orig_encode_video
        _retake_module._encode_audio_for_retake = orig_encode_audio
        _retake_module.encode_prompts = orig_encode_prompts


def _audio_latent_to_output_audio(
    *,
    audio_latent: torch.Tensor,
    pipeline,
    duration: float,
) -> "Audio":
    """Decode an audio latent to a stereo Audio object for muxing into MP4."""
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.types import Audio
    from .core import align_waveform_length

    audio_decoder = pipeline.model_ledger.audio_decoder()
    vocoder = pipeline.model_ledger.vocoder()

    with torch.no_grad():
        decoded = vae_decode_audio(
            audio_latent.to(dtype=next(audio_decoder.parameters()).dtype),
            audio_decoder,
            vocoder,
        )

    wave = decoded.waveform.detach().float()
    if wave.ndim == 3 and wave.shape[0] == 1:
        wave = wave.squeeze(0)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)

    wave = align_waveform_length(wave, int(duration * decoded.sampling_rate))
    if wave.shape[0] == 1:
        wave = wave.expand(2, -1).contiguous()
    elif wave.shape[0] > 2:
        wave = wave[:2].contiguous()

    return Audio(waveform=wave.cpu(), sampling_rate=decoded.sampling_rate)


# ---------------------------------------------------------------------------
# Final video rendering with best parameters
# ---------------------------------------------------------------------------

def render_final_video(
    *,
    mode: str,
    best: dict,
    pipeline,
    src_video: str,
    cached_video_latent: torch.Tensor,
    base_audio_latent: torch.Tensor,
    base_pos_context: EmbeddingsProcessorOutput,
    base_neg_context: EmbeddingsProcessorOutput,
    retake_kwargs: dict,
    output_dir: Path,
    num_frames: int,
    frame_rate: float,
    audio_sr: int,
    audio_opt_last_steps: int,
    skip_baseline: bool = False,
) -> None:
    """Render the optimized video and optionally a baseline to disk."""
    import torchaudio
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.types import Audio
    from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
    from .core import align_waveform_length

    optimize_text = mode in ("text", "both")
    optimize_audio = mode in ("audio", "both")

    best_audio_latent = best.get("audio_latent")
    best_delta_v = best.get("delta_v")

    # Build optimized audio latent and text context
    audio_for_render = (
        best_audio_latent.to(dtype=base_audio_latent.dtype)
        if best_audio_latent is not None
        else base_audio_latent
    )
    if optimize_text and best_delta_v is not None:
        pos_ctx_best = EmbeddingsProcessorOutput(
            video_encoding=base_pos_context.video_encoding + best_delta_v.to(dtype=base_pos_context.video_encoding.dtype),
            audio_encoding=base_pos_context.audio_encoding,
            attention_mask=base_pos_context.attention_mask,
        )
    else:
        pos_ctx_best = base_pos_context

    duration = num_frames / frame_rate
    src_audio = decode_audio_from_file(src_video, pipeline.device, max_duration=duration)
    source_out_audio = None
    if src_audio is not None:
        wave = src_audio.waveform.squeeze(0).float()
        if src_audio.sampling_rate != audio_sr:
            wave = torchaudio.functional.resample(wave, orig_freq=src_audio.sampling_rate, new_freq=audio_sr)
        wave = align_waveform_length(wave, int(duration * audio_sr))
        if wave.shape[0] == 1:
            wave = wave.expand(2, -1).contiguous()
        elif wave.shape[0] > 2:
            wave = wave[:2].contiguous()
        source_out_audio = Audio(waveform=wave.cpu(), sampling_rate=audio_sr)

    optimized_out_audio = source_out_audio
    if optimize_audio and best_audio_latent is not None:
        try:
            optimized_out_audio = _audio_latent_to_output_audio(
                audio_latent=audio_for_render,
                pipeline=pipeline,
                duration=duration,
            )
            log.info("[%s] Using decoded optimized audio latent in optimized MP4.", mode)
        except Exception:
            log.warning(
                "[%s] Failed to decode optimized audio latent for MP4; falling back to source audio.",
                mode,
                exc_info=True,
            )

    orig_encode_video = _retake_module._encode_video_for_retake
    orig_encode_audio = _retake_module._encode_audio_for_retake
    orig_encode_prompts = _retake_module.encode_prompts if optimize_text else None

    def _cached_video(video_encoder, video_path, output_shape, dtype, device):  # noqa: ARG001
        return cached_video_latent

    def _injected_audio(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return audio_for_render

    def _patched_encode_prompts(prompts, model_ledger, **kwargs):  # noqa: ARG001
        return [pos_ctx_best, base_neg_context]

    _retake_module._encode_video_for_retake = _cached_video
    _retake_module._encode_audio_for_retake = _injected_audio
    if optimize_text:
        _retake_module.encode_prompts = _patched_encode_prompts
    try:
        with torch.no_grad():
            video_iter, _ = pipeline(video_path=src_video, **retake_kwargs)
        encode_video(
            video=video_iter,
            fps=int(round(frame_rate)),
            audio=optimized_out_audio,
            output_path=str(output_dir / f"best_optimized_video_{mode}.mp4"),
            video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
        )
        log.info("[%s] Saved optimized video.", mode)
    finally:
        _retake_module._encode_video_for_retake = orig_encode_video
        _retake_module._encode_audio_for_retake = orig_encode_audio
        if optimize_text:
            _retake_module.encode_prompts = orig_encode_prompts

    if skip_baseline:
        return

    baseline_out_audio = source_out_audio
    if optimize_audio:
        try:
            baseline_out_audio = _audio_latent_to_output_audio(
                audio_latent=base_audio_latent,
                pipeline=pipeline,
                duration=duration,
            )
            log.info("[%s] Using decoded baseline audio latent in baseline MP4.", mode)
        except Exception:
            log.warning(
                "[%s] Failed to decode baseline audio latent for MP4; falling back to source audio.",
                mode,
                exc_info=True,
            )

    # Baseline: base audio + base text context
    def _base_audio(audio_encoder, waveform, waveform_sr, output_shape, dtype):  # noqa: ARG001
        return base_audio_latent

    def _base_text(prompts, model_ledger, **kwargs):  # noqa: ARG001
        return [base_pos_context, base_neg_context]

    _retake_module._encode_video_for_retake = _cached_video
    _retake_module._encode_audio_for_retake = _base_audio
    if optimize_text:
        _retake_module.encode_prompts = _base_text
    try:
        with torch.no_grad():
            video_iter, _ = pipeline(video_path=src_video, **retake_kwargs)
        encode_video(
            video=video_iter,
            fps=int(round(frame_rate)),
            audio=baseline_out_audio,
            output_path=str(output_dir / f"baseline_video_{mode}.mp4"),
            video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
        )
        log.info("[%s] Saved baseline video.", mode)
    finally:
        _retake_module._encode_video_for_retake = orig_encode_video
        _retake_module._encode_audio_for_retake = orig_encode_audio
        if optimize_text:
            _retake_module.encode_prompts = orig_encode_prompts
