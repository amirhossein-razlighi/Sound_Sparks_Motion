"""Unified gradient optimization loop for three experiment modes.

Modes
-----
text  : optimize a soft delta on the Gemma text embedding (video_encoding).
audio : optimize the audio latent (same parameter as the existing flow-loss
        pipeline, but now using a CLIP alignment loss).
both  : jointly optimize text delta + audio latent.

Loss
----
Primary: CLIP alignment — 1 - cosine_sim(CLIP(gen_frames), CLIP(edit_prompt))
Optional addons:
  - L2 regularization on the audio latent (keep it close to source audio)
  - L2 regularization on the text delta (keep perturbation small)
"""
from __future__ import annotations

import csv
import gc
import logging
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
import ltx_pipelines.retake as _retake_module
import ltx_pipelines.utils.samplers as _samplers_module

from .clip_loss import compute_clip_video_loss
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
# Main optimization loop
# ---------------------------------------------------------------------------

def gradient_optimize_multimodal(
    *,
    mode: str,  # "text" | "audio" | "both"
    args,
    is_main: bool,
    output_dir: Path,
    # Text optimization inputs
    base_pos_context: EmbeddingsProcessorOutput,
    base_neg_context: EmbeddingsProcessorOutput,
    # Audio optimization inputs
    base_audio_latent: torch.Tensor,
    base_audio_latent_fp32: torch.Tensor,
    # Shared rendering inputs
    cached_video_latent: torch.Tensor,
    retake_input_video: str,
    pipeline,
    retake_kwargs: dict,
    # CLIP loss
    clip_model,
    text_embedding: torch.Tensor,
    eval_sample_start: int,
) -> dict:
    """Run gradient optimization for one of three modes and return best result.

    Returns a dict with keys:
        "clip_loss"      : best total loss value (lower is better)
        "clip_score"     : 1 - clip_loss (higher is better, [0, 1])
        "delta_v"        : best text delta tensor (or None if mode=="audio")
        "audio_latent"   : best audio latent tensor (or None if mode=="text")
        "mode"           : the optimization mode string
    """
    if mode not in ("text", "audio", "both"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'text', 'audio', or 'both'.")

    optimize_text = mode in ("text", "both")
    optimize_audio = mode in ("audio", "both")

    # ---- Build parameters ----
    params: list[torch.nn.Parameter] = []
    delta_v: torch.nn.Parameter | None = None
    audio_latent: torch.nn.Parameter | None = None

    if optimize_text:
        delta_v = torch.nn.Parameter(
            torch.zeros_like(base_pos_context.video_encoding.float())
        )
        params.append(delta_v)
        log.info(
            "[%s] Text delta shape: %s, %.1fK params",
            mode, tuple(delta_v.shape), delta_v.numel() / 1e3,
        )

    if optimize_audio:
        audio_latent = torch.nn.Parameter(base_audio_latent_fp32.clone())
        params.append(audio_latent)
        log.info(
            "[%s] Audio latent shape: %s, %.1fK params",
            mode, tuple(audio_latent.shape), audio_latent.numel() / 1e3,
        )

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ---- Resume from checkpoint ----
    best_latent_path = output_dir / f"best_audio_latent_{mode}.pt"
    best_delta_path = output_dir / f"best_text_delta_{mode}.pt"
    num_iters = args.iterations

    best: dict = {
        "clip_loss": float("inf"),
        "clip_score": float("-inf"),
        "delta_v": delta_v.detach().clone() if delta_v is not None else None,
        "audio_latent": audio_latent.detach().clone() if audio_latent is not None else None,
        "mode": mode,
    }

    if getattr(args, "resume", False):
        if optimize_audio and best_latent_path.exists():
            loaded = torch.load(best_latent_path, map_location="cpu")
            audio_latent.data.copy_(loaded.to(audio_latent.device, dtype=audio_latent.dtype))
            best["audio_latent"] = audio_latent.detach().clone()
            log.info("[%s] Resumed audio latent from %s", mode, best_latent_path)
        if optimize_text and best_delta_path.exists():
            loaded = torch.load(best_delta_path, map_location="cpu")
            delta_v.data.copy_(loaded.to(delta_v.device, dtype=delta_v.dtype))
            best["delta_v"] = delta_v.detach().clone()
            log.info("[%s] Resumed text delta from %s", mode, best_delta_path)
        num_iters = 0  # skip optimization, just do final render

    # ---- CSV log ----
    csv_path = output_dir / f"optimization_log_{mode}.csv"
    csv_file = csv_path.open("w", newline="") if is_main else open("/dev/null", "w", newline="")
    writer = csv.writer(csv_file)
    if is_main:
        writer.writerow(["iter", "clip_loss", "clip_score", "audio_reg", "text_reg", "total_loss", "grad_norm", "is_best"])

    try:
        for it in range(1, num_iters + 1):
            optimizer.zero_grad(set_to_none=True)

            # Build the positive context for this iteration
            if optimize_text and delta_v is not None:
                pos_ctx_iter = EmbeddingsProcessorOutput(
                    video_encoding=base_pos_context.video_encoding + delta_v.to(dtype=base_pos_context.video_encoding.dtype),
                    audio_encoding=base_pos_context.audio_encoding,
                    attention_mask=base_pos_context.attention_mask,
                )
            else:
                pos_ctx_iter = base_pos_context

            # Audio latent to inject
            audio_for_render = (
                audio_latent.to(dtype=base_audio_latent.dtype)
                if audio_latent is not None
                else base_audio_latent
            )

            gen_frames = render_with_injected_latents(
                pipeline=pipeline,
                src_video=retake_input_video,
                injected_audio_latent=audio_for_render,
                cached_video_latent=cached_video_latent,
                retake_kwargs=retake_kwargs,
                max_frames=args.max_eval_frames,
                frame_stride=args.frame_stride,
                resize_to=None,
                audio_opt_last_steps=args.audio_opt_last_steps,
                eval_sample_start=eval_sample_start,
                pos_context=pos_ctx_iter,
                neg_context=base_neg_context,
                inject_text_context=optimize_text,
            )

            if gen_frames.shape[0] < 1:
                raise RuntimeError("Generated video has no frames.")

            # CLIP alignment loss
            clip_loss_t = compute_clip_video_loss(
                frames_chw=gen_frames,
                text_embedding=text_embedding,
                clip_model=clip_model,
                max_frames=args.clip_max_frames,
            )

            # Regularization
            audio_reg_t = torch.tensor(0.0, device=clip_loss_t.device)
            if optimize_audio and audio_latent is not None and args.latent_reg_weight > 0:
                audio_reg_t = args.latent_reg_weight * torch.mean(
                    (audio_latent - base_audio_latent_fp32) ** 2
                )

            text_reg_t = torch.tensor(0.0, device=clip_loss_t.device)
            if optimize_text and delta_v is not None and args.text_reg_weight > 0:
                text_reg_t = args.text_reg_weight * torch.mean(delta_v ** 2)

            total_t = clip_loss_t + audio_reg_t + text_reg_t
            total_t.backward()

            grad_norm = 0.0
            if args.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip).item())
            elif params[0].grad is not None:
                grad_norm = float(sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5)

            optimizer.step()

            clip_loss = float(clip_loss_t.detach().item())
            clip_score = 1.0 - clip_loss
            audio_reg = float(audio_reg_t.detach().item())
            text_reg = float(text_reg_t.detach().item())
            total = float(total_t.detach().item())

            is_best = total < best["clip_loss"]
            if is_best:
                best["clip_loss"] = total
                best["clip_score"] = clip_score
                if delta_v is not None:
                    best["delta_v"] = delta_v.detach().clone()
                if audio_latent is not None:
                    best["audio_latent"] = audio_latent.detach().clone()

            if is_main:
                writer.writerow([it, clip_loss, clip_score, audio_reg, text_reg, total, grad_norm, int(is_best)])
                csv_file.flush()
                log.info(
                    "[%s] iter %3d/%d  clip_loss=%.4f  clip_score=%.4f  total=%.4f  grad_norm=%.3f%s",
                    mode, it, num_iters, clip_loss, clip_score, total, grad_norm,
                    "  ★" if is_best else "",
                )
    finally:
        csv_file.close()

    return best


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
    filename_suffix: str = "",
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
            output_path=str(output_dir / f"best_optimized_video_{mode}{filename_suffix}.mp4"),
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
