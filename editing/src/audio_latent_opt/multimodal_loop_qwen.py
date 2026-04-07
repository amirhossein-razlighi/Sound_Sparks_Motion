"""Gradient optimization loop — Qwen2.5-VL variant.

Same three modes (text / audio / both) as multimodal_loop.py but uses
compute_qwen_video_loss instead of compute_clip_video_loss.

Rendering helpers (render_with_injected_latents, pre_encode_base_contexts,
render_final_video) are reused directly from multimodal_loop.
"""
from __future__ import annotations

import csv
import gc
import logging
import math
from pathlib import Path

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput

from .attn_vis import (
    LTXAttentionCapture,
    extract_qwen_attention_maps,
    frames_to_numpy,
    log_attn_frames_to_wandb,
)
from .multimodal_loop import (
    pre_encode_base_contexts,
    render_final_video,
    render_with_injected_latents,
)
from .qwen_loss import compute_qwen_video_loss

log = logging.getLogger(__name__)


def gradient_optimize_multimodal_qwen(
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
    # Qwen2.5-VL loss
    qwen_model,
    cached_qwen_inputs: dict,
    yes_token_id: int,
    no_token_id: int,
    eval_sample_start: int,
    visualize_retake_kwargs: dict | None = None,
    num_frames: int = 0,
    frame_rate: float = 25.0,
    audio_sr: int = 44100,
    wandb_run=None,
    extract_attn_maps: bool = False,
) -> dict:
    """Run gradient optimization for one mode and return the best result.

    Returns a dict with keys:
        "qwen_loss"   : best total loss (lower is better)
        "qwen_score"  : best Qwen yes-probability estimate (higher is better)
        "delta_v"     : best text delta tensor (or None if mode == "audio")
        "audio_latent": best audio latent tensor (or None if mode == "text")
        "mode"        : the optimization mode string
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
        log.info("[%s] Text delta shape: %s, %.1fK params", mode, tuple(delta_v.shape), delta_v.numel() / 1e3)

    if optimize_audio:
        audio_latent = torch.nn.Parameter(base_audio_latent_fp32.clone())
        params.append(audio_latent)
        log.info("[%s] Audio latent shape: %s, %.1fK params", mode, tuple(audio_latent.shape), audio_latent.numel() / 1e3)

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ---- Resume from checkpoint ----
    best_latent_path = output_dir / f"best_audio_latent_{mode}.pt"
    best_delta_path = output_dir / f"best_text_delta_{mode}.pt"
    num_iters = args.iterations

    best: dict = {
        "qwen_loss": float("inf"),
        "qwen_score": float("-inf"),
        "delta_v": delta_v.detach().clone() if delta_v is not None else None,
        "audio_latent": audio_latent.detach().clone() if audio_latent is not None else None,
        "mode": mode,
        "best_iter": 0,
        # Keep these aliases so render_final_video (imported from multimodal_loop) works unchanged
        "clip_loss": float("inf"),
        "clip_score": float("-inf"),
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
    csv_path = output_dir / f"optimization_log_qwen_{mode}.csv"
    csv_file = csv_path.open("w", newline="") if is_main else open("/dev/null", "w", newline="")
    csv_writer = csv.writer(csv_file)
    if is_main:
        csv_writer.writerow([
            "iter", "qwen_nll", "qwen_yes_prob",
            "audio_reg", "text_reg", "total_loss", "grad_norm", "is_best",
        ])

    # ---- W&B Table (per-iteration metrics — sortable/filterable in W&B UI) ----
    wandb_table = None
    if wandb_run is not None and is_main:
        try:
            import wandb as _wandb
            wandb_table = _wandb.Table(
                columns=["iter", "yes_prob", "qwen_nll", "audio_reg",
                         "text_reg", "total_loss", "grad_norm", "is_best"]
            )
        except Exception:
            wandb_table = None

    # ---- TensorBoard (optional) ----
    tb_writer = None
    if is_main and _TB_AVAILABLE:
        tb_dir = output_dir / "tensorboard"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        log.info("[%s] TensorBoard logs → %s", mode, tb_dir)
    elif is_main:
        log.warning("tensorboard not installed — metrics logged to CSV only.")

    preview_every = max(int(getattr(args, "visualize_every_iters", 0) or 0), 0)
    preview_root = output_dir / "visualizations"

    # ---- Save baseline audio (before any optimisation) ----
    if is_main and optimize_audio and preview_every > 0:
        baseline_audio_dir = output_dir / "visualizations" / "iter_000"
        baseline_audio_dir.mkdir(parents=True, exist_ok=True)
        _save_preview_audio(
            it=0,
            mode=mode,
            label="baseline",
            audio_latent=base_audio_latent_fp32,
            pipeline=pipeline,
            audio_sr=audio_sr,
            preview_dir=baseline_audio_dir,
            wandb_run=wandb_run,
        )

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

            # Qwen2.5-VL alignment loss
            qwen_loss_t = compute_qwen_video_loss(
                frames_chw=gen_frames,
                qwen_model=qwen_model,
                cached_inputs=cached_qwen_inputs,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                max_frames=args.qwen_max_frames,
                img_size=args.qwen_img_size,
            )

            # Regularization (same as multimodal_loop)
            audio_reg_t = torch.tensor(0.0, device=qwen_loss_t.device)
            if optimize_audio and audio_latent is not None and args.latent_reg_weight > 0:
                audio_reg_t = args.latent_reg_weight * torch.mean(
                    (audio_latent - base_audio_latent_fp32) ** 2
                )

            text_reg_t = torch.tensor(0.0, device=qwen_loss_t.device)
            if optimize_text and delta_v is not None and args.text_reg_weight > 0:
                text_reg_t = args.text_reg_weight * torch.mean(delta_v ** 2)

            total_t = qwen_loss_t + audio_reg_t + text_reg_t
            total_t.backward()

            grad_norm = 0.0
            if args.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip).item())
            elif params[0].grad is not None:
                grad_norm = float(sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5)

            optimizer.step()

            qwen_loss = float(qwen_loss_t.detach().item())
            qwen_score = math.exp(-qwen_loss)
            audio_reg = float(audio_reg_t.detach().item())
            text_reg = float(text_reg_t.detach().item())
            total = float(total_t.detach().item())

            is_best = total < best["qwen_loss"]
            if is_best:
                best["qwen_loss"] = total
                best["qwen_score"] = qwen_score
                best["clip_loss"] = total    # alias for render_final_video compat
                best["clip_score"] = qwen_score
                best["best_iter"] = it
                if delta_v is not None:
                    best["delta_v"] = delta_v.detach().clone()
                if audio_latent is not None:
                    best["audio_latent"] = audio_latent.detach().clone()

            iters_without_improvement = it - best.get("best_iter", 0)

            if is_main:
                csv_writer.writerow([it, qwen_loss, qwen_score, audio_reg, text_reg, total, grad_norm, int(is_best)])
                csv_file.flush()
                if wandb_table is not None:
                    wandb_table.add_data(it, qwen_score, qwen_loss, audio_reg,
                                         text_reg, total, grad_norm, int(is_best))
                log.info(
                    "[%s] iter %3d/%d  qwen_nll=%.4f  yes_prob=%.4f  total=%.4f  grad_norm=%.3f  no_improve=%d%s",
                    mode, it, num_iters, qwen_loss, qwen_score, total, grad_norm,
                    iters_without_improvement, "  ★" if is_best else "",
                )
                if tb_writer is not None:
                    tb_writer.add_scalar(f"{mode}/qwen_nll", qwen_loss, it)
                    tb_writer.add_scalar(f"{mode}/qwen_yes_prob", qwen_score, it)
                    tb_writer.add_scalar(f"{mode}/total_loss", total, it)
                    tb_writer.add_scalar(f"{mode}/grad_norm", grad_norm, it)
                    tb_writer.add_scalar(f"{mode}/iters_without_improvement", iters_without_improvement, it)
                    if optimize_audio:
                        tb_writer.add_scalar(f"{mode}/audio_reg", audio_reg, it)
                    if optimize_text:
                        tb_writer.add_scalar(f"{mode}/text_reg", text_reg, it)
                    if is_best:
                        tb_writer.add_scalar(f"{mode}/best_qwen_yes_prob", qwen_score, it)

                if wandb_run is not None:
                    wandb_payload = {
                        "global/iter": it,
                        f"{mode}/qwen_nll": qwen_loss,
                        f"{mode}/qwen_yes_prob": qwen_score,
                        f"{mode}/total_loss": total,
                        f"{mode}/grad_norm": grad_norm,
                        f"{mode}/iters_without_improvement": iters_without_improvement,
                        f"{mode}/is_best": int(is_best),
                    }
                    if optimize_audio:
                        wandb_payload[f"{mode}/audio_reg"] = audio_reg
                    if optimize_text:
                        wandb_payload[f"{mode}/text_reg"] = text_reg
                    if is_best:
                        wandb_payload[f"{mode}/best_qwen_yes_prob"] = qwen_score
                    wandb_run.log(wandb_payload, step=it)

                if (
                    preview_every > 0
                    and it % preview_every == 0
                    and visualize_retake_kwargs is not None
                    and num_frames > 0
                ):
                    preview_dir = preview_root / f"iter_{it:03d}"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    log.info("[%s] Rendering best-so-far preview at iter %d...", mode, it)

                    # ---- Render preview (optionally with LTX cross-attention hooks) ----
                    attn_cap = None
                    if extract_attn_maps and wandb_run is not None:
                        attn_cap = LTXAttentionCapture(pipeline, block_fraction=0.5)
                        attn_cap.__enter__()

                    render_final_video(
                        mode=mode,
                        best=best,
                        pipeline=pipeline,
                        src_video=retake_input_video,
                        cached_video_latent=cached_video_latent,
                        base_audio_latent=base_audio_latent,
                        base_pos_context=base_pos_context,
                        base_neg_context=base_neg_context,
                        retake_kwargs=visualize_retake_kwargs,
                        output_dir=preview_dir,
                        num_frames=num_frames,
                        frame_rate=frame_rate,
                        audio_sr=audio_sr,
                        audio_opt_last_steps=args.audio_opt_last_steps,
                        skip_baseline=True,
                    )

                    if attn_cap is not None:
                        attn_cap.__exit__(None, None, None)

                    # ---- Save optimized audio at this snapshot ----
                    if optimize_audio and audio_latent is not None:
                        _save_preview_audio(
                            it=it,
                            mode=mode,
                            label="optimized",
                            audio_latent=audio_latent.detach(),
                            pipeline=pipeline,
                            audio_sr=audio_sr,
                            preview_dir=preview_dir,
                            wandb_run=wandb_run,
                        )

                    if wandb_run is not None:
                        try:
                            import wandb

                            preview_path = preview_dir / f"best_optimized_video_{mode}.mp4"
                            if preview_path.exists():
                                wandb_run.log(
                                    {
                                        f"{mode}/preview_video": wandb.Video(
                                            str(preview_path),
                                            format="mp4",
                                            caption=f"{mode} best-so-far at iter {it}",
                                        )
                                    },
                                    step=it,
                                )
                        except Exception:
                            log.exception("[%s] Failed to log preview video to W&B at iter %d", mode, it)

                    # ---- Attention map extraction & W&B logging ----
                    if extract_attn_maps and wandb_run is not None:
                        _log_attention_maps(
                            it=it,
                            mode=mode,
                            label="optimized",
                            gen_frames=gen_frames,
                            qwen_model=qwen_model,
                            cached_qwen_inputs=cached_qwen_inputs,
                            qwen_max_frames=args.qwen_max_frames,
                            qwen_img_size=args.qwen_img_size,
                            attn_cap=attn_cap,
                            cached_video_latent=cached_video_latent,
                            wandb_run=wandb_run,
                        )

            early_stop_limit = getattr(args, "early_stopping", 0)
            if early_stop_limit > 0 and iters_without_improvement >= early_stop_limit:
                log.info(
                    "[%s] Early stopping: no improvement for %d consecutive iters (best at iter %d).",
                    mode, iters_without_improvement, best.get("best_iter", 0),
                )
                break
    finally:
        csv_file.close()
        if tb_writer is not None:
            tb_writer.close()
        # Log the per-iteration metrics table to W&B
        if wandb_table is not None and wandb_run is not None:
            try:
                wandb_run.log({f"{mode}/metrics_table": wandb_table})
            except Exception:
                log.warning("[%s] Failed to log W&B metrics table", mode, exc_info=True)

    return best


# ---------------------------------------------------------------------------
# Attention map extraction helper (called at preview steps)
# ---------------------------------------------------------------------------

def _log_attention_maps(
    *,
    it: int,
    mode: str,
    label: str,
    gen_frames: torch.Tensor,
    qwen_model,
    cached_qwen_inputs: dict,
    qwen_max_frames: int,
    qwen_img_size: int,
    attn_cap: "LTXAttentionCapture | None",
    cached_video_latent: torch.Tensor,
    wandb_run,
) -> None:
    """Extract Qwen + LTX attention maps and log overlays to W&B.

    Called only at preview_every steps inside a no_grad context (gen_frames
    has already been detached from the compute graph by the time we get here
    since optimizer.step() was called before the preview render).

    Memory notes (H100):
      - Qwen output_attentions=True: ~140 MB extra; freed before returning.
      - LTX hooks: accumulate one [Tv] vector per step; ~200 KB total.
    """
    frames_np = frames_to_numpy(gen_frames)  # [N, H, W, 3] uint8

    # ---- 1. Qwen self-attention: where does the scorer look? ----
    log.info("[%s] Extracting Qwen attention maps at iter %d...", mode, it)
    qwen_result = extract_qwen_attention_maps(
        frames_chw=gen_frames.detach(),
        qwen_model=qwen_model,
        cached_inputs=cached_qwen_inputs,
        max_frames=qwen_max_frames,
        img_size=qwen_img_size,
        num_layers_to_avg=4,
    )
    if qwen_result is not None:
        # qwen_result["attn_spatial"]: [grid_t, grid_h, grid_w]
        attn_q = qwen_result["attn_spatial"]  # [gt, gh, gw]
        # Mean over temporal grid → 2-D spatial map for overlay
        attn_q_spatial = attn_q.mean(axis=0)  # [gh, gw]
        log_attn_frames_to_wandb(
            wandb_run=wandb_run,
            tag=f"{mode}/attn_qwen_{label}",
            frames_np=frames_np,
            heatmap=attn_q_spatial,
            step=it,
            caption=f"Qwen scorer attention — {label} iter {it}",
        )
        log.info("[%s] Qwen attention logged (grid %s).", mode, qwen_result["grid_thw"])

    # ---- 2. LTX audio→video cross-attention: where does audio drive video? ----
    if attn_cap is not None:
        for key, tag_suffix, caption_suffix in [
            ("audio_to_video", "attn_ltx_audio", "LTX audio→video attention"),
            ("text_to_video",  "attn_ltx_text",  "LTX text→video attention"),
        ]:
            attn_grid = attn_cap.reshape_to_video_grid(key, cached_video_latent)
            if attn_grid is not None:
                # Average over temporal latent dimension → 2-D
                if attn_grid.ndim == 3:
                    attn_2d = attn_grid.mean(axis=0)
                else:
                    attn_2d = attn_grid.squeeze()
                log_attn_frames_to_wandb(
                    wandb_run=wandb_run,
                    tag=f"{mode}/{tag_suffix}_{label}",
                    frames_np=frames_np,
                    heatmap=attn_2d,
                    step=it,
                    caption=f"{caption_suffix} — {label} iter {it}",
                )
                log.info("[%s] LTX %s attention logged.", mode, key)


# ---------------------------------------------------------------------------
# Audio decode + save helper (called at preview steps)
# ---------------------------------------------------------------------------

def _save_preview_audio(
    *,
    it: int,
    mode: str,
    label: str,
    audio_latent: torch.Tensor,
    pipeline,
    audio_sr: int,
    preview_dir: Path,
    wandb_run,
) -> None:
    """Decode an audio latent back to waveform and save as WAV.

    Uses the pipeline's audio_decoder + vocoder so the round-trip matches
    exactly what the model hears.  All work is done under torch.no_grad().

    Files are saved as:
        <preview_dir>/<label>_audio_<mode>.wav

    Also logs to W&B as wandb.Audio when wandb_run is not None.
    """
    try:
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from .core import save_audio_wav

        audio_decoder = pipeline.model_ledger.audio_decoder()
        vocoder = pipeline.model_ledger.vocoder()

        with torch.no_grad():
            decoded = vae_decode_audio(
                audio_latent.to(dtype=next(audio_decoder.parameters()).dtype),
                audio_decoder,
                vocoder,
            )

        wav_path = preview_dir / f"{label}_audio_{mode}.wav"
        save_audio_wav(decoded.waveform, decoded.sampling_rate, str(wav_path))
        log.info("[%s] Saved %s audio iter %d → %s (sr=%d)",
                 mode, label, it, wav_path.name, decoded.sampling_rate)

        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log(
                    {
                        f"{mode}/{label}_audio": wandb.Audio(
                            str(wav_path),
                            sample_rate=decoded.sampling_rate,
                            caption=f"{mode} {label} audio — iter {it}",
                        )
                    },
                    step=it,
                )
            except Exception:
                log.warning("[%s] Failed to log audio to W&B at iter %d", mode, it, exc_info=True)

    except Exception:
        log.warning("[%s] Failed to decode/save audio at iter %d", mode, it, exc_info=True)


__all__ = [
    "gradient_optimize_multimodal_qwen",
    "pre_encode_base_contexts",
    "render_final_video",
    "render_with_injected_latents",
]
