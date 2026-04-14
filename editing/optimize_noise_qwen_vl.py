#!/usr/bin/env python3
"""Joint noise + latent optimization using Qwen2.5-VL as the alignment loss.

Extends the standard text/audio latent optimization with a third learnable
parameter: **video_delta** — a small perturbation in the source video's VAE
latent space.  This directly modifies the denoising trajectory in two ways:

  1. **Starting point**: x_t = (V₀ + Δ) + σ·ε   (the initial noised state)
  2. **Clean-latent blending**: at every denoising step, the retake mask
     blends the denoised prediction with clean_latent = V₀ + Δ.

Because clean_latent is accessed at *every* step, gradients flow through it
during the last-N steps without needing full backprop through the entire
diffusion chain — the same mechanism that makes audio_latent optimization
work.  Memory overhead is negligible (~4 MB for the delta + ~8 MB Adam state).

Novel ideas implemented here
-----------------------------
1. **Latent-space trajectory steering**: jointly optimizing text (semantic
   direction), audio (modality conditioning), and video latent (denoising
   trajectory) provides three complementary axes of control.

2. **Cosine warm-in for video_delta**: the video_delta regularization weight
   starts HIGH and decays via cosine schedule.  This lets text/audio
   establish the edit direction first; video_delta only kicks in once the
   semantic target is stable, avoiding destructive interference.

3. **Frequency-decoupled regularization**: low spatial-frequency components
   of video_delta are penalized more strongly than high-frequency ones.
   This preserves global appearance (colour, lighting) while allowing
   fine-grained motion / texture changes that matter for the edit.

Usage
-----
    python editing/optimize_noise_qwen_vl.py \\
        --src-video input_videos/rabbit.mp4 \\
        --edit-prompt "The rabbit raises its paw and waves" \\
        --output-dir results/noise_opt/rabbit_wave \\
        --opt-mode both_vd \\
        --quantization fp8-cast --gradient-checkpointing
"""
from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

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
from audio_latent_opt.multimodal_loop import (
    pre_encode_base_contexts,
    render_with_injected_latents,
    render_final_video,
    render_baseline_video,
)
from audio_latent_opt.multimodal_loop_qwen import _clear_cuda_cache
from audio_latent_opt.perceptual_loss import (
    adaptive_reg_weight,
    cache_source_frames,
    compute_perceptual_quality_loss,
)
from audio_latent_opt.qwen_loss import (
    DEFAULT_QWEN_MOTION_QUESTION,
    QWEN_IMG_SIZE,
    build_qwen_model,
    build_qwen_rubric_inputs,
    compute_qwen_video_loss,
)
from audio_latent_opt.runtime import build_retake_kwargs, prepare_retake_input_video

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_pipelines.utils.constants import detect_params

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_mode(mode: str) -> str:
    """Map our extended mode names to the standard modes understood by
    render_final_video (which checks ``mode in ("text", "both")`` etc.).
    """
    has_text = "text" in mode or mode.startswith("both") or mode == "all"
    has_audio = "audio" in mode or mode.startswith("both") or mode == "all"
    if has_text and has_audio:
        return "both"
    if has_text:
        return "text"
    if has_audio:
        return "audio"
    return "audio"  # fallback: render_final_video needs at least one axis


# ---------------------------------------------------------------------------
# Frequency-decoupled regularisation for video_delta
# ---------------------------------------------------------------------------

def freq_decoupled_reg(delta: torch.Tensor, low_freq_weight: float = 3.0) -> torch.Tensor:
    """L2 regularisation that penalises low spatial frequencies more strongly.

    Rationale: low-frequency changes (colour shifts, brightness) are visually
    disruptive and don't help motion edits.  High-frequency changes (edges,
    textures, local motion cues) are what the edit needs.

    Implementation: 3×3 average-pool across spatial dims to extract the
    low-freq component; penalise it with ``low_freq_weight`` × the base L2
    penalty, and the residual (high-freq) with 1× the base penalty.
    """
    # delta: (1, C, T, H, W)  — all in latent space (small spatial dims)
    if delta.dim() != 5 or delta.shape[3] < 3 or delta.shape[4] < 3:
        return torch.mean(delta ** 2)

    # Low-freq: 3×3 spatial average (keeps temporal + channel dims intact)
    low = F.avg_pool3d(delta, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    high = delta - low

    return low_freq_weight * torch.mean(low ** 2) + torch.mean(high ** 2)


def cosine_warm_in(it: int, total: int, max_val: float, min_val: float) -> float:
    """Cosine schedule that starts at *max_val* and decays to *min_val*.

    Used for video_delta regularisation: heavy reg early (delta stays near
    zero while text/audio establish direction) → lighter reg later (delta
    contributes once the semantic target is stable).
    """
    if total <= 1:
        return min_val
    t = min(it / total, 1.0)
    return min_val + 0.5 * (max_val - min_val) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Core optimisation loop
# ---------------------------------------------------------------------------

def gradient_optimize_with_video_delta(
    *,
    mode: str,
    args,
    output_dir: Path,
    # Text
    base_pos_context: EmbeddingsProcessorOutput,
    base_neg_context: EmbeddingsProcessorOutput,
    # Audio
    base_audio_latent: torch.Tensor,
    base_audio_latent_fp32: torch.Tensor,
    # Video
    cached_video_latent: torch.Tensor,
    retake_input_video: str,
    pipeline,
    retake_kwargs: dict,
    # Qwen
    qwen_model,
    cached_qwen_inputs: dict,
    yes_token_id: int,
    no_token_id: int,
    eval_sample_start: int,
    # Optional
    visualize_retake_kwargs: dict | None = None,
    num_frames: int = 0,
    frame_rate: float = 25.0,
    audio_sr: int = 44100,
    wandb_run=None,
    cached_src_frames: torch.Tensor | None = None,
) -> dict:
    """Optimisation loop that jointly optimises text, audio, AND video latent delta."""

    optimize_text = "text" in mode or mode.startswith("both") or mode == "all"
    optimize_audio = "audio" in mode or mode.startswith("both") or mode == "all"
    optimize_vd = "vd" in mode or mode == "all"

    # ---- Build parameters ----
    params: list[torch.nn.Parameter] = []
    delta_v: torch.nn.Parameter | None = None
    audio_latent: torch.nn.Parameter | None = None
    video_delta: torch.nn.Parameter | None = None

    if optimize_text:
        delta_v = torch.nn.Parameter(
            torch.zeros_like(base_pos_context.video_encoding.float())
        )
        params.append(delta_v)
        log.info("[%s] Text delta: shape=%s  (%.1fK params)",
                 mode, tuple(delta_v.shape), delta_v.numel() / 1e3)

    if optimize_audio:
        audio_latent = torch.nn.Parameter(base_audio_latent_fp32.clone())
        params.append(audio_latent)
        log.info("[%s] Audio latent: shape=%s  (%.1fK params)",
                 mode, tuple(audio_latent.shape), audio_latent.numel() / 1e3)

    if optimize_vd:
        video_delta = torch.nn.Parameter(
            torch.zeros(
                cached_video_latent.shape,
                dtype=torch.float32,
                device=cached_video_latent.device,
            )
        )
        params.append(video_delta)
        log.info("[%s] Video delta: shape=%s  (%.1fK params, %.2f MB)",
                 mode, tuple(video_delta.shape), video_delta.numel() / 1e3,
                 video_delta.numel() * 4 / 1e6)

    optimizer = torch.optim.Adam(params, lr=args.lr)

    lr_schedule = getattr(args, "lr_schedule", "constant")
    scheduler = None
    if lr_schedule == "cosine" and args.iterations > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations, eta_min=args.lr * 0.01,
        )

    num_iters = args.iterations
    best: dict = {
        "qwen_loss": float("inf"),
        "qwen_score": float("-inf"),
        "delta_v": delta_v.detach().clone() if delta_v is not None else None,
        "audio_latent": audio_latent.detach().clone() if audio_latent is not None else None,
        "video_delta": video_delta.detach().clone() if video_delta is not None else None,
        "mode": mode,
        "best_iter": 0,
        # Compat aliases for render_final_video
        "clip_loss": float("inf"),
        "clip_score": float("-inf"),
    }

    # ---- CSV log ----
    csv_path = output_dir / f"optimization_log_{mode}.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    rubric_metric_names = [
        str(item["name"])
        for item in cached_qwen_inputs.get("rubric_items", [])
    ]
    csv_header = [
        "iter", "qwen_nll", "qwen_yes_prob",
        "audio_reg", "text_reg", "vd_reg", "perceptual_loss",
        "total_loss", "grad_norm", "is_best",
    ]
    for name in rubric_metric_names:
        csv_header.extend([f"{name}_nll", f"{name}_yes_prob"])
    csv_writer.writerow(csv_header)

    vd_reg_base = getattr(args, "video_delta_reg_weight", 0.1)
    vd_reg_max = getattr(args, "video_delta_reg_max", vd_reg_base * 5.0)
    vd_low_freq_weight = getattr(args, "vd_low_freq_weight", 3.0)

    try:
        for it in range(1, num_iters + 1):
            optimizer.zero_grad(set_to_none=True)

            # ---- Build per-iteration video latent ----
            if video_delta is not None:
                perturbed_video_latent = (
                    cached_video_latent + video_delta.to(dtype=cached_video_latent.dtype)
                )
            else:
                perturbed_video_latent = cached_video_latent

            # ---- Build per-iteration text context ----
            if optimize_text and delta_v is not None:
                pos_ctx_iter = EmbeddingsProcessorOutput(
                    video_encoding=(
                        base_pos_context.video_encoding
                        + delta_v.to(dtype=base_pos_context.video_encoding.dtype)
                    ),
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

            # ---- Differentiable render ----
            gen_frames = render_with_injected_latents(
                pipeline=pipeline,
                src_video=retake_input_video,
                injected_audio_latent=audio_for_render,
                cached_video_latent=perturbed_video_latent,
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

            # ---- Perceptual loss (backward with retain_graph) ----
            perceptual_loss_t = torch.tensor(0.0, device=gen_frames.device)
            lpips_weight = getattr(args, "lpips_weight", 0.0)
            temporal_weight = getattr(args, "temporal_weight", 0.0)
            need_perceptual = cached_src_frames is not None and (lpips_weight > 0 or temporal_weight > 0)
            if need_perceptual:
                perceptual_loss_t, _ = compute_perceptual_quality_loss(
                    gen_frames=gen_frames,
                    src_frames=cached_src_frames,
                    lpips_weight=lpips_weight,
                    temporal_weight=temporal_weight,
                    backbone=getattr(args, "lpips_backbone", "alex"),
                    max_lpips_frames=min(16, gen_frames.shape[0]),
                    max_temporal_pairs=min(12, gen_frames.shape[0] - 1),
                    backward=False,
                )
                if perceptual_loss_t.requires_grad:
                    perceptual_loss_t.backward(retain_graph=True)
                    perceptual_loss_t = perceptual_loss_t.detach()

            # ---- Qwen loss (with optional gradient accumulation) ----
            rubric_weight_overrides = None
            qwen_gradient_rubric = getattr(args, "qwen_gradient_rubric", "motion")
            if qwen_gradient_rubric != "full":
                rubric_weight_overrides = {
                    "motion": 0.0, "entities": 0.0, "overall": 0.0,
                    str(qwen_gradient_rubric): 1.0,
                }

            qwen_grad_accum = max(1, int(getattr(args, "qwen_grad_accum_steps", 1)))

            if qwen_grad_accum == 1:
                qwen_loss_t, qwen_details = compute_qwen_video_loss(
                    frames_chw=gen_frames,
                    qwen_model=qwen_model,
                    cached_inputs=cached_qwen_inputs,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    max_frames=args.qwen_max_frames,
                    img_size=args.qwen_img_size,
                    backward=True,
                    return_details=True,
                    sample_mode=getattr(args, "qwen_sample_mode", "linspace"),
                    contiguous_start_frame=getattr(args, "qwen_contiguous_start_frame", 0),
                    rubric_weight_overrides=rubric_weight_overrides,
                )
            else:
                gf_det = gen_frames.detach().requires_grad_(True)
                accum_loss = torch.zeros((), device=gen_frames.device)
                qwen_details = []
                base_sample_mode = getattr(args, "qwen_sample_mode", "linspace")
                for accum_i in range(qwen_grad_accum):
                    pass_mode = base_sample_mode if accum_i == 0 else "contiguous_random"
                    _result = compute_qwen_video_loss(
                        frames_chw=gf_det,
                        qwen_model=qwen_model,
                        cached_inputs=cached_qwen_inputs,
                        yes_token_id=yes_token_id,
                        no_token_id=no_token_id,
                        max_frames=args.qwen_max_frames,
                        img_size=args.qwen_img_size,
                        backward=False,
                        return_details=(accum_i == 0),
                        sample_mode=pass_mode,
                        contiguous_start_frame=getattr(args, "qwen_contiguous_start_frame", 0),
                        rubric_weight_overrides=rubric_weight_overrides,
                    )
                    if accum_i == 0:
                        loss_i, qwen_details = _result
                    else:
                        loss_i = _result
                    (loss_i / qwen_grad_accum).backward()
                    accum_loss = accum_loss + loss_i.detach()
                qwen_loss_t = accum_loss / qwen_grad_accum
                if gf_det.grad is not None and gen_frames.grad_fn is not None:
                    gen_frames.backward(gradient=gf_det.grad)

            # ---- Regularisation ----
            reg_schedule = getattr(args, "reg_schedule", "constant")

            # Audio reg
            audio_reg_t = torch.tensor(0.0, device=gen_frames.device)
            if optimize_audio and audio_latent is not None:
                audio_reg_w = adaptive_reg_weight(
                    args.latent_reg_weight, it, num_iters, schedule=reg_schedule,
                )
                audio_reg_t = audio_reg_w * torch.mean(
                    (audio_latent - base_audio_latent_fp32) ** 2
                )

            # Text reg
            text_reg_t = torch.tensor(0.0, device=gen_frames.device)
            if optimize_text and delta_v is not None:
                text_reg_w = adaptive_reg_weight(
                    args.text_reg_weight, it, num_iters, schedule=reg_schedule,
                )
                text_reg_t = text_reg_w * torch.mean(delta_v ** 2)

            # Video delta reg — cosine warm-in: strong early → weaker later
            vd_reg_t = torch.tensor(0.0, device=gen_frames.device)
            if optimize_vd and video_delta is not None:
                vd_reg_w = cosine_warm_in(it, num_iters, vd_reg_max, vd_reg_base)
                vd_reg_t = vd_reg_w * freq_decoupled_reg(
                    video_delta.unsqueeze(0) if video_delta.dim() == 4 else video_delta,
                    low_freq_weight=vd_low_freq_weight,
                )

            reg_t = audio_reg_t + text_reg_t + vd_reg_t
            if reg_t.requires_grad:
                reg_t.backward()

            # ---- Gradient clip + step ----
            grad_norm = 0.0
            if args.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip).item())
            elif params[0].grad is not None:
                grad_norm = float(sum(
                    p.grad.norm().item() ** 2 for p in params if p.grad is not None
                ) ** 0.5)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # ---- Logging ----
            qwen_loss = float(qwen_loss_t.detach().item())
            qwen_score = math.exp(-qwen_loss)
            audio_reg = float(audio_reg_t.detach().item())
            text_reg = float(text_reg_t.detach().item())
            vd_reg = float(vd_reg_t.detach().item())
            perceptual_loss = float(perceptual_loss_t.detach().item()) if torch.is_tensor(perceptual_loss_t) else 0.0
            total = float(qwen_loss_t.detach().item()) + audio_reg + text_reg + vd_reg + perceptual_loss

            is_best = total < best["qwen_loss"]
            if is_best:
                best["qwen_loss"] = total
                best["qwen_score"] = qwen_score
                best["clip_loss"] = total
                best["clip_score"] = qwen_score
                best["best_iter"] = it
                if delta_v is not None:
                    best["delta_v"] = delta_v.detach().clone()
                if audio_latent is not None:
                    best["audio_latent"] = audio_latent.detach().clone()
                if video_delta is not None:
                    best["video_delta"] = video_delta.detach().clone()

            iters_without_improvement = it - best.get("best_iter", 0)

            # CSV
            detail_by_name = {str(item["name"]): item for item in qwen_details}
            rubric_values = []
            for name in rubric_metric_names:
                item = detail_by_name.get(name, {})
                rubric_values.extend([item.get("nll", ""), item.get("yes_prob", "")])
            csv_writer.writerow([
                it, qwen_loss, qwen_score, audio_reg, text_reg, vd_reg,
                perceptual_loss, total, grad_norm, int(is_best),
                *rubric_values,
            ])
            csv_file.flush()

            rubric_log = ""
            if qwen_details:
                rubric_log = "  rubric=" + ", ".join(
                    f"{item['name']}:{float(item['yes_prob']):.4f}"
                    for item in qwen_details
                )
            vd_norm_log = ""
            if video_delta is not None:
                vd_norm_log = f"  vd_norm={video_delta.data.norm():.4f}"

            log.info(
                "[%s] iter %3d/%d  yes_prob=%.4f  total=%.4f  "
                "vd_reg=%.4f  grad=%.3f  no_impr=%d%s%s%s",
                mode, it, num_iters, qwen_score, total,
                vd_reg, grad_norm, iters_without_improvement,
                "  ★" if is_best else "", vd_norm_log, rubric_log,
            )

            if wandb_run is not None:
                payload = {
                    "global/iter": it,
                    f"{mode}/qwen_yes_prob": qwen_score,
                    f"{mode}/qwen_nll": qwen_loss,
                    f"{mode}/total_loss": total,
                    f"{mode}/grad_norm": grad_norm,
                    f"{mode}/is_best": int(is_best),
                }
                if optimize_audio:
                    payload[f"{mode}/audio_reg"] = audio_reg
                if optimize_text:
                    payload[f"{mode}/text_reg"] = text_reg
                if optimize_vd:
                    payload[f"{mode}/vd_reg"] = vd_reg
                    payload[f"{mode}/vd_norm"] = float(video_delta.data.norm())
                for item in qwen_details:
                    name = str(item["name"])
                    payload[f"{mode}/rubric/{name}_nll"] = float(item["nll"])
                    payload[f"{mode}/rubric/{name}_yes_prob"] = float(item["yes_prob"])
                wandb_run.log(payload, step=it)

            # ---- Periodic preview ----
            preview_every = max(int(getattr(args, "visualize_every_iters", 0) or 0), 0)
            if (
                preview_every > 0
                and it % preview_every == 0
                and visualize_retake_kwargs is not None
                and num_frames > 0
            ):
                preview_dir = output_dir / "visualizations" / f"iter_{it:03d}"
                preview_dir.mkdir(parents=True, exist_ok=True)

                # Build perturbed video latent for the best checkpoint
                best_vl = cached_video_latent
                if best.get("video_delta") is not None:
                    best_vl = cached_video_latent + best["video_delta"].to(
                        dtype=cached_video_latent.dtype
                    )

                render_final_video(
                    mode=_render_mode(mode),
                    best=best,
                    pipeline=pipeline,
                    src_video=retake_input_video,
                    cached_video_latent=best_vl,
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
                _clear_cuda_cache()

            # Early stopping
            early_stop_limit = getattr(args, "early_stopping", 0)
            if early_stop_limit > 0 and iters_without_improvement >= early_stop_limit:
                log.info("[%s] Early stopping at iter %d (no improvement for %d iters).",
                         mode, it, iters_without_improvement)
                break

            del gen_frames, qwen_loss_t, audio_reg_t, text_reg_t, vd_reg_t, reg_t
            _clear_cuda_cache()

    finally:
        csv_file.close()

    return best


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    wandb_run = None
    if wandb is not None and os.environ.get("WANDB_DISABLED", "").lower() not in {"1", "true"}:
        prompt_slug = re.sub(r"[^a-z0-9]+", "-", args.edit_prompt.lower())[:40].strip("-")
        job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%m%d-%H%M%S"))
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"{prompt_slug}-{args.opt_mode}-{job_id}",
            config=vars(args),
            dir=os.environ.get("WANDB_DIR"),
        )

    try:
        # ---- Shape ----
        height, width, num_frames, frame_rate = compute_target_shape(
            args.src_video, args.height, args.width, args.num_frames, args.frame_rate,
        )
        duration = num_frames / frame_rate
        log.info("Video: %dx%d, %d frames @ %.1f fps (%.2fs)", width, height, num_frames, frame_rate, duration)

        # ---- Quantization ----
        retake_quant = resolve_quantization_policy(
            args.retake_quantization if args.retake_quantization is not None else args.quantization
        )

        # ---- Input video ----
        retake_input_video = prepare_retake_input_video(
            args=args, is_main=True, output_dir=output_dir,
            height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
        )

        # ---- Guiders ----
        params = detect_params(args.checkpoint_path)
        video_guider_params, audio_guider_params = build_guiders_for_mode(
            args=args, params=params, use_low_memory_guidance=args.low_memory_guidance,
        )

        # ---- Pipeline ----
        log.info("Loading RetakePipeline...")
        pipeline = build_retake_pipeline(
            checkpoint_path=args.checkpoint_path,
            gemma_root=args.gemma_root,
            loras=_parse_loras(args.loras),
            device=device,
            quant_policy=retake_quant,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        # ---- Cache source latents ----
        cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
            pipeline=pipeline, src_video=str(retake_input_video),
            height=height, width=width, num_frames=num_frames,
            audio_sr=args.audio_sr, device=device,
        )
        base_audio_latent_fp32 = base_audio_latent.float().detach()
        log.info("Video latent shape: %s (%.2f MB fp32)",
                 tuple(cached_video_latent.shape),
                 cached_video_latent.numel() * 4 / 1e6)

        # ---- Text contexts ----
        base_pos_context, base_neg_context = pre_encode_base_contexts(
            pipeline=pipeline, pos_prompt=args.edit_prompt,
            neg_prompt=args.negative_prompt, device=device,
        )

        # ---- Qwen ----
        log.info("Loading Qwen2.5-VL (%s)...", args.qwen_model)
        qwen_model, qwen_processor = build_qwen_model(
            args.qwen_model, device=device,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        qwen_num_frames = args.qwen_max_frames + (args.qwen_max_frames % 2)
        cached_qwen_inputs, yes_token_id, no_token_id = build_qwen_rubric_inputs(
            processor=qwen_processor, edit_prompt=args.edit_prompt,
            num_frames=qwen_num_frames, img_size=args.qwen_img_size, device=device,
            motion_question=args.qwen_motion_question,
        )

        # ---- Retake kwargs ----
        retake_kwargs = build_retake_kwargs(
            args=args, frame_rate=frame_rate, duration=duration,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )
        eval_sample_start = 0

        # ---- Source frames for perceptual loss ----
        cached_src_frames = None
        if args.lpips_weight > 0 or args.temporal_weight > 0:
            cached_src_frames = cache_source_frames(
                pipeline=pipeline, src_video=str(retake_input_video),
                cached_video_latent=cached_video_latent,
                base_audio_latent=base_audio_latent,
                base_pos_context=base_pos_context,
                base_neg_context=base_neg_context,
                retake_kwargs=retake_kwargs,
                max_frames=args.max_eval_frames,
                frame_stride=args.frame_stride,
                eval_sample_start=eval_sample_start,
            )

        # ---- Final-render kwargs ----
        final_retake_kwargs = dict(retake_kwargs)
        if args.final_retake_num_inference_steps is not None:
            final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps
        final_vg, final_ag = build_guiders_for_mode(args=args, params=params, use_low_memory_guidance=False)
        final_retake_kwargs["video_guider_params"] = final_vg
        final_retake_kwargs["audio_guider_params"] = final_ag

        # ---- Baseline ----
        if args.save_final_videos:
            log.info("Rendering baseline video...")
            render_baseline_video(
                pipeline=pipeline, src_video=str(retake_input_video),
                cached_video_latent=cached_video_latent,
                base_audio_latent=base_audio_latent,
                base_pos_context=base_pos_context,
                base_neg_context=base_neg_context,
                retake_kwargs=final_retake_kwargs,
                output_path=output_dir / "baseline_video.mp4",
                num_frames=num_frames, frame_rate=frame_rate, audio_sr=waveform_sr,
            )

        # ---- Optimise ----
        modes = [m.strip() for m in args.opt_mode.split(",")]
        all_results: dict[str, dict] = {}

        for mode in modes:
            log.info("=" * 60)
            log.info("  Optimisation mode: %s", mode.upper())
            log.info("=" * 60)

            mode_dir = output_dir / f"mode_{mode}"
            mode_dir.mkdir(parents=True, exist_ok=True)

            best = gradient_optimize_with_video_delta(
                mode=mode, args=args, output_dir=mode_dir,
                base_pos_context=base_pos_context,
                base_neg_context=base_neg_context,
                base_audio_latent=base_audio_latent,
                base_audio_latent_fp32=base_audio_latent_fp32,
                cached_video_latent=cached_video_latent,
                retake_input_video=str(retake_input_video),
                pipeline=pipeline, retake_kwargs=retake_kwargs,
                qwen_model=qwen_model,
                cached_qwen_inputs=cached_qwen_inputs,
                yes_token_id=yes_token_id, no_token_id=no_token_id,
                eval_sample_start=eval_sample_start,
                visualize_retake_kwargs=final_retake_kwargs,
                num_frames=num_frames, frame_rate=frame_rate,
                audio_sr=waveform_sr, wandb_run=wandb_run,
                cached_src_frames=cached_src_frames,
            )
            all_results[mode] = best

            # Save
            if best.get("audio_latent") is not None:
                torch.save(best["audio_latent"].cpu(), mode_dir / f"best_audio_latent_{mode}.pt")
            if best.get("delta_v") is not None:
                torch.save(best["delta_v"].cpu(), mode_dir / f"best_text_delta_{mode}.pt")
            if best.get("video_delta") is not None:
                torch.save(best["video_delta"].cpu(), mode_dir / f"best_video_delta_{mode}.pt")
            torch.save(
                {"mode": mode, "qwen_loss": best["qwen_loss"], "qwen_score": best["qwen_score"]},
                mode_dir / f"best_params_{mode}.pt",
            )

            log.info("[%s] Done — best yes_prob: %.4f (total: %.4f, iter %d)",
                     mode, best["qwen_score"], best["qwen_loss"], best.get("best_iter", 0))

            # ---- Final render with best video_delta ----
            if args.save_final_videos:
                best_vl = cached_video_latent
                if best.get("video_delta") is not None:
                    best_vl = cached_video_latent + best["video_delta"].to(
                        dtype=cached_video_latent.dtype
                    )
                render_final_video(
                    mode=mode, best=best, pipeline=pipeline,
                    src_video=str(retake_input_video),
                    cached_video_latent=best_vl,
                    base_audio_latent=base_audio_latent,
                    base_pos_context=base_pos_context,
                    base_neg_context=base_neg_context,
                    retake_kwargs=final_retake_kwargs,
                    output_dir=mode_dir,
                    num_frames=num_frames, frame_rate=frame_rate,
                    audio_sr=waveform_sr,
                    audio_opt_last_steps=args.audio_opt_last_steps,
                    skip_baseline=True,
                )

            gc.collect()
            torch.cuda.empty_cache()

        # ---- Summary ----
        log.info("")
        log.info("=" * 60)
        log.info("COMPARISON SUMMARY")
        log.info("=" * 60)
        for mode, r in sorted(all_results.items(), key=lambda x: -x[1]["qwen_score"]):
            log.info("  %-14s  yes_prob=%.4f  total=%.4f", mode, r["qwen_score"], r["qwen_loss"])
        log.info("Outputs: %s", output_dir)

    finally:
        if wandb_run is not None:
            wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--src-video", required=True)
    p.add_argument("--edit-prompt", required=True)
    p.add_argument("--output-dir", required=True)

    # Mode — extends the standard modes with video-delta variants
    p.add_argument(
        "--opt-mode", default="both_vd",
        help=(
            "Comma-separated optimisation modes.  Standard: text, audio, both.  "
            "With video delta: vd (video delta only), both_vd (text+audio+vd), "
            "audio_vd (audio+vd), text_vd (text+vd), all (= both_vd)."
        ),
    )

    # Video delta
    p.add_argument("--video-delta-reg-weight", type=float, default=0.1,
                   help="Base L2 reg weight for video_delta (decays from 5× via cosine).")
    p.add_argument("--video-delta-reg-max", type=float, default=None,
                   help="Max reg weight at start of cosine warm-in (default: 5× base).")
    p.add_argument("--vd-low-freq-weight", type=float, default=3.0,
                   help="Extra penalty on low-frequency video_delta components.")

    # Qwen2.5-VL
    p.add_argument("--qwen-model", default=DEFAULT_QWEN_ROOT)
    p.add_argument("--qwen-max-frames", type=int, default=8)
    p.add_argument("--qwen-img-size", type=int, default=QWEN_IMG_SIZE)
    p.add_argument("--qwen-sample-mode", default="linspace",
                   choices=["linspace", "normal", "contiguous", "contiguous_random"])
    p.add_argument("--qwen-contiguous-start-frame", type=int, default=0)
    p.add_argument("--qwen-gradient-rubric", default="motion",
                   choices=["motion", "entities", "overall", "full"])
    p.add_argument("--qwen-grad-accum-steps", type=int, default=1)
    p.add_argument("--qwen-motion-question", default=DEFAULT_QWEN_MOTION_QUESTION)

    # Regularisation
    p.add_argument("--latent-reg-weight", type=float, default=0.01)
    p.add_argument("--text-reg-weight", type=float, default=0.001)
    p.add_argument("--reg-schedule", default="constant",
                   choices=["constant", "linear_warmup", "cosine_increase"])
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--temporal-weight", type=float, default=0.0)
    p.add_argument("--lpips-backbone", default="alex", choices=["alex", "vgg"])
    p.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"])

    # Optimisation
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--audio-opt-last-steps", type=int, default=6)
    p.add_argument("--visualize-every-iters", type=int, default=10)
    p.add_argument("--early-stopping", type=int, default=0)
    p.add_argument("--max-eval-frames", type=int, default=33)
    p.add_argument("--frame-stride", type=int, default=1)

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
    p.add_argument("--save-final-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--static-prompt", default="")
    p.add_argument("--clip-similarity-diag", action=argparse.BooleanOptionalAction, default=False)

    # Guidance
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--audio-cfg-scale", type=float, default=None)
    p.add_argument("--a2v-scale", type=float, default=None)
    p.add_argument("--low-memory-guidance", action=argparse.BooleanOptionalAction, default=True)

    # Quantisation + checkpointing
    p.add_argument("--quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--retake-quantization", default=None, choices=["fp8-cast", "fp8-scaled-mm"])
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ltx-noise-opt"))

    # Paths
    p.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    p.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    p.add_argument("--lora", dest="loras", nargs="+", metavar=("PATH", "STRENGTH"),
                   action="append", default=[])

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.qwen_max_frames % 2 != 0:
        args.qwen_max_frames += 1
    if args.qwen_img_size % 28 != 0:
        args.qwen_img_size = (args.qwen_img_size // 28 + 1) * 28
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    if args.video_delta_reg_max is None:
        args.video_delta_reg_max = args.video_delta_reg_weight * 5.0

    args.ti2v_num_inference_steps = args.num_inference_steps
    args.clip_max_frames = args.qwen_max_frames

    run(args)


if __name__ == "__main__":
    main()
