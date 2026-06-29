"""Main gradient optimization loop for audio-text conditioned video editing.

Supervises the optimization with Qwen2.5-VL as the alignment signal:
  loss = -log P("yes" | video_frames, "Does this video show: {edit_prompt}?")

Optimization modes
------------------
text  : learn a soft delta on the Gemma text embedding (video_encoding).
audio : learn a perturbation of the audio latent fed to the Retake pipeline.
both  : jointly optimize text delta + audio latent.

Rendering infrastructure (differentiable Retake forward pass, text pre-encoding,
final video export) lives in multimodal_loop.py and is imported from there.

Optional addons
---------------
  - L2 regularization on the audio latent and/or text delta
  - LPIPS / temporal consistency perceptual regularizer
  - Cosine LR annealing, early stopping
  - TensorBoard + W&B logging, per-iteration video/audio previews
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
from .perceptual_loss import (
    adaptive_reg_weight,
    compute_perceptual_quality_loss,
)
from .qwen_loss import compute_qwen_video_loss

log = logging.getLogger(__name__)


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
    cached_src_frames: torch.Tensor | None = None,
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
    # Anchor for the L2 audio regularizer. `None` disables the audio reg term.
    # Defaults below reproduce the original behaviour (init from source latent,
    # regularize toward the source latent).
    audio_reg_anchor: torch.Tensor | None = None

    if optimize_text:
        delta_v = torch.nn.Parameter(
            torch.zeros_like(base_pos_context.video_encoding.float())
        )
        params.append(delta_v)
        log.info("[%s] Text delta shape: %s, %.1fK params", mode, tuple(delta_v.shape), delta_v.numel() / 1e3)

    if optimize_audio:
        # --audio-init selects how the optimized audio latent is initialised.
        #   source : encoded source-audio latent (default; original behaviour, ours)
        #   zero   : zeros — capacity-matched control with no audio prior
        #   random : Gaussian noise scale-matched to the source latent's std
        # --audio-reg-anchor selects the L2 reg anchor (source / init / none).
        audio_init = getattr(args, "audio_init", "source")
        if audio_init == "source":
            init_tensor = base_audio_latent_fp32.clone()
        elif audio_init == "zero":
            init_tensor = torch.zeros_like(base_audio_latent_fp32)
        elif audio_init == "random":
            # Scale-matched, zero-mean Gaussian with same shape as the source
            # latent. Seeded by --audio-init-seed (falls back to --seed) so the
            # draw is reproducible and can be varied independently of the
            # diffusion seed.
            init_seed = getattr(args, "audio_init_seed", None)
            if init_seed is None:
                init_seed = getattr(args, "seed", 42)
            std = base_audio_latent_fp32.std().clamp_min(1e-6)
            gen = torch.Generator(device=base_audio_latent_fp32.device)
            gen.manual_seed(int(init_seed))
            noise = torch.randn(
                base_audio_latent_fp32.shape,
                generator=gen,
                device=base_audio_latent_fp32.device,
                dtype=base_audio_latent_fp32.dtype,
            )
            init_tensor = noise * std
        else:
            raise ValueError(f"Unknown audio_init: {audio_init!r}. Expected source/zero/random.")

        audio_latent = torch.nn.Parameter(init_tensor)
        params.append(audio_latent)

        reg_anchor_mode = getattr(args, "audio_reg_anchor", "source")
        if reg_anchor_mode == "source":
            audio_reg_anchor = base_audio_latent_fp32
        elif reg_anchor_mode == "init":
            audio_reg_anchor = init_tensor.detach().clone()
        elif reg_anchor_mode == "none":
            audio_reg_anchor = None
        else:
            raise ValueError(f"Unknown audio_reg_anchor: {reg_anchor_mode!r}. Expected source/init/none.")

        log.info(
            "[%s] Audio latent shape: %s, %.1fK params  (init=%s, reg_anchor=%s)",
            mode, tuple(audio_latent.shape), audio_latent.numel() / 1e3,
            audio_init, reg_anchor_mode,
        )

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ---- LR scheduler (cosine annealing to prevent late-stage adversarial drift) ----
    lr_schedule = getattr(args, "lr_schedule", "constant")
    scheduler = None
    if lr_schedule == "cosine" and args.iterations > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations, eta_min=args.lr * 0.01,
        )
        log.info("[%s] Using cosine LR schedule: %.6f → %.6f", mode, args.lr, args.lr * 0.01)

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
    rubric_metric_names = [
        str(item["name"])
        for item in cached_qwen_inputs.get("rubric_items", [])
    ]
    if is_main:
        csv_header = [
            "iter", "qwen_nll", "qwen_yes_prob",
            "audio_reg", "text_reg", "perceptual_loss",
            "lpips_raw", "temporal_raw",
            "total_loss", "grad_norm", "is_best",
        ]
        for name in rubric_metric_names:
            csv_header.extend([f"{name}_nll", f"{name}_yes_prob"])
        csv_writer.writerow(csv_header)

    # ---- W&B Table (per-iteration metrics — sortable/filterable in W&B UI) ----
    wandb_table = None
    if wandb_run is not None and is_main:
        try:
            import wandb as _wandb
            wandb_table = _wandb.Table(
                columns=(
                    ["iter", "yes_prob", "qwen_nll", "audio_reg",
                     "text_reg", "perceptual_loss", "lpips_raw", "temporal_raw",
                     "total_loss", "grad_norm", "is_best"]
                    + [
                        col
                        for name in rubric_metric_names
                        for col in (f"{name}_nll", f"{name}_yes_prob")
                    ]
                )
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

            # Perceptual quality loss — computed BEFORE Qwen loss because
            # Qwen's backward=True frees the graph.  Perceptual loss must
            # use retain_graph=True so Qwen can still backprop afterward.
            perceptual_loss_t = torch.tensor(0.0, device=gen_frames.device)
            perceptual_details: dict[str, float] = {}
            lpips_weight = getattr(args, "lpips_weight", 0.0)
            temporal_weight = getattr(args, "temporal_weight", 0.0)
            need_perceptual = cached_src_frames is not None and (lpips_weight > 0 or temporal_weight > 0)
            if need_perceptual:
                perceptual_loss_t, perceptual_details = compute_perceptual_quality_loss(
                    gen_frames=gen_frames,
                    src_frames=cached_src_frames,
                    lpips_weight=lpips_weight,
                    temporal_weight=temporal_weight,
                    backbone=getattr(args, "lpips_backbone", "alex"),
                    max_lpips_frames=min(16, gen_frames.shape[0]),
                    max_temporal_pairs=min(12, gen_frames.shape[0] - 1),
                    backward=False,  # we backward manually with retain_graph
                )
                if perceptual_loss_t.requires_grad:
                    perceptual_loss_t.backward(retain_graph=True)
                    perceptual_loss_t = perceptual_loss_t.detach()

            # Qwen2.5-VL alignment loss
            # With qwen_grad_accum_steps > 1: run N Qwen passes per render using
            # different random frame windows (contiguous_random). Gradient is
            # accumulated on a detached copy of gen_frames so each Qwen graph is
            # freed after its own backward — no N× memory overhead. The averaged
            # gradient is then chained through the rendering graph once at the end.
            rubric_weight_overrides = None
            qwen_gradient_rubric = getattr(args, "qwen_gradient_rubric", "motion")
            if qwen_gradient_rubric != "full":
                rubric_weight_overrides = {
                    "motion": 0.0,
                    "entities": 0.0,
                    "overall": 0.0,
                    str(qwen_gradient_rubric): 1.0,
                }

            qwen_grad_accum = max(1, int(getattr(args, "qwen_grad_accum_steps", 1)))

            if qwen_grad_accum == 1:
                # Original single-pass path — backward=True frees the graph.
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
                # Multi-pass gradient accumulation.
                # Detach gen_frames → each Qwen graph freed after its own backward,
                # not N copies held simultaneously. Averaged gradient then propagates
                # through the rendering graph (retained by perceptual backward, or
                # still intact if perceptual was skipped).
                gf_det = gen_frames.detach().requires_grad_(True)
                accum_loss = torch.zeros((), device=gen_frames.device)
                qwen_details = []
                base_sample_mode = getattr(args, "qwen_sample_mode", "linspace")
                for accum_i in range(qwen_grad_accum):
                    # Pass 0: user's chosen mode (e.g. linspace gives full-arc coverage).
                    # Passes 1+: contiguous_random for diversity — different window each
                    # time so the averaged gradient is not just N copies of the same estimate.
                    pass_sample_mode = base_sample_mode if accum_i == 0 else "contiguous_random"
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
                        sample_mode=pass_sample_mode,
                        contiguous_start_frame=getattr(args, "qwen_contiguous_start_frame", 0),
                        rubric_weight_overrides=rubric_weight_overrides,
                    )
                    if accum_i == 0:
                        loss_i, qwen_details = _result
                    else:
                        loss_i = _result  # plain tensor when return_details=False
                    # Scale by 1/N and backward: frees this Qwen graph,
                    # accumulates gradient in gf_det.grad.
                    (loss_i / qwen_grad_accum).backward()
                    accum_loss = accum_loss + loss_i.detach()

                qwen_loss_t = accum_loss / qwen_grad_accum

                # Chain the averaged Qwen gradient through the rendering graph.
                # gen_frames still has its grad_fn (rendering kept alive by
                # perceptual retain_graph=True, or untouched if no perceptual).
                if gf_det.grad is not None and gen_frames.grad_fn is not None:
                    gen_frames.backward(gradient=gf_det.grad)

            # Adaptive regularization (increase over time to prevent late-stage adversarial drift)
            reg_schedule = getattr(args, "reg_schedule", "constant")
            audio_reg_w = adaptive_reg_weight(
                args.latent_reg_weight, it, num_iters, schedule=reg_schedule,
            )
            text_reg_w = adaptive_reg_weight(
                args.text_reg_weight, it, num_iters, schedule=reg_schedule,
            )

            # Regularization
            audio_reg_t = torch.tensor(0.0, device=qwen_loss_t.device)
            if (
                optimize_audio
                and audio_latent is not None
                and audio_reg_w > 0
                and audio_reg_anchor is not None
            ):
                audio_reg_t = audio_reg_w * torch.mean(
                    (audio_latent - audio_reg_anchor) ** 2
                )

            text_reg_t = torch.tensor(0.0, device=qwen_loss_t.device)
            if optimize_text and delta_v is not None and text_reg_w > 0:
                text_reg_t = text_reg_w * torch.mean(delta_v ** 2)

            # perceptual_loss_t is detached (already backpropped); add for logging only
            total_t = qwen_loss_t + audio_reg_t + text_reg_t
            reg_t = audio_reg_t + text_reg_t
            if reg_t.requires_grad:
                reg_t.backward()

            grad_norm = 0.0
            if args.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip).item())
            elif params[0].grad is not None:
                grad_norm = float(sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            qwen_loss = float(qwen_loss_t.detach().item())
            qwen_score = math.exp(-qwen_loss)
            audio_reg = float(audio_reg_t.detach().item())
            text_reg = float(text_reg_t.detach().item())
            perceptual_loss = float(perceptual_loss_t.detach().item()) if torch.is_tensor(perceptual_loss_t) else 0.0
            total = float(total_t.detach().item()) + perceptual_loss  # include perceptual for logging/best selection

            best_min_loss_delta = max(float(getattr(args, "best_min_loss_delta", 0.0) or 0.0), 0.0)
            is_best = total < (best["qwen_loss"] - best_min_loss_delta)
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
                detail_by_name = {str(item["name"]): item for item in qwen_details}
                rubric_values = []
                for name in rubric_metric_names:
                    item = detail_by_name.get(name, {})
                    rubric_values.extend([
                        item.get("nll", ""),
                        item.get("yes_prob", ""),
                    ])
                csv_writer.writerow([
                    it, qwen_loss, qwen_score, audio_reg, text_reg,
                    perceptual_loss,
                    perceptual_details.get("lpips_raw", ""),
                    perceptual_details.get("temporal_raw", ""),
                    total, grad_norm, int(is_best),
                    *rubric_values,
                ])
                csv_file.flush()
                if wandb_table is not None:
                    wandb_table.add_data(
                        it, qwen_score, qwen_loss, audio_reg, text_reg,
                        perceptual_loss,
                        perceptual_details.get("lpips_raw", 0.0),
                        perceptual_details.get("temporal_raw", 0.0),
                        total, grad_norm, int(is_best),
                        *rubric_values,
                    )
                rubric_log = ""
                if qwen_details:
                    rubric_log = "  rubric=" + ", ".join(
                        f"{item['name']}:{float(item['yes_prob']):.4f}"
                        for item in qwen_details
                    )
                percep_log = ""
                if perceptual_details:
                    percep_log = f"  lpips={perceptual_details.get('lpips_raw', 0):.4f}  temporal={perceptual_details.get('temporal_raw', 0):.4f}"
                log.info(
                    "[%s] iter %3d/%d  qwen_nll=%.4f  yes_prob=%.4f  total=%.4f  grad_norm=%.3f  no_improve=%d%s%s%s",
                    mode, it, num_iters, qwen_loss, qwen_score, total, grad_norm,
                    iters_without_improvement, "  ★" if is_best else "", percep_log, rubric_log,
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
                    if perceptual_details:
                        tb_writer.add_scalar(f"{mode}/perceptual_total", perceptual_loss, it)
                        for pk, pv in perceptual_details.items():
                            tb_writer.add_scalar(f"{mode}/perceptual/{pk}", pv, it)
                    if is_best:
                        tb_writer.add_scalar(f"{mode}/best_qwen_yes_prob", qwen_score, it)
                    for item in qwen_details:
                        name = str(item["name"])
                        tb_writer.add_scalar(f"{mode}/rubric/{name}_nll", float(item["nll"]), it)
                        tb_writer.add_scalar(f"{mode}/rubric/{name}_yes_prob", float(item["yes_prob"]), it)

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
                    if perceptual_details:
                        wandb_payload[f"{mode}/perceptual_total"] = perceptual_loss
                        for pk, pv in perceptual_details.items():
                            wandb_payload[f"{mode}/perceptual/{pk}"] = pv
                    if is_best:
                        wandb_payload[f"{mode}/best_qwen_yes_prob"] = qwen_score
                    for item in qwen_details:
                        name = str(item["name"])
                        wandb_payload[f"{mode}/rubric/{name}_nll"] = float(item["nll"])
                        wandb_payload[f"{mode}/rubric/{name}_yes_prob"] = float(item["yes_prob"])
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
                    attn_cap_active = False
                    try:
                        if extract_attn_maps and wandb_run is not None:
                            attn_cap = LTXAttentionCapture(pipeline, block_fraction=0.5)
                            attn_cap.__enter__()
                            attn_cap_active = True

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
                            attn_cap_active = False

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
                                            f"media/video/{mode}/preview": wandb.Video(
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
                                gen_frames=gen_frames.detach(),
                                qwen_model=qwen_model,
                                cached_qwen_inputs=cached_qwen_inputs,
                                qwen_max_frames=args.qwen_max_frames,
                                qwen_img_size=args.qwen_img_size,
                                attn_cap=attn_cap,
                                cached_video_latent=cached_video_latent,
                                wandb_run=wandb_run,
                                save_dir=preview_dir / "attention",
                            )
                    finally:
                        if attn_cap is not None and attn_cap_active:
                            attn_cap.__exit__(None, None, None)
                        _clear_cuda_cache()

            early_stop_limit = getattr(args, "early_stopping", 0)
            if early_stop_limit > 0 and iters_without_improvement >= early_stop_limit:
                log.info(
                    "[%s] Early stopping: no improvement for %d consecutive iters (best at iter %d).",
                    mode, iters_without_improvement, best.get("best_iter", 0),
                )
                del gen_frames, qwen_loss_t, audio_reg_t, text_reg_t, total_t, reg_t, qwen_details
                _clear_cuda_cache()
                break

            del gen_frames, qwen_loss_t, audio_reg_t, text_reg_t, perceptual_loss_t, total_t, reg_t, qwen_details, perceptual_details
            _clear_cuda_cache()
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
    save_dir: Path | None = None,
) -> None:
    """Extract Qwen + LTX attention maps and log overlays to W&B.

    Called only at preview_every steps inside a no_grad context (gen_frames
    has already been detached from the compute graph by the time we get here
    since optimizer.step() was called before the preview render).

    Memory notes:
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
            tag=f"media/attention/{mode}/qwen_{label}",
            frames_np=frames_np,
            heatmap=attn_q_spatial,
            step=it,
            caption=f"Qwen scorer attention — {label} iter {it}",
            save_dir=save_dir,
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
                    tag=f"media/attention/{mode}/{tag_suffix}_{label}",
                    frames_np=frames_np,
                    heatmap=attn_2d,
                    step=it,
                    caption=f"{caption_suffix} — {label} iter {it}",
                    save_dir=save_dir,
                )
                log.info("[%s] LTX %s attention logged.", mode, key)
            else:
                log.warning("[%s] LTX %s attention was not captured at iter %d.", mode, key, it)


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
                        f"media/audio/{mode}/{label}": wandb.Audio(
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
