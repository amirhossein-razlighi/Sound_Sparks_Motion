#!/usr/bin/env python3
"""Video-latent residual ablation (rebuttal).

Instead of optimizing the audio and/or text conditioning, optimize a learnable
residual delta_z directly in the video VAE latent domain:

    z_vid_used = z_vid_source + delta_z          (delta_z init = 0)

delta_z is updated by backprop from the Qwen-VL motion critic (the SAME loss as
ours), with text and audio conditioning FROZEN. This probes a different control
space: tuning the video latent itself — the most direct handle on the output —
versus tuning the audio-conditioning pathway.

This is structurally identical to the audio-latent method (an external learnable
tensor added to an injected latent), so it needs none of the LoRA-specific
machinery: the video latent is a grad-carrying INPUT to the frozen DiT, so
gradients reach delta_z through the existing last-`audio_opt_last_steps`
differentiable window, gradient checkpointing works, and the pipeline's
`transformer.requires_grad_(False)` never touches delta_z (it lives outside the
transformer).

Additive entry point: reuses the parser, setup, render, and loss code from
optimize_qwen_vl.py / motion_opt without modifying any of it.

Usage (same config flow as optimize_qwen_vl.py, plus --zvid-* flags):
    python editing/optimize_zvid_residual.py \
        --src-video ... --edit-prompt ... --output-dir results/rebuttal/<s>/zvid \
        [--zvid-reg-weight 0.0]
"""
from __future__ import annotations

import csv
import gc
import logging
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))           # optimize_qwen_vl
sys.path.insert(0, str(Path(__file__).parent / "src"))   # motion_opt

from optimize_qwen_vl import build_parser, _write_run_config  # reuse, no main changes

from motion_opt.core import (
    _parse_loras,
    build_cached_source_latents,
    build_guiders_for_mode,
    compute_target_shape,
)
from motion_opt.models import build_retake_pipeline, resolve_quantization_policy
from motion_opt.multimodal_loop import (
    render_baseline_video,
    render_final_video,
    render_with_injected_latents,
)
from motion_opt.multimodal_loop_qwen import pre_encode_base_contexts
from motion_opt.perceptual_loss import adaptive_reg_weight
from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss
from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video

from ltx_pipelines.utils.constants import detect_params

log = logging.getLogger(__name__)

MODE = "zvid"


def _rubric_overrides(args):
    gr = getattr(args, "qwen_gradient_rubric", "motion")
    if gr == "full":
        return None
    return {"motion": 0.0, "entities": 0.0, "overall": 0.0, gr: 1.0}


def gradient_optimize_zvid(
    *,
    args,
    output_dir: Path,
    pipeline,
    delta_z: torch.nn.Parameter,
    base_video_latent_fp32: torch.Tensor,
    video_dtype: torch.dtype,
    base_pos_context,
    base_neg_context,
    base_audio_latent,
    retake_input_video: str,
    retake_kwargs: dict,
    qwen_model,
    cached_qwen_inputs: dict,
    yes_token_id: int,
    no_token_id: int,
) -> dict:
    """Optimize delta_z (a residual on the source video latent) with the Qwen
    motion critic. Text + audio conditioning are frozen."""
    optimizer = torch.optim.Adam([delta_z], lr=args.lr)
    scheduler = None
    if getattr(args, "lr_schedule", "constant") == "cosine" and args.iterations > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations, eta_min=args.lr * 0.01
        )

    rubric_overrides = _rubric_overrides(args)
    qwen_grad_accum = max(1, int(getattr(args, "qwen_grad_accum_steps", 1)))
    base_sample_mode = getattr(args, "qwen_sample_mode", "linspace")
    reg_weight = float(getattr(args, "zvid_reg_weight", 0.0) or 0.0)
    reg_schedule = getattr(args, "reg_schedule", "constant")

    best = {
        "qwen_loss": float("inf"), "qwen_score": float("-inf"), "best_iter": 0,
        "delta_z": delta_z.detach().clone(),
        # render_final_video reads these; None -> base audio/text (the residual does the edit)
        "delta_v": None, "audio_latent": None,
    }

    csv_path = output_dir / f"optimization_log_qwen_{MODE}.csv"
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["iter", "qwen_nll", "qwen_yes_prob", "zvid_reg", "total_loss", "grad_norm", "is_best"])

    try:
        for it in range(1, args.iterations + 1):
            optimizer.zero_grad(set_to_none=True)

            # z_vid_used = source video latent + residual (compute in fp32, inject in model dtype)
            z_in = (base_video_latent_fp32 + delta_z).to(video_dtype)

            gen_frames = render_with_injected_latents(
                pipeline=pipeline,
                src_video=retake_input_video,
                injected_audio_latent=base_audio_latent,      # FROZEN source audio
                cached_video_latent=z_in,                     # source + learnable residual
                retake_kwargs=retake_kwargs,
                max_frames=args.max_eval_frames,
                frame_stride=args.frame_stride,
                resize_to=None,
                audio_opt_last_steps=args.audio_opt_last_steps,
                eval_sample_start=0,
                pos_context=base_pos_context,                 # FROZEN text
                neg_context=base_neg_context,
                inject_text_context=True,
            )
            if gen_frames.shape[0] < 1:
                raise RuntimeError("Generated video has no frames.")

            # Qwen motion critic loss (single pass or grad accumulation).
            if qwen_grad_accum == 1:
                qwen_loss_t, _ = compute_qwen_video_loss(
                    frames_chw=gen_frames, qwen_model=qwen_model, cached_inputs=cached_qwen_inputs,
                    yes_token_id=yes_token_id, no_token_id=no_token_id,
                    max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                    backward=True, return_details=True, sample_mode=base_sample_mode,
                    contiguous_start_frame=getattr(args, "qwen_contiguous_start_frame", 0),
                    rubric_weight_overrides=rubric_overrides,
                )
            else:
                gf_det = gen_frames.detach().requires_grad_(True)
                accum = torch.zeros((), device=gen_frames.device)
                for accum_i in range(qwen_grad_accum):
                    pass_mode = base_sample_mode if accum_i == 0 else "contiguous_random"
                    loss_i = compute_qwen_video_loss(
                        frames_chw=gf_det, qwen_model=qwen_model, cached_inputs=cached_qwen_inputs,
                        yes_token_id=yes_token_id, no_token_id=no_token_id,
                        max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                        backward=False, return_details=False, sample_mode=pass_mode,
                        contiguous_start_frame=getattr(args, "qwen_contiguous_start_frame", 0),
                        rubric_weight_overrides=rubric_overrides,
                    )
                    (loss_i / qwen_grad_accum).backward()
                    accum = accum + loss_i.detach()
                qwen_loss_t = accum / qwen_grad_accum
                if gf_det.grad is not None and gen_frames.grad_fn is not None:
                    gen_frames.backward(gradient=gf_det.grad)

            # Optional L2 reg on the residual (anchor delta_z toward 0 == video latent
            # toward source). Default 0; set --zvid-reg-weight to match the audio
            # method's latent_reg_weight for an anchored, apples-to-apples comparison.
            zvid_reg = 0.0
            if reg_weight > 0:
                w = adaptive_reg_weight(reg_weight, it, args.iterations, schedule=reg_schedule)
                if w > 0:
                    reg_t = w * torch.mean(delta_z ** 2)
                    reg_t.backward()
                    zvid_reg = float(reg_t.detach().item())

            if args.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_([delta_z], max_norm=args.grad_clip).item())
            else:
                grad_norm = float(delta_z.grad.detach().norm().item()) if delta_z.grad is not None else 0.0
            if grad_norm == 0.0:
                log.warning("[zvid] grad_norm is 0 — no gradient reached delta_z. "
                            "Check 0 < audio_opt_last_steps (%d) < denoising steps.", args.audio_opt_last_steps)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            qwen_loss = float(qwen_loss_t.detach().item())
            qwen_score = math.exp(-qwen_loss)
            total = qwen_loss + zvid_reg

            is_best = total < best["qwen_loss"]
            if is_best:
                best.update(qwen_loss=total, qwen_score=qwen_score, best_iter=it,
                            delta_z=delta_z.detach().clone())

            writer.writerow([it, qwen_loss, qwen_score, zvid_reg, total, grad_norm, int(is_best)])
            csv_file.flush()
            no_improve = it - best["best_iter"]
            log.info("[zvid] iter %3d/%d  qwen_nll=%.4f  yes_prob=%.4f  zvid_reg=%.4f  total=%.4f  "
                     "grad_norm=%.3f  no_improve=%d%s", it, args.iterations, qwen_loss, qwen_score,
                     zvid_reg, total, grad_norm, no_improve, "  ★" if is_best else "")

            del gen_frames, qwen_loss_t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            early = getattr(args, "early_stopping", 0)
            if early > 0 and no_improve >= early:
                log.info("[zvid] Early stopping at iter %d (best iter %d).", it, best["best_iter"])
                break
    finally:
        csv_file.close()

    return best


def run(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(args, output_dir)

    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video, args.height, args.width, args.num_frames, args.frame_rate
    )
    duration = num_frames / frame_rate
    log.info("Video shape: %dx%d, %d frames @ %.1f fps (%.2fs)", width, height, num_frames, frame_rate, duration)

    retake_quant = resolve_quantization_policy(
        args.retake_quantization if args.retake_quantization is not None else args.quantization
    )
    retake_input_video = prepare_retake_input_video(
        args=args, is_main=True, output_dir=output_dir,
        height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
    )

    params = detect_params(args.checkpoint_path)
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args, params=params, use_low_memory_guidance=args.low_memory_guidance
    )

    log.info("Loading RetakePipeline (checkpoint: %s)...", args.checkpoint_path)
    pipeline = build_retake_pipeline(
        checkpoint_path=args.checkpoint_path, gemma_root=args.gemma_root,
        loras=_parse_loras(args.loras), device=device,
        quant_policy=retake_quant, gradient_checkpointing=args.gradient_checkpointing,
    )

    cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
        pipeline=pipeline, src_video=str(retake_input_video),
        height=height, width=width, num_frames=num_frames, audio_sr=args.audio_sr, device=device,
    )
    base_pos_context, base_neg_context = pre_encode_base_contexts(
        pipeline=pipeline, pos_prompt=args.edit_prompt, neg_prompt=args.negative_prompt, device=device
    )

    log.info("Loading Qwen2.5-VL model (%s)...", args.qwen_model)
    qwen_model, qwen_processor = build_qwen_model(
        args.qwen_model, device=device, gradient_checkpointing=args.gradient_checkpointing
    )
    qwen_num_frames = args.qwen_max_frames + (args.qwen_max_frames % 2)
    cached_qwen_inputs, yes_token_id, no_token_id = build_qwen_rubric_inputs(
        processor=qwen_processor, edit_prompt=args.edit_prompt, num_frames=qwen_num_frames,
        img_size=args.qwen_img_size, device=device, motion_question=args.qwen_motion_question,
        gradient_rubric=args.qwen_gradient_rubric,
    )

    retake_kwargs = build_retake_kwargs(
        args=args, frame_rate=frame_rate, duration=duration,
        video_guider_params=video_guider_params, audio_guider_params=audio_guider_params,
    )

    final_retake_kwargs = dict(retake_kwargs)
    if args.final_retake_num_inference_steps is not None:
        final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps
    final_vg, final_ag = build_guiders_for_mode(args=args, params=params, use_low_memory_guidance=False)
    final_retake_kwargs["video_guider_params"] = final_vg
    final_retake_kwargs["audio_guider_params"] = final_ag

    # ---- Baseline render (delta_z = 0): pure base-LTX retake, identical to other variants ----
    if args.save_final_videos:
        log.info("Rendering baseline video (delta_z = 0)...")
        render_baseline_video(
            pipeline=pipeline, src_video=str(retake_input_video),
            cached_video_latent=cached_video_latent, base_audio_latent=base_audio_latent,
            base_pos_context=base_pos_context, base_neg_context=base_neg_context,
            retake_kwargs=final_retake_kwargs, output_path=output_dir / "baseline_video.mp4",
            num_frames=num_frames, frame_rate=frame_rate, audio_sr=waveform_sr,
        )

    # ---- Optimize the video-latent residual ----
    mode_dir = output_dir / f"mode_{MODE}"
    mode_dir.mkdir(parents=True, exist_ok=True)

    base_video_latent_fp32 = cached_video_latent.float().detach()
    delta_z = torch.nn.Parameter(torch.zeros_like(base_video_latent_fp32))
    log.info("[zvid] Residual delta_z shape: %s, %.1fK params", tuple(delta_z.shape), delta_z.numel() / 1e3)

    log.info("=" * 60)
    log.info("Starting video-latent residual optimization (delta_z on z_vid)")
    log.info("=" * 60)
    best = gradient_optimize_zvid(
        args=args, output_dir=mode_dir, pipeline=pipeline, delta_z=delta_z,
        base_video_latent_fp32=base_video_latent_fp32, video_dtype=cached_video_latent.dtype,
        base_pos_context=base_pos_context, base_neg_context=base_neg_context,
        base_audio_latent=base_audio_latent, retake_input_video=str(retake_input_video),
        retake_kwargs=retake_kwargs, qwen_model=qwen_model, cached_qwen_inputs=cached_qwen_inputs,
        yes_token_id=yes_token_id, no_token_id=no_token_id,
    )

    torch.save(best["delta_z"].cpu(), mode_dir / f"best_delta_z_{MODE}.pt")
    torch.save({"mode": MODE, "qwen_loss": best["qwen_loss"], "qwen_score": best["qwen_score"],
                "best_iter": best["best_iter"]}, mode_dir / f"best_params_{MODE}.pt")
    log.info("[zvid] Best Qwen yes_prob: %.4f (total loss: %.4f) at iter %d",
             best["qwen_score"], best["qwen_loss"], best["best_iter"])

    # ---- Final render: source video latent + best residual, base text+audio ----
    if args.save_final_videos:
        z_best = (base_video_latent_fp32 + best["delta_z"].to(base_video_latent_fp32.device)).to(cached_video_latent.dtype)
        log.info("[zvid] Rendering optimized video (z_vid + best residual, base text+audio)...")
        render_final_video(
            mode=MODE, best=best, pipeline=pipeline, src_video=str(retake_input_video),
            cached_video_latent=z_best, base_audio_latent=base_audio_latent,
            base_pos_context=base_pos_context, base_neg_context=base_neg_context,
            retake_kwargs=final_retake_kwargs, output_dir=mode_dir,
            num_frames=num_frames, frame_rate=frame_rate, audio_sr=waveform_sr,
            audio_opt_last_steps=args.audio_opt_last_steps, skip_baseline=True,
        )

    log.info("Outputs saved to: %s", output_dir)


def main() -> None:
    parser = build_parser()
    g = parser.add_argument_group("video-latent residual ablation")
    g.add_argument("--zvid-reg-weight", type=float, default=0.0,
                   help="L2 reg on the residual delta_z (anchor video latent to source). "
                        "Default 0; set to e.g. --latent-reg-weight's value for a fair anchored run.")
    args = parser.parse_args()

    if args.qwen_max_frames % 2 != 0:
        args.qwen_max_frames += 1
    if args.qwen_img_size % 28 != 0:
        args.qwen_img_size = (args.qwen_img_size // 28 + 1) * 28
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    args.ti2v_num_inference_steps = args.num_inference_steps
    args.clip_max_frames = args.qwen_max_frames
    args.opt_mode = MODE
    # Never enhance the prompt (final render of this mode doesn't inject base text).
    args.enhance_prompt = False

    run(args)


if __name__ == "__main__":
    main()
