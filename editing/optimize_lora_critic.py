#!/usr/bin/env python3
"""LoRA capacity-control ablation (rebuttal).

Trains a LoRA adapter on the FROZEN LTX-2 DiT using the SAME Qwen-VL critic,
the SAME losses, iterations, LR, and schedules as our method — but with NO
learnable text or audio conditioning latents. The only trainable parameters are
the injected LoRA weights (extra free capacity bolted onto the diffusion model).

Purpose: answer the reviewers' "is the gain just added capacity?" concern. If a
capacity-matched (indeed, far larger) LoRA optimized identically cannot
reproduce the motion edit that tuning the audio-conditioning latent achieves,
then the audio pathway is providing structured control that raw free parameters
do not.

This entry point is ADDITIVE: it reuses the parser, setup, render, and loss code
from optimize_qwen_vl.py / motion_opt without modifying any of it. Gradients
reach the LoRA params through the same last-`audio_opt_last_steps` differentiable
denoising window the audio/text optimization already uses.

Usage (same config flow as optimize_qwen_vl.py, plus --lora-* flags):
    python editing/optimize_lora_critic.py \
        --src-video ... --edit-prompt ... --output-dir results/rebuttal/<s>/lora_r64 \
        --lora-rank 64 --lora-alpha 64
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
from motion_opt.lora_critic import inject_lora_into_pipeline, lora_state_dict, save_lora, set_lora_state
from motion_opt.models import build_retake_pipeline, resolve_quantization_policy
from motion_opt.multimodal_loop import (
    render_baseline_video,
    render_final_video,
    render_with_injected_latents,
)
from motion_opt.multimodal_loop_qwen import pre_encode_base_contexts
from motion_opt.qwen_loss import build_qwen_model, build_qwen_rubric_inputs, compute_qwen_video_loss
from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video

from ltx_pipelines.utils.constants import detect_params

log = logging.getLogger(__name__)

MODE = "lora"


def _rubric_overrides(args):
    """Match the main loop: single-rubric runs optimize only that question."""
    gr = getattr(args, "qwen_gradient_rubric", "motion")
    if gr == "full":
        return None
    return {"motion": 0.0, "entities": 0.0, "overall": 0.0, gr: 1.0}


def gradient_optimize_lora(
    *,
    args,
    output_dir: Path,
    pipeline,
    lora_params,
    peft_transformer,
    base_pos_context,
    base_neg_context,
    base_audio_latent,
    cached_video_latent,
    retake_input_video: str,
    retake_kwargs: dict,
    qwen_model,
    cached_qwen_inputs: dict,
    yes_token_id: int,
    no_token_id: int,
) -> dict:
    """Optimize ONLY the LoRA params with the Qwen motion critic — the SAME main
    loss our method uses, and nothing else. No LPIPS / temporal / L2 reg: this is a
    pure test of whether extra free parameters + the motion critic can produce the
    edit. Text and audio conditioning are frozen (base contexts + base source
    audio latent)."""
    optimizer = torch.optim.Adam(lora_params, lr=args.lr)
    scheduler = None
    if getattr(args, "lr_schedule", "constant") == "cosine" and args.iterations > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations, eta_min=args.lr * 0.01
        )

    rubric_overrides = _rubric_overrides(args)
    qwen_grad_accum = max(1, int(getattr(args, "qwen_grad_accum_steps", 1)))
    base_sample_mode = getattr(args, "qwen_sample_mode", "linspace")

    best = {
        "qwen_loss": float("inf"), "qwen_score": float("-inf"),
        "best_iter": 0, "lora_state": None,
        # render_final_video reads these; None -> base text/audio (LoRA does the edit)
        "delta_v": None, "audio_latent": None,
    }

    csv_path = output_dir / f"optimization_log_qwen_{MODE}.csv"
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["iter", "qwen_nll", "qwen_yes_prob", "grad_norm", "is_best"])

    try:
        for it in range(1, args.iterations + 1):
            optimizer.zero_grad(set_to_none=True)

            gen_frames = render_with_injected_latents(
                pipeline=pipeline,
                src_video=retake_input_video,
                injected_audio_latent=base_audio_latent,      # FROZEN source audio
                cached_video_latent=cached_video_latent,
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

            # Qwen motion critic loss (single pass or grad accumulation) — the ONLY
            # loss. No perceptual / preservation / L2 reg terms.
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

            if args.grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, max_norm=args.grad_clip).item())
            else:
                # Report the TRUE grad norm even without clipping, so grad_norm=0
                # unambiguously means no gradient reached the LoRA (e.g. a bad
                # audio_opt_last_steps window) rather than "clipping disabled".
                sq = sum(float(p.grad.detach().norm().item()) ** 2 for p in lora_params if p.grad is not None)
                grad_norm = sq ** 0.5
            if grad_norm == 0.0:
                log.warning("[lora] grad_norm is 0 — no gradient reached the LoRA. "
                            "Check that 0 < audio_opt_last_steps (%d) < denoising steps.",
                            args.audio_opt_last_steps)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            qwen_loss = float(qwen_loss_t.detach().item())
            qwen_score = math.exp(-qwen_loss)

            is_best = qwen_loss < best["qwen_loss"]
            if is_best:
                best.update(qwen_loss=qwen_loss, qwen_score=qwen_score, best_iter=it,
                            lora_state=lora_state_dict(peft_transformer))

            writer.writerow([it, qwen_loss, qwen_score, grad_norm, int(is_best)])
            csv_file.flush()
            no_improve = it - best["best_iter"]
            log.info("[lora] iter %3d/%d  qwen_nll=%.4f  yes_prob=%.4f  grad_norm=%.3f  "
                     "no_improve=%d%s", it, args.iterations, qwen_loss, qwen_score,
                     grad_norm, no_improve, "  ★" if is_best else "")

            del gen_frames, qwen_loss_t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            early = getattr(args, "early_stopping", 0)
            if early > 0 and no_improve >= early:
                log.info("[lora] Early stopping at iter %d (best iter %d).", it, best["best_iter"])
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

    # ---- Shape / quantization / input video (identical to optimize_qwen_vl) ----
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

    # Final-render kwargs (full guidance), used for baseline + final optimized video.
    final_retake_kwargs = dict(retake_kwargs)
    if args.final_retake_num_inference_steps is not None:
        final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps
    final_vg, final_ag = build_guiders_for_mode(args=args, params=params, use_low_memory_guidance=False)
    final_retake_kwargs["video_guider_params"] = final_vg
    final_retake_kwargs["audio_guider_params"] = final_ag

    # ---- Baseline render BEFORE injecting LoRA (identical unedited reference) ----
    if args.save_final_videos:
        log.info("Rendering baseline video (no LoRA)...")
        render_baseline_video(
            pipeline=pipeline, src_video=str(retake_input_video),
            cached_video_latent=cached_video_latent, base_audio_latent=base_audio_latent,
            base_pos_context=base_pos_context, base_neg_context=base_neg_context,
            retake_kwargs=final_retake_kwargs, output_path=output_dir / "baseline_video.mp4",
            num_frames=num_frames, frame_rate=frame_rate, audio_sr=waveform_sr,
        )

    # ---- Inject trainable LoRA and optimize it with the Qwen critic ----
    mode_dir = output_dir / f"mode_{MODE}"
    mode_dir.mkdir(parents=True, exist_ok=True)

    peft_transformer, lora_params, restore_transformer = inject_lora_into_pipeline(
        pipeline,
        rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()] or None,
    )
    try:
        log.info("=" * 60)
        log.info("Starting LoRA capacity-control optimization (rank=%d, alpha=%d)", args.lora_rank, args.lora_alpha)
        log.info("=" * 60)
        best = gradient_optimize_lora(
            args=args, output_dir=mode_dir, pipeline=pipeline,
            lora_params=lora_params, peft_transformer=peft_transformer,
            base_pos_context=base_pos_context, base_neg_context=base_neg_context,
            base_audio_latent=base_audio_latent, cached_video_latent=cached_video_latent,
            retake_input_video=str(retake_input_video), retake_kwargs=retake_kwargs,
            qwen_model=qwen_model, cached_qwen_inputs=cached_qwen_inputs,
            yes_token_id=yes_token_id, no_token_id=no_token_id,
        )

        # Restore best LoRA weights, save, and render the final optimized video.
        set_lora_state(peft_transformer, best["lora_state"])
        save_lora(peft_transformer, mode_dir / f"best_lora_{MODE}.pt")
        torch.save({"mode": MODE, "qwen_loss": best["qwen_loss"], "qwen_score": best["qwen_score"],
                    "best_iter": best["best_iter"], "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha},
                   mode_dir / f"best_params_{MODE}.pt")
        log.info("[lora] Best Qwen yes_prob: %.4f (total loss: %.4f) at iter %d",
                 best["qwen_score"], best["qwen_loss"], best["best_iter"])

        if args.save_final_videos:
            log.info("[lora] Rendering optimized video (LoRA active, base text+audio)...")
            render_final_video(
                mode=MODE, best=best, pipeline=pipeline, src_video=str(retake_input_video),
                cached_video_latent=cached_video_latent, base_audio_latent=base_audio_latent,
                base_pos_context=base_pos_context, base_neg_context=base_neg_context,
                retake_kwargs=final_retake_kwargs, output_dir=mode_dir,
                num_frames=num_frames, frame_rate=frame_rate, audio_sr=waveform_sr,
                audio_opt_last_steps=args.audio_opt_last_steps, skip_baseline=True,
            )
    finally:
        restore_transformer()

    log.info("Outputs saved to: %s", output_dir)


def main() -> None:
    parser = build_parser()
    g = parser.add_argument_group("LoRA capacity-control ablation")
    g.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (capacity). Larger = more free params.")
    g.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha scaling.")
    g.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    g.add_argument("--lora-targets", default="to_q,to_k,to_v,to_out.0",
                   help="Comma-separated target module suffixes for LoRA.")
    args = parser.parse_args()

    if args.qwen_max_frames % 2 != 0:
        args.qwen_max_frames += 1
    if args.qwen_img_size % 28 != 0:
        args.qwen_img_size = (args.qwen_img_size // 28 + 1) * 28
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    args.ti2v_num_inference_steps = args.num_inference_steps
    args.clip_max_frames = args.qwen_max_frames
    args.opt_mode = MODE  # informational; this entry point always optimizes LoRA
    # The LoRA control must NEVER enhance the prompt — the final render (mode=lora
    # does not inject the base text context) would otherwise rewrite it. Force off.
    args.enhance_prompt = False

    run(args)


if __name__ == "__main__":
    main()
