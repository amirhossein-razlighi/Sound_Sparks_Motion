#!/usr/bin/env python3
"""Benchmark timing script for Qwen-VL optimization.

Identical logic to optimize_qwen_vl.py / multimodal_loop_qwen.py but strips
every non-essential I/O path (W&B, TensorBoard, preview renders, CLIP
diagnostics, attention maps, audio saves, CSV writes, baseline render) so
measured wall-clock time reflects only model forward/backward passes and the
optimizer step.

Reports per-iteration time and total elapsed time via tqdm.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

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
from audio_latent_opt.multimodal_loop_qwen import (
    pre_encode_base_contexts,
    render_final_video,
    render_with_injected_latents,
)
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

# Silence everything except warnings — we don't want library info spam
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _benchmark_optimize_loop(
    *,
    mode: str,
    args,
    output_dir: Path,
    base_pos_context,
    base_neg_context,
    base_audio_latent: torch.Tensor,
    base_audio_latent_fp32: torch.Tensor,
    cached_video_latent: torch.Tensor,
    retake_input_video: str,
    pipeline,
    retake_kwargs: dict,
    qwen_model,
    cached_qwen_inputs: dict,
    yes_token_id: int,
    no_token_id: int,
    eval_sample_start: int,
    num_frames: int,
    frame_rate: float,
    audio_sr: int,
    cached_src_frames: torch.Tensor | None = None,
) -> dict:
    """Stripped optimization loop: same math, no logging/viz side-effects."""
    optimize_text = mode in ("text", "both")
    optimize_audio = mode in ("audio", "both")

    params: list[torch.nn.Parameter] = []
    delta_v: torch.nn.Parameter | None = None
    audio_latent: torch.nn.Parameter | None = None

    if optimize_text:
        delta_v = torch.nn.Parameter(
            torch.zeros_like(base_pos_context.video_encoding.float())
        )
        params.append(delta_v)

    if optimize_audio:
        audio_latent = torch.nn.Parameter(base_audio_latent_fp32.clone())
        params.append(audio_latent)

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
        "mode": mode,
        "best_iter": 0,
        "clip_loss": float("inf"),
        "clip_score": float("-inf"),
    }

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

    iter_times: list[float] = []

    pbar = tqdm(range(1, num_iters + 1), desc=f"[{mode}] optimizing", unit="iter", dynamic_ncols=True)
    for it in pbar:
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

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
                backward=False,
            )
            if perceptual_loss_t.requires_grad:
                perceptual_loss_t.backward(retain_graph=True)
                perceptual_loss_t = perceptual_loss_t.detach()

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
                    loss_i = _result
                (loss_i / qwen_grad_accum).backward()
                accum_loss = accum_loss + loss_i.detach()

            qwen_loss_t = accum_loss / qwen_grad_accum
            if gf_det.grad is not None and gen_frames.grad_fn is not None:
                gen_frames.backward(gradient=gf_det.grad)

        reg_schedule = getattr(args, "reg_schedule", "constant")
        audio_reg_w = adaptive_reg_weight(args.latent_reg_weight, it, num_iters, schedule=reg_schedule)
        text_reg_w = adaptive_reg_weight(args.text_reg_weight, it, num_iters, schedule=reg_schedule)

        audio_reg_t = torch.tensor(0.0, device=qwen_loss_t.device)
        if optimize_audio and audio_latent is not None and audio_reg_w > 0:
            audio_reg_t = audio_reg_w * torch.mean((audio_latent - base_audio_latent_fp32) ** 2)

        text_reg_t = torch.tensor(0.0, device=qwen_loss_t.device)
        if optimize_text and delta_v is not None and text_reg_w > 0:
            text_reg_t = text_reg_w * torch.mean(delta_v ** 2)

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
        perceptual_loss = float(perceptual_loss_t.detach().item()) if torch.is_tensor(perceptual_loss_t) else 0.0
        total = float(total_t.detach().item()) + perceptual_loss

        best_min_loss_delta = max(float(getattr(args, "best_min_loss_delta", 0.0) or 0.0), 0.0)
        is_best = total < (best["qwen_loss"] - best_min_loss_delta)
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

        iters_without_improvement = it - best.get("best_iter", 0)

        iter_t = time.perf_counter() - t0
        iter_times.append(iter_t)

        pbar.set_postfix(
            nll=f"{qwen_loss:.4f}",
            yes=f"{qwen_score:.4f}",
            total=f"{total:.4f}",
            best="★" if is_best else " ",
            no_imp=iters_without_improvement,
            t=f"{iter_t:.1f}s",
        )

        del gen_frames, qwen_loss_t, audio_reg_t, text_reg_t, perceptual_loss_t, total_t, reg_t, qwen_details, perceptual_details
        _clear_cuda_cache()

        early_stop_limit = getattr(args, "early_stopping", 0)
        if early_stop_limit > 0 and iters_without_improvement >= early_stop_limit:
            print(f"\n[{mode}] Early stop: no improvement for {iters_without_improvement} iters (best at iter {best.get('best_iter', 0)}).")
            break

    best["iter_times"] = iter_times
    return best


def benchmark_run(args) -> None:
    # Disable everything that is not the core optimization loop
    save_final_videos = args.save_final_videos  # remember original intent
    args.save_final_videos = False              # keep False during timed loop
    args.clip_similarity_diag = False
    args.visualize_every_iters = 0
    os.environ["WANDB_DISABLED"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # ---- Setup phase ----
    print(f"\n{'='*60}")
    print(f"  BENCHMARK — {args.opt_mode.upper()}")
    print(f"  Iterations: {args.iterations}  LR: {args.lr}  Steps: {args.num_inference_steps}")
    print(f"  Qwen frames: {args.qwen_max_frames}  img_size: {args.qwen_img_size}")
    print(f"  Quantization: {args.quantization}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("[setup] Computing target shape...")
    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video, args.height, args.width, args.num_frames, args.frame_rate,
    )
    print(f"[setup] Shape: {width}x{height}, {num_frames} frames @ {frame_rate:.1f} fps")

    retake_quant = resolve_quantization_policy(
        args.retake_quantization if args.retake_quantization is not None else args.quantization
    )

    print("[setup] Preparing input video...")
    retake_input_video = prepare_retake_input_video(
        args=args, is_main=True, output_dir=output_dir,
        height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
    )

    params_cfg = detect_params(args.checkpoint_path)
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args, params=params_cfg, use_low_memory_guidance=args.low_memory_guidance,
    )

    print("[setup] Loading RetakePipeline...")
    t0 = time.perf_counter()
    loras = _parse_loras(args.loras)
    pipeline = build_retake_pipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=loras,
        device=device,
        quant_policy=retake_quant,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(f"[setup] Pipeline loaded in {time.perf_counter() - t0:.1f}s")

    print("[setup] Encoding source video/audio...")
    t0 = time.perf_counter()
    cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
        pipeline=pipeline,
        src_video=str(retake_input_video),
        height=height, width=width, num_frames=num_frames,
        audio_sr=args.audio_sr, device=device,
    )
    base_audio_latent_fp32 = base_audio_latent.float().detach()
    print(f"[setup] Source encoded in {time.perf_counter() - t0:.1f}s")

    print("[setup] Pre-encoding text contexts (Gemma)...")
    t0 = time.perf_counter()
    base_pos_context, base_neg_context = pre_encode_base_contexts(
        pipeline=pipeline, pos_prompt=args.edit_prompt,
        neg_prompt=args.negative_prompt, device=device,
    )
    print(f"[setup] Text encoded in {time.perf_counter() - t0:.1f}s")

    print("[setup] Loading Qwen2.5-VL...")
    t0 = time.perf_counter()
    qwen_model, qwen_processor = build_qwen_model(
        args.qwen_model, device=device, gradient_checkpointing=args.gradient_checkpointing,
    )
    qwen_num_frames = args.qwen_max_frames + (args.qwen_max_frames % 2)
    cached_qwen_inputs, yes_token_id, no_token_id = build_qwen_rubric_inputs(
        processor=qwen_processor, edit_prompt=args.edit_prompt,
        num_frames=qwen_num_frames, img_size=args.qwen_img_size,
        device=device, motion_question=args.qwen_motion_question,
    )
    print(f"[setup] Qwen loaded in {time.perf_counter() - t0:.1f}s  yes_id={yes_token_id} no_id={no_token_id}")

    retake_kwargs = build_retake_kwargs(
        args=args, frame_rate=frame_rate, duration=num_frames / frame_rate,
        video_guider_params=video_guider_params, audio_guider_params=audio_guider_params,
    )

    cached_src_frames = None
    if args.lpips_weight > 0 or args.temporal_weight > 0:
        print("[setup] Caching source frames for perceptual loss...")
        t0 = time.perf_counter()
        cached_src_frames = cache_source_frames(
            pipeline=pipeline, src_video=str(retake_input_video),
            cached_video_latent=cached_video_latent,
            base_audio_latent=base_audio_latent,
            base_pos_context=base_pos_context, base_neg_context=base_neg_context,
            retake_kwargs=retake_kwargs, max_frames=args.max_eval_frames,
            frame_stride=args.frame_stride, eval_sample_start=0,
        )
        print(f"[setup] Cached {cached_src_frames.shape[0]} source frames in {time.perf_counter() - t0:.1f}s")

    t_setup_done = time.perf_counter()
    print(f"\n[setup] Total setup time: {t_setup_done - t_start:.1f}s\n")

    # Build final-render kwargs (full-quality, no low-memory guidance) — same as original
    final_retake_kwargs = dict(retake_kwargs)
    if args.final_retake_num_inference_steps is not None:
        final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps
    final_vg, final_ag = build_guiders_for_mode(args=args, params=params_cfg, use_low_memory_guidance=False)
    final_retake_kwargs["video_guider_params"] = final_vg
    final_retake_kwargs["audio_guider_params"] = final_ag

    # ---- Optimization loop ----
    modes = [m.strip() for m in args.opt_mode.split(",")]
    all_results: dict[str, dict] = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*60}")

        mode_dir = output_dir / f"mode_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)

        t_mode_start = time.perf_counter()
        best = _benchmark_optimize_loop(
            mode=mode,
            args=args,
            output_dir=mode_dir,
            base_pos_context=base_pos_context,
            base_neg_context=base_neg_context,
            base_audio_latent=base_audio_latent,
            base_audio_latent_fp32=base_audio_latent_fp32,
            cached_video_latent=cached_video_latent,
            retake_input_video=str(retake_input_video),
            pipeline=pipeline,
            retake_kwargs=retake_kwargs,
            qwen_model=qwen_model,
            cached_qwen_inputs=cached_qwen_inputs,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            eval_sample_start=0,
            num_frames=num_frames,
            frame_rate=frame_rate,
            audio_sr=waveform_sr,
            cached_src_frames=cached_src_frames,
        )
        t_mode_done = time.perf_counter()
        all_results[mode] = best

        iter_times = best.pop("iter_times", [])
        n = len(iter_times)
        if n > 0:
            warmup = 1  # first iter includes CUDA warmup
            cold = iter_times[:warmup]
            warm = iter_times[warmup:]
            print(f"\n[{mode}] Timing summary ({n} iters):")
            print(f"  Warmup iter(s) : {', '.join(f'{t:.2f}s' for t in cold)}")
            if warm:
                print(f"  Warm avg/min/max: {sum(warm)/len(warm):.2f}s / {min(warm):.2f}s / {max(warm):.2f}s")
            print(f"  Total loop time : {t_mode_done - t_mode_start:.1f}s")

        print(f"[{mode}] Best → yes_prob={best['qwen_score']:.4f}  total_loss={best['qwen_loss']:.4f}  iter={best.get('best_iter', 0)}")

        # ---- Final video render (NOT timed — sanity check only) ----
        if save_final_videos:
            print(f"\n[{mode}] Rendering final video (untimed — sanity check)...")
            render_final_video(
                mode=mode,
                best=best,
                pipeline=pipeline,
                src_video=str(retake_input_video),
                cached_video_latent=cached_video_latent,
                base_audio_latent=base_audio_latent,
                base_pos_context=base_pos_context,
                base_neg_context=base_neg_context,
                retake_kwargs=final_retake_kwargs,
                output_dir=mode_dir,
                num_frames=num_frames,
                frame_rate=frame_rate,
                audio_sr=waveform_sr,
                audio_opt_last_steps=args.audio_opt_last_steps,
                skip_baseline=True,
            )
            final_path = mode_dir / f"best_optimized_video_{mode}.mp4"
            if final_path.exists():
                print(f"[{mode}] Final video saved → {final_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Setup time  : {t_setup_done - t_start:.1f}s")
    print(f"  Opt time    : {t_total - (t_setup_done - t_start):.1f}s")
    print(f"  Total time  : {t_total:.1f}s  (excludes final render)")
    print(f"{'='*60}\n")


def build_parser():
    # Reuse the parser from optimize_qwen_vl so args stay in sync
    from optimize_qwen_vl import build_parser as _base_parser
    return _base_parser()


def main() -> None:
    args = build_parser().parse_args()

    if args.qwen_max_frames % 2 != 0:
        args.qwen_max_frames += 1
    if args.qwen_img_size % 28 != 0:
        args.qwen_img_size = (args.qwen_img_size // 28 + 1) * 28
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    args.ti2v_num_inference_steps = args.num_inference_steps
    args.clip_max_frames = args.qwen_max_frames

    benchmark_run(args)


if __name__ == "__main__":
    main()
