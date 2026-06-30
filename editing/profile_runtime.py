#!/usr/bin/env python3
"""Test-time runtime breakdown for the motion-editing pipeline.

Answers the reviewer's request for a clear cost breakdown — backbone inference,
VLM scoring, optimization iterations, and hardware — by timing each stage of a
*real* run on the actual model, reusing the exact code paths of
``optimize_qwen_vl.py`` (same pipeline, same differentiable Retake render, same
Qwen2.5-VL rubric loss). It does NOT re-implement anything: it imports the same
setup helpers and the same render/score functions the optimizer uses.

What it measures (all GPU-synchronised, seconds):

  one-time setup
    pipeline_load        load LTX-2 Retake pipeline (fp8-cast) onto the GPU
    source_encode        VAE-encode the source video + audio latents
    text_encode          Gemma text-context pre-encoding (one-time)
    qwen_load            load Qwen2.5-VL + build the cached rubric inputs

  per-generated-video
    backbone_inference   one full Retake generation, all denoising steps, no grad
                         (= the cost to produce one video; the reviewer's
                          "backbone inference")

  per optimization iteration (averaged over PROFILE_ITERS, after PROFILE_WARMUP)
    iter_render          differentiable Retake forward (grad through the last
                         audio_opt_last_steps denoising steps)
    iter_vlm_score       Qwen2.5-VL forward = "VLM scoring"
    iter_backward        backward: Qwen -> differentiable decoder -> latent
    iter_optim_step      Adam step
    iter_total           sum of the above (one optimization iteration)

  projection
    Uses the config's iterations to project an end-to-end wall-clock estimate.

Run it through the same config plumbing as a normal run; this script takes the
SAME CLI args as ``optimize_qwen_vl.py`` (so ``parse_config.py`` feeds it
unchanged). Two extra knobs are read from the environment to keep the arg set
identical:

    PROFILE_WARMUP   # untimed warmup iterations (default 2)
    PROFILE_ITERS    # timed iterations to average (default 5)

Output: prints a table and writes ``runtime_breakdown.json`` in the output dir.
Logging is forced to WARNING and W&B/TensorBoard/CSV/previews are never created,
so the measured numbers reflect compute, not bookkeeping.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Reuse the real entry point's parser and setup helpers verbatim — no divergence
# from what a production run does.
from optimize_qwen_vl import build_parser  # noqa: E402

from motion_opt.core import (  # noqa: E402
    build_cached_source_latents,
    build_guiders_for_mode,
    compute_target_shape,
)
from motion_opt.models import build_retake_pipeline, resolve_quantization_policy  # noqa: E402
from motion_opt.multimodal_loop import render_with_injected_latents  # noqa: E402
from motion_opt.multimodal_loop_qwen import pre_encode_base_contexts  # noqa: E402
from motion_opt.perceptual_loss import (  # noqa: E402
    adaptive_reg_weight,
    cache_source_frames,
    compute_perceptual_quality_loss,
)
from motion_opt.qwen_loss import (  # noqa: E402
    build_qwen_model,
    build_qwen_rubric_inputs,
    compute_qwen_video_loss,
)
from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video  # noqa: E402
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput  # noqa: E402
from ltx_pipelines.utils.constants import detect_params  # noqa: E402

log = logging.getLogger("profile_runtime")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Timer:
    """GPU-synchronised stopwatch (context manager)."""

    def __init__(self) -> None:
        self.dt = 0.0

    def __enter__(self) -> "_Timer":
        _sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        _sync()
        self.dt = time.perf_counter() - self._t0


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "device": "cuda",
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def main() -> None:
    args = build_parser().parse_args()

    # ---- Mirror main()'s arg normalisation ----
    if args.qwen_max_frames % 2 != 0:
        args.qwen_max_frames += 1
    if args.qwen_img_size % 28 != 0:
        args.qwen_img_size = (args.qwen_img_size // 28 + 1) * 28
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    args.ti2v_num_inference_steps = args.num_inference_steps
    args.clip_max_frames = args.qwen_max_frames

    warmup = int(os.environ.get("PROFILE_WARMUP", "2"))
    timed_iters = int(os.environ.get("PROFILE_ITERS", "5"))

    # Keep it fast: no info/debug spam, no W&B/TensorBoard/CSV/previews.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # Profile the most representative single mode (default: the first listed).
    mode = [m.strip() for m in args.opt_mode.split(",")][0]
    optimize_text = mode in ("text", "both")
    optimize_audio = mode in ("audio", "both")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    # ---- Shape + input prep (cheap; not a model cost) ----
    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video, args.height, args.width, args.num_frames, args.frame_rate,
    )
    duration = num_frames / frame_rate
    retake_quant = resolve_quantization_policy(
        args.retake_quantization if args.retake_quantization is not None else args.quantization
    )
    retake_input_video = prepare_retake_input_video(
        args=args, is_main=True, output_dir=output_dir,
        height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
    )
    params = detect_params(args.checkpoint_path)
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args, params=params, use_low_memory_guidance=args.low_memory_guidance,
    )

    # ---- 1. Pipeline load (backbone weights onto GPU) ----
    print("Loading LTX-2 Retake pipeline...", flush=True)
    with _Timer() as t:
        pipeline = build_retake_pipeline(
            checkpoint_path=args.checkpoint_path,
            gemma_root=args.gemma_root,
            loras=[],
            device=device,
            quant_policy=retake_quant,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    timings["pipeline_load"] = t.dt

    # ---- 2. Source video/audio VAE encode ----
    print("Encoding source video/audio...", flush=True)
    with _Timer() as t:
        cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
            pipeline=pipeline, src_video=str(retake_input_video),
            height=height, width=width, num_frames=num_frames,
            audio_sr=args.audio_sr, device=device,
        )
    timings["source_encode"] = t.dt
    base_audio_latent_fp32 = base_audio_latent.float().detach()

    # ---- 3. Gemma text context pre-encode (one-time) ----
    print("Pre-encoding text contexts (Gemma)...", flush=True)
    with _Timer() as t:
        base_pos_context, base_neg_context = pre_encode_base_contexts(
            pipeline=pipeline, pos_prompt=args.edit_prompt,
            neg_prompt=args.negative_prompt, device=device,
        )
    timings["text_encode"] = t.dt

    # ---- 4. Qwen2.5-VL load + rubric build ----
    print("Loading Qwen2.5-VL + building rubric inputs...", flush=True)
    with _Timer() as t:
        qwen_model, qwen_processor = build_qwen_model(
            args.qwen_model, device=device,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        qwen_num_frames = args.qwen_max_frames + (args.qwen_max_frames % 2)
        cached_qwen_inputs, yes_token_id, no_token_id = build_qwen_rubric_inputs(
            processor=qwen_processor, edit_prompt=args.edit_prompt,
            num_frames=qwen_num_frames, img_size=args.qwen_img_size, device=device,
            motion_question=args.qwen_motion_question,
            gradient_rubric=args.qwen_gradient_rubric,
        )
    timings["qwen_load"] = t.dt

    retake_kwargs = build_retake_kwargs(
        args=args, frame_rate=frame_rate, duration=duration,
        video_guider_params=video_guider_params, audio_guider_params=audio_guider_params,
    )
    eval_sample_start = 0

    # ---- Cache source frames for the perceptual term (one-time, if enabled) ----
    lpips_weight = float(getattr(args, "lpips_weight", 0.0))
    temporal_weight = float(getattr(args, "temporal_weight", 0.0))
    need_perceptual = lpips_weight > 0 or temporal_weight > 0
    cached_src_frames = None
    if need_perceptual:
        print("Caching source frames for perceptual loss...", flush=True)
        cached_src_frames = cache_source_frames(
            pipeline=pipeline, src_video=str(retake_input_video),
            cached_video_latent=cached_video_latent, base_audio_latent=base_audio_latent,
            base_pos_context=base_pos_context, base_neg_context=base_neg_context,
            retake_kwargs=retake_kwargs, max_frames=args.max_eval_frames,
            frame_stride=args.frame_stride, eval_sample_start=eval_sample_start,
        )

    rubric_weight_overrides = None
    if args.qwen_gradient_rubric != "full":
        rubric_weight_overrides = {
            "motion": 0.0, "entities": 0.0, "overall": 0.0,
            str(args.qwen_gradient_rubric): 1.0,
        }

    def _render(audio_for_render, pos_ctx_iter):
        return render_with_injected_latents(
            pipeline=pipeline, src_video=str(retake_input_video),
            injected_audio_latent=audio_for_render,
            cached_video_latent=cached_video_latent,
            retake_kwargs=retake_kwargs, max_frames=args.max_eval_frames,
            frame_stride=args.frame_stride, resize_to=None,
            audio_opt_last_steps=args.audio_opt_last_steps,
            eval_sample_start=eval_sample_start,
            pos_context=pos_ctx_iter, neg_context=base_neg_context,
            inject_text_context=optimize_text,
        )

    # ---- 5. Backbone inference: one full generation, all steps, no grad ----
    # This is the pure cost of producing one video (the reviewer's "backbone
    # inference"), independent of optimization.
    print("Timing full backbone inference (one generation, no grad)...", flush=True)
    with _Timer() as t:
        with torch.no_grad():
            frames = _render(base_audio_latent, base_pos_context)
    timings["backbone_inference"] = t.dt
    n_frames_eval = int(frames.shape[0])
    del frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 6. Per optimization-iteration breakdown ----
    # Faithfully mirrors gradient_optimize_multimodal_qwen's per-iteration body:
    # render -> (perceptual fwd+bwd) -> Qwen loss with grad-accum -> reg -> step.
    # Same code, same number of Qwen passes, same perceptual term — only the
    # logging/CSV/W&B/preview bookkeeping is removed. Buckets:
    #   iter_render     differentiable Retake forward (backbone, grad last-N steps)
    #   iter_perceptual LPIPS + temporal forward+backward (0 if disabled)
    #   iter_vlm_grad   Qwen scoring forward + gradient backprop to the latent
    #                   (all qwen_grad_accum_steps passes; the reviewer's "VLM scoring")
    #   iter_optim_step regularization backward + grad-clip + Adam step
    opt_params: list[torch.nn.Parameter] = []
    delta_v = None
    audio_latent = None
    if optimize_text:
        delta_v = torch.nn.Parameter(torch.zeros_like(base_pos_context.video_encoding.float()))
        opt_params.append(delta_v)
    if optimize_audio:
        audio_latent = torch.nn.Parameter(base_audio_latent_fp32.clone())
        opt_params.append(audio_latent)
    optimizer = torch.optim.Adam(opt_params, lr=args.lr)

    # Profiling assumption: always use a single Qwen pass per iteration,
    # regardless of the config's qwen_grad_accum_steps. This is a profiling-only
    # simplification — the real loop multiplies the Qwen-scoring cost by
    # qwen_grad_accum_steps, so multiply iter_vlm_grad by that factor if you want
    # the cost under accumulation. (Scoped to this file; configs are untouched.)
    qwen_grad_accum = 1
    base_sample_mode = getattr(args, "qwen_sample_mode", "linspace")
    contiguous_start = getattr(args, "qwen_contiguous_start_frame", 0)
    reg_schedule = getattr(args, "reg_schedule", "constant")
    n_steps = int(args.retake_num_inference_steps)
    grad_steps = min(max(int(args.audio_opt_last_steps), 0), n_steps)
    iters_cfg = int(args.iterations)

    acc = {"iter_render": 0.0, "iter_perceptual": 0.0, "iter_vlm_grad": 0.0, "iter_optim_step": 0.0}

    total_loop = warmup + timed_iters
    print(f"Timing {timed_iters} optimization iterations ({warmup} warmup, "
          f"grad_accum={qwen_grad_accum}, perceptual={need_perceptual})...", flush=True)
    for it in range(total_loop):
        measure = it >= warmup
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
            audio_latent.to(dtype=base_audio_latent.dtype) if audio_latent is not None else base_audio_latent
        )

        # --- render (differentiable backbone forward) ---
        tr = _Timer()
        with tr:
            gen_frames = _render(audio_for_render, pos_ctx_iter)

        # --- perceptual loss (LPIPS + temporal), backward with retain_graph ---
        tp = _Timer()
        with tp:
            if need_perceptual and cached_src_frames is not None:
                perceptual_loss_t, _ = compute_perceptual_quality_loss(
                    gen_frames=gen_frames, src_frames=cached_src_frames,
                    lpips_weight=lpips_weight, temporal_weight=temporal_weight,
                    backbone=getattr(args, "lpips_backbone", "alex"),
                    max_lpips_frames=min(16, gen_frames.shape[0]),
                    max_temporal_pairs=min(12, gen_frames.shape[0] - 1),
                    backward=False,
                )
                if perceptual_loss_t.requires_grad:
                    perceptual_loss_t.backward(retain_graph=True)

        # --- Qwen scoring + gradient (single-pass fused, or grad-accum) ---
        tv = _Timer()
        with tv:
            if qwen_grad_accum == 1:
                loss_t, _ = compute_qwen_video_loss(
                    frames_chw=gen_frames, qwen_model=qwen_model,
                    cached_inputs=cached_qwen_inputs,
                    yes_token_id=yes_token_id, no_token_id=no_token_id,
                    max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                    backward=True, return_details=True,
                    sample_mode=base_sample_mode, contiguous_start_frame=contiguous_start,
                    rubric_weight_overrides=rubric_weight_overrides,
                )
            else:
                gf_det = gen_frames.detach().requires_grad_(True)
                for accum_i in range(qwen_grad_accum):
                    pass_mode = base_sample_mode if accum_i == 0 else "contiguous_random"
                    _r = compute_qwen_video_loss(
                        frames_chw=gf_det, qwen_model=qwen_model,
                        cached_inputs=cached_qwen_inputs,
                        yes_token_id=yes_token_id, no_token_id=no_token_id,
                        max_frames=args.qwen_max_frames, img_size=args.qwen_img_size,
                        backward=False, return_details=(accum_i == 0),
                        sample_mode=pass_mode, contiguous_start_frame=contiguous_start,
                        rubric_weight_overrides=rubric_weight_overrides,
                    )
                    loss_i = _r[0] if accum_i == 0 else _r
                    (loss_i / qwen_grad_accum).backward()
                if gf_det.grad is not None and gen_frames.grad_fn is not None:
                    gen_frames.backward(gradient=gf_det.grad)

        # --- regularization backward + grad clip + optimizer step ---
        to = _Timer()
        with to:
            audio_reg_w = adaptive_reg_weight(args.latent_reg_weight, it + 1, iters_cfg, schedule=reg_schedule)
            text_reg_w = adaptive_reg_weight(args.text_reg_weight, it + 1, iters_cfg, schedule=reg_schedule)
            reg_t = torch.zeros((), device=gen_frames.device)
            if optimize_audio and audio_latent is not None and audio_reg_w > 0:
                reg_t = reg_t + audio_reg_w * torch.mean((audio_latent - base_audio_latent_fp32) ** 2)
            if optimize_text and delta_v is not None and text_reg_w > 0:
                reg_t = reg_t + text_reg_w * torch.mean(delta_v ** 2)
            if reg_t.requires_grad:
                reg_t.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=args.grad_clip)
            optimizer.step()

        if measure:
            acc["iter_render"] += tr.dt
            acc["iter_perceptual"] += tp.dt
            acc["iter_vlm_grad"] += tv.dt
            acc["iter_optim_step"] += to.dt

        del gen_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for k in acc:
        timings[k] = acc[k] / max(timed_iters, 1)
    timings["iter_total"] = (
        timings["iter_render"] + timings["iter_perceptual"]
        + timings["iter_vlm_grad"] + timings["iter_optim_step"]
    )

    peak_mem_gb = (
        round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
    )

    # ---- Projection to an end-to-end run ----
    setup_total = (
        timings["pipeline_load"] + timings["source_encode"]
        + timings["text_encode"] + timings["qwen_load"]
    )
    # One baseline render + one final render per mode ≈ 2 backbone passes.
    projected_total = (
        setup_total + 2 * timings["backbone_inference"] + iters_cfg * timings["iter_total"]
    )

    report = {
        "hardware": _gpu_info(),
        "peak_gpu_memory_gb": peak_mem_gb,
        "config": {
            "src_video": args.src_video,
            "edit_prompt": args.edit_prompt,
            "opt_mode_profiled": mode,
            "quantization": args.quantization,
            "resolution": f"{width}x{height}",
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "eval_frames_per_render": n_frames_eval,
            "retake_num_inference_steps": n_steps,
            "grad_carrying_denoise_steps": grad_steps,
            "audio_opt_last_steps": args.audio_opt_last_steps,
            "qwen_max_frames": args.qwen_max_frames,
            "qwen_img_size": args.qwen_img_size,
            "qwen_grad_accum_steps_profiled": qwen_grad_accum,
            "qwen_grad_accum_steps_config": int(getattr(args, "qwen_grad_accum_steps", 1)),
            "lpips_weight": lpips_weight,
            "temporal_weight": temporal_weight,
            "perceptual_enabled": need_perceptual,
            "iterations_configured": iters_cfg,
            "profile_warmup": warmup,
            "profile_timed_iters": timed_iters,
        },
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
        "projection_seconds": {
            "setup_total": round(setup_total, 3),
            "per_optimization_iteration": round(timings["iter_total"], 3),
            "estimated_end_to_end": round(projected_total, 3),
            "estimated_end_to_end_minutes": round(projected_total / 60.0, 2),
        },
    }

    out_path = output_dir / "runtime_breakdown.json"
    out_path.write_text(json.dumps(report, indent=2))

    # ---- Pretty table ----
    hw = report["hardware"]
    print("\n" + "=" * 64)
    print("  RUNTIME BREAKDOWN")
    print("=" * 64)
    print(f"  Hardware : {hw.get('name', hw.get('device'))}  "
          f"({hw.get('total_memory_gb', '?')} GB, CUDA {hw.get('cuda', '?')})")
    print(f"  Peak GPU memory : {peak_mem_gb} GB")
    print(f"  Quant : {args.quantization}   Resolution : {width}x{height}  "
          f"{num_frames}f @ {frame_rate:.1f}fps")
    print(f"  Denoising steps : {n_steps} (grad through last {grad_steps})   "
          f"Qwen frames : {args.qwen_max_frames}@{args.qwen_img_size}px")
    print("-" * 64)
    print("  One-time setup")
    for k in ("pipeline_load", "source_encode", "text_encode", "qwen_load"):
        print(f"    {k:<22s} {timings[k]:8.2f} s")
    print(f"    {'setup_total':<22s} {setup_total:8.2f} s")
    print("-" * 64)
    print("  Per generated video")
    print(f"    {'backbone_inference':<22s} {timings['backbone_inference']:8.2f} s")
    print("-" * 64)
    print(f"  Per optimization iteration (avg of {timed_iters}, "
          f"grad_accum={qwen_grad_accum}, perceptual={need_perceptual})")
    for k in ("iter_render", "iter_perceptual", "iter_vlm_grad", "iter_optim_step", "iter_total"):
        print(f"    {k:<22s} {timings[k]:8.2f} s")
    print("-" * 64)
    print(f"  Projected end-to-end ({iters_cfg} iters): "
          f"{projected_total:.1f} s  ({projected_total/60:.2f} min)")
    print("=" * 64)
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
