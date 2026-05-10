#!/usr/bin/env python3
"""PPO-vs-direct-optimization ablation for Qwen-guided video editing.

This runner shares the expensive setup across methods for one scenario/seed,
then runs:

  1. gradient         direct differentiable Qwen-guided Adam optimization
  2. ppo_qwen_calls  clipped PPO with the same Qwen-call budget as gradient
  3. ppo_wallclock   clipped PPO with the same optimization-loop wall time

PPO is deliberately a black-box baseline over the same editable text/audio
variables.  It does not train LTX or Qwen; it samples full-video edits and
updates a per-instance Gaussian policy from scalar rewards.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from motion_opt.clip_loss import build_clip_model, encode_text_for_clip
from motion_opt.core import (
    _parse_loras,
    build_cached_source_latents,
    build_guiders_for_mode,
    compute_target_shape,
)
from motion_opt.models import build_retake_pipeline, resolve_quantization_policy
from motion_opt.multimodal_loop import render_baseline_video
from motion_opt.multimodal_loop_qwen import (
    gradient_optimize_multimodal_qwen,
    pre_encode_base_contexts,
    render_final_video,
)
from motion_opt.perceptual_loss import cache_source_frames
from motion_opt.ppo_loop_qwen import active_qwen_rubric_count, ppo_optimize_multimodal_qwen
from motion_opt.qwen_loss import QWEN_IMG_SIZE, build_qwen_model, build_qwen_rubric_inputs
from motion_opt.runtime import build_retake_kwargs, prepare_retake_input_video
from ltx_pipelines.utils.constants import detect_params

from optimize_qwen_vl import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GEMMA_ROOT,
    DEFAULT_QWEN_ROOT,
    _write_clip_similarity_diagnostics,
    build_parser as build_qwen_parser,
)

log = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _write_run_config(args: argparse.Namespace, output_dir: Path, *, method_case: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "args": _jsonable(vars(args)),
        "derived": {
            "output_dir": str(output_dir),
            "method_case": method_case,
        },
        "env": {
            key: os.environ.get(key)
            for key in [
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_SUBMIT_DIR",
                "WANDB_PROJECT",
                "WANDB_ENTITY",
                "WANDB_TAGS",
                "WANDB_MODE",
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
            ]
            if os.environ.get(key) is not None
        },
        "argv": sys.argv,
    }
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def _write_case_metadata(args: argparse.Namespace, case_dir: Path, method_case: str) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(f"{args.edit_prompt}\n")
    (case_dir / "static_prompt.txt").write_text(f"{args.static_prompt}\n")
    (case_dir / "negative_prompt.txt").write_text(f"{args.negative_prompt}\n")
    (case_dir / "method_case.txt").write_text(f"{method_case}\n")
    (case_dir / "qwen_motion_question.txt").write_text(f"{args.qwen_motion_question}\n")
    _write_run_config(args, case_dir, method_case=method_case)


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in csv.DictReader(f))


def _save_best_params(best: dict[str, Any], mode: str, mode_dir: Path) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    if best.get("audio_latent") is not None:
        torch.save(best["audio_latent"].cpu(), mode_dir / f"best_audio_latent_{mode}.pt")
    if best.get("delta_v") is not None:
        torch.save(best["delta_v"].cpu(), mode_dir / f"best_text_delta_{mode}.pt")
    torch.save(
        {
            "mode": mode,
            "qwen_loss": best.get("qwen_loss", float("inf")),
            "qwen_score": best.get("qwen_score", float("-inf")),
            "best_iter": best.get("best_iter", 0),
            "qwen_calls": best.get("qwen_calls", 0),
            "loop_seconds": best.get("loop_seconds", 0.0),
        },
        mode_dir / f"best_params_{mode}.pt",
    )


def _render_final_and_diagnostics(
    *,
    args: argparse.Namespace,
    method_root: Path,
    mode_dir: Path,
    mode: str,
    best: dict[str, Any],
    pipeline,
    retake_input_video: str,
    cached_video_latent: torch.Tensor,
    base_audio_latent: torch.Tensor,
    base_pos_context,
    base_neg_context,
    final_retake_kwargs: dict,
    num_frames: int,
    frame_rate: float,
    audio_sr: int,
    device: torch.device,
) -> str:
    if not args.save_final_videos:
        return ""

    render_final_video(
        mode=mode,
        best=best,
        pipeline=pipeline,
        src_video=retake_input_video,
        cached_video_latent=cached_video_latent,
        base_audio_latent=base_audio_latent,
        base_pos_context=base_pos_context,
        base_neg_context=base_neg_context,
        retake_kwargs=final_retake_kwargs,
        output_dir=mode_dir,
        num_frames=num_frames,
        frame_rate=frame_rate,
        audio_sr=audio_sr,
        audio_opt_last_steps=args.audio_opt_last_steps,
        skip_baseline=True,
    )

    final_video_path = mode_dir / f"best_optimized_video_{mode}.mp4"
    if args.clip_similarity_diag and final_video_path.exists():
        baseline_path = method_root / "baseline_video.mp4"
        if baseline_path.exists():
            static_prompt = args.static_prompt or (
                f"A static frame before the edit: {args.edit_prompt} has not happened yet."
            )
            log.info("[%s] Loading CLIP diagnostic model (%s)...", mode, args.clip_similarity_diag_model)
            clip_model, clip_tokenizer = build_clip_model(args.clip_similarity_diag_model, device)
            static_embedding = encode_text_for_clip(static_prompt, clip_model, clip_tokenizer, device)
            edit_embedding = encode_text_for_clip(args.edit_prompt, clip_model, clip_tokenizer, device)
            _write_clip_similarity_diagnostics(
                baseline_path=baseline_path,
                optimized_path=final_video_path,
                mode=mode,
                output_dir=mode_dir,
                clip_model=clip_model,
                static_embedding=static_embedding,
                edit_embedding=edit_embedding,
                max_frames=args.clip_similarity_diag_max_frames,
                batch_size=args.clip_similarity_diag_batch_size,
                device=device,
                wandb_run=None,
            )
            del clip_model, clip_tokenizer, static_embedding, edit_embedding
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            log.warning("[%s] Skipping CLIP diagnostics; missing baseline at %s", mode, baseline_path)

    return str(final_video_path) if final_video_path.exists() else ""


def _summary_row(
    *,
    args: argparse.Namespace,
    method_case: str,
    mode: str,
    mode_dir: Path,
    best: dict[str, Any],
    actual_rollouts_or_iters: int,
    qwen_calls: int,
    loop_seconds: float,
    budget_type: str,
    budget_value: float | int | None,
    final_video_path: str,
) -> dict[str, Any]:
    return {
        "scenario": args.scenario_name,
        "seed": args.seed,
        "method": method_case,
        "mode": mode,
        "budget_type": budget_type,
        "budget_value": budget_value if budget_value is not None else "",
        "actual_iterations_or_rollouts": actual_rollouts_or_iters,
        "qwen_calls": qwen_calls,
        "loop_seconds": loop_seconds,
        "best_iter": int(best.get("best_iter", 0) or 0),
        "best_qwen_yes_prob": float(best.get("qwen_score", float("nan"))),
        "best_total_loss": float(best.get("qwen_loss", float("nan"))),
        "output_dir": str(mode_dir),
        "final_video": final_video_path,
    }


def _write_summaries(rows: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "ablation_summary.json").open("w") as f:
        json.dump({"runs": rows}, f, indent=2, sort_keys=True)

    csv_path = output_dir / "ppo_vs_optimize_summary.csv"
    fieldnames = [
        "scenario", "seed", "method", "mode", "budget_type", "budget_value",
        "actual_iterations_or_rollouts", "qwen_calls", "loop_seconds",
        "best_iter", "best_qwen_yes_prob", "best_total_loss",
        "output_dir", "final_video",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote ablation summaries to %s and %s", output_dir / "ablation_summary.json", csv_path)


def _run_gradient_case(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    mode: str,
    setup: dict[str, Any],
    active_rubric_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method_case = "gradient"
    case_dir = output_dir / method_case
    mode_dir = case_dir / f"mode_{mode}"
    _write_case_metadata(args, case_dir, method_case)
    mode_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Running direct gradient optimization: mode=%s", mode)
    log.info("=" * 60)

    start = time.perf_counter()
    best = gradient_optimize_multimodal_qwen(
        mode=mode,
        args=args,
        is_main=True,
        output_dir=mode_dir,
        base_pos_context=setup["base_pos_context"],
        base_neg_context=setup["base_neg_context"],
        base_audio_latent=setup["base_audio_latent"],
        base_audio_latent_fp32=setup["base_audio_latent_fp32"],
        cached_video_latent=setup["cached_video_latent"],
        retake_input_video=str(setup["retake_input_video"]),
        pipeline=setup["pipeline"],
        retake_kwargs=setup["retake_kwargs"],
        qwen_model=setup["qwen_model"],
        cached_qwen_inputs=setup["cached_qwen_inputs"],
        yes_token_id=setup["yes_token_id"],
        no_token_id=setup["no_token_id"],
        eval_sample_start=0,
        visualize_retake_kwargs=setup["final_retake_kwargs"] if args.visualize_every_iters > 0 else None,
        num_frames=setup["num_frames"],
        frame_rate=setup["frame_rate"],
        audio_sr=setup["waveform_sr"],
        wandb_run=None,
        extract_attn_maps=False,
        cached_src_frames=setup["cached_src_frames"],
    )
    loop_seconds = time.perf_counter() - start

    csv_path = mode_dir / f"optimization_log_qwen_{mode}.csv"
    actual_iters = _count_csv_rows(csv_path)
    qwen_calls = actual_iters * max(1, int(args.qwen_grad_accum_steps)) * active_rubric_count
    best["qwen_calls"] = qwen_calls
    best["loop_seconds"] = loop_seconds

    _save_best_params(best, mode, mode_dir)
    final_video_path = _render_final_and_diagnostics(
        args=args,
        method_root=output_dir,
        mode_dir=mode_dir,
        mode=mode,
        best=best,
        pipeline=setup["pipeline"],
        retake_input_video=str(setup["retake_input_video"]),
        cached_video_latent=setup["cached_video_latent"],
        base_audio_latent=setup["base_audio_latent"],
        base_pos_context=setup["base_pos_context"],
        base_neg_context=setup["base_neg_context"],
        final_retake_kwargs=setup["final_retake_kwargs"],
        num_frames=setup["num_frames"],
        frame_rate=setup["frame_rate"],
        audio_sr=setup["waveform_sr"],
        device=setup["device"],
    )

    row = _summary_row(
        args=args,
        method_case=method_case,
        mode=mode,
        mode_dir=mode_dir,
        best=best,
        actual_rollouts_or_iters=actual_iters,
        qwen_calls=qwen_calls,
        loop_seconds=loop_seconds,
        budget_type="gradient",
        budget_value="",
        final_video_path=final_video_path,
    )
    return best, row


def _run_ppo_case(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    method_case: str,
    mode: str,
    setup: dict[str, Any],
    qwen_call_budget: int | None,
    wallclock_budget_seconds: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir = output_dir / method_case
    mode_dir = case_dir / f"mode_{mode}"
    _write_case_metadata(args, case_dir, method_case)

    log.info("=" * 60)
    log.info(
        "Running PPO optimization: case=%s mode=%s qwen_budget=%s wallclock_budget=%s",
        method_case,
        mode,
        qwen_call_budget,
        wallclock_budget_seconds,
    )
    log.info("=" * 60)

    best = ppo_optimize_multimodal_qwen(
        mode=mode,
        args=args,
        is_main=True,
        output_dir=mode_dir,
        base_pos_context=setup["base_pos_context"],
        base_neg_context=setup["base_neg_context"],
        base_audio_latent=setup["base_audio_latent"],
        base_audio_latent_fp32=setup["base_audio_latent_fp32"],
        cached_video_latent=setup["cached_video_latent"],
        retake_input_video=str(setup["retake_input_video"]),
        pipeline=setup["pipeline"],
        retake_kwargs=setup["retake_kwargs"],
        qwen_model=setup["qwen_model"],
        cached_qwen_inputs=setup["cached_qwen_inputs"],
        yes_token_id=setup["yes_token_id"],
        no_token_id=setup["no_token_id"],
        eval_sample_start=0,
        cached_src_frames=setup["cached_src_frames"],
        qwen_call_budget=qwen_call_budget,
        wallclock_budget_seconds=wallclock_budget_seconds,
        method_name=method_case,
    )

    csv_path = mode_dir / f"optimization_log_qwen_ppo_{mode}.csv"
    actual_rollouts = _count_csv_rows(csv_path)
    qwen_calls = int(best.get("qwen_calls", 0) or 0)
    loop_seconds = float(best.get("loop_seconds", 0.0) or 0.0)

    _save_best_params(best, mode, mode_dir)
    final_video_path = _render_final_and_diagnostics(
        args=args,
        method_root=output_dir,
        mode_dir=mode_dir,
        mode=mode,
        best=best,
        pipeline=setup["pipeline"],
        retake_input_video=str(setup["retake_input_video"]),
        cached_video_latent=setup["cached_video_latent"],
        base_audio_latent=setup["base_audio_latent"],
        base_pos_context=setup["base_pos_context"],
        base_neg_context=setup["base_neg_context"],
        final_retake_kwargs=setup["final_retake_kwargs"],
        num_frames=setup["num_frames"],
        frame_rate=setup["frame_rate"],
        audio_sr=setup["waveform_sr"],
        device=setup["device"],
    )

    if method_case == "ppo_qwen_calls":
        budget_type = "qwen_calls"
        budget_value = qwen_call_budget
    else:
        budget_type = "wallclock_seconds"
        budget_value = wallclock_budget_seconds

    row = _summary_row(
        args=args,
        method_case=method_case,
        mode=mode,
        mode_dir=mode_dir,
        best=best,
        actual_rollouts_or_iters=actual_rollouts,
        qwen_calls=qwen_calls,
        loop_seconds=loop_seconds,
        budget_type=budget_type,
        budget_value=budget_value,
        final_video_path=final_video_path,
    )
    return best, row


def _build_shared_setup(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    height, width, num_frames, frame_rate = compute_target_shape(
        args.src_video,
        args.height,
        args.width,
        args.num_frames,
        args.frame_rate,
    )
    duration = num_frames / frame_rate
    log.info("Video shape: %dx%d, %d frames @ %.1f fps (%.2fs)", width, height, num_frames, frame_rate, duration)

    retake_quant = resolve_quantization_policy(
        args.retake_quantization if args.retake_quantization is not None else args.quantization
    )

    retake_input_video = prepare_retake_input_video(
        args=args,
        is_main=True,
        output_dir=output_dir,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
    )

    params = detect_params(args.checkpoint_path)
    video_guider_params, audio_guider_params = build_guiders_for_mode(
        args=args,
        params=params,
        use_low_memory_guidance=args.low_memory_guidance,
    )

    log.info("Loading RetakePipeline (checkpoint: %s)...", args.checkpoint_path)
    pipeline = build_retake_pipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=_parse_loras(args.loras),
        device=device,
        quant_policy=retake_quant,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    log.info("Encoding source video/audio...")
    cached_video_latent, base_audio_latent, waveform_sr = build_cached_source_latents(
        pipeline=pipeline,
        src_video=str(retake_input_video),
        height=height,
        width=width,
        num_frames=num_frames,
        audio_sr=args.audio_sr,
        device=device,
    )
    base_audio_latent_fp32 = base_audio_latent.float().detach()

    log.info("Pre-encoding text contexts...")
    base_pos_context, base_neg_context = pre_encode_base_contexts(
        pipeline=pipeline,
        pos_prompt=args.edit_prompt,
        neg_prompt=args.negative_prompt,
        device=device,
    )

    log.info("Loading Qwen2.5-VL model (%s)...", args.qwen_model)
    qwen_model, qwen_processor = build_qwen_model(
        args.qwen_model,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    qwen_num_frames = args.qwen_max_frames
    if qwen_num_frames % 2 != 0:
        qwen_num_frames += 1
    cached_qwen_inputs, yes_token_id, no_token_id = build_qwen_rubric_inputs(
        processor=qwen_processor,
        edit_prompt=args.edit_prompt,
        num_frames=qwen_num_frames,
        img_size=args.qwen_img_size,
        device=device,
        motion_question=args.qwen_motion_question,
    )

    retake_kwargs = build_retake_kwargs(
        args=args,
        frame_rate=frame_rate,
        duration=duration,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
    )

    cached_src_frames = None
    if args.lpips_weight > 0 or args.temporal_weight > 0:
        log.info("Caching source frames for perceptual losses...")
        cached_src_frames = cache_source_frames(
            pipeline=pipeline,
            src_video=str(retake_input_video),
            cached_video_latent=cached_video_latent,
            base_audio_latent=base_audio_latent,
            base_pos_context=base_pos_context,
            base_neg_context=base_neg_context,
            retake_kwargs=retake_kwargs,
            max_frames=args.max_eval_frames,
            frame_stride=args.frame_stride,
            eval_sample_start=0,
        )

    final_retake_kwargs = dict(retake_kwargs)
    if args.final_retake_num_inference_steps is not None:
        final_retake_kwargs["num_inference_steps"] = args.final_retake_num_inference_steps
    final_vg, final_ag = build_guiders_for_mode(args=args, params=params, use_low_memory_guidance=False)
    final_retake_kwargs["video_guider_params"] = final_vg
    final_retake_kwargs["audio_guider_params"] = final_ag

    if args.save_final_videos:
        baseline_path = output_dir / "baseline_video.mp4"
        log.info("Rendering shared baseline video to %s...", baseline_path)
        render_baseline_video(
            pipeline=pipeline,
            src_video=str(retake_input_video),
            cached_video_latent=cached_video_latent,
            base_audio_latent=base_audio_latent,
            base_pos_context=base_pos_context,
            base_neg_context=base_neg_context,
            retake_kwargs=final_retake_kwargs,
            output_path=baseline_path,
            num_frames=num_frames,
            frame_rate=frame_rate,
            audio_sr=waveform_sr,
        )

    return {
        "device": device,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "waveform_sr": waveform_sr,
        "retake_input_video": retake_input_video,
        "pipeline": pipeline,
        "cached_video_latent": cached_video_latent,
        "base_audio_latent": base_audio_latent,
        "base_audio_latent_fp32": base_audio_latent_fp32,
        "base_pos_context": base_pos_context,
        "base_neg_context": base_neg_context,
        "qwen_model": qwen_model,
        "cached_qwen_inputs": cached_qwen_inputs,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "retake_kwargs": retake_kwargs,
        "final_retake_kwargs": final_retake_kwargs,
        "cached_src_frames": cached_src_frames,
    }


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(args, output_dir, method_case=None)

    setup_start = time.perf_counter()
    setup = _build_shared_setup(args, output_dir)
    setup_seconds = time.perf_counter() - setup_start
    log.info("Shared setup finished in %.1fs", setup_seconds)

    method_cases = [case.strip().lower() for case in args.method_cases.split(",") if case.strip()]
    modes = [mode.strip() for mode in args.opt_mode.split(",") if mode.strip()]
    active_count = active_qwen_rubric_count(args, setup["cached_qwen_inputs"])
    summary_rows: list[dict[str, Any]] = []
    gradient_budgets: dict[str, dict[str, float | int]] = {}

    for mode in modes:
        gradient_best: dict[str, Any] | None = None
        for method_case in method_cases:
            if method_case == "gradient":
                gradient_best, row = _run_gradient_case(
                    args=args,
                    output_dir=output_dir,
                    mode=mode,
                    setup=setup,
                    active_rubric_count=active_count,
                )
                summary_rows.append(row)
                gradient_budgets[mode] = {
                    "qwen_calls": int(row["qwen_calls"]),
                    "loop_seconds": float(row["loop_seconds"]),
                }
            elif method_case in {"ppo_qwen_calls", "ppo_wallclock"}:
                explicit_qwen_budget = int(args.ppo_qwen_call_budget) if int(args.ppo_qwen_call_budget) > 0 else None
                explicit_time_budget = (
                    float(args.ppo_wallclock_budget_seconds)
                    if float(args.ppo_wallclock_budget_seconds) > 0
                    else None
                )
                if method_case == "ppo_qwen_calls":
                    qwen_budget = explicit_qwen_budget
                    if qwen_budget is None:
                        if mode not in gradient_budgets:
                            raise RuntimeError(
                                "ppo_qwen_calls requires a prior gradient case or --ppo-qwen-call-budget."
                            )
                        qwen_budget = int(gradient_budgets[mode]["qwen_calls"])
                    time_budget = None
                else:
                    qwen_budget = None
                    time_budget = explicit_time_budget
                    if time_budget is None:
                        if mode not in gradient_budgets:
                            raise RuntimeError(
                                "ppo_wallclock requires a prior gradient case or --ppo-wallclock-budget-seconds."
                            )
                        time_budget = float(gradient_budgets[mode]["loop_seconds"])

                _, row = _run_ppo_case(
                    args=args,
                    output_dir=output_dir,
                    method_case=method_case,
                    mode=mode,
                    setup=setup,
                    qwen_call_budget=qwen_budget,
                    wallclock_budget_seconds=time_budget,
                )
                summary_rows.append(row)
            else:
                raise ValueError(
                    f"Unknown method case {method_case!r}. Use gradient, ppo_qwen_calls, ppo_wallclock."
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if gradient_best is None and any(case.startswith("ppo_") for case in method_cases):
            log.info("[%s] PPO cases used explicit budgets; no gradient result was run.", mode)

    _write_summaries(summary_rows, output_dir)

    log.info("")
    log.info("=" * 80)
    log.info("PPO VS DIRECT OPTIMIZATION SUMMARY")
    log.info("=" * 80)
    for row in summary_rows:
        log.info(
            "%-16s mode=%-5s best_yes=%.4f total=%.4f calls=%s loop=%.1fs final=%s",
            row["method"],
            row["mode"],
            row["best_qwen_yes_prob"],
            row["best_total_loss"],
            row["qwen_calls"],
            row["loop_seconds"],
            row["final_video"] or "n/a",
        )
    log.info("Outputs saved to: %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    p = build_qwen_parser()
    p.description = __doc__
    p.set_defaults(
        checkpoint_path=DEFAULT_CHECKPOINT,
        gemma_root=DEFAULT_GEMMA_ROOT,
        qwen_model=DEFAULT_QWEN_ROOT,
    )

    p.add_argument("--scenario-name", default="custom")
    p.add_argument(
        "--method-cases",
        default="gradient,ppo_qwen_calls,ppo_wallclock",
        help="Comma-separated cases: gradient, ppo_qwen_calls, ppo_wallclock.",
    )

    p.add_argument("--ppo-rollouts-per-update", type=int, default=4)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--ppo-clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-lr", type=float, default=1e-3)
    p.add_argument("--ppo-init-std-audio", type=float, default=0.02)
    p.add_argument("--ppo-init-std-text", type=float, default=0.01)
    p.add_argument("--ppo-entropy-weight", type=float, default=0.0)
    p.add_argument("--ppo-max-updates", type=int, default=0)
    p.add_argument("--ppo-qwen-call-budget", type=int, default=0)
    p.add_argument("--ppo-wallclock-budget-seconds", type=float, default=0.0)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.qwen_max_frames % 2 != 0:
        args.qwen_max_frames += 1
        log.warning("--qwen-max-frames rounded up to %d (must be even).", args.qwen_max_frames)
    if args.qwen_img_size % 28 != 0:
        args.qwen_img_size = (args.qwen_img_size // 28 + 1) * 28
        log.warning("--qwen-img-size rounded up to %d (must be divisible by 28).", args.qwen_img_size)
    if args.retake_num_inference_steps is None:
        args.retake_num_inference_steps = args.num_inference_steps
    args.ti2v_num_inference_steps = args.num_inference_steps
    args.clip_max_frames = args.qwen_max_frames
    if not hasattr(args, "qwen_img_size"):
        args.qwen_img_size = QWEN_IMG_SIZE

    run(args)


if __name__ == "__main__":
    main()
