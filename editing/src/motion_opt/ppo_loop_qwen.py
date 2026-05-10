"""PPO baseline loop for Qwen-guided per-video editing ablations.

This module intentionally treats the existing differentiable editor as a
black-box environment: one sampled text/audio latent edit renders one complete
video, receives one Qwen-derived reward, and then PPO updates a Gaussian policy
over the same editable continuous variables used by the direct optimizer.
"""
from __future__ import annotations

import csv
import gc
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput

from .multimodal_loop import render_with_injected_latents
from .perceptual_loss import adaptive_reg_weight, compute_perceptual_quality_loss
from .qwen_loss import compute_qwen_video_loss

log = logging.getLogger(__name__)

_LOG_2PI = math.log(2.0 * math.pi)


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def active_qwen_rubric_count(args, cached_qwen_inputs: dict[str, Any]) -> int:
    """Return how many Qwen model calls one objective evaluation performs."""
    rubric_items = cached_qwen_inputs.get("rubric_items", [])
    if not rubric_items:
        return 1
    if getattr(args, "qwen_gradient_rubric", "motion") == "full":
        return max(1, sum(1 for item in rubric_items if float(item.get("weight", 0.0)) > 0.0))
    return 1


def _rubric_weight_overrides(args) -> dict[str, float] | None:
    qwen_gradient_rubric = getattr(args, "qwen_gradient_rubric", "motion")
    if qwen_gradient_rubric == "full":
        return None
    return {
        "motion": 0.0,
        "entities": 0.0,
        "overall": 0.0,
        str(qwen_gradient_rubric): 1.0,
    }


def _log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    log_std = log_std.clamp(math.log(1e-4), math.log(1.0))
    inv_std = torch.exp(-log_std)
    z = (action - mean) * inv_std
    return (-0.5 * z.pow(2) - log_std - 0.5 * _LOG_2PI).sum()


def _entropy_per_dim(log_stds: list[torch.Tensor]) -> torch.Tensor:
    if not log_stds:
        return torch.tensor(0.0)
    entropies = [0.5 * (1.0 + _LOG_2PI) + ls.clamp(math.log(1e-4), math.log(1.0)) for ls in log_stds]
    return torch.stack(entropies).mean()


def _policy_device(audio_mean: torch.Tensor | None, text_mean: torch.Tensor | None) -> torch.device:
    if audio_mean is not None:
        return audio_mean.device
    if text_mean is not None:
        return text_mean.device
    raise ValueError("PPO policy has no mean tensors.")


def _sample_policy(
    *,
    audio_mean: torch.nn.Parameter | None,
    text_mean: torch.nn.Parameter | None,
    log_std_audio: torch.nn.Parameter | None,
    log_std_text: torch.nn.Parameter | None,
) -> dict[str, Any]:
    audio_delta = None
    text_delta = None
    log_prob = torch.zeros((), device=_policy_device(audio_mean, text_mean))

    if audio_mean is not None and log_std_audio is not None:
        std = torch.exp(log_std_audio.clamp(math.log(1e-4), math.log(1.0)))
        audio_delta = audio_mean + std * torch.randn_like(audio_mean)
        log_prob = log_prob + _log_prob(audio_delta, audio_mean, log_std_audio)

    if text_mean is not None and log_std_text is not None:
        std = torch.exp(log_std_text.clamp(math.log(1e-4), math.log(1.0)))
        text_delta = text_mean + std * torch.randn_like(text_mean)
        log_prob = log_prob + _log_prob(text_delta, text_mean, log_std_text)

    return {
        "audio_delta": audio_delta,
        "text_delta": text_delta,
        "old_log_prob": float(log_prob.detach().item()),
    }


def _current_log_prob(
    rollout: dict[str, Any],
    *,
    audio_mean: torch.nn.Parameter | None,
    text_mean: torch.nn.Parameter | None,
    log_std_audio: torch.nn.Parameter | None,
    log_std_text: torch.nn.Parameter | None,
) -> torch.Tensor:
    device = _policy_device(audio_mean, text_mean)
    log_prob = torch.zeros((), device=device)

    if audio_mean is not None and log_std_audio is not None:
        action = rollout["audio_delta"].to(device=device, dtype=audio_mean.dtype)
        log_prob = log_prob + _log_prob(action, audio_mean, log_std_audio)

    if text_mean is not None and log_std_text is not None:
        action = rollout["text_delta"].to(device=device, dtype=text_mean.dtype)
        log_prob = log_prob + _log_prob(action, text_mean, log_std_text)

    return log_prob


def _build_context(
    *,
    base_pos_context: EmbeddingsProcessorOutput,
    text_delta: torch.Tensor | None,
) -> EmbeddingsProcessorOutput:
    if text_delta is None:
        return base_pos_context
    return EmbeddingsProcessorOutput(
        video_encoding=base_pos_context.video_encoding + text_delta.to(dtype=base_pos_context.video_encoding.dtype),
        audio_encoding=base_pos_context.audio_encoding,
        attention_mask=base_pos_context.attention_mask,
    )


@torch.no_grad()
def _evaluate_rollout(
    *,
    mode: str,
    args,
    update_idx: int,
    rollout_idx: int,
    rollout: dict[str, Any],
    base_pos_context: EmbeddingsProcessorOutput,
    base_neg_context: EmbeddingsProcessorOutput,
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
    cached_src_frames: torch.Tensor | None,
    qwen_calls_before: int,
    qwen_calls_per_eval: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    optimize_text = mode in ("text", "both")
    optimize_audio = mode in ("audio", "both")

    audio_delta = rollout.get("audio_delta")
    text_delta = rollout.get("text_delta")

    audio_for_render = base_audio_latent
    if optimize_audio and audio_delta is not None:
        audio_for_render = (base_audio_latent_fp32 + audio_delta).to(dtype=base_audio_latent.dtype)

    pos_ctx_iter = base_pos_context
    if optimize_text and text_delta is not None:
        pos_ctx_iter = _build_context(base_pos_context=base_pos_context, text_delta=text_delta)

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

    qwen_loss_t, qwen_details = compute_qwen_video_loss(
        frames_chw=gen_frames,
        qwen_model=qwen_model,
        cached_inputs=cached_qwen_inputs,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        max_frames=args.qwen_max_frames,
        img_size=args.qwen_img_size,
        backward=False,
        return_details=True,
        sample_mode=getattr(args, "qwen_sample_mode", "linspace"),
        contiguous_start_frame=getattr(args, "qwen_contiguous_start_frame", 0),
        rubric_weight_overrides=_rubric_weight_overrides(args),
    )

    perceptual_loss_t = torch.tensor(0.0, device=gen_frames.device)
    perceptual_details: dict[str, float] = {}
    lpips_weight = getattr(args, "lpips_weight", 0.0)
    temporal_weight = getattr(args, "temporal_weight", 0.0)
    if cached_src_frames is not None and (lpips_weight > 0 or temporal_weight > 0):
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

    audio_reg_w = adaptive_reg_weight(args.latent_reg_weight, update_idx, max(int(args.iterations), 1), schedule=args.reg_schedule)
    text_reg_w = adaptive_reg_weight(args.text_reg_weight, update_idx, max(int(args.iterations), 1), schedule=args.reg_schedule)

    audio_reg_t = torch.tensor(0.0, device=qwen_loss_t.device)
    if optimize_audio and audio_delta is not None and audio_reg_w > 0:
        audio_reg_t = audio_reg_w * torch.mean(audio_delta.float().pow(2))

    text_reg_t = torch.tensor(0.0, device=qwen_loss_t.device)
    if optimize_text and text_delta is not None and text_reg_w > 0:
        text_reg_t = text_reg_w * torch.mean(text_delta.float().pow(2))

    qwen_loss = float(qwen_loss_t.detach().item())
    qwen_score = math.exp(-qwen_loss)
    perceptual_loss = float(perceptual_loss_t.detach().item())
    audio_reg = float(audio_reg_t.detach().item())
    text_reg = float(text_reg_t.detach().item())
    total = qwen_loss + audio_reg + text_reg + perceptual_loss
    reward = -total

    result = {
        "method": "ppo",
        "update": update_idx,
        "rollout": rollout_idx,
        "qwen_nll": qwen_loss,
        "qwen_yes_prob": qwen_score,
        "audio_reg": audio_reg,
        "text_reg": text_reg,
        "perceptual_loss": perceptual_loss,
        "lpips_raw": perceptual_details.get("lpips_raw", ""),
        "temporal_raw": perceptual_details.get("temporal_raw", ""),
        "total_loss": total,
        "reward": reward,
        "elapsed_seconds": elapsed_seconds,
        "qwen_calls": qwen_calls_before + qwen_calls_per_eval,
        "qwen_details": qwen_details,
        "audio_latent": (base_audio_latent_fp32 + audio_delta).detach().clone()
        if optimize_audio and audio_delta is not None
        else None,
        "delta_v": text_delta.detach().clone() if optimize_text and text_delta is not None else None,
    }

    del gen_frames, qwen_loss_t, perceptual_loss_t, audio_reg_t, text_reg_t
    return result


def ppo_optimize_multimodal_qwen(
    *,
    mode: str,
    args,
    is_main: bool,
    output_dir: Path,
    base_pos_context: EmbeddingsProcessorOutput,
    base_neg_context: EmbeddingsProcessorOutput,
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
    cached_src_frames: torch.Tensor | None = None,
    qwen_call_budget: int | None = None,
    wallclock_budget_seconds: float | None = None,
    method_name: str = "ppo",
) -> dict[str, Any]:
    """Run one-step clipped PPO over text/audio edit latents."""
    if mode not in ("text", "audio", "both"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'text', 'audio', or 'both'.")

    optimize_text = mode in ("text", "both")
    optimize_audio = mode in ("audio", "both")
    device = base_audio_latent_fp32.device

    audio_mean = None
    text_mean = None
    log_std_audio = None
    log_std_text = None
    policy_params: list[torch.nn.Parameter] = []

    if optimize_audio:
        audio_mean = torch.nn.Parameter(torch.zeros_like(base_audio_latent_fp32))
        log_std_audio = torch.nn.Parameter(torch.tensor(math.log(float(args.ppo_init_std_audio)), device=device))
        policy_params.extend([audio_mean, log_std_audio])
        log.info("[%s:%s] PPO audio policy shape: %s", method_name, mode, tuple(audio_mean.shape))

    if optimize_text:
        text_mean = torch.nn.Parameter(torch.zeros_like(base_pos_context.video_encoding.float()))
        log_std_text = torch.nn.Parameter(torch.tensor(math.log(float(args.ppo_init_std_text)), device=device))
        policy_params.extend([text_mean, log_std_text])
        log.info("[%s:%s] PPO text policy shape: %s", method_name, mode, tuple(text_mean.shape))

    if not policy_params:
        raise ValueError(f"Mode {mode!r} has no optimizable variables.")

    optimizer = torch.optim.Adam(policy_params, lr=float(args.ppo_lr))
    qwen_calls_per_eval = active_qwen_rubric_count(args, cached_qwen_inputs)
    rollouts_per_update = max(1, int(args.ppo_rollouts_per_update))

    if int(args.ppo_max_updates) > 0:
        max_updates = int(args.ppo_max_updates)
    elif qwen_call_budget is not None and qwen_call_budget > 0:
        calls_per_update = max(1, rollouts_per_update * qwen_calls_per_eval)
        max_updates = max(1, math.ceil(qwen_call_budget / calls_per_update))
    elif wallclock_budget_seconds is not None and wallclock_budget_seconds > 0:
        max_updates = 100000
    else:
        max_updates = max(1, int(args.iterations))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"optimization_log_qwen_ppo_{mode}.csv"
    rubric_metric_names = [str(item["name"]) for item in cached_qwen_inputs.get("rubric_items", [])]

    best: dict[str, Any] = {
        "qwen_loss": float("inf"),
        "qwen_score": float("-inf"),
        "delta_v": text_mean.detach().clone() if text_mean is not None else None,
        "audio_latent": base_audio_latent_fp32.detach().clone() if audio_mean is not None else None,
        "mode": mode,
        "best_iter": 0,
        "clip_loss": float("inf"),
        "clip_score": float("-inf"),
        "qwen_calls": 0,
        "loop_seconds": 0.0,
    }

    fieldnames = [
        "iter", "method", "update", "rollout",
        "qwen_nll", "qwen_yes_prob",
        "audio_reg", "text_reg", "perceptual_loss",
        "lpips_raw", "temporal_raw",
        "total_loss", "reward", "advantage",
        "elapsed_seconds", "qwen_calls",
        "policy_loss", "approx_kl", "clip_fraction", "entropy",
        "grad_norm", "is_best",
    ]
    for name in rubric_metric_names:
        fieldnames.extend([f"{name}_nll", f"{name}_yes_prob"])

    csv_file = csv_path.open("w", newline="") if is_main else open("/dev/null", "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if is_main:
        writer.writeheader()

    qwen_calls = 0
    rollout_counter = 0
    start_time = time.perf_counter()
    iters_without_improvement = 0

    try:
        for update_idx in range(1, max_updates + 1):
            if wallclock_budget_seconds is not None and time.perf_counter() - start_time >= wallclock_budget_seconds:
                log.info("[%s:%s] Wall-clock budget reached before update %d.", method_name, mode, update_idx)
                break
            if qwen_call_budget is not None and qwen_calls >= qwen_call_budget:
                log.info("[%s:%s] Qwen-call budget reached before update %d.", method_name, mode, update_idx)
                break

            batch: list[dict[str, Any]] = []
            for local_rollout_idx in range(1, rollouts_per_update + 1):
                if qwen_call_budget is not None and qwen_calls + qwen_calls_per_eval > qwen_call_budget and batch:
                    break
                if wallclock_budget_seconds is not None and time.perf_counter() - start_time >= wallclock_budget_seconds and batch:
                    break

                sample = _sample_policy(
                    audio_mean=audio_mean,
                    text_mean=text_mean,
                    log_std_audio=log_std_audio,
                    log_std_text=log_std_text,
                )
                rollout_counter += 1
                elapsed = time.perf_counter() - start_time
                result = _evaluate_rollout(
                    mode=mode,
                    args=args,
                    update_idx=update_idx,
                    rollout_idx=local_rollout_idx,
                    rollout=sample,
                    base_pos_context=base_pos_context,
                    base_neg_context=base_neg_context,
                    base_audio_latent=base_audio_latent,
                    base_audio_latent_fp32=base_audio_latent_fp32,
                    cached_video_latent=cached_video_latent,
                    retake_input_video=retake_input_video,
                    pipeline=pipeline,
                    retake_kwargs=retake_kwargs,
                    qwen_model=qwen_model,
                    cached_qwen_inputs=cached_qwen_inputs,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    eval_sample_start=eval_sample_start,
                    cached_src_frames=cached_src_frames,
                    qwen_calls_before=qwen_calls,
                    qwen_calls_per_eval=qwen_calls_per_eval,
                    elapsed_seconds=elapsed,
                )
                result["method"] = method_name
                qwen_calls += qwen_calls_per_eval

                result_audio_latent = result.pop("audio_latent")
                result_delta_v = result.pop("delta_v")
                is_best = result["total_loss"] < best["qwen_loss"]
                if is_best:
                    best["qwen_loss"] = result["total_loss"]
                    best["qwen_score"] = result["qwen_yes_prob"]
                    best["clip_loss"] = result["total_loss"]
                    best["clip_score"] = result["qwen_yes_prob"]
                    best["best_iter"] = rollout_counter
                    best["qwen_calls"] = qwen_calls
                    best["loop_seconds"] = time.perf_counter() - start_time
                    if result_delta_v is not None:
                        best["delta_v"] = result_delta_v.detach().clone()
                    if result_audio_latent is not None:
                        best["audio_latent"] = result_audio_latent.detach().clone()
                    iters_without_improvement = 0
                else:
                    iters_without_improvement += 1

                sample["audio_delta"] = sample["audio_delta"].detach().cpu() if sample.get("audio_delta") is not None else None
                sample["text_delta"] = sample["text_delta"].detach().cpu() if sample.get("text_delta") is not None else None
                sample.update(result)
                sample["iter"] = rollout_counter
                sample["is_best"] = int(is_best)
                batch.append(sample)

                log.info(
                    "[%s:%s] rollout %3d  update=%d  qwen_nll=%.4f  yes_prob=%.4f  total=%.4f  reward=%.4f  calls=%d%s",
                    method_name, mode, rollout_counter, update_idx,
                    result["qwen_nll"], result["qwen_yes_prob"], result["total_loss"], result["reward"],
                    qwen_calls, "  *" if is_best else "",
                )
                _clear_cuda_cache()

                early_stop_limit = int(getattr(args, "early_stopping", 0) or 0)
                if early_stop_limit > 0 and iters_without_improvement >= early_stop_limit:
                    log.info(
                        "[%s:%s] Early stopping: no improvement for %d rollouts (best rollout %d).",
                        method_name, mode, iters_without_improvement, best.get("best_iter", 0),
                    )
                    break

            if not batch:
                break

            rewards = torch.tensor([float(item["reward"]) for item in batch], device=device)
            advantages = rewards - rewards.mean()
            if len(batch) > 1 and float(advantages.std(unbiased=False).item()) > 1e-8:
                advantages = advantages / (advantages.std(unbiased=False) + 1e-8)

            policy_loss_value = 0.0
            approx_kl_value = 0.0
            clip_fraction_value = 0.0
            entropy_value = 0.0
            grad_norm = 0.0

            for _ in range(max(1, int(args.ppo_epochs))):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                kls = []
                clipped = []

                for rollout, adv in zip(batch, advantages, strict=True):
                    new_log_prob = _current_log_prob(
                        rollout,
                        audio_mean=audio_mean,
                        text_mean=text_mean,
                        log_std_audio=log_std_audio,
                        log_std_text=log_std_text,
                    )
                    old_log_prob = torch.tensor(float(rollout["old_log_prob"]), device=device)
                    log_ratio = (new_log_prob - old_log_prob).clamp(-20.0, 20.0)
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = ratio.clamp(1.0 - float(args.ppo_clip_eps), 1.0 + float(args.ppo_clip_eps))
                    losses.append(-torch.minimum(ratio * adv, clipped_ratio * adv))
                    kls.append(old_log_prob - new_log_prob.detach())
                    clipped.append((torch.abs(ratio.detach() - 1.0) > float(args.ppo_clip_eps)).float())

                entropy = _entropy_per_dim([ls for ls in [log_std_audio, log_std_text] if ls is not None]).to(device)
                policy_loss = torch.stack(losses).mean() - float(args.ppo_entropy_weight) * entropy
                policy_loss.backward()

                if float(getattr(args, "grad_clip", 0.0) or 0.0) > 0:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(policy_params, max_norm=float(args.grad_clip)).item())
                else:
                    grad_norm = float(
                        sum(p.grad.norm().item() ** 2 for p in policy_params if p.grad is not None) ** 0.5
                    )

                optimizer.step()
                with torch.no_grad():
                    if log_std_audio is not None:
                        log_std_audio.clamp_(math.log(1e-4), math.log(1.0))
                    if log_std_text is not None:
                        log_std_text.clamp_(math.log(1e-4), math.log(1.0))

                policy_loss_value = float(policy_loss.detach().item())
                approx_kl_value = float(torch.stack(kls).mean().item()) if kls else 0.0
                clip_fraction_value = float(torch.stack(clipped).mean().item()) if clipped else 0.0
                entropy_value = float(entropy.detach().item())

            if is_main:
                for row_item, adv in zip(batch, advantages, strict=True):
                    detail_by_name = {str(item["name"]): item for item in row_item.get("qwen_details", [])}
                    row = {
                        key: row_item.get(key, "")
                        for key in [
                            "iter", "method", "update", "rollout",
                            "qwen_nll", "qwen_yes_prob",
                            "audio_reg", "text_reg", "perceptual_loss",
                            "lpips_raw", "temporal_raw",
                            "total_loss", "reward", "elapsed_seconds",
                            "qwen_calls", "is_best",
                        ]
                    }
                    row["advantage"] = float(adv.detach().item())
                    row["policy_loss"] = policy_loss_value
                    row["approx_kl"] = approx_kl_value
                    row["clip_fraction"] = clip_fraction_value
                    row["entropy"] = entropy_value
                    row["grad_norm"] = grad_norm
                    for name in rubric_metric_names:
                        item = detail_by_name.get(name, {})
                        row[f"{name}_nll"] = item.get("nll", "")
                        row[f"{name}_yes_prob"] = item.get("yes_prob", "")
                    writer.writerow(row)
                csv_file.flush()

            early_stop_limit = int(getattr(args, "early_stopping", 0) or 0)
            if early_stop_limit > 0 and iters_without_improvement >= early_stop_limit:
                break

        best["qwen_calls"] = max(int(best.get("qwen_calls", 0)), qwen_calls)
        best["loop_seconds"] = time.perf_counter() - start_time
        return best
    finally:
        csv_file.close()
        _clear_cuda_cache()


__all__ = [
    "active_qwen_rubric_count",
    "ppo_optimize_multimodal_qwen",
]
