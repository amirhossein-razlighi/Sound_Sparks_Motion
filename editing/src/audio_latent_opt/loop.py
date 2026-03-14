from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.distributed as dist

from .distributed import is_distributed
from .losses import compute_total_loss


def gradient_optimize_audio_latent(
    *,
    args,
    is_main: bool,
    world_size: int,
    output_dir: Path,
    base_audio_latent: torch.Tensor,
    base_audio_latent_fp32: torch.Tensor,
    cached_video_latent: torch.Tensor,
    retake_input_video: str,
    pipeline,
    retake_kwargs: dict,
    resize_to: tuple[int, int],
    target_flows: torch.Tensor,
    raft_model: torch.nn.Module,
    raft_transforms,
    render_with_injected_audio_latent,
) -> dict:
    audio_latent = torch.nn.Parameter(base_audio_latent_fp32.clone())
    optimizer = torch.optim.Adam([audio_latent], lr=args.lr)

    csv_path = output_dir / "optimization_log.csv"
    with (csv_path.open("w", newline="") if is_main else open("/dev/null", "w", newline="")) as f_csv:
        writer = csv.writer(f_csv)
        if is_main:
            writer.writerow([
                "iter",
                "total_loss",
                "flow_mse",
                "mag_curve_mse",
                "latent_reg",
                "grad_norm",
                "is_best",
            ])

        best = {
            "loss": float("inf"),
            "latent": audio_latent.detach().clone(),
            "flow_mse": float("inf"),
            "mag_mse": float("inf"),
        }

        for it in range(1, args.iterations + 1):
            optimizer.zero_grad(set_to_none=True)

            injected_latent = audio_latent.to(dtype=base_audio_latent.dtype)
            gen_frames = render_with_injected_audio_latent(
                pipeline=pipeline,
                src_video=retake_input_video,
                injected_audio_latent=injected_latent,
                cached_video_latent=cached_video_latent,
                retake_kwargs=retake_kwargs,
                max_frames=args.max_eval_frames,
                frame_stride=args.frame_stride,
                resize_to=resize_to,
            )
            if gen_frames.shape[0] < 2:
                raise RuntimeError("Generated video has fewer than 2 frames; cannot compute flow objective.")

            total_t, flow_mse_t, mag_mse_t, latent_reg_t = compute_total_loss(
                gen_frames_chw=gen_frames,
                target_flows=target_flows,
                raft_model=raft_model,
                raft_transforms=raft_transforms,
                flow_weight=args.flow_weight,
                mag_curve_weight=args.mag_curve_weight,
                audio_latent=audio_latent,
                base_audio_latent_fp32=base_audio_latent_fp32,
                latent_reg_weight=args.latent_reg_weight,
            )
            total_t.backward()

            if world_size > 1 and audio_latent.grad is not None and is_distributed():
                dist.all_reduce(audio_latent.grad, op=dist.ReduceOp.SUM)
                audio_latent.grad.div_(world_size)

            grad_norm = float(audio_latent.grad.norm().item()) if audio_latent.grad is not None else 0.0
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([audio_latent], max_norm=args.grad_clip)
            optimizer.step()

            total = float(total_t.detach().item())
            flow_mse = float(flow_mse_t.detach().item())
            mag_mse = float(mag_mse_t.detach().item())
            latent_reg = float(latent_reg_t.detach().item())

            is_best = total < best["loss"]
            if is_main:
                writer.writerow([it, total, flow_mse, mag_mse, latent_reg, grad_norm, int(is_best)])
                f_csv.flush()

            if is_best:
                best["loss"] = total
                best["latent"] = audio_latent.detach().clone()
                best["flow_mse"] = flow_mse
                best["mag_mse"] = mag_mse

    return best
