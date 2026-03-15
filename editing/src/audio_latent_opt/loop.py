from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.distributed as dist

from .distributed import is_distributed
from .core import flow_objective_torch


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
    roi_frame_masks: torch.Tensor | None,
    source_frames_lpips: torch.Tensor | None,
    lpips_model,
    eval_sample_start: int,
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
                "lpips",
                "grad_norm",
                "is_best",
            ])

        best = {
            "loss": float("inf"),
            "latent": audio_latent.detach().clone(),
            "flow_mse": float("inf"),
            "mag_mse": float("inf"),
            "lpips": float("inf"),
        }

        num_iters = args.iterations
        best_latent_path = output_dir / "best_audio_latent.pt"
        if args.resume and best_latent_path.exists():
            loaded = torch.load(best_latent_path, map_location="cpu").to(device=audio_latent.device, dtype=audio_latent.dtype)
            audio_latent.data.copy_(loaded)
            best["latent"] = audio_latent.detach().clone()
            num_iters = 0

            params_path = output_dir / "best_latent_params.pt"
            if params_path.exists():
                saved_params = torch.load(params_path, map_location="cpu")
                best["loss"] = saved_params.get("best_loss", float("inf"))
                best["flow_mse"] = saved_params.get("best_flow_mse", float("inf"))
                best["mag_mse"] = saved_params.get("best_mag_mse", float("inf"))
                best["lpips"] = saved_params.get("best_lpips", float("inf"))

        for it in range(1, num_iters + 1):
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
                audio_opt_last_steps=args.audio_opt_last_steps,
                eval_sample_start=eval_sample_start,
            )
            if gen_frames.shape[0] < 2:
                raise RuntimeError("Generated video has fewer than 2 frames; cannot compute flow objective.")

            flow_total, flow_mse_t, mag_mse_t = flow_objective_torch(
                gen_frames_chw=gen_frames,
                target_flows=target_flows,
                raft_model=raft_model,
                raft_transforms=raft_transforms,
                flow_weight=args.flow_weight,
                mag_curve_weight=args.mag_curve_weight,
                roi_frame_masks=roi_frame_masks,
            )

            lpips_loss_t = torch.tensor(0.0, device=audio_latent.device)
            if args.lpips_weight > 0.0 and lpips_model is not None and source_frames_lpips is not None:
                n_frames = min(gen_frames.shape[0], source_frames_lpips.shape[0])
                gen_lpips = gen_frames[:n_frames] * 2.0 - 1.0
                src_lpips = source_frames_lpips[:n_frames]

                if args.lpips_max_frames > 0 and n_frames > args.lpips_max_frames:
                    idx = torch.linspace(
                        0,
                        n_frames - 1,
                        steps=args.lpips_max_frames,
                        device=gen_lpips.device,
                    ).round().long()
                    gen_lpips = gen_lpips.index_select(0, idx)
                    src_lpips = src_lpips.index_select(0, idx)

                lpips_loss_val = lpips_model(gen_lpips, src_lpips).mean()
                lpips_loss_t = args.lpips_weight * lpips_loss_val

            latent_reg_t = args.latent_reg_weight * torch.mean((audio_latent - base_audio_latent_fp32) ** 2)
            total_t = flow_total + latent_reg_t + lpips_loss_t
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
            lpips_val = float(lpips_loss_t.detach().item())

            is_best = total < best["loss"]
            if is_main:
                writer.writerow([it, total, flow_mse, mag_mse, latent_reg, lpips_val, grad_norm, int(is_best)])
                f_csv.flush()

            if is_best:
                best["loss"] = total
                best["latent"] = audio_latent.detach().clone()
                best["flow_mse"] = flow_mse
                best["mag_mse"] = mag_mse
                best["lpips"] = lpips_val

    return best
