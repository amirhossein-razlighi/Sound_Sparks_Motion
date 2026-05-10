from __future__ import annotations

import torch

from .core import flow_objective_torch


def compute_total_loss(
    *,
    gen_frames_chw: torch.Tensor,
    target_flows: torch.Tensor,
    raft_model: torch.nn.Module,
    raft_transforms,
    flow_weight: float,
    mag_curve_weight: float,
    audio_latent: torch.Tensor,
    base_audio_latent_fp32: torch.Tensor,
    latent_reg_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flow_total, flow_mse_t, mag_mse_t = flow_objective_torch(
        gen_frames_chw=gen_frames_chw,
        target_flows=target_flows,
        raft_model=raft_model,
        raft_transforms=raft_transforms,
        flow_weight=flow_weight,
        mag_curve_weight=mag_curve_weight,
    )
    latent_reg_t = latent_reg_weight * torch.mean((audio_latent - base_audio_latent_fp32) ** 2)
    total_t = flow_total + latent_reg_t
    return total_t, flow_mse_t, mag_mse_t, latent_reg_t
